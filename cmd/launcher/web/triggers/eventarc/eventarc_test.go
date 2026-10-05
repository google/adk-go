// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package eventarc

import (
	"bytes"
	"iter"
	"log"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"

	"github.com/gorilla/mux"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/cmd/launcher"
	"google.golang.org/adk/v2/server/authn"
	"google.golang.org/adk/v2/session"
)

const (
	testAudience       = "https://example.run.app/api/apps/noop/trigger/eventarc"
	testServiceAccount = "push@project.iam.gserviceaccount.com"
)

var authArgs = []string{"-oidc_audience=" + testAudience, "-oidc_service_accounts=" + testServiceAccount}

// newTestServer parses args, wires the route onto a fresh router, and returns
// the router with the session service the trigger writes into. A non-nil auth
// replaces the authenticator Parse built, standing in for a token that verifies.
func newTestServer(t *testing.T, l *eventarcLauncher, args []string, auth authn.Authenticator) (*mux.Router, session.Service) {
	t.Helper()
	if _, err := l.Parse(args); err != nil {
		t.Fatalf("Parse(%q) failed: %v", args, err)
	}
	if auth != nil {
		l.auth = auth
	}
	a, err := agent.New(agent.Config{
		Name: "noop",
		Run: func(agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(func(*session.Event, error) bool) {}
		},
	})
	if err != nil {
		t.Fatalf("agent.New() failed: %v", err)
	}
	sessions := session.InMemoryService()
	router := mux.NewRouter()
	if err := l.SetupSubrouters(router, &launcher.Config{
		SessionService: sessions,
		AgentLoader:    agent.NewSingleLoader(a),
	}); err != nil {
		t.Fatalf("SetupSubrouters() failed: %v", err)
	}
	return router, sessions
}

// postEvent sends a binary-mode CloudEvent whose ce-source names source.
func postEvent(router http.Handler, source, authorization string) *httptest.ResponseRecorder {
	req := httptest.NewRequest(http.MethodPost, "/api/apps/noop/trigger/eventarc", strings.NewReader(`{}`))
	req.Header.Set("ce-id", "1")
	req.Header.Set("ce-type", "google.cloud.storage.object.v1.finalized")
	req.Header.Set("ce-source", source)
	req.Header.Set("ce-specversion", "1.0")
	if authorization != "" {
		req.Header.Set("Authorization", authorization)
	}
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	return rec
}

func sessionCount(t *testing.T, sessions session.Service, userID string) int {
	t.Helper()
	resp, err := sessions.List(t.Context(), &session.ListRequest{AppName: "noop", UserID: userID})
	if err != nil {
		t.Fatalf("List() failed: %v", err)
	}
	return len(resp.Sessions)
}

func TestUnauthenticatedCallerCannotImpersonate(t *testing.T) {
	for _, tc := range []struct {
		name          string
		authorization string
	}{
		{name: "no credential"},
		{name: "malformed bearer token", authorization: "Bearer not-a-jwt"},
		{name: "non-bearer scheme", authorization: "Basic dXNlcjpwYXNz"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			router, sessions := newTestServer(t, NewLauncher().(*eventarcLauncher), authArgs, nil)

			rec := postEvent(router, "victim", tc.authorization)

			if rec.Code != http.StatusUnauthorized {
				t.Errorf("status = %d, want %d", rec.Code, http.StatusUnauthorized)
			}
			if n := sessionCount(t, sessions, "victim"); n != 0 {
				t.Errorf("an unauthenticated request ran the agent as the user it named: %d sessions for %q", n, "victim")
			}
		})
	}
}

// Auth is opt-in, so without the flags the route behaves as it always has.
// This also proves sessionCount can see the run the test above forbids.
func TestNoAuthFlagsKeepsRouteOpen(t *testing.T) {
	router, sessions := newTestServer(t, NewLauncher().(*eventarcLauncher), nil, nil)

	rec := postEvent(router, "victim", "")

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d; body: %s", rec.Code, http.StatusOK, rec.Body)
	}
	if n := sessionCount(t, sessions, "victim"); n != 1 {
		t.Errorf("sessions for %q = %d, want 1", "victim", n)
	}
}

func TestVerifiedCallerRunsAsDeliveryUser(t *testing.T) {
	verified := authn.NewCustom(func(*http.Request) (*authn.Caller, error) {
		return &authn.Caller{UserID: "push-sa-subject"}, nil
	})
	router, sessions := newTestServer(t, NewLauncher().(*eventarcLauncher), authArgs, verified)

	rec := postEvent(router, "delivery-user", "")

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d; body: %s", rec.Code, http.StatusOK, rec.Body)
	}
	if n := sessionCount(t, sessions, "delivery-user"); n != 1 {
		t.Errorf("sessions for the delivery's user = %d, want 1", n)
	}
	if n := sessionCount(t, sessions, "push-sa-subject"); n != 0 {
		t.Errorf("sessions for the authenticated service account = %d, want 0", n)
	}
}

func TestRefusedCallerGets403(t *testing.T) {
	refused := authn.NewCustom(func(*http.Request) (*authn.Caller, error) {
		return nil, authn.ErrForbidden
	})
	router, sessions := newTestServer(t, NewLauncher().(*eventarcLauncher), authArgs, refused)

	rec := postEvent(router, "victim", "")

	if rec.Code != http.StatusForbidden {
		t.Errorf("status = %d, want %d", rec.Code, http.StatusForbidden)
	}
	if n := sessionCount(t, sessions, "victim"); n != 0 {
		t.Errorf("a refused caller ran the agent: %d sessions", n)
	}
}

func TestStartupWarnsOnlyWhenUnauthenticated(t *testing.T) {
	for _, tc := range []struct {
		name     string
		args     []string
		wantWarn bool
	}{
		{name: "no auth flags", wantWarn: true},
		{name: "auth flags", args: authArgs, wantWarn: false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var buf bytes.Buffer
			log.SetOutput(&buf)
			t.Cleanup(func() { log.SetOutput(os.Stderr) })

			newTestServer(t, NewLauncher().(*eventarcLauncher), tc.args, nil)

			if got := strings.Contains(buf.String(), "unauthenticated"); got != tc.wantWarn {
				t.Errorf("logged an unauthenticated warning = %t, want %t; log: %q", got, tc.wantWarn, buf.String())
			}
		})
	}
}

func TestParseTrimsSpaceAroundServiceAccounts(t *testing.T) {
	args := []string{"-oidc_audience=" + testAudience, "-oidc_service_accounts=" + testServiceAccount + ", other@project.iam.gserviceaccount.com"}
	if _, err := NewLauncher().Parse(args); err != nil {
		t.Errorf("Parse(%q) = %v, want no error", args, err)
	}
}

func TestParseRejectsIncompleteAuthFlags(t *testing.T) {
	for _, tc := range []struct {
		name string
		args []string
	}{
		{name: "audience without service accounts", args: []string{"-oidc_audience=" + testAudience}},
		{name: "service accounts without audience", args: []string{"-oidc_service_accounts=" + testServiceAccount}},
		{name: "trailing comma", args: []string{"-oidc_audience=" + testAudience, "-oidc_service_accounts=" + testServiceAccount + ","}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := NewLauncher().Parse(tc.args); err == nil {
				t.Errorf("Parse(%q) succeeded, want an error", tc.args)
			}
		})
	}
}
