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

package pubsub

import (
	"bytes"
	"fmt"
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

func TestParse(t *testing.T) {
	tests := []struct {
		name       string
		args       []string
		wantPrefix string
		wantRetry  int
		wantErr    bool
	}{
		{
			name:       "default values",
			args:       []string{},
			wantPrefix: "/api",
			wantRetry:  3,
			wantErr:    false,
		},
		{
			name:       "custom prefix and retries",
			args:       []string{"-path_prefix=/custom", "-trigger_max_retries=5"},
			wantPrefix: "/custom",
			wantRetry:  5,
			wantErr:    false,
		},
		{
			name:       "invalid retry count",
			args:       []string{"-trigger_max_retries=-1"},
			wantPrefix: "/api",
			wantRetry:  3,
			wantErr:    true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			l := NewLauncher().(*pubsubLauncher)
			_, err := l.Parse(tt.args)
			if (err != nil) != tt.wantErr {
				t.Errorf("Parse() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if tt.wantErr {
				return
			}
			if l.config.pathPrefix != tt.wantPrefix {
				t.Errorf("Parse() pathPrefix = %v, want %v", l.config.pathPrefix, tt.wantPrefix)
			}
			if l.config.triggerMaxRetries != tt.wantRetry {
				t.Errorf("Parse() triggerMaxRetries = %v, want %v", l.config.triggerMaxRetries, tt.wantRetry)
			}
		})
	}
}

func TestSetupSubrouters(t *testing.T) {
	l := NewLauncher().(*pubsubLauncher)
	_, _ = l.Parse([]string{"-path_prefix=/api"})

	router := mux.NewRouter()
	config := &launcher.Config{}

	err := l.SetupSubrouters(router, config)
	if err != nil {
		t.Fatalf("SetupSubrouters() failed: %v", err)
	}

	// Verify route is registered
	req := httptest.NewRequest(http.MethodPost, "/api/apps/my-app/trigger/pubsub", nil)
	var match mux.RouteMatch
	if !router.Match(req, &match) {
		t.Errorf("SetupSubrouters() did not register expected route")
	}
}

const (
	testAudience       = "https://example.run.app/api/apps/noop/trigger/pubsub"
	testServiceAccount = "push@project.iam.gserviceaccount.com"
)

var authArgs = []string{"-oidc_audience=" + testAudience, "-oidc_service_accounts=" + testServiceAccount}

// newTestServer parses args, wires the route onto a fresh router, and returns
// the router with the session service the trigger writes into. A non-nil auth
// replaces the authenticator Parse built, standing in for a token that verifies.
func newTestServer(t *testing.T, l *pubsubLauncher, args []string, auth authn.Authenticator) (*mux.Router, session.Service) {
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

// postEvent sends a push delivery whose subscription field names source.
func postEvent(router http.Handler, source, authorization string) *httptest.ResponseRecorder {
	body := fmt.Sprintf(`{"message":{"data":"aGk=","messageId":"1"},"subscription":%q}`, source)
	req := httptest.NewRequest(http.MethodPost, "/api/apps/noop/trigger/pubsub", strings.NewReader(body))
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
			router, sessions := newTestServer(t, NewLauncher().(*pubsubLauncher), authArgs, nil)

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
	router, sessions := newTestServer(t, NewLauncher().(*pubsubLauncher), nil, nil)

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
	router, sessions := newTestServer(t, NewLauncher().(*pubsubLauncher), authArgs, verified)

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
	router, sessions := newTestServer(t, NewLauncher().(*pubsubLauncher), authArgs, refused)

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

			newTestServer(t, NewLauncher().(*pubsubLauncher), tc.args, nil)

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
		{name: "padded audience", args: []string{"-oidc_audience=" + testAudience + " ", "-oidc_service_accounts=" + testServiceAccount}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := NewLauncher().Parse(tc.args); err == nil {
				t.Errorf("Parse(%q) succeeded, want an error", tc.args)
			}
		})
	}
}
