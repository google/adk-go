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

package adkrest

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/artifact"
	"google.golang.org/adk/v2/server/authn"
	"google.golang.org/adk/v2/server/authz"
	"google.golang.org/adk/v2/session"
)

// newAuthenticatedServer builds an assembled server gated by Custom
// authenticator that returns an caller's identity with authenticatedUserID.
func newAuthenticatedServer(t *testing.T, authenticatedUserID string) *Server {
	t.Helper()

	rootAgent, err := agent.New(agent.Config{Name: testAppName, Description: "root agent"})
	if err != nil {
		t.Fatalf("agent.New() failed: %v", err)
	}
	srv, err := NewServer(ServerConfig{
		SessionService:  session.InMemoryService(),
		ArtifactService: artifact.InMemoryService(),
		AgentLoader:     agent.NewSingleLoader(rootAgent),
		Authenticator: authn.NewCustom(
			func(r *http.Request) (*authn.Caller, error) {
				return &authn.Caller{
					UserID: authenticatedUserID,
				}, nil
			}),
		Authorizer: authz.NewStrict(),
	})
	if err != nil {
		t.Fatalf("NewServer() failed: %v", err)
	}
	return srv
}

// TestAuthPublicEndpointsSkipAuth checks the endpoints that must stay reachable
// without credentials even when an authenticator is configured.
func TestAuthPublicEndpointsSkipAuth(t *testing.T) {
	srv := newAuthenticatedServer(t, "")

	for _, path := range []string{"/health", "/version"} {
		t.Run(path, func(t *testing.T) {
			rr := httptest.NewRecorder()
			srv.ServeHTTP(rr, httptest.NewRequest(http.MethodGet, path, nil))
			if got, want := rr.Code, http.StatusOK; got != want {
				t.Errorf("GET %s status = %d, want %d (public endpoint must skip auth); body: %s",
					path, got, want, rr.Body.String())
			}
		})
	}
}

// TestAuthDisabledByDefault confirms the change is backward compatible: with no
// authenticator, a protected endpoint is reachable without credentials.
func TestAuthDisabledByDefault(t *testing.T) {
	rootAgent, err := agent.New(agent.Config{Name: testAppName, Description: "root agent"})
	if err != nil {
		t.Fatalf("agent.New() failed: %v", err)
	}
	srv, err := NewServer(ServerConfig{
		SessionService: session.InMemoryService(),
		AgentLoader:    agent.NewSingleLoader(rootAgent),
	})
	if err != nil {
		t.Fatalf("NewServer() failed: %v", err)
	}

	rr := httptest.NewRecorder()
	srv.ServeHTTP(rr, httptest.NewRequest(http.MethodGet, "/list-apps", nil))
	if got, want := rr.Code, http.StatusOK; got != want {
		t.Errorf("GET /list-apps status = %d, want %d (no authenticator should not gate)", got, want)
	}
}

// TestAuthPreflightNotBlocked checks a CORS preflight is answered without
// credentials. No route declares OPTIONS, so the request reaches the fallback
// handler, which is not wrapped with auth.
func TestAuthPreflightNotBlocked(t *testing.T) {
	srv := newAuthenticatedServer(t, "")

	rr := httptest.NewRecorder()
	srv.ServeHTTP(rr, httptest.NewRequest(http.MethodOptions, "/list-apps", nil))
	if got, want := rr.Code, http.StatusNoContent; got != want {
		t.Errorf("OPTIONS /list-apps status = %d, want %d (preflight must not require auth)", got, want)
	}
}

// TestCrossUserAuthorization exercises the full authorization path through the
// user-scoped controllers: the caller is authenticated as "alice" with an
// [authz.strict] authorizer, so a request for alice's own resources is allowed
// while a request naming another user is answered 403 before the handler acts.
func TestCrossUserAuthorization(t *testing.T) {
	srv := newAuthenticatedServer(t, "alice")

	const base = "/apps/" + testAppName + "/users/"

	tests := []struct {
		name     string
		method   string
		path     string
		wantCode int
	}{
		{"own session create", http.MethodPost, base + "alice/sessions/s1", http.StatusOK},
		{"own session list", http.MethodGet, base + "alice/sessions", http.StatusOK},
		{"other user session create", http.MethodPost, base + "bob/sessions/s2", http.StatusForbidden},
		{"other user session list", http.MethodGet, base + "bob/sessions", http.StatusForbidden},
		{"other user session get", http.MethodGet, base + "bob/sessions/s1", http.StatusForbidden},
		{"other user session delete", http.MethodDelete, base + "bob/sessions/s1", http.StatusForbidden},
		{"other user artifacts list", http.MethodGet, base + "bob/sessions/s1/artifacts", http.StatusForbidden},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rr := httptest.NewRecorder()
			srv.ServeHTTP(rr, httptest.NewRequest(tt.method, tt.path, nil))
			if rr.Code != tt.wantCode {
				t.Errorf("%s %s status = %d, want %d; body: %s",
					tt.method, tt.path, rr.Code, tt.wantCode, rr.Body.String())
			}
		})
	}
}

// TestDebugEventGraphCrossUserForbidden guards the debug API's one user-scoped
// route. The event-graph endpoint carries a {user_id}, so it must enforce the
// same authorization as the other user-scoped controllers rather than reading
// another user's session events. Authenticated as "alice":
//   - bob's event graph is 403, refused before the session is looked up;
//   - alice's own event graph is not 403 (authorization passes; the missing
//     session then yields a different status).
func TestDebugEventGraphCrossUserForbidden(t *testing.T) {
	rootAgent, err := agent.New(agent.Config{Name: testAppName, Description: "root agent"})
	if err != nil {
		t.Fatalf("agent.New() failed: %v", err)
	}
	srv, err := NewServer(ServerConfig{
		SessionService: session.InMemoryService(),
		AgentLoader:    agent.NewSingleLoader(rootAgent),
		Authenticator: authn.NewCustom(func(*http.Request) (*authn.Caller, error) {
			return &authn.Caller{UserID: "alice"}, nil
		}),
		Authorizer:     authz.NewStrict(),
		DebugAPIConfig: DebugAPIConfig{IncludeDebugAPI: true},
	})
	if err != nil {
		t.Fatalf("NewServer() failed: %v", err)
	}

	const suffix = "/sessions/s1/events/e1/graph"

	rr := httptest.NewRecorder()
	srv.ServeHTTP(rr, httptest.NewRequest(http.MethodGet, "/apps/"+testAppName+"/users/bob"+suffix, nil))
	if got, want := rr.Code, http.StatusForbidden; got != want {
		t.Errorf("GET bob's event graph status = %d, want %d (must be refused before the session lookup); body: %s",
			got, want, rr.Body.String())
	}

	rr = httptest.NewRecorder()
	srv.ServeHTTP(rr, httptest.NewRequest(http.MethodGet, "/apps/"+testAppName+"/users/alice"+suffix, nil))
	if rr.Code == http.StatusForbidden {
		t.Errorf("GET alice's own event graph returned 403; authorization must pass for the caller's own user")
	}
}
