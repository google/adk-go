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
	"google.golang.org/adk/v2/session"
)

func TestServerHealth(t *testing.T) {
	server, err := NewServer(ServerConfig{})
	if err != nil {
		t.Fatalf("NewServer() failed: %v", err)
	}

	recorder := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodGet, "/health", nil)
	server.ServeHTTP(recorder, request)

	if got, want := recorder.Code, http.StatusOK; got != want {
		t.Errorf("GET /health status = %d, want %d", got, want)
	}
	if got, want := recorder.Header().Get("Content-Type"), "application/json"; got != want {
		t.Errorf("GET /health Content-Type = %q, want %q", got, want)
	}
	if got, want := recorder.Body.String(), "{\"status\":\"ok\"}\n"; got != want {
		t.Errorf("GET /health body = %q, want %q", got, want)
	}
}

// debugRoutes are every route owned by the debug API router, including the
// event graph route, whose path does not contain "debug".
var debugRoutes = []string{
	"/debug/trace/evt1",
	"/debug/trace/session/sess1",
	"/apps/app1/users/user1/sessions/sess1/events/evt1/graph",
}

// muxNotFound is what gorilla/mux writes when no route matches. A registered
// handler that happens to answer 404 writes its own body, so the body is what
// distinguishes "not routed" from "routed and rejected".
const muxNotFound = "404 page not found\n"

func TestNewServerDebugAPIGate(t *testing.T) {
	for _, tc := range []struct {
		name       string
		include    bool
		wantRouted bool
	}{
		{name: "omitted by default", include: false, wantRouted: false},
		{name: "included when opted in", include: true, wantRouted: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			srv, err := NewServer(ServerConfig{
				SessionService: session.InMemoryService(),
				AgentLoader:    agent.NewSingleLoader(nil),
				DebugAPIConfig: DebugAPIConfig{IncludeDebugAPI: tc.include},
			})
			if err != nil {
				t.Fatalf("NewServer() error = %v", err)
			}
			for _, route := range debugRoutes {
				rr := httptest.NewRecorder()
				srv.ServeHTTP(rr, httptest.NewRequest(http.MethodGet, route, nil))
				routed := rr.Body.String() != muxNotFound
				if routed != tc.wantRouted {
					t.Errorf("GET %s: routed = %v, want %v (code %d, body %q)",
						route, routed, tc.wantRouted, rr.Code, rr.Body.String())
				}
			}
		})
	}
}
