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
	"regexp"
	"testing"

	"github.com/gorilla/mux"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/server/authn"
	"google.golang.org/adk/v2/session"
)

// publicEndpoints are the path templates intentionally reachable without
// authentication. Every other endpoint the router exposes must require an
// authenticated user; [TestAllEndpointsRequireAuthentication] enforces that.
//
// Adding an endpoint here is a deliberate decision to expose it unauthenticated.
var publicEndpoints = map[string]bool{
	"/health":  true,
	"/version": true,
}

// pathVarPattern matches a mux path variable such as {app_name}.
var pathVarPattern = regexp.MustCompile(`\{[^}]+\}`)

// concretePath replaces every {var} segment of a mux path template with a
// placeholder so the path matches its route. Every adkrest route uses the
// default [^/]+ matcher, so any non-empty, slash-free value matches.
func concretePath(template string) string {
	return pathVarPattern.ReplaceAllString(template, "x")
}

// primaryMethod returns a verb the route serves, preferring a non-HEAD one so
// the probe exercises the route's real handler.
func primaryMethod(methods []string) string {
	for _, m := range methods {
		if m != http.MethodHead && m != http.MethodOptions {
			return m
		}
	}
	if len(methods) > 0 {
		return methods[0]
	}
	return http.MethodGet
}

// TestAllEndpointsRequireAuthentication walks every route the adkrest server
// registers and asserts that an unauthenticated request to it is rejected with
// 401 — unless the endpoint is on the small [publicEndpoints] allowlist.
//
// It fails if a new endpoint ships reachable without authentication and is not
// deliberately allowlisted, so authentication coverage cannot silently regress
// as endpoints are added. It walks the live router rather than a hand-kept list,
// so it cannot fall out of sync with the routes actually served.
func TestAllEndpointsRequireAuthentication(t *testing.T) {
	rootAgent, err := agent.New(agent.Config{Name: testAppName, Description: "root agent"})
	if err != nil {
		t.Fatalf("agent.New() failed: %v", err)
	}
	srv, err := NewServer(ServerConfig{
		SessionService: session.InMemoryService(),
		AgentLoader:    agent.NewSingleLoader(rootAgent),
		// NewHeader rejects a request that lacks the named header, so a
		// credential-less request to a protected route is answered 401 by the
		// middleware before the handler runs.
		Authenticator: authn.NewHeader("non-existing"),
		// Opt into the debug API so its routes are covered too.
		DebugAPIConfig: DebugAPIConfig{IncludeDebugAPI: true},
	})
	if err != nil {
		t.Fatalf("NewServer() failed: %v", err)
	}

	seenPublic := map[string]bool{}
	protected := 0

	walkErr := srv.router.Walk(func(route *mux.Route, _ *mux.Router, _ []*mux.Route) error {
		template, err := route.GetPathTemplate()
		if err != nil {
			// A route with no path template (none in adkrest); nothing to probe.
			return nil
		}
		methods, err := route.GetMethods()
		if err != nil {
			// A route with no method matcher matches every verb; probe with GET.
			methods = []string{http.MethodGet}
		}
		method := primaryMethod(methods)
		path := concretePath(template)

		rr := httptest.NewRecorder()
		// The request deliberately carries no credentials.
		srv.ServeHTTP(rr, httptest.NewRequest(method, path, nil))

		if publicEndpoints[template] {
			seenPublic[template] = true
			if rr.Code == http.StatusUnauthorized {
				t.Errorf("%s %s is allowlisted as public but returned 401; a public endpoint must be reachable without authentication",
					method, template)
			}
			return nil
		}

		protected++
		if rr.Code != http.StatusUnauthorized {
			t.Errorf("%s %s returned %d without credentials, want 401. Every endpoint must require an authenticated user; if this one is intentionally public, add %q to publicEndpoints. Body: %s",
				method, template, rr.Code, template, rr.Body.String())
		}
		return nil
	})
	if walkErr != nil {
		t.Fatalf("router.Walk() failed: %v", walkErr)
	}

	t.Logf("checked %d protected route registrations and %d public endpoints", protected, len(seenPublic))

	// Guard against the walk finding nothing: a refactor that stopped
	// registering routes would otherwise make this test vacuously pass.
	if protected == 0 {
		t.Fatal("no protected routes were walked; the router exposed nothing to check")
	}
	// Keep the allowlist honest: every entry must still be a real route.
	for tmpl := range publicEndpoints {
		if !seenPublic[tmpl] {
			t.Errorf("public endpoint %q was not found among the server's routes; remove it from publicEndpoints or fix the route", tmpl)
		}
	}
}
