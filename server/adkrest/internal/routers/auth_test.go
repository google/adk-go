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

package routers_test

import (
	"net/http"
	"net/http/httptest"
	"regexp"
	"testing"

	"github.com/gorilla/mux"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/server/adkrest/controllers"
	"google.golang.org/adk/v2/server/adkrest/internal/routers"
	"google.golang.org/adk/v2/server/adkrest/internal/services"
	"google.golang.org/adk/v2/server/authn"
	"google.golang.org/adk/v2/session"
)

// routerAuthCase pairs a router with the names of the routes it intends to serve
// without authentication. Any route not named here must require an
// authenticated user.
type routerAuthCase struct {
	name string
	// router is the subrouter under test.
	router routers.Router
	// publicRoutes maps the [routers.Route].Name of each intentionally public
	// route to true. A nil/empty map means every route requires authentication.
	publicRoutes map[string]bool
}

// routePathVar matches a mux path variable such as {app_name}.
var routePathVar = regexp.MustCompile(`\{[^}]+\}`)

// concreteRoutePath fills every {var} in a route pattern with a placeholder so
// the path matches its route. Every adkrest route uses the default [^/]+
// matcher, so any non-empty, slash-free value matches.
func concreteRoutePath(pattern string) string {
	return routePathVar.ReplaceAllString(pattern, "x")
}

// primaryRouteMethod returns a verb the route serves, preferring a non-HEAD one.
func primaryRouteMethod(methods []string) string {
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

// authTestRouters builds one instance of every router adkrest mounts, each with
// the minimal dependencies its constructor needs. Handlers on protected routes
// never run in these tests (the auth middleware answers 401 first), so services
// that are awkward to fake can be nil.
func authTestRouters(t *testing.T) []routerAuthCase {
	t.Helper()

	rootAgent, err := agent.New(agent.Config{Name: "test-app", Description: "root agent"})
	if err != nil {
		t.Fatalf("agent.New() failed: %v", err)
	}
	loader := agent.NewSingleLoader(rootAgent)
	sessionSvc := session.InMemoryService()
	telemetry, err := services.NewDebugTelemetryWithConfig(&services.DebugTelemetryConfig{})
	if err != nil {
		t.Fatalf("NewDebugTelemetryWithConfig() failed: %v", err)
	}
	runtimeController := controllers.NewRuntimeAPIControllerWithConfig(controllers.RuntimeAPIControllerConfig{
		SessionService: sessionSvc,
		AgentLoader:    loader,
	})

	return []routerAuthCase{
		{
			name:   "sessions",
			router: routers.NewSessionsAPIRouter(controllers.NewSessionsAPIController(sessionSvc)),
		},
		{
			name:   "runtime",
			router: routers.NewRuntimeAPIRouter(runtimeController),
		},
		{
			name:   "apps",
			router: routers.NewAppsAPIRouter(controllers.NewAppsAPIController(loader)),
		},
		{
			name:   "artifacts",
			router: routers.NewArtifactsAPIRouter(controllers.NewArtifactsAPIController(nil)),
		},
		{
			name:         "version",
			router:       routers.NewVersionAPIRouter(controllers.NewVersionAPIController()),
			publicRoutes: map[string]bool{"GetVersion": true},
		},
		{
			name:   "agentgraph",
			router: routers.NewAgentGraphAPIRouter(controllers.NewAgentGraphAPIController(loader)),
		},
		{
			name:   "tests",
			router: &routers.TestsAPIRouter{},
		},
		{
			name:   "eval",
			router: &routers.EvalAPIRouter{},
		},
		{
			name:   "debug",
			router: routers.NewDebugAPIRouter(controllers.NewDebugAPIController(sessionSvc, loader, telemetry)),
		},
	}
}

// TestRouterAuthentication checks, for every router adkrest exposes, which of its
// routes require an authenticated user. Each router is mounted in isolation with
// an authenticator that rejects credential-less requests; every route is then
// probed without credentials and must answer 401 unless it is intentionally
// public (only /version is). It also asserts each route's [routers.Route].Public
// flag matches that intent, so a route cannot silently change its auth posture.
func TestRouterAuthentication(t *testing.T) {
	// NewHeader rejects a request that lacks the named header, so a
	// credential-less request to a protected route is answered 401 by the
	// middleware before the handler runs.
	authenticator := authn.NewHeader("non-existing-header")

	for _, tc := range authTestRouters(t) {
		t.Run(tc.name, func(t *testing.T) {
			m := mux.NewRouter()
			routers.SetupSubRouters(m, authenticator, tc.router)

			routes := tc.router.Routes()
			if len(routes) == 0 {
				t.Fatalf("router %q exposes no routes", tc.name)
			}

			for _, route := range routes {
				route := route
				wantPublic := tc.publicRoutes[route.Name]
				t.Run(route.Name, func(t *testing.T) {
					// The declared Public flag must match the intended policy,
					// so flipping it is caught here and not only in behavior.
					if route.Public != wantPublic {
						t.Errorf("Route %q Public = %v, want %v", route.Name, route.Public, wantPublic)
					}

					method := primaryRouteMethod(route.Methods)
					path := concreteRoutePath(route.Pattern)
					rr := httptest.NewRecorder()
					// The request deliberately carries no credentials.
					m.ServeHTTP(rr, httptest.NewRequest(method, path, nil))

					if wantPublic {
						if rr.Code == http.StatusUnauthorized {
							t.Errorf("%s %s (%s) is public but returned 401; a public route must be reachable without authentication",
								method, path, route.Name)
						}
						return
					}
					if rr.Code != http.StatusUnauthorized {
						t.Errorf("%s %s (%s) returned %d without credentials, want 401; this route must require an authenticated user",
							method, path, route.Name, rr.Code)
					}
				})
			}
		})
	}
}
