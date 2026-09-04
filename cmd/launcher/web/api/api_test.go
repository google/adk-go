// Copyright 2025 Google LLC
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

package api

import (
	"fmt"
	"iter"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gorilla/mux"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/cmd/launcher"
	"google.golang.org/adk/v2/session"
)

// stubREST mimics the shape of the real ADK REST router: gorilla/mux with
// StrictSlash(true), a GET and a POST sharing one path, and PUT/PATCH/DELETE
// routes. ran records which handler served the request, so a test can assert
// that the intended handler ran rather than only that some handler returned
// 200.
type stubREST struct {
	http.Handler
	ran string
}

func newStubREST() *stubREST {
	s := &stubREST{}
	router := mux.NewRouter().StrictSlash(true)
	mark := func(name string) http.HandlerFunc {
		return func(w http.ResponseWriter, r *http.Request) {
			s.ran = name
			w.Header().Set("X-Handler", name)
			_, _ = fmt.Fprint(w, name)
		}
	}
	router.Methods(http.MethodGet, http.MethodHead).Path("/list-apps").Handler(mark("ListApps"))
	router.Methods(http.MethodGet).Path("/apps/{app}/users/{user}/sessions").Handler(mark("ListSessions"))
	router.Methods(http.MethodPost).Path("/apps/{app}/users/{user}/sessions").Handler(mark("CreateSession"))
	router.Methods(http.MethodPatch).Path("/apps/{app}/users/{user}/sessions/{session}").Handler(mark("UpdateSession"))
	router.Methods(http.MethodDelete).Path("/apps/{app}/users/{user}/sessions/{session}").Handler(mark("DeleteSession"))
	router.Methods(http.MethodPut).Path("/tests/{test}").Handler(mark("CreateTest"))
	router.Methods(http.MethodOptions).Path("/list-apps").Handler(mark("OptionsListApps"))
	s.Handler = router
	return s
}

// callAPI registers inner under prefix on a fresh base router shaped like the
// one the web launcher builds, then serves one request against it.
func callAPI(t *testing.T, prefix string, inner http.Handler, method, target string) *httptest.ResponseRecorder {
	t.Helper()
	router := mux.NewRouter().StrictSlash(true)
	registerAPIRoutes(router, prefix, inner)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, httptest.NewRequest(method, target, nil))
	return rec
}

func TestNormalizeOrigin(t *testing.T) {
	tests := []struct {
		name string
		addr string
		want string
	}{
		{name: "bare host and port gets http", addr: "localhost:8080", want: "http://localhost:8080"},
		{name: "bare host gets http", addr: "ui.example.com", want: "http://ui.example.com"},
		{name: "http kept", addr: "http://localhost:4200", want: "http://localhost:4200"},
		{name: "https kept", addr: "https://ui.example.com", want: "https://ui.example.com"},
		{name: "trailing slash stripped", addr: "http://localhost:8080/", want: "http://localhost:8080"},
		{name: "path stripped", addr: "https://ui.example.com/app/index.html", want: "https://ui.example.com"},
		{name: "query stripped", addr: "http://localhost:8080/?a=b", want: "http://localhost:8080"},
		{name: "wildcard passes through", addr: "*", want: "*"},
		{name: "empty passes through", addr: "", want: ""},
		{name: "surrounding space trimmed", addr: "  localhost:8080  ", want: "http://localhost:8080"},
		{name: "ipv6 host and port", addr: "[::1]:8080", want: "http://[::1]:8080"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := normalizeOrigin(tt.addr); got != tt.want {
				t.Errorf("normalizeOrigin(%q) = %q, want %q", tt.addr, got, tt.want)
			}
		})
	}
}

func TestCORSHeaders(t *testing.T) {
	tests := []struct {
		name          string
		addr          string
		requestOrigin string
		wantOrigin    string
		wantVary      string
	}{
		{
			name:       "default flag value gains a scheme",
			addr:       "localhost:8080",
			wantOrigin: "http://localhost:8080",
			wantVary:   "Origin",
		},
		{
			name:       "explicit origin kept",
			addr:       "https://ui.example.com",
			wantOrigin: "https://ui.example.com",
			wantVary:   "Origin",
		},
		{
			name:       "wildcard sets no Vary",
			addr:       "*",
			wantOrigin: "*",
			wantVary:   "",
		},
		{
			name:          "request Origin is never reflected",
			addr:          "localhost:8080",
			requestOrigin: "http://evil.example.com",
			wantOrigin:    "http://localhost:8080",
			wantVary:      "Origin",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			called := false
			h := corsWithArgs(tt.addr)(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				called = true
			}))
			req := httptest.NewRequest(http.MethodGet, "/list-apps", nil)
			if tt.requestOrigin != "" {
				req.Header.Set("Origin", tt.requestOrigin)
			}
			rec := httptest.NewRecorder()
			h.ServeHTTP(rec, req)

			if !called {
				t.Error("next handler was not called for a non-preflight request")
			}
			gotOrigin := rec.Header().Get("Access-Control-Allow-Origin")
			if gotOrigin != tt.wantOrigin {
				t.Errorf("Access-Control-Allow-Origin = %q, want %q", gotOrigin, tt.wantOrigin)
			}
			if gotOrigin != "*" && !strings.Contains(gotOrigin, "://") {
				t.Errorf("Access-Control-Allow-Origin = %q, want a value carrying a scheme", gotOrigin)
			}
			if got := rec.Header().Get("Vary"); got != tt.wantVary {
				t.Errorf("Vary = %q, want %q", got, tt.wantVary)
			}
			allow := rec.Header().Get("Access-Control-Allow-Methods")
			for _, m := range []string{"GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"} {
				if !strings.Contains(allow, m) {
					t.Errorf("Access-Control-Allow-Methods = %q, want it to list %s", allow, m)
				}
			}
		})
	}
}

func TestCORSPreflightStopsAtMiddleware(t *testing.T) {
	called := false
	h := corsWithArgs("localhost:8080")(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		called = true
		w.WriteHeader(http.StatusTeapot)
	}))
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest(http.MethodOptions, "/list-apps", nil))

	if called {
		t.Error("OPTIONS preflight reached the next handler, want it answered by the middleware")
	}
	if rec.Code != http.StatusOK {
		t.Errorf("status = %d, want %d", rec.Code, http.StatusOK)
	}
	if got := rec.Header().Get("Access-Control-Allow-Origin"); got != "http://localhost:8080" {
		t.Errorf("Access-Control-Allow-Origin = %q, want %q", got, "http://localhost:8080")
	}
}

// TestRegisterAPIRoutesForwardsEveryMethod covers the blanket 405 the outer
// mount used to return for PUT, PATCH and HEAD.
func TestRegisterAPIRoutesForwardsEveryMethod(t *testing.T) {
	prefixes := []struct {
		name   string
		prefix string
	}{
		{name: "default prefix", prefix: "/api"},
		{name: "custom prefix", prefix: "/adk"},
		{name: "empty prefix", prefix: ""},
	}
	tests := []struct {
		name    string
		method  string
		path    string
		wantRan string
	}{
		{name: "GET", method: http.MethodGet, path: "/list-apps", wantRan: "ListApps"},
		{name: "HEAD", method: http.MethodHead, path: "/list-apps", wantRan: "ListApps"},
		{name: "POST", method: http.MethodPost, path: "/apps/a/users/u/sessions", wantRan: "CreateSession"},
		{name: "PUT", method: http.MethodPut, path: "/tests/t1", wantRan: "CreateTest"},
		{name: "PATCH", method: http.MethodPatch, path: "/apps/a/users/u/sessions/s1", wantRan: "UpdateSession"},
		{name: "DELETE", method: http.MethodDelete, path: "/apps/a/users/u/sessions/s1", wantRan: "DeleteSession"},
		{name: "OPTIONS", method: http.MethodOptions, path: "/list-apps", wantRan: "OptionsListApps"},
	}
	for _, p := range prefixes {
		for _, tt := range tests {
			t.Run(p.name+"/"+tt.name, func(t *testing.T) {
				stub := newStubREST()
				rec := callAPI(t, p.prefix, stub, tt.method, p.prefix+tt.path)
				if rec.Code != http.StatusOK {
					t.Errorf("status = %d, want %d (405 means the outer mount rejected %s)", rec.Code, http.StatusOK, tt.method)
				}
				if stub.ran != tt.wantRan {
					t.Errorf("handler that ran = %q, want %q", stub.ran, tt.wantRan)
				}
			})
		}
	}
}

// TestRegisterAPIRoutesTrailingSlash covers the guaranteed 404, and the POST
// that silently ran a GET handler, on any URL with a trailing slash.
func TestRegisterAPIRoutesTrailingSlash(t *testing.T) {
	tests := []struct {
		name    string
		prefix  string
		method  string
		target  string
		wantRan string
	}{
		{name: "GET list-apps", prefix: "/api", method: http.MethodGet, target: "/api/list-apps/", wantRan: "ListApps"},
		{name: "GET list-apps custom prefix", prefix: "/adk", method: http.MethodGet, target: "/adk/list-apps/", wantRan: "ListApps"},
		{name: "GET list-apps empty prefix", prefix: "", method: http.MethodGet, target: "/list-apps/", wantRan: "ListApps"},
		{name: "GET sessions", prefix: "/api", method: http.MethodGet, target: "/api/apps/a/users/u/sessions/", wantRan: "ListSessions"},
		{name: "POST sessions", prefix: "/api", method: http.MethodPost, target: "/api/apps/a/users/u/sessions/", wantRan: "CreateSession"},
		{name: "POST sessions custom prefix", prefix: "/adk", method: http.MethodPost, target: "/adk/apps/a/users/u/sessions/", wantRan: "CreateSession"},
		{name: "PUT test", prefix: "/api", method: http.MethodPut, target: "/api/tests/t1/", wantRan: "CreateTest"},
		{name: "no trailing slash still works", prefix: "/api", method: http.MethodGet, target: "/api/list-apps", wantRan: "ListApps"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			stub := newStubREST()
			rec := callAPI(t, tt.prefix, stub, tt.method, tt.target)
			if stub.ran != tt.wantRan {
				t.Errorf("handler that ran = %q, want %q (status %d, Location %q)", stub.ran, tt.wantRan, rec.Code, rec.Header().Get("Location"))
			}
			if rec.Code != http.StatusOK {
				t.Errorf("status = %d, want %d", rec.Code, http.StatusOK)
			}
			if got := rec.Header().Get("X-Handler"); got != tt.wantRan {
				t.Errorf("X-Handler = %q, want %q", got, tt.wantRan)
			}
		})
	}
}

// TestTrailingSlashPostNeverBecomesGet states the outcome the redirect used to
// break: the POST must not be answered by the GET handler registered on the
// same path.
func TestTrailingSlashPostNeverBecomesGet(t *testing.T) {
	stub := newStubREST()
	rec := callAPI(t, "/api", stub, http.MethodPost, "/api/apps/a/users/u/sessions/")

	if stub.ran == "ListSessions" {
		t.Fatal("POST with a trailing slash ran ListSessions, the GET handler on the same path")
	}
	if rec.Code >= 300 && rec.Code < 400 {
		if rec.Code == http.StatusMovedPermanently || rec.Code == http.StatusFound {
			t.Fatalf("POST answered with %d, which a client re-issues as GET; want 200, 308 or 404", rec.Code)
		}
	}
	if stub.ran != "CreateSession" {
		t.Errorf("handler that ran = %q, want %q", stub.ran, "CreateSession")
	}
}

// TestRedirectLocationKeepsPrefix covers a redirect written below the mount,
// which is served a path with the prefix already stripped.
func TestRedirectLocationKeepsPrefix(t *testing.T) {
	const target = "/apps/a/users/u/sessions"
	tests := []struct {
		name       string
		prefix     string
		method     string
		code       int
		wantStatus int
		wantLoc    string
	}{
		{
			name: "GET keeps 301", prefix: "/api", method: http.MethodGet,
			code: http.StatusMovedPermanently, wantStatus: http.StatusMovedPermanently, wantLoc: "/api" + target,
		},
		{
			name: "GET custom prefix", prefix: "/adk", method: http.MethodGet,
			code: http.StatusMovedPermanently, wantStatus: http.StatusMovedPermanently, wantLoc: "/adk" + target,
		},
		{
			name: "POST is promoted to 308", prefix: "/api", method: http.MethodPost,
			code: http.StatusMovedPermanently, wantStatus: http.StatusPermanentRedirect, wantLoc: "/api" + target,
		},
		{
			name: "POST 302 is promoted to 307", prefix: "/api", method: http.MethodPost,
			code: http.StatusFound, wantStatus: http.StatusTemporaryRedirect, wantLoc: "/api" + target,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			redirector := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				http.Redirect(w, r, target, tt.code)
			})
			rec := callAPI(t, tt.prefix, redirector, tt.method, tt.prefix+"/redirect-me")

			if rec.Code != tt.wantStatus {
				t.Errorf("status = %d, want %d", rec.Code, tt.wantStatus)
			}
			got := rec.Header().Get("Location")
			if got != tt.wantLoc {
				t.Errorf("Location = %q, want %q", got, tt.wantLoc)
			}
			if !strings.HasPrefix(got, tt.prefix+"/") {
				t.Errorf("Location = %q, want it to start with %q", got, tt.prefix+"/")
			}
		})
	}
}

// TestRedirectFollowUpReachesIntendedHandler follows the rewritten Location the
// way a client would, and checks it resolves to the POST handler rather than
// the GET one.
func TestRedirectFollowUpReachesIntendedHandler(t *testing.T) {
	redirector := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, "/apps/a/users/u/sessions", http.StatusMovedPermanently)
	})
	first := callAPI(t, "/api", redirector, http.MethodPost, "/api/redirect-me")
	if first.Code != http.StatusPermanentRedirect {
		t.Fatalf("status = %d, want %d", first.Code, http.StatusPermanentRedirect)
	}

	stub := newStubREST()
	// 308 tells the client to repeat the request unchanged, so the follow-up is
	// still a POST.
	second := callAPI(t, "/api", stub, http.MethodPost, first.Header().Get("Location"))
	if second.Code != http.StatusOK {
		t.Errorf("follow-up status = %d, want %d", second.Code, http.StatusOK)
	}
	if stub.ran != "CreateSession" {
		t.Errorf("follow-up ran %q, want %q", stub.ran, "CreateSession")
	}
}

// TestPrefixIsSegmentAware covers PathPrefix("/api") capturing /apifoo.
func TestPrefixIsSegmentAware(t *testing.T) {
	tests := []struct {
		name        string
		target      string
		wantAPI     bool
		wantOutside bool
	}{
		{name: "prefix root", target: "/api", wantAPI: true},
		{name: "under prefix", target: "/api/list-apps", wantAPI: true},
		{name: "prefix as a word start", target: "/apifoo", wantOutside: true},
		{name: "prefix as a word start with path", target: "/apifoo/bar", wantOutside: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			stub := newStubREST()
			apiHit, outsideHit := false, false
			router := mux.NewRouter().StrictSlash(true)
			registerAPIRoutes(router, "/api", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				apiHit = true
				stub.ServeHTTP(w, r)
			}))
			router.PathPrefix("/apifoo").HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				outsideHit = true
			})

			rec := httptest.NewRecorder()
			router.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, tt.target, nil))

			if apiHit != tt.wantAPI {
				t.Errorf("API mount hit = %v, want %v (status %d, Location %q)", apiHit, tt.wantAPI, rec.Code, rec.Header().Get("Location"))
			}
			if outsideHit != tt.wantOutside {
				t.Errorf("route outside the mount hit = %v, want %v", outsideHit, tt.wantOutside)
			}
		})
	}
}

// TestSetupSubroutersServesRealRESTAPI drives the real registration path with
// the real ADK REST server, so the fixes are checked against the router the
// launcher actually mounts.
func TestSetupSubroutersServesRealRESTAPI(t *testing.T) {
	agnt, err := agent.New(agent.Config{
		Name: "HelloWorldAgent",
		Run: func(ic agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				event := session.NewEvent(ic, ic.InvocationID())
				event.Content = genai.NewContentFromText("hi", genai.RoleModel)
				yield(event, nil)
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New() error = %v", err)
	}

	l := NewLauncher()
	if _, err := l.Parse([]string{"-webui_address", "localhost:8080", "-path_prefix", "/api"}); err != nil {
		t.Fatalf("Parse() error = %v", err)
	}
	router := mux.NewRouter().StrictSlash(true)
	if err := l.SetupSubrouters(router, &launcher.Config{
		AgentLoader:    agent.NewSingleLoader(agnt),
		SessionService: session.InMemoryService(),
	}); err != nil {
		t.Fatalf("SetupSubrouters() error = %v", err)
	}

	tests := []struct {
		name   string
		method string
		target string
	}{
		{name: "GET list-apps", method: http.MethodGet, target: "/api/list-apps"},
		{name: "GET list-apps trailing slash", method: http.MethodGet, target: "/api/list-apps/"},
		{name: "HEAD list-apps", method: http.MethodHead, target: "/api/list-apps"},
		{name: "HEAD list-apps trailing slash", method: http.MethodHead, target: "/api/list-apps/"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rec := httptest.NewRecorder()
			router.ServeHTTP(rec, httptest.NewRequest(tt.method, tt.target, nil))
			if rec.Code != http.StatusOK {
				t.Errorf("status = %d, want %d (body %q, Location %q)", rec.Code, http.StatusOK, rec.Body.String(), rec.Header().Get("Location"))
			}
			if got := rec.Header().Get("Access-Control-Allow-Origin"); got != "http://localhost:8080" {
				t.Errorf("Access-Control-Allow-Origin = %q, want %q", got, "http://localhost:8080")
			}
		})
	}

	// A verb the path does not serve must be refused by the REST router, which
	// knows what that path supports, rather than by the mount, which does not.
	// The Allow header is the tell: only the REST router sets one.
	//
	// PURGE is here because the mount used to match on a fixed list of verbs.
	// Anything outside that list was refused by the mount with a bare 405 and
	// no Allow header at all, whatever the path underneath served.
	for _, method := range []string{http.MethodPatch, http.MethodPut, http.MethodDelete, "PURGE"} {
		t.Run("unsupported verb "+method+" is refused by the REST router", func(t *testing.T) {
			rec := httptest.NewRecorder()
			router.ServeHTTP(rec, httptest.NewRequest(method, "/api/list-apps", nil))
			if rec.Code != http.StatusMethodNotAllowed {
				t.Errorf("status = %d, want %d (404 or a 405 without Allow means the mount answered, not the REST router)",
					rec.Code, http.StatusMethodNotAllowed)
			}
			if got := rec.Header().Get("Allow"); got != "GET, HEAD" {
				t.Errorf("Allow = %q, want %q", got, "GET, HEAD")
			}
		})
	}
}
