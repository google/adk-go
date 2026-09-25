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
	"errors"
	"fmt"
	"iter"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/gorilla/mux"
	"github.com/gorilla/websocket"
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

func TestCORSHeaders(t *testing.T) {
	tests := []struct {
		name          string
		addr          string
		allowed       []string
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
			name:          "unlisted request Origin is not reflected",
			addr:          "localhost:8080",
			allowed:       []string{"https://ui.example.com"},
			requestOrigin: "http://evil.example.com",
			wantOrigin:    "http://localhost:8080",
			wantVary:      "Origin",
		},
		{
			name:          "allowed request Origin is reflected",
			addr:          "localhost:8080",
			allowed:       []string{"https://ui.example.com"},
			requestOrigin: "https://ui.example.com",
			wantOrigin:    "https://ui.example.com",
			wantVary:      "Origin",
		},
		{
			name:          "allowed bare host is read as http",
			addr:          "localhost:8080",
			allowed:       []string{"ui.example.com:3000"},
			requestOrigin: "http://ui.example.com:3000",
			wantOrigin:    "http://ui.example.com:3000",
			wantVary:      "Origin",
		},
		{
			name:          "star in the allowed origins allows every origin",
			addr:          "localhost:8080",
			allowed:       []string{"*"},
			requestOrigin: "https://any.example.com",
			wantOrigin:    "*",
			wantVary:      "",
		},
		{
			// A reflected origin would need Vary, and "*" already admits it.
			name:          "star web UI address wins over an allowed origin",
			addr:          "*",
			allowed:       []string{"https://ui.example.com"},
			requestOrigin: "https://ui.example.com",
			wantOrigin:    "*",
			wantVary:      "",
		},
		{
			name:          "empty web UI address still reflects an allowed origin",
			addr:          "",
			allowed:       []string{"https://ui.example.com"},
			requestOrigin: "https://ui.example.com",
			wantOrigin:    "https://ui.example.com",
			wantVary:      "Origin",
		},
		{
			// With no web UI origin there is no fallback header, but an
			// allowed origin would get one, so the response still varies.
			name:          "empty web UI address still varies for an unlisted origin",
			addr:          "",
			allowed:       []string{"https://ui.example.com"},
			requestOrigin: "http://evil.example.com",
			wantOrigin:    "",
			wantVary:      "Origin",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			called := false
			h := corsWithArgs(tt.addr, tt.allowed)(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
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
			if gotOrigin != "" && gotOrigin != "*" && !strings.Contains(gotOrigin, "://") {
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
	h := corsWithArgs("localhost:8080", nil)(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
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

// TestWebSocketUpgradeThroughMount covers /run_live, which takes over the
// connection instead of writing a response.
//
// The prefixed mount wraps the ResponseWriter to rewrite redirects, and
// gorilla/websocket type-asserts that writer to http.Hijacker directly rather
// than following Unwrap. A wrapper missing Hijack therefore turns every
// upgrade into a 500, on the default /api prefix but not on an empty one.
// httptest.NewServer is used rather than a recorder because only a real
// connection can be hijacked.
func TestWebSocketUpgradeThroughMount(t *testing.T) {
	for _, prefix := range []string{"/api", ""} {
		t.Run("prefix="+prefix, func(t *testing.T) {
			upgrader := websocket.Upgrader{}
			echo := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				conn, err := upgrader.Upgrade(w, r, nil)
				if err != nil {
					// Upgrade has already written the error response.
					return
				}
				defer func() { _ = conn.Close() }()
				_ = conn.WriteMessage(websocket.TextMessage, []byte("live"))
			})

			router := mux.NewRouter().StrictSlash(true)
			inner := mux.NewRouter().StrictSlash(true)
			inner.Path("/run_live").Handler(echo)
			registerAPIRoutes(router, prefix, inner)

			srv := httptest.NewServer(router)
			defer srv.Close()

			url := "ws" + strings.TrimPrefix(srv.URL, "http") + prefix + "/run_live"
			conn, resp, err := websocket.DefaultDialer.Dial(url, nil)
			if err != nil {
				status := "no response"
				if resp != nil {
					status = resp.Status
				}
				t.Fatalf("Dial(%s) error = %v (%s), want a successful upgrade", url, err, status)
			}
			defer func() { _ = conn.Close() }()

			// Without a deadline, an upgrade that succeeds but never sends a
			// frame hangs until the package timeout, taking every other test in
			// the package down with it instead of failing this one.
			if err := conn.SetReadDeadline(time.Now().Add(10 * time.Second)); err != nil {
				t.Fatalf("SetReadDeadline() error = %v", err)
			}

			_, msg, err := conn.ReadMessage()
			if err != nil {
				t.Fatalf("ReadMessage() error = %v, want the server's frame", err)
			}
			if got := string(msg); got != "live" {
				t.Errorf("ReadMessage() = %q, want %q", got, "live")
			}
		})
	}
}

// TestHijackReportsNotSupported covers the branch taken when the writer
// underneath the mount cannot be hijacked.
//
// The wrapper has to report that as http.ErrNotSupported rather than a bare
// error, because http.ResponseController finds this method before it follows
// Unwrap. A caller asking errors.Is(err, http.ErrNotSupported) would otherwise
// get a different answer purely because the mount is in the way.
func TestHijackReportsNotSupported(t *testing.T) {
	// httptest.ResponseRecorder implements no Hijacker, which is the case.
	w := &redirectRewriter{ResponseWriter: httptest.NewRecorder(), prefix: "/api"}

	conn, brw, err := w.Hijack()

	if err == nil {
		t.Fatal("Hijack() error = nil, want a failure on a writer that cannot be hijacked")
	}
	if conn != nil || brw != nil {
		t.Errorf("Hijack() = (%v, %v), want both nil alongside the error", conn, brw)
	}
	if !errors.Is(err, http.ErrNotSupported) {
		t.Errorf("Hijack() error = %v, want it to wrap http.ErrNotSupported", err)
	}
}

// TestRunLiveUpgradesThroughTheRealMount drives the actual REST server rather
// than a stub inner router.
//
// The stub version proves the mount can carry an upgrade. This proves the
// endpoint that was broken can. The session does not exist, so the server
// accepts the upgrade and then closes; what matters is that the handshake
// completes at all, which is what returned 500 before Hijack was forwarded.
func TestRunLiveUpgradesThroughTheRealMount(t *testing.T) {
	agnt, err := agent.New(agent.Config{
		Name: "HelloWorldAgent",
		Run: func(ic agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {}
		},
	})
	if err != nil {
		t.Fatalf("agent.New() error = %v", err)
	}

	l := NewLauncher()
	if _, err := l.Parse([]string{"-path_prefix", "/api"}); err != nil {
		t.Fatalf("Parse() error = %v", err)
	}
	router := mux.NewRouter().StrictSlash(true)
	if err := l.SetupSubrouters(router, &launcher.Config{
		AgentLoader:    agent.NewSingleLoader(agnt),
		SessionService: session.InMemoryService(),
	}); err != nil {
		t.Fatalf("SetupSubrouters() error = %v", err)
	}

	srv := httptest.NewServer(router)
	defer srv.Close()

	url := "ws" + strings.TrimPrefix(srv.URL, "http") +
		"/api/run_live?app_name=HelloWorldAgent&user_id=u&session_id=does-not-exist"
	conn, resp, err := websocket.DefaultDialer.Dial(url, nil)
	if err != nil {
		status := "no response"
		if resp != nil {
			status = resp.Status
		}
		t.Fatalf("Dial(%s) error = %v (%s), want the handshake to complete", url, err, status)
	}
	defer func() { _ = conn.Close() }()

	if resp.StatusCode != http.StatusSwitchingProtocols {
		t.Errorf("handshake status = %d, want %d", resp.StatusCode, http.StatusSwitchingProtocols)
	}

	// The server closes because the session is missing. Reading that close is
	// what confirms the connection was live, and the deadline keeps a silent
	// server from hanging the package.
	if err := conn.SetReadDeadline(time.Now().Add(10 * time.Second)); err != nil {
		t.Fatalf("SetReadDeadline() error = %v", err)
	}
	if _, _, err := conn.ReadMessage(); err == nil {
		t.Log("server sent a frame before closing, which is also fine")
	}
}

// TestSetupSubroutersPassesWebUIOriginToRESTServer pins that -webui_address
// reaches the REST server's origin check, not only the CORS header. The header
// alone tells a browser it may read the response; without the same value on the
// check, the request never gets one.
func TestSetupSubroutersPassesWebUIOriginToRESTServer(t *testing.T) {
	const devUI = "http://localhost:4200"

	agnt, err := agent.New(agent.Config{Name: "HelloWorldAgent"})
	if err != nil {
		t.Fatalf("agent.New() error = %v", err)
	}
	l := NewLauncher()
	if _, err := l.Parse([]string{"-webui_address", devUI, "-path_prefix", "/api"}); err != nil {
		t.Fatalf("Parse() error = %v", err)
	}
	router := mux.NewRouter().StrictSlash(true)
	if err := l.SetupSubrouters(router, &launcher.Config{
		AgentLoader:    agent.NewSingleLoader(agnt),
		SessionService: session.InMemoryService(),
	}); err != nil {
		t.Fatalf("SetupSubrouters() error = %v", err)
	}

	for _, tc := range []struct {
		name       string
		origin     string
		wantStatus int
	}{
		{name: "configured web UI origin", origin: devUI, wantStatus: http.StatusOK},
		{name: "any other origin", origin: "http://evil.com", wantStatus: http.StatusForbidden},
	} {
		t.Run(tc.name, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodGet, "/api/list-apps", nil)
			req.Header.Set("Origin", tc.origin)
			rec := httptest.NewRecorder()
			router.ServeHTTP(rec, req)

			if rec.Code != tc.wantStatus {
				t.Errorf("status = %d, want %d (body %q)", rec.Code, tc.wantStatus, rec.Body.String())
			}
		})
	}
}

// TestSetupSubroutersPassesBindHostToRESTServer pins that the launcher's bind
// address reaches the REST server's Host check.
//
// That check is the only one that sees a rebound page's same-origin GET: a
// browser sends no Origin header on one, so the Host it names is the only tell.
// It arms on a declared loopback bind and stays off without one, so a dropped
// BindHost leaves the request served rather than failing anywhere visible.
func TestSetupSubroutersPassesBindHostToRESTServer(t *testing.T) {
	agnt, err := agent.New(agent.Config{Name: "HelloWorldAgent"})
	if err != nil {
		t.Fatalf("agent.New() error = %v", err)
	}

	for _, tc := range []struct {
		name       string
		bindHost   string
		host       string
		wantStatus int
	}{
		{
			name:       "rebound host on a loopback bind",
			bindHost:   "127.0.0.1",
			host:       "evil.com:8080",
			wantStatus: http.StatusForbidden,
		},
		{
			name:       "loopback host on a loopback bind",
			bindHost:   "127.0.0.1",
			host:       "localhost:8080",
			wantStatus: http.StatusOK,
		},
		{
			// A routable bind is an operator exposing the server on purpose,
			// so it is reachable under whatever name resolves to it.
			name:       "rebound host on an all-interfaces bind",
			bindHost:   "0.0.0.0",
			host:       "evil.com:8080",
			wantStatus: http.StatusOK,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			l := NewLauncher()
			if _, err := l.Parse([]string{"-path_prefix", "/api"}); err != nil {
				t.Fatalf("Parse() error = %v", err)
			}
			router := mux.NewRouter().StrictSlash(true)
			if err := l.SetupSubrouters(router, &launcher.Config{
				AgentLoader:    agent.NewSingleLoader(agnt),
				SessionService: session.InMemoryService(),
				BindHost:       tc.bindHost,
			}); err != nil {
				t.Fatalf("SetupSubrouters() error = %v", err)
			}

			req := httptest.NewRequest(http.MethodGet, "/api/list-apps", nil)
			req.Host = tc.host
			rec := httptest.NewRecorder()
			router.ServeHTTP(rec, req)

			if rec.Code != tc.wantStatus {
				t.Errorf("status = %d, want %d (body %q)", rec.Code, tc.wantStatus, rec.Body.String())
			}
		})
	}
}

// TestSetupSubroutersOrigins pins the two directions the allowed origins move
// in: this launcher adds the web UI origin to the server-wide list, and the
// REST server and its CORS headers honor the whole list rather than that one
// entry.
//
// Without the second half, an origin the operator allowed server-wide passes
// the launcher's guard and is then refused inside the REST server, or admitted
// but unable to read the response, on the API alone.
func TestSetupSubroutersOrigins(t *testing.T) {
	agnt, err := agent.New(agent.Config{Name: "HelloWorldAgent"})
	if err != nil {
		t.Fatalf("agent.New() error = %v", err)
	}

	for _, tc := range []struct {
		name string
		// configured is what the operator allowed server-wide, as the web
		// launcher would have collected it before this launcher runs.
		configured []string
		origin     string
		wantStatus int
		wantCORS   string // Access-Control-Allow-Origin
	}{
		{
			name:       "the web UI origin is served",
			origin:     "http://localhost:8080",
			wantStatus: http.StatusOK,
			wantCORS:   "http://localhost:8080",
		},
		{
			name:       "a server-wide origin is served and may read the response",
			configured: []string{"https://ui.example.com"},
			origin:     "https://ui.example.com",
			wantStatus: http.StatusOK,
			wantCORS:   "https://ui.example.com",
		},
		{
			name:       "an unlisted origin is refused",
			configured: []string{"https://ui.example.com"},
			origin:     "https://evil.example.com",
			wantStatus: http.StatusForbidden,
			wantCORS:   "http://localhost:8080",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			l := NewLauncher()
			if _, err := l.Parse([]string{"-path_prefix", "/api", "-webui_address", "localhost:8080"}); err != nil {
				t.Fatalf("Parse() error = %v", err)
			}
			config := &launcher.Config{
				AgentLoader:    agent.NewSingleLoader(agnt),
				SessionService: session.InMemoryService(),
				BindHost:       "127.0.0.1",
				AllowedOrigins: tc.configured,
			}
			router := mux.NewRouter().StrictSlash(true)
			if err := l.SetupSubrouters(router, config); err != nil {
				t.Fatalf("SetupSubrouters() error = %v", err)
			}

			if !slices.Contains(config.AllowedOrigins, "localhost:8080") {
				t.Errorf("AllowedOrigins = %q, want it to contain the -webui_address value", config.AllowedOrigins)
			}

			// A Host that differs from every Origin here, so that no request
			// passes as same-origin and each result comes from the list alone.
			req := httptest.NewRequest(http.MethodGet, "/api/list-apps", nil)
			req.Host = "127.0.0.1:8080"
			req.Header.Set("Origin", tc.origin)
			rec := httptest.NewRecorder()
			router.ServeHTTP(rec, req)

			if rec.Code != tc.wantStatus {
				t.Errorf("GET /api/list-apps with Origin %q = %d, want %d", tc.origin, rec.Code, tc.wantStatus)
			}
			if got := rec.Header().Get("Access-Control-Allow-Origin"); got != tc.wantCORS {
				t.Errorf("GET /api/list-apps with Origin %q: Access-Control-Allow-Origin = %q, want %q", tc.origin, got, tc.wantCORS)
			}
		})
	}
}
