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
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
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

const (
	testAppName  = "test-app"
	testUserID   = "test-user"
	testSessions = "/apps/" + testAppName + "/users/" + testUserID + "/sessions"
)

// newAssembledServer builds the whole ADK REST server the way a caller does,
// with an agent loader and a session service, and serves it over a real
// [httptest.Server].
//
// The transport matters here: net/http strips the body from a HEAD response in
// the server, not in the handler, so a [httptest.ResponseRecorder] would record
// a body that never reaches the wire.
func newAssembledServer(t *testing.T) *httptest.Server {
	t.Helper()

	rootAgent, err := agent.New(agent.Config{Name: testAppName, Description: "root agent"})
	if err != nil {
		t.Fatalf("agent.New() failed: %v", err)
	}
	server, err := NewServer(ServerConfig{
		SessionService: session.InMemoryService(),
		AgentLoader:    agent.NewSingleLoader(rootAgent),
		// The debug API is opt-in. These tests cover the assembled server the
		// web UI talks to, and the web UI is the debug tool, so they opt in.
		DebugAPIConfig: DebugAPIConfig{IncludeDebugAPI: true},
	})
	if err != nil {
		t.Fatalf("NewServer() failed: %v", err)
	}
	testServer := httptest.NewServer(server)
	t.Cleanup(testServer.Close)
	return testServer
}

// do sends one request to the test server and returns the response and its body.
func do(t *testing.T, testServer *httptest.Server, method, path string) (*http.Response, string) {
	t.Helper()

	req, err := http.NewRequestWithContext(t.Context(), method, testServer.URL+path, nil)
	if err != nil {
		t.Fatalf("http.NewRequestWithContext(%s %s) failed: %v", method, path, err)
	}
	resp, err := testServer.Client().Do(req)
	if err != nil {
		t.Fatalf("%s %s failed: %v", method, path, err)
	}
	body, readErr := io.ReadAll(resp.Body)
	closeErr := resp.Body.Close()
	if readErr != nil {
		t.Fatalf("%s %s reading body failed: %v", method, path, readErr)
	}
	if closeErr != nil {
		t.Fatalf("%s %s closing body failed: %v", method, path, closeErr)
	}
	return resp, string(body)
}

// TestHeadMatchesGet covers one route from each router. Every route declared
// GET alone, so gorilla/mux rejected HEAD before the handler ran and monitoring,
// link checkers and caches all saw 405.
func TestHeadMatchesGet(t *testing.T) {
	testServer := newAssembledServer(t)

	paths := []struct {
		name string
		path string
	}{
		{name: "apps", path: "/list-apps"},
		{name: "version", path: "/version"},
		{name: "sessions", path: testSessions},
		{name: "eval", path: "/dev/apps/" + testAppName + "/metrics-info"},
		{name: "agent graph", path: "/dev/apps/" + testAppName + "/build_graph"},
		{name: "debug", path: "/debug/trace/session/test-session"},
	}

	for _, tt := range paths {
		t.Run(tt.name, func(t *testing.T) {
			getResp, getBody := do(t, testServer, http.MethodGet, tt.path)
			if got, want := getResp.StatusCode, http.StatusOK; got != want {
				t.Fatalf("GET %s status = %d, want %d; body: %s", tt.path, got, want, getBody)
			}

			headResp, headBody := do(t, testServer, http.MethodHead, tt.path)
			if got, want := headResp.StatusCode, getResp.StatusCode; got != want {
				t.Errorf("HEAD %s status = %d, want the same as GET, %d", tt.path, got, want)
			}
			if headBody != "" {
				t.Errorf("HEAD %s body = %q, want it empty", tt.path, headBody)
			}
		})
	}
}

// TestOptionsDoesNotDeleteSession is the regression that matters most here.
// Several routes declared Methods(DELETE, OPTIONS), so a CORS preflight ran the
// delete handler and destroyed data on any server mounting adkrest without CORS
// middleware in front of it.
func TestOptionsDoesNotDeleteSession(t *testing.T) {
	testServer := newAssembledServer(t)
	sessionPath := testSessions + "/keep-me"

	createResp, createBody := do(t, testServer, http.MethodPost, sessionPath)
	if got, want := createResp.StatusCode, http.StatusOK; got != want {
		t.Fatalf("POST %s status = %d, want %d; body: %s", sessionPath, got, want, createBody)
	}

	optionsResp, optionsBody := do(t, testServer, http.MethodOptions, sessionPath)
	t.Logf("OPTIONS %s status = %d; body: %s", sessionPath, optionsResp.StatusCode, optionsBody)

	getResp, getBody := do(t, testServer, http.MethodGet, sessionPath)
	if got, want := getResp.StatusCode, http.StatusOK; got != want {
		t.Fatalf("GET %s after OPTIONS status = %d, want %d; the preflight deleted the session. Body: %s",
			sessionPath, got, want, getBody)
	}

	var got struct {
		ID string `json:"id"`
	}
	if err := json.Unmarshal([]byte(getBody), &got); err != nil {
		t.Fatalf("json.Unmarshal(%q) failed: %v", getBody, err)
	}
	if want := "keep-me"; got.ID != want {
		t.Errorf("GET %s after OPTIONS returned session id %q, want %q", sessionPath, got.ID, want)
	}
}

// TestMethodNotAllowedHasAllowHeader checks the header RFC 9110 requires on a
// 405. gorilla/mux omits it, which leaves a client unable to discover what the
// resource does support.
func TestMethodNotAllowedHasAllowHeader(t *testing.T) {
	testServer := newAssembledServer(t)

	tests := []struct {
		name        string
		method      string
		path        string
		wantAllowed []string
	}{
		{
			name:        "patch on version",
			method:      http.MethodPatch,
			path:        "/version",
			wantAllowed: []string{http.MethodGet, http.MethodHead},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			resp, body := do(t, testServer, tt.method, tt.path)
			if got, want := resp.StatusCode, http.StatusMethodNotAllowed; got != want {
				t.Fatalf("%s %s status = %d, want %d; body: %s", tt.method, tt.path, got, want, body)
			}
			allow := resp.Header.Get("Allow")
			if allow == "" {
				t.Fatalf("%s %s returned no Allow header, want one listing %v", tt.method, tt.path, tt.wantAllowed)
			}
			for _, method := range tt.wantAllowed {
				if !strings.Contains(allow, method) {
					t.Errorf("%s %s Allow = %q, want it to list %s", tt.method, tt.path, allow, method)
				}
			}
		})
	}
}

// probedMethods are the verbs the matrix sends at every path below.
var probedMethods = []string{
	http.MethodGet, http.MethodHead, http.MethodPost,
	http.MethodPut, http.MethodPatch, http.MethodDelete,
}

// parseAllow reads an Allow header into a sorted list, so a test can compare it
// against the exact set of methods a path serves rather than only checking that
// one method appears in it.
func parseAllow(allow string) []string {
	var got []string
	for _, method := range strings.Split(allow, ",") {
		if method = strings.TrimSpace(method); method != "" {
			got = append(got, method)
		}
	}
	slices.Sort(got)
	return got
}

// TestMethodNotAllowedMatrix sends every verb at paths that serve only some of
// them.
//
// Which mismatches produced a 405 used to depend on registration order.
// gorilla/mux records a method mismatch on the route that matched the path,
// then clears it as soon as a later route's method matcher succeeds, whatever
// path that route is on. DELETE /version therefore fell through to the
// NotFoundHandler and returned a bare 404, while PATCH /version returned 405
// only because the one PATCH route happens to be registered before /version.
func TestMethodNotAllowedMatrix(t *testing.T) {
	testServer := newAssembledServer(t)
	devApp := "/dev/apps/" + testAppName

	tests := []struct {
		name string
		path string
		// allowed is every method the path serves, and so the exact contents
		// of the Allow header on a 405.
		allowed []string
	}{
		{
			name:    "version",
			path:    "/version",
			allowed: []string{http.MethodGet, http.MethodHead},
		},
		{
			name:    "list-apps",
			path:    "/list-apps",
			allowed: []string{http.MethodGet, http.MethodHead},
		},
		{
			name:    "run",
			path:    "/run",
			allowed: []string{http.MethodPost},
		},
		{
			name:    "session collection",
			path:    testSessions,
			allowed: []string{http.MethodGet, http.MethodHead, http.MethodPost},
		},
		{
			name: "single session",
			path: testSessions + "/some-session",
			allowed: []string{
				http.MethodGet, http.MethodHead, http.MethodPost,
				http.MethodPatch, http.MethodDelete,
			},
		},
		{
			name:    "artifact",
			path:    testSessions + "/some-session/artifacts/report.txt",
			allowed: []string{http.MethodGet, http.MethodHead, http.MethodDelete},
		},
		{
			name:    "test case",
			path:    devApp + "/tests/some-test",
			allowed: []string{http.MethodGet, http.MethodHead, http.MethodPut, http.MethodDelete},
		},
		{
			name:    "eval case",
			path:    devApp + "/eval_sets/set-1/evals/case-1",
			allowed: []string{http.MethodGet, http.MethodHead, http.MethodPut, http.MethodDelete},
		},
	}

	for _, tt := range tests {
		want := slices.Sorted(slices.Values(tt.allowed))
		for _, method := range probedMethods {
			if slices.Contains(tt.allowed, method) {
				continue
			}
			t.Run(tt.name+"/"+method, func(t *testing.T) {
				resp, body := do(t, testServer, method, tt.path)
				if got := resp.StatusCode; got != http.StatusMethodNotAllowed {
					t.Fatalf("%s %s status = %d, want %d (404 means the router dropped the method mismatch); body: %s",
						method, tt.path, got, http.StatusMethodNotAllowed, body)
				}
				if got := parseAllow(resp.Header.Get("Allow")); !slices.Equal(got, want) {
					t.Errorf("%s %s Allow = %v, want exactly %v", method, tt.path, got, want)
				}
			})
		}
	}
}

// TestUnmatchedPathIsStillNotFound guards the other side of the fallback: a
// path no route serves must stay a 404, not become a 405 with an empty Allow.
func TestUnmatchedPathIsStillNotFound(t *testing.T) {
	testServer := newAssembledServer(t)

	for _, method := range probedMethods {
		t.Run(method, func(t *testing.T) {
			resp, _ := do(t, testServer, method, "/no-such-endpoint")
			if got := resp.StatusCode; got != http.StatusNotFound {
				t.Errorf("%s /no-such-endpoint status = %d, want %d", method, got, http.StatusNotFound)
			}
			if got := resp.Header.Get("Allow"); got != "" {
				t.Errorf("%s /no-such-endpoint Allow = %q, want no Allow header", method, got)
			}
		})
	}
}

// doNoRedirect sends one request without following redirects, so a redirect is
// visible to the test instead of being replayed as a GET by the client.
func doNoRedirect(t *testing.T, testServer *httptest.Server, method, path string) (*http.Response, string) {
	t.Helper()

	req, err := http.NewRequestWithContext(t.Context(), method, testServer.URL+path, nil)
	if err != nil {
		t.Fatalf("http.NewRequestWithContext(%s %s) failed: %v", method, path, err)
	}
	client := &http.Client{
		CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse },
	}
	resp, err := client.Do(req)
	if err != nil {
		t.Fatalf("%s %s failed: %v", method, path, err)
	}
	body, readErr := io.ReadAll(resp.Body)
	closeErr := resp.Body.Close()
	if readErr != nil {
		t.Fatalf("%s %s reading body failed: %v", method, path, readErr)
	}
	if closeErr != nil {
		t.Fatalf("%s %s closing body failed: %v", method, path, closeErr)
	}
	return resp, string(body)
}

// TestTrailingSlashKeepsMethodAndBody covers a bare [NewServer], with no
// launcher mount in front of it.
//
// A trailing slash used to be answered with a 301 to the same path without one.
// A client re-issues a redirected POST as a GET, so
// POST /apps/x/users/y/sessions/ came back as a session list and created
// nothing at all.
func TestTrailingSlashKeepsMethodAndBody(t *testing.T) {
	testServer := newAssembledServer(t)
	const sessionID = "slashed-session"
	path := testSessions + "/" + sessionID

	resp, body := doNoRedirect(t, testServer, http.MethodPost, path+"/")
	if got := resp.StatusCode; got != http.StatusOK {
		t.Fatalf("POST %s/ status = %d, want %d (a 3xx here is the redirect that drops the method and the body); body: %s",
			path, got, http.StatusOK, body)
	}

	getResp, getBody := do(t, testServer, http.MethodGet, path)
	if got := getResp.StatusCode; got != http.StatusOK {
		t.Fatalf("GET %s after the POST status = %d, want %d; the session was never created. Body: %s",
			path, got, http.StatusOK, getBody)
	}
	var got struct {
		ID string `json:"id"`
	}
	if err := json.Unmarshal([]byte(getBody), &got); err != nil {
		t.Fatalf("json.Unmarshal(%q) failed: %v", getBody, err)
	}
	if got.ID != sessionID {
		t.Errorf("GET %s returned session id %q, want %q", path, got.ID, sessionID)
	}
}

// TestTrailingSlashOnSafeAndRejectedMethods covers the rest of the trailing
// slash behaviour: a GET is served directly rather than redirected, and a verb
// the path does not serve is still refused with an Allow header.
func TestTrailingSlashOnSafeAndRejectedMethods(t *testing.T) {
	testServer := newAssembledServer(t)

	t.Run("GET is served without a redirect", func(t *testing.T) {
		resp, body := doNoRedirect(t, testServer, http.MethodGet, "/list-apps/")
		if got := resp.StatusCode; got != http.StatusOK {
			t.Fatalf("GET /list-apps/ status = %d, want %d; body: %s", got, http.StatusOK, body)
		}
		var apps []string
		if err := json.Unmarshal([]byte(body), &apps); err != nil {
			t.Fatalf("json.Unmarshal(%q) failed: %v", body, err)
		}
		if !slices.Contains(apps, testAppName) {
			t.Errorf("GET /list-apps/ returned %v, want it to list %q", apps, testAppName)
		}
	})

	t.Run("unsupported verb is refused", func(t *testing.T) {
		resp, body := doNoRedirect(t, testServer, http.MethodDelete, "/version/")
		if got := resp.StatusCode; got != http.StatusMethodNotAllowed {
			t.Fatalf("DELETE /version/ status = %d, want %d; body: %s", got, http.StatusMethodNotAllowed, body)
		}
		want := []string{http.MethodGet, http.MethodHead}
		if got := parseAllow(resp.Header.Get("Allow")); !slices.Equal(got, want) {
			t.Errorf("DELETE /version/ Allow = %v, want exactly %v", got, want)
		}
	})
}

// TestOptionsDescribesTheResource pins the answer to a preflight on a resource
// that exists.
//
// No route declares OPTIONS, because a route that did would run its mutating
// handler on a preflight — that is how OPTIONS used to delete sessions. But
// answering 405 while listing every verb except the one asked for is
// self-contradictory, so the fallback answers it: 204 with an Allow header that
// includes OPTIONS.
func TestOptionsDescribesTheResource(t *testing.T) {
	testServer := newAssembledServer(t)

	resp, body := do(t, testServer, http.MethodOptions, testSessions+"/some-session")
	if got, want := resp.StatusCode, http.StatusNoContent; got != want {
		t.Errorf("OPTIONS status = %d, want %d; body: %s", got, want, body)
	}
	allow := parseAllow(resp.Header.Get("Allow"))
	want := []string{http.MethodDelete, http.MethodGet, http.MethodHead, http.MethodOptions, http.MethodPatch, http.MethodPost}
	if !slices.Equal(allow, want) {
		t.Errorf("Allow = %v, want %v", allow, want)
	}
}
