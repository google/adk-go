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

// This file guards the contract between the prebuilt Angular UI embedded at
// cmd/launcher/web/webui/distr and the routes this server registers.
//
// On 2026-06-30 that bundle was refreshed from google/adk-web and silently
// picked up an upstream breaking change: ADK v2 had moved 30 developer
// endpoints under a /dev/apps/{app_name}/ prefix. The Go server kept serving
// the old paths, most of the UI 404ed in the browser for two months, and
// nothing caught it, because the bundle is inert //go:embed data that no test
// reads. The test below closes that hole: it proves every endpoint in the
// golden list actually routes. The golden list must be updated deliberately
// when the bundle is refreshed.

package adkrest_test

import (
	"net/http"
	"net/http/httptest"
	"regexp"
	"strings"
	"testing"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/artifact"
	"google.golang.org/adk/memory"
	"google.golang.org/adk/server/adkrest"
	"google.golang.org/adk/session"
)

// uiEndpoint is one request the ADK web UI makes.
type uiEndpoint struct {
	// method is the HTTP method the UI uses.
	method string
	// template is the URL path as the Angular code builds it, with named
	// placeholders standing in for the interpolated values. Half 2 compares it
	// against the bundle after both sides are reduced by canonicalTemplate.
	template string
	// note records why an entry needs explaining. Optional.
	note string
	// webSocket marks the one endpoint the UI opens as a WebSocket rather than
	// an XHR, so it is built from getWSServerUrl() and not apiServerDomain.
	webSocket bool
}

// uiGolden is every endpoint the embedded ADK web UI calls: 39 method/template
// pairs over 30 distinct path templates, derived from the 42 request sites in
// the shipped bundle.
//
// Adding a route to the server is not enough on its own, and neither is adding
// an entry here. A refreshed bundle that calls something new fails half 2 until
// the route exists and the entry is added.
var uiGolden = []uiEndpoint{
	// Endpoints the UI calls without the developer prefix. These are the
	// production API and their paths did not move at v2.
	{method: http.MethodPost, template: "/run_sse"},
	{method: http.MethodGet, template: "/list-apps"},
	{method: http.MethodGet, template: "/version"},
	{method: http.MethodGet, template: "/apps/{app_name}/users/{user_id}/sessions"},
	{method: http.MethodPost, template: "/apps/{app_name}/users/{user_id}/sessions"},
	{method: http.MethodGet, template: "/apps/{app_name}/users/{user_id}/sessions/{session_id}"},
	{
		method:   http.MethodPatch,
		template: "/apps/{app_name}/users/{user_id}/sessions/{session_id}",
		note: "the UI's updateSession(), which it uses to rename a session and to write session state. " +
			"Served by UpdateSessionHandler; before that route existed this answered 405 and renaming a session " +
			"silently did nothing, because the UI subscribes with an empty error callback and swallows it.",
	},
	{method: http.MethodDelete, template: "/apps/{app_name}/users/{user_id}/sessions/{session_id}"},
	{method: http.MethodGet, template: "/apps/{app_name}/users/{user_id}/sessions/{session_id}/artifacts/{artifact_name}"},
	{method: http.MethodGet, template: "/apps/{app_name}/users/{user_id}/sessions/{session_id}/artifacts/{artifact_name}/versions/{version}"},
	{
		method:    http.MethodGet,
		template:  "/run_live",
		webSocket: true,
		note:      "opened as a WebSocket; a plain GET cannot complete the handshake, so half 1 only checks that the route exists",
	},

	// Developer endpoints. Everything below moved under /dev/apps/{app_name}
	// at ADK v2; serving any of them at the pre-v2 path leaves it unreachable
	// from the UI. This is the group the 2026-06-30 refresh broke.
	{method: http.MethodGet, template: "/dev/apps/{app_name}/build_graph"},
	{method: http.MethodGet, template: "/dev/apps/{app_name}/build_graph_image"},
	{method: http.MethodGet, template: "/dev/apps/{app_name}/builder"},
	{method: http.MethodPost, template: "/dev/apps/{app_name}/builder/save"},
	{method: http.MethodPost, template: "/dev/apps/{app_name}/builder/cancel"},
	{method: http.MethodGet, template: "/dev/apps/{app_name}/eval_sets"},
	{
		method:   http.MethodPost,
		template: "/dev/apps/{app_name}/eval-sets",
		note:     "the UI creates an eval set at the hyphenated spelling and reads it back at the underscored one",
	},
	{method: http.MethodGet, template: "/dev/apps/{app_name}/eval_sets/{eval_set_id}"},
	{method: http.MethodDelete, template: "/dev/apps/{app_name}/eval_sets/{eval_set_id}"},
	{method: http.MethodGet, template: "/dev/apps/{app_name}/eval_sets/{eval_set_id}/evals"},
	{method: http.MethodGet, template: "/dev/apps/{app_name}/eval_sets/{eval_set_id}/evals/{eval_case_id}"},
	{method: http.MethodPut, template: "/dev/apps/{app_name}/eval_sets/{eval_set_id}/evals/{eval_case_id}"},
	{method: http.MethodDelete, template: "/dev/apps/{app_name}/eval_sets/{eval_set_id}/evals/{eval_case_id}"},
	{method: http.MethodPost, template: "/dev/apps/{app_name}/eval_sets/{eval_set_id}/add_session"},
	{method: http.MethodPost, template: "/dev/apps/{app_name}/eval_sets/{eval_set_id}/run_eval"},
	{method: http.MethodGet, template: "/dev/apps/{app_name}/eval_results"},
	{method: http.MethodGet, template: "/dev/apps/{app_name}/eval_results/{eval_result_id}"},
	{method: http.MethodGet, template: "/dev/apps/{app_name}/metrics-info"},
	{method: http.MethodGet, template: "/dev/apps/{app_name}/debug/trace/{event_id}"},
	{method: http.MethodGet, template: "/dev/apps/{app_name}/debug/trace/session/{session_id}"},
	{method: http.MethodGet, template: "/dev/apps/{app_name}/users/{user_id}/sessions/{session_id}/events/{event_id}/graph"},
	{method: http.MethodGet, template: "/dev/apps/{app_name}/tests"},
	{method: http.MethodGet, template: "/dev/apps/{app_name}/tests/{test_name}"},
	{method: http.MethodPut, template: "/dev/apps/{app_name}/tests/{test_name}"},
	{method: http.MethodDelete, template: "/dev/apps/{app_name}/tests/{test_name}"},
	{method: http.MethodPost, template: "/dev/apps/{app_name}/tests/rebuild"},
	{method: http.MethodPost, template: "/dev/apps/{app_name}/tests/run"},
}

// uiContractApp is the name of the agent the half 1 server loads, and the value
// substituted for {app_name}.
const uiContractApp = "ui_contract_agent"

// uiPathValues gives each placeholder in uiGolden a concrete value.
//
// {test_name} deliberately avoids "rebuild" and "run": those are literal
// sibling routes, and using one as a test name would silently exercise the
// wrong route.
var uiPathValues = map[string]string{
	"app_name":       uiContractApp,
	"user_id":        "ui-contract-user",
	"session_id":     "ui-contract-session",
	"artifact_name":  "ui-contract-artifact",
	"version":        "0",
	"event_id":       "ui-contract-event",
	"eval_set_id":    "ui-contract-eval-set",
	"eval_case_id":   "ui-contract-eval-case",
	"eval_result_id": "ui-contract-eval-result",
	"test_name":      "ui-contract-test",
}

// placeholderRE matches one {name} placeholder in a golden template.
var placeholderRE = regexp.MustCompile(`\{[a-z_]+\}`)

// concretePath substitutes a real value for every placeholder in a golden
// template. An unrecognised placeholder is a test bug, not a server bug: left
// alone it would be sent literally, which still routes and would hide the
// mistake.
func concretePath(t *testing.T, tmpl string) string {
	t.Helper()
	return placeholderRE.ReplaceAllStringFunc(tmpl, func(ph string) string {
		name := strings.Trim(ph, "{}")
		value, ok := uiPathValues[name]
		if !ok {
			t.Fatalf("golden template %q uses placeholder %q with no value in uiPathValues; add one", tmpl, ph)
		}
		return value
	})
}

// TestUIContractEveryUIEndpointRoutes is half 1: every endpoint the web UI
// calls reaches a handler on a real server.
//
// It asserts routing and nothing else. 200, 400, 500 and 501 are all answers
// from a route that exists, and an unimplemented developer endpoint answering
// 501 is the intended behaviour, not a failure. Exactly two responses mean the
// request never reached a handler: 405, and the plain-text 404 the mux writes
// for an unknown path. A handler's own 404 ("no such session") carries a JSON
// body and passes.
func TestUIContractEveryUIEndpointRoutes(t *testing.T) {
	server := newUIContractServer(t)

	for _, endpoint := range uiGolden {
		path := concretePath(t, endpoint.template)
		name := endpoint.method + " " + endpoint.template
		t.Run(name, func(t *testing.T) {
			request := httptest.NewRequestWithContext(t.Context(), endpoint.method, path, strings.NewReader("{}"))
			request.Header.Set("Content-Type", "application/json")
			if endpoint.webSocket {
				// The UI sends these three as query parameters. A
				// httptest.ResponseRecorder cannot be hijacked, so the
				// handshake fails whatever we send; the point is only that the
				// request reaches the handler at all.
				request.URL.RawQuery = "app_name=" + uiContractApp + "&user_id=u&session_id=s"
			}
			recorder := httptest.NewRecorder()
			server.ServeHTTP(recorder, request)

			body := recorder.Body.String()
			if recorder.Code == http.StatusMethodNotAllowed {
				t.Fatalf("%s %s: server answered 405 Method Not Allowed, so the path is routed but not for this method.\n"+
					"The embedded web UI makes this exact request, so it is broken in the browser.\n"+
					"Fix: register %s for %q in server/adkrest/internal/routers.\nAllow header: %q%s",
					endpoint.method, path, endpoint.method, endpoint.template,
					recorder.Header().Get("Allow"), noteSuffix(endpoint))
			}
			if isMuxNotFound(recorder.Code, body) {
				t.Fatalf("%s %s: server answered the mux's %q, so no route matches this path at all.\n"+
					"The embedded web UI makes this exact request, so it is broken in the browser.\n"+
					"Fix: register %q in server/adkrest/internal/routers.%s",
					endpoint.method, path, strings.TrimSpace(body), endpoint.template, noteSuffix(endpoint))
			}
			t.Logf("%s %s -> %d", endpoint.method, path, recorder.Code)
		})
	}
}

// muxNotFoundBody is what gorilla/mux's default NotFoundHandler writes. It is
// the only 404 that means "no route matched"; every 404 a handler writes itself
// carries a JSON body.
const muxNotFoundBody = "404 page not found"

func isMuxNotFound(code int, body string) bool {
	return code == http.StatusNotFound && strings.TrimSpace(body) == muxNotFoundBody
}

func noteSuffix(endpoint uiEndpoint) string {
	if endpoint.note == "" {
		return ""
	}
	return "\nNote: " + endpoint.note
}

// newUIContractServer builds a real server with in-memory services, so half 1
// exercises the same routing table production uses.
func newUIContractServer(t *testing.T) *adkrest.Server {
	t.Helper()

	// main roots this at a graph workflow agent; 1.x has no graph engine, and
	// the routing table this test asserts does not depend on the agent's kind.
	rootAgent, err := agent.New(agent.Config{Name: uiContractApp})
	if err != nil {
		t.Fatalf("agent.New() failed: %v", err)
	}

	server, err := adkrest.NewServer(adkrest.ServerConfig{
		SessionService:  session.InMemoryService(),
		ArtifactService: artifact.InMemoryService(),
		MemoryService:   memory.InMemoryService(),
		AgentLoader:     agent.NewSingleLoader(rootAgent),
	})
	if err != nil {
		t.Fatalf("adkrest.NewServer() failed: %v", err)
	}
	return server
}

// Half 2 of this test on main reads the shipped Angular bundle and requires the
// golden list above to match every API path it builds. It is not carried over:
// 1.x ships an older bundle whose minified code constructs URLs differently, so
// the patterns extract nothing and the assertion would be about main's bundle
// rather than this branch's. Half 1 above still covers what the backport is
// for -- that every endpoint the UI calls reaches a handler.
