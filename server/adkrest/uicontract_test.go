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
// reads. The two tests below close that hole: one proves every endpoint in the
// golden list below actually routes, the other reads the shipped bundle from
// disk and proves the golden list still lists everything the bundle calls. Both
// halves must be updated deliberately when the bundle is refreshed.

package adkrest_test

import (
	"maps"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
	"testing"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/workflowagent"
	"google.golang.org/adk/v2/artifact"
	"google.golang.org/adk/v2/memory"
	"google.golang.org/adk/v2/server/adkrest"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/workflow"
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

// canonicalTemplate reduces a path template to the form both halves compare on:
// query string dropped, every placeholder collapsed to "{}".
//
// The bundle is minified, so its interpolated variables are renamed on every
// build. Only the shape of the path is stable, so only the shape is compared.
func canonicalTemplate(tmpl string) string {
	path, _, _ := strings.Cut(tmpl, "?")
	return placeholderRE.ReplaceAllString(path, "{}")
}

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

	echo := workflow.NewFunctionNode("echo",
		func(_ agent.Context, in string) (string, error) { return in, nil },
		workflow.NodeConfig{},
	)
	rootAgent, err := workflowagent.New(workflowagent.Config{
		Name:  uiContractApp,
		Edges: workflow.Chain(workflow.Start, echo),
	})
	if err != nil {
		t.Fatalf("workflowagent.New() failed: %v", err)
	}

	server, err := adkrest.NewServer(adkrest.ServerConfig{
		SessionService:  session.InMemoryService(),
		ArtifactService: artifact.InMemoryService(),
		MemoryService:   memory.InMemoryService(),
		AgentLoader:     agent.NewSingleLoader(rootAgent),
		// Three of the UI's endpoints are debug-trace routes, which are opt-in
		// since #1413. The UI is a developer tool and is run with the flag on,
		// so this asserts the contract a developer actually gets. Without it,
		// those three would look unrouted and this test would report the
		// gate as a regression.
		DebugAPIConfig: adkrest.DebugAPIConfig{IncludeDebugAPI: true},
	})
	if err != nil {
		t.Fatalf("adkrest.NewServer() failed: %v", err)
	}
	return server
}

// bundleDir is the embedded web UI, relative to this package.
const bundleDir = "../../cmd/launcher/web/webui/distr"

// bundleCallSite describes one family of API calls in the Angular bundle.
//
// The minifier renames every local variable on each build, so these match on
// the two things that survive: the field the base URL is read from, and the
// shape of the path appended to it. minMatches is a floor, not an exact count;
// it exists so a bundle refresh that changes how URLs are built fails loudly
// instead of extracting nothing and quietly passing.
type bundleCallSite struct {
	name       string
	pattern    *regexp.Regexp
	minMatches int
	// fixedMethod is the verb for a site whose method is not written next to
	// the URL. Only the WebSocket site needs it; everywhere else the verb is
	// read out of the bundle by verbAfterURL.
	fixedMethod string
}

// urlChars matches the body of a URL inside a template literal: literal URL
// characters, or a ${...} interpolation.
const urlChars = "(?:[\\w\\-./?=&]|\\$\\{[^{}`]*\\})*"

// bundleCallSites covers the three shapes the UI builds an XHR URL in, plus the
// one WebSocket URL, which is built from getWSServerUrl() rather than
// apiServerDomain because it needs a bare host without a scheme.
var bundleCallSites = []bundleCallSite{
	{
		// `${this.apiServerDomain}/dev/apps/${A}/tests`
		name:       "template literal interpolating apiServerDomain",
		pattern:    regexp.MustCompile("`\\$\\{[\\w$.]+\\.apiServerDomain\\}(" + urlChars + ")`"),
		minMatches: 5,
	},
	{
		// this.apiServerDomain+`/dev/apps/${A}/eval_sets`
		name:       "apiServerDomain concatenated with a template literal",
		pattern:    regexp.MustCompile("\\.apiServerDomain\\s*\\+\\s*`(" + urlChars + ")`"),
		minMatches: 28,
	},
	{
		// this.apiServerDomain+"/run_sse"
		name:       "apiServerDomain concatenated with a string literal",
		pattern:    regexp.MustCompile(`\.apiServerDomain\s*\+\s*"([\w\-./?=&]*)"`),
		minMatches: 3,
	},
	{
		// `ws://${Ur.getWSServerUrl()}/run_live?app_name=...`
		name:        "WebSocket URL built from getWSServerUrl",
		pattern:     regexp.MustCompile("\\$\\{[\\w$.]+\\.getWSServerUrl\\(\\)\\}(" + urlChars + ")`"),
		minMatches:  1,
		fixedMethod: http.MethodGet,
	},
}

// minBundleTemplates is a floor on the total distinct templates extracted. A
// bundle that yields fewer than this has either been replaced by something that
// is not the ADK web UI, or is being read with regexes that no longer fit.
// Either way the extraction is not trustworthy and must fail rather than skip.
const minBundleTemplates = 10

// TestUIContractGoldenListMatchesBundle is half 2: the golden list is still in
// step with the bundle actually shipped.
//
// It reads cmd/launcher/web/webui/distr from disk, extracts every API path the
// Angular code builds, and requires each to be in uiGolden. A bundle refresh
// that introduces an endpoint therefore fails here, at the point the bundle
// changes, instead of shipping as a 404 nobody sees until a user opens the tab.
//
// It fails, never skips, on a missing bundle or an extraction that comes back
// empty. Skipping is how the original incident went unnoticed for two months.
func TestUIContractGoldenListMatchesBundle(t *testing.T) {
	source, bundlePath := readUIBundle(t)
	bundleTemplates := extractBundleTemplates(t, source, bundlePath)

	golden := map[string]map[string]bool{}
	for _, endpoint := range uiGolden {
		key := canonicalTemplate(endpoint.template)
		if golden[key] == nil {
			golden[key] = map[string]bool{}
		}
		golden[key][endpoint.method] = true
	}

	t.Run("bundle_calls_are_all_in_the_golden_list", func(t *testing.T) {
		for _, tmpl := range sortedKeys(bundleTemplates) {
			known, ok := golden[tmpl]
			if !ok {
				t.Errorf("the embedded web UI bundle calls %q (in %s), which the golden list does not know about.\n"+
					"This is the 2026-06-30 failure mode: the bundle was refreshed and now addresses an endpoint the server does not route, "+
					"so that part of the UI 404s in the browser with nothing to show for it.\n"+
					"Fix, in order: (1) register the route in server/adkrest/internal/routers, (2) add the matching entry to uiGolden in this file.",
					tmpl, filepath.Base(bundlePath))
				continue
			}
			for _, method := range sortedKeys(bundleTemplates[tmpl]) {
				if known[method] {
					continue
				}
				// A verb change is the same failure as a new path: the request
				// reaches a route that does not serve that method and 405s.
				t.Errorf("the bundle now calls %s %q, a method the golden list does not have (it has %v).\n"+
					"The UI changed the verb it uses on an endpoint that already existed, so this request 405s in the browser.\n"+
					"Fix, in order: (1) add %s to that route's Methods() in server/adkrest/internal/routers, (2) update uiGolden here.",
					method, tmpl, sortedKeys(known), method)
			}
		}
	})

	t.Run("golden_entries_are_all_still_called_by_the_bundle", func(t *testing.T) {
		for _, tmpl := range sortedKeys(golden) {
			called, ok := bundleTemplates[tmpl]
			if !ok {
				t.Errorf("the golden list has %q (methods %v) but the shipped bundle %s never calls it.\n"+
					"Either the UI dropped the endpoint and the entry is now stale, or the extraction regexes no longer see how the UI builds it.\n"+
					"Fix: confirm against the bundle, then either drop the entry or repair bundleCallSites.",
					tmpl, sortedKeys(golden[tmpl]), filepath.Base(bundlePath))
				continue
			}
			for _, method := range sortedKeys(golden[tmpl]) {
				if !called[method] {
					t.Errorf("the golden list has %s %q but the shipped bundle never issues that verb against it (it issues %v).\n"+
						"Fix: confirm against the bundle, then either drop the entry or repair bundleCallSites.",
						method, tmpl, sortedKeys(called))
				}
			}
		}
	})
}

// readUIBundle returns the contents of the embedded UI's main bundle.
//
// The filename carries a content hash that changes on every rebuild, so it is
// globbed rather than hardcoded. Exactly one match is required: several would
// mean stale bundles are being shipped alongside the current one, and this test
// would be reading an arbitrary one of them.
func readUIBundle(t *testing.T) (source, path string) {
	t.Helper()

	matches, err := filepath.Glob(filepath.Join(bundleDir, "main-*.js"))
	if err != nil {
		t.Fatalf("globbing %s/main-*.js failed: %v", bundleDir, err)
	}
	if len(matches) != 1 {
		t.Fatalf("found %d files matching %s/main-*.js, want exactly 1: %v\n"+
			"The embedded ADK web UI ships one main bundle. Zero means the bundle is missing "+
			"and this test cannot check anything; more than one means stale bundles are being shipped.",
			len(matches), bundleDir, matches)
	}
	content, err := os.ReadFile(matches[0])
	if err != nil {
		t.Fatalf("reading the embedded UI bundle %s failed: %v", matches[0], err)
	}
	return string(content), matches[0]
}

// extractBundleTemplates returns each distinct API path template the bundle
// builds, mapped to how many call sites build it.
// verbRE finds the HTTP verb a call site uses. The UI assigns the URL to a
// local and then passes it to Angular's HttpClient or to fetch, so the verb sits
// just after the URL literal rather than inside it.
var verbRE = regexp.MustCompile(`\.http\.(get|post|put|patch|delete)\(|method:\s*"(GET|POST|PUT|PATCH|DELETE)"`)

// verbWindow bounds how far after a URL literal the verb may appear. Wide
// enough for the assignment and return that separate them, narrow enough not to
// reach the next call site.
const verbWindow = 260

// verbAfterURL reports the HTTP verb belonging to the URL literal ending at end.
func verbAfterURL(source string, end int) (string, bool) {
	window := source[end:min(end+verbWindow, len(source))]
	match := verbRE.FindStringSubmatch(window)
	if match == nil {
		return "", false
	}
	if match[1] != "" {
		return strings.ToUpper(match[1]), true
	}
	return match[2], true
}

// extractBundleTemplates returns the endpoints the bundle calls, as a set of
// methods per canonical path template.
//
// The method matters as much as the path. Upstream switching a verb on an
// endpoint that already exists is the same failure as adding a new one: the
// call 405s in the browser. Comparing paths alone would miss it.
func extractBundleTemplates(t *testing.T, source, path string) map[string]map[string]bool {
	t.Helper()

	interpolationRE := regexp.MustCompile(`\$\{[^{}]*\}`)
	templates := map[string]map[string]bool{}
	total := 0
	for _, site := range bundleCallSites {
		found := site.pattern.FindAllStringSubmatchIndex(source, -1)
		if len(found) < site.minMatches {
			t.Fatalf("the %s pattern matched %d call site(s) in %s, want at least %d.\n"+
				"The bundle has changed how it builds URLs, so this test is no longer reading it. "+
				"Repair the pattern in bundleCallSites before trusting any result from this file.\nPattern: %s",
				site.name, len(found), filepath.Base(path), site.minMatches, site.pattern)
		}
		for _, loc := range found {
			raw, _, _ := strings.Cut(source[loc[2]:loc[3]], "?")
			template := interpolationRE.ReplaceAllString(raw, "{}")

			method := site.fixedMethod
			if method == "" {
				resolved, ok := verbAfterURL(source, loc[1])
				if !ok {
					t.Fatalf("could not find the HTTP verb for the %s call site building %q in %s.\n"+
						"The bundle has changed how it issues requests, so this test can no longer tell a GET from a PUT "+
						"and would stop catching a verb change. Repair verbRE or verbWindow.",
						site.name, template, filepath.Base(path))
				}
				method = resolved
			}

			if templates[template] == nil {
				templates[template] = map[string]bool{}
			}
			templates[template][method] = true
			total++
		}
	}
	if len(templates) < minBundleTemplates {
		t.Fatalf("extracted only %d distinct API path template(s) from %s, want at least %d.\n"+
			"An extraction this thin is not trustworthy; treat it as a broken test, not a passing one.",
			len(templates), filepath.Base(path), minBundleTemplates)
	}
	t.Logf("extracted %d distinct API path templates from %d call sites in %s", len(templates), total, filepath.Base(path))
	return templates
}

func sortedKeys[V any](m map[string]V) []string {
	return slices.Sorted(maps.Keys(m))
}
