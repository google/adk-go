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

package controllers_test

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"google.golang.org/adk/v2/server/adkrest/controllers"
	"google.golang.org/adk/v2/server/adkrest/internal/routers"
)

// muxNotFoundBody is what gorilla/mux writes for a path it does not route.
// The endpoints in devfeatures.go exist so a client can tell a deliberate "this
// server does not implement that" apart from an unrouted path, and the only
// thing distinguishing the two at the wire level is the body.
const muxNotFoundBody = "404 page not found"

// serve runs a handler against a bare GET and returns the recorder.
func serve(t *testing.T, h http.HandlerFunc, target string) *httptest.ResponseRecorder {
	t.Helper()
	rr := httptest.NewRecorder()
	h(rr, httptest.NewRequest(http.MethodGet, target, nil))
	return rr
}

func TestNewNotImplementedHandler(t *testing.T) {
	const (
		feature = "the widget tab"
		detail  = "ADK Go has no widgets."
	)
	rr := serve(t, controllers.NewNotImplementedHandler(feature, detail), "/dev/apps/test-app/widgets")

	if got, want := rr.Code, http.StatusNotImplemented; got != want {
		t.Errorf("not-implemented handler status = %d, want %d", got, want)
	}
	if got, want := rr.Header().Get("Content-Type"), "application/json"; !strings.Contains(got, want) {
		t.Errorf("not-implemented handler Content-Type = %q, want it to contain %q", got, want)
	}

	raw := rr.Body.String()
	if strings.Contains(raw, muxNotFoundBody) {
		t.Errorf("not-implemented handler body = %q, want a JSON body rather than mux's unrouted-path text", raw)
	}

	var got controllers.ErrorResponse
	if err := json.Unmarshal(rr.Body.Bytes(), &got); err != nil {
		t.Fatalf("json.Unmarshal(%q) failed: %v", raw, err)
	}
	if got.Error == "" {
		t.Error("not-implemented handler error field is empty, want a readable message")
	}
	if !strings.Contains(got.Error, feature) {
		t.Errorf("not-implemented handler error = %q, want it to name the feature %q", got.Error, feature)
	}
	if got.Detail != detail {
		t.Errorf("not-implemented handler detail = %q, want %q", got.Detail, detail)
	}
}

// TestNotImplementedHandlerOmitsEmptyDetail pins the omitempty tag: a handler
// built without a detail must leave the key out rather than send an empty
// string a client would display.
func TestNotImplementedHandlerOmitsEmptyDetail(t *testing.T) {
	rr := serve(t, controllers.NewNotImplementedHandler("the widget tab", ""), "/dev/apps/test-app/widgets")

	if got, want := rr.Code, http.StatusNotImplemented; got != want {
		t.Errorf("not-implemented handler status = %d, want %d", got, want)
	}

	var body map[string]json.RawMessage
	if err := json.Unmarshal(rr.Body.Bytes(), &body); err != nil {
		t.Fatalf("json.Unmarshal(%q) failed: %v", rr.Body.String(), err)
	}
	if _, ok := body["detail"]; ok {
		t.Errorf("not-implemented handler body = %q, want no \"detail\" key when the detail is empty", rr.Body.String())
	}
	if _, ok := body["error"]; !ok {
		t.Errorf("not-implemented handler body = %q, want an \"error\" key", rr.Body.String())
	}
}

// TestMetricsInfoHandler asserts the raw bytes on purpose. A nil slice
// serialises as null, and a decoded []any is nil either way, so a decoded
// comparison cannot tell an empty list from a missing one. The client can.
func TestMetricsInfoHandler(t *testing.T) {
	rr := serve(t, controllers.MetricsInfoHandler, "/dev/apps/test-app/metrics-info")

	if got, want := rr.Code, http.StatusOK; got != want {
		t.Errorf("metrics-info status = %d, want %d", got, want)
	}
	if got, want := rr.Body.String(), `"metricsInfo":[]`; !strings.Contains(got, want) {
		t.Errorf("metrics-info raw body = %q, want it to contain %q", got, want)
	}
}

// TestAgentBuilderConfigHandler pins the empty document. The UI reads an empty
// body as "nothing to edit here" and disables the builder toggle.
func TestAgentBuilderConfigHandler(t *testing.T) {
	rr := serve(t, controllers.AgentBuilderConfigHandler, "/dev/apps/test-app/builder")

	if got, want := rr.Code, http.StatusOK; got != want {
		t.Errorf("builder config status = %d, want %d", got, want)
	}
	if got := rr.Body.String(); got != "" {
		t.Errorf("builder config body = %q, want an empty body", got)
	}
	if got, want := rr.Header().Get("Content-Type"), "text/plain; charset=UTF-8"; got != want {
		t.Errorf("builder config Content-Type = %q, want %q", got, want)
	}
}

// TestEvalDetailReportsNotInstalled guards the phrase the web UI matches on.
// A failed eval run's error detail is searched for "not installed"; when it
// matches the UI raises its own eval-unavailable notice, and when it does not
// it reports a generic transport error instead.
func TestEvalDetailReportsNotInstalled(t *testing.T) {
	// Exercise the detail routers/eval.go actually composes, not a copy of it.
	var runEval http.HandlerFunc
	for _, route := range (&routers.EvalAPIRouter{}).Routes() {
		if route.Name == "RunEval" {
			runEval = route.HandlerFunc
			break
		}
	}
	if runEval == nil {
		t.Fatal("EvalAPIRouter.Routes() has no RunEval route")
	}

	rr := serve(t, runEval, "/dev/apps/test-app/eval_sets/set-1/run_eval")

	if got, want := rr.Code, http.StatusNotImplemented; got != want {
		t.Errorf("run_eval status = %d, want %d", got, want)
	}

	var got controllers.ErrorResponse
	if err := json.Unmarshal(rr.Body.Bytes(), &got); err != nil {
		t.Fatalf("json.Unmarshal(%q) failed: %v", rr.Body.String(), err)
	}
	if !strings.Contains(got.Detail, controllers.NotInstalledDetail) {
		t.Errorf("run_eval detail = %q, want it to contain %q", got.Detail, controllers.NotInstalledDetail)
	}
}
