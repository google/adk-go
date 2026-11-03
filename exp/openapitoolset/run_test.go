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

package openapitoolset

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"google.golang.org/adk/v2/tool/toolauth"
)

// runTool runs t with args through the core run path, which needs only a
// context.Context. The agent-context header-provider wiring is covered by the
// end-to-end test in integration_test.go.
func runTool(t *testing.T, tool *RESTAPITool, args map[string]any) map[string]any {
	t.Helper()
	res, err := tool.run(context.Background(), args, nil)
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	return res
}

func newServerTool(t *testing.T, handler http.HandlerFunc, operation map[string]any) (*RESTAPITool, *httptest.Server) {
	t.Helper()
	srv := httptest.NewServer(handler)
	t.Cleanup(srv.Close)
	op := operation
	if op == nil {
		op = map[string]any{"operationId": "test_op", "responses": map[string]any{"200": map[string]any{"description": "ok"}}}
	}
	tool, err := NewRESTAPITool(RESTAPIToolConfig{
		Name:      "test_tool",
		Endpoint:  Endpoint{BaseURL: srv.URL, Path: "/api", Method: "get"},
		Operation: op,
	})
	if err != nil {
		t.Fatal(err)
	}
	return tool, srv
}

func TestRunSuccessObject(t *testing.T) {
	tool, _ := newServerTool(t, func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"result": "success"}`)
	}, nil)
	res := runTool(t, tool, map[string]any{})
	if res["result"] != "success" {
		t.Errorf("result = %v, want success", res)
	}
}

func TestRunSuccessArrayWrapped(t *testing.T) {
	tool, _ := newServerTool(t, func(w http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(w, `[1, 2, 3]`)
	}, nil)
	res := runTool(t, tool, map[string]any{})
	arr, ok := res["result"].([]any)
	if !ok || len(arr) != 3 {
		t.Errorf("result = %v, want a 3-element array under result", res)
	}
}

func TestRunSuccessNonJSONText(t *testing.T) {
	tool, _ := newServerTool(t, func(w http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(w, `not json`)
	}, nil)
	res := runTool(t, tool, map[string]any{})
	if res["text"] != "not json" {
		t.Errorf("result = %v, want text", res)
	}
}

func TestRunHTTPFailure(t *testing.T) {
	tool, _ := newServerTool(t, func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = io.WriteString(w, "Internal Server Error")
	}, nil)
	res := runTool(t, tool, map[string]any{})
	errMsg, _ := res["error"].(string)
	if !strings.Contains(errMsg, "Status Code: 500, Internal Server Error") {
		t.Errorf("error = %q, want it to carry the status and body", errMsg)
	}
	if !strings.Contains(errMsg, "Tool test_tool execution failed") {
		t.Errorf("error = %q, want the standard preamble", errMsg)
	}
}

func TestRunDotSegmentPathParamFailsClosed(t *testing.T) {
	var called bool
	srv := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) { called = true }))
	t.Cleanup(srv.Close)
	tool, err := NewRESTAPITool(RESTAPIToolConfig{
		Name:     "test_tool",
		Endpoint: Endpoint{BaseURL: srv.URL, Path: "/files/{name}", Method: "get"},
		Operation: map[string]any{
			"operationId": "test_op",
			"parameters":  []any{map[string]any{"name": "name", "in": "path", "required": true, "schema": map[string]any{"type": "string"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	res := runTool(t, tool, map[string]any{"name": "../../admin/secret"})
	if errMsg, _ := res["error"].(string); !strings.Contains(errMsg, "parent-directory") {
		t.Errorf("error = %q, want parent-directory", errMsg)
	}
	if called {
		t.Error("a traversal path must not issue an HTTP request")
	}
}

func TestRunTimeout(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		time.Sleep(200 * time.Millisecond)
		_, _ = io.WriteString(w, `{}`)
	}))
	t.Cleanup(srv.Close)
	tool, err := NewRESTAPITool(RESTAPIToolConfig{
		Name:       "test_tool",
		Endpoint:   Endpoint{BaseURL: srv.URL, Path: "/api", Method: "get"},
		HTTPClient: &http.Client{Timeout: 20 * time.Millisecond},
	})
	if err != nil {
		t.Fatal(err)
	}
	res := runTool(t, tool, map[string]any{})
	errMsg, _ := res["error"].(string)
	if !strings.Contains(errMsg, "Request timed out.") {
		t.Errorf("error = %q, want a timeout message", errMsg)
	}
	if strings.Contains(errMsg, srv.URL) {
		t.Errorf("error = %q, must not leak the request URL", errMsg)
	}
}

func TestRunTransportErrorHidesURL(t *testing.T) {
	// A server that is closed immediately gives a connection error; the URL
	// (which can carry tokens in its query) must not appear in the message.
	srv := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
	url := srv.URL
	srv.Close()
	tool, err := NewRESTAPITool(RESTAPIToolConfig{
		Name:       "test_tool",
		Endpoint:   Endpoint{BaseURL: url, Path: "/api", Method: "get"},
		HTTPClient: &http.Client{Timeout: time.Second},
	})
	if err != nil {
		t.Fatal(err)
	}
	res := runTool(t, tool, map[string]any{})
	errMsg, _ := res["error"].(string)
	if errMsg == "" {
		t.Fatal("expected an error result")
	}
	if strings.Contains(errMsg, url) {
		t.Errorf("error = %q, must not leak the request URL", errMsg)
	}
}

func TestRunRequiresAuthErrors(t *testing.T) {
	tool, err := NewRESTAPITool(RESTAPIToolConfig{
		Name:     "test_tool",
		Endpoint: Endpoint{BaseURL: "https://example.com", Path: "/api", Method: "get"},
	})
	if err != nil {
		t.Fatal(err)
	}
	// A spec-parsed tool carries the auth scheme; set one directly here.
	tool.authScheme = &toolauth.Scheme{Type: toolauth.SchemeTypeOAuth2}
	res := runTool(t, tool, map[string]any{})
	if errMsg, _ := res["error"].(string); !strings.Contains(errMsg, "authentication") {
		t.Errorf("error = %q, want it to mention authentication", errMsg)
	}
}

func TestRunExtraHeadersApplied(t *testing.T) {
	// The headers from a header provider (resolved by Run) are applied to the
	// request; run receives them directly here.
	var gotHeader string
	tool, _ := newServerTool(t, func(w http.ResponseWriter, r *http.Request) {
		gotHeader = r.Header.Get("X-Correlation-ID")
		_, _ = io.WriteString(w, `{"ok": true}`)
	}, nil)
	if _, err := tool.run(context.Background(), map[string]any{}, map[string]string{"X-Correlation-ID": "abc-123"}); err != nil {
		t.Fatal(err)
	}
	if gotHeader != "abc-123" {
		t.Errorf("header = %q, want abc-123", gotHeader)
	}
}

func TestRunSendsBodyAndQuery(t *testing.T) {
	// End-to-end: an array body is sent as the JSON payload and echoed back.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer func() { _ = r.Body.Close() }()
		data, _ := io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(append([]byte(`{"body": `), append(data, '}')...))
	}))
	t.Cleanup(srv.Close)
	tool, err := NewRESTAPITool(RESTAPIToolConfig{
		Name:     "create_items",
		Endpoint: Endpoint{BaseURL: srv.URL, Path: "/items", Method: "post"},
		Operation: map[string]any{
			"operationId": "create_items",
			"requestBody": map[string]any{"content": map[string]any{"application/json": map[string]any{
				"schema": map[string]any{"type": "array", "items": map[string]any{"type": "object"}},
			}}},
			"responses": map[string]any{"200": map[string]any{"description": "OK"}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	res := runTool(t, tool, map[string]any{"array": []any{map[string]any{"item_prop1": "a", "item_prop2": float64(1)}}})
	body, ok := res["body"].([]any)
	if !ok || len(body) != 1 {
		t.Fatalf("response = %v, want {body:[...]}", res)
	}
	item := body[0].(map[string]any)
	if item["item_prop1"] != "a" {
		t.Errorf("echoed item = %v", item)
	}
}
