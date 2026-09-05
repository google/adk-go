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
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
	"time"

	"github.com/getkin/kin-openapi/openapi3"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/auth"
)

func TestOperationToolRunBuildsAuthenticatedRequest(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got, want := r.Method, http.MethodGet; got != want {
			t.Errorf("method = %q, want %q", got, want)
		}
		if got, want := r.URL.EscapedPath(), "/v1/pets/a%2Fb"; got != want {
			t.Errorf("escaped path = %q, want %q", got, want)
		}
		if got, want := r.URL.Query().Get("limit"), "3"; got != want {
			t.Errorf("limit = %q, want %q", got, want)
		}
		if got, want := r.Header.Get("X-Trace"), "trace-1"; got != want {
			t.Errorf("X-Trace = %q, want %q", got, want)
		}
		if got, want := r.Header.Get("Authorization"), "Bearer secret"; got != want {
			t.Errorf("Authorization = %q, want %q", got, want)
		}
		cookie, err := r.Cookie("session")
		if err != nil || cookie.Value != "cookie-1" {
			t.Errorf("session cookie = %v, %v; want cookie-1", cookie, err)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"a/b","ok":true}`))
	}))
	t.Cleanup(server.Close)

	spec := strings.ReplaceAll(string(readFixture(t)), "https://api.example.test", server.URL)
	ts, err := New(context.Background(), []byte(spec), Config{
		HTTPClient: server.Client(),
		Auth:       auth.StaticToken("secret"),
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	tools, err := ts.Tools(nil)
	if err != nil {
		t.Fatalf("Tools() error = %v", err)
	}
	getPet := findOperationTool(t, tools, "get_pet")

	result, err := getPet.Run(newTestAgentContext(), map[string]any{
		"id":      "a/b",
		"limit":   3,
		"x_trace": "trace-1",
		"session": "cookie-1",
	})
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	if got, want := result["id"], "a/b"; got != want {
		t.Errorf("result[id] = %v, want %v", got, want)
	}
	if got, want := result["ok"], true; got != want {
		t.Errorf("result[ok] = %v, want %v", got, want)
	}
}

func TestOperationToolRunBuildsJSONBody(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got, want := r.Header.Get("Content-Type"), "application/json"; got != want {
			t.Errorf("Content-Type = %q, want %q", got, want)
		}
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("ReadAll() error = %v", err)
		}
		if got, want := string(body), `{"age":4,"name":"Mochi"}`; got != want {
			t.Errorf("body = %s, want %s", got, want)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"updated":true}`))
	}))
	t.Cleanup(server.Close)

	spec := strings.ReplaceAll(string(readFixture(t)), "https://api.example.test", server.URL)
	ts, err := New(context.Background(), []byte(spec), Config{HTTPClient: server.Client()})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	tools, _ := ts.Tools(nil)
	updatePet := findOperationTool(t, tools, "update_pet")
	result, err := updatePet.Run(newTestAgentContext(), map[string]any{
		"id":   "pet-1",
		"name": "Mochi",
		"age":  4,
	})
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	if got, want := result["updated"], true; got != want {
		t.Errorf("result[updated] = %v, want %v", got, want)
	}
}

func TestOperationToolRunRejectsMissingRequiredArgument(t *testing.T) {
	ts, err := New(context.Background(), readFixture(t), Config{})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	tools, _ := ts.Tools(nil)
	getPet := findOperationTool(t, tools, "get_pet")
	_, err = getPet.Run(newTestAgentContext(), map[string]any{"id": "pet-1"})
	if err == nil || !strings.Contains(err.Error(), `required argument "limit"`) {
		t.Fatalf("Run() error = %v, want missing limit", err)
	}
}

func TestOperationToolRunBoundsHTTPErrorBody(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadGateway)
		_, _ = w.Write([]byte(strings.Repeat("x", 100)))
	}))
	t.Cleanup(server.Close)

	spec := strings.ReplaceAll(string(readFixture(t)), "https://api.example.test", server.URL)
	ts, err := New(context.Background(), []byte(spec), Config{
		HTTPClient:       server.Client(),
		MaxResponseBytes: 16,
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	tools, _ := ts.Tools(nil)
	getPet := findOperationTool(t, tools, "get_pet")
	_, err = getPet.Run(newTestAgentContext(), map[string]any{
		"id":      "pet-1",
		"limit":   1,
		"x_trace": "trace-1",
	})
	if err == nil {
		t.Fatal("Run() error = nil, want HTTP error")
	}
	if got := err.Error(); !strings.Contains(got, "502 Bad Gateway") || strings.Contains(got, strings.Repeat("x", 17)) {
		t.Errorf("Run() error = %q, want status and bounded body", got)
	}
}

func TestQueryParameterSerialization(t *testing.T) {
	tests := []struct {
		name      string
		parameter argumentBinding
		value     any
		want      string
	}{
		{
			name:      "form array exploded",
			parameter: argumentBinding{parameterName: "tag", style: openapi3.SerializationForm, explode: true},
			value:     []string{"a", "b"},
			want:      "tag=a&tag=b",
		},
		{
			name:      "form object exploded",
			parameter: argumentBinding{parameterName: "filter", style: openapi3.SerializationForm, explode: true},
			value:     map[string]any{"role": "admin", "active": true},
			want:      "active=true&role=admin",
		},
		{
			name:      "deep object",
			parameter: argumentBinding{parameterName: "filter", style: openapi3.SerializationDeepObject, explode: true},
			value:     map[string]any{"role": "admin", "active": true},
			want:      "filter%5Bactive%5D=true&filter%5Brole%5D=admin",
		},
		{
			name:      "pipe delimited",
			parameter: argumentBinding{parameterName: "tag", style: openapi3.SerializationPipeDelimited},
			value:     []string{"a", "b"},
			want:      "tag=a%7Cb",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			query := make(url.Values)
			if err := addQueryParameter(query, tt.parameter, tt.value); err != nil {
				t.Fatalf("addQueryParameter() error = %v", err)
			}
			if got := query.Encode(); got != tt.want {
				t.Errorf("query = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestPathAndCookieParameterSerialization(t *testing.T) {
	path, err := serializePathParameter(argumentBinding{
		parameterName: "id",
		style:         openapi3.SerializationMatrix,
		explode:       true,
	}, []string{"3", "4"})
	if err != nil {
		t.Fatalf("serializePathParameter() error = %v", err)
	}
	if got, want := path, ";id=3;id=4"; got != want {
		t.Errorf("path = %q, want %q", got, want)
	}

	cookies, err := serializeCookies(argumentBinding{
		parameterName: "session",
		style:         openapi3.SerializationForm,
		explode:       true,
	}, map[string]any{"role": "admin", "active": true})
	if err != nil {
		t.Fatalf("serializeCookies() error = %v", err)
	}
	if got, want := fmt.Sprint(cookies), "[active=true role=admin]"; got != want {
		t.Errorf("cookies = %q, want %q", got, want)
	}
}

func TestOptionalRequestBodyValidatesRequiredFieldsOnlyWhenPresent(t *testing.T) {
	operation := &operationTool{
		name: "update_pet",
		body: &bodyBinding{
			fields: []argumentBinding{
				{argumentName: "name", parameterName: "name", required: true},
				{argumentName: "age", parameterName: "age"},
			},
		},
	}
	body, err := operation.buildRequestBody(map[string]any{})
	if err != nil {
		t.Fatalf("buildRequestBody(empty) error = %v", err)
	}
	if body != nil {
		t.Errorf("buildRequestBody(empty) = %v, want nil", body)
	}

	_, err = operation.buildRequestBody(map[string]any{"age": 4})
	if err == nil || !strings.Contains(err.Error(), `required argument "name"`) {
		t.Fatalf("buildRequestBody(age) error = %v, want missing name", err)
	}
}

type testAgentContext struct {
	agent.Context
	ctx context.Context
}

func newTestAgentContext() *testAgentContext {
	return &testAgentContext{ctx: context.Background()}
}

func (c *testAgentContext) Deadline() (time.Time, bool) { return c.ctx.Deadline() }
func (c *testAgentContext) Done() <-chan struct{}       { return c.ctx.Done() }
func (c *testAgentContext) Err() error                  { return c.ctx.Err() }
func (c *testAgentContext) Value(key any) any           { return c.ctx.Value(key) }

func ExampleNewFromURL() {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = fmt.Fprint(w, `{"openapi":"3.0.3","info":{"title":"Status","version":"1"},"servers":[{"url":"https://example.test"}],"paths":{"/status":{"get":{"operationId":"getStatus","summary":"Read service status","responses":{"200":{"description":"OK"}}}}}}`)
	}))
	defer server.Close()

	ts, err := NewFromURL(context.Background(), server.URL, Config{HTTPClient: server.Client()})
	if err != nil {
		fmt.Println(err)
		return
	}
	tools, err := ts.Tools(nil)
	if err != nil {
		fmt.Println(err)
		return
	}
	for _, apiTool := range tools {
		fmt.Printf("%s: %s\n", apiTool.Name(), apiTool.Description())
	}
	// Output:
	// get_status: Read service status
}
