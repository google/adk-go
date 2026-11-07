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

package openapitoolset_test

import (
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"slices"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/exp/openapitoolset"
	"google.golang.org/adk/v2/internal/testutil"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
)

func loadFixtureExternal(t *testing.T) []byte {
	t.Helper()
	data, err := os.ReadFile("testdata/test.yaml")
	if err != nil {
		t.Fatal(err)
	}
	return data
}

// declarationInRequest runs an agent whose only toolset is the one built from
// spec, once, and returns the function declaration named toolName from the
// first model request. This is how the conformance suite observes an OpenAPI
// toolset: through the declarations it offers the model.
func declarationInRequest(t *testing.T, cfg openapitoolset.Config, toolName string) *genai.FunctionDeclaration {
	t.Helper()
	ts, err := openapitoolset.New(cfg)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	m := &testutil.MockModel{Responses: []*genai.Content{genai.NewContentFromText("done", genai.RoleModel)}}
	a, err := llmagent.New(llmagent.Config{
		Name:     "root_agent",
		Model:    m,
		Toolsets: []tool.Toolset{ts},
	})
	if err != nil {
		t.Fatalf("llmagent.New: %v", err)
	}
	runner := testutil.NewTestAgentRunner(t, a)
	if _, err := testutil.CollectEvents(runner.Run(t, "s1", "hi")); err != nil {
		t.Fatalf("run: %v", err)
	}
	if len(m.Requests) == 0 {
		t.Fatal("model received no request")
	}
	for _, genaiTool := range m.Requests[0].Config.Tools {
		for _, decl := range genaiTool.FunctionDeclarations {
			if decl.Name == toolName {
				return decl
			}
		}
	}
	t.Fatalf("function declaration %q not found in request", toolName)
	return nil
}

// propertyNames returns the sorted property names of a declaration's parameter
// schema.
func propertyNames(t *testing.T, decl *genai.FunctionDeclaration) []string {
	t.Helper()
	schema, ok := decl.ParametersJsonSchema.(map[string]any)
	if !ok {
		t.Fatalf("ParametersJsonSchema is %T, want map", decl.ParametersJsonSchema)
	}
	props, ok := schema["properties"].(map[string]any)
	if !ok {
		t.Fatalf("properties is %T, want map", schema["properties"])
	}
	names := make([]string, 0, len(props))
	for n := range props {
		names = append(names, n)
	}
	slices.Sort(names)
	return names
}

// TestDeclarationGetJSONSchema mirrors u2349: the operation's JSON schema is
// observed as the tool's parameters schema in the model request.
func TestDeclarationGetJSONSchema(t *testing.T) {
	spec := []byte(`{
  "openapi": "3.0.0",
  "info": {"title": "Test API", "version": "1.0.0"},
  "paths": {"/test": {"post": {
    "operationId": "test_operation",
    "summary": "Test Summary",
    "description": "Test Description",
    "parameters": [
      {"name": "param1", "in": "query", "schema": {"type": "string"}, "description": "Parameter 1"},
      {"name": "param2", "in": "header", "schema": {"type": "string"}, "description": "Parameter 2"}
    ],
    "requestBody": {"description": "body", "content": {"application/json": {"schema": {"type": "object", "properties": {"prop1": {"type": "string", "description": "Property 1"}, "prop2": {"type": "integer", "description": "Property 2"}}}}}},
    "responses": {"200": {"description": "Success", "content": {"application/json": {"schema": {"type": "string"}}}}, "400": {"description": "Client Error"}}
  }}}
}`)
	decl := declarationInRequest(t, openapitoolset.Config{Spec: spec}, "test_operation")
	schema := decl.ParametersJsonSchema.(map[string]any)
	if schema["title"] != "test_operation_Arguments" {
		t.Errorf("title = %v, want test_operation_Arguments", schema["title"])
	}
	if schema["type"] != "object" {
		t.Errorf("type = %v, want object", schema["type"])
	}
	props := schema["properties"].(map[string]any)
	for _, want := range []string{"param1", "prop1"} {
		if _, ok := props[want]; !ok {
			t.Errorf("%q missing from %v", want, props)
		}
	}
}

// TestDeclarationDedupeParamNames mirrors u2428: a path, query and header
// parameter all named "test" become test, test_0, test_1.
func TestDeclarationDedupeParamNames(t *testing.T) {
	spec := []byte(`{
  "openapi": "3.0.0",
  "info": {"title": "Test API", "version": "1.0"},
  "servers": [{"url": "https://example.com"}],
  "paths": {"/items/{test}": {"get": {
    "operationId": "test_operation",
    "parameters": [
      {"name": "test", "in": "path", "required": true, "description": "Parameter 1", "schema": {"type": "string"}},
      {"name": "test", "in": "query", "description": "Parameter 2", "schema": {"type": "string"}},
      {"name": "test", "in": "header", "description": "Parameter 3", "schema": {"type": "string"}}
    ],
    "responses": {"200": {"description": "Success"}}
  }}}
}`)
	decl := declarationInRequest(t, openapitoolset.Config{Spec: spec}, "test_operation")
	got := propertyNames(t, decl)
	want := []string{"test", "test_0", "test_1"}
	if !slices.Equal(got, want) {
		t.Errorf("property names = %v, want %v", got, want)
	}
	props := decl.ParametersJsonSchema.(map[string]any)["properties"].(map[string]any)
	wantDesc := map[string]string{"test": "Parameter 1", "test_0": "Parameter 2", "test_1": "Parameter 3"}
	for name, desc := range wantDesc {
		got, _ := props[name].(map[string]any)["description"].(string)
		if got != desc {
			t.Errorf("%s description = %q, want %q", name, got, desc)
		}
	}
}

// TestDeclarationEmptyObjectBody mirrors u2482: an object body with no
// properties contributes no parameters.
func TestDeclarationEmptyObjectBody(t *testing.T) {
	spec := []byte(`{
  "openapi": "3.0.0",
  "info": {"title": "Test API", "version": "1.0.0"},
  "paths": {"/test": {"post": {"requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}}}}}
}`)
	decl := declarationInRequest(t, openapitoolset.Config{Spec: spec}, "test_post")
	if got := propertyNames(t, decl); len(got) != 0 {
		t.Errorf("property names = %v, want empty", got)
	}
}

// TestDeclarationPreservePropertyNamesBodyParams mirrors u2440: one toolset
// preserves camelCase body names, another converts them to snake_case.
func TestDeclarationPreservePropertyNamesBodyParams(t *testing.T) {
	spec := []byte(`{
  "info": {"title": "Test API", "version": "1.0"},
  "openapi": "3.0.0",
  "servers": [{"url": "https://api.example.com"}],
  "paths": {"/users": {"post": {
    "operationId": "createUser",
    "requestBody": {"content": {"application/json": {"schema": {"type": "object", "properties": {"emailAddress": {"type": "string"}, "firstName": {"type": "string"}, "lastName": {"type": "string"}}}}}},
    "responses": {"200": {"description": "OK"}}
  }}}
}`)
	preserve := declarationInRequest(t, openapitoolset.Config{Spec: spec, PreservePropertyNames: true}, "create_user")
	if got := propertyNames(t, preserve); !slices.Equal(got, []string{"emailAddress", "firstName", "lastName"}) {
		t.Errorf("preserve names = %v, want camelCase", got)
	}
	def := declarationInRequest(t, openapitoolset.Config{Spec: spec}, "create_user")
	if got := propertyNames(t, def); !slices.Equal(got, []string{"email_address", "first_name", "last_name"}) {
		t.Errorf("default names = %v, want snake_case", got)
	}
}

// TestToolCountFromFixture mirrors u2424/u2427: the calendar fixture yields
// exactly five tools, offered to the model.
func TestToolCountFromFixture(t *testing.T) {
	data := loadFixtureExternal(t)
	ts, err := openapitoolset.New(openapitoolset.Config{Spec: data})
	if err != nil {
		t.Fatal(err)
	}
	m := &testutil.MockModel{Responses: []*genai.Content{genai.NewContentFromText("done", genai.RoleModel)}}
	a, err := llmagent.New(llmagent.Config{Name: "root_agent", Model: m, Toolsets: []tool.Toolset{ts}})
	if err != nil {
		t.Fatal(err)
	}
	runner := testutil.NewTestAgentRunner(t, a)
	if _, err := testutil.CollectEvents(runner.Run(t, "s1", "hi")); err != nil {
		t.Fatal(err)
	}
	var count int
	for _, genaiTool := range m.Requests[0].Config.Tools {
		count += len(genaiTool.FunctionDeclarations)
	}
	if count != 5 {
		t.Errorf("declared %d tools, want 5", count)
	}
}

// functionResponses returns every function response across the events.
func functionResponses(events []*session.Event) []*genai.FunctionResponse {
	var out []*genai.FunctionResponse
	for _, ev := range events {
		if ev.LLMResponse.Content == nil {
			continue
		}
		for _, p := range ev.LLMResponse.Content.Parts {
			if p.FunctionResponse != nil {
				out = append(out, p.FunctionResponse)
			}
		}
	}
	return out
}

// TestEndToEndCallEchoesBody mirrors u2483: the model calls an OpenAPI tool with
// an array body, the tool POSTs it to the (loopback) server, and the echoed
// body comes back in the function response. It exercises the whole path,
// including the agent-context header-provider wiring in Run.
func TestEndToEndCallEchoesBody(t *testing.T) {
	var providerSeen bool
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("X-Correlation-ID") == "corr-1" {
			providerSeen = true
		}
		defer func() { _ = r.Body.Close() }()
		data, _ := io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(append([]byte(`{"body": `), append(data, '}')...))
	}))
	t.Cleanup(srv.Close)

	ts, err := openapitoolset.New(openapitoolset.Config{
		HeaderProvider: func(agent.ReadonlyContext) (map[string]string, error) {
			return map[string]string{"X-Correlation-ID": "corr-1"}, nil
		},
		SpecMap: map[string]any{
			"openapi": "3.0.0",
			"info":    map[string]any{"title": "Test API", "version": "1.0.0"},
			"servers": []any{map[string]any{"url": srv.URL}},
			"paths": map[string]any{
				"/items": map[string]any{"post": map[string]any{
					"operationId": "create_items",
					"requestBody": map[string]any{"content": map[string]any{"application/json": map[string]any{
						"schema": map[string]any{"type": "array", "items": map[string]any{
							"type":       "object",
							"properties": map[string]any{"item_prop1": map[string]any{"type": "string"}},
						}},
					}}},
					"responses": map[string]any{"200": map[string]any{"description": "OK"}},
				}},
			},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	m := &testutil.MockModel{Responses: []*genai.Content{
		{Role: genai.RoleModel, Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{
			Name: "create_items",
			Args: map[string]any{"array": []any{map[string]any{"item_prop1": "a"}}},
		}}}},
		genai.NewContentFromText("done", genai.RoleModel),
	}}
	a, err := llmagent.New(llmagent.Config{Name: "root_agent", Model: m, Toolsets: []tool.Toolset{ts}})
	if err != nil {
		t.Fatal(err)
	}
	runner := testutil.NewTestAgentRunner(t, a)
	events, err := testutil.CollectEvents(runner.Run(t, "s1", "create the items"))
	if err != nil {
		t.Fatal(err)
	}

	responses := functionResponses(events)
	if len(responses) == 0 {
		t.Fatal("no function response emitted")
	}
	fr := responses[0]
	if fr.Name != "create_items" {
		t.Fatalf("function response name = %q", fr.Name)
	}
	body, ok := fr.Response["body"].([]any)
	if !ok || len(body) != 1 {
		t.Fatalf("function response = %v, want {body:[...]}", fr.Response)
	}
	if item, _ := body[0].(map[string]any); item["item_prop1"] != "a" {
		t.Errorf("echoed item = %v, want item_prop1=a", body[0])
	}
	if !providerSeen {
		t.Error("header provider header did not reach the server")
	}
}
