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
	"slices"
	"testing"
)

// parseOp decodes a JSON operation object and parses its parameters.
func parseOp(t *testing.T, jsonOp string, preserveNames bool) []*apiParameter {
	t.Helper()
	v, err := decodeJSON([]byte(jsonOp))
	if err != nil {
		t.Fatalf("decode operation: %v", err)
	}
	params, _, err := parseOperation(v.(*object), preserveNames)
	if err != nil {
		t.Fatalf("parseOperation: %v", err)
	}
	return params
}

const sampleOperation = `{
  "operationId": "test_operation",
  "summary": "Test Summary",
  "description": "Test Description",
  "parameters": [
    {"name": "param1", "in": "query", "schema": {"type": "string"}, "description": "Parameter 1"},
    {"name": "param2", "in": "header", "schema": {"type": "string"}, "description": "Parameter 2"}
  ],
  "requestBody": {
    "content": {"application/json": {"schema": {"type": "object", "properties": {
      "prop1": {"type": "string", "description": "Property 1"},
      "prop2": {"type": "integer", "description": "Property 2"}
    }}}},
    "description": "Request body description"
  },
  "responses": {
    "200": {"description": "Success", "content": {"application/json": {"schema": {"type": "string"}}}},
    "400": {"description": "Client Error"}
  },
  "security": [{"oauth2": ["resource: read", "resource: write"]}]
}`

func TestOperationParameters(t *testing.T) {
	params := parseOp(t, sampleOperation, false)
	if len(params) != 4 { // 2 params + 2 body props
		t.Fatalf("got %d params, want 4", len(params))
	}
	if params[0].originalName != "param1" || params[0].location != "query" {
		t.Errorf("params[0] = %q/%q, want param1/query", params[0].originalName, params[0].location)
	}
	if params[1].originalName != "param2" || params[1].location != "header" {
		t.Errorf("params[1] = %q/%q, want param2/header", params[1].originalName, params[1].location)
	}
	if params[2].originalName != "prop1" || params[2].location != "body" {
		t.Errorf("params[2] = %q/%q, want prop1/body", params[2].originalName, params[2].location)
	}
}

func TestOperationParametersKeepDescription(t *testing.T) {
	// A parameter's description is copied into its schema when the schema has none.
	params := parseOp(t, sampleOperation, false)
	if d, _ := params[0].schema["description"].(string); d != "Parameter 1" {
		t.Errorf("param1 schema description = %q, want Parameter 1", d)
	}
}

func TestRequestBodyObjectRequired(t *testing.T) {
	op := `{
  "operationId": "createSpace",
  "requestBody": {"content": {"application/json": {"schema": {"type": "object", "required": ["spaceName"], "properties": {"spaceName": {"type": "string"}, "description": {"type": "string"}}}}}}
}`
	params := parseOp(t, op, false)
	schema := parametersSchema("createSpace", params)
	required := schema["required"].([]string)
	if !slices.Equal(required, []string{"space_name"}) {
		t.Errorf("required = %v, want [space_name]", required)
	}
}

func TestRequestBodyArray(t *testing.T) {
	op := `{
  "requestBody": {"content": {"application/json": {"schema": {"type": "array", "items": {"type": "object", "properties": {
    "item_prop1": {"type": "string", "description": "Item Property 1"},
    "item_prop2": {"type": "integer", "description": "Item Property 2"}
  }}}}}}
}`
	params := parseOp(t, op, false)
	if len(params) != 1 {
		t.Fatalf("got %d params, want 1", len(params))
	}
	p := params[0]
	if p.originalName != "array" || p.location != "body" {
		t.Errorf("param = %q/%q, want array/body", p.originalName, p.location)
	}
	if p.schema["type"] != "array" {
		t.Errorf("schema type = %v, want array", p.schema["type"])
	}
	items := p.schema["items"].(map[string]any)
	if items["type"] != "object" {
		t.Errorf("items type = %v, want object", items["type"])
	}
}

func TestRequestBodyNoName(t *testing.T) {
	// A scalar body keeps adk-python's empty original name, resolved to "body".
	op := `{"requestBody": {"content": {"application/json": {"schema": {"type": "string"}}}}}`
	params := parseOp(t, op, false)
	if len(params) != 1 {
		t.Fatalf("got %d params, want 1", len(params))
	}
	if params[0].originalName != "" {
		t.Errorf("originalName = %q, want empty", params[0].originalName)
	}
	if params[0].location != "body" {
		t.Errorf("location = %q, want body", params[0].location)
	}
	if params[0].pyName != "body" {
		t.Errorf("pyName = %q, want body", params[0].pyName)
	}
}

func TestRequestBodyOneOfAssignsName(t *testing.T) {
	op := `{
  "operationId": "one_of_request",
  "requestBody": {"content": {"application/json": {"schema": {"oneOf": [{"type": "object", "properties": {"type": {"type": "string"}, "stage": {"type": "string"}}}], "discriminator": {"propertyName": "type"}}}}},
  "responses": {"200": {"description": "ok"}}
}`
	params := parseOp(t, op, false)
	if len(params) != 1 {
		t.Fatalf("got %d params, want 1", len(params))
	}
	if params[0].originalName != "body" || params[0].pyName != "body" {
		t.Errorf("param = %q/%q, want body/body", params[0].originalName, params[0].pyName)
	}
	schema := parametersSchema("one_of_request", params)
	props := schema["properties"].(map[string]any)
	if _, ok := props["body"]; !ok {
		t.Error("body not in properties")
	}
	if _, ok := props[""]; ok {
		t.Error("empty-named property present")
	}
}

func TestRequestBodyEmptyObject(t *testing.T) {
	op := `{"requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}}}`
	params := parseOp(t, op, false)
	if len(params) != 0 {
		t.Fatalf("got %d params, want 0", len(params))
	}
}

func TestDedupeParamNames(t *testing.T) {
	params := []*apiParameter{
		{originalName: "test", pyName: "test"},
		{originalName: "test", pyName: "test"},
		{originalName: "test", pyName: "test"},
	}
	dedupeParamNames(params)
	want := []string{"test", "test_0", "test_1"}
	for i, p := range params {
		if p.pyName != want[i] {
			t.Errorf("params[%d].pyName = %q, want %q", i, p.pyName, want[i])
		}
	}
}

func TestGetJSONSchema(t *testing.T) {
	params := parseOp(t, sampleOperation, false)
	schema := parametersSchema("test_operation", params)
	if schema["title"] != "test_operation_Arguments" {
		t.Errorf("title = %v, want test_operation_Arguments", schema["title"])
	}
	if schema["type"] != "object" {
		t.Errorf("type = %v, want object", schema["type"])
	}
	props := schema["properties"].(map[string]any)
	if _, ok := props["param1"]; !ok {
		t.Error("param1 not in properties")
	}
	if _, ok := props["prop1"]; !ok {
		t.Error("prop1 not in properties")
	}
	// Nothing is required in the sample, but the key is always present.
	if got := schema["required"].([]string); len(got) != 0 {
		t.Errorf("required = %v, want []", got)
	}
}

func TestGetJSONSchemaUnnamedTitle(t *testing.T) {
	schema := parametersSchema("", nil)
	if schema["title"] != "unnamed_Arguments" {
		t.Errorf("title = %v, want unnamed_Arguments", schema["title"])
	}
}

func TestPreservePropertyNames(t *testing.T) {
	op := `{
  "operationId": "createUser",
  "requestBody": {"content": {"application/json": {"schema": {"type": "object", "properties": {
    "firstName": {"type": "string"}, "lastName": {"type": "string"}, "emailAddress": {"type": "string"}
  }}}}}
}`
	preserve := parseOp(t, op, true)
	names := map[string]bool{}
	for _, p := range preserve {
		names[p.pyName] = true
	}
	for _, want := range []string{"firstName", "lastName", "emailAddress"} {
		if !names[want] {
			t.Errorf("preserve: %q missing from %v", want, names)
		}
	}

	def := parseOp(t, op, false)
	names = map[string]bool{}
	for _, p := range def {
		names[p.pyName] = true
	}
	for _, want := range []string{"first_name", "last_name", "email_address"} {
		if !names[want] {
			t.Errorf("default: %q missing from %v", want, names)
		}
	}
}

func TestRequiredSchemeName(t *testing.T) {
	tests := []struct {
		name string
		json string
		want string
	}{
		{"none", `null`, ""},
		{"empty list", `[]`, ""},
		{"single", `[{"oauth2": ["a", "b"]}]`, "oauth2"},
		{"empty requirement", `[{}]`, ""},
		{"optional listed first", `[{}, {"apiKey": []}]`, ""},
		{"optional listed last", `[{"apiKey": []}, {}]`, ""},
		{"required", `[{"apiKey": []}]`, "apiKey"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			v, err := decodeJSON([]byte(tc.json))
			if err != nil {
				t.Fatal(err)
			}
			if got := requiredSchemeName(v); got != tc.want {
				t.Errorf("requiredSchemeName(%s) = %q, want %q", tc.json, got, tc.want)
			}
		})
	}
}

func TestParseOperationFalseSchemaError(t *testing.T) {
	op := `{"operationId": "x", "parameters": [{"name": "p", "in": "query", "schema": false}]}`
	v, err := decodeJSON([]byte(op))
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := parseOperation(v.(*object), false); err == nil {
		t.Fatal("parseOperation succeeded, want error for a false schema")
	}
}

func TestParseOperationSkipsRefParameter(t *testing.T) {
	// An unresolved $ref parameter is skipped, not rejected.
	op := `{"operationId": "x", "parameters": [{"$ref": "#/components/parameters/Missing"}, {"name": "p", "in": "query", "schema": {"type": "string"}}]}`
	params := parseOp(t, op, false)
	if len(params) != 1 || params[0].originalName != "p" {
		t.Fatalf("params = %+v, want only p", params)
	}
}
