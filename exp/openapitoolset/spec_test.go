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
	"strings"
	"testing"
)

// parse decodes a JSON spec and parses it, failing the test on error.
func parse(t *testing.T, jsonSpec string) []*parsedOperation {
	t.Helper()
	v, err := decodeJSON([]byte(jsonSpec))
	if err != nil {
		t.Fatalf("decode spec: %v", err)
	}
	spec, ok := v.(*object)
	if !ok {
		t.Fatalf("spec decoded to %T, want object", v)
	}
	ops, err := parseSpec(spec, false)
	if err != nil {
		t.Fatalf("parseSpec: %v", err)
	}
	return ops
}

// findParam returns the parameter with the given originalName and location.
func findParam(params []*apiParameter, name, location string) *apiParameter {
	for _, p := range params {
		if p.originalName == name && p.location == location {
			return p
		}
	}
	return nil
}

const minimalSpec = `{
  "openapi": "3.1.0",
  "info": {"title": "Minimal API", "version": "1.0.0"},
  "paths": {
    "/test": {
      "get": {
        "summary": "Test GET endpoint",
        "operationId": "testGet",
        "responses": {
          "200": {
            "description": "Successful response",
            "content": {"application/json": {"schema": {"type": "string"}}}
          }
        }
      }
    }
  }
}`

func TestParseMinimalSpec(t *testing.T) {
	ops := parse(t, minimalSpec)
	if len(ops) != 1 {
		t.Fatalf("got %d operations, want 1", len(ops))
	}
	op := ops[0]
	if op.name != "test_get" {
		t.Errorf("name = %q, want test_get", op.name)
	}
	if op.endpoint.Path != "/test" {
		t.Errorf("path = %q, want /test", op.endpoint.Path)
	}
	if op.endpoint.Method != "get" {
		t.Errorf("method = %q, want get", op.endpoint.Method)
	}
}

func TestParseSpecNoOperationID(t *testing.T) {
	spec := `{
  "openapi": "3.1.0",
  "info": {"title": "Minimal API", "version": "1.0.0"},
  "paths": {"/test": {"get": {"responses": {"200": {"description": "ok"}}}}}
}`
	ops := parse(t, spec)
	if len(ops) != 1 {
		t.Fatalf("got %d operations, want 1", len(ops))
	}
	if ops[0].name != "test_get" {
		t.Errorf("name = %q, want test_get (auto-generated)", ops[0].name)
	}
}

func TestParseSpecMultipleMethods(t *testing.T) {
	spec := `{
  "openapi": "3.1.0",
  "info": {"title": "API", "version": "1.0.0"},
  "paths": {
    "/test": {
      "get": {"operationId": "testGet", "responses": {"200": {"description": "ok"}}},
      "post": {"operationId": "testPost", "responses": {"200": {"description": "ok"}}}
    }
  }
}`
	ops := parse(t, spec)
	if len(ops) != 2 {
		t.Fatalf("got %d operations, want 2", len(ops))
	}
	names := map[string]bool{}
	for _, op := range ops {
		names[op.name] = true
	}
	if !names["test_get"] || !names["test_post"] {
		t.Errorf("names = %v, want test_get and test_post", names)
	}
}

func TestParseSpecWithParameters(t *testing.T) {
	spec := `{
  "openapi": "3.1.0",
  "info": {"title": "API", "version": "1.0.0"},
  "paths": {
    "/test": {
      "get": {
        "operationId": "testGet",
        "parameters": [
          {"name": "param1", "in": "query", "schema": {"type": "string"}},
          {"name": "param2", "in": "header", "schema": {"type": "integer"}}
        ],
        "responses": {"200": {"description": "ok"}}
      }
    }
  }
}`
	ops := parse(t, spec)
	params := ops[0].params
	if len(params) != 2 {
		t.Fatalf("got %d params, want 2", len(params))
	}
	if params[0].originalName != "param1" || params[0].location != "query" {
		t.Errorf("param[0] = %q/%q, want param1/query", params[0].originalName, params[0].location)
	}
	if params[1].originalName != "param2" || params[1].location != "header" {
		t.Errorf("param[1] = %q/%q, want param2/header", params[1].originalName, params[1].location)
	}
}

func TestParseSpecWithRequestBody(t *testing.T) {
	spec := `{
  "openapi": "3.1.0",
  "info": {"title": "API", "version": "1.0.0"},
  "paths": {
    "/test": {
      "post": {
        "operationId": "testPostWithBody",
        "requestBody": {"content": {"application/json": {"schema": {"type": "object", "properties": {"name": {"type": "string"}}}}}},
        "responses": {"200": {"description": "ok"}}
      }
    }
  }
}`
	ops := parse(t, spec)
	op := ops[0]
	if op.name != "test_post_with_body" {
		t.Errorf("name = %q, want test_post_with_body", op.name)
	}
	if len(op.params) != 1 || op.params[0].originalName != "name" {
		t.Fatalf("params = %+v, want one named name", op.params)
	}
	if op.params[0].location != "body" {
		t.Errorf("location = %q, want body", op.params[0].location)
	}
}

func TestParseSpecWithReference(t *testing.T) {
	spec := `{
  "openapi": "3.1.0",
  "info": {"title": "API with Refs", "version": "1.0.0"},
  "paths": {
    "/test_ref": {
      "get": {
        "operationId": "testGetRef",
        "parameters": [{"name": "body", "in": "query", "schema": {"$ref": "#/components/schemas/MySchema"}}],
        "responses": {"200": {"description": "ok"}}
      }
    }
  },
  "components": {"schemas": {"MySchema": {"type": "object", "properties": {"name": {"type": "string"}}}}}
}`
	ops := parse(t, spec)
	p := ops[0].params[0]
	if p.schema["type"] != "object" {
		t.Errorf("resolved ref type = %v, want object", p.schema["type"])
	}
	props, ok := p.schema["properties"].(map[string]any)
	if !ok || props["name"] == nil {
		t.Errorf("resolved ref properties = %v, want name", p.schema["properties"])
	}
}

func TestParseSpecWithCircularReference(t *testing.T) {
	spec := `{
  "openapi": "3.1.0",
  "info": {"title": "Circular", "version": "1.0.0"},
  "paths": {
    "/circular": {
      "get": {
        "operationId": "getCircular",
        "parameters": [{"name": "a", "in": "query", "schema": {"$ref": "#/components/schemas/A"}}],
        "responses": {"200": {"description": "ok"}}
      }
    }
  },
  "components": {"schemas": {
    "A": {"type": "object", "properties": {"b": {"$ref": "#/components/schemas/B"}}},
    "B": {"type": "object", "properties": {"a": {"$ref": "#/components/schemas/A"}}}
  }}
}`
	// The cycle must not hang or crash; it is broken by dropping the repeated $ref.
	ops := parse(t, spec)
	if len(ops) != 1 {
		t.Fatalf("got %d operations, want 1", len(ops))
	}
}

func TestParseNoPaths(t *testing.T) {
	ops := parse(t, `{"openapi": "3.1.0", "info": {"title": "No Paths", "version": "1.0.0"}}`)
	if len(ops) != 0 {
		t.Errorf("got %d operations, want 0", len(ops))
	}
}

func TestParseEmptyPathItem(t *testing.T) {
	ops := parse(t, `{"openapi": "3.1.0", "info": {"title": "x", "version": "1"}, "paths": {"/empty": null}}`)
	if len(ops) != 0 {
		t.Errorf("got %d operations, want 0", len(ops))
	}
}

func TestParseSpecGlobalAuthScheme(t *testing.T) {
	spec := `{
  "openapi": "3.1.0",
  "info": {"title": "API", "version": "1.0.0"},
  "security": [{"api_key": []}],
  "components": {"securitySchemes": {"api_key": {"type": "apiKey", "in": "header", "name": "X-API-Key"}}},
  "paths": {"/test": {"get": {"operationId": "testGet", "responses": {"200": {"description": "ok"}}}}}
}`
	op := parse(t, spec)[0]
	if op.authScheme == nil {
		t.Fatal("authScheme is nil, want apiKey")
	}
	if op.authScheme.Type != "apiKey" {
		t.Errorf("authScheme.Type = %q, want apiKey", op.authScheme.Type)
	}
}

func TestParseSpecLocalAuthScheme(t *testing.T) {
	spec := `{
  "openapi": "3.1.0",
  "info": {"title": "API", "version": "1.0.0"},
  "components": {"securitySchemes": {"local_auth": {"type": "http", "scheme": "bearer"}}},
  "paths": {"/test": {"get": {"operationId": "testGet", "security": [{"local_auth": []}], "responses": {"200": {"description": "ok"}}}}}
}`
	op := parse(t, spec)[0]
	if op.authScheme == nil || op.authScheme.Type != "http" {
		t.Fatalf("authScheme = %+v, want http", op.authScheme)
	}
	if op.authScheme.Scheme != "bearer" {
		t.Errorf("authScheme.Scheme = %q, want bearer", op.authScheme.Scheme)
	}
}

func TestParseSpecServers(t *testing.T) {
	spec := `{
  "openapi": "3.1.0",
  "info": {"title": "API", "version": "1.0.0"},
  "servers": [{"url": "https://api.example.com"}, {"url": "http://localhost:8000"}],
  "paths": {"/test": {"get": {"operationId": "testGet", "responses": {"200": {"description": "ok"}}}}}
}`
	op := parse(t, spec)[0]
	if op.endpoint.BaseURL != "https://api.example.com" {
		t.Errorf("baseURL = %q, want https://api.example.com", op.endpoint.BaseURL)
	}
}

func TestParseSpecNoServers(t *testing.T) {
	op := parse(t, minimalSpec)[0]
	if op.endpoint.BaseURL != "" {
		t.Errorf("baseURL = %q, want empty", op.endpoint.BaseURL)
	}
}

func TestParseSpecDescription(t *testing.T) {
	tests := []struct {
		name, descField, want string
	}{
		{"description", `"description": "This is a test description.",`, "This is a test description."},
		{"empty", `"description": "", "summary": "",`, ""},
		{"none", ``, ""},
		{"summary_fallback", `"summary": "A summary",`, "A summary"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			spec := `{
  "openapi": "3.1.0",
  "info": {"title": "API", "version": "1.0.0"},
  "paths": {"/test": {"get": {"operationId": "testGet", ` + tc.descField + ` "responses": {"200": {"description": "ok"}}}}}
}`
			op := parse(t, spec)[0]
			if op.description != tc.want {
				t.Errorf("description = %q, want %q", op.description, tc.want)
			}
		})
	}
}

func TestParseExternalRefError(t *testing.T) {
	spec := `{
  "openapi": "3.1.0",
  "info": {"title": "x", "version": "1"},
  "paths": {"/external": {"get": {"parameters": [{"name": "a", "in": "query", "schema": {"$ref": "external_file.json#/x"}}], "responses": {"200": {"description": "ok"}}}}}
}`
	v, err := decodeJSON([]byte(spec))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := parseSpec(v.(*object), false); err == nil {
		t.Fatal("parseSpec succeeded, want error for external reference")
	} else if !strings.Contains(err.Error(), "external") {
		t.Errorf("error = %v, want it to mention external references", err)
	}
}

func TestParseSpecDeepRefs(t *testing.T) {
	spec := `{
  "openapi": "3.1.0",
  "info": {"title": "Deep Refs", "version": "1.0.0"},
  "paths": {
    "/path1": {
      "post": {
        "operationId": "postPath1",
        "requestBody": {"content": {"application/json": {"schema": {"$ref": "#/components/schemas/Request1"}}}},
        "responses": {"200": {"description": "ok"}}
      }
    }
  },
  "components": {"schemas": {
    "Request1": {"type": "object", "properties": {"req1_prop1": {"$ref": "#/components/schemas/Level1_1"}}},
    "Level1_1": {"type": "object", "properties": {"level1_1_prop1": {"$ref": "#/components/schemas/Level2_1"}}},
    "Level2_1": {"type": "object", "properties": {"level2_1_prop1": {"$ref": "#/components/schemas/Level3"}}},
    "Level3": {"type": "integer"}
  }}
}`
	op := parse(t, spec)[0]
	if op.name != "post_path1" {
		t.Fatalf("name = %q, want post_path1", op.name)
	}
	p := findParam(op.params, "req1_prop1", "body")
	if p == nil {
		t.Fatal("req1_prop1 not found")
	}
	// Walk req1_prop1 -> level1_1_prop1 -> level2_1_prop1 -> integer.
	l1 := p.schema["properties"].(map[string]any)["level1_1_prop1"].(map[string]any)
	l2 := l1["properties"].(map[string]any)["level2_1_prop1"].(map[string]any)
	if l2["type"] != "integer" {
		t.Errorf("deeply nested type = %v, want integer", l2["type"])
	}
}

func TestParseSpecDuplicateParameterNames(t *testing.T) {
	spec := `{
  "openapi": "3.1.0",
  "info": {"title": "Dup", "version": "1.0.0"},
  "paths": {
    "/duplicate": {
      "post": {
        "operationId": "createWithDuplicate",
        "parameters": [{"name": "name", "in": "query", "schema": {"type": "string"}}],
        "requestBody": {"content": {"application/json": {"schema": {"type": "object", "properties": {"name": {"type": "integer"}}}}}},
        "responses": {"200": {"description": "ok"}}
      }
    }
  }
}`
	op := parse(t, spec)[0]
	if len(op.params) != 2 {
		t.Fatalf("got %d params, want 2", len(op.params))
	}
	query := findParam(op.params, "name", "query")
	body := findParam(op.params, "name", "body")
	if query == nil || query.pyName != "name" {
		t.Errorf("query param = %+v, want py_name name", query)
	}
	if body == nil || body.pyName != "name_0" {
		t.Errorf("body param = %+v, want py_name name_0", body)
	}
}

func TestParseSpecPathLevelParameters(t *testing.T) {
	spec := `{
  "openapi": "3.1.0",
  "info": {"title": "Combine", "version": "1.0.0"},
  "paths": {
    "/test": {
      "parameters": [{"name": "global_param", "in": "query", "schema": {"type": "string"}}],
      "get": {
        "parameters": [{"name": "local_param", "in": "header", "schema": {"type": "integer"}}],
        "operationId": "testGet",
        "responses": {"200": {"description": "ok"}}
      }
    }
  }
}`
	op := parse(t, spec)[0]
	if len(op.params) != 2 {
		t.Fatalf("got %d params, want 2", len(op.params))
	}
	if findParam(op.params, "global_param", "query") == nil {
		t.Error("global_param (query) not found")
	}
	if findParam(op.params, "local_param", "header") == nil {
		t.Error("local_param (header) not found")
	}
}

func TestParseSpecOperationParamOverridesPathLevel(t *testing.T) {
	spec := `{
  "openapi": "3.1.0",
  "info": {"title": "Accounts", "version": "1.0.0"},
  "paths": {
    "/accounts/{accountId}": {
      "parameters": [
        {"name": "accountId", "in": "path", "required": true, "schema": {"type": "string"}, "description": "Shared account id"},
        {"name": "accountId", "in": "header", "schema": {"type": "string"}}
      ],
      "get": {
        "operationId": "getAccount",
        "parameters": [{"name": "accountId", "in": "path", "required": true, "schema": {"type": "string"}, "description": "Account id, e.g. ACC-123"}],
        "responses": {"200": {"description": "ok"}}
      }
    }
  }
}`
	op := parse(t, spec)[0]
	var gotOrder [][2]string
	for _, p := range op.params {
		gotOrder = append(gotOrder, [2]string{p.originalName, p.location})
	}
	want := [][2]string{{"accountId", "path"}, {"accountId", "header"}}
	if len(gotOrder) != 2 || gotOrder[0] != want[0] || gotOrder[1] != want[1] {
		t.Fatalf("param order = %v, want %v", gotOrder, want)
	}
	// The operation-level path parameter wins, so its description is used.
	if d, _ := op.params[0].schema["description"].(string); d != "Account id, e.g. ACC-123" {
		t.Errorf("accountId description = %q, want the operation-level one", d)
	}
}

func TestParseSpecPathParamRequiredWhenOverrideOmitsIt(t *testing.T) {
	spec := `{
  "openapi": "3.1.0",
  "info": {"title": "Accounts", "version": "1.0.0"},
  "paths": {
    "/accounts/{accountId}": {
      "parameters": [{"name": "accountId", "in": "path", "required": true, "schema": {"type": "string"}}],
      "get": {
        "operationId": "getAccount",
        "parameters": [
          {"name": "accountId", "in": "path", "schema": {"type": "string"}},
          {"name": "expand", "in": "query", "schema": {"type": "string"}}
        ],
        "responses": {"200": {"description": "ok"}}
      }
    }
  }
}`
	op := parse(t, spec)[0]
	required := map[string]bool{}
	for _, p := range op.params {
		required[p.originalName] = p.required
	}
	if !required["accountId"] {
		t.Error("accountId should be required even though the override omits the flag")
	}
	if required["expand"] {
		t.Error("expand should not be required")
	}
}

func TestParseSpecInvalidTypeSanitized(t *testing.T) {
	spec := `{
  "openapi": "3.1.0",
  "info": {"title": "Any type", "version": "1.0.0"},
  "paths": {"/test": {"get": {"operationId": "testAnyType", "responses": {"200": {"description": "ok", "content": {"application/json": {"schema": {"type": "Any"}}}}}}}}
}`
	ops := parse(t, spec)
	if len(ops) != 1 || ops[0].name != "test_any_type" {
		t.Fatalf("ops = %+v, want one named test_any_type", ops)
	}
}

func TestParseSpecNestedInvalidTypes(t *testing.T) {
	spec := `{
  "openapi": "3.1.0",
  "info": {"title": "Nested invalid", "version": "1.0.0"},
  "paths": {"/test": {"post": {
    "operationId": "testNestedInvalid",
    "requestBody": {"content": {"application/json": {"schema": {"type": "object", "properties": {
      "valid_prop": {"type": "string"},
      "invalid_prop": {"type": "Unknown"},
      "nested_obj": {"type": "object", "properties": {"deeply_invalid": {"type": "CustomType"}}}
    }}}}},
    "responses": {"200": {"description": "ok"}}
  }}}
}`
	op := parse(t, spec)[0]
	names := map[string]bool{}
	for _, p := range op.params {
		names[p.originalName] = true
	}
	for _, want := range []string{"valid_prop", "invalid_prop", "nested_obj"} {
		if !names[want] {
			t.Errorf("param %q missing; got %v", want, names)
		}
	}
	// The invalid nested type is dropped.
	invalid := findParam(op.params, "invalid_prop", "body")
	if _, ok := invalid.schema["type"]; ok {
		t.Errorf("invalid_prop kept its type; want it dropped")
	}
}

func TestSanitizeSchemaTypes(t *testing.T) {
	tests := []struct {
		name string
		in   string
		// check receives the sanitized schema under paths./x.get.
		check func(t *testing.T, typ any, ok bool)
	}{
		{
			name: "drops invalid",
			in:   `{"type": "Any"}`,
			check: func(t *testing.T, _ any, ok bool) {
				if ok {
					t.Error("type kept; want dropped")
				}
			},
		},
		{
			name: "keeps valid",
			in:   `{"type": "string"}`,
			check: func(t *testing.T, typ any, ok bool) {
				if !ok || typ != "string" {
					t.Errorf("type = %v, %v; want string", typ, ok)
				}
			},
		},
		{
			name: "filters list",
			in:   `{"type": ["string", "Any", "null", "Unknown"]}`,
			check: func(t *testing.T, typ any, ok bool) {
				list, _ := typ.([]any)
				if !ok || len(list) != 2 || list[0] != "string" || list[1] != "null" {
					t.Errorf("type = %v; want [string null]", typ)
				}
			},
		},
		{
			name: "drops all-invalid list",
			in:   `{"type": ["Any", "Unknown"]}`,
			check: func(t *testing.T, _ any, ok bool) {
				if ok {
					t.Error("type kept; want dropped")
				}
			},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			v, err := decodeJSON([]byte(`{"schema": ` + tc.in + `}`))
			if err != nil {
				t.Fatal(err)
			}
			root := v.(*object)
			sanitizeSchemaTypes(root, false)
			schema := root.get("schema").(*object)
			typ, ok := schema.lookup("type")
			tc.check(t, typ, ok)
		})
	}
}

func TestSanitizeSchemaTypesLeavesSecuritySchemes(t *testing.T) {
	// A securitySchemes entry is not below a schema/schemas key, so its
	// apiKey "type" must survive; a sibling schema's invalid type is dropped.
	v, err := decodeJSON([]byte(`{
  "components": {
    "schemas": {"InvalidSchema": {"type": "Any"}},
    "securitySchemes": {"api_key": {"type": "apiKey", "in": "header", "name": "X-API-Key"}}
  }
}`))
	if err != nil {
		t.Fatal(err)
	}
	root := v.(*object)
	sanitizeSchemaTypes(root, false)
	components := root.get("components").(*object)
	invalid := components.get("schemas").(*object).get("InvalidSchema").(*object)
	if _, ok := invalid.lookup("type"); ok {
		t.Error("InvalidSchema kept its type; want dropped")
	}
	apiKey := components.get("securitySchemes").(*object).get("api_key").(*object)
	if apiKey.get("type") != "apiKey" {
		t.Errorf("securityScheme type = %v, want apiKey", apiKey.get("type"))
	}
}

func TestSanitizeSchemaTypesKeepsPropertyNamedType(t *testing.T) {
	// A property literally named "type" (common in calendar-style specs) must
	// not be treated as a type keyword.
	v, err := decodeJSON([]byte(`{
  "schema": {"type": "object", "properties": {"type": {"type": "string"}}}
}`))
	if err != nil {
		t.Fatal(err)
	}
	root := v.(*object)
	sanitizeSchemaTypes(root, false)
	props := root.get("schema").(*object).get("properties").(*object)
	typeProp, ok := props.lookup("type")
	if !ok {
		t.Fatal("property named type was dropped")
	}
	if typeProp.(*object).get("type") != "string" {
		t.Errorf("type property schema = %v, want type string", typeProp)
	}
}
