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

import "testing"

func TestNewRESTAPITool(t *testing.T) {
	tl, err := NewRESTAPITool(RESTAPIToolConfig{
		Name:        "one_of_request",
		Description: "",
		Endpoint:    Endpoint{BaseURL: "https://example.com", Path: "/one_of_request", Method: "post"},
		Operation: map[string]any{
			"operationId": "one_of_request",
			"requestBody": map[string]any{
				"content": map[string]any{
					"application/json": map[string]any{
						"schema": map[string]any{
							"oneOf":         []any{map[string]any{"type": "object", "properties": map[string]any{"type": map[string]any{"type": "string"}, "stage": map[string]any{"type": "string"}}}},
							"discriminator": map[string]any{"propertyName": "type"},
						},
					},
				},
			},
			"responses": map[string]any{"200": map[string]any{"description": "ok"}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if tl.Name() != "one_of_request" {
		t.Errorf("name = %q", tl.Name())
	}
	props := tl.parameters["properties"].(map[string]any)
	if _, ok := props["body"]; !ok {
		t.Errorf("properties = %v, want body", props)
	}
	if _, ok := props[""]; ok {
		t.Error("empty-named property present")
	}
}

func TestNewRESTAPIToolEmptyOperation(t *testing.T) {
	tl, err := NewRESTAPITool(RESTAPIToolConfig{
		Name:     "items_get_direct",
		Endpoint: Endpoint{BaseURL: "http://localhost:1", Path: "/items", Method: "get"},
	})
	if err != nil {
		t.Fatal(err)
	}
	props := tl.parameters["properties"].(map[string]any)
	if len(props) != 0 {
		t.Errorf("properties = %v, want empty", props)
	}
	if tl.authScheme != nil {
		t.Errorf("authScheme = %+v, want nil (RESTAPITool ignores the operation's security)", tl.authScheme)
	}
}

func TestNewRESTAPIToolRequiresName(t *testing.T) {
	if _, err := NewRESTAPITool(RESTAPIToolConfig{}); err == nil {
		t.Fatal("NewRESTAPITool succeeded, want error without a name")
	}
}

func TestNewRESTAPIToolNameTruncated(t *testing.T) {
	long := ""
	for range 80 {
		long += "a"
	}
	tl, err := NewRESTAPITool(RESTAPIToolConfig{Name: long})
	if err != nil {
		t.Fatal(err)
	}
	if len(tl.Name()) != maxToolNameLength {
		t.Errorf("name length = %d, want %d", len(tl.Name()), maxToolNameLength)
	}
}

func TestDeclaration(t *testing.T) {
	ts, err := New(Config{Spec: loadFixture(t)})
	if err != nil {
		t.Fatal(err)
	}
	get, _ := ts.Tool("calendar_calendars_get")
	decl := get.Declaration()
	if decl.Name != "calendar_calendars_get" {
		t.Errorf("decl.Name = %q", decl.Name)
	}
	schema, ok := decl.ParametersJsonSchema.(map[string]any)
	if !ok {
		t.Fatalf("ParametersJsonSchema is %T, want map", decl.ParametersJsonSchema)
	}
	if schema["title"] != "calendar.calendars.get_Arguments" {
		t.Errorf("title = %v, want calendar.calendars.get_Arguments", schema["title"])
	}
	if schema["type"] != "object" {
		t.Errorf("type = %v, want object", schema["type"])
	}
	props := schema["properties"].(map[string]any)
	if _, ok := props["calendar_id"]; !ok {
		t.Errorf("calendar_id missing from %v", props)
	}
}
