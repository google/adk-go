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

package setmodelresponsetool_test

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/jsonschema-go/jsonschema"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/internal/toolinternal"
)

func TestDeclaration_Shapes(t *testing.T) {
	const wantDescription = "Set your final response using the required output schema.\n\n" +
		"Use this tool to provide your final structured answer instead\n" +
		"of outputting text directly."

	record := &genai.Schema{
		Type:       genai.TypeObject,
		Properties: map[string]*genai.Schema{"city": {Type: genai.TypeString}},
		Required:   []string{"city"},
	}
	recordLower := &genai.Schema{
		Type:       "object",
		Properties: map[string]*genai.Schema{"city": {Type: "string"}},
	}
	recordList := &genai.Schema{
		Type: genai.TypeArray,
		Items: &genai.Schema{
			Type:       genai.TypeObject,
			Title:      "Item",
			Properties: map[string]*genai.Schema{"id": {Type: genai.TypeInteger}},
		},
	}
	recordListLower := &genai.Schema{
		Type: "array",
		Items: &genai.Schema{
			Type:       "object",
			Properties: map[string]*genai.Schema{"id": {Type: "integer"}},
		},
	}
	primitiveList := &genai.Schema{Type: genai.TypeArray, Items: &genai.Schema{Type: genai.TypeString}}
	freeForm := &genai.Schema{Type: genai.TypeObject}
	primitive := &genai.Schema{Type: genai.TypeString}
	empty := &genai.Schema{}
	raw := &jsonschema.Schema{
		Type:       "object",
		Properties: map[string]*jsonschema.Schema{"result": {Type: "string"}},
	}

	tests := []struct {
		name string
		tool toolinternal.FunctionTool
		// samePointer, when set, is the schema ParametersJsonSchema must be
		// identical to; otherwise want is compared by value.
		samePointer *genai.Schema
		want        any
	}{
		{name: "record", tool: newTool(t, record), samePointer: record},
		{name: "record_lowercase_type", tool: newTool(t, recordLower), samePointer: recordLower},
		{
			name: "record_list",
			tool: newTool(t, recordList),
			want: &genai.Schema{
				Type: genai.TypeObject,
				Properties: map[string]*genai.Schema{"items": {
					Type: genai.TypeArray,
					Items: &genai.Schema{
						Type:       genai.TypeObject,
						Title:      "Item",
						Properties: map[string]*genai.Schema{"id": {Type: genai.TypeInteger}},
					},
				}},
				Required: []string{"items"},
			},
		},
		{
			name: "record_list_lowercase_type",
			tool: newTool(t, recordListLower),
			want: &genai.Schema{
				Type: genai.TypeObject,
				Properties: map[string]*genai.Schema{"items": {
					Type: "array",
					Items: &genai.Schema{
						Type:       "object",
						Properties: map[string]*genai.Schema{"id": {Type: "integer"}},
					},
				}},
				Required: []string{"items"},
			},
		},
		{
			name: "primitive_list",
			tool: newTool(t, primitiveList),
			want: &genai.Schema{
				Type: genai.TypeObject,
				Properties: map[string]*genai.Schema{"response": {
					Type:  genai.TypeArray,
					Items: &genai.Schema{Type: genai.TypeString},
				}},
				Required: []string{"response"},
			},
		},
		{
			name: "free_form_object",
			tool: newTool(t, freeForm),
			want: &genai.Schema{
				Type:       genai.TypeObject,
				Properties: map[string]*genai.Schema{"response": {Type: genai.TypeObject}},
				Required:   []string{"response"},
			},
		},
		{
			name: "primitive_root",
			tool: newTool(t, primitive),
			want: &genai.Schema{
				Type:       genai.TypeObject,
				Properties: map[string]*genai.Schema{"response": {Type: genai.TypeString}},
				Required:   []string{"response"},
			},
		},
		{
			name: "empty_type_schema",
			tool: newTool(t, empty),
			want: &genai.Schema{
				Type:       genai.TypeObject,
				Properties: map[string]*genai.Schema{"response": {}},
				Required:   []string{"response"},
			},
		},
		{
			name: "raw_json_schema",
			tool: newRawTool(t, raw),
			want: &jsonschema.Schema{
				Type: "object",
				Properties: map[string]*jsonschema.Schema{"response": {
					Type:       "object",
					Properties: map[string]*jsonschema.Schema{"result": {Type: "string"}},
				}},
				Required: []string{"response"},
			},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := tc.tool.Name(); got != "set_model_response" {
				t.Errorf("Name() = %q, want %q", got, "set_model_response")
			}
			if tc.tool.IsLongRunning() {
				t.Error("IsLongRunning() = true, want false")
			}
			if got := tc.tool.Description(); got != wantDescription {
				t.Errorf("Description() = %q, want %q", got, wantDescription)
			}
			decl := tc.tool.Declaration()
			if decl == nil {
				t.Fatal("Declaration() = nil")
			}
			if decl.Name != "set_model_response" || decl.Description != wantDescription {
				t.Errorf("Declaration() name, description = %q, %q; want %q, %q", decl.Name, decl.Description, "set_model_response", wantDescription)
			}
			if tc.samePointer != nil {
				if got, ok := decl.ParametersJsonSchema.(*genai.Schema); !ok || got != tc.samePointer {
					t.Errorf("ParametersJsonSchema = %#v, want the developer's schema pointer %p", decl.ParametersJsonSchema, tc.samePointer)
				}
				return
			}
			if diff := cmp.Diff(tc.want, decl.ParametersJsonSchema); diff != "" {
				t.Errorf("ParametersJsonSchema mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// TestDeclaration_RawJSONSchemaWithDefs checks that a raw schema's root
// definitions ($defs or definitions) move to the wrapper object, so its
// references still resolve from the root of the declared parameters, and that
// the caller's schema is left as it was.
func TestDeclaration_RawJSONSchemaWithDefs(t *testing.T) {
	item := func() map[string]*jsonschema.Schema {
		return map[string]*jsonschema.Schema{"Item": {
			Type:       "object",
			Properties: map[string]*jsonschema.Schema{"id": {Type: "integer"}},
		}}
	}
	tests := []struct {
		name string
		// raw returns a new copy of the developer's schema on each call.
		raw  func() *jsonschema.Schema
		want *jsonschema.Schema
	}{
		{
			name: "defs",
			raw: func() *jsonschema.Schema {
				return &jsonschema.Schema{
					Type:  "array",
					Items: &jsonschema.Schema{Ref: "#/$defs/Item"},
					Defs:  item(),
				}
			},
			want: &jsonschema.Schema{
				Type: "object",
				Properties: map[string]*jsonschema.Schema{"response": {
					Type:  "array",
					Items: &jsonschema.Schema{Ref: "#/$defs/Item"},
				}},
				Required: []string{"response"},
				Defs:     item(),
			},
		},
		{
			name: "definitions",
			raw: func() *jsonschema.Schema {
				return &jsonschema.Schema{
					Type:        "array",
					Items:       &jsonschema.Schema{Ref: "#/definitions/Item"},
					Definitions: item(),
				}
			},
			want: &jsonschema.Schema{
				Type: "object",
				Properties: map[string]*jsonschema.Schema{"response": {
					Type:  "array",
					Items: &jsonschema.Schema{Ref: "#/definitions/Item"},
				}},
				Required:    []string{"response"},
				Definitions: item(),
			},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			raw := tc.raw()

			got := newRawTool(t, raw).Declaration().ParametersJsonSchema

			if diff := cmp.Diff(tc.want, got); diff != "" {
				t.Errorf("ParametersJsonSchema mismatch (-want +got):\n%s", diff)
			}
			if diff := cmp.Diff(tc.raw(), raw); diff != "" {
				t.Errorf("caller's schema was modified (-want +got):\n%s", diff)
			}
			if s, ok := got.(*jsonschema.Schema); ok {
				if _, err := s.Resolve(nil); err != nil {
					t.Errorf("Resolve() on the declared parameters: %v", err)
				}
			}
		})
	}
}
