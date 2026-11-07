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

package workflow

import (
	"encoding/json"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/jsonschema-go/jsonschema"
	"google.golang.org/genai"
)

type schemaInner struct {
	N int `json:"n"`
}

type schemaShapes struct {
	Name   string        `json:"name" jsonschema:"the name"`
	Score  float64       `json:"score,omitempty"`
	OK     bool          `json:"ok"`
	Note   *string       `json:"note"`
	Tags   []string      `json:"tags"`
	Inner  schemaInner   `json:"inner"`
	Maybe  *schemaInner  `json:"maybe"`
	Many   []schemaInner `json:"many"`
	Count  uint          `json:"count"`
	Opaque json.RawMessage
}

func TestModelOutputSchema(t *testing.T) {
	inner := &genai.Schema{
		Type:       genai.TypeObject,
		Properties: map[string]*genai.Schema{"n": {Type: genai.TypeInteger}},
		Required:   []string{"n"},
	}
	nullableInner := &genai.Schema{
		Type:       genai.TypeObject,
		Nullable:   genai.Ptr(true),
		Properties: map[string]*genai.Schema{"n": {Type: genai.TypeInteger}},
		Required:   []string{"n"},
	}
	for _, tc := range []struct {
		name   string
		schema *jsonschema.Schema // nil: derived from schemaShapes
		want   *genai.Schema
	}{
		{
			name: "struct shapes",
			want: &genai.Schema{
				Type: genai.TypeObject,
				Properties: map[string]*genai.Schema{
					"name":  {Type: genai.TypeString, Description: "the name"},
					"score": {Type: genai.TypeNumber},
					"ok":    {Type: genai.TypeBoolean},
					"note":  {Type: genai.TypeString, Nullable: genai.Ptr(true)},
					"tags":  {Type: genai.TypeArray, Nullable: genai.Ptr(true), Items: &genai.Schema{Type: genai.TypeString}},
					"inner": inner,
					"maybe": nullableInner,
					"many":  {Type: genai.TypeArray, Nullable: genai.Ptr(true), Items: inner},
					// uint's minimum: 0 is dropped; ValidateOutput still checks it.
					"count": {Type: genai.TypeInteger},
					// json.RawMessage is a byte array with bounds on each item.
					"Opaque": {Type: genai.TypeArray, Nullable: genai.Ptr(true), Items: &genai.Schema{Type: genai.TypeInteger}},
				},
				Required: []string{"name", "ok", "note", "tags", "inner", "maybe", "many", "count", "Opaque"},
			},
		},
		{
			name: "string enum",
			schema: &jsonschema.Schema{Type: "object", Properties: map[string]*jsonschema.Schema{
				"level": {Type: "string", Enum: []any{"low", "high"}},
			}},
			want: &genai.Schema{Type: genai.TypeObject, Properties: map[string]*genai.Schema{
				"level": {Type: genai.TypeString, Enum: []string{"low", "high"}},
			}},
		},
		{
			name: "value constraints and annotations dropped",
			schema: &jsonschema.Schema{
				Schema:        "https://json-schema.org/draft/2020-12/schema",
				ID:            "https://example.com/report",
				Anchor:        "report",
				Type:          "object",
				MinProperties: jsonschema.Ptr(1),
				MaxProperties: jsonschema.Ptr(5),
				Properties: map[string]*jsonschema.Schema{
					"code": {
						Type: "string", Pattern: "^[A-Z]+$", Format: "email",
						MinLength: jsonschema.Ptr(1), MaxLength: jsonschema.Ptr(9),
						Default: json.RawMessage(`"X"`), Examples: []any{"AB"},
						Deprecated: true, ReadOnly: true, WriteOnly: true, Comment: "c",
					},
					"list": {
						Type: "array", Items: &jsonschema.Schema{Type: "number", MultipleOf: jsonschema.Ptr(0.5), ExclusiveMinimum: jsonschema.Ptr(0.0), ExclusiveMaximum: jsonschema.Ptr(9.0)},
						MinItems: jsonschema.Ptr(1), MaxItems: jsonschema.Ptr(3), UniqueItems: true,
					},
				},
			},
			want: &genai.Schema{Type: genai.TypeObject, Properties: map[string]*genai.Schema{
				"code": {Type: genai.TypeString},
				"list": {Type: genai.TypeArray, Items: &genai.Schema{Type: genai.TypeNumber}},
			}},
		},
		{
			name: "properties with a schema for extra keys",
			schema: &jsonschema.Schema{
				Type:                 "object",
				Properties:           map[string]*jsonschema.Schema{"name": {Type: "string"}},
				AdditionalProperties: &jsonschema.Schema{Type: "string"},
			},
		},
		{
			name: "integer enum",
			schema: &jsonschema.Schema{Type: "object", Properties: map[string]*jsonschema.Schema{
				"level": {Type: "integer", Enum: []any{1, 2}},
			}},
		},
		{
			name: "untyped property",
			schema: &jsonschema.Schema{Type: "object", Properties: map[string]*jsonschema.Schema{
				"any": {},
			}},
		},
		{
			name: "map property",
			schema: &jsonschema.Schema{Type: "object", Properties: map[string]*jsonschema.Schema{
				"m": {Type: "object", AdditionalProperties: &jsonschema.Schema{Type: "integer"}},
			}},
		},
		{
			name: "anyOf property",
			schema: &jsonschema.Schema{Type: "object", Properties: map[string]*jsonschema.Schema{
				"v": {AnyOf: []*jsonschema.Schema{{Type: "string"}, {Type: "integer"}}},
			}},
		},
		{
			name: "const property",
			schema: &jsonschema.Schema{Type: "object", Properties: map[string]*jsonschema.Schema{
				"v": {Type: "string", Const: jsonschema.Ptr[any]("x")},
			}},
		},
		{
			name: "two non-null types",
			schema: &jsonschema.Schema{Type: "object", Properties: map[string]*jsonschema.Schema{
				"v": {Types: []string{"string", "integer"}},
			}},
		},
		{
			name:   "array without items",
			schema: &jsonschema.Schema{Type: "object", Properties: map[string]*jsonschema.Schema{"v": {Type: "array"}}},
		},
		{
			name:   "nested object without properties",
			schema: &jsonschema.Schema{Type: "object", Properties: map[string]*jsonschema.Schema{"v": {Type: "object"}}},
		},
		{
			name:   "root object without properties",
			schema: &jsonschema.Schema{Type: "object"},
		},
		{
			name:   "root string",
			schema: &jsonschema.Schema{Type: "string"},
		},
		{
			name:   "root array of objects",
			schema: &jsonschema.Schema{Type: "array", Items: &jsonschema.Schema{Type: "object", Properties: map[string]*jsonschema.Schema{"n": {Type: "integer"}}}},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			resolved, err := resolvedSchema[schemaShapes](tc.schema)
			if err != nil {
				t.Fatalf("resolvedSchema: %v", err)
			}

			got := modelOutputSchema(resolved)

			if diff := cmp.Diff(tc.want, got); diff != "" {
				t.Errorf("modelOutputSchema mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestModelOutputSchemaNil(t *testing.T) {
	if got := modelOutputSchema(nil); got != nil {
		t.Errorf("modelOutputSchema(nil) = %v, want nil", got)
	}
}
