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

// falseSchema returns the schema jsonschema-go writes for additionalProperties on a
// Go struct: it matches nothing, so no extra keys are allowed. Each use needs
// its own value, because a resolved schema must be a tree.
func falseSchema() *jsonschema.Schema { return &jsonschema.Schema{Not: &jsonschema.Schema{}} }

// closedObject returns an object schema with the given properties and no
// other keys allowed.
func closedObject(props map[string]*jsonschema.Schema) *jsonschema.Schema {
	return &jsonschema.Schema{Type: "object", Properties: props, AdditionalProperties: falseSchema()}
}

func TestModelOutputSchema(t *testing.T) {
	inner := &genai.Schema{
		Type:       genai.TypeObject,
		Properties: map[string]*genai.Schema{"n": {Type: genai.TypeInteger}},
		Required:   []string{"n"},
	}
	nullableInner := &genai.Schema{
		Type:       genai.TypeObject,
		Nullable:   new(true),
		Properties: inner.Properties,
		Required:   inner.Required,
	}
	for _, tc := range []struct {
		name   string
		schema *jsonschema.Schema // nil: derived from schemaShapes
		want   *genai.Schema
	}{
		{
			// Pointer and slice fields keep Nullable, so set_model_response
			// accepts a null for them as ValidateOutput does.
			name: "struct shapes",
			want: &genai.Schema{
				Type: genai.TypeObject,
				Properties: map[string]*genai.Schema{
					"name":  {Type: genai.TypeString, Description: "the name"},
					"score": {Type: genai.TypeNumber},
					"ok":    {Type: genai.TypeBoolean},
					"note":  {Type: genai.TypeString, Nullable: new(true)},
					"tags":  {Type: genai.TypeArray, Nullable: new(true), Items: &genai.Schema{Type: genai.TypeString}},
					"inner": inner,
					"maybe": nullableInner,
					"many":  {Type: genai.TypeArray, Nullable: new(true), Items: inner},
					// uint's minimum: 0 is dropped; ValidateOutput still checks it.
					"count": {Type: genai.TypeInteger},
					// json.RawMessage is a byte array with bounds on each item.
					"Opaque": {Type: genai.TypeArray, Nullable: new(true), Items: &genai.Schema{Type: genai.TypeInteger}},
				},
				Required: []string{"name", "ok", "note", "tags", "inner", "maybe", "many", "count", "Opaque"},
			},
		},
		{
			name: "string enum and titles",
			schema: &jsonschema.Schema{
				Type: "object", Title: "Report", AdditionalProperties: falseSchema(),
				Properties: map[string]*jsonschema.Schema{
					"level": {Type: "string", Title: "Level", Enum: []any{"low", "high"}},
				},
			},
			want: &genai.Schema{Type: genai.TypeObject, Title: "Report", Properties: map[string]*genai.Schema{
				"level": {Type: genai.TypeString, Title: "Level", Enum: []string{"low", "high"}},
			}},
		},
		{
			name: "single-element type lists",
			schema: &jsonschema.Schema{
				Types: []string{"object"}, AdditionalProperties: falseSchema(),
				Properties: map[string]*jsonschema.Schema{"city": {Types: []string{"string"}}},
			},
			want: &genai.Schema{Type: genai.TypeObject, Properties: map[string]*genai.Schema{
				"city": {Type: genai.TypeString},
			}},
		},
		{
			name: "value constraints and annotations dropped",
			schema: &jsonschema.Schema{
				Schema:               "https://json-schema.org/draft/2020-12/schema",
				ID:                   "https://example.com/report",
				Anchor:               "report",
				Type:                 "object",
				AdditionalProperties: falseSchema(),
				MinProperties:        new(1),
				MaxProperties:        new(5),
				Properties: map[string]*jsonschema.Schema{
					"code": {
						Type: "string", Pattern: "^[A-Z]+$", Format: "email",
						MinLength: new(1), MaxLength: new(9),
						Default: json.RawMessage(`"X"`), Examples: []any{"AB"},
						Deprecated: true, ReadOnly: true, WriteOnly: true, Comment: "c",
					},
					"list": {
						Type: "array", Items: &jsonschema.Schema{Type: "number", MultipleOf: new(0.5), ExclusiveMinimum: new(0.0), ExclusiveMaximum: new(9.0)},
						MinItems: new(1), MaxItems: new(3), UniqueItems: true,
					},
				},
			},
			want: &genai.Schema{Type: genai.TypeObject, Properties: map[string]*genai.Schema{
				"code": {Type: genai.TypeString},
				"list": {Type: genai.TypeArray, Items: &genai.Schema{Type: genai.TypeNumber}},
			}},
		},
		// set_model_response rejects keys its schema does not list, so an
		// object that accepts extra keys cannot be sent.
		{
			name:   "open root object",
			schema: &jsonschema.Schema{Type: "object", Properties: map[string]*jsonschema.Schema{"city": {Type: "string"}}},
		},
		{
			name: "open nested object",
			schema: closedObject(map[string]*jsonschema.Schema{
				"place": {Type: "object", Properties: map[string]*jsonschema.Schema{"city": {Type: "string"}}},
			}),
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
			name: "required name with no property",
			schema: &jsonschema.Schema{
				Type: "object", AdditionalProperties: falseSchema(), Required: []string{"city", "country"},
				Properties: map[string]*jsonschema.Schema{"city": {Type: "string"}},
			},
		},
		{
			// A pointer Output: set_model_response always answers with an
			// object, so a null answer would have no way through.
			name: "nullable root object",
			schema: &jsonschema.Schema{
				Types: []string{"null", "object"}, AdditionalProperties: falseSchema(),
				Properties: map[string]*jsonschema.Schema{"city": {Type: "string"}},
			},
		},
		{
			name:   "integer enum",
			schema: closedObject(map[string]*jsonschema.Schema{"level": {Type: "integer", Enum: []any{1, 2}}}),
		},
		{
			name:   "non-string value in a string enum",
			schema: closedObject(map[string]*jsonschema.Schema{"level": {Type: "string", Enum: []any{"low", 2}}}),
		},
		{
			name:   "string enum on an integer",
			schema: closedObject(map[string]*jsonschema.Schema{"level": {Type: "integer", Enum: []any{"1", "2"}}}),
		},
		{
			name:   "untyped property",
			schema: closedObject(map[string]*jsonschema.Schema{"any": {}}),
		},
		{
			name:   "map property",
			schema: closedObject(map[string]*jsonschema.Schema{"m": {Type: "object", AdditionalProperties: &jsonschema.Schema{Type: "integer"}}}),
		},
		{
			name:   "anyOf property",
			schema: closedObject(map[string]*jsonschema.Schema{"v": {Type: "string", AnyOf: []*jsonschema.Schema{{Type: "string"}, {Type: "string", MinLength: new(2)}}}}),
		},
		{
			name:   "const property",
			schema: closedObject(map[string]*jsonschema.Schema{"v": {Type: "string", Const: jsonschema.Ptr[any]("x")}}),
		},
		{
			name:   "two non-null types",
			schema: closedObject(map[string]*jsonschema.Schema{"v": {Types: []string{"string", "integer"}}}),
		},
		{
			name:   "array without items",
			schema: closedObject(map[string]*jsonschema.Schema{"v": {Type: "array"}}),
		},
		{
			name:   "nested object without properties",
			schema: closedObject(map[string]*jsonschema.Schema{"v": {Type: "object", AdditionalProperties: falseSchema()}}),
		},
		{
			name:   "root object without properties",
			schema: &jsonschema.Schema{Type: "object", AdditionalProperties: falseSchema()},
		},
		{
			name:   "root string",
			schema: &jsonschema.Schema{Type: "string"},
		},
		{
			name:   "root array of objects",
			schema: &jsonschema.Schema{Type: "array", Items: closedObject(map[string]*jsonschema.Schema{"n": {Type: "integer"}})},
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
