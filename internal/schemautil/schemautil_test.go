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

package schemautil_test

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/internal/schemautil"
)

func TestGeminiToJSONSchema(t *testing.T) {
	tests := []struct {
		name    string
		schema  *genai.Schema
		want    map[string]any
		wantErr bool
	}{
		{
			name:    "nil schema returns error",
			schema:  nil,
			wantErr: true,
		},
		{
			name: "unmarshalable schema returns error",
			schema: &genai.Schema{
				Type:    genai.TypeString,
				Example: make(chan int),
			},
			wantErr: true,
		},
		{
			name:   "unset type omits type constraint",
			schema: &genai.Schema{},
			want:   map[string]any{},
		},
		{
			name:   "unspecified type omits type constraint",
			schema: &genai.Schema{Type: genai.TypeUnspecified},
			want:   map[string]any{},
		},
		{
			name:   "explicit null type maps to null",
			schema: &genai.Schema{Type: genai.TypeNULL},
			want:   map[string]any{"type": "null"},
		},
		{
			name:   "type is lowercased",
			schema: &genai.Schema{Type: genai.TypeString},
			want:   map[string]any{"type": "string"},
		},
		{
			name:   "boolean type is lowercased",
			schema: &genai.Schema{Type: genai.TypeBoolean},
			want:   map[string]any{"type": "boolean"},
		},
		{
			name: "direct fields are preserved",
			schema: &genai.Schema{
				Type:        genai.TypeString,
				Title:       "City",
				Description: "A city name",
				Default:     "Paris",
				Enum:        []string{"Paris", "Rome"},
				Format:      "enum",
				Example:     "Rome",
			},
			want: map[string]any{
				"type":        "string",
				"title":       "City",
				"description": "A city name",
				"default":     "Paris",
				"enum":        []any{"Paris", "Rome"},
				"format":      "enum",
				"example":     "Rome",
			},
		},
		{
			name: "nullable true is emitted",
			schema: &genai.Schema{
				Type:     genai.TypeString,
				Nullable: new(true),
			},
			want: map[string]any{
				"type":     "string",
				"nullable": true,
			},
		},
		{
			name: "nullable false is omitted",
			schema: &genai.Schema{
				Type:     genai.TypeString,
				Nullable: new(false),
			},
			want: map[string]any{
				"type": "string",
			},
		},
		{
			name: "string constraints are preserved on string type",
			schema: &genai.Schema{
				Type:      genai.TypeString,
				Pattern:   "^a.*",
				MinLength: new(int64(2)),
				MaxLength: new(int64(8)),
			},
			want: map[string]any{
				"type":      "string",
				"pattern":   "^a.*",
				"minLength": float64(2),
				"maxLength": float64(8),
			},
		},
		{
			name: "string constraints are dropped for non-string type",
			schema: &genai.Schema{
				Type:      genai.TypeInteger,
				Pattern:   "^a.*",
				MinLength: new(int64(2)),
				MaxLength: new(int64(8)),
				Minimum:   new(1.0),
			},
			want: map[string]any{
				"type":    "integer",
				"minimum": float64(1),
			},
		},
		{
			name: "numeric constraints are dropped for string type",
			schema: &genai.Schema{
				Type:    genai.TypeString,
				Minimum: new(1.0),
				Maximum: new(5.0),
				Pattern: "x",
			},
			want: map[string]any{
				"type":    "string",
				"pattern": "x",
			},
		},
		{
			name: "numeric constraints are kept for number type",
			schema: &genai.Schema{
				Type:    genai.TypeNumber,
				Minimum: new(0.5),
				Maximum: new(9.5),
			},
			want: map[string]any{
				"type":    "number",
				"minimum": float64(0.5),
				"maximum": float64(9.5),
			},
		},
		{
			name: "numeric constraints are kept for integer type",
			schema: &genai.Schema{
				Type:    genai.TypeInteger,
				Minimum: new(1.0),
				Maximum: new(10.0),
			},
			want: map[string]any{
				"type":    "integer",
				"minimum": float64(1),
				"maximum": float64(10),
			},
		},
		{
			name: "array items and constraints are converted recursively",
			schema: &genai.Schema{
				Type: genai.TypeArray,
				Items: &genai.Schema{
					Type:      genai.TypeString,
					MaxLength: new(int64(4)),
				},
				MinItems: new(int64(1)),
				MaxItems: new(int64(3)),
			},
			want: map[string]any{
				"type": "array",
				"items": map[string]any{
					"type":      "string",
					"maxLength": float64(4),
				},
				"minItems": float64(1),
				"maxItems": float64(3),
			},
		},
		{
			name:   "array without items omits items key",
			schema: &genai.Schema{Type: genai.TypeArray},
			want:   map[string]any{"type": "array"},
		},
		{
			name: "array constraints are dropped for non-array type",
			schema: &genai.Schema{
				Type:     genai.TypeString,
				Items:    &genai.Schema{Type: genai.TypeString},
				MinItems: new(int64(1)),
				MaxItems: new(int64(3)),
			},
			want: map[string]any{"type": "string"},
		},
		{
			name: "object properties and constraints are converted recursively",
			schema: &genai.Schema{
				Type: genai.TypeObject,
				Properties: map[string]*genai.Schema{
					"name": {
						Type:      genai.TypeString,
						MaxLength: new(int64(10)),
					},
					"tags": {
						Type:  genai.TypeArray,
						Items: &genai.Schema{Type: genai.TypeString},
					},
				},
				Required:      []string{"name"},
				MinProperties: new(int64(1)),
				MaxProperties: new(int64(2)),
			},
			want: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"name": map[string]any{
						"type":      "string",
						"maxLength": float64(10),
					},
					"tags": map[string]any{
						"type":  "array",
						"items": map[string]any{"type": "string"},
					},
				},
				"required":      []any{"name"},
				"minProperties": float64(1),
				"maxProperties": float64(2),
			},
		},
		{
			name: "object constraints are dropped for non-object type",
			schema: &genai.Schema{
				Type: genai.TypeString,
				Properties: map[string]*genai.Schema{
					"name": {Type: genai.TypeString},
				},
				Required:      []string{"name"},
				MinProperties: new(int64(1)),
				MaxProperties: new(int64(2)),
			},
			want: map[string]any{"type": "string"},
		},
		{
			name: "propertyOrdering is stripped at every depth while user property named propertyOrdering is preserved",
			schema: &genai.Schema{
				Type:             genai.TypeObject,
				PropertyOrdering: []string{"nested", "propertyOrdering", "list"},
				Properties: map[string]*genai.Schema{
					"propertyOrdering": {
						Type: genai.TypeString,
					},
					"nested": {
						Type:             genai.TypeObject,
						PropertyOrdering: []string{"leaf"},
						Properties: map[string]*genai.Schema{
							"leaf": {Type: genai.TypeString},
						},
					},
					"list": {
						Type: genai.TypeArray,
						Items: &genai.Schema{
							Type:             genai.TypeObject,
							PropertyOrdering: []string{"item"},
							Properties: map[string]*genai.Schema{
								"item": {Type: genai.TypeInteger},
							},
						},
					},
				},
				AnyOf: []*genai.Schema{
					{
						Type:             genai.TypeObject,
						PropertyOrdering: []string{"branch"},
						Properties: map[string]*genai.Schema{
							"branch": {Type: genai.TypeBoolean},
						},
					},
				},
				Default: map[string]any{
					"propertyOrdering": []string{"keep"},
					"type":             "STRING",
				},
			},
			want: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"propertyOrdering": map[string]any{
						"type": "string",
					},
					"nested": map[string]any{
						"type": "object",
						"properties": map[string]any{
							"leaf": map[string]any{"type": "string"},
						},
					},
					"list": map[string]any{
						"type": "array",
						"items": map[string]any{
							"type": "object",
							"properties": map[string]any{
								"item": map[string]any{"type": "integer"},
							},
						},
					},
				},
				"anyOf": []any{
					map[string]any{
						"type": "object",
						"properties": map[string]any{
							"branch": map[string]any{"type": "boolean"},
						},
					},
				},
				"default": map[string]any{
					"propertyOrdering": []any{"keep"},
					"type":             "STRING",
				},
			},
		},
		{
			name: "anyOf subschemas are converted recursively",
			schema: &genai.Schema{
				AnyOf: []*genai.Schema{
					{Type: genai.TypeString},
					{Type: genai.TypeInteger, Minimum: new(0.0)},
				},
			},
			want: map[string]any{
				"anyOf": []any{
					map[string]any{"type": "string"},
					map[string]any{"type": "integer", "minimum": float64(0)},
				},
			},
		},
		{
			name: "explicit non-object type is preserved alongside anyOf",
			schema: &genai.Schema{
				Type: genai.TypeString,
				AnyOf: []*genai.Schema{
					{Type: genai.TypeString},
					{Type: genai.TypeInteger},
				},
			},
			want: map[string]any{
				"type": "string",
				"anyOf": []any{
					map[string]any{"type": "string"},
					map[string]any{"type": "integer"},
				},
			},
		},
		{
			name: "object union placeholder with non-object anyOf branches omits outer object type",
			schema: &genai.Schema{
				Type: genai.TypeObject,
				AnyOf: []*genai.Schema{
					{Type: genai.TypeString},
					{Type: genai.TypeInteger},
				},
			},
			want: map[string]any{
				"anyOf": []any{
					map[string]any{"type": "string"},
					map[string]any{"type": "integer"},
				},
			},
		},
		{
			name: "object union placeholder keeps properties and required",
			schema: &genai.Schema{
				Type: genai.TypeObject,
				Properties: map[string]*genai.Schema{
					"name": {Type: genai.TypeString},
				},
				Required: []string{"name"},
				AnyOf: []*genai.Schema{
					{Type: genai.TypeString},
					{Type: genai.TypeInteger},
				},
			},
			want: map[string]any{
				"properties": map[string]any{
					"name": map[string]any{"type": "string"},
				},
				"required": []any{"name"},
				"anyOf": []any{
					map[string]any{"type": "string"},
					map[string]any{"type": "integer"},
				},
			},
		},
		{
			name: "object anyOf of objects and unspecified types keeps object type",
			schema: &genai.Schema{
				Type: genai.TypeObject,
				AnyOf: []*genai.Schema{
					{Type: genai.TypeObject, Required: []string{"a"}},
					{Type: genai.TypeUnspecified},
					{},
				},
			},
			want: map[string]any{
				"type": "object",
				"anyOf": []any{
					map[string]any{"type": "object", "required": []any{"a"}},
					map[string]any{},
					map[string]any{},
				},
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := schemautil.GeminiToJSONSchema(tc.schema)
			if (err != nil) != tc.wantErr {
				t.Fatalf("GeminiToJSONSchema() error = %v, wantErr %v", err, tc.wantErr)
			}
			if tc.wantErr {
				return
			}
			if diff := cmp.Diff(tc.want, got); diff != "" {
				t.Errorf("GeminiToJSONSchema() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}
