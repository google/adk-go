// Copyright 2025 Google LLC
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

package shared

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	"google.golang.org/genai"
)

func TestSchemaToMap(t *testing.T) {
	tests := []struct {
		name    string
		schema  *genai.Schema
		wantErr bool
		want    map[string]any
	}{
		{
			name:   "nil schema",
			schema: nil,
			want:   nil,
		},
		{
			name:   "string type",
			schema: &genai.Schema{Type: genai.TypeString},
			want:   map[string]any{"type": "string"}, // Marshals as "STRING" if using standard json, but we lower it
		},
		{
			name:    "invalid type",
			schema:  &genai.Schema{Example: make(chan int)},
			wantErr: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := SchemaToMap(tc.schema)
			if (err != nil) != tc.wantErr {
				t.Fatalf("SchemaToMap() error = %v, wantErr %v", err, tc.wantErr)
			} else {
				if got["type"] != tc.want["type"] {
					t.Fatalf("unexpected map: %+v, want %+v", got, tc.want)
				}
			}
		})
	}
}

func TestNormalizeSchema(t *testing.T) {
	tests := []struct {
		name    string
		schema  any
		want    map[string]any
		wantErr bool
	}{
		{
			name:    "nil schema",
			schema:  nil,
			wantErr: true,
		},
		{
			name:   "map schema",
			schema: map[string]any{"type": "object"},
			want:   map[string]any{"type": "object"},
		},
		{
			name: "struct schema",
			schema: struct {
				Type string `json:"type"`
			}{Type: "array"},
			want: map[string]any{"type": "array"},
		},
		{
			name:    "invalid schema",
			schema:  func() {}, // unmarshalable
			wantErr: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := NormalizeSchema(tc.schema)
			if (err != nil) != tc.wantErr {
				t.Fatalf("NormalizeSchema() error = %v, wantErr %v", err, tc.wantErr)
			}
			if !tc.wantErr && got["type"] != tc.want["type"] {
				t.Fatalf("NormalizeSchema() = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestResponseFormatName(t *testing.T) {
	tests := []struct {
		name   string
		schema *genai.Schema
		want   string
	}{
		{
			name:   "nil schema",
			schema: nil,
			want:   DefaultResponseFormatName,
		},
		{
			name:   "no title",
			schema: &genai.Schema{Type: genai.TypeObject},
			want:   DefaultResponseFormatName,
		},
		{
			name:   "title already a legal name",
			schema: &genai.Schema{Title: "Custom-Title_2"},
			want:   "Custom-Title_2",
		},
		{
			name:   "title of exactly 64 characters is kept",
			schema: &genai.Schema{Title: strings.Repeat("a", 64)},
			want:   strings.Repeat("a", 64),
		},
		{
			name:   "title with a space falls back to the default",
			schema: &genai.Schema{Title: "Weather Report"},
			want:   DefaultResponseFormatName,
		},
		{
			name:   "title with a trailing newline falls back to the default",
			schema: &genai.Schema{Title: "Weather\n"},
			want:   DefaultResponseFormatName,
		},
		{
			name:   "non-ascii title falls back to the default",
			schema: &genai.Schema{Title: "天気予報"},
			want:   DefaultResponseFormatName,
		},
		{
			name:   "title longer than 64 characters falls back to the default",
			schema: &genai.Schema{Title: strings.Repeat("a", 65)},
			want:   DefaultResponseFormatName,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := ResponseFormatName(tc.schema); got != tc.want {
				t.Errorf("ResponseFormatName() = %q, want %q", got, tc.want)
			}
		})
	}
}

func TestEnforceStrictOpenAISchema(t *testing.T) {
	t.Run("recursive root ref inlines definition and preserves defs", func(t *testing.T) {
		schema := map[string]any{
			"$ref": "#/$defs/TreeNode",
			"$defs": map[string]any{
				"TreeNode": map[string]any{
					"type": "object",
					"properties": map[string]any{
						"value": map[string]any{"type": "string"},
						"children": map[string]any{
							"type": "array",
							"items": map[string]any{
								"$ref":        "#/$defs/TreeNode",
								"description": "should be stripped on nested ref",
							},
						},
					},
				},
			},
		}

		EnforceStrictOpenAISchema(schema)

		want := map[string]any{
			"type":                 "object",
			"additionalProperties": false,
			"required":             []string{"children", "value"},
			"properties": map[string]any{
				"value": map[string]any{"type": "string"},
				"children": map[string]any{
					"type":  "array",
					"items": map[string]any{"$ref": "#/$defs/TreeNode"},
				},
			},
			"$defs": map[string]any{
				"TreeNode": map[string]any{
					"type":                 "object",
					"additionalProperties": false,
					"required":             []string{"children", "value"},
					"properties": map[string]any{
						"value": map[string]any{"type": "string"},
						"children": map[string]any{
							"type":  "array",
							"items": map[string]any{"$ref": "#/$defs/TreeNode"},
						},
					},
				},
			},
		}
		if !reflect.DeepEqual(schema, want) {
			t.Fatalf("EnforceStrictOpenAISchema() =\n%#v\nwant\n%#v", schema, want)
		}

		// Mutating the inlined root must not alias $defs.
		rootProps := schema["properties"].(map[string]any)
		rootProps["extra"] = map[string]any{"type": "boolean"}
		defProps := schema["$defs"].(map[string]any)["TreeNode"].(map[string]any)["properties"].(map[string]any)
		if _, leaked := defProps["extra"]; leaked {
			t.Fatal("root properties aliased $defs.TreeNode.properties")
		}
	})

	t.Run("chained root ref follows defs chain", func(t *testing.T) {
		schema := map[string]any{
			"$ref": "#/$defs/Alias",
			"$defs": map[string]any{
				"Alias": map[string]any{"$ref": "#/$defs/Target"},
				"Target": map[string]any{
					"type":       "object",
					"properties": map[string]any{"id": map[string]any{"type": "string"}},
				},
			},
		}

		EnforceStrictOpenAISchema(schema)

		if _, hasRef := schema["$ref"]; hasRef {
			t.Fatalf("expected root $ref to be inlined, got %#v", schema)
		}
		if schema["type"] != "object" || schema["additionalProperties"] != false {
			t.Fatalf("expected strict object at root, got %#v", schema)
		}
		if _, hasDefs := schema["$defs"]; !hasDefs {
			t.Fatalf("expected $defs to be preserved, got %#v", schema)
		}
	})

	t.Run("cyclic root ref terminates without infinite loop", func(t *testing.T) {
		schema := map[string]any{
			"$ref":        "#/$defs/A",
			"description": "extra",
			"$defs": map[string]any{
				"A": map[string]any{"$ref": "#/$defs/B"},
				"B": map[string]any{"$ref": "#/$defs/A"},
			},
		}

		EnforceStrictOpenAISchema(schema)

		want := map[string]any{"$ref": "#/$defs/A"}
		if !reflect.DeepEqual(schema, want) {
			t.Fatalf("cyclic root ref got %#v, want %#v", schema, want)
		}
	})

	t.Run("root keys take precedence over referenced def keys", func(t *testing.T) {
		schema := map[string]any{
			"$ref":        "#/$defs/Node",
			"description": "root description",
			"$defs": map[string]any{
				"Node": map[string]any{
					"type":        "object",
					"description": "def description",
					"properties":  map[string]any{"name": map[string]any{"type": "string"}},
				},
			},
		}

		EnforceStrictOpenAISchema(schema)

		if got := schema["description"]; got != "root description" {
			t.Fatalf("root description = %v, want %q", got, "root description")
		}
	})

	t.Run("unresolvable external root ref strips siblings without panic", func(t *testing.T) {
		schema := map[string]any{
			"$ref":        "https://example.com/schema.json",
			"description": "should be stripped",
			"$defs": map[string]any{
				"Local": map[string]any{"type": "object"},
			},
		}

		EnforceStrictOpenAISchema(schema)

		want := map[string]any{"$ref": "https://example.com/schema.json"}
		if !reflect.DeepEqual(schema, want) {
			t.Fatalf("external ref = %#v, want %#v", schema, want)
		}
	})

	t.Run("nested ref still strips sibling keywords", func(t *testing.T) {
		schema := map[string]any{
			"type": "object",
			"properties": map[string]any{
				"child": map[string]any{
					"$ref":        "#/$defs/Child",
					"description": "not allowed alongside $ref",
				},
			},
			"$defs": map[string]any{
				"Child": map[string]any{"type": "object"},
			},
		}

		EnforceStrictOpenAISchema(schema)

		child := schema["properties"].(map[string]any)["child"].(map[string]any)
		wantChild := map[string]any{"$ref": "#/$defs/Child"}
		if !reflect.DeepEqual(child, wantChild) {
			t.Fatalf("nested ref = %#v, want %#v", child, wantChild)
		}
	})

	t.Run("normalize schema numeric constraints preserved across root ref clone", func(t *testing.T) {
		raw := map[string]any{
			"$ref": "#/$defs/Score",
			"$defs": map[string]any{
				"Score": map[string]any{
					"type": "object",
					"properties": map[string]any{
						"value": map[string]any{
							"type":    "integer",
							"minimum": 1,
						},
					},
				},
			},
		}

		normalized, err := NormalizeSchema(raw)
		if err != nil {
			t.Fatalf("NormalizeSchema() error = %v", err)
		}
		EnforceStrictOpenAISchema(normalized)

		encoded, err := json.Marshal(normalized)
		if err != nil {
			t.Fatalf("json.Marshal() error = %v", err)
		}
		var roundTrip map[string]any
		if err := json.Unmarshal(encoded, &roundTrip); err != nil {
			t.Fatalf("json.Unmarshal() error = %v", err)
		}
		minVal := roundTrip["properties"].(map[string]any)["value"].(map[string]any)["minimum"]
		if _, isNum := minVal.(float64); !isNum {
			t.Fatalf("minimum decoded as %T (%v), want numeric JSON value", minVal, minVal)
		}
	})
}
