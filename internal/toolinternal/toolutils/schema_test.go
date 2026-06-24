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

package toolutils

import (
	"slices"
	"testing"

	"github.com/google/jsonschema-go/jsonschema"
)

// hasCompositionWithSiblings reports whether any node in the schema tree carries
// an anyOf/oneOf composition alongside a sibling keyword — the shape Vertex
// rejects.
func hasCompositionWithSiblings(s *jsonschema.Schema) bool {
	if s == nil {
		return false
	}
	if len(s.AnyOf) > 0 || len(s.OneOf) > 0 {
		siblings := s.Description != "" || s.Title != "" || s.Type != "" ||
			len(s.Types) > 0 || len(s.Properties) > 0 || len(s.Required) > 0
		if siblings {
			return true
		}
	}
	bad := false
	forEachChildSchema(s, func(c *jsonschema.Schema) {
		if hasCompositionWithSiblings(c) {
			bad = true
		}
	})
	return bad
}

func TestSanitizeSchemaForVertex(t *testing.T) {
	t.Run("collapses_nullable_primitive_union", func(t *testing.T) {
		in := &jsonschema.Schema{
			Type: "object",
			Properties: map[string]*jsonschema.Schema{
				"field": {
					Description: "an optional value",
					AnyOf:       []*jsonschema.Schema{{Type: "string"}, {Type: "null"}},
				},
			},
		}
		out := SanitizeSchemaForVertex(in)
		field := out.Properties["field"]
		if len(field.AnyOf) != 0 {
			t.Errorf("anyOf not collapsed: %+v", field.AnyOf)
		}
		if !slices.Equal(field.Types, []string{"string", "null"}) {
			t.Errorf("types = %v, want [string null]", field.Types)
		}
		if field.Description != "an optional value" {
			t.Errorf("description dropped: %q", field.Description)
		}
	})

	t.Run("strips_siblings_for_structured_members", func(t *testing.T) {
		in := &jsonschema.Schema{
			Description: "optional model",
			Title:       "Maybe",
			AnyOf: []*jsonschema.Schema{
				{Type: "object", Properties: map[string]*jsonschema.Schema{"a": {Type: "string"}}},
				{Type: "null"},
			},
		}
		out := SanitizeSchemaForVertex(in)
		if hasCompositionWithSiblings(out) {
			t.Errorf("composition still carries sibling keys: %+v", out)
		}
		if len(out.AnyOf) != 2 {
			t.Errorf("anyOf members dropped: %+v", out.AnyOf)
		}
	})

	t.Run("handles_oneOf", func(t *testing.T) {
		in := &jsonschema.Schema{
			Description: "x",
			OneOf:       []*jsonschema.Schema{{Type: "integer"}, {Type: "number"}},
		}
		out := SanitizeSchemaForVertex(in)
		if len(out.OneOf) != 0 {
			t.Errorf("oneOf not collapsed: %+v", out.OneOf)
		}
		if !slices.Equal(out.Types, []string{"integer", "number"}) {
			t.Errorf("types = %v, want [integer number]", out.Types)
		}
	})

	t.Run("does_not_mutate_input", func(t *testing.T) {
		in := &jsonschema.Schema{
			Properties: map[string]*jsonschema.Schema{
				"f": {Description: "d", AnyOf: []*jsonschema.Schema{{Type: "string"}, {Type: "null"}}},
			},
		}
		_ = SanitizeSchemaForVertex(in)
		if len(in.Properties["f"].AnyOf) != 2 {
			t.Errorf("input schema was mutated: %+v", in.Properties["f"])
		}
	})

	t.Run("nil_returns_nil", func(t *testing.T) {
		if got := SanitizeSchemaForVertex(nil); got != nil {
			t.Errorf("got %+v, want nil", got)
		}
	})
}

func TestSanitizeJSONSchemaForVertex_Map(t *testing.T) {
	t.Run("collapses_nested_nullable_primitive_union", func(t *testing.T) {
		in := map[string]any{
			"type": "object",
			"properties": map[string]any{
				"age": map[string]any{
					"description": "optional age",
					"anyOf": []any{
						map[string]any{"type": "integer"},
						map[string]any{"type": "null"},
					},
				},
			},
		}
		out, _ := SanitizeJSONSchemaForVertex(in).(map[string]any)
		age, _ := out["properties"].(map[string]any)["age"].(map[string]any)
		if _, has := age["anyOf"]; has {
			t.Errorf("anyOf not collapsed: %+v", age)
		}
		if age["description"] != "optional age" {
			t.Errorf("description dropped: %+v", age)
		}
		if gotTypes, _ := age["type"].([]any); !slices.Equal(gotTypes, []any{"integer", "null"}) {
			t.Errorf("type = %v, want [integer null]", age["type"])
		}
	})

	t.Run("strips_siblings_for_structured_members", func(t *testing.T) {
		in := map[string]any{
			"description": "optional model",
			"title":       "Maybe",
			"anyOf": []any{
				map[string]any{"type": "object", "properties": map[string]any{"a": map[string]any{"type": "string"}}},
				map[string]any{"type": "null"},
			},
		}
		out, _ := SanitizeJSONSchemaForVertex(in).(map[string]any)
		if _, has := out["description"]; has {
			t.Errorf("sibling description not stripped: %+v", out)
		}
		if _, has := out["title"]; has {
			t.Errorf("sibling title not stripped: %+v", out)
		}
		if members, _ := out["anyOf"].([]any); len(members) != 2 {
			t.Errorf("anyOf members dropped: %+v", out["anyOf"])
		}
	})

	t.Run("does_not_treat_property_named_anyOf_as_composition", func(t *testing.T) {
		in := map[string]any{
			"type": "object",
			"properties": map[string]any{
				"anyOf": map[string]any{"type": "string", "description": "a field called anyOf"},
			},
		}
		out, _ := SanitizeJSONSchemaForVertex(in).(map[string]any)
		field, _ := out["properties"].(map[string]any)["anyOf"].(map[string]any)
		if field["description"] != "a field called anyOf" {
			t.Errorf("property named anyOf was corrupted: %+v", field)
		}
	})

	t.Run("passes_through_non_schema_value", func(t *testing.T) {
		if got := SanitizeJSONSchemaForVertex("not a schema"); got != "not a schema" {
			t.Errorf("got %v, want passthrough", got)
		}
	})

	t.Run("does_not_mutate_input", func(t *testing.T) {
		in := map[string]any{
			"properties": map[string]any{
				"f": map[string]any{
					"description": "d",
					"anyOf":       []any{map[string]any{"type": "string"}, map[string]any{"type": "null"}},
				},
			},
		}
		_ = SanitizeJSONSchemaForVertex(in)
		f, _ := in["properties"].(map[string]any)["f"].(map[string]any)
		if _, has := f["anyOf"]; !has {
			t.Errorf("input map was mutated: %+v", f)
		}
	})
}
