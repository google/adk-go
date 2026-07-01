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
	"reflect"
	"slices"
	"strings"
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

	t.Run("leaves_valid_schema_unchanged", func(t *testing.T) {
		in := &jsonschema.Schema{
			Type:     "object",
			Required: []string{"name"},
			Properties: map[string]*jsonschema.Schema{
				"name":   {Type: "string", Description: "the name"},
				"age":    {Type: "integer", Minimum: ptr(0.0)},
				"factor": {Type: "number", MultipleOf: ptr(2.0)},
				"tags":   {Type: "array", Items: &jsonschema.Schema{Type: "string"}},
				"email":  {Type: "string", Format: "email"},
			},
			AllOf: []*jsonschema.Schema{{Required: []string{"age"}}},
		}
		out := SanitizeSchemaForVertex(in)
		if !reflect.DeepEqual(out, in) {
			t.Errorf("schema with no offending composition changed shape:\n got = %+v\nwant = %+v", out, in)
		}
	})

	t.Run("preserves_sibling_free_structured_composition", func(t *testing.T) {
		in := &jsonschema.Schema{
			AnyOf: []*jsonschema.Schema{
				{Type: "object", Properties: map[string]*jsonschema.Schema{"a": {Type: "string"}}},
				{Type: "object", Properties: map[string]*jsonschema.Schema{"b": {Type: "integer"}}},
			},
		}
		out := SanitizeSchemaForVertex(in)
		if !reflect.DeepEqual(out, in) {
			t.Errorf("already-valid composition changed:\n got = %+v\nwant = %+v", out, in)
		}
	})

	t.Run("collapses_sibling_free_primitive_union", func(t *testing.T) {
		in := &jsonschema.Schema{AnyOf: []*jsonschema.Schema{{Type: "string"}, {Type: "null"}}}
		out := SanitizeSchemaForVertex(in)
		if len(out.AnyOf) != 0 {
			t.Errorf("anyOf not collapsed: %+v", out.AnyOf)
		}
		if !slices.Equal(out.Types, []string{"string", "null"}) {
			t.Errorf("types = %v, want [string null]", out.Types)
		}
	})

	t.Run("member_with_constraint_is_not_folded", func(t *testing.T) {
		// A member carrying a validation keyword (format) is not a plain type, so
		// folding into a type array — which would drop the keyword — must not
		// happen; the composition is kept and only siblings are stripped.
		in := &jsonschema.Schema{
			Description: "optional email",
			AnyOf: []*jsonschema.Schema{
				{Type: "string", Format: "email"},
				{Type: "null"},
			},
		}
		out := SanitizeSchemaForVertex(in)
		if len(out.Types) != 0 {
			t.Errorf("constrained union folded into types %v", out.Types)
		}
		if len(out.AnyOf) != 2 {
			t.Errorf("composition members changed: %+v", out.AnyOf)
		}
		if hasCompositionWithSiblings(out) {
			t.Errorf("siblings not stripped: %+v", out)
		}
	})

	t.Run("member_with_numeric_constraint_is_not_folded", func(t *testing.T) {
		// multipleOf is a validation keyword like any other; a member carrying it
		// is not plain, so folding (which would silently drop multipleOf) must
		// not happen.
		in := &jsonschema.Schema{
			Description: "optional even number",
			AnyOf: []*jsonschema.Schema{
				{Type: "number", MultipleOf: ptr(2.0)},
				{Type: "null"},
			},
		}
		out := SanitizeSchemaForVertex(in)
		if len(out.Types) != 0 {
			t.Errorf("union with a multipleOf member folded into types %v", out.Types)
		}
		if len(out.AnyOf) != 2 {
			t.Errorf("composition members changed: %+v", out.AnyOf)
		}
	})

	t.Run("collapses_anyof_of_two_primitives", func(t *testing.T) {
		// Parity with adk-python test_to_gemini_schema_any_of: a union of plain
		// types folds into a type array.
		in := &jsonschema.Schema{AnyOf: []*jsonschema.Schema{{Type: "string"}, {Type: "integer"}}}
		out := SanitizeSchemaForVertex(in)
		if len(out.AnyOf) != 0 {
			t.Errorf("anyOf not collapsed: %+v", out.AnyOf)
		}
		if !slices.Equal(out.Types, []string{"string", "integer"}) {
			t.Errorf("types = %v, want [string integer]", out.Types)
		}
	})

	t.Run("collapses_union_with_type_list_member", func(t *testing.T) {
		// Parity with adk-python test_sanitize_schema_formats_for_gemini_nullable:
		// a member may itself carry a type array; its types merge into the fold.
		in := &jsonschema.Schema{
			Description: "x",
			AnyOf: []*jsonschema.Schema{
				{Type: "string"},
				{Types: []string{"integer", "null"}},
			},
		}
		out := SanitizeSchemaForVertex(in)
		if len(out.AnyOf) != 0 {
			t.Errorf("anyOf not collapsed: %+v", out.AnyOf)
		}
		if !slices.Equal(out.Types, []string{"string", "integer", "null"}) {
			t.Errorf("types = %v, want [string integer null]", out.Types)
		}
		if out.Description != "x" {
			t.Errorf("description dropped: %q", out.Description)
		}
	})

	t.Run("dedupes_repeated_types", func(t *testing.T) {
		in := &jsonschema.Schema{
			AnyOf: []*jsonschema.Schema{{Type: "string"}, {Type: "string"}, {Type: "null"}},
		}
		out := SanitizeSchemaForVertex(in)
		if !slices.Equal(out.Types, []string{"string", "null"}) {
			t.Errorf("types = %v, want [string null]", out.Types)
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

	t.Run("leaves_valid_schema_unchanged", func(t *testing.T) {
		in := map[string]any{
			"type":     "object",
			"required": []any{"name"},
			"properties": map[string]any{
				"name":  map[string]any{"type": "string", "description": "the name"},
				"tags":  map[string]any{"type": "array", "items": map[string]any{"type": "string"}},
				"email": map[string]any{"type": "string", "format": "email"},
			},
		}
		out := SanitizeJSONSchemaForVertex(in)
		if !reflect.DeepEqual(out, in) {
			t.Errorf("schema with no offending composition changed shape:\n got = %#v\nwant = %#v", out, in)
		}
	})

	t.Run("does_not_recurse_into_instance_data", func(t *testing.T) {
		// default holds an instance value, not a subschema; an anyOf inside it
		// must be left untouched.
		in := map[string]any{
			"type": "string",
			"default": map[string]any{
				"anyOf": []any{
					map[string]any{"type": "string"},
					map[string]any{"type": "null"},
				},
			},
		}
		out := SanitizeJSONSchemaForVertex(in)
		if !reflect.DeepEqual(out, in) {
			t.Errorf("instance data under default was rewritten:\n got = %#v\nwant = %#v", out, in)
		}
	})

	t.Run("allOf_passes_through", func(t *testing.T) {
		in := map[string]any{
			"allOf": []any{
				map[string]any{"type": "object", "properties": map[string]any{"a": map[string]any{"type": "string"}}},
				map[string]any{"required": []any{"a"}},
			},
		}
		out := SanitizeJSONSchemaForVertex(in)
		if !reflect.DeepEqual(out, in) {
			t.Errorf("allOf was modified:\n got = %#v\nwant = %#v", out, in)
		}
	})

	t.Run("collapses_union_with_type_list_member", func(t *testing.T) {
		// Parity with adk-python test_sanitize_schema_formats_for_gemini_nullable:
		// a member carrying a type array is merged into the folded type array.
		in := map[string]any{
			"description": "x",
			"anyOf": []any{
				map[string]any{"type": "string"},
				map[string]any{"type": []any{"integer", "null"}},
			},
		}
		out, _ := SanitizeJSONSchemaForVertex(in).(map[string]any)
		if _, has := out["anyOf"]; has {
			t.Errorf("anyOf not collapsed: %+v", out)
		}
		if gotTypes, _ := out["type"].([]any); !slices.Equal(gotTypes, []any{"string", "integer", "null"}) {
			t.Errorf("type = %v, want [string integer null]", out["type"])
		}
		if out["description"] != "x" {
			t.Errorf("description dropped: %+v", out)
		}
	})

	t.Run("passes_through_boolean_subschema", func(t *testing.T) {
		// Parity with adk-python boolean-schema tests: MCP servers use `true` for
		// an unconstrained field; it must pass through without panicking.
		in := map[string]any{
			"type": "object",
			"properties": map[string]any{
				"model": true,
				"refId": map[string]any{"type": "string"},
			},
		}
		out := SanitizeJSONSchemaForVertex(in)
		if !reflect.DeepEqual(out, in) {
			t.Errorf("boolean subschema not preserved:\n got = %#v\nwant = %#v", out, in)
		}
	})

	t.Run("sanitizes_inside_defs", func(t *testing.T) {
		in := map[string]any{
			"type": "object",
			"$defs": map[string]any{
				"Foo": map[string]any{
					"description": "d",
					"anyOf": []any{
						map[string]any{"type": "string"},
						map[string]any{"type": "null"},
					},
				},
			},
		}
		out, _ := SanitizeJSONSchemaForVertex(in).(map[string]any)
		foo, _ := out["$defs"].(map[string]any)["Foo"].(map[string]any)
		if _, has := foo["anyOf"]; has {
			t.Errorf("anyOf under $defs not collapsed: %+v", foo)
		}
		if gotTypes, _ := foo["type"].([]any); !slices.Equal(gotTypes, []any{"string", "null"}) {
			t.Errorf("type = %v, want [string null]", foo["type"])
		}
		if foo["description"] != "d" {
			t.Errorf("description dropped: %+v", foo)
		}
	})
}

// TestSubschemaKeyListsCoverStruct keeps jsonSubschemaKeys/jsonSubschemaMapKeys
// in sync with jsonschema.Schema: it fails if the struct has a subschema-bearing
// field whose JSON key is in neither list. Such a gap would make the map-form
// walk skip that location and leave an anyOf/oneOf-plus-siblings shape
// unsanitized. (The struct-form walk uses forEachChildSchema and needs no list.)
func TestSubschemaKeyListsCoverStruct(t *testing.T) {
	covered := map[string]bool{}
	for _, k := range jsonSubschemaKeys {
		covered[k] = true
	}
	for _, k := range jsonSubschemaMapKeys {
		covered[k] = true
	}

	// Wire keys for the subschema fields the library tags json:"-".
	untagged := map[string]string{
		"Items":             "items",
		"ItemsArray":        "items",
		"DependencySchemas": "dependencies",
	}
	subschemaTypes := map[reflect.Type]bool{
		reflect.TypeOf((*jsonschema.Schema)(nil)):          true,
		reflect.TypeOf([]*jsonschema.Schema(nil)):          true,
		reflect.TypeOf(map[string]*jsonschema.Schema(nil)): true,
	}

	st := reflect.TypeOf(jsonschema.Schema{})
	for i := 0; i < st.NumField(); i++ {
		f := st.Field(i)
		if !subschemaTypes[f.Type] {
			continue
		}
		wire := strings.Split(f.Tag.Get("json"), ",")[0]
		if wire == "" || wire == "-" {
			k, ok := untagged[f.Name]
			if !ok {
				t.Errorf("subschema field %s is json:%q with no wire-key override; add it to untagged and to the key lists", f.Name, f.Tag.Get("json"))
				continue
			}
			wire = k
		}
		if !covered[wire] {
			t.Errorf("subschema field %s (JSON key %q) is not covered by jsonSubschemaKeys/jsonSubschemaMapKeys", f.Name, wire)
		}
	}
}

func ptr[T any](v T) *T { return &v }
