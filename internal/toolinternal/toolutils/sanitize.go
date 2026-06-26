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

	"github.com/google/jsonschema-go/jsonschema"
)

var (
	schemaPtrType   = reflect.TypeOf((*jsonschema.Schema)(nil))
	schemaSliceType = reflect.TypeOf([]*jsonschema.Schema(nil))
	schemaMapType   = reflect.TypeOf(map[string]*jsonschema.Schema(nil))
)

// SanitizeSchemaForVertex returns a deep copy of s with every `anyOf`/`oneOf`
// composition normalized so it never sits alongside sibling keywords — the
// shape Vertex function-calling rejects ("schema specified other fields
// alongside any_of. When using any_of, it must be the only field set."),
// commonly the optional fields pydantic/MCP servers emit
// (`{"description":..., "anyOf":[{"type":"string"},{"type":"null"}]}`).
//
// Primitive unions collapse into a type array (siblings preserved); structured
// compositions are reduced to the composition keyword alone. The result is
// valid for both the Vertex and Gemini backends. Returns nil for nil.
func SanitizeSchemaForVertex(s *jsonschema.Schema) *jsonschema.Schema {
	if s == nil {
		return nil
	}
	out := s.CloneSchemas()
	sanitizeSchemaNode(out)
	return out
}

// SanitizeJSONSchemaForVertex is SanitizeSchemaForVertex for a schema held in an
// `any`: a *jsonschema.Schema, or the map[string]any that MCP tools arrive as on
// the client side. Other values pass through; the map is deep-copied, not mutated.
func SanitizeJSONSchemaForVertex(v any) any {
	switch s := v.(type) {
	case *jsonschema.Schema:
		return SanitizeSchemaForVertex(s)
	case map[string]any:
		out, _ := deepCopyJSONValue(s).(map[string]any)
		sanitizeJSONNode(out)
		return out
	default:
		return v
	}
}

func sanitizeSchemaNode(s *jsonschema.Schema) {
	if s == nil {
		return
	}
	forEachChildSchema(s, sanitizeSchemaNode)

	if len(s.AnyOf) > 0 {
		s.AnyOf = normalizeComposition(s, s.AnyOf)
	}
	if len(s.OneOf) > 0 {
		s.OneOf = normalizeComposition(s, s.OneOf)
	}
}

// normalizeComposition removes the Vertex-rejected "composition + siblings"
// shape from s. It returns the members to keep on the composition keyword (nil
// when the composition was collapsed into a type array on s).
func normalizeComposition(s *jsonschema.Schema, members []*jsonschema.Schema) []*jsonschema.Schema {
	if collapseTypeUnion(s, members) {
		return nil
	}
	// Members are structured: the composition must be the only field on s.
	anyOf, oneOf := s.AnyOf, s.OneOf
	*s = jsonschema.Schema{AnyOf: anyOf, OneOf: oneOf}
	return members
}

// collapseTypeUnion folds a composition of plain type schemas (e.g. the
// `string | null` that pydantic emits for Optional[str]) into a type array on
// s, leaving s's other keywords intact. It reports whether the fold happened.
func collapseTypeUnion(s *jsonschema.Schema, members []*jsonschema.Schema) bool {
	var nonNull []string
	hasNull := false
	for _, m := range members {
		if !isPlainTypeSchema(m) {
			return false
		}
		for _, t := range memberTypes(m) {
			if t == "null" {
				hasNull = true
				continue
			}
			if !slices.Contains(nonNull, t) {
				nonNull = append(nonNull, t)
			}
		}
	}

	types := nonNull
	if hasNull {
		types = append(types, "null")
	}
	switch {
	case len(types) == 0:
		return false
	case len(types) == 1:
		s.Type = types[0]
		s.Types = nil
	default:
		s.Type = ""
		s.Types = types
	}
	return true
}

func memberTypes(m *jsonschema.Schema) []string {
	if m.Type != "" {
		return []string{m.Type}
	}
	return m.Types
}

// isPlainTypeSchema reports whether m carries a JSON type and nothing else, so a
// composition member can be folded into a parent type array without dropping
// information. Emptiness is checked structurally — every field other than
// Type/Types must be zero — so a member carrying any other keyword (format,
// multipleOf, a nested schema, ...) is left to the structured-composition path
// instead of being silently flattened.
func isPlainTypeSchema(m *jsonschema.Schema) bool {
	if m == nil || (m.Type == "" && len(m.Types) == 0) {
		return false
	}
	v := reflect.ValueOf(m).Elem()
	t := v.Type()
	for i := 0; i < t.NumField(); i++ {
		switch t.Field(i).Name {
		case "Type", "Types":
			// The type is exactly what we fold; every other field must be empty.
		default:
			if !v.Field(i).IsZero() {
				return false
			}
		}
	}
	return true
}

// forEachChildSchema applies f to every immediate sub-schema of s.
func forEachChildSchema(s *jsonschema.Schema, f func(*jsonschema.Schema)) {
	v := reflect.ValueOf(s).Elem()
	for i := 0; i < v.NumField(); i++ {
		fv := v.Field(i)
		switch fv.Type() {
		case schemaPtrType:
			if c, _ := fv.Interface().(*jsonschema.Schema); c != nil {
				f(c)
			}
		case schemaSliceType:
			for _, c := range fv.Interface().([]*jsonschema.Schema) {
				if c != nil {
					f(c)
				}
			}
		case schemaMapType:
			for _, c := range fv.Interface().(map[string]*jsonschema.Schema) {
				if c != nil {
					f(c)
				}
			}
		}
	}
}

// jsonSubschemaKeys and jsonSubschemaMapKeys list where subschemas live, so the
// map-form walk recurses there only — never into instance data (enum, const,
// default, examples) or a property whose name happens to be a schema keyword.
var (
	jsonSubschemaKeys = []string{
		"items", "additionalItems", "additionalProperties", "propertyNames",
		"contains", "not", "if", "then", "else",
		"unevaluatedItems", "unevaluatedProperties", "contentSchema",
		"prefixItems", "allOf", "anyOf", "oneOf",
	}
	jsonSubschemaMapKeys = []string{
		"properties", "patternProperties", "$defs", "definitions", "dependentSchemas",
	}
)

// sanitizeJSONNode is sanitizeSchemaNode for the map[string]any form. It mutates
// node in place; callers pass a deep copy.
func sanitizeJSONNode(node map[string]any) {
	for _, k := range jsonSubschemaKeys {
		recurseJSONSchema(node[k])
	}
	for _, k := range jsonSubschemaMapKeys {
		if m, ok := node[k].(map[string]any); ok {
			for _, child := range m {
				recurseJSONSchema(child)
			}
		}
	}
	normalizeJSONComposition(node, "anyOf")
	normalizeJSONComposition(node, "oneOf")
}

// recurseJSONSchema sanitizes v when it is a subschema or an array of subschemas.
func recurseJSONSchema(v any) {
	switch t := v.(type) {
	case map[string]any:
		sanitizeJSONNode(t)
	case []any:
		for _, e := range t {
			if child, ok := e.(map[string]any); ok {
				sanitizeJSONNode(child)
			}
		}
	}
}

// normalizeJSONComposition is normalizeComposition for the map form.
func normalizeJSONComposition(node map[string]any, keyword string) {
	members, ok := node[keyword].([]any)
	if !ok || len(members) == 0 {
		return
	}
	if types, ok := collapseJSONTypeUnion(members); ok {
		delete(node, keyword)
		if len(types) == 1 {
			node["type"] = types[0]
		} else {
			node["type"] = types
		}
		return
	}
	// Members are structured: the composition must be the only field on node.
	for k := range node {
		if k != "anyOf" && k != "oneOf" {
			delete(node, k)
		}
	}
}

// collapseJSONTypeUnion is collapseTypeUnion for the map form. Null sorts last.
func collapseJSONTypeUnion(members []any) ([]any, bool) {
	var nonNull []any
	hasNull := false
	for _, m := range members {
		mm, ok := m.(map[string]any)
		if !ok {
			return nil, false
		}
		ts, ok := plainJSONMemberTypes(mm)
		if !ok {
			return nil, false
		}
		for _, t := range ts {
			if t == "null" {
				hasNull = true
				continue
			}
			if !slices.Contains(nonNull, any(t)) {
				nonNull = append(nonNull, t)
			}
		}
	}
	types := nonNull
	if hasNull {
		types = append(types, "null")
	}
	if len(types) == 0 {
		return nil, false
	}
	return types, true
}

// plainJSONMemberTypes is isPlainTypeSchema for the map form: a member carrying
// only a "type" keyword (string or array of strings).
func plainJSONMemberTypes(m map[string]any) ([]string, bool) {
	if len(m) != 1 {
		return nil, false
	}
	switch t := m["type"].(type) {
	case string:
		return []string{t}, true
	case []any:
		out := make([]string, 0, len(t))
		for _, e := range t {
			s, ok := e.(string)
			if !ok {
				return nil, false
			}
			out = append(out, s)
		}
		return out, true
	default:
		return nil, false
	}
}

// deepCopyJSONValue deep-copies a JSON-decoded value (map[string]any/[]any;
// scalars are immutable) so sanitizing never mutates the caller's schema.
func deepCopyJSONValue(v any) any {
	switch t := v.(type) {
	case map[string]any:
		m := make(map[string]any, len(t))
		for k, e := range t {
			m[k] = deepCopyJSONValue(e)
		}
		return m
	case []any:
		s := make([]any, len(t))
		for i, e := range t {
			s[i] = deepCopyJSONValue(e)
		}
		return s
	default:
		return v
	}
}
