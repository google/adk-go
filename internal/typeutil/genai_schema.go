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

package typeutil

import (
	"bytes"
	"encoding/json"
	"fmt"
	"maps"
	"reflect"
	"strings"

	"github.com/google/jsonschema-go/jsonschema"
	"google.golang.org/genai"
)

var genaiTypes = map[string]genai.Type{
	"string":  genai.TypeString,
	"integer": genai.TypeInteger,
	"number":  genai.TypeNumber,
	"boolean": genai.TypeBoolean,
	"array":   genai.TypeArray,
	"object":  genai.TypeObject,
}

// ToGenaiSchema converts the shape of s: types, properties, required fields,
// array items, string enums, title and description. Value constraints
// (bounds, lengths, pattern) are dropped: ValidateOutput enforces them, and
// some providers reject them in a response schema (#2036). Annotations such as
// format, default and examples are dropped too; ValidateOutput does not check
// them either. A "null" in a type list becomes Nullable.
// It reports false for anything else, such as an untyped (true) schema from an
// `any` field, a map or other open object, a required name with no property,
// a composition keyword, a reference, or an enum that is not on a string.
// s is not modified.
func ToGenaiSchema(s *jsonschema.Schema) (*genai.Schema, bool) {
	if s == nil || !onlyKnownKeywords(*s) {
		return nil, false
	}
	typ, nullable, ok := schemaType(s)
	if !ok || (len(s.Enum) > 0 && typ != genai.TypeString) {
		return nil, false
	}
	out := &genai.Schema{Type: typ, Title: s.Title, Description: s.Description}
	if nullable {
		out.Nullable = new(true)
	}
	for _, v := range s.Enum {
		str, ok := v.(string)
		if !ok {
			return nil, false
		}
		out.Enum = append(out.Enum, str)
	}

	switch typ {
	case genai.TypeObject:
		if len(s.Properties) == 0 || !IsFalseSchema(s.AdditionalProperties) {
			return nil, false
		}
		for _, name := range s.Required {
			if _, ok := s.Properties[name]; !ok {
				return nil, false
			}
		}
		out.Properties = make(map[string]*genai.Schema, len(s.Properties))
		for name, prop := range s.Properties {
			p, ok := ToGenaiSchema(prop)
			if !ok {
				return nil, false
			}
			out.Properties[name] = p
		}
		out.Required = append([]string(nil), s.Required...)
	case genai.TypeArray:
		items, ok := ToGenaiSchema(s.Items)
		if !ok {
			return nil, false
		}
		out.Items = items
	}
	return out, true
}

// onlyKnownKeywords reports whether s uses nothing beyond the keywords
// ToGenaiSchema converts or deliberately drops. It clears those and checks
// that nothing is left, so a keyword added to jsonschema.Schema later is
// rejected rather than silently lost.
func onlyKnownKeywords(s jsonschema.Schema) bool {
	// Converted.
	s.Type, s.Types, s.Enum = "", nil, nil
	s.Title, s.Description = "", ""
	s.Properties, s.Required, s.Items = nil, nil, nil
	// Dropped. additionalProperties: false only forbids extra keys, which
	// ValidateOutput and set_model_response both enforce; ToGenaiSchema
	// requires it on every object. Any other value describes a map.
	if IsFalseSchema(s.AdditionalProperties) {
		s.AdditionalProperties = nil
	}
	s.PropertyOrder = nil
	s.Minimum, s.Maximum, s.ExclusiveMinimum, s.ExclusiveMaximum, s.MultipleOf = nil, nil, nil, nil, nil
	s.MinLength, s.MaxLength, s.Pattern, s.Format = nil, nil, "", ""
	s.MinItems, s.MaxItems, s.UniqueItems = nil, nil, false
	s.MinProperties, s.MaxProperties = nil, nil
	s.Default, s.Examples, s.Comment = nil, nil, ""
	s.Schema, s.ID, s.Anchor = "", "", ""
	s.Deprecated, s.ReadOnly, s.WriteOnly = false, false, false
	return reflect.ValueOf(s).IsZero()
}

// IsFalseSchema reports whether s is the schema that matches nothing, which
// jsonschema-go writes as {"not": {}}. It is what jsonschema-go infers as the
// additionalProperties of every struct.
func IsFalseSchema(s *jsonschema.Schema) bool {
	return s != nil && reflect.DeepEqual(*s, jsonschema.Schema{Not: &jsonschema.Schema{}})
}

// schemaType maps the JSON Schema type of s to a genai type. A type list may
// pair one type with "null", which jsonschema-go emits for pointers and
// slices; that becomes Nullable. No type, or more than one non-null type, has
// no genai equivalent.
func schemaType(s *jsonschema.Schema) (genai.Type, bool, bool) {
	if s.Type != "" {
		t, ok := genaiTypes[s.Type]
		return t, false, ok
	}
	var name string
	nullable := false
	for _, t := range s.Types {
		if t == "null" {
			nullable = true
			continue
		}
		if name != "" {
			return "", false, false
		}
		name = t
	}
	t, ok := genaiTypes[name]
	return t, nullable, ok
}

// ClearTitles clears the Title of s and of every schema it reaches through
// Items, Properties and AnyOf. A schema that reaches itself is visited once.
// s may be nil.
func ClearTitles(s *genai.Schema) {
	clearTitles(s, map[*genai.Schema]bool{})
}

func clearTitles(s *genai.Schema, seen map[*genai.Schema]bool) {
	if s == nil || seen[s] {
		return
	}
	seen[s] = true
	s.Title = ""
	clearTitles(s.Items, seen)
	for _, p := range s.Properties {
		clearTitles(p, seen)
	}
	for _, arm := range s.AnyOf {
		clearTitles(arm, seen)
	}
}

// FlattenNullableForVertex returns a copy of s in which every sub-schema of
// the form anyOf [X, {"type": "null"}] becomes X plus "nullable": true, as
// adk-python does because Vertex AI rejects a sub-schema without a type
// (tools/_function_tool_declarations.py:351-420). The sub-schema's other
// keywords (description, default) are kept unless X sets the same keyword. A
// union with more than one non-null arm keeps its anyOf without the null arm
// and gets "nullable": true; a union without exactly one null arm is left
// alone. s is not modified, and the result is always a new schema. It fails
// only when s does not round-trip through JSON.
//
// The rewrite works on the JSON value, as adk-python's does, so every keyword
// of X survives without listing the fields of jsonschema.Schema here.
func FlattenNullableForVertex(s *jsonschema.Schema) (*jsonschema.Schema, error) {
	b, err := json.Marshal(s)
	if err != nil {
		return nil, fmt.Errorf("marshal parameters: %w", err)
	}
	dec := json.NewDecoder(bytes.NewReader(b))
	dec.UseNumber()
	var v any
	if err := dec.Decode(&v); err != nil {
		return nil, fmt.Errorf("decode parameters: %w", err)
	}
	b, err = json.Marshal(sanitizeForVertex(v))
	if err != nil {
		return nil, fmt.Errorf("marshal flattened parameters: %w", err)
	}
	out := &jsonschema.Schema{}
	if err := json.Unmarshal(b, out); err != nil {
		return nil, fmt.Errorf("unmarshal flattened parameters: %w", err)
	}
	return out, nil
}

// sanitizeForVertex ports _sanitize_json_schema_for_vertex: children first,
// then the value itself.
func sanitizeForVertex(v any) any {
	switch v := v.(type) {
	case []any:
		out := make([]any, len(v))
		for i, item := range v {
			out[i] = sanitizeForVertex(item)
		}
		return out
	case map[string]any:
		out := make(map[string]any, len(v))
		for k, item := range v {
			out[k] = sanitizeForVertex(item)
		}
		return flattenOptionalAnyOf(out)
	default:
		return v
	}
}

// flattenOptionalAnyOf ports _flatten_optional_any_of for one schema whose
// children are already rewritten.
func flattenOptionalAnyOf(s map[string]any) map[string]any {
	anyOf, ok := s["anyOf"].([]any)
	if !ok || len(anyOf) < 2 {
		return s
	}
	var nulls []any
	for _, arm := range anyOf {
		if m, ok := arm.(map[string]any); ok && m["type"] == "null" {
			nulls = append(nulls, arm)
		}
	}
	if len(nulls) != 1 {
		return s
	}
	var rest []any
	for _, arm := range anyOf {
		if !reflect.DeepEqual(arm, nulls[0]) {
			rest = append(rest, arm)
		}
	}
	if len(rest) == 1 {
		if x, ok := rest[0].(map[string]any); ok {
			flat := maps.Clone(x)
			for k, val := range s {
				if _, set := flat[k]; !set && k != "anyOf" {
					flat[k] = val
				}
			}
			flat["nullable"] = true
			return flat
		}
	}
	flat := maps.Clone(s)
	flat["anyOf"] = rest
	flat["nullable"] = true
	return flat
}

// GenaiSchemaToJSONMap converts a genai schema to the JSON-schema map both
// APIs take, with every "type" lowercased at every depth.
func GenaiSchemaToJSONMap(s *genai.Schema) (map[string]any, error) {
	if s == nil {
		return nil, nil
	}
	bytes, err := json.Marshal(s)
	if err != nil {
		return nil, fmt.Errorf("typeutil: marshal schema: %w", err)
	}
	var result map[string]any
	if err := json.Unmarshal(bytes, &result); err != nil {
		return nil, fmt.Errorf("typeutil: unmarshal schema: %w", err)
	}
	lowercaseSchemaTypes(result)
	return result, nil
}

// lowercaseSchemaTypes rewrites genai's upper-case type names, at every depth
// of a decoded schema, to the lower case JSON Schema uses.
func lowercaseSchemaTypes(val any) {
	switch v := val.(type) {
	case map[string]any:
		if t, ok := v["type"]; ok {
			switch tVal := t.(type) {
			case string:
				v["type"] = strings.ToLower(tVal)
			case []any:
				for i, item := range tVal {
					if str, ok := item.(string); ok {
						tVal[i] = strings.ToLower(str)
					}
				}
			}
		}
		for _, child := range v {
			lowercaseSchemaTypes(child)
		}
	case []any:
		for _, child := range v {
			lowercaseSchemaTypes(child)
		}
	}
}
