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
	"reflect"
	"slices"

	"github.com/google/jsonschema-go/jsonschema"
	"google.golang.org/genai"
)

// modelOutputSchema returns the schema an agent node asks the model to answer
// with, derived from the node's output schema, or nil to leave the model
// unconstrained as it was before typed nodes passed their schema on.
//
// Only a closed, non-null object with at least one property qualifies. A
// string Output sent as a response schema makes the model answer with a
// quoted JSON string, and the set_model_response tool needs object parameters,
// so primitive, slice, map and nullable (pointer) outputs stay unconstrained.
// Every object must also set additionalProperties to false, as a Go struct
// does: set_model_response rejects keys its schema does not list, so an open
// object would turn an answer the node accepts into an error.
//
// A nullable property (a pointer or slice field) keeps Nullable. The
// set_model_response tool accepts a null argument only when its schema is
// nullable, so dropping it would turn a null the node accepts into an error.
//
// The result is a hint, not the contract: BaseNode.ValidateOutput still checks
// the reply against the full schema. So a schema that genai.Schema cannot
// carry yields nil rather than an error, and constructing a node never fails
// because of it.
//
// adk-python has no typed node: an LLM agent node there is sent the agent's
// own output_schema, whatever its shape (workflow/_llm_agent_wrapper.py). The
// limits here exist because the schema is derived from a Go type the agent's
// author never declared, so it must not break a node that works without it.
func modelOutputSchema(resolved *jsonschema.Resolved) *genai.Schema {
	if resolved == nil {
		return nil
	}
	root := resolved.Schema()
	if root == nil || slices.Contains(root.Types, "null") {
		return nil
	}
	s, ok := toGenaiSchema(root)
	if !ok || s.Type != genai.TypeObject {
		return nil
	}
	return s
}

var genaiTypes = map[string]genai.Type{
	"string":  genai.TypeString,
	"integer": genai.TypeInteger,
	"number":  genai.TypeNumber,
	"boolean": genai.TypeBoolean,
	"array":   genai.TypeArray,
	"object":  genai.TypeObject,
}

// toGenaiSchema converts the shape of s: types, properties, required fields,
// array items, string enums, title and description. Value constraints
// (bounds, lengths, pattern) are dropped: ValidateOutput enforces them, and
// some providers reject them in a response schema (#2036). Annotations such as
// format, default and examples are dropped too; ValidateOutput does not check
// them either. A "null" in a type list becomes Nullable.
// It reports false for anything else, such as an untyped (true) schema from an
// `any` field, a map or other open object, a required name with no property,
// a composition keyword, a reference, or an enum that is not on a string.
func toGenaiSchema(s *jsonschema.Schema) (*genai.Schema, bool) {
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
		if len(s.Properties) == 0 || !isFalseSchema(s.AdditionalProperties) {
			return nil, false
		}
		for _, name := range s.Required {
			if _, ok := s.Properties[name]; !ok {
				return nil, false
			}
		}
		out.Properties = make(map[string]*genai.Schema, len(s.Properties))
		for name, prop := range s.Properties {
			p, ok := toGenaiSchema(prop)
			if !ok {
				return nil, false
			}
			out.Properties[name] = p
		}
		out.Required = append([]string(nil), s.Required...)
	case genai.TypeArray:
		items, ok := toGenaiSchema(s.Items)
		if !ok {
			return nil, false
		}
		out.Items = items
	}
	return out, true
}

// onlyKnownKeywords reports whether s uses nothing beyond the keywords
// toGenaiSchema converts or deliberately drops. It clears those and checks
// that nothing is left, so a keyword added to jsonschema.Schema later is
// rejected rather than silently lost.
func onlyKnownKeywords(s jsonschema.Schema) bool {
	// Converted.
	s.Type, s.Types, s.Enum = "", nil, nil
	s.Title, s.Description = "", ""
	s.Properties, s.Required, s.Items = nil, nil, nil
	// Dropped. additionalProperties: false only forbids extra keys, which
	// ValidateOutput and set_model_response both enforce; toGenaiSchema
	// requires it on every object. Any other value describes a map.
	if isFalseSchema(s.AdditionalProperties) {
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

// isFalseSchema reports whether s is the schema that matches nothing, which
// jsonschema-go writes as {"not": {}}.
func isFalseSchema(s *jsonschema.Schema) bool {
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
