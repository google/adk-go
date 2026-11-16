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
	"fmt"
	"maps"
	"regexp"
	"slices"
	"strings"

	"github.com/google/jsonschema-go/jsonschema"
	"google.golang.org/genai"
)

// conform validates v against schema the way adk-python's pydantic
// validation of node data does (utils/_schema_utils.py,
// validate_node_data): it takes the JSON form of v, fills schema defaults
// for absent properties, validates, and then drops a null from every
// property the schema lists as optional (model_dump(exclude_none=True)).
// Defaults are filled before validating because pydantic fills a default
// while validating, so a property with a default is never missing.
//
// form is the conformed JSON form; it never shares a map or slice with v,
// so v is never modified. unchanged reports that form is a faithful copy
// of v: nothing needed encoding/json to convert, no default was filled and
// no null was dropped. On failure conform returns the error and a nil
// form.
func conform(v any, schema *jsonschema.Resolved) (form any, unchanged bool, err error) {
	form, converted, err := jsonForm(v)
	if err != nil {
		return nil, false, err
	}
	root := schema.Schema()
	filled, err := fillDefaults(root, root, form)
	if err != nil {
		return nil, false, err
	}
	if err := schema.Validate(form); err != nil {
		return nil, false, err
	}
	stripped, err := stripNulls(root, root, form)
	if err != nil {
		return nil, false, err
	}
	return form, !converted && !filled && !stripped, nil
}

// conformed returns the conformed form of v, or nil and the error.
func conformed(v any, schema *jsonschema.Resolved) (any, error) {
	form, _, err := conform(v, schema)
	if err != nil {
		return nil, err
	}
	return form, nil
}

// jsonForm returns a copy of v built only from the values encoding/json
// decodes into: map[string]any, []any, string, bool, numbers and nil.
// Maps, slices and scalars already in that shape are copied and keep their
// Go scalar types (an int stays an int). Any other value (a struct, a
// pointer, a typed map or slice) is converted with json.Marshal and
// json.Unmarshal, so a struct arrives as a map[string]any; converted
// reports that at least one value inside v was converted this way.
func jsonForm(v any) (form any, converted bool, err error) {
	switch val := v.(type) {
	case nil, string, bool,
		int, int8, int16, int32, int64,
		uint, uint8, uint16, uint32, uint64,
		float32, float64:
		return v, false, nil
	case map[string]any:
		m := make(map[string]any, len(val))
		for k, item := range val {
			f, c, err := jsonForm(item)
			if err != nil {
				return nil, false, err
			}
			m[k] = f
			converted = converted || c
		}
		return m, converted, nil
	case []any:
		s := make([]any, len(val))
		for i, item := range val {
			f, c, err := jsonForm(item)
			if err != nil {
				return nil, false, err
			}
			s[i] = f
			converted = converted || c
		}
		return s, converted, nil
	default:
		raw, err := json.Marshal(v)
		if err != nil {
			return nil, false, fmt.Errorf("encoding %T as JSON: %w", v, err)
		}
		var decoded any
		if err := json.Unmarshal(raw, &decoded); err != nil {
			return nil, false, fmt.Errorf("decoding %T from JSON: %w", v, err)
		}
		return decoded, true, nil
	}
}

// fillDefaults adds the default of every property of s that is absent from
// an object in v, at every position walkObjects reaches, whether the
// property is required or not. A present value, null included, is never
// replaced, and a default of null adds nothing. Each default is decoded
// from its JSON on every call, so the filled values share nothing with the
// schema. v is modified in place; it must be a JSON form (see jsonForm).
// changed reports that a default was added.
func fillDefaults(root, s *jsonschema.Schema, v any) (changed bool, err error) {
	return walkObjects(root, s, v, fillObjectDefaults)
}

// fillObjectDefaults fills the defaults of s's properties that m lacks.
func fillObjectDefaults(s *jsonschema.Schema, m map[string]any) (bool, error) {
	var changed bool
	for name, prop := range s.Properties {
		if prop == nil || len(prop.Default) == 0 {
			continue
		}
		if _, present := m[name]; present {
			continue
		}
		var def any
		if err := json.Unmarshal(prop.Default, &def); err != nil {
			return false, fmt.Errorf("default of property %q: %w", name, err)
		}
		if def == nil {
			// pydantic leaves a None default out of model_dump(exclude_none=True).
			continue
		}
		m[name] = def
		changed = true
	}
	return changed, nil
}

// stripNulls deletes from every object in v a key whose value is null when
// the schema describing the object lists the key under properties and not
// under required, at every position walkObjects reaches. Keys the schema
// does not list are left alone. v is modified in place; changed reports
// that a key was deleted. It never fails; err is walkObjects' error.
func stripNulls(root, s *jsonschema.Schema, v any) (changed bool, err error) {
	return walkObjects(root, s, v, stripObjectNulls)
}

// stripObjectNulls deletes from m the optional properties of s that hold
// null.
func stripObjectNulls(s *jsonschema.Schema, m map[string]any) (bool, error) {
	var changed bool
	for name := range s.Properties {
		cur, present := m[name]
		if !present || cur != nil || slices.Contains(s.Required, name) {
			continue
		}
		delete(m, name)
		changed = true
	}
	return changed, nil
}

// objectVisitor changes the object m, which the schema s describes, and
// reports whether it changed anything.
type objectVisitor func(s *jsonschema.Schema, m map[string]any) (bool, error)

// walkObjects calls visit with every object in v and each schema that
// describes it: s itself, and the schemas reached from s at every depth
// through properties, patternProperties (for each key properties does not
// declare and the pattern matches), additionalProperties (for keys neither
// declares), prefixItems (item i), items (the items after prefixItems),
// allOf (every member, for the same value) and a local $ref. root is the
// schema document a $ref is resolved against: "#", "#/$defs/<name>" and
// "#/definitions/<name>" are followed; any other reference is ignored.
// visit sees an object after the values it holds have been walked, so a
// value visit adds is not walked. changed reports that a visit changed
// something; the first visit error ends the walk and is returned.
//
// Schemas reached only through anyOf or oneOf are not walked.
func walkObjects(root, s *jsonschema.Schema, v any, visit objectVisitor) (changed bool, err error) {
	return walkObjectsAt(root, s, v, nil, visit)
}

// walkObjectsAt is walkObjects for one value position. followed holds the
// references already followed at this position, so a $ref chain that
// loops back on itself, directly or through allOf, stops; it starts empty
// at every property, item and additional property, where the value is one
// level deeper.
func walkObjectsAt(root, s *jsonschema.Schema, v any, followed map[string]bool, visit objectVisitor) (bool, error) {
	if s == nil {
		return false, nil
	}
	var changed bool
	switch val := v.(type) {
	case map[string]any:
		c, err := walkObject(root, s, val, visit)
		if err != nil {
			return false, err
		}
		changed = c
	case []any:
		for i, item := range val {
			itemSchema := s.Items
			if i < len(s.PrefixItems) {
				itemSchema = s.PrefixItems[i]
			}
			c, err := walkObjectsAt(root, itemSchema, item, nil, visit)
			if err != nil {
				return false, err
			}
			changed = changed || c
		}
	}
	for _, member := range s.AllOf {
		c, err := walkObjectsAt(root, member, v, followed, visit)
		if err != nil {
			return false, err
		}
		changed = changed || c
	}
	if s.Ref == "" || followed[s.Ref] {
		return changed, nil
	}
	target := localRef(root, s.Ref)
	if target == nil {
		return changed, nil
	}
	if followed == nil {
		followed = map[string]bool{}
	}
	followed[s.Ref] = true
	c, err := walkObjectsAt(root, target, v, followed, visit)
	if err != nil {
		return false, err
	}
	return changed || c, nil
}

// walkObject walks each value of m with the schemas s gives its key (see
// keySchemas), and then visits m.
func walkObject(root, s *jsonschema.Schema, m map[string]any, visit objectVisitor) (bool, error) {
	var changed bool
	for name, cur := range m {
		schemas, err := keySchemas(s, name)
		if err != nil {
			return false, err
		}
		for _, ks := range schemas {
			c, err := walkObjectsAt(root, ks, cur, nil, visit)
			if err != nil {
				return false, err
			}
			changed = changed || c
		}
	}
	c, err := visit(s, m)
	if err != nil {
		return false, err
	}
	return changed || c, nil
}

// keySchemas returns the schemas s gives the value of the key name: the
// declared property; else the schema of every patternProperties pattern
// that matches name, in pattern order; else additionalProperties, which
// may be nil.
func keySchemas(s *jsonschema.Schema, name string) ([]*jsonschema.Schema, error) {
	if prop, declared := s.Properties[name]; declared {
		return []*jsonschema.Schema{prop}, nil
	}
	var matched []*jsonschema.Schema
	for _, pattern := range slices.Sorted(maps.Keys(s.PatternProperties)) {
		re, err := regexp.Compile(pattern)
		if err != nil {
			return nil, fmt.Errorf("patternProperties[%q]: %w", pattern, err)
		}
		if re.MatchString(name) {
			matched = append(matched, s.PatternProperties[pattern])
		}
	}
	if len(matched) > 0 {
		return matched, nil
	}
	return []*jsonschema.Schema{s.AdditionalProperties}, nil
}

// localRef returns the schema a local reference names in root, or nil
// when the reference is not local or names nothing.
func localRef(root *jsonschema.Schema, ref string) *jsonschema.Schema {
	if ref == "#" {
		return root
	}
	if name, ok := strings.CutPrefix(ref, "#/$defs/"); ok {
		return root.Defs[name]
	}
	if name, ok := strings.CutPrefix(ref, "#/definitions/"); ok {
		return root.Definitions[name]
	}
	return nil
}

// schemaExpectsObject reports whether schema's root type is, or includes,
// "object".
func schemaExpectsObject(schema *jsonschema.Resolved) bool {
	root := schema.Schema()
	return root != nil && (root.Type == "object" || slices.Contains(root.Types, "object"))
}

// asContent returns the *genai.Content that data holds, directly or as a
// genai.Content value.
func asContent(data any) (*genai.Content, bool) {
	switch c := data.(type) {
	case *genai.Content:
		return c, true
	case genai.Content:
		return &c, true
	}
	return nil, false
}
