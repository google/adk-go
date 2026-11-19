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

package fencing

import (
	"maps"
	"slices"
)

// valuePositions are the schema keywords whose values are instance data, not
// schemas, so nothing under them is fenced.
var valuePositions = map[string]bool{
	"default":  true,
	"enum":     true,
	"examples": true,
	"example":  true,
	"const":    true,
}

// schemaMapKeys are the schema keywords whose value maps names to schemas, so a
// name there such as "default" is a name and not a keyword.
var schemaMapKeys = map[string]bool{
	"properties":        true,
	"patternProperties": true,
	"$defs":             true,
	"definitions":       true,
	"dependentSchemas":  true,
}

// Schema returns a copy of a JSON-decoded schema with every prose description fenced and untrusted markers elided.
//
// schema has the shape encoding/json decodes into an any: map[string]any,
// []any, string, float64, bool or nil. In the copy:
//
//   - A "description" key of a schema with a string value is passed through
//     [ToolDescription]. With any other value it is walked like any other value.
//   - The values under "default", "enum", "examples", "example" and "const" are
//     instance data: nothing in them is fenced, so a "description" key there
//     stays data. Their strings and keys are still elided.
//   - The map under "properties", "patternProperties", "$defs", "definitions"
//     or "dependentSchemas" maps names to schemas: each name is elided and each
//     value is walked as a schema, so a property called "default" is a schema.
//   - Every other key and every other string, at any depth, is passed through
//     [ElideToolDescriptionMarkers] without being fenced.
//   - Numbers, booleans and nil are returned as they are.
//
// The input is not modified, and no map or slice of the result is shared with
// the input. A nil map[string]any or nil []any comes back as the same typed
// nil, at any depth. A value of any other dynamic type, such as an int or a
// *jsonschema.Schema, is returned unchanged and so may share memory with the
// input.
//
// When elision makes two keys of one map equal, the value of the key that
// sorts last in the input wins, so the result does not depend on map order.
// adk-python keeps the key that comes later in its insertion order instead.
//
// Schema is safe for concurrent use, including on one shared input, as long as
// nothing writes to the input meanwhile.
func Schema(schema any) any {
	return walk(schema, false)
}

// walk copies v. inValue reports whether v lies under a value position, where
// nothing is fenced and keywords are data.
func walk(v any, inValue bool) any {
	switch v := v.(type) {
	case map[string]any:
		if v == nil {
			return v
		}
		out := make(map[string]any, len(v))
		for _, k := range slices.Sorted(maps.Keys(v)) {
			out[ElideToolDescriptionMarkers(k)] = walkEntry(k, v[k], inValue)
		}
		return out
	case []any:
		if v == nil {
			return v
		}
		out := make([]any, len(v))
		for i, item := range v {
			out[i] = walk(item, inValue)
		}
		return out
	case string:
		return ElideToolDescriptionMarkers(v)
	}
	return v
}

// walkEntry copies the value val found under key in a map walked with inValue.
func walkEntry(key string, val any, inValue bool) any {
	if !inValue {
		switch {
		case key == "description":
			if s, ok := val.(string); ok {
				return ToolDescription(s)
			}
		case valuePositions[key]:
			return walk(val, true)
		case schemaMapKeys[key]:
			if m, ok := val.(map[string]any); ok {
				return schemaMap(m)
			}
		}
	}
	return walk(val, inValue)
}

// schemaMap copies a map from names to schemas: it elides each name and walks
// each value as a schema.
func schemaMap(m map[string]any) map[string]any {
	if m == nil {
		return nil
	}
	out := make(map[string]any, len(m))
	for _, name := range slices.Sorted(maps.Keys(m)) {
		out[ElideToolDescriptionMarkers(name)] = walk(m[name], false)
	}
	return out
}
