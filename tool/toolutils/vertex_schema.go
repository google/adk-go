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
	"bytes"
	"encoding/json"
	"reflect"
)

// SanitizeSchemaForVertex removes annotation siblings from a schema node
// whose only other keyword is a single anyOf or oneOf. Vertex AI rejects
// anyOf/oneOf beside title, description, default, and the other annotation
// keywords.
//
// Annotations stay when the union also has a validation keyword (type,
// properties, required, discriminator, minimum, and so on), and a node that
// has both anyOf and oneOf is left as it is. Those are the same shapes
// adk-python leaves alone.
//
// Nested schemas are walked. Instance values (const, enum, default,
// examples) are not. Dictionaries of schemas — properties,
// patternProperties, $defs, definitions, and dependentSchemas — are walked
// entry by entry and are not themselves treated as schemas.
// See https://github.com/google/adk-go/issues/1659.
func SanitizeSchemaForVertex(schema any) any {
	schema, _ = sanitizeSchemaForVertex(schema)
	return schema
}

func sanitizeSchemaForVertex(schema any) (any, bool) {
	switch s := schema.(type) {
	case map[string]any:
		return sanitizeSchemaMap(s)
	case []any:
		var out []any
		for i, value := range s {
			sanitized, changed := sanitizeSchemaForVertex(value)
			if !changed {
				continue
			}
			if out == nil {
				// Copy the whole slice on the first change so a rewrite of s[0] is kept:
				// append(nil, s[:0]...) is still nil and would drop the change.
				out = make([]any, len(s))
				copy(out, s)
			}
			out[i] = sanitized
		}
		if out == nil {
			return schema, false
		}
		return out, true
	default:
		if isScalar(schema) {
			return schema, false
		}
		return sanitizeTypedSchema(schema)
	}
}

func sanitizeTypedSchema(schema any) (any, bool) {
	encoded, err := json.Marshal(schema)
	if err != nil {
		return schema, false
	}
	var generic any
	decoder := json.NewDecoder(bytes.NewReader(encoded))
	decoder.UseNumber()
	if err := decoder.Decode(&generic); err != nil {
		return schema, false
	}
	sanitized, changed := sanitizeSchemaForVertex(generic)
	if !changed {
		return schema, false
	}
	return sanitized, true
}

func isScalar(value any) bool {
	if value == nil {
		return true
	}
	switch reflect.ValueOf(value).Kind() {
	case reflect.Bool, reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr,
		reflect.Float32, reflect.Float64, reflect.String:
		return true
	default:
		return false
	}
}

func sanitizeSchemaMap(schema map[string]any) (any, bool) {
	var out map[string]any
	for key, value := range schema {
		var sanitized any
		var changed bool
		switch {
		case isInstanceValue(key):
			continue
		case isSchemaDictionary(key):
			sanitized, changed = sanitizeSchemaDictionary(value)
		default:
			sanitized, changed = sanitizeSchemaForVertex(value)
		}
		if !changed {
			continue
		}
		if out == nil {
			out = make(map[string]any, len(schema))
			for copiedKey, copiedValue := range schema {
				out[copiedKey] = copiedValue
			}
		}
		out[key] = sanitized
	}
	childrenChanged := out != nil
	if out == nil {
		out = schema
	}

	var unionKey string
	var unionValue any
	for _, key := range []string{"anyOf", "oneOf"} {
		if value, ok := out[key]; ok {
			if unionKey != "" {
				// both anyOf and oneOf — leave alone (same as adk-python)
				return out, childrenChanged
			}
			unionKey = key
			unionValue = value
		}
	}
	if unionKey == "" {
		return out, childrenChanged
	}

	for key := range out {
		if key != "anyOf" && key != "oneOf" && !isAnnotation(key) {
			return out, childrenChanged
		}
	}
	// Already a bare union with no annotation siblings — keep original.
	if !childrenChanged {
		onlyUnion := true
		for key := range out {
			if key != unionKey {
				onlyUnion = false
				break
			}
		}
		if onlyUnion {
			return out, false
		}
	}
	return map[string]any{unionKey: unionValue}, true
}

// isInstanceValue reports keywords whose values are data, not schemas.
// default and examples are also annotations, so a bare union still drops
// the keyword itself; only the value is left unwalked.
func isInstanceValue(key string) bool {
	switch key {
	case "const", "default", "enum", "example", "examples":
		return true
	default:
		return false
	}
}

// isSchemaDictionary reports keywords whose value is a map from a name to a
// schema. The map is not a schema, so a property named anyOf must not be
// treated as the anyOf keyword.
func isSchemaDictionary(key string) bool {
	switch key {
	case "$defs", "definitions", "dependentSchemas", "patternProperties", "properties":
		return true
	default:
		return false
	}
}

func sanitizeSchemaDictionary(value any) (any, bool) {
	dict, ok := value.(map[string]any)
	if !ok {
		return value, false
	}
	var out map[string]any
	for key, child := range dict {
		sanitized, changed := sanitizeSchemaForVertex(child)
		if !changed {
			continue
		}
		if out == nil {
			out = make(map[string]any, len(dict))
			for copiedKey, copiedValue := range dict {
				out[copiedKey] = copiedValue
			}
		}
		out[key] = sanitized
	}
	if out == nil {
		return value, false
	}
	return out, true
}

func isAnnotation(key string) bool {
	switch key {
	case "$comment", "$id", "$schema", "default", "deprecated", "description", "examples", "readOnly", "title", "writeOnly":
		return true
	default:
		return false
	}
}
