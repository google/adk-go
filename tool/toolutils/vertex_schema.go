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

// SanitizeSchemaForVertex strips annotation siblings (title, description,
// default, …) from maps that contain anyOf/oneOf so Vertex AI accepts the
// declaration. Validation keywords beside a union (type, properties,
// required, discriminator, …) and maps that are not schemas are left
// unchanged — the same shapes adk-python leaves alone. Nested schemas are
// walked recursively. See https://github.com/google/adk-go/issues/1659.
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
				// Copy prefixes with make+copy so a rewrite of s[0] is kept:
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
		sanitized, changed := sanitizeSchemaForVertex(value)
		if changed && out == nil {
			out = make(map[string]any, len(schema))
			for copiedKey, copiedValue := range schema {
				out[copiedKey] = copiedValue
			}
		}
		if out != nil {
			out[key] = sanitized
		}
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

func isAnnotation(key string) bool {
	switch key {
	case "$comment", "$id", "$schema", "default", "deprecated", "description", "examples", "readOnly", "title", "writeOnly":
		return true
	default:
		return false
	}
}
