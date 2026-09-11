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

package utils

import (
	"bytes"
	"encoding/json"
	"sort"

	"github.com/google/jsonschema-go/jsonschema"
)

// unorderedArrayKeywords are JSON Schema keywords whose array value is a set,
// so two schemas that differ only in the order of these arrays are equivalent.
// jsonschema-go builds "required" in struct-field declaration order, and a
// "type" union may be supplied in any order, so both are sorted before
// comparison to avoid spurious mismatches between equivalent schemas.
var unorderedArrayKeywords = map[string]bool{
	"required": true,
	"type":     true,
}

// CanonicalSchemaJSON marshals the schema to JSON, parses it back, and
// re-emits it with object keys sorted alphabetically (recursively). Arrays keep
// their original order, except the unordered set keywords (see
// unorderedArrayKeywords), whose elements are sorted so that equivalent schemas
// canonicalize identically.
func CanonicalSchemaJSON(s *jsonschema.Schema) ([]byte, error) {
	raw, err := json.Marshal(s)
	if err != nil {
		return nil, err
	}
	var v any
	if err := json.Unmarshal(raw, &v); err != nil {
		return nil, err
	}
	return canonicalize(v)
}

// canonicalize recursively serializes v with sorted map keys. Slices keep their
// order unless they are the value of an unordered set keyword, in which case
// their elements are sorted. Primitive values are encoded via json.Marshal.
func canonicalize(v any) ([]byte, error) {
	switch val := v.(type) {
	case map[string]any:
		if val == nil {
			return []byte("null"), nil
		}
		keys := make([]string, 0, len(val))
		for k := range val {
			keys = append(keys, k)
		}
		sort.Strings(keys)

		var buf bytes.Buffer
		buf.WriteByte('{')
		for i, k := range keys {
			if i > 0 {
				buf.WriteByte(',')
			}
			keyBytes, err := json.Marshal(k)
			if err != nil {
				return nil, err
			}
			buf.Write(keyBytes)
			buf.WriteByte(':')

			var valBytes []byte
			if arr, ok := val[k].([]any); ok && unorderedArrayKeywords[k] {
				valBytes, err = canonicalizeUnorderedArray(arr)
			} else {
				valBytes, err = canonicalize(val[k])
			}
			if err != nil {
				return nil, err
			}
			buf.Write(valBytes)
		}
		buf.WriteByte('}')
		return buf.Bytes(), nil

	case []any:
		if val == nil {
			return []byte("null"), nil
		}
		var buf bytes.Buffer
		buf.WriteByte('[')
		for i, item := range val {
			if i > 0 {
				buf.WriteByte(',')
			}
			itemBytes, err := canonicalize(item)
			if err != nil {
				return nil, err
			}
			buf.Write(itemBytes)
		}
		buf.WriteByte(']')
		return buf.Bytes(), nil

	default:
		return json.Marshal(val)
	}
}

// canonicalizeUnorderedArray canonicalizes each element and sorts the encoded
// results, so a set-valued keyword compares equal regardless of the order its
// elements were declared in.
func canonicalizeUnorderedArray(items []any) ([]byte, error) {
	encoded := make([][]byte, len(items))
	for i, item := range items {
		b, err := canonicalize(item)
		if err != nil {
			return nil, err
		}
		encoded[i] = b
	}
	sort.Slice(encoded, func(i, j int) bool {
		return bytes.Compare(encoded[i], encoded[j]) < 0
	})

	var buf bytes.Buffer
	buf.WriteByte('[')
	for i, b := range encoded {
		if i > 0 {
			buf.WriteByte(',')
		}
		buf.Write(b)
	}
	buf.WriteByte(']')
	return buf.Bytes(), nil
}
