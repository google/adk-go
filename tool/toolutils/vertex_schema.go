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

import "encoding/json"

// SanitizeSchemaForVertex rewrites JSON Schema maps so that anyOf/oneOf
// subschemas do not carry sibling keywords. Vertex AI rejects declarations
// where any_of/one_of appears alongside other fields (e.g. description/title
// emitted by pydantic for Optional/Union). Sibling fields are merged into
// each branch. See https://github.com/google/adk-go/issues/1659.
func SanitizeSchemaForVertex(schema any) any {
	switch s := schema.(type) {
	case map[string]any:
		return sanitizeSchemaMap(s)
	case []any:
		out := make([]any, len(s))
		for i, v := range s {
			out[i] = SanitizeSchemaForVertex(v)
		}
		return out
	default:
		// jsonschema.Schema and other typed values — round-trip via JSON.
		b, err := json.Marshal(schema)
		if err != nil {
			return schema
		}
		var generic any
		if err := json.Unmarshal(b, &generic); err != nil {
			return schema
		}
		return SanitizeSchemaForVertex(generic)
	}
}

func sanitizeSchemaMap(schema map[string]any) map[string]any {
	out := make(map[string]any, len(schema))
	for k, v := range schema {
		out[k] = SanitizeSchemaForVertex(v)
	}

	for _, key := range []string{"anyOf", "oneOf"} {
		raw, ok := out[key]
		if !ok {
			continue
		}
		branches, ok := raw.([]any)
		if !ok || len(branches) == 0 {
			continue
		}
		siblings := map[string]any{}
		for k, v := range out {
			if k == key {
				continue
			}
			siblings[k] = v
		}
		if len(siblings) == 0 {
			continue
		}
		merged := make([]any, len(branches))
		for i, branch := range branches {
			bm, ok := branch.(map[string]any)
			if !ok {
				merged[i] = branch
				continue
			}
			nb := make(map[string]any, len(bm)+len(siblings))
			for k, v := range siblings {
				nb[k] = v
			}
			for k, v := range bm {
				nb[k] = v // branch wins on conflict
			}
			merged[i] = nb
		}
		return map[string]any{key: merged}
	}
	return out
}
