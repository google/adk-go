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

package converters

import (
	"maps"
	"slices"
	"strings"
	"unicode"
)

// userKeyFields hold map-valued genai fields whose keys are user data (tool
// arguments and results, part metadata) and must reach the caller unchanged.
var userKeyFields = map[string]bool{"args": true, "response": true, "partMetadata": true}

// CamelizeKeys returns a copy of a decoded JSON value with snake_case object
// keys renamed to the lowerCamelCase names in genai's JSON tags, so that the
// result decodes into a genai type with encoding/json. Python agents
// (pydantic's model_dump without by_alias) and the adk-go Agent Engine server
// (helper.ConvertSnake) write snake_case keys; camelCase input passes through
// unchanged.
//
// Values under args, response and partMetadata are user data and are returned
// unchanged, at any depth. When a key and its camelCase form are both present,
// the camelCase value is kept; when several snake_case keys collapse to one
// name, the first in sorted order is kept. The result does not depend on map
// iteration order.
//
// CamelizeKeys is total and does not mutate v. Values other than
// map[string]any and []any are returned as they are, and the result shares
// them with v. It is safe for concurrent use.
func CamelizeKeys(v any) any {
	switch v := v.(type) {
	case map[string]any:
		out := make(map[string]any, len(v))
		for _, k := range slices.Sorted(maps.Keys(v)) {
			ck := snakeToCamel(k)
			if ck != k {
				if _, ok := v[ck]; ok {
					continue
				}
				if _, ok := out[ck]; ok {
					continue
				}
			}
			if userKeyFields[ck] {
				out[ck] = v[k]
				continue
			}
			out[ck] = CamelizeKeys(v[k])
		}
		return out
	case []any:
		out := make([]any, len(v))
		for i, val := range v {
			out[i] = CamelizeKeys(val)
		}
		return out
	default:
		return v
	}
}

func snakeToCamel(s string) string {
	if !strings.Contains(s, "_") {
		return s
	}
	var b strings.Builder
	upper := false
	for _, r := range s {
		if r == '_' {
			upper = b.Len() > 0
			continue
		}
		if upper {
			r = unicode.ToUpper(r)
			upper = false
		}
		b.WriteRune(r)
	}
	return b.String()
}

// StdBase64Fields rewrites, in place, every thoughtSignature string and every
// data string inside an inlineData object under v, at any depth (a function
// response carries its own media parts), from URL-safe or unpadded base64 to
// standard padded base64. Pydantic writes bytes as URL-safe base64, while
// encoding/json decodes []byte only from standard base64, so a Python agent's
// value would otherwise fail the whole decode. Values under args, response and
// partMetadata are user data and are left as written. Strings that are not
// base64 at all are left for the decoder to reject.
//
// v must be a value returned by CamelizeKeys, which the caller owns: the
// containers StdBase64Fields writes to must not be shared.
func StdBase64Fields(v any) {
	switch v := v.(type) {
	case map[string]any:
		for k, val := range v {
			if userKeyFields[k] {
				continue
			}
			switch k {
			case "thoughtSignature":
				if s, ok := val.(string); ok {
					v[k] = stdBase64(s)
				}
			case "inlineData":
				if blob, ok := val.(map[string]any); ok {
					if s, ok := blob["data"].(string); ok {
						blob["data"] = stdBase64(s)
					}
				}
			}
			StdBase64Fields(val)
		}
	case []any:
		for _, val := range v {
			StdBase64Fields(val)
		}
	}
}

// stdBase64 converts URL-safe base64 to the standard alphabet and restores
// missing padding. Standard padded input is returned unchanged.
func stdBase64(s string) string {
	if strings.ContainsAny(s, "-_") {
		s = strings.NewReplacer("-", "+", "_", "/").Replace(s)
	}
	if r := len(s) % 4; r != 0 {
		s += strings.Repeat("=", 4-r)
	}
	return s
}
