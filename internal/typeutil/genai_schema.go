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
	"encoding/json"
	"fmt"
	"strings"

	"google.golang.org/genai"
)

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
