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

// Package schemautil converts between the schema forms ADK handles.
package schemautil

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"google.golang.org/genai"
)

// GeminiToJSONSchema converts a Gemini Schema into a JSON Schema map.
func GeminiToJSONSchema(schema *genai.Schema) (map[string]any, error) {
	if schema == nil {
		return nil, errors.New("schemautil: gemini schema is nil")
	}
	data, err := json.Marshal(schema)
	if err != nil {
		return nil, fmt.Errorf("schemautil: marshal gemini schema: %w", err)
	}
	var raw map[string]any
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, fmt.Errorf("schemautil: unmarshal gemini schema: %w", err)
	}
	normalizeGeminiSchemaMap(raw)
	return raw, nil
}

// normalizeGeminiSchemaMap normalizes a marshaled genai.Schema map in place into
// JSON Schema, recursively normalizing schema child nodes. Like adk-python's
// gemini_to_json_schema, it keeps the OpenAPI "nullable" keyword and leaves
// enum values as strings whatever the type, so the result is close to, but
// not strictly, standard JSON Schema.
func normalizeGeminiSchemaMap(raw map[string]any) {
	delete(raw, "propertyOrdering")

	typeStr, _ := raw["type"].(string)
	geminiType := genai.Type(strings.ToUpper(typeStr))

	// Omitted/unspecified types impose no JSON Schema type constraint. Vertex
	// function declarations also mark a union as OBJECT plus anyOf; that OBJECT
	// is a placeholder, so the anyOf branches carry the type.
	isUnionPlaceholder := false
	if geminiType == genai.TypeObject {
		if anyOf, ok := raw["anyOf"].([]any); ok {
			for _, elem := range anyOf {
				if subSchema, ok := elem.(map[string]any); ok {
					subTypeStr, _ := subSchema["type"].(string)
					subType := genai.Type(strings.ToUpper(subTypeStr))
					if subType != "" && subType != genai.TypeUnspecified && subType != genai.TypeObject {
						isUnionPlaceholder = true
						break
					}
				}
			}
		}
	}

	if geminiType == "" || geminiType == genai.TypeUnspecified || isUnionPlaceholder {
		delete(raw, "type")
	} else {
		raw["type"] = strings.ToLower(string(geminiType))
	}

	if nullable, ok := raw["nullable"].(bool); ok && !nullable {
		delete(raw, "nullable")
	}

	if geminiType != genai.TypeString {
		delete(raw, "pattern")
		delete(raw, "minLength")
		delete(raw, "maxLength")
	}

	if geminiType != genai.TypeNumber && geminiType != genai.TypeInteger {
		delete(raw, "minimum")
		delete(raw, "maximum")
	}

	if geminiType != genai.TypeArray {
		delete(raw, "items")
		delete(raw, "minItems")
		delete(raw, "maxItems")
	} else if items, ok := raw["items"].(map[string]any); ok {
		normalizeGeminiSchemaMap(items)
	}

	if geminiType != genai.TypeObject {
		delete(raw, "properties")
		delete(raw, "required")
		delete(raw, "minProperties")
		delete(raw, "maxProperties")
	} else if props, ok := raw["properties"].(map[string]any); ok {
		for _, v := range props {
			if propSchema, ok := v.(map[string]any); ok {
				normalizeGeminiSchemaMap(propSchema)
			}
		}
	}

	if anyOf, ok := raw["anyOf"].([]any); ok {
		for _, elem := range anyOf {
			if subSchema, ok := elem.(map[string]any); ok {
				normalizeGeminiSchemaMap(subSchema)
			}
		}
	}
}
