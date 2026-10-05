// Copyright 2025 Google LLC
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

package shared

import (
	"bytes"
	"encoding/json"
	"fmt"
	"sort"
	"strings"

	"google.golang.org/genai"
)

// NormalizeSchema round-trips a JSON schema of any Go shape into a map,
// keeping its numbers as written.
func NormalizeSchema(schema any) (map[string]any, error) {
	if schema == nil {
		return nil, ErrEmptyJSONSchema
	}
	data, err := json.Marshal(schema)
	if err != nil {
		return nil, fmt.Errorf("openai: marshal json schema: %w", err)
	}
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.UseNumber()
	var result map[string]any
	if err := decoder.Decode(&result); err != nil {
		return nil, fmt.Errorf("openai: unmarshal json schema: %w", err)
	}
	preserveSchemaNumbers(result)
	return result, nil
}

// preserveSchemaNumbers keeps numeric constraints as raw JSON. The OpenAI SDK
// otherwise serializes json.Number values as strings.
func preserveSchemaNumbers(val any) any {
	switch v := val.(type) {
	case json.Number:
		return json.RawMessage(v.String())
	case map[string]any:
		for key, child := range v {
			v[key] = preserveSchemaNumbers(child)
		}
	case []any:
		for i, child := range v {
			v[i] = preserveSchemaNumbers(child)
		}
	}
	return val
}

// EnforceStrictOpenAISchema recursively walks the schema and enforces the rules
// required by OpenAI's structured outputs with strict=true: every object type
// carries properties, additionalProperties=false and a required array naming
// every property, and a $ref keeps no siblings. An object that declares no
// properties is given an empty properties map and an empty required array
// alongside additionalProperties=false, because the API rejects the whole
// request when any object in the schema omits one of the three. Any
// additionalProperties the caller wrote is replaced: strict mode accepts only
// false, so a schema spelling a map as additionalProperties={"type":"string"}
// becomes an empty object rather than the 400 it would otherwise draw.
//
// Treating a property-less object that way diverges from adk-python
// deliberately. Its _enforce_strict_openai_schema rewrites an object only when
// the schema already carries a properties key, which leaves one without to fail
// the same request.
func EnforceStrictOpenAISchema(val any) {
	schema, ok := val.(map[string]any)
	if !ok {
		return
	}

	if _, hasRef := schema["$ref"]; hasRef {
		for key := range schema {
			if key != "$ref" {
				delete(schema, key)
			}
		}
		return
	}

	t, hasType := schema["type"]
	isObj := hasType && t == "object"
	propsMap, _ := schema["properties"].(map[string]any)

	if isObj {
		if propsMap == nil {
			propsMap = map[string]any{}
			schema["properties"] = propsMap
		}
		schema["additionalProperties"] = false
		req := make([]string, 0, len(propsMap))
		for k := range propsMap {
			req = append(req, k)
		}
		sort.Strings(req)
		schema["required"] = req
	}

	if defsVal, ok := schema["$defs"]; ok {
		if defsMap, ok := defsVal.(map[string]any); ok {
			for _, defn := range defsMap {
				EnforceStrictOpenAISchema(defn)
			}
		}
	}

	for _, prop := range propsMap {
		EnforceStrictOpenAISchema(prop)
	}

	for _, key := range []string{"anyOf", "oneOf", "allOf"} {
		if arrVal, ok := schema[key]; ok {
			if arr, ok := arrVal.([]any); ok {
				for _, item := range arr {
					EnforceStrictOpenAISchema(item)
				}
			}
		}
	}

	if itemsVal, ok := schema["items"]; ok {
		if _, isMap := itemsVal.(map[string]any); isMap {
			EnforceStrictOpenAISchema(itemsVal)
		}
	}
}

// SchemaToMap converts a genai schema to the JSON-schema map both APIs take,
// with its type names lowercased.
func SchemaToMap(schema *genai.Schema) (map[string]any, error) {
	if schema == nil {
		return nil, nil
	}
	bytes, err := json.Marshal(schema)
	if err != nil {
		return nil, fmt.Errorf("openai: marshal schema: %w", err)
	}
	var result map[string]any
	if err := json.Unmarshal(bytes, &result); err != nil {
		return nil, fmt.Errorf("openai: unmarshal schema: %w", err)
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
