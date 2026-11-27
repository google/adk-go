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
	"maps"
	"regexp"
	"slices"
	"strings"

	"google.golang.org/genai"
)

// DefaultResponseFormatName is the structured-output format name both
// endpoints send when the caller's schema has no title OpenAI would accept.
const DefaultResponseFormatName = "adk_response"

// responseFormatNamePattern is the rule openai-go documents on the format name
// of both endpoints: "Must be a-z, A-Z, 0-9, or contain underscores and
// dashes, with a maximum length of 64."
var responseFormatNamePattern = regexp.MustCompile(`^[a-zA-Z0-9_-]{1,64}$`)

// ResponseFormatName returns the structured-output format name for a schema:
// its title when OpenAI accepts it as a name, and DefaultResponseFormatName
// otherwise. genai puts no constraint on Title, so a title such as
// "Weather Report" would otherwise fail every request. The title still reaches
// the model unchanged in the schema body, which the API does not restrict.
//
// adk-python's LiteLLM path copies the title straight into the name
// (models/lite_llm.py, _to_litellm_response_format) and fails the same way.
func ResponseFormatName(schema *genai.Schema) string {
	if schema != nil && responseFormatNamePattern.MatchString(schema.Title) {
		return schema.Title
	}
	return DefaultResponseFormatName
}

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
//
// When the root schema itself is a $ref into $defs (for example recursive model
// schemas produced by JSON schema generators), the referenced definition is
// inlined at the root instead of stripping $defs alongside $ref, because OpenAI
// rejects a root $ref and $defs still need to remain available for nested refs.
func EnforceStrictOpenAISchema(val any) {
	schema, ok := val.(map[string]any)
	if !ok {
		return
	}

	// Inline a $ref pointing into sibling $defs (for example recursive schemas
	// whose root is {"$ref": "#/$defs/...", "$defs": {...}}), keeping root
	// keys and stopping on cyclic or unresolvable references.
	if defsMap, ok := schema["$defs"].(map[string]any); ok {
		seen := map[string]bool{}
		for {
			refStr, ok := schema["$ref"].(string)
			if !ok {
				break
			}
			name, ok := strings.CutPrefix(refStr, "#/$defs/")
			if !ok || seen[name] {
				break
			}
			target, ok := defsMap[name].(map[string]any)
			if !ok {
				break
			}
			seen[name] = true
			delete(schema, "$ref")
			for k, v := range target {
				if _, exists := schema[k]; !exists {
					schema[k] = cloneSchemaValue(v)
				}
			}
		}
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
		// Strict mode rejects "required": null, so an empty propsMap must
		// still yield a non-nil slice.
		required := slices.AppendSeq(make([]string, 0, len(propsMap)), maps.Keys(propsMap))
		slices.Sort(required)
		schema["required"] = required
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

func cloneSchemaValue(val any) any {
	switch v := val.(type) {
	case map[string]any:
		cloned := make(map[string]any, len(v))
		for k, child := range v {
			cloned[k] = cloneSchemaValue(child)
		}
		return cloned
	case []any:
		cloned := make([]any, len(v))
		for i, child := range v {
			cloned[i] = cloneSchemaValue(child)
		}
		return cloned
	case []string:
		return slices.Clone(v)
	case json.RawMessage:
		return slices.Clone(v)
	case []byte:
		return slices.Clone(v)
	default:
		return v
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
