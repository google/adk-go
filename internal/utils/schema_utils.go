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

package utils

import (
	"encoding/json"
	"fmt"
	"math"
	"reflect"

	"google.golang.org/genai"
)

// matchType checks if the value matches the schema type.
func matchType(value any, schema *genai.Schema, isInput bool) (bool, error) {
	if schema == nil {
		return false, fmt.Errorf("schema is nil")
	}

	if value == nil {
		return false, nil
	}

	switch schema.Type {
	case genai.TypeString:
		_, ok := value.(string)
		return ok, nil
	case genai.TypeInteger:
		f, ok := value.(float64)
		if !ok {
			return false, nil
		}
		return f == math.Trunc(f), nil
	case genai.TypeBoolean:
		_, ok := value.(bool)
		return ok, nil
	case genai.TypeNumber:
		_, ok := value.(float64)
		return ok, nil
	case genai.TypeArray:
		val := reflect.ValueOf(value)
		if val.Kind() != reflect.Slice {
			return false, nil
		}
		if schema.Items == nil {
			return false, fmt.Errorf("array schema missing items definition")
		}
		for i := 0; i < val.Len(); i++ {
			ok, err := matchType(val.Index(i).Interface(), schema.Items, isInput)
			if err != nil {
				return false, fmt.Errorf("array item %d: %w", i, err)
			}
			if !ok {
				return false, nil
			}
		}
		return true, nil
	case genai.TypeObject:
		obj, ok := value.(map[string]any)
		if !ok {
			return false, nil
		}
		err := ValidateMapOnSchema(obj, schema, isInput)
		return err == nil, err
	default:
		return false, fmt.Errorf("unsupported type: %s", schema.Type)
	}
}

// ValidateMapOnSchema validates a map against a schema.
func ValidateMapOnSchema(args map[string]any, schema *genai.Schema, isInput bool) error {
	if schema == nil {
		return fmt.Errorf("schema cannot be nil")
	}

	properties := schema.Properties
	if properties == nil {
		properties = make(map[string]*genai.Schema)
	}

	argType := "input"
	if !isInput {
		argType = "output"
	}

	for key, value := range args {
		propSchema, exists := properties[key]
		if !exists {
			// Note: OpenAPI schemas can allow additional properties. This implementation assumes strictness.
			return fmt.Errorf("%s arg: '%q' does not exist in schema properties", argType, key)
		}
		ok, err := matchType(value, propSchema, isInput)
		if err != nil {
			return fmt.Errorf("%s arg: '%q' validation failed: %w", argType, key, err)
		}
		if !ok {
			return fmt.Errorf("%s arg: '%q' type mismatch, expected schema type %s, got value %v of type %T", argType, key, propSchema.Type, value, value)
		}
	}

	for _, requiredKey := range schema.Required {
		if _, exists := args[requiredKey]; !exists {
			return fmt.Errorf("%q args does not contain required key: '%q'", argType, requiredKey)
		}
	}
	return nil
}

// ValidateOutputSchema validates an output JSON string against a schema.
func ValidateOutputSchema(output string, schema *genai.Schema) (map[string]any, error) {
	if schema == nil {
		return nil, fmt.Errorf("schema cannot be nil")
	}
	var outputMap map[string]any
	err := json.Unmarshal([]byte(output), &outputMap)
	if err != nil {
		return nil, fmt.Errorf("failed to parse output JSON: %w", err)
	}

	if err := ValidateMapOnSchema(outputMap, schema, false); err != nil { // isInput = false
		return nil, err
	}
	return outputMap, nil
}
