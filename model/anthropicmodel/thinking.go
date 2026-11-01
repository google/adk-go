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

package anthropicmodel

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/anthropics/anthropic-sdk-go"
	"google.golang.org/genai"
)

func validateThinkingConfig(tc *genai.ThinkingConfig) error {
	if tc == nil {
		return nil
	}
	if tc.ThinkingLevel != "" {
		return fmt.Errorf("%w: ThinkingConfig.ThinkingLevel is not supported on Claude; configure ClientConfig.Effort instead", ErrInvalidConfig)
	}
	if tc.ThinkingBudget == nil {
		return fmt.Errorf("%w: ThinkingConfig.ThinkingBudget must be set explicitly when ThinkingConfig is provided (0 to disable, >0 for manual budget, <0 for adaptive thinking)", ErrInvalidConfig)
	}
	// IncludeThoughts is ignored by design; thought parts are always returned when thinking is active.
	return nil
}

func isThinkingEnabled(tc *genai.ThinkingConfig) bool {
	return tc != nil && tc.ThinkingBudget != nil && *tc.ThinkingBudget != 0
}

func buildThinkingParam(cfg *genai.GenerateContentConfig) anthropic.ThinkingConfigParamUnion {
	if cfg == nil || cfg.ThinkingConfig == nil || cfg.ThinkingConfig.ThinkingBudget == nil {
		return anthropic.ThinkingConfigParamUnion{}
	}
	budget := *cfg.ThinkingConfig.ThinkingBudget
	switch {
	case budget == 0:
		disabled := anthropic.NewThinkingConfigDisabledParam()
		return anthropic.ThinkingConfigParamUnion{OfDisabled: &disabled}
	case budget < 0:
		return anthropic.ThinkingConfigParamUnion{
			OfAdaptive: &anthropic.ThinkingConfigAdaptiveParam{
				Display: anthropic.ThinkingConfigAdaptiveDisplaySummarized,
			},
		}
	default:
		return anthropic.ThinkingConfigParamOfEnabled(int64(budget))
	}
}

func validateOutputFormatConfig(cfg *genai.GenerateContentConfig) error {
	if cfg == nil {
		return nil
	}
	if cfg.ResponseSchema != nil && cfg.ResponseJsonSchema != nil {
		return fmt.Errorf("%w: ResponseSchema and ResponseJsonSchema cannot both be set", ErrInvalidConfig)
	}
	mime := strings.ToLower(strings.TrimSpace(cfg.ResponseMIMEType))
	if mime != "" && mime != "text/plain" && mime != "application/json" {
		return fmt.Errorf("%w: ResponseMIMEType %q is not supported", ErrInvalidConfig, cfg.ResponseMIMEType)
	}
	if mime == "text/plain" && (cfg.ResponseSchema != nil || cfg.ResponseJsonSchema != nil) {
		return fmt.Errorf("%w: ResponseMIMEType %q cannot be combined with ResponseSchema or ResponseJsonSchema", ErrInvalidConfig, cfg.ResponseMIMEType)
	}
	if mime == "application/json" && cfg.ResponseSchema == nil && cfg.ResponseJsonSchema == nil {
		return fmt.Errorf("%w: ResponseMIMEType \"application/json\" requires ResponseSchema or ResponseJsonSchema", ErrInvalidConfig)
	}
	if cfg.ResponseSchema != nil || cfg.ResponseJsonSchema != nil {
		if _, _, err := buildOutputFormatSchema(cfg); err != nil {
			return err
		}
	}
	return nil
}

const wrappedOutputSchemaKey = "result"

func buildOutputFormatSchema(cfg *genai.GenerateContentConfig) (map[string]any, bool, error) {
	if cfg == nil || (cfg.ResponseSchema == nil && cfg.ResponseJsonSchema == nil) {
		return nil, false, nil
	}
	var (
		rawSchema any
		fieldName string
	)
	if cfg.ResponseJsonSchema != nil {
		rawSchema = cfg.ResponseJsonSchema
		fieldName = "ResponseJsonSchema"
	} else {
		rawSchema = cfg.ResponseSchema
		fieldName = "ResponseSchema"
	}

	data, err := json.Marshal(rawSchema)
	if err != nil {
		return nil, false, fmt.Errorf("%w: failed to marshal %s: %v", ErrInvalidConfig, fieldName, err)
	}
	var schemaMap map[string]any
	if err := json.Unmarshal(data, &schemaMap); err != nil || len(schemaMap) == 0 {
		return nil, false, fmt.Errorf("%w: %s must be a non-empty JSON schema", ErrInvalidConfig, fieldName)
	}
	lowercaseSchemaTypes(schemaMap)
	if _, hasType := schemaMap["type"]; !hasType {
		if _, hasProps := schemaMap["properties"]; hasProps {
			schemaMap["type"] = "object"
		}
	}
	normalizeOutputSchemaObjects(schemaMap)
	if rootType, _ := schemaMap["type"].(string); rootType == "object" {
		return schemaMap, false, nil
	}
	rewriteWrappedSchemaRefs(schemaMap)
	wrapped := map[string]any{
		"type": "object",
		"properties": map[string]any{
			wrappedOutputSchemaKey: schemaMap,
		},
		"required":             []string{wrappedOutputSchemaKey},
		"additionalProperties": false,
	}
	for _, defsKey := range []string{"$defs", "definitions"} {
		if defs, ok := schemaMap[defsKey]; ok {
			wrapped[defsKey] = defs
			delete(schemaMap, defsKey)
		}
	}
	return wrapped, true, nil
}

func rewriteWrappedSchemaRefs(node any) {
	switch v := node.(type) {
	case map[string]any:
		if ref, ok := v["$ref"].(string); ok {
			switch {
			case ref == "#":
				v["$ref"] = "#/properties/" + wrappedOutputSchemaKey
			case ref == "#/$defs" || strings.HasPrefix(ref, "#/$defs/") ||
				ref == "#/definitions" || strings.HasPrefix(ref, "#/definitions/"):
				// $defs and definitions are hoisted to the wrapper root.
			case strings.HasPrefix(ref, "#/"):
				v["$ref"] = "#/properties/" + wrappedOutputSchemaKey + ref[1:]
			}
		}
		for _, child := range v {
			rewriteWrappedSchemaRefs(child)
		}
	case []any:
		for _, elem := range v {
			rewriteWrappedSchemaRefs(elem)
		}
	}
}

func normalizeOutputSchemaObjects(node any) {
	switch v := node.(type) {
	case map[string]any:
		if t, _ := v["type"].(string); t == "object" {
			if _, ok := v["properties"]; !ok {
				v["properties"] = map[string]any{}
			}
			if _, ok := v["additionalProperties"]; !ok {
				v["additionalProperties"] = false
			}
		}
		for _, child := range v {
			normalizeOutputSchemaObjects(child)
		}
	case []any:
		for _, elem := range v {
			normalizeOutputSchemaObjects(elem)
		}
	}
}

func buildOutputConfigParam(effort Effort, cfg *genai.GenerateContentConfig) (anthropic.OutputConfigParam, bool, error) {
	var out anthropic.OutputConfigParam
	if effort != "" {
		out.Effort = anthropic.OutputConfigEffort(effort)
	}
	schemaMap, unwrap, err := buildOutputFormatSchema(cfg)
	if err != nil {
		return anthropic.OutputConfigParam{}, false, err
	}
	if schemaMap != nil {
		out.Format = anthropic.JSONOutputFormatParam{
			Schema: schemaMap,
		}
	}
	return out, unwrap, nil
}
