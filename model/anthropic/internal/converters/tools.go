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

package converters

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/google/jsonschema-go/jsonschema"
	"google.golang.org/genai"
)

// ToolsToAnthropicTools converts genai Tools to Anthropic ToolUnionParams.
func ToolsToAnthropicTools(tools []*genai.Tool) []anthropic.ToolUnionParam {
	if len(tools) == 0 {
		return nil
	}

	var result []anthropic.ToolUnionParam
	for _, tool := range tools {
		if tool == nil || len(tool.FunctionDeclarations) == 0 {
			continue
		}
		for _, fd := range tool.FunctionDeclarations {
			if fd == nil {
				continue
			}
			toolParam := FunctionDeclarationToTool(fd)
			result = append(result, toolParam)
		}
	}
	return result
}

// extractFunctionParams extracts properties and required fields from a FunctionDeclaration.
// Parameters takes precedence over ParametersJsonSchema.
// ParametersJsonSchema currently supports:
//   - map[string]any with "properties" and "required" keys
//   - *jsonschema.Schema
//
// Other ParametersJsonSchema types are ignored.
func extractFunctionParams(fd *genai.FunctionDeclaration) (properties map[string]any, required []string) {
	properties = map[string]any{}

	if fd.Parameters != nil {
		if props := schemaPropertiesToMap(fd.Parameters.Properties); props != nil {
			properties = props
		}
		required = fd.Parameters.Required
	} else if fd.ParametersJsonSchema != nil {
		switch schema := fd.ParametersJsonSchema.(type) {
		case map[string]any:
			if props, ok := schema["properties"].(map[string]any); ok {
				properties = props
			}
			required = extractRequiredFields(schema["required"])
		case *jsonschema.Schema:
			if props := jsonSchemaToProperties(schema); props != nil {
				properties = props
			}
			if len(schema.Required) > 0 {
				required = schema.Required
			}
		}
	}

	return properties, required
}

// FunctionDeclarationToTool converts a genai FunctionDeclaration to an Anthropic ToolUnionParam.
func FunctionDeclarationToTool(fd *genai.FunctionDeclaration) anthropic.ToolUnionParam {
	properties, required := extractFunctionParams(fd)

	return anthropic.ToolUnionParam{
		OfTool: &anthropic.ToolParam{
			Name:        fd.Name,
			Description: anthropic.String(fd.Description),
			InputSchema: anthropic.ToolInputSchemaParam{
				Properties: properties,
				Required:   required,
			},
		},
	}
}

// extractRequiredFields extracts required field names from various input types.
// Supports []any (from JSON unmarshalling) and []string (from manual construction).
func extractRequiredFields(v any) []string {
	if v == nil {
		return nil
	}
	switch req := v.(type) {
	case []string:
		return req
	case []any:
		result := make([]string, 0, len(req))
		for _, r := range req {
			if s, ok := r.(string); ok {
				result = append(result, s)
			}
		}
		return result
	default:
		return nil
	}
}

// jsonSchemaToProperties converts a jsonschema.Schema to a properties map.
// Returns nil if schema or its properties are nil, consistent with schemaPropertiesToMap.
func jsonSchemaToProperties(schema *jsonschema.Schema) map[string]any {
	if schema == nil || schema.Properties == nil {
		return nil
	}

	props := make(map[string]any)
	for name, propSchema := range schema.Properties {
		props[name] = jsonSchemaPropertyToMap(propSchema)
	}
	return props
}

// jsonSchemaPropertyToMap converts a single jsonschema.Schema property to a map.
func jsonSchemaPropertyToMap(schema *jsonschema.Schema) map[string]any {
	if schema == nil {
		return nil
	}

	result := make(map[string]any)

	if schema.Type != "" {
		result["type"] = string(schema.Type)
	}
	if schema.Description != "" {
		result["description"] = schema.Description
	}
	if len(schema.Enum) > 0 {
		result["enum"] = schema.Enum
	}
	if schema.Items != nil {
		result["items"] = jsonSchemaPropertyToMap(schema.Items)
	}
	if schema.Properties != nil {
		result["properties"] = jsonSchemaToProperties(schema)
	}
	if len(schema.Required) > 0 {
		result["required"] = schema.Required
	}

	return result
}

// schemaPropertiesToMap converts genai Schema properties to a map for Anthropic.
func schemaPropertiesToMap(props map[string]*genai.Schema) map[string]any {
	if props == nil {
		return nil
	}

	result := make(map[string]any)
	for name, schema := range props {
		if schema == nil {
			continue
		}
		result[name] = SchemaToMap(schema)
	}
	return result
}

// SchemaToMap converts a genai.Schema to a map[string]any suitable for Anthropic.
func SchemaToMap(schema *genai.Schema) map[string]any {
	if schema == nil {
		return nil
	}

	result := make(map[string]any)

	// Type
	if schema.Type != "" {
		result["type"] = strings.ToLower(string(schema.Type))
	}

	// Description
	if schema.Description != "" {
		result["description"] = schema.Description
	}

	// Enum
	if len(schema.Enum) > 0 {
		result["enum"] = schema.Enum
	}

	// Format
	if schema.Format != "" {
		result["format"] = schema.Format
	}

	// Items (for arrays)
	if schema.Items != nil {
		result["items"] = SchemaToMap(schema.Items)
	}

	// Properties (for objects)
	if len(schema.Properties) > 0 {
		result["properties"] = schemaPropertiesToMap(schema.Properties)
	}

	// Required
	if len(schema.Required) > 0 {
		result["required"] = schema.Required
	}

	// Nullable
	if schema.Nullable != nil && *schema.Nullable {
		result["nullable"] = true
	}

	// Default
	if schema.Default != nil {
		result["default"] = schema.Default
	}

	// Min/Max constraints
	if schema.Minimum != nil {
		result["minimum"] = *schema.Minimum
	}
	if schema.Maximum != nil {
		result["maximum"] = *schema.Maximum
	}
	if schema.MinLength != nil {
		result["minLength"] = *schema.MinLength
	}
	if schema.MaxLength != nil {
		result["maxLength"] = *schema.MaxLength
	}
	if schema.MinItems != nil {
		result["minItems"] = *schema.MinItems
	}
	if schema.MaxItems != nil {
		result["maxItems"] = *schema.MaxItems
	}

	// Pattern
	if schema.Pattern != "" {
		result["pattern"] = schema.Pattern
	}

	// AnyOf
	if len(schema.AnyOf) > 0 {
		anyOf := make([]map[string]any, 0, len(schema.AnyOf))
		for _, s := range schema.AnyOf {
			if m := SchemaToMap(s); m != nil {
				anyOf = append(anyOf, m)
			}
		}
		if len(anyOf) > 0 {
			result["anyOf"] = anyOf
		}
	}

	return result
}

// SchemaToJSONString converts a genai.Schema to a pretty-printed JSON string.
// Used for prompt-based JSON output fallback on Vertex AI.
func SchemaToJSONString(schema *genai.Schema) string {
	m := SchemaToMap(schema)
	b, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		return "{}"
	}
	return string(b)
}

// toolChoiceKind represents the resolved tool choice type.
type toolChoiceKind int

const (
	toolChoiceNone toolChoiceKind = iota // omit tool_choice
	toolChoiceAuto
	toolChoiceAny
	toolChoiceTool
)

// resolvedToolChoice holds the result of resolving a ToolConfig into a tool choice decision.
type resolvedToolChoice struct {
	kind     toolChoiceKind
	toolName string // populated when kind == toolChoiceTool
}

// resolveToolChoice extracts the tool choice decision from a ToolConfig.
// Returns an error for unsupported configurations (multiple AllowedFunctionNames,
// unknown FunctionCallingConfig modes).
func resolveToolChoice(config *genai.ToolConfig) (resolvedToolChoice, error) {
	if config == nil || config.FunctionCallingConfig == nil {
		return resolvedToolChoice{kind: toolChoiceNone}, nil
	}

	fcc := config.FunctionCallingConfig

	if len(fcc.AllowedFunctionNames) > 1 {
		return resolvedToolChoice{}, fmt.Errorf(
			"Anthropic does not support multiple AllowedFunctionNames (got %d); use a single function name or remove the restriction",
			len(fcc.AllowedFunctionNames),
		)
	}

	switch fcc.Mode {
	case genai.FunctionCallingConfigModeNone:
		return resolvedToolChoice{kind: toolChoiceNone}, nil

	case genai.FunctionCallingConfigModeAuto:
		return resolvedToolChoice{kind: toolChoiceAuto}, nil

	case genai.FunctionCallingConfigModeAny:
		if len(fcc.AllowedFunctionNames) == 1 {
			return resolvedToolChoice{kind: toolChoiceTool, toolName: fcc.AllowedFunctionNames[0]}, nil
		}
		return resolvedToolChoice{kind: toolChoiceAny}, nil

	default:
		return resolvedToolChoice{}, fmt.Errorf(
			"unsupported FunctionCallingConfig mode %q; supported modes are: ModeNone, ModeAuto, ModeAny",
			fcc.Mode,
		)
	}
}

// ToolConfigToToolChoice converts a genai.ToolConfig to Anthropic's tool_choice parameter.
// Returns a zero-value union param when no tool_choice should be set (nil config, ModeNone),
// which is safe to assign unconditionally as the SDK omits it during serialization.
//
// Mapping:
//   - ModeNone -> zero value (omitted)
//   - ModeAuto -> "auto" (model decides whether to use tools)
//   - ModeAny -> "any" (model must use a tool)
//   - ModeAny + single AllowedFunctionNames -> "tool" with specific name
//
// Returns an error if AllowedFunctionNames contains more than one function name,
// or if the FunctionCallingConfig mode is not recognized.
func ToolConfigToToolChoice(config *genai.ToolConfig) (anthropic.ToolChoiceUnionParam, error) {
	resolved, err := resolveToolChoice(config)
	if err != nil {
		return anthropic.ToolChoiceUnionParam{}, err
	}

	switch resolved.kind {
	case toolChoiceNone:
		return anthropic.ToolChoiceUnionParam{}, nil
	case toolChoiceAuto:
		return anthropic.ToolChoiceUnionParam{
			OfAuto: &anthropic.ToolChoiceAutoParam{},
		}, nil
	case toolChoiceAny:
		return anthropic.ToolChoiceUnionParam{
			OfAny: &anthropic.ToolChoiceAnyParam{},
		}, nil
	case toolChoiceTool:
		return anthropic.ToolChoiceUnionParam{
			OfTool: &anthropic.ToolChoiceToolParam{
				Name: resolved.toolName,
			},
		}, nil
	default:
		return anthropic.ToolChoiceUnionParam{}, fmt.Errorf("unexpected tool choice kind: %d", resolved.kind)
	}
}
