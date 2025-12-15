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

// FunctionDeclarationToTool converts a genai FunctionDeclaration to an Anthropic ToolUnionParam.
//
// If Parameters is set, it takes precedence over ParametersJsonSchema.
// ParametersJsonSchema currently supports:
//   - map[string]any with "properties" and "required" keys
//   - *jsonschema.Schema
//
// Other ParametersJsonSchema types are ignored.
func FunctionDeclarationToTool(fd *genai.FunctionDeclaration) anthropic.ToolUnionParam {
	inputSchema := anthropic.ToolInputSchemaParam{
		// Anthropic tools require an object schema at the root.
		// The SDK defaults to type "object" when not specified.
		Properties: map[string]any{},
	}

	// Convert parameters schema - Parameters takes precedence over ParametersJsonSchema
	if fd.Parameters != nil {
		props := schemaPropertiesToMap(fd.Parameters.Properties)
		if props != nil {
			inputSchema.Properties = props
		}
		if len(fd.Parameters.Required) > 0 {
			inputSchema.Required = fd.Parameters.Required
		}
	} else if fd.ParametersJsonSchema != nil {
		switch schema := fd.ParametersJsonSchema.(type) {
		case map[string]any:
			if props, ok := schema["properties"].(map[string]any); ok {
				inputSchema.Properties = props
			}
			inputSchema.Required = extractRequiredFields(schema["required"])
		case *jsonschema.Schema:
			if props := jsonSchemaToProperties(schema); props != nil {
				inputSchema.Properties = props
			}
			if len(schema.Required) > 0 {
				inputSchema.Required = schema.Required
			}
		}
	}

	return anthropic.ToolUnionParam{
		OfTool: &anthropic.ToolParam{
			Name:        fd.Name,
			Description: anthropic.String(fd.Description),
			InputSchema: inputSchema,
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
		result[name] = schemaToMap(schema)
	}
	return result
}

// schemaToMap converts a genai.Schema to a map[string]any suitable for Anthropic.
func schemaToMap(schema *genai.Schema) map[string]any {
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
		result["items"] = schemaToMap(schema.Items)
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
			if m := schemaToMap(s); m != nil {
				anyOf = append(anyOf, m)
			}
		}
		if len(anyOf) > 0 {
			result["anyOf"] = anyOf
		}
	}

	return result
}
