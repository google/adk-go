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
	"regexp"
	"slices"
	"strings"

	"github.com/anthropics/anthropic-sdk-go"
	"google.golang.org/genai"
)

var validToolUseIDPattern = regexp.MustCompile(`^[a-zA-Z0-9_-]+$`)

// toolUseIDSanitizer maps invalid or empty tool_use IDs to deterministic
// fallback IDs (`toolu_fallback_<n>`), pairing FunctionCall and FunctionResponse
// parts across a conversation in the same order as adk-python.
type toolUseIDSanitizer struct {
	mapping             map[string]string
	unpairedCallsByName map[string][]string
	unpairedCallsList   []string
	nextFallback        int
}

func newToolUseIDSanitizer() *toolUseIDSanitizer {
	return &toolUseIDSanitizer{
		mapping:             make(map[string]string),
		unpairedCallsByName: make(map[string][]string),
	}
}

func (s *toolUseIDSanitizer) sanitize(toolID, toolName string, isCall bool) string {
	if toolID != "" && validToolUseIDPattern.MatchString(toolID) {
		return toolID
	}
	if toolID != "" {
		if mapped, ok := s.mapping[toolID]; ok {
			return mapped
		}
	}

	if isCall || toolID != "" {
		assignedID := fmt.Sprintf("toolu_fallback_%d", s.nextFallback)
		s.nextFallback++
		if toolID != "" {
			s.mapping[toolID] = assignedID
		} else {
			if toolName != "" {
				s.unpairedCallsByName[toolName] = append(s.unpairedCallsByName[toolName], assignedID)
			}
			s.unpairedCallsList = append(s.unpairedCallsList, assignedID)
		}
		return assignedID
	}

	// Response with empty toolID: pair with the oldest pending call for the same tool name,
	// or the oldest pending call overall.
	if toolName != "" && len(s.unpairedCallsByName[toolName]) > 0 {
		assignedID := s.unpairedCallsByName[toolName][0]
		s.unpairedCallsByName[toolName] = s.unpairedCallsByName[toolName][1:]
		if idx := slices.Index(s.unpairedCallsList, assignedID); idx >= 0 {
			s.unpairedCallsList = slices.Delete(s.unpairedCallsList, idx, idx+1)
		}
		return assignedID
	}
	if len(s.unpairedCallsList) > 0 {
		assignedID := s.unpairedCallsList[0]
		s.unpairedCallsList = s.unpairedCallsList[1:]
		for name, ids := range s.unpairedCallsByName {
			if idx := slices.Index(ids, assignedID); idx >= 0 {
				s.unpairedCallsByName[name] = slices.Delete(ids, idx, idx+1)
				break
			}
		}
		return assignedID
	}

	assignedID := fmt.Sprintf("toolu_fallback_%d", s.nextFallback)
	s.nextFallback++
	return assignedID
}

// validateTools ensures every genai.Tool only contains FunctionDeclarations.
func validateTools(tools []*genai.Tool) (bool, error) {
	hasDecls := false
	for _, t := range tools {
		if t == nil {
			continue
		}
		if t.GoogleSearch != nil {
			return false, fmt.Errorf("%w: %w: Tools.GoogleSearch is not supported by Claude", ErrInvalidConfig, ErrUnsupportedToolType)
		}
		if t.GoogleSearchRetrieval != nil {
			return false, fmt.Errorf("%w: %w: Tools.GoogleSearchRetrieval is not supported by Claude", ErrInvalidConfig, ErrUnsupportedToolType)
		}
		if t.Retrieval != nil {
			return false, fmt.Errorf("%w: %w: Tools.Retrieval is not supported by Claude", ErrInvalidConfig, ErrUnsupportedToolType)
		}
		if t.CodeExecution != nil {
			return false, fmt.Errorf("%w: %w: Tools.CodeExecution is not supported by Claude", ErrInvalidConfig, ErrUnsupportedToolType)
		}
		if t.EnterpriseWebSearch != nil {
			return false, fmt.Errorf("%w: %w: Tools.EnterpriseWebSearch is not supported by Claude", ErrInvalidConfig, ErrUnsupportedToolType)
		}
		if t.GoogleMaps != nil {
			return false, fmt.Errorf("%w: %w: Tools.GoogleMaps is not supported by Claude", ErrInvalidConfig, ErrUnsupportedToolType)
		}
		if t.ComputerUse != nil {
			return false, fmt.Errorf("%w: %w: Tools.ComputerUse is not supported by Claude", ErrInvalidConfig, ErrUnsupportedToolType)
		}
		if t.URLContext != nil {
			return false, fmt.Errorf("%w: %w: Tools.URLContext is not supported by Claude", ErrInvalidConfig, ErrUnsupportedToolType)
		}
		if t.FileSearch != nil {
			return false, fmt.Errorf("%w: %w: Tools.FileSearch is not supported by Claude", ErrInvalidConfig, ErrUnsupportedToolType)
		}
		if t.ParallelAISearch != nil {
			return false, fmt.Errorf("%w: %w: Tools.ParallelAISearch is not supported by Claude", ErrInvalidConfig, ErrUnsupportedToolType)
		}
		if len(t.MCPServers) > 0 {
			return false, fmt.Errorf("%w: %w: Tools.MCPServers is not supported by Claude", ErrInvalidConfig, ErrUnsupportedToolType)
		}
		if t.ExaAISearch != nil {
			return false, fmt.Errorf("%w: %w: Tools.ExaAISearch is not supported by Claude", ErrInvalidConfig, ErrUnsupportedToolType)
		}
		if len(t.FunctionDeclarations) == 0 {
			return false, fmt.Errorf("%w: %w: Tools entry contains no FunctionDeclarations", ErrInvalidConfig, ErrUnsupportedToolType)
		}
		for _, decl := range t.FunctionDeclarations {
			if decl == nil || strings.TrimSpace(decl.Name) == "" {
				return false, fmt.Errorf("%w: Tools.FunctionDeclarations entry has an empty Name", ErrInvalidConfig)
			}
			hasDecls = true
		}
	}
	return hasDecls, nil
}

func validateToolConfig(tc *genai.ToolConfig, hasTools bool, thinkingCfg *genai.ThinkingConfig) error {
	if tc == nil {
		return nil
	}
	if tc.RetrievalConfig != nil {
		return fmt.Errorf("%w: ToolConfig.RetrievalConfig is not supported", ErrInvalidConfig)
	}
	if tc.IncludeServerSideToolInvocations != nil && *tc.IncludeServerSideToolInvocations {
		return fmt.Errorf("%w: ToolConfig.IncludeServerSideToolInvocations is not supported", ErrInvalidConfig)
	}
	if tc.FunctionCallingConfig == nil {
		return nil
	}
	fcc := tc.FunctionCallingConfig
	if fcc.StreamFunctionCallArguments != nil && *fcc.StreamFunctionCallArguments {
		return fmt.Errorf("%w: ToolConfig.FunctionCallingConfig.StreamFunctionCallArguments is not supported", ErrInvalidConfig)
	}

	mode := strings.ToUpper(strings.TrimSpace(string(fcc.Mode)))
	switch genai.FunctionCallingConfigMode(mode) {
	case "", genai.FunctionCallingConfigModeAuto:
		if len(fcc.AllowedFunctionNames) > 0 {
			return fmt.Errorf("%w: ToolConfig.FunctionCallingConfig.AllowedFunctionNames is not supported with mode AUTO", ErrInvalidConfig)
		}
	case genai.FunctionCallingConfigModeAny:
		if !hasTools {
			return fmt.Errorf("%w: ToolConfig.FunctionCallingConfig.Mode ANY requires at least one declared function tool", ErrInvalidConfig)
		}
		if isThinkingEnabled(thinkingCfg) {
			return fmt.Errorf("%w: ToolConfig.FunctionCallingConfig.Mode ANY cannot be combined with enabled ThinkingConfig on Claude", ErrInvalidConfig)
		}
		if len(fcc.AllowedFunctionNames) > 1 {
			return fmt.Errorf("%w: ToolConfig.FunctionCallingConfig.AllowedFunctionNames with multiple names is not supported by Claude", ErrInvalidConfig)
		}
		if len(fcc.AllowedFunctionNames) == 1 && strings.TrimSpace(fcc.AllowedFunctionNames[0]) == "" {
			return fmt.Errorf("%w: ToolConfig.FunctionCallingConfig.AllowedFunctionNames must not contain an empty name", ErrInvalidConfig)
		}
	case genai.FunctionCallingConfigModeNone:
		if len(fcc.AllowedFunctionNames) > 0 {
			return fmt.Errorf("%w: ToolConfig.FunctionCallingConfig.AllowedFunctionNames is not supported with mode NONE", ErrInvalidConfig)
		}
	default:
		return fmt.Errorf("%w: unsupported ToolConfig.FunctionCallingConfig.Mode %q", ErrInvalidConfig, fcc.Mode)
	}
	return nil
}

func buildToolsAndChoice(cfg *genai.GenerateContentConfig, reqTools map[string]any) ([]anthropic.ToolUnionParam, anthropic.ToolChoiceUnionParam, error) {
	var tools []anthropic.ToolUnionParam
	if cfg != nil {
		for _, t := range cfg.Tools {
			if t == nil {
				continue
			}
			for _, decl := range t.FunctionDeclarations {
				if decl == nil {
					continue
				}
				toolParam, err := convertFunctionDeclaration(decl)
				if err != nil {
					return nil, anthropic.ToolChoiceUnionParam{}, err
				}
				tools = append(tools, anthropic.ToolUnionParam{OfTool: &toolParam})
			}
		}
	}

	var choice anthropic.ToolChoiceUnionParam
	hasTools := len(tools) > 0 || len(reqTools) > 0

	if cfg != nil && cfg.ToolConfig != nil && cfg.ToolConfig.FunctionCallingConfig != nil {
		fcc := cfg.ToolConfig.FunctionCallingConfig
		mode := genai.FunctionCallingConfigMode(strings.ToUpper(strings.TrimSpace(string(fcc.Mode))))
		switch mode {
		case genai.FunctionCallingConfigModeNone:
			if len(tools) > 0 {
				noneParam := anthropic.NewToolChoiceNoneParam()
				choice = anthropic.ToolChoiceUnionParam{OfNone: &noneParam}
			}
		case genai.FunctionCallingConfigModeAny:
			if len(fcc.AllowedFunctionNames) == 1 {
				choice = anthropic.ToolChoiceParamOfTool(strings.TrimSpace(fcc.AllowedFunctionNames[0]))
			} else {
				choice = anthropic.ToolChoiceUnionParam{OfAny: &anthropic.ToolChoiceAnyParam{}}
			}
		default:
			if hasTools && len(tools) > 0 {
				choice = anthropic.ToolChoiceUnionParam{OfAuto: &anthropic.ToolChoiceAutoParam{}}
			}
		}
	} else if hasTools && len(tools) > 0 {
		choice = anthropic.ToolChoiceUnionParam{OfAuto: &anthropic.ToolChoiceAutoParam{}}
	}

	return tools, choice, nil
}

func convertFunctionDeclaration(decl *genai.FunctionDeclaration) (anthropic.ToolParam, error) {
	inputSchema, err := buildToolInputSchema(decl)
	if err != nil {
		return anthropic.ToolParam{}, err
	}
	toolParam := anthropic.ToolParam{
		Name:        strings.TrimSpace(decl.Name),
		InputSchema: inputSchema,
	}
	if decl.Description != "" {
		toolParam.Description = anthropic.String(decl.Description)
	}
	return toolParam, nil
}

func buildToolInputSchema(decl *genai.FunctionDeclaration) (anthropic.ToolInputSchemaParam, error) {
	var rawSchema any
	switch {
	case decl.ParametersJsonSchema != nil:
		rawSchema = decl.ParametersJsonSchema
	case decl.Parameters != nil:
		rawSchema = decl.Parameters
	default:
		return anthropic.ToolInputSchemaParam{
			Properties: map[string]any{},
		}, nil
	}

	data, err := json.Marshal(rawSchema)
	if err != nil {
		return anthropic.ToolInputSchemaParam{}, fmt.Errorf("%w: failed to marshal tool schema for %q", ErrInvalidConfig, decl.Name)
	}

	var schemaMap map[string]any
	if err := json.Unmarshal(data, &schemaMap); err != nil || schemaMap == nil {
		return anthropic.ToolInputSchemaParam{}, fmt.Errorf("%w: tool schema for %q must be a JSON object", ErrInvalidConfig, decl.Name)
	}

	lowercaseSchemaTypes(schemaMap)

	var out anthropic.ToolInputSchemaParam
	if props, ok := schemaMap["properties"]; ok && props != nil {
		out.Properties = props
	} else {
		out.Properties = map[string]any{}
	}

	if reqRaw, ok := schemaMap["required"]; ok {
		switch reqTyped := reqRaw.(type) {
		case []string:
			out.Required = slices.Clone(reqTyped)
		case []any:
			for _, item := range reqTyped {
				if s, ok := item.(string); ok {
					out.Required = append(out.Required, s)
				}
			}
		}
	}

	for k, v := range schemaMap {
		if k == "type" || k == "properties" || k == "required" {
			continue
		}
		if out.ExtraFields == nil {
			out.ExtraFields = make(map[string]any)
		}
		out.ExtraFields[k] = v
	}

	return out, nil
}

// lowercaseSchemaTypes recursively normalizes JSON schema "type" values to
// lowercase (matching adk-python's lowercase_schema_types).
func lowercaseSchemaTypes(node any) {
	switch v := node.(type) {
	case map[string]any:
		if t, ok := v["type"]; ok {
			switch tv := t.(type) {
			case string:
				v["type"] = strings.ToLower(tv)
			case []any:
				for i, elem := range tv {
					if s, ok := elem.(string); ok {
						tv[i] = strings.ToLower(s)
					}
				}
			}
		}
		for _, child := range v {
			lowercaseSchemaTypes(child)
		}
	case []any:
		for _, elem := range v {
			lowercaseSchemaTypes(elem)
		}
	}
}
