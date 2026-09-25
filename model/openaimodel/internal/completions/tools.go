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

package completions

import (
	"fmt"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/shared"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/model/openaimodel/internal/openaicommon"
)

// convertTools converts function declarations into Chat Completions tools,
// whose declaration nests under a "function" object that the flat Responses
// shape does not have.
func convertTools(cfg *genai.GenerateContentConfig) ([]openai.ChatCompletionToolUnionParam, error) {
	if cfg == nil || len(cfg.Tools) == 0 {
		return nil, nil
	}
	var tools []openai.ChatCompletionToolUnionParam
	for i, tool := range cfg.Tools {
		if err := openaicommon.EnsureFunctionToolOnly(i, tool); err != nil {
			return nil, err
		}
		for _, decl := range tool.FunctionDeclarations {
			fn, err := convertFunctionDeclaration(decl)
			if err != nil {
				return nil, err
			}
			tools = append(tools, openai.ChatCompletionToolUnionParam{OfFunction: fn})
		}
	}
	return tools, nil
}

// convertFunctionDeclaration converts one function declaration into a Chat
// Completions function tool.
func convertFunctionDeclaration(fn *genai.FunctionDeclaration) (*openai.ChatCompletionFunctionToolParam, error) {
	if fn == nil {
		return nil, fmt.Errorf("openai: nil function declaration")
	}
	if fn.Name == "" {
		return nil, fmt.Errorf("openai: function declaration missing name")
	}

	paramsMap, err := openaicommon.SchemaToMap(fn.Parameters)
	if err != nil {
		return nil, err
	}
	if paramsMap == nil && fn.ParametersJsonSchema != nil {
		paramsMap, err = openaicommon.NormalizeSchema(fn.ParametersJsonSchema)
		if err != nil {
			return nil, err
		}
	}
	if paramsMap == nil {
		paramsMap = map[string]any{
			"type":       "object",
			"properties": map[string]any{},
		}
	}

	def := shared.FunctionDefinitionParam{
		Name:       fn.Name,
		Parameters: paramsMap,
	}
	if fn.Description != "" {
		def.Description = param.NewOpt(fn.Description)
	}
	return &openai.ChatCompletionFunctionToolParam{Function: def}, nil
}

// convertToolChoice translates a tool config into the Chat Completions
// tool_choice, which names a function one level deeper than Responses does.
func convertToolChoice(toolCfg *genai.ToolConfig) (*openai.ChatCompletionToolChoiceOptionUnionParam, error) {
	if toolCfg == nil || toolCfg.FunctionCallingConfig == nil {
		return nil, nil
	}
	cfg := toolCfg.FunctionCallingConfig
	choice := &openai.ChatCompletionToolChoiceOptionUnionParam{}
	switch cfg.Mode {
	case "", genai.FunctionCallingConfigModeUnspecified, genai.FunctionCallingConfigModeAuto:
		if len(cfg.AllowedFunctionNames) == 0 {
			// Nothing named, so the provider's own default stands.
			return nil, nil
		}
		choice.OfAllowedTools = allowedToolParam(cfg.AllowedFunctionNames, openai.ChatCompletionAllowedToolsModeAuto)
	case genai.FunctionCallingConfigModeNone:
		choice.OfAuto = param.NewOpt("none")
	case genai.FunctionCallingConfigModeAny:
		if len(cfg.AllowedFunctionNames) == 0 {
			choice.OfAuto = param.NewOpt("required")
		} else if name, ok := soleFunctionName(cfg.AllowedFunctionNames); ok {
			// Naming the one function says the same as allowed_tools, in the
			// older form that compatible providers accept and many that do not
			// know allowed_tools reject.
			choice.OfFunctionToolChoice = &openai.ChatCompletionNamedToolChoiceParam{
				Function: openai.ChatCompletionNamedToolChoiceFunctionParam{Name: name},
			}
		} else {
			choice.OfAllowedTools = allowedToolParam(cfg.AllowedFunctionNames, openai.ChatCompletionAllowedToolsModeRequired)
		}
	default:
		return nil, fmt.Errorf("openai: unsupported tool calling mode %q", cfg.Mode)
	}

	if !param.IsOmitted(choice.OfAuto) || choice.OfAllowedTools != nil || choice.OfFunctionToolChoice != nil {
		return choice, nil
	}
	return nil, nil
}

// soleFunctionName reports the one function names allows, skipping the empty
// entries allowedToolParam also skips, and false for none or several.
func soleFunctionName(names []string) (string, bool) {
	sole := ""
	for _, name := range names {
		switch {
		case name == "":
		case sole != "":
			return "", false
		default:
			sole = name
		}
	}
	return sole, sole != ""
}

// allowedToolParam builds the allowed-tools choice. Each entry nests the
// name under "function", unlike the flat Responses form.
func allowedToolParam(names []string, mode openai.ChatCompletionAllowedToolsMode) *openai.ChatCompletionAllowedToolChoiceParam {
	tools := make([]map[string]any, 0, len(names))
	for _, name := range names {
		if name == "" {
			continue
		}
		tools = append(tools, map[string]any{
			"type":     "function",
			"function": map[string]any{"name": name},
		})
	}
	if len(tools) == 0 {
		return nil
	}
	return &openai.ChatCompletionAllowedToolChoiceParam{
		AllowedTools: openai.ChatCompletionAllowedToolsParam{Mode: mode, Tools: tools},
	}
}
