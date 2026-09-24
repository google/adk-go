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

package responses

import (
	"fmt"

	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared/constant"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/model/openaimodel/internal/openaicommon"
)

// convertTools takes our generic tool definitions and converts them into
// OpenAI's specific tool format. We ensure that only function tools are
// supported and properly declared.
func convertTools(cfg *genai.GenerateContentConfig) ([]responses.ToolUnionParam, error) {
	if cfg == nil || len(cfg.Tools) == 0 {
		return nil, nil
	}
	var tools []responses.ToolUnionParam
	for i, tool := range cfg.Tools {
		if err := openaicommon.EnsureFunctionToolOnly(i, tool); err != nil {
			return nil, err
		}
		for _, decl := range tool.FunctionDeclarations {
			fn, err := convertFunctionDeclaration(decl)
			if err != nil {
				return nil, err
			}
			tools = append(tools, responses.ToolUnionParam{OfFunction: fn})
		}
	}
	return tools, nil
}

// convertFunctionDeclaration takes a generic genai.FunctionDeclaration and
// converts it into an OpenAI-specific responses.FunctionToolParam. We handle
// the function's name, description, and importantly, convert its parameters
// from a generic schema format to a map[string]any that the OpenAI API expects.
//
// Strict is pinned off rather than left unset: an absent flag lets the API pick
// the mode from the schema shape and add the caller's optional arguments to
// required. Pinning it on would do that permanently, so off is the only value
// that keeps an optional argument optional; adk-python's Responses path agrees.
func convertFunctionDeclaration(fn *genai.FunctionDeclaration) (*responses.FunctionToolParam, error) {
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
		// If no parameters are defined, we default to an empty object schema.
		paramsMap = map[string]any{
			"type":       "object",
			"properties": map[string]any{},
		}
	}

	fnParam := &responses.FunctionToolParam{
		Name:       fn.Name,
		Type:       constant.Function("function"),
		Parameters: paramsMap,
		Strict:     param.NewOpt(false),
	}
	if fn.Description != "" {
		fnParam.Description = param.NewOpt(fn.Description)
	}
	return fnParam, nil
}

// convertToolChoice takes our generic genai.ToolConfig and translates its
// FunctionCallingConfig into an OpenAI-specific tool choice parameter.
// We handle different function calling modes (Auto, None, Any) and
// incorporate any allowed function names into the appropriate OpenAI format.
func convertToolChoice(toolCfg *genai.ToolConfig) (*responses.ResponseNewParamsToolChoiceUnion, error) {
	if toolCfg == nil || toolCfg.FunctionCallingConfig == nil {
		return nil, nil
	}
	cfg := toolCfg.FunctionCallingConfig
	choice := &responses.ResponseNewParamsToolChoiceUnion{}
	switch cfg.Mode {
	case "", genai.FunctionCallingConfigModeUnspecified, genai.FunctionCallingConfigModeAuto:
		if len(cfg.AllowedFunctionNames) == 0 {
			// If no specific functions are allowed, we don't set a tool choice,
			// letting OpenAI decide (which is effectively 'auto').
			return nil, nil
		}
		// If specific functions are allowed in auto mode, we specify them.
		choice.OfAllowedTools = allowedToolParam(cfg.AllowedFunctionNames, responses.ToolChoiceAllowedModeAuto)
	case genai.FunctionCallingConfigModeNone:
		// Explicitly disable tool calling.
		choice.OfToolChoiceMode = param.NewOpt(responses.ToolChoiceOptionsNone)
	case genai.FunctionCallingConfigModeAny:
		if len(cfg.AllowedFunctionNames) == 0 {
			// If 'any' is specified without allowed names, it means the model
			// can call any tool.
			choice.OfToolChoiceMode = param.NewOpt(responses.ToolChoiceOptionsRequired)
		} else {
			// If 'any' is specified with allowed names, the model must call
			// one of the allowed tools.
			choice.OfAllowedTools = allowedToolParam(cfg.AllowedFunctionNames, responses.ToolChoiceAllowedModeRequired)
		}
	default:
		return nil, fmt.Errorf("openai: unsupported tool calling mode %q", cfg.Mode)
	}

	if !param.IsOmitted(choice.OfToolChoiceMode) || choice.OfAllowedTools != nil {
		return choice, nil
	}
	return nil, nil
}

func allowedToolParam(names []string, mode responses.ToolChoiceAllowedMode) *responses.ToolChoiceAllowedParam {
	tools := make([]map[string]any, 0, len(names))
	for _, name := range names {
		if name == "" {
			continue
		}
		tools = append(tools, map[string]any{
			"type": "function",
			"name": name,
		})
	}
	if len(tools) == 0 {
		return nil
	}
	return &responses.ToolChoiceAllowedParam{
		Mode:  mode,
		Type:  constant.AllowedTools("allowed_tools"),
		Tools: tools,
	}
}
