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

// Package anthropic implements the model.LLM interface backed by Claude models
// served via Vertex AI.

package anthropic

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/packages/param"
	"github.com/anthropics/anthropic-sdk-go/shared/constant"

	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

// RequestBuilder helps build the request for the Anthropic API.
type RequestBuilder struct {
	modelName string
	maxTokens int64
}

func (builder *RequestBuilder) FromLLMRequest(req *model.LLMRequest) (*anthropic.MessageNewParams, error) {
	messages := make([]anthropic.MessageParam, 0, len(req.Contents))
	for _, content := range req.Contents {
		message, err := builder.buildMessageFromContent(content)
		if err != nil {
			return nil, err
		}
		messages = append(messages, message)
	}

	systemInstruction := builder.buildSystemInstruction(req.Config.SystemInstruction)
	params := &anthropic.MessageNewParams{
		Messages:  messages,
		System:    systemInstruction,
		Model:     anthropic.Model(builder.modelName),
		MaxTokens: builder.maxTokens,
	}
	if err := builder.appendConfigOptions(params, req.Config); err != nil {
		return nil, err
	}
	return params, nil
}

func (builder *RequestBuilder) buildMessageFromContent(content *genai.Content) (anthropic.MessageParam, error) {
	blocks := make([]anthropic.ContentBlockParamUnion, 0, len(content.Parts))
	for _, part := range content.Parts {
		block, err := builder.buildContentBlockFromPart(part)
		if err != nil {
			return anthropic.MessageParam{}, err
		}
		blocks = append(blocks, block)
	}

	role := anthropic.MessageParamRoleUser
	if content.Role == "model" || content.Role == "assistant" {
		role = anthropic.MessageParamRoleAssistant
	}

	return anthropic.MessageParam{
		Role:    role,
		Content: blocks,
	}, nil
}

func (builder *RequestBuilder) buildContentBlockFromPart(part *genai.Part) (anthropic.ContentBlockParamUnion, error) {
	switch {
	case part.Text != "":
		return anthropic.NewTextBlock(part.Text), nil
	case part.FunctionCall != nil:
		call := part.FunctionCall
		return anthropic.NewToolUseBlock(call.ID, call.Args, call.Name), nil
	case part.FunctionResponse != nil:
		funcResponse := part.FunctionResponse
		content := stringifyFunctionResponse(funcResponse.Response)
		return anthropic.NewToolResultBlock(funcResponse.ID, content, false), nil
	case isImagePart(part):
		encodingData := base64.StdEncoding.EncodeToString(part.InlineData.Data)
		return anthropic.NewImageBlockBase64(part.InlineData.MIMEType, encodingData), nil
	case part.ExecutableCode != nil:
		code := fmt.Sprintf("Code:```%s\n%s\n```", part.ExecutableCode.Language, part.ExecutableCode.Code)
		return anthropic.NewTextBlock(code), nil
	case part.CodeExecutionResult != nil:
		output := fmt.Sprintf("Execution Result:```code_output\n%s\n```", part.CodeExecutionResult.Output)
		return anthropic.NewTextBlock(output), nil
	default:
		return anthropic.ContentBlockParamUnion{}, fmt.Errorf("unsupported part type %+v", part)
	}
}

func (builder *RequestBuilder) buildSystemInstruction(content *genai.Content) []anthropic.TextBlockParam {
	if content == nil {
		return nil
	}

	var blocks []anthropic.TextBlockParam
	for _, part := range content.Parts {
		if part == nil || part.Text == "" {
			continue
		}
		blocks = append(blocks, anthropic.TextBlockParam{
			Text: part.Text,
			Type: constant.ValueOf[constant.Text](),
		})
	}
	return blocks
}

func (builder *RequestBuilder) appendConfigOptions(params *anthropic.MessageNewParams, cfg *genai.GenerateContentConfig) error {
	if cfg == nil {
		return nil
	}
	if cfg.MaxOutputTokens > 0 {
		params.MaxTokens = int64(cfg.MaxOutputTokens)
	}
	if cfg.Temperature != nil {
		params.Temperature = param.NewOpt(float64(*cfg.Temperature))
	}
	if cfg.TopP != nil {
		params.TopP = param.NewOpt(float64(*cfg.TopP))
	}
	if cfg.TopK != nil {
		params.TopK = param.NewOpt(int64(*cfg.TopK))
	}
	if len(cfg.StopSequences) > 0 {
		params.StopSequences = cfg.StopSequences
	}

	if len(cfg.Tools) > 0 {
		toolParams := make([]anthropic.ToolUnionParam, 0)
		for _, tool := range cfg.Tools {
			for _, fn := range tool.FunctionDeclarations {
				toolParam, err := builder.buildToToolParam(fn)
				if err != nil {
					return err
				}
				toolParams = append(toolParams, toolParam)
			}
		}
		params.Tools = toolParams
		if len(toolParams) > 0 {
			params.ToolChoice = anthropic.ToolChoiceParamOfTool("auto")
		}
	}
	return nil
}

func (builder *RequestBuilder) buildToToolParam(fn *genai.FunctionDeclaration) (anthropic.ToolUnionParam, error) {
	if fn.Name == "" {
		return anthropic.ToolUnionParam{}, fmt.Errorf("function declaration missing name")
	}

	inputSchema := anthropic.ToolInputSchemaParam{
		Type: "object",
	}
	if fn.Parameters != nil && len(fn.Parameters.Properties) > 0 {
		properties := make(map[string]any, len(fn.Parameters.Properties))
		for key, property := range fn.Parameters.Properties {
			normalizeSchemaType(property)
			properties[key] = property
		}
		if len(properties) > 0 {
			inputSchema.Properties = properties
		}
		if len(fn.Parameters.Required) > 0 {
			inputSchema.Required = append([]string(nil), fn.Parameters.Required...)
		}
	}

	toolParam := anthropic.ToolParam{
		Name:        fn.Name,
		InputSchema: inputSchema,
		Type:        anthropic.ToolTypeCustom,
	}
	if fn.Description != "" {
		toolParam.Description = param.NewOpt(fn.Description)
	}
	return anthropic.ToolUnionParam{OfTool: &toolParam}, nil
}

func normalizeSchemaType(schema *genai.Schema) {
	schema.Type = genai.Type(strings.ToLower(string(schema.Type)))
	for _, item := range schema.AnyOf {
		normalizeSchemaType(item)
	}
	for _, prop := range schema.Properties {
		normalizeSchemaType(prop)
	}
	if schema.Items != nil {
		normalizeSchemaType(schema.Items)
	}
}

func isImagePart(part *genai.Part) bool {
	if part == nil || part.InlineData == nil {
		return false
	}
	mime := strings.ToLower(part.InlineData.MIMEType)
	return strings.HasPrefix(mime, "image")
}

func stringifyFunctionResponse(resp map[string]any) string {
	if resp == nil {
		return ""
	}
	if result, ok := resp["result"]; ok && result != nil {
		return fmt.Sprint(result)
	}
	if output, ok := resp["output"]; ok && output != nil {
		return fmt.Sprint(output)
	}
	data, err := json.Marshal(resp)
	if err != nil {
		return fmt.Sprint(resp)
	}
	return string(data)
}
