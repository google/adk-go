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

package openai

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared"
	"github.com/openai/openai-go/v3/shared/constant"
	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

// buildOpenAIParams converts a generic LLMRequest into the OpenAI-specific
// responses.ResponseNewParams format, preparing it for an API call.
func buildOpenAIParams(modelName string, req *model.LLMRequest) (responses.ResponseNewParams, error) {
	if req == nil {
		return responses.ResponseNewParams{}, ErrRequestNil
	}

	params := responses.ResponseNewParams{
		Model: shared.ResponsesModel(modelName),
	}
	if req.Model != "" {
		params.Model = shared.ResponsesModel(req.Model)
	}

	// We convert the generic content parts into OpenAI's input format.
	input, err := convertContents(req.Contents)
	if err != nil {
		return responses.ResponseNewParams{}, err
	}
	if len(input) == 0 {
		return responses.ResponseNewParams{}, ErrNoContents
	}
	params.Input = responses.ResponseNewParamsInputUnion{
		OfInputItemList: input,
	}

	// Apply generation configuration settings like temperature and max output tokens.
	if err := applyGenerationConfig(&params, req.Config); err != nil {
		return responses.ResponseNewParams{}, err
	}

	// Convert any specified tools into the OpenAI tool format.
	tools, err := convertTools(req.Config)
	if err != nil {
		return responses.ResponseNewParams{}, err
	}
	if len(tools) > 0 {
		params.Tools = tools
	}

	// Handle tool choice configuration, if provided.
	if cfg := req.Config; cfg != nil && cfg.ToolConfig != nil {
		choice, err := convertToolChoice(cfg.ToolConfig)
		if err != nil {
			return responses.ResponseNewParams{}, err
		}
		if choice != nil {
			params.ToolChoice = *choice
		}
	}

	return params, nil
}

func convertContents(contents []*genai.Content) (responses.ResponseInputParam, error) {
	var (
		items     responses.ResponseInputParam
		tracker   callTracker
		textParts []string
		curRole   genai.Role = genai.RoleUser
		// flushText is a helper function that takes any accumulated text parts
		// and converts them into a message, then appends it to our items.
		flushText = func() error {
			if len(textParts) == 0 {
				return nil
			}
			msg, err := newMessage(curRole, textParts)
			if err != nil {
				return err
			}
			if msg != nil {
				items = append(items, responses.ResponseInputItemUnionParam{OfMessage: msg})
			}
			textParts = textParts[:0]
			return nil
		}
	)

	for _, content := range contents {
		if content == nil || len(content.Parts) == 0 {
			continue
		}
		curRole = genai.Role(content.Role)
		for _, part := range content.Parts {
			switch {
			case part == nil:
				continue
			case part.Text != "":
				textParts = append(textParts, part.Text)
			case part.FunctionCall != nil:
				// If we encounter a function call, we first flush any accumulated text.
				if err := flushText(); err != nil {
					return nil, err
				}
				callParam, err := tracker.newFunctionCall(part.FunctionCall)
				if err != nil {
					return nil, err
				}
				items = append(items, responses.ResponseInputItemUnionParam{OfFunctionCall: callParam})
			case part.FunctionResponse != nil:
				// Similarly, for a function response, we flush text before adding the response.
				if err := flushText(); err != nil {
					return nil, err
				}
				respParam, err := tracker.newFunctionResponse(part.FunctionResponse)
				if err != nil {
					return nil, err
				}
				items = append(items, responses.ResponseInputItemUnionParam{OfFunctionCallOutput: respParam})
			default:
				return nil, fmt.Errorf("openai: unsupported content part %T", part)
			}
		}
		// After processing all parts in a content block, we flush any remaining text.
		if err := flushText(); err != nil {
			return nil, err
		}
	}

	return items, nil
}

func newMessage(role genai.Role, texts []string) (*responses.EasyInputMessageParam, error) {
	if len(texts) == 0 {
		return nil, nil
	}
	msgRole, err := normalizeRole(role)
	if err != nil {
		return nil, err
	}
	contentList := make(responses.ResponseInputMessageContentListParam, 0, len(texts))
	for _, txt := range texts {
		if strings.TrimSpace(txt) == "" {
			continue
		}
		textParam := responses.ResponseInputTextParam{
			Text: txt,
			Type: constant.InputText("input_text"),
		}
		contentList = append(contentList, responses.ResponseInputContentUnionParam{
			OfInputText: &textParam,
		})
	}
	if len(contentList) == 0 {
		return nil, nil
	}
	return &responses.EasyInputMessageParam{
		Role: msgRole,
		Type: responses.EasyInputMessageTypeMessage,
		Content: responses.EasyInputMessageContentUnionParam{
			OfInputItemContentList: contentList,
		},
	}, nil
}

func normalizeRole(role genai.Role) (responses.EasyInputMessageRole, error) {
	switch role {
	case "", genai.RoleUser:
		return responses.EasyInputMessageRoleUser, nil
	case genai.RoleModel:
		return responses.EasyInputMessageRoleAssistant, nil
	case "system":
		return responses.EasyInputMessageRoleSystem, nil
	case "developer":
		return responses.EasyInputMessageRoleDeveloper, nil
	default:
		return "", fmt.Errorf("openai: unsupported role %q", role)
	}
}

// callTracker helps us manage function call IDs, ensuring that function responses
// can be correctly associated with their corresponding calls, especially when IDs are not
// explicitly provided in the input.
type callTracker struct {
	nextID  int
	pending []string
}

// newFunctionCall converts a generic genai.FunctionCall into an OpenAI-specific
// ResponseFunctionToolCallParam. We generate a unique callID if one isn't
// provided, and then marshal the function arguments into a JSON string.
func (t *callTracker) newFunctionCall(fc *genai.FunctionCall) (*responses.ResponseFunctionToolCallParam, error) {
	if fc.Name == "" {
		return nil, ErrFunctionCallMissingName
	}
	callID := fc.ID
	if callID == "" {
		callID = fmt.Sprintf("adk-openai-call-%d", t.nextID)
		t.nextID++
	}
	t.pending = append(t.pending, callID)
	argsValue := fc.Args
	if argsValue == nil {
		argsValue = map[string]any{}
	}
	args, err := json.Marshal(argsValue)
	if err != nil {
		return nil, fmt.Errorf("openai: marshal function args: %w", err)
	}
	return &responses.ResponseFunctionToolCallParam{
		Name:      fc.Name,
		CallID:    callID,
		Arguments: string(args),
		Type:      constant.FunctionCall("function_call"),
	}, nil
}

// newFunctionResponse converts a generic genai.FunctionResponse into an OpenAI-specific
// ResponseInputItemFunctionCallOutputParam. We try to match the response to a pending
// function call. If an explicit callID is provided, we find and remove it from our
// pending list. Otherwise, we assume it corresponds to the oldest pending call.
func (t *callTracker) newFunctionResponse(fr *genai.FunctionResponse) (*responses.ResponseInputItemFunctionCallOutputParam, error) {
	callID := fr.ID
	if callID == "" {
		if len(t.pending) == 0 {
			return nil, fmt.Errorf("openai: response for %q missing call id", fr.Name)
		}
		callID = t.pending[0]
		t.pending = t.pending[1:]
	} else {
		found := false
		for i, pending := range t.pending {
			if pending == callID {
				t.pending = append(t.pending[:i], t.pending[i+1:]...)
				found = true
				break
			}
		}
		if !found {
			return nil, fmt.Errorf("openai: received function response for unknown or already completed call id %q", callID)
		}
	}
	payload, err := json.Marshal(fr.Response)
	if err != nil {
		return nil, fmt.Errorf("openai: marshal function response: %w", err)
	}
	return &responses.ResponseInputItemFunctionCallOutputParam{
		CallID: callID,
		Output: responses.ResponseInputItemFunctionCallOutputOutputUnionParam{
			OfString: param.NewOpt(string(payload)),
		},
		Type: constant.FunctionCallOutput("function_call_output"),
	}, nil
}

// applyGenerationConfig translates our generic generation configuration into
// OpenAI-specific parameters. We also validate and return errors for features
// that are not supported by the OpenAI Responses API.
func applyGenerationConfig(params *responses.ResponseNewParams, cfg *genai.GenerateContentConfig) error {
	if cfg == nil {
		return nil
	}
	if cfg.Temperature != nil {
		params.Temperature = param.NewOpt(float64(*cfg.Temperature))
	}
	if cfg.TopP != nil {
		params.TopP = param.NewOpt(float64(*cfg.TopP))
	}
	if cfg.TopK != nil {
		return ErrTopKNotSupported
	}
	if cfg.MaxOutputTokens > 0 {
		params.MaxOutputTokens = param.NewOpt(int64(cfg.MaxOutputTokens))
	}
	if len(cfg.StopSequences) > 0 {
		return ErrStopSequencesNotSupported
	}
	if cfg.CandidateCount > 1 {
		return ErrMultipleCandidatesNotSupported
	}
	if cfg.FrequencyPenalty != nil || cfg.PresencePenalty != nil {
		return ErrPenaltiesNotSupported
	}
	if cfg.ResponseLogprobs {
		if cfg.Logprobs != nil {
			params.TopLogprobs = param.NewOpt(int64(*cfg.Logprobs))
		} else {
			params.TopLogprobs = param.NewOpt(int64(1))
		}
	}
	if cfg.SystemInstruction != nil {
		inst, err := flattenContentText(cfg.SystemInstruction)
		if err != nil {
			return fmt.Errorf("openai: system instruction: %w", err)
		}
		if inst != "" {
			params.Instructions = param.NewOpt(inst)
		}
	}
	if cfg.ResponseMIMEType != "" && cfg.ResponseMIMEType != "text/plain" && cfg.ResponseMIMEType != "application/json" {
		return fmt.Errorf("openai: response mime type %q is not supported", cfg.ResponseMIMEType)
	}
	if cfg.ResponseMIMEType == "application/json" || cfg.ResponseSchema != nil || cfg.ResponseJsonSchema != nil {
		format, err := newJSONSchemaFormat(cfg)
		if err != nil {
			return err
		}
		params.Text = responses.ResponseTextConfigParam{
			Format: responses.ResponseFormatTextConfigUnionParam{
				OfJSONSchema: format,
			},
		}
	}
	if cfg.Labels != nil {
		return ErrLabelsNotSupported
	}
	if cfg.SafetySettings != nil {
		return ErrSafetySettingsNotSupported
	}
	return nil
}

func flattenContentText(content *genai.Content) (string, error) {
	if content == nil {
		return "", nil
	}
	var b strings.Builder
	for _, part := range content.Parts {
		if part == nil {
			continue
		}
		if part.Text == "" {
			return "", fmt.Errorf("non-text system instruction part %T", part)
		}
		if b.Len() > 0 {
			b.WriteString("\n")
		}
		b.WriteString(part.Text)
	}
	return b.String(), nil
}

// newJSONSchemaFormat constructs an OpenAI-specific JSON schema format from our
// generic GenerateContentConfig. We handle cases where the schema is provided
// directly or needs to be converted, and assign a name to it.
func newJSONSchemaFormat(cfg *genai.GenerateContentConfig) (*responses.ResponseFormatTextJSONSchemaConfigParam, error) {
	var (
		schema map[string]any
		err    error
	)
	switch {
	case cfg.ResponseJsonSchema != nil:
		schema, err = normalizeSchema(cfg.ResponseJsonSchema)
	case cfg.ResponseSchema != nil:
		schema, err = schemaToMap(cfg.ResponseSchema)
	default:
		return nil, ErrJSONResponseWithoutSchema
	}
	if err != nil {
		return nil, err
	}
	name := "adk_response"
	if cfg.ResponseSchema != nil && cfg.ResponseSchema.Title != "" {
		name = cfg.ResponseSchema.Title
	}
	return &responses.ResponseFormatTextJSONSchemaConfigParam{
		Name:   name,
		Schema: schema,
		Type:   constant.JSONSchema("json_schema"),
	}, nil
}

func normalizeSchema(schema any) (map[string]any, error) {
	switch s := schema.(type) {
	case map[string]any:
		return s, nil
	case nil:
		return nil, ErrEmptyJSONSchema
	default:
		bytes, err := json.Marshal(s)
		if err != nil {
			return nil, fmt.Errorf("openai: marshal json schema: %w", err)
		}
		var result map[string]any
		if err := json.Unmarshal(bytes, &result); err != nil {
			return nil, fmt.Errorf("openai: unmarshal json schema: %w", err)
		}
		return result, nil
	}
}
