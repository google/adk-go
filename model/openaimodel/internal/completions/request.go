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
	"encoding/json"
	"fmt"
	"strings"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/shared"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"

	"google.golang.org/adk/v2/model/openaimodel/internal/openaicommon"
)

// The message roles Chat Completions accepts from a caller. The tool role is
// absent on purpose: it comes from a function response, never from a content
// role.
const (
	roleUser      = "user"
	roleAssistant = "assistant"
	roleSystem    = "system"
	roleDeveloper = "developer"
)

// buildParams converts a generic LLMRequest into Chat Completions request
// params.
func buildParams(modelName string, req *model.LLMRequest) (openai.ChatCompletionNewParams, error) {
	if req == nil {
		return openai.ChatCompletionNewParams{}, openaicommon.ErrRequestNil
	}

	params := openai.ChatCompletionNewParams{Model: shared.ChatModel(modelName)}
	if req.Model != "" {
		params.Model = shared.ChatModel(req.Model)
	}

	messages, err := convertContents(req.Contents)
	if err != nil {
		return openai.ChatCompletionNewParams{}, err
	}
	if len(messages) == 0 {
		return openai.ChatCompletionNewParams{}, openaicommon.ErrNoContents
	}
	params.Messages = messages

	// Runs after the messages exist, because the system instruction joins them
	// as the first one rather than as a field of its own.
	if err := applyGenerationConfig(&params, req.Config); err != nil {
		return openai.ChatCompletionNewParams{}, err
	}

	tools, err := convertTools(req.Config)
	if err != nil {
		return openai.ChatCompletionNewParams{}, err
	}
	if len(tools) > 0 {
		params.Tools = tools
	}

	if cfg := req.Config; cfg != nil && cfg.ToolConfig != nil {
		choice, err := convertToolChoice(cfg.ToolConfig)
		if err != nil {
			return openai.ChatCompletionNewParams{}, err
		}
		// Only alongside tools: the API rejects a tool_choice sent without
		// them, "none" included, so a turn whose toolset came up empty would
		// fail. adk-python's LiteLLM path drops it in the same case.
		if choice != nil && len(tools) > 0 {
			params.ToolChoice = *choice
		}
	}

	return params, nil
}

// convertContents converts ADK contents into a Chat Completions message
// list. One content can produce several messages, because a tool result is a
// message of its own rather than a part of the turn that carried it.
func convertContents(contents []*genai.Content) ([]openai.ChatCompletionMessageParamUnion, error) {
	var (
		messages []openai.ChatCompletionMessageParamUnion
		tracker  openaicommon.CallTracker
	)
	for _, content := range contents {
		if content == nil || len(content.Parts) == 0 {
			continue
		}
		converted, err := convertContent(content, &tracker)
		if err != nil {
			return nil, err
		}
		messages = append(messages, converted...)
	}
	return messages, nil
}

// convertContent converts one content into the messages it becomes: a tool
// message per function response, then at most one message for the rest.
func convertContent(content *genai.Content, tracker *openaicommon.CallTracker) ([]openai.ChatCompletionMessageParamUnion, error) {
	role, err := normalizeRole(genai.Role(content.Role))
	if err != nil {
		return nil, err
	}

	var (
		messages []openai.ChatCompletionMessageParamUnion
		texts    []string
		calls    []openai.ChatCompletionMessageToolCallUnionParam
	)
	for _, part := range content.Parts {
		switch {
		case part == nil:
			continue
		case part.Thought:
			// Chat Completions has no field for prior-turn reasoning, and
			// replaying it as assistant text would hand the model its own
			// scratchpad back as an answer.
			continue
		case part.Text != "":
			texts = append(texts, part.Text)
		case part.FunctionCall != nil:
			call, err := newToolCall(tracker, part.FunctionCall)
			if err != nil {
				return nil, err
			}
			calls = append(calls, *call)
		case part.FunctionResponse != nil:
			msg, err := newToolMessage(tracker, part.FunctionResponse)
			if err != nil {
				return nil, err
			}
			messages = append(messages, *msg)
		default:
			return nil, fmt.Errorf("openai: unsupported content part %T", part)
		}
	}

	text := joinText(texts)
	switch {
	case role == roleAssistant:
		if text == "" && len(calls) == 0 {
			break
		}
		msg := openai.ChatCompletionAssistantMessageParam{ToolCalls: calls}
		if text != "" {
			msg.Content.OfString = param.NewOpt(text)
		}
		messages = append(messages, openai.ChatCompletionMessageParamUnion{OfAssistant: &msg})
	case text == "":
	case role == roleSystem:
		messages = append(messages, openai.SystemMessage(text))
	case role == roleDeveloper:
		messages = append(messages, openai.DeveloperMessage(text))
	default:
		messages = append(messages, openai.UserMessage(text))
	}
	return messages, nil
}

// normalizeRole maps a genai content role onto a Chat Completions message role. The
// system and developer roles survive rather than folding into user, which is
// what adk-python's LiteLLM path does, because Chat Completions has both.
func normalizeRole(role genai.Role) (string, error) {
	switch role {
	case "", genai.RoleUser:
		return roleUser, nil
	case genai.RoleModel:
		return roleAssistant, nil
	case roleSystem:
		return roleSystem, nil
	case roleDeveloper:
		return roleDeveloper, nil
	default:
		return "", fmt.Errorf("openai: unsupported role %q", role)
	}
}

// joinText flattens a turn's text parts into the one string a Chat
// Completions message carries, dropping parts holding only whitespace.
func joinText(texts []string) string {
	var b strings.Builder
	for _, text := range texts {
		if strings.TrimSpace(text) == "" {
			continue
		}
		if b.Len() > 0 {
			b.WriteString("\n")
		}
		b.WriteString(text)
	}
	return b.String()
}

// newToolCall converts a function call into the tool call an assistant
// message carries.
func newToolCall(tracker *openaicommon.CallTracker, fc *genai.FunctionCall) (*openai.ChatCompletionMessageToolCallUnionParam, error) {
	if fc.Name == "" {
		return nil, openaicommon.ErrFunctionCallMissingName
	}
	callID := tracker.TakeCallID(fc)
	args, err := openaicommon.MarshalFunctionArgs(fc.Args)
	if err != nil {
		return nil, err
	}
	return &openai.ChatCompletionMessageToolCallUnionParam{
		OfFunction: &openai.ChatCompletionMessageFunctionToolCallParam{
			ID: callID,
			Function: openai.ChatCompletionMessageFunctionToolCallFunctionParam{
				Name:      fc.Name,
				Arguments: args,
			},
		},
	}, nil
}

// newToolMessage converts a function response into the tool-role message
// that answers its call.
func newToolMessage(tracker *openaicommon.CallTracker, fr *genai.FunctionResponse) (*openai.ChatCompletionMessageParamUnion, error) {
	callID, err := tracker.ResolveResponseID(fr)
	if err != nil {
		return nil, err
	}
	payload, err := json.Marshal(fr.Response)
	if err != nil {
		return nil, fmt.Errorf("openai: marshal function response: %w", err)
	}
	msg := openai.ToolMessage(string(payload), callID)
	return &msg, nil
}

// serviceTiers maps genai's processing tiers onto the Chat Completions
// equivalents, reading an unspecified tier the way the Responses path does.
var serviceTiers = map[genai.ServiceTier]openai.ChatCompletionNewParamsServiceTier{
	genai.ServiceTierUnspecified: openai.ChatCompletionNewParamsServiceTierDefault,
	genai.ServiceTierStandard:    openai.ChatCompletionNewParamsServiceTierDefault,
	genai.ServiceTierFlex:        openai.ChatCompletionNewParamsServiceTierFlex,
	genai.ServiceTierPriority:    openai.ChatCompletionNewParamsServiceTierPriority,
}

// unsupportedConfigFields is openaicommon.UnsupportedConfigFields without the fields
// Chat Completions can express, so this endpoint does not refuse a setting it
// supports.
var unsupportedConfigFields = openaicommon.ConfigFieldsWithout("Seed")

// applyGenerationConfig translates the generation config onto Chat
// Completions params. Stop sequences, the penalties and seed are translated
// here and rejected on the Responses path, because only this endpoint has them.
func applyGenerationConfig(params *openai.ChatCompletionNewParams, cfg *genai.GenerateContentConfig) error {
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
		// Chat Completions has no top_k either, and forwarding it would trade a
		// clear error for a field the provider ignores.
		return openaicommon.ErrTopKNotSupported
	}
	if cfg.MaxOutputTokens > 0 {
		params.MaxCompletionTokens = param.NewOpt(int64(cfg.MaxOutputTokens))
	}
	if len(cfg.StopSequences) > 0 {
		// Sent whatever its length: OpenAI documents a limit of four, but
		// compatible providers differ, and a 400 naming the field diagnoses it
		// better than a guess made here.
		params.Stop = openai.ChatCompletionNewParamsStopUnion{OfStringArray: cfg.StopSequences}
	}
	if cfg.CandidateCount > 1 {
		// The API has "n", but a genai response carries one candidate and the
		// rest of this package assumes it.
		return openaicommon.ErrMultipleCandidatesNotSupported
	}
	if cfg.FrequencyPenalty != nil {
		params.FrequencyPenalty = param.NewOpt(float64(*cfg.FrequencyPenalty))
	}
	if cfg.PresencePenalty != nil {
		params.PresencePenalty = param.NewOpt(float64(*cfg.PresencePenalty))
	}
	if cfg.Seed != nil {
		params.Seed = param.NewOpt(int64(*cfg.Seed))
	}
	if cfg.ResponseLogprobs {
		// top_logprobs is refused without this, so it is set whenever logprobs
		// are asked for at all.
		params.Logprobs = param.NewOpt(true)
		if cfg.Logprobs != nil {
			params.TopLogprobs = param.NewOpt(int64(*cfg.Logprobs))
		}
	}
	if cfg.SystemInstruction != nil {
		inst, err := openaicommon.FlattenContentText(cfg.SystemInstruction)
		if err != nil {
			return fmt.Errorf("openai: system instruction: %w", err)
		}
		if inst != "" {
			// There is no instructions field here, so the instruction leads the
			// conversation as a system message.
			params.Messages = append([]openai.ChatCompletionMessageParamUnion{openai.SystemMessage(inst)}, params.Messages...)
		}
	}
	if cfg.ResponseMIMEType != "" && cfg.ResponseMIMEType != "text/plain" && cfg.ResponseMIMEType != "application/json" {
		return fmt.Errorf("%w: %s", openaicommon.ErrUnsupportedMIMEType, cfg.ResponseMIMEType)
	}
	if cfg.ResponseMIMEType == "application/json" || cfg.ResponseSchema != nil || cfg.ResponseJsonSchema != nil {
		format, err := newResponseFormat(cfg)
		if err != nil {
			return err
		}
		params.ResponseFormat = *format
	}
	if cfg.Labels != nil {
		return openaicommon.ErrLabelsNotSupported
	}
	if cfg.SafetySettings != nil {
		return openaicommon.ErrSafetySettingsNotSupported
	}
	// IncludeThoughts is accepted and ignored: this endpoint has no reasoning
	// summary to ask for, and refusing it would break a config that works
	// against Responses.
	effort, err := openaicommon.ReasoningEffortFor(cfg.ThinkingConfig)
	if err != nil {
		return err
	}
	params.ReasoningEffort = effort
	if cfg.ServiceTier != "" {
		tier, ok := serviceTiers[cfg.ServiceTier]
		if !ok {
			return fmt.Errorf("%w: ServiceTier %q", openaicommon.ErrUnsupportedConfigField, cfg.ServiceTier)
		}
		params.ServiceTier = tier
	}
	if err := openaicommon.RejectUntranslatableValues(cfg); err != nil {
		return err
	}
	// Last, so the named errors above win when a caller sets both.
	return openaicommon.RejectConfigFields(unsupportedConfigFields, cfg)
}

// newResponseFormat builds the response_format Chat Completions takes,
// carrying the same schema the Responses path puts under text.format.
func newResponseFormat(cfg *genai.GenerateContentConfig) (*openai.ChatCompletionNewParamsResponseFormatUnion, error) {
	if cfg.ResponseSchema == nil && cfg.ResponseJsonSchema == nil {
		obj := shared.NewResponseFormatJSONObjectParam()
		return &openai.ChatCompletionNewParamsResponseFormatUnion{OfJSONObject: &obj}, nil
	}
	var (
		schema map[string]any
		err    error
	)
	if cfg.ResponseJsonSchema != nil {
		schema, err = openaicommon.NormalizeSchema(cfg.ResponseJsonSchema)
	} else {
		schema, err = openaicommon.SchemaToMap(cfg.ResponseSchema)
	}
	if err != nil {
		return nil, err
	}
	openaicommon.EnforceStrictOpenAISchema(schema)
	name := "adk_response"
	if cfg.ResponseSchema != nil && cfg.ResponseSchema.Title != "" {
		name = cfg.ResponseSchema.Title
	}
	return &openai.ChatCompletionNewParamsResponseFormatUnion{
		OfJSONSchema: &shared.ResponseFormatJSONSchemaParam{
			JSONSchema: shared.ResponseFormatJSONSchemaJSONSchemaParam{
				Name:   name,
				Schema: schema,
				Strict: param.NewOpt(true),
			},
		},
	}, nil
}
