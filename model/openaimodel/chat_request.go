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

package openaimodel

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/shared"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"
)

// The message roles Chat Completions accepts from a caller. The tool role is
// absent on purpose: it comes from a function response, never from a content
// role.
const (
	chatRoleUser      = "user"
	chatRoleAssistant = "assistant"
	chatRoleSystem    = "system"
	chatRoleDeveloper = "developer"
)

// buildChatParams converts a generic LLMRequest into Chat Completions request
// params.
func buildChatParams(modelName string, req *model.LLMRequest) (openai.ChatCompletionNewParams, error) {
	if req == nil {
		return openai.ChatCompletionNewParams{}, ErrRequestNil
	}

	params := openai.ChatCompletionNewParams{Model: shared.ChatModel(modelName)}
	if req.Model != "" {
		params.Model = shared.ChatModel(req.Model)
	}

	messages, err := convertChatMessages(req.Contents)
	if err != nil {
		return openai.ChatCompletionNewParams{}, err
	}
	if len(messages) == 0 {
		return openai.ChatCompletionNewParams{}, ErrNoContents
	}
	params.Messages = messages

	// Runs after the messages exist, because the system instruction joins them
	// as the first one rather than as a field of its own.
	if err := applyChatGenerationConfig(&params, req.Config); err != nil {
		return openai.ChatCompletionNewParams{}, err
	}

	tools, err := convertChatTools(req.Config)
	if err != nil {
		return openai.ChatCompletionNewParams{}, err
	}
	if len(tools) > 0 {
		params.Tools = tools
	}

	if cfg := req.Config; cfg != nil && cfg.ToolConfig != nil {
		choice, err := convertChatToolChoice(cfg.ToolConfig)
		if err != nil {
			return openai.ChatCompletionNewParams{}, err
		}
		if choice != nil {
			params.ToolChoice = *choice
		}
	}

	return params, nil
}

// convertChatMessages converts ADK contents into a Chat Completions message
// list. One content can produce several messages, because a tool result is a
// message of its own rather than a part of the turn that carried it.
func convertChatMessages(contents []*genai.Content) ([]openai.ChatCompletionMessageParamUnion, error) {
	var (
		messages []openai.ChatCompletionMessageParamUnion
		tracker  callTracker
	)
	for _, content := range contents {
		if content == nil || len(content.Parts) == 0 {
			continue
		}
		converted, err := convertChatContent(content, &tracker)
		if err != nil {
			return nil, err
		}
		messages = append(messages, converted...)
	}
	return messages, nil
}

// convertChatContent converts one content into the messages it becomes: a tool
// message per function response, then at most one message for the rest.
func convertChatContent(content *genai.Content, tracker *callTracker) ([]openai.ChatCompletionMessageParamUnion, error) {
	role, err := chatRole(genai.Role(content.Role))
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
			call, err := newChatToolCall(tracker, part.FunctionCall)
			if err != nil {
				return nil, err
			}
			calls = append(calls, *call)
		case part.FunctionResponse != nil:
			msg, err := newChatToolMessage(tracker, part.FunctionResponse)
			if err != nil {
				return nil, err
			}
			messages = append(messages, *msg)
		default:
			return nil, fmt.Errorf("openai: unsupported content part %T", part)
		}
	}

	text := joinChatText(texts)
	switch {
	case role == chatRoleAssistant:
		if text == "" && len(calls) == 0 {
			break
		}
		msg := openai.ChatCompletionAssistantMessageParam{ToolCalls: calls}
		if text != "" {
			msg.Content.OfString = param.NewOpt(text)
		}
		messages = append(messages, openai.ChatCompletionMessageParamUnion{OfAssistant: &msg})
	case text == "":
	case role == chatRoleSystem:
		messages = append(messages, openai.SystemMessage(text))
	case role == chatRoleDeveloper:
		messages = append(messages, openai.DeveloperMessage(text))
	default:
		messages = append(messages, openai.UserMessage(text))
	}
	return messages, nil
}

// chatRole maps a genai content role onto a Chat Completions message role. The
// system and developer roles survive rather than folding into user, which is
// what adk-python's LiteLLM path does, because Chat Completions has both.
func chatRole(role genai.Role) (string, error) {
	switch role {
	case "", genai.RoleUser:
		return chatRoleUser, nil
	case genai.RoleModel:
		return chatRoleAssistant, nil
	case chatRoleSystem:
		return chatRoleSystem, nil
	case chatRoleDeveloper:
		return chatRoleDeveloper, nil
	default:
		return "", fmt.Errorf("openai: unsupported role %q", role)
	}
}

// joinChatText flattens a turn's text parts into the one string a Chat
// Completions message carries, dropping parts holding only whitespace.
func joinChatText(texts []string) string {
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

// newChatToolCall converts a function call into the tool call an assistant
// message carries.
func newChatToolCall(tracker *callTracker, fc *genai.FunctionCall) (*openai.ChatCompletionMessageToolCallUnionParam, error) {
	if fc.Name == "" {
		return nil, ErrFunctionCallMissingName
	}
	callID := tracker.takeCallID(fc)
	args, err := marshalFunctionArgs(fc.Args)
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

// newChatToolMessage converts a function response into the tool-role message
// that answers its call.
func newChatToolMessage(tracker *callTracker, fr *genai.FunctionResponse) (*openai.ChatCompletionMessageParamUnion, error) {
	callID, err := tracker.resolveResponseID(fr)
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

// chatServiceTiers maps genai's processing tiers onto the Chat Completions
// equivalents, reading an unspecified tier the way the Responses path does.
var chatServiceTiers = map[genai.ServiceTier]openai.ChatCompletionNewParamsServiceTier{
	genai.ServiceTierUnspecified: openai.ChatCompletionNewParamsServiceTierDefault,
	genai.ServiceTierStandard:    openai.ChatCompletionNewParamsServiceTierDefault,
	genai.ServiceTierFlex:        openai.ChatCompletionNewParamsServiceTierFlex,
	genai.ServiceTierPriority:    openai.ChatCompletionNewParamsServiceTierPriority,
}

// chatUnsupportedConfigFields is unsupportedConfigFields without the fields
// Chat Completions can express, so this endpoint does not refuse a setting it
// supports.
var chatUnsupportedConfigFields = configFieldsWithout("Seed")

// configFieldsWithout returns unsupportedConfigFields minus the named entries.
func configFieldsWithout(names ...string) []configField {
	drop := make(map[string]bool, len(names))
	for _, name := range names {
		drop[name] = true
	}
	kept := make([]configField, 0, len(unsupportedConfigFields))
	for _, field := range unsupportedConfigFields {
		if !drop[field.name] {
			kept = append(kept, field)
		}
	}
	return kept
}

// applyChatGenerationConfig translates the generation config onto Chat
// Completions params. Stop sequences, the penalties and seed are translated
// here and rejected on the Responses path, because only this endpoint has them.
func applyChatGenerationConfig(params *openai.ChatCompletionNewParams, cfg *genai.GenerateContentConfig) error {
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
		return ErrTopKNotSupported
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
		return ErrMultipleCandidatesNotSupported
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
		inst, err := flattenContentText(cfg.SystemInstruction)
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
		return fmt.Errorf("%w: %s", ErrUnsupportedMIMEType, cfg.ResponseMIMEType)
	}
	if cfg.ResponseMIMEType == "application/json" || cfg.ResponseSchema != nil || cfg.ResponseJsonSchema != nil {
		format, err := newChatResponseFormat(cfg)
		if err != nil {
			return err
		}
		params.ResponseFormat = *format
	}
	if cfg.Labels != nil {
		return ErrLabelsNotSupported
	}
	if cfg.SafetySettings != nil {
		return ErrSafetySettingsNotSupported
	}
	if err := applyChatThinkingConfig(params, cfg.ThinkingConfig); err != nil {
		return err
	}
	if cfg.ServiceTier != "" {
		tier, ok := chatServiceTiers[cfg.ServiceTier]
		if !ok {
			return fmt.Errorf("%w: ServiceTier %q", ErrUnsupportedConfigField, cfg.ServiceTier)
		}
		params.ServiceTier = tier
	}
	if err := rejectUntranslatableValues(cfg); err != nil {
		return err
	}
	// Last, so the named errors above win when a caller sets both.
	for _, field := range chatUnsupportedConfigFields {
		if field.isSet(cfg) {
			return fmt.Errorf("%w: %s", ErrUnsupportedConfigField, field.name)
		}
	}
	return nil
}

// newChatResponseFormat builds the response_format Chat Completions takes,
// carrying the same schema the Responses path puts under text.format.
func newChatResponseFormat(cfg *genai.GenerateContentConfig) (*openai.ChatCompletionNewParamsResponseFormatUnion, error) {
	if cfg.ResponseSchema == nil && cfg.ResponseJsonSchema == nil {
		obj := shared.NewResponseFormatJSONObjectParam()
		return &openai.ChatCompletionNewParamsResponseFormatUnion{OfJSONObject: &obj}, nil
	}
	var (
		schema map[string]any
		err    error
	)
	if cfg.ResponseJsonSchema != nil {
		schema, err = normalizeSchema(cfg.ResponseJsonSchema)
	} else {
		schema, err = schemaToMap(cfg.ResponseSchema)
	}
	if err != nil {
		return nil, err
	}
	enforceStrictOpenAISchema(schema)
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

// applyChatThinkingConfig maps genai's thinking config onto reasoning_effort,
// reusing the Responses level mapping because both endpoints take the same
// effort values.
//
// IncludeThoughts is accepted and ignored: this endpoint has no reasoning
// summary to ask for, and refusing it would break a config that works against
// Responses.
func applyChatThinkingConfig(params *openai.ChatCompletionNewParams, cfg *genai.ThinkingConfig) error {
	if cfg == nil {
		return nil
	}
	if cfg.ThinkingBudget != nil && *cfg.ThinkingBudget < dynamicThinkingBudget {
		return fmt.Errorf("%w: ThinkingConfig.ThinkingBudget %d", ErrUnsupportedConfigField, *cfg.ThinkingBudget)
	}
	// A level outranks a budget, but only when it names one.
	level := cfg.ThinkingLevel
	if level == genai.ThinkingLevelUnspecified && cfg.ThinkingBudget != nil {
		level = ""
	}
	switch {
	case level != "":
		effort, ok := reasoningEfforts[level]
		if !ok {
			return fmt.Errorf("%w: ThinkingConfig.ThinkingLevel %q", ErrUnsupportedConfigField, level)
		}
		params.ReasoningEffort = effort
	case cfg.ThinkingBudget != nil:
		switch *cfg.ThinkingBudget {
		case 0:
			params.ReasoningEffort = shared.ReasoningEffortNone
		case dynamicThinkingBudget:
			// The caller asked the model to decide, so no effort is sent.
		default:
			params.ReasoningEffort = shared.ReasoningEffortMedium
		}
	}
	return nil
}
