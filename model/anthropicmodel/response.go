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
	"bytes"
	"encoding/json"
	"fmt"

	"github.com/anthropics/anthropic-sdk-go"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"
)

const (
	// StopReasonMetadataKey is the key in [model.LLMResponse.CustomMetadata] holding
	// the raw Claude stop_reason when it does not map 1:1 to a [genai.FinishReason]
	// (such as "pause_turn", "refusal", "model_context_window_exceeded", or unknown values).
	StopReasonMetadataKey = "anthropic_stop_reason"

	// CacheCreationInputTokensMetadataKey is the key in [model.LLMResponse.CustomMetadata]
	// holding the int64 count of prompt-cache creation tokens when > 0.
	CacheCreationInputTokensMetadataKey = "anthropic_cache_creation_input_tokens"
)

func convertMessageToLLMResponse(msg *anthropic.Message, unwrapOutput bool) (*model.LLMResponse, error) {
	if msg == nil {
		return nil, fmt.Errorf("%w: received nil message from API", ErrAPIStatus)
	}

	truncatedByTokenLimit := msg.StopReason == anthropic.StopReasonMaxTokens ||
		msg.StopReason == anthropic.StopReasonModelContextWindowExceeded

	parts := make([]*genai.Part, 0, len(msg.Content))
	for i := range msg.Content {
		if truncatedByTokenLimit && msg.Content[i].Type == "tool_use" {
			// Do not emit a FunctionCall part from a turn cut off by the token limit,
			// as its arguments may be incomplete or empty.
			continue
		}
		part, ok, err := convertContentBlock(&msg.Content[i], unwrapOutput)
		if err != nil {
			return nil, err
		}
		if ok {
			parts = append(parts, part)
		}
	}

	finishReason, rawStopReason := mapStopReason(msg.StopReason)
	usageMeta, cacheCreationTokens := extractUsageMetadata(msg.Usage)

	var customMeta map[string]any
	if rawStopReason != "" {
		customMeta = make(map[string]any)
		customMeta[StopReasonMetadataKey] = rawStopReason
	}
	if cacheCreationTokens > 0 {
		if customMeta == nil {
			customMeta = make(map[string]any)
		}
		customMeta[CacheCreationInputTokensMetadataKey] = cacheCreationTokens
	}

	return &model.LLMResponse{
		Content: &genai.Content{
			Role:  genai.RoleModel,
			Parts: parts,
		},
		UsageMetadata:  usageMeta,
		FinishReason:   finishReason,
		ModelVersion:   string(msg.Model),
		Partial:        false,
		TurnComplete:   true,
		CustomMetadata: customMeta,
	}, nil
}

func convertContentBlock(block *anthropic.ContentBlockUnion, unwrapOutput bool) (*genai.Part, bool, error) {
	switch block.Type {
	case "thinking":
		part := &genai.Part{
			Thought: true,
			Text:    block.Thinking,
		}
		if block.Signature != "" {
			part.ThoughtSignature = []byte(block.Signature)
		}
		return part, true, nil

	case "redacted_thinking":
		return &genai.Part{
			Thought:          true,
			ThoughtSignature: []byte(block.Data),
		}, true, nil

	case "text":
		text := block.Text
		if unwrapOutput {
			text = unwrapOutputSchemaText(text)
		}
		return genai.NewPartFromText(text), true, nil

	case "tool_use":
		args := make(map[string]any)
		trimmed := bytes.TrimSpace(block.Input)
		if len(trimmed) > 0 && !bytes.Equal(trimmed, []byte("null")) {
			if err := json.Unmarshal(trimmed, &args); err != nil {
				return nil, false, fmt.Errorf("anthropicmodel: failed to decode tool_use input for %q", block.Name)
			}
		}
		return &genai.Part{
			FunctionCall: &genai.FunctionCall{
				ID:   block.ID,
				Name: block.Name,
				Args: args,
			},
		}, true, nil

	default:
		return nil, false, nil
	}
}

func unwrapOutputSchemaText(text string) string {
	var wrapper map[string]json.RawMessage
	if err := json.Unmarshal([]byte(text), &wrapper); err != nil {
		return text
	}
	raw, ok := wrapper[wrappedOutputSchemaKey]
	if !ok || len(wrapper) != 1 {
		return text
	}
	return string(bytes.TrimSpace(raw))
}

func mapStopReason(stopReason anthropic.StopReason) (genai.FinishReason, string) {
	switch stopReason {
	case "":
		return genai.FinishReasonUnspecified, ""
	case anthropic.StopReasonEndTurn, anthropic.StopReasonStopSequence, anthropic.StopReasonToolUse:
		return genai.FinishReasonStop, ""
	case anthropic.StopReasonMaxTokens:
		return genai.FinishReasonMaxTokens, ""
	case anthropic.StopReasonModelContextWindowExceeded:
		return genai.FinishReasonMaxTokens, string(stopReason)
	case anthropic.StopReasonRefusal:
		return genai.FinishReasonSafety, string(stopReason)
	case anthropic.StopReasonPauseTurn:
		return genai.FinishReasonStop, string(stopReason)
	default:
		return genai.FinishReasonOther, string(stopReason)
	}
}

func extractUsageMetadata(usage anthropic.Usage) (*genai.GenerateContentResponseUsageMetadata, int64) {
	inputTokens := max(usage.InputTokens, 0)
	cacheRead := max(usage.CacheReadInputTokens, 0)
	cacheCreate := max(usage.CacheCreationInputTokens, 0)
	outputTokens := max(usage.OutputTokens, 0)

	promptTotal := inputTokens + cacheRead + cacheCreate

	var thinkingTokens int64
	if usage.OutputTokensDetails.ThinkingTokens > 0 || usage.OutputTokensDetails.JSON.ThinkingTokens.Valid() {
		thinkingTokens = max(usage.OutputTokensDetails.ThinkingTokens, 0)
		if thinkingTokens > outputTokens {
			thinkingTokens = outputTokens
		}
	}

	candidatesTokens := outputTokens - thinkingTokens

	return &genai.GenerateContentResponseUsageMetadata{
		PromptTokenCount:        int32(promptTotal),
		CandidatesTokenCount:    int32(candidatesTokens),
		TotalTokenCount:         int32(promptTotal + outputTokens),
		CachedContentTokenCount: int32(cacheRead),
		ThoughtsTokenCount:      int32(thinkingTokens),
	}, cacheCreate
}
