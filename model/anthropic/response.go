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
	"encoding/json"
	"fmt"

	"github.com/anthropics/anthropic-sdk-go"

	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

func parsePartialStreamEvent(event anthropic.MessageStreamEventUnion) *model.LLMResponse {
	deltaEvent, ok := event.AsAny().(anthropic.ContentBlockDeltaEvent)
	if !ok {
		return nil
	}

	var part *genai.Part
	switch v := deltaEvent.Delta.AsAny().(type) {
	case anthropic.TextDelta:
		if v.Text != "" {
			part = genai.NewPartFromText(v.Text)
		}
	case anthropic.ThinkingDelta:
		if v.Thinking != "" {
			part = &genai.Part{Text: v.Thinking, Thought: true}
		}
	default:
		// Unsupported delta type for partial response
		return nil
	}

	if part != nil {
		content := genai.NewContentFromParts([]*genai.Part{part}, genai.RoleModel)
		return &model.LLMResponse{
			Content: content,
		}
	}
	return nil
}

type ResponseBuilder struct{}

func (builder *ResponseBuilder) FromMessage(message *anthropic.Message) (*model.LLMResponse, error) {
	parts := make([]*genai.Part, 0, len(message.Content))
	for _, block := range message.Content {
		part, err := builder.buildPartFromContentBlock(block)
		if err != nil {
			return nil, err
		}
		if part != nil {
			parts = append(parts, part)
		}
	}
	content := genai.NewContentFromParts(parts, genai.RoleModel)

	llmResponse := &model.LLMResponse{
		Content:        content,
		FinishReason:   builder.buildFinishReason(message.StopReason),
		UsageMetadata:  builder.extractUsage(message.Usage),
		CustomMetadata: make(map[string]any, 0),
	}
	if message.StopReason != "" {
		llmResponse.CustomMetadata["stop_reason"] = message.StopReason
	}
	if message.StopSequence != "" {
		llmResponse.CustomMetadata["stop_sequence"] = message.StopSequence
	}
	return llmResponse, nil
}

func (builder *ResponseBuilder) buildPartFromContentBlock(block anthropic.ContentBlockUnion) (*genai.Part, error) {
	switch v := block.AsAny().(type) {
	case anthropic.TextBlock:
		return genai.NewPartFromText(v.Text), nil
	case anthropic.ToolUseBlock:
		args := make(map[string]any)
		if len(v.Input) > 0 {
			if err := json.Unmarshal(v.Input, &args); err != nil {
				return nil, fmt.Errorf("failed to decode tool input: %w", err)
			}
		}
		return &genai.Part{
			FunctionCall: &genai.FunctionCall{
				ID:   v.ID,
				Name: v.Name,
				Args: args,
			},
		}, nil
	default:
		return nil, fmt.Errorf("not supported '%T' yet", v)
	}
}

func (builder *ResponseBuilder) extractUsage(usage anthropic.Usage) *genai.GenerateContentResponseUsageMetadata {
	total := usage.InputTokens + usage.OutputTokens
	return &genai.GenerateContentResponseUsageMetadata{
		PromptTokenCount:        int32(usage.InputTokens),
		CandidatesTokenCount:    int32(usage.OutputTokens),
		TotalTokenCount:         int32(total),
		CachedContentTokenCount: int32(usage.CacheReadInputTokens),
	}
}

func (builder *ResponseBuilder) buildFinishReason(stop anthropic.StopReason) genai.FinishReason {
	switch stop {
	case anthropic.StopReasonEndTurn, anthropic.StopReasonStopSequence, anthropic.StopReasonToolUse:
		return genai.FinishReasonStop
	case anthropic.StopReasonMaxTokens:
		return genai.FinishReasonMaxTokens
	default:
		return genai.FinishReasonUnspecified
	}
}
