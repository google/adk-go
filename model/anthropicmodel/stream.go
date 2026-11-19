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
	"context"
	"encoding/json"
	"slices"
	"strings"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"
)

func (m *anthropicModel) generateStreaming(
	ctx context.Context,
	params anthropic.MessageNewParams,
	reqOpts []option.RequestOption,
	unwrapOutput bool,
	yield func(*model.LLMResponse, error) bool,
) {
	stream := m.client.Messages.NewStreaming(ctx, params, slices.Clone(reqOpts)...)
	defer func() {
		_ = stream.Close()
	}()

	var acc anthropic.Message
	toolInputBuffers := make(map[int64]*strings.Builder)

	for stream.Next() {
		if err := ctx.Err(); err != nil {
			yield(nil, err)
			return
		}

		event := stream.Current()
		if err := acc.Accumulate(event); err != nil {
			yield(nil, classifyError(err))
			return
		}

		switch event.Type {
		case "content_block_start":
			switch event.ContentBlock.Type {
			case "text":
				if !unwrapOutput && event.ContentBlock.Text != "" {
					if !yieldPartialPart(yield, string(acc.Model), genai.NewPartFromText(event.ContentBlock.Text)) {
						return
					}
				}
			case "thinking":
				if event.ContentBlock.Thinking != "" {
					if !yieldPartialPart(yield, string(acc.Model), &genai.Part{
						Thought: true,
						Text:    event.ContentBlock.Thinking,
					}) {
						return
					}
				}
			case "tool_use":
				toolInputBuffers[event.Index] = &strings.Builder{}
			}

		case "content_block_delta":
			switch event.Delta.Type {
			case "text_delta":
				if !unwrapOutput && event.Delta.Text != "" {
					if !yieldPartialPart(yield, string(acc.Model), genai.NewPartFromText(event.Delta.Text)) {
						return
					}
				}
			case "thinking_delta":
				if event.Delta.Thinking != "" {
					if !yieldPartialPart(yield, string(acc.Model), &genai.Part{
						Thought: true,
						Text:    event.Delta.Thinking,
					}) {
						return
					}
				}
			case "input_json_delta":
				if event.Delta.PartialJSON != "" {
					buf, ok := toolInputBuffers[event.Index]
					if !ok {
						buf = &strings.Builder{}
						toolInputBuffers[event.Index] = buf
					}
					buf.WriteString(event.Delta.PartialJSON)
				}
			}
		}
	}

	if err := stream.Err(); err != nil {
		yield(nil, classifyError(err))
		return
	}
	if err := ctx.Err(); err != nil {
		yield(nil, err)
		return
	}

	if acc.StopReason != anthropic.StopReasonMaxTokens && acc.StopReason != anthropic.StopReasonModelContextWindowExceeded {
		for idx, buf := range toolInputBuffers {
			if idx >= 0 && int(idx) < len(acc.Content) && buf.Len() > 0 {
				acc.Content[idx].Input = json.RawMessage(buf.String())
			}
		}
	}

	finalResp, err := convertMessageToLLMResponse(&acc, unwrapOutput)
	if err != nil {
		yield(nil, err)
		return
	}
	yield(finalResp, nil)
}

func yieldPartialPart(
	yield func(*model.LLMResponse, error) bool,
	modelVersion string,
	part *genai.Part,
) bool {
	return yield(&model.LLMResponse{
		Content: &genai.Content{
			Role:  genai.RoleModel,
			Parts: []*genai.Part{part},
		},
		ModelVersion: modelVersion,
		Partial:      true,
		TurnComplete: false,
	}, nil)
}
