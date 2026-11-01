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
	pendingDeltas := make(map[int64]*blockDeltas)
	toolInputBuffers := make(map[int64]*strings.Builder)

	for stream.Next() {
		if err := ctx.Err(); err != nil {
			yield(nil, err)
			return
		}

		event := stream.Current()
		if err := accumulate(&acc, event, pendingDeltas); err != nil {
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
	flushDeltas(&acc, pendingDeltas)

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

// blockDeltas buffers the text and thinking deltas of one content block until
// they are appended to the accumulated message.
type blockDeltas struct {
	text, thinking strings.Builder
}

// accumulate applies event to acc like acc.Accumulate, except that text and
// thinking deltas are buffered in pending: Accumulate appends each one to its
// block by string concatenation, copying the whole block per delta. The
// buffers are appended before every event that may read or replace block
// text; call flushDeltas before reading block text from acc.
func accumulate(acc *anthropic.Message, event anthropic.MessageStreamEventUnion, pending map[int64]*blockDeltas) error {
	switch event.Type {
	case "content_block_delta":
		if event.Delta.Type != "text_delta" && event.Delta.Type != "thinking_delta" {
			return acc.Accumulate(event)
		}
		// Accumulate still validates the block index and returns its error; with
		// the delta blanked it leaves the block unchanged.
		text, thinking := event.Delta.Text, event.Delta.Thinking
		event.Delta.Text, event.Delta.Thinking = "", ""
		if err := acc.Accumulate(event); err != nil {
			return err
		}
		d := pending[event.Index]
		if d == nil {
			d = &blockDeltas{}
			pending[event.Index] = d
		}
		if event.Delta.Type == "text_delta" {
			d.text.WriteString(text)
		} else {
			d.thinking.WriteString(thinking)
		}
		return nil
	case "content_block_start", "message_delta":
		// Neither reads nor replaces the text of existing blocks.
		return acc.Accumulate(event)
	default:
		// content_block_stop and message_stop render block text into the raw
		// JSON, message_start replaces the blocks.
		flushDeltas(acc, pending)
		return acc.Accumulate(event)
	}
}

// flushDeltas appends the pending deltas to their content blocks in acc.
func flushDeltas(acc *anthropic.Message, pending map[int64]*blockDeltas) {
	for i, d := range pending {
		acc.Content[i].Text += d.text.String()
		acc.Content[i].Thinking += d.thinking.String()
	}
	clear(pending)
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
