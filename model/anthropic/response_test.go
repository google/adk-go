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

package anthropic

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/packages/ssestream"
	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"google.golang.org/genai"

	"google.golang.org/adk/model"
)

type mockDecoder struct {
	events  []ssestream.Event
	index   int
	current ssestream.Event
	closed  bool
}

func (d *mockDecoder) Event() ssestream.Event {
	return d.current
}

func (d *mockDecoder) Next() bool {
	if d.index >= len(d.events) {
		return false
	}
	d.current = d.events[d.index]
	d.index++
	return true
}

func (d *mockDecoder) Close() error {
	d.closed = true
	return nil
}

func (d *mockDecoder) Err() error {
	return nil
}

func TestResponseBuilder_FromMessage(t *testing.T) {
	messageJSON := `{
		"id": "msg_1",
		"type": "message",
		"role": "assistant",
		"model": "claude-3",
		"stop_reason": "end_turn",
		"stop_sequence": "!",
		"content": [
			{"type": "text", "text": "Final answer."},
			{"type": "tool_use", "id": "call-1", "name": "lookup", "input": {"arg": "value"}}
		],
		"usage": {
			"cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
			"cache_creation_input_tokens": 0,
			"cache_read_input_tokens": 2,
			"input_tokens": 5,
			"output_tokens": 3,
			"server_tool_use": {"web_search_requests": 0},
			"service_tier": "standard"
		}
	}`

	msg := mustUnmarshalMessage(t, messageJSON)
	builder := ResponseBuilder{}

	got, err := builder.FromMessage(msg)
	if err != nil {
		t.Fatalf("ResponseBuilder.FromMessage() error = %v", err)
	}

	want := &model.LLMResponse{
		Content: genai.NewContentFromParts(
			[]*genai.Part{
				genai.NewPartFromText("Final answer."),
				{
					FunctionCall: &genai.FunctionCall{
						ID:   "call-1",
						Name: "lookup",
						Args: map[string]any{"arg": "value"},
					},
				},
			},
			genai.RoleModel,
		),
		UsageMetadata: &genai.GenerateContentResponseUsageMetadata{
			PromptTokenCount:        5,
			CandidatesTokenCount:    3,
			TotalTokenCount:         8,
			CachedContentTokenCount: 2,
		},
		FinishReason: genai.FinishReasonStop,
		CustomMetadata: map[string]any{
			"stop_reason":   anthropic.StopReason("end_turn"),
			"stop_sequence": "!",
		},
	}

	if diff := cmp.Diff(want, got, cmpopts.IgnoreFields(model.LLMResponse{}, "AvgLogprobs")); diff != "" {
		t.Fatalf("ResponseBuilder.FromMessage() diff (-want +got):\n%s", diff)
	}
}

func TestParsePartialStreamEvent(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		raw  string
		want *model.LLMResponse
	}{
		{
			name: "text_delta",
			raw:  `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"partial"}}`,
			want: &model.LLMResponse{
				Content: genai.NewContentFromText("partial", genai.RoleModel),
				Partial: true,
			},
		},
		{
			name: "thinking_delta",
			raw:  `{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"reason"}}`,
			want: &model.LLMResponse{
				Content: genai.NewContentFromParts(
					[]*genai.Part{{Text: "reason", Thought: true}},
					genai.RoleModel,
				),
				Partial: true,
			},
		},
		{
			name: "unsupported_delta",
			raw:  `{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{}"}}`,
			want: nil,
		},
		{
			name: "non_delta_event",
			raw:  `{"type":"message_start","message":{}}`,
			want: nil,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			event := mustUnmarshalEvent(t, tt.raw)
			if diff := cmp.Diff(tt.want, parsePartialStreamEvent(event)); diff != "" {
				t.Fatalf("parsePartialStreamEvent() diff (-want +got):\n%s", diff)
			}
		})
	}
}

func TestReadStreamEvents(t *testing.T) {
	t.Run("nil_stream", func(t *testing.T) {
		t.Parallel()
		seq := readStreamEvents(nil)
		calls := 0
		for resp, err := range seq {
			calls++
			if resp != nil {
				t.Fatalf("expected nil response, got %+v", resp)
			}
			if err == nil || !strings.Contains(err.Error(), "stream is empty") {
				t.Fatalf("unexpected error: %v", err)
			}
		}
		if calls != 1 {
			t.Fatalf("expected one yield, got %d", calls)
		}
	})

	t.Run("yields_partial_and_final_responses", func(t *testing.T) {
		decoder := &mockDecoder{
			events: []ssestream.Event{
				newSSEvent("message_start", `{"type":"message_start","message":{"type":"message","role":"assistant","model":"claude-3","content":[]}}`),
				newSSEvent("content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`),
				newSSEvent("content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}`),
				newSSEvent("content_block_stop", `{"type":"content_block_stop","index":0}`),
				newSSEvent("message_delta", `{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":""},"usage":{"cache_creation_input_tokens":0,"cache_read_input_tokens":0,"input_tokens":0,"output_tokens":1,"server_tool_use":{"web_search_requests":0}}}`),
				newSSEvent("message_stop", `{"type":"message_stop"}`),
			},
		}
		stream := ssestream.NewStream[anthropic.MessageStreamEventUnion](decoder, nil)

		var partialTexts []string
		var finalTexts []string
		for resp, err := range readStreamEvents(stream) {
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if resp == nil {
				continue
			}
			if resp.TurnComplete {
				finalTexts = append(finalTexts, resp.Content.Parts[0].Text)
				if resp.FinishReason != genai.FinishReasonStop {
					t.Fatalf("finish reason = %v, want stop", resp.FinishReason)
				}
			} else {
				partialTexts = append(partialTexts, resp.Content.Parts[0].Text)
			}
		}

		if diff := cmp.Diff([]string{"Hello"}, partialTexts); diff != "" {
			t.Fatalf("partial texts diff (-want +got):\n%s", diff)
		}
		if diff := cmp.Diff([]string{"Hello"}, finalTexts); diff != "" {
			t.Fatalf("final texts diff (-want +got):\n%s", diff)
		}
		if !decoder.closed {
			t.Fatal("stream was not closed")
		}
	})
}

func mustUnmarshalEvent(t *testing.T, raw string) anthropic.MessageStreamEventUnion {
	t.Helper()
	var event anthropic.MessageStreamEventUnion
	if err := json.Unmarshal([]byte(raw), &event); err != nil {
		t.Fatalf("unmarshal event: %v", err)
	}
	return event
}

func mustUnmarshalMessage(t *testing.T, raw string) *anthropic.Message {
	t.Helper()
	var msg anthropic.Message
	if err := json.Unmarshal([]byte(raw), &msg); err != nil {
		t.Fatalf("unmarshal message: %v", err)
	}
	return &msg
}

func newSSEvent(eventType, payload string) ssestream.Event {
	return ssestream.Event{
		Type: eventType,
		Data: []byte(payload),
	}
}
