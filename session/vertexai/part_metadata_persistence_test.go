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

package vertexai

import (
	"testing"

	aiplatformpb "cloud.google.com/go/aiplatform/apiv1beta1/aiplatformpb"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
)

// A part's PartMetadata has no SessionEvent column: aiplatformpb.Part
// carries only the part data, the thought flag and the thought signature.
// anthropicmodel records the original block type there for a thinking block
// with empty text (key "anthropic_content_block_type"), because the part
// alone cannot be told apart from a redacted_thinking block. raw_event is
// therefore the only slot that can hold it, and eventNeedsRawEvent is what
// decides whether raw_event gets written at all.
//
// When it did not list PartMetadata the key reached no slot: the part
// reloaded without it and the next request replayed the block as
// redacted_thinking instead of thinking.
//
// These run offline because the replay-based suite needs a recording that
// only a live Agent Engine project can produce.
func TestEventNeedsRawEventForPartMetadata(t *testing.T) {
	t.Parallel()

	thoughtPart := func() *genai.Part {
		return &genai.Part{
			Thought:          true,
			ThoughtSignature: []byte("sig_empty_thinking"),
			PartMetadata:     map[string]any{"anthropic_content_block_type": "thinking"},
		}
	}
	eventWithPart := func(part *genai.Part) *session.Event {
		return &session.Event{
			ID:          "evt-1",
			Author:      "model",
			LLMResponse: model.LLMResponse{Content: &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{part}}},
		}
	}

	if !eventNeedsRawEvent(eventWithPart(thoughtPart())) {
		t.Error("eventNeedsRawEvent() = false for a part carrying PartMetadata, so it would be written nowhere and lost on reload")
	}

	// A thought part without metadata keeps the legacy wire format, so
	// existing replay recordings remain valid.
	bare := thoughtPart()
	bare.PartMetadata = nil
	if eventNeedsRawEvent(eventWithPart(bare)) {
		t.Error("eventNeedsRawEvent() = true for a thought part without PartMetadata, which would change the wire format for unrelated events")
	}
}

func TestPartMetadataRoundTripsThroughRawEvent(t *testing.T) {
	t.Parallel()

	want := &session.Event{
		ID:     "evt-1",
		Author: "model",
		LLMResponse: model.LLMResponse{
			Content: &genai.Content{
				Role: genai.RoleModel,
				Parts: []*genai.Part{
					{
						Thought:          true,
						ThoughtSignature: []byte("sig_empty_thinking"),
						PartMetadata:     map[string]any{"anthropic_content_block_type": "thinking"},
					},
					{Text: "done"},
				},
			},
		},
	}

	raw, err := eventToRawEvent(want)
	if err != nil {
		t.Fatalf("eventToRawEvent() error = %v", err)
	}
	got, err := eventFromRawEvent(raw)
	if err != nil {
		t.Fatalf("eventFromRawEvent() error = %v", err)
	}

	if got.Content == nil || len(got.Content.Parts) != 2 {
		t.Fatalf("Content = %v, want 2 parts", got.Content)
	}
	part := got.Content.Parts[0]
	if !part.Thought || string(part.ThoughtSignature) != "sig_empty_thinking" {
		t.Errorf("part = %+v, want the signed thought part", part)
	}
	if got, want := part.PartMetadata["anthropic_content_block_type"], "thinking"; got != want {
		t.Errorf("PartMetadata[anthropic_content_block_type] = %v, want %v", got, want)
	}
}

// TestPartMetadataDroppedByLegacyContentColumns pins why the raw_event gate
// above exists: the structured content columns cannot carry PartMetadata.
// If a future aiplatformpb.Part grows a field for it and the conversions
// learn to use it, this test fails and the gate can go.
func TestPartMetadataDroppedByLegacyContentColumns(t *testing.T) {
	t.Parallel()

	event := &session.Event{
		Author: "model",
		LLMResponse: model.LLMResponse{
			Content: &genai.Content{
				Role: genai.RoleModel,
				Parts: []*genai.Part{
					{
						Thought:          true,
						ThoughtSignature: []byte("sig_empty_thinking"),
						PartMetadata:     map[string]any{"anthropic_content_block_type": "thinking"},
					},
				},
			},
		},
	}

	content, err := createAiplatformpbContent(event)
	if err != nil {
		t.Fatalf("createAiplatformpbContent() error = %v", err)
	}
	got := aiplatformToGenaiContent(&aiplatformpb.SessionEvent{Content: content})
	if got == nil || len(got.Parts) != 1 {
		t.Fatalf("aiplatformToGenaiContent() = %v, want 1 part", got)
	}
	if len(got.Parts[0].PartMetadata) != 0 {
		t.Errorf("PartMetadata = %v after the legacy column round trip, want it dropped (that is the gap raw_event covers)", got.Parts[0].PartMetadata)
	}
}
