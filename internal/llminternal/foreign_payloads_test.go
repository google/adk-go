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

package llminternal_test

import (
	"encoding/json"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/internal/llminternal"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
)

func TestConvertForeignEvent_RetainsPayloadParts(t *testing.T) {
	t.Parallel()

	for _, tc := range []struct {
		name string
		part *genai.Part
	}{
		{"inline data", &genai.Part{InlineData: &genai.Blob{MIMEType: "image/png", Data: []byte{1}}}},
		{"file data", &genai.Part{FileData: &genai.FileData{FileURI: "gs://test/image"}}},
		{"executable code", &genai.Part{ExecutableCode: &genai.ExecutableCode{Code: "1 + 1"}}},
		{"code result", &genai.Part{CodeExecutionResult: &genai.CodeExecutionResult{Output: "2"}}},
		{"server tool call", &genai.Part{ToolCall: &genai.ToolCall{ID: "call", Args: map[string]any{"input": "test"}}}},
		{"server tool response", &genai.Part{ToolResponse: &genai.ToolResponse{ID: "call", Response: map[string]any{"output": "test"}}}},
		{"audio transcription", &genai.Part{AudioTranscription: &genai.Transcription{Text: "spoken answer"}}},
	} {
		for _, signed := range []bool{false, true} {
			for _, metadata := range []bool{false, true} {
				name := "unsigned " + tc.name
				if signed {
					name = "signed " + tc.name
				}
				if metadata {
					name += " with metadata"
				}
				t.Run(name, func(t *testing.T) {
					part := *tc.part
					if metadata {
						part.VideoMetadata = &genai.VideoMetadata{EndOffset: time.Second}
						part.MediaResolution = &genai.PartMediaResolution{Level: "MEDIA_RESOLUTION_HIGH"}
						part.MediaProcessing = genai.MediaProcessingStatic
						part.SpeechMetadata = &genai.SpeechMetadata{Speaker: "speaker", Style: "calm"}
					}
					if signed {
						part.ThoughtSignature = []byte{3, 4}
					}
					event := &session.Event{
						Author: "otherAgent",
						LLMResponse: model.LLMResponse{Content: &genai.Content{
							Role: "model", Parts: []*genai.Part{
								{},
								{ThoughtSignature: []byte{1, 2}},
								{PartMetadata: map[string]any{"source": "test"}},
								{VideoMetadata: &genai.VideoMetadata{EndOffset: time.Second}},
								{MediaResolution: &genai.PartMediaResolution{Level: "MEDIA_RESOLUTION_HIGH"}},
								{MediaProcessing: genai.MediaProcessingStatic},
								{SpeechMetadata: &genai.SpeechMetadata{Speaker: "speaker", Style: "calm"}},
								&part,
							},
						}},
					}
					before, err := json.Marshal(event)
					if err != nil {
						t.Fatal("could not snapshot the stored event")
					}
					got := llminternal.ConvertForeignEvent(event)
					if got == nil || got.Content == nil {
						t.Fatal("foreign payload was discarded")
					}
					want := []*genai.Part{{Text: "For context:"}, &part}
					if !cmp.Equal(got.Content.Parts, want) {
						t.Error("foreign conversion did not retain only the payload and its attached metadata")
					}
					after, err := json.Marshal(event)
					if err != nil {
						t.Fatal("could not snapshot the stored event after conversion")
					}
					if !cmp.Equal(before, after) {
						t.Error("foreign conversion modified the stored event")
					}
				})
			}
		}
	}
}

func TestConvertForeignEvent_DropsThoughtPayloadParts(t *testing.T) {
	t.Parallel()

	for _, tc := range []struct {
		name string
		part *genai.Part
	}{
		{"inline data", &genai.Part{InlineData: &genai.Blob{MIMEType: "image/png", Data: []byte{1}}}},
		{"file data", &genai.Part{FileData: &genai.FileData{FileURI: "gs://test/image"}}},
		{"executable code", &genai.Part{ExecutableCode: &genai.ExecutableCode{Code: "1 + 1"}}},
		{"code result", &genai.Part{CodeExecutionResult: &genai.CodeExecutionResult{Output: "2"}}},
		{"server tool call", &genai.Part{ToolCall: &genai.ToolCall{ID: "call", Args: map[string]any{"input": "test"}}}},
		{"server tool response", &genai.Part{ToolResponse: &genai.ToolResponse{ID: "call", Response: map[string]any{"output": "test"}}}},
		{"audio transcription", &genai.Part{AudioTranscription: &genai.Transcription{Text: "spoken answer"}}},
	} {
		for _, text := range []string{"", "private reasoning"} {
			for _, mixed := range []bool{false, true} {
				name := tc.name + " without text"
				if text != "" {
					name = tc.name + " with text"
				}
				if mixed {
					name += " beside public payload"
				}
				t.Run(name, func(t *testing.T) {
					thought := *tc.part
					thought.Thought = true
					thought.Text = text
					parts := []*genai.Part{&thought}
					var want []*genai.Part
					if mixed {
						public := *tc.part
						parts = append(parts, &genai.Part{Text: "public answer"}, &public)
						want = []*genai.Part{{Text: "For context:"}, {Text: "[otherAgent] said: public answer"}, &public}
					}
					event := &session.Event{
						Author: "otherAgent",
						LLMResponse: model.LLMResponse{Content: &genai.Content{
							Role: "model", Parts: parts,
						}},
					}
					before, err := json.Marshal(event)
					if err != nil {
						t.Fatal("could not snapshot the stored event")
					}
					got := llminternal.ConvertForeignEvent(event)
					if !mixed {
						if got != nil {
							t.Error("foreign thought-only payload was relayed")
						}
					} else if got == nil || got.Content == nil || !cmp.Equal(got.Content.Parts, want) {
						t.Error("foreign conversion did not retain only the public payloads")
					}
					after, err := json.Marshal(event)
					if err != nil {
						t.Fatal("could not snapshot the stored event after conversion")
					}
					if !cmp.Equal(before, after) {
						t.Error("foreign conversion modified the stored event")
					}
				})
			}
		}
	}
}
