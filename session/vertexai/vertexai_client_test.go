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

package vertexai

import (
	"testing"

	aiplatformpb "cloud.google.com/go/aiplatform/apiv1beta1/aiplatformpb"
	"google.golang.org/genai"

	"google.golang.org/adk/model"
	"google.golang.org/adk/session"
)

// TestFunctionCallRoundTrip verifies that FunctionCall.ID and
// FunctionResponse.ID survive a serialise/deserialise round trip through
// the aiplatformpb proto. ADK's LLMAgent depends on these IDs to pair
// FunctionCall events with their FunctionResponse counterparts; if they
// are dropped on read the pairing silently breaks and tool outputs cannot
// be associated with the call that produced them.
func TestFunctionCallRoundTrip(t *testing.T) {
	event := &session.Event{
		LLMResponse: model.LLMResponse{
			Content: &genai.Content{
				Role: string(genai.RoleModel),
				Parts: []*genai.Part{
					{
						FunctionCall: &genai.FunctionCall{
							ID:   "adk-abc-123",
							Name: "lua_sandbox",
							Args: map[string]any{"code": "return 1"},
						},
					},
					{
						FunctionResponse: &genai.FunctionResponse{
							ID:       "adk-abc-123",
							Name:     "lua_sandbox",
							Response: map[string]any{"output": "1"},
						},
					},
				},
			},
		},
	}

	pbContent, err := createAiplatformpbContent(event)
	if err != nil {
		t.Fatalf("createAiplatformpbContent: %v", err)
	}

	roundTripped := aiplatformToGenaiContent(&aiplatformpb.SessionEvent{Content: pbContent})
	if roundTripped == nil {
		t.Fatalf("aiplatformToGenaiContent returned nil content")
	}
	if got, want := len(roundTripped.Parts), 2; got != want {
		t.Fatalf("part count: got %d, want %d", got, want)
	}

	fc := roundTripped.Parts[0].FunctionCall
	if fc == nil {
		t.Fatalf("expected FunctionCall on parts[0], got %+v", roundTripped.Parts[0])
	}
	if got, want := fc.ID, "adk-abc-123"; got != want {
		t.Fatalf("FunctionCall.ID after round trip: got %q, want %q", got, want)
	}
	if got, want := fc.Name, "lua_sandbox"; got != want {
		t.Fatalf("FunctionCall.Name after round trip: got %q, want %q", got, want)
	}

	fr := roundTripped.Parts[1].FunctionResponse
	if fr == nil {
		t.Fatalf("expected FunctionResponse on parts[1], got %+v", roundTripped.Parts[1])
	}
	if got, want := fr.ID, "adk-abc-123"; got != want {
		t.Fatalf("FunctionResponse.ID after round trip: got %q, want %q", got, want)
	}
	if got, want := fr.Name, "lua_sandbox"; got != want {
		t.Fatalf("FunctionResponse.Name after round trip: got %q, want %q", got, want)
	}
}
