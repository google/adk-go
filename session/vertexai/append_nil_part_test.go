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

	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
)

// TestAppendEvent_dropsNilParts checks that a nil part, which adk-python's
// Content cannot hold, is left out of the stored event instead of reaching
// the converter, and that the caller's event keeps it.
func TestAppendEvent_dropsNilParts(t *testing.T) {
	fake := &fakeVertexAiSessionService{}
	svc := &vertexAiService{client: newFakeVertexAiClient(t, fake)}
	sess := &localSession{appName: "123", userID: "user1", sessionID: "owned", state: map[string]any{}}
	text := genai.NewPartFromText("x")
	event := &session.Event{
		ID:          "event1",
		Author:      "user",
		LLMResponse: model.LLMResponse{Content: &genai.Content{Role: "user", Parts: []*genai.Part{nil, text}}},
	}

	var err error
	func() {
		defer func() {
			if r := recover(); r != nil {
				t.Fatalf("AppendEvent() panicked: %v", r)
			}
		}()
		err = svc.AppendEvent(t.Context(), sess, event)
	}()

	if err != nil {
		t.Fatalf("AppendEvent() error = %v", err)
	}
	if got := len(fake.events); got != 1 {
		t.Fatalf("AppendEvent RPCs stored %d events, want 1", got)
	}
	parts := fake.events[0].GetContent().GetParts()
	if len(parts) != 1 || parts[0].GetText() != "x" {
		t.Errorf("stored parts = %v, want the one text part \"x\"", parts)
	}
	if got := event.Content.Parts; len(got) != 2 || got[0] != nil || got[1] != text {
		t.Errorf("caller's event parts = %v, want [nil, the text part]", got)
	}
}
