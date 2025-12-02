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

package llmagent

import (
	"reflect"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/model"
	"google.golang.org/adk/session"
)

type MockOutputSchema struct {
	Message    string  `json:"message"`
	Confidence float64 `json:"confidence"`
}

// createTestEvent is a helper to build events for tests.
func createTestEvent(author, contentText string, isFinal bool) *session.Event {
	var parts []*genai.Part
	if contentText != "" {
		parts = append(parts, &genai.Part{Text: contentText})
	}

	var content *genai.Content
	if len(parts) > 0 {
		content = &genai.Content{Role: "model", Parts: parts}
	}

	return &session.Event{
		InvocationID: "test_invocation",
		Author:       author,
		LLMResponse:  model.LLMResponse{Content: content, Partial: !isFinal},
		Actions:      session.EventActions{StateDelta: make(map[string]any)},
	}
}

func TestLlmAgent_MaybeSaveOutputToState(t *testing.T) {
	// Define the structure for our test cases
	testCases := []struct {
		name             string
		agentConfig      Config
		event            *session.Event
		wantStateDelta   map[string]any
		customEventParts []*genai.Part // For multi-part test
	}{
		{
			name:           "skips when event author differs from agentConfig name",
			agentConfig:    Config{Name: "agent_a", OutputKey: "result"},
			event:          createTestEvent("agent_b", "Response from B", true),
			wantStateDelta: map[string]any{},
		},
		{
			name:           "saves when event author matches agentConfig name",
			agentConfig:    Config{Name: "test_agent", OutputKey: "result"},
			event:          createTestEvent("test_agent", "Test response", true),
			wantStateDelta: map[string]any{"result": "Test response"},
		},
		{
			name:           "skips when output_key is not set",
			agentConfig:    Config{Name: "test_agent"}, // No OutputKey
			event:          createTestEvent("test_agent", "Test response", true),
			wantStateDelta: map[string]any{},
		},
		{
			name:           "skips for non-final responses",
			agentConfig:    Config{Name: "test_agent", OutputKey: "result"},
			event:          createTestEvent("test_agent", "*genai.Partial response", false),
			wantStateDelta: map[string]any{},
		},
		{
			name:           "skips when event has no content text",
			agentConfig:    Config{Name: "test_agent", OutputKey: "result"},
			event:          createTestEvent("test_agent", "", true),
			wantStateDelta: map[string]any{},
		},
		{
			name:        "concatenates multiple text parts",
			agentConfig: Config{Name: "test_agent", OutputKey: "result"},
			event:       createTestEvent("test_agent", "", true), // Base event
			customEventParts: []*genai.Part{
				{Text: "Hello "},
				{Text: "world"},
				{Text: "!"},
			},
			wantStateDelta: map[string]any{"result": "Hello world!"},
		},
		{
			name:           "skips on case-sensitive name mismatch",
			agentConfig:    Config{Name: "TestAgent", OutputKey: "result"},
			event:          createTestEvent("testagent", "Test response", true),
			wantStateDelta: map[string]any{},
		},
		// TODO tests with OutputSchema
	}

	// Iterate over the test cases
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// --- Setup for specific cases ---
			if tc.customEventParts != nil {
				tc.event.Content = &genai.Content{Role: "model", Parts: tc.customEventParts}
			}

			// --- Execution ---
			// The method modifies the event in-place, just like the Python version.
			createdAgent, err := New(tc.agentConfig)
			if err != nil {
				t.Fatalf("failed to create agent: %v", err)
			}
			createdLlmAgent, ok := createdAgent.(*llmAgent)
			if !ok {
				t.Fatalf("failed to convert to llmagent")
			}
			createdLlmAgent.maybeSaveOutputToState(tc.event)

			// --- Assertion ---
			gotStateDelta := tc.event.Actions.StateDelta
			if !reflect.DeepEqual(gotStateDelta, tc.wantStateDelta) {
				t.Errorf("stateDelta mismatch:\ngot = %v\nwant = %v", gotStateDelta, tc.wantStateDelta)
			}
		})
	}
}
