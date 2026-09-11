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

	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
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
			name:        "skips function call events",
			agentConfig: Config{Name: "test_agent", OutputKey: "result"},
			event:       createTestEvent("test_agent", "", true),
			customEventParts: []*genai.Part{
				{FunctionCall: &genai.FunctionCall{ID: "call_1", Name: "read_state"}},
			},
			wantStateDelta: map[string]any{},
		},
		{
			name:        "skips function response events",
			agentConfig: Config{Name: "test_agent", OutputKey: "result"},
			event:       createTestEvent("test_agent", "", true),
			customEventParts: []*genai.Part{
				{FunctionResponse: &genai.FunctionResponse{Name: "read_state", Response: map[string]any{"result": "SECRET_42"}}},
			},
			wantStateDelta: map[string]any{},
		},
		{
			name:           "skips when event has no content text",
			agentConfig:    Config{Name: "test_agent", OutputKey: "result"},
			event:          createTestEvent("test_agent", "", true),
			wantStateDelta: map[string]any{},
		},
		{
			name:        "skips thought-only text",
			agentConfig: Config{Name: "test_agent", OutputKey: "result"},
			event:       createTestEvent("test_agent", "", true),
			customEventParts: []*genai.Part{
				{Text: "hidden thought", Thought: true},
			},
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

			// Output is stamped by the node wrapper, not here (adk-python
			// __maybe_save_output_to_state parity).
			if tc.event.Output != nil {
				t.Errorf("event.Output = %v, want nil (only state_delta may be written here)", tc.event.Output)
			}
		})
	}
}

// newSessionWithEvent returns an in-memory session.Session preloaded
// with a single user-authored event.
func newSessionWithEvent(t *testing.T, text string) session.Session {
	t.Helper()
	svc := session.InMemoryService()
	createResp, err := svc.Create(t.Context(), &session.CreateRequest{
		AppName: "app", UserID: "u", SessionID: "s",
	})
	if err != nil {
		t.Fatalf("session.Create: %v", err)
	}
	ev := session.NewEvent(t.Context(), "inv-existing")
	ev.Author = "user"
	ev.LLMResponse = model.LLMResponse{Content: &genai.Content{
		Role:  genai.RoleUser,
		Parts: []*genai.Part{genai.NewPartFromText(text)},
	}}
	if err := svc.AppendEvent(t.Context(), createResp.Session, ev); err != nil {
		t.Fatalf("AppendEvent: %v", err)
	}
	getResp, err := svc.Get(t.Context(), &session.GetRequest{
		AppName: "app", UserID: "u", SessionID: "s",
	})
	if err != nil {
		t.Fatalf("session.Get: %v", err)
	}
	return getResp.Session
}

func seedEvent(t *testing.T, text string) *session.Event {
	t.Helper()
	ev := session.NewEvent(t.Context(), "inv-seed")
	ev.Author = "user"
	ev.LLMResponse = model.LLMResponse{Content: &genai.Content{
		Role:  genai.RoleUser,
		Parts: []*genai.Part{genai.NewPartFromText(text)},
	}}
	return ev
}

// TestWrappedSession_SeedNotPersisted asserts the single_turn
// node-input contract: the seed is visible through the wrapped view but
// never written to the underlying session history.
func TestWrappedSession_SeedNotPersisted(t *testing.T) {
	t.Parallel()

	base := newSessionWithEvent(t, "existing turn")
	baseLen := base.Events().Len()
	seed := seedEvent(t, "transient node input")
	wrapped := newWrappedSession(base, seed)

	if got, want := wrapped.Events().Len(), baseLen+1; got != want {
		t.Errorf("wrapped.Events().Len() = %d, want %d", got, want)
	}
	if got := wrapped.Events().At(wrapped.Events().Len() - 1); got != seed {
		t.Errorf("last wrapped event = %v, want the seed", got)
	}

	if got := base.Events().Len(); got != baseLen {
		t.Errorf("underlying session length = %d, want %d; seed must not persist", got, baseLen)
	}
	for ev := range base.Events().All() {
		if ev == seed {
			t.Fatal("seed leaked into the underlying session history")
		}
	}
}
