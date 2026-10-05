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
	"strings"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/artifact"
	artifactinternal "google.golang.org/adk/v2/internal/artifact"
	icontext "google.golang.org/adk/v2/internal/context"
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

func newArtifactInvocationContext(t *testing.T, svc artifact.Service) agent.InvocationContext {
	t.Helper()
	var artifacts agent.Artifacts
	if svc != nil {
		artifacts = &artifactinternal.Artifacts{
			Service:   svc,
			AppName:   "app",
			UserID:    "u",
			SessionID: "s",
		}
	}
	return icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{
		Artifacts: artifacts,
	})
}

func TestLlmAgent_MaybeSaveOutputToArtifact(t *testing.T) {
	t.Parallel()

	t.Run("saves final text as artifact and replaces event body with reference", func(t *testing.T) {
		t.Parallel()
		svc := artifact.InMemoryService()
		ic := newArtifactInvocationContext(t, svc)
		created, err := New(Config{
			Name:           "writer",
			OutputArtifact: "design.md",
			OutputKey:      "latest_design",
		})
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		a := created.(*llmAgent)

		ev := createTestEvent("writer", "", true)
		ev.Content = &genai.Content{
			Role: genai.RoleModel,
			Parts: []*genai.Part{
				{Text: "internal reasoning", Thought: true},
				{Text: "# Design v1\n"},
				{Text: "Body text"},
			},
		}

		if err := a.maybeSaveOutput(ic, ev); err != nil {
			t.Fatalf("maybeSaveOutput: %v", err)
		}
		if got, want := ev.Actions.ArtifactDelta["design.md"], int64(1); got != want {
			t.Errorf("ArtifactDelta[\"design.md\"] = %d, want %d", got, want)
		}
		if got, want := ev.Actions.StateDelta["latest_design"], any("# Design v1\nBody text"); got != want {
			t.Errorf("StateDelta[\"latest_design\"] = %v, want %v", got, want)
		}
		if len(ev.Content.Parts) != 2 {
			t.Fatalf("len(ev.Content.Parts) = %d, want 2 (thought + reference)", len(ev.Content.Parts))
		}
		if !ev.Content.Parts[0].Thought || ev.Content.Parts[0].Text != "internal reasoning" {
			t.Errorf("Parts[0] = %+v, want preserved thought part", ev.Content.Parts[0])
		}
		if got, want := ev.Content.Parts[1].Text, `Saved artifact "design.md" (version 1).`; got != want {
			t.Errorf("Parts[1].Text = %q, want %q", got, want)
		}

		loaded, err := svc.Load(t.Context(), &artifact.LoadRequest{
			AppName: "app", UserID: "u", SessionID: "s", FileName: "design.md", Version: 1,
		})
		if err != nil {
			t.Fatalf("svc.Load: %v", err)
		}
		if got, want := loaded.Part.Text, "# Design v1\nBody text"; got != want {
			t.Errorf("loaded artifact text = %q, want %q", got, want)
		}
	})

	t.Run("skips non-final, foreign-author, tool, and thought-only events", func(t *testing.T) {
		t.Parallel()
		cases := []struct {
			name  string
			event *session.Event
		}{
			{
				name:  "different author",
				event: createTestEvent("other", "body", true),
			},
			{
				name:  "partial event",
				event: createTestEvent("writer", "body", false),
			},
			{
				name: "function call with text and LongRunningToolIDs",
				event: &session.Event{
					Author:             "writer",
					LongRunningToolIDs: []string{"fc-1"},
					LLMResponse: model.LLMResponse{
						Content: &genai.Content{
							Role: genai.RoleModel,
							Parts: []*genai.Part{
								{Text: "calling tool"},
								{FunctionCall: &genai.FunctionCall{ID: "fc-1", Name: "slow"}},
							},
						},
					},
				},
			},
			{
				name: "function response with SkipSummarization and text",
				event: &session.Event{
					Author: "writer",
					Actions: session.EventActions{
						SkipSummarization: true,
					},
					LLMResponse: model.LLMResponse{
						Content: &genai.Content{
							Role: genai.RoleUser,
							Parts: []*genai.Part{
								{FunctionResponse: &genai.FunctionResponse{ID: "fc-1", Name: "sub", Response: map[string]any{"result": "ok"}}},
								{Text: "ok"},
							},
						},
					},
				},
			},
			{
				name: "thought-only event",
				event: &session.Event{
					Author: "writer",
					LLMResponse: model.LLMResponse{
						Content: &genai.Content{
							Role:  genai.RoleModel,
							Parts: []*genai.Part{{Text: "thinking", Thought: true}},
						},
					},
				},
			},
		}

		for _, tc := range cases {
			t.Run(tc.name, func(t *testing.T) {
				t.Parallel()
				svc := artifact.InMemoryService()
				ic := newArtifactInvocationContext(t, svc)
				created, err := New(Config{Name: "writer", OutputArtifact: "design.md"})
				if err != nil {
					t.Fatalf("New: %v", err)
				}
				a := created.(*llmAgent)
				if err := a.maybeSaveOutput(ic, tc.event); err != nil {
					t.Fatalf("maybeSaveOutput: %v", err)
				}
				if len(tc.event.Actions.ArtifactDelta) != 0 {
					t.Errorf("ArtifactDelta = %v, want empty", tc.event.Actions.ArtifactDelta)
				}
				listed, err := svc.List(t.Context(), &artifact.ListRequest{AppName: "app", UserID: "u", SessionID: "s"})
				if err != nil {
					t.Fatalf("svc.List: %v", err)
				}
				if len(listed.FileNames) != 0 {
					t.Errorf("saved artifacts = %v, want none", listed.FileNames)
				}
			})
		}
	})

	t.Run("errors when artifact service is not configured", func(t *testing.T) {
		t.Parallel()
		ic := newArtifactInvocationContext(t, nil)
		created, err := New(Config{Name: "writer", OutputArtifact: "design.md"})
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		a := created.(*llmAgent)
		ev := createTestEvent("writer", "content", true)
		if err := a.maybeSaveOutput(ic, ev); err == nil {
			t.Fatal("maybeSaveOutput succeeded without an artifact service, want error")
		}
	})

	t.Run("validates OutputSchema before saving artifact", func(t *testing.T) {
		t.Parallel()
		svc := artifact.InMemoryService()
		ic := newArtifactInvocationContext(t, svc)
		schema := &genai.Schema{
			Type: genai.TypeObject,
			Properties: map[string]*genai.Schema{
				"title": {Type: genai.TypeString},
			},
			Required: []string{"title"},
		}
		created, err := New(Config{
			Name:           "writer",
			Mode:           ModeSingleTurn,
			OutputArtifact: "spec.json",
			OutputKey:      "spec",
			OutputSchema:   schema,
		})
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		a := created.(*llmAgent)

		badEv := createTestEvent("writer", `{"wrong": "SECRET_PAYLOAD_99"}`, true)
		err = a.maybeSaveOutput(ic, badEv)
		if err == nil {
			t.Fatal("maybeSaveOutput succeeded on invalid schema output, want error")
		}
		if strings.Contains(err.Error(), "SECRET_PAYLOAD_99") {
			t.Errorf("validation error leaked model output: %v", err)
		}
		listed, err := svc.List(t.Context(), &artifact.ListRequest{AppName: "app", UserID: "u", SessionID: "s"})
		if err != nil {
			t.Fatalf("svc.List: %v", err)
		}
		if len(listed.FileNames) != 0 {
			t.Fatalf("invalid output was saved to artifacts: %v", listed.FileNames)
		}

		goodEv := createTestEvent("writer", `{"title":"v1"}`, true)
		if err := a.maybeSaveOutput(ic, goodEv); err != nil {
			t.Fatalf("maybeSaveOutput on valid JSON: %v", err)
		}
		if goodEv.Output != nil {
			t.Errorf("goodEv.Output after maybeSaveOutput = %#v, want nil", goodEv.Output)
		}
		if err := ProcessLLMAgentOutput(created, goodEv); err != nil {
			t.Fatalf("ProcessLLMAgentOutput after OutputArtifact save: %v", err)
		}
		gotState, ok := goodEv.Actions.StateDelta["spec"].(map[string]any)
		if !ok || gotState["title"] != "v1" {
			t.Errorf("StateDelta[\"spec\"] = %#v, want parsed map with title=v1", goodEv.Actions.StateDelta["spec"])
		}
		if got, want := goodEv.Output, any(`Saved artifact "spec.json" (version 1).`); got != want {
			t.Errorf("goodEv.Output = %#v, want %#v", got, want)
		}
	})

	t.Run("skips conversational text turns in task mode", func(t *testing.T) {
		t.Parallel()
		svc := artifact.InMemoryService()
		ic := newArtifactInvocationContext(t, svc)
		created, err := New(Config{
			Name:           "tasker",
			Mode:           ModeTask,
			OutputArtifact: "report.md",
		})
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		a := created.(*llmAgent)
		ev := createTestEvent("tasker", "Which format do you prefer?", true)
		if err := a.maybeSaveOutput(ic, ev); err != nil {
			t.Fatalf("maybeSaveOutput: %v", err)
		}
		if len(ev.Actions.ArtifactDelta) != 0 {
			t.Errorf("ArtifactDelta = %v, want empty in task mode", ev.Actions.ArtifactDelta)
		}
	})
}
