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

package llminternal

import (
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool/toolconfirmation"
)

type mockAgent struct {
	agent.Agent
	name string
}

func (m *mockAgent) Name() string {
	return m.name
}

type mockInvocationContext struct {
	agent.InvocationContext
	invocationID string
	agentName    string
	branch       string
}

func (m *mockInvocationContext) InvocationID() string {
	return m.invocationID
}

func (m *mockInvocationContext) Agent() agent.Agent {
	return &mockAgent{name: m.agentName}
}

func (m *mockInvocationContext) Branch() string {
	return m.branch
}

func (m *mockInvocationContext) Deadline() (time.Time, bool)     { return time.Time{}, false }
func (m *mockInvocationContext) Done() <-chan struct{}           { return nil }
func (m *mockInvocationContext) Err() error                      { return nil }
func (m *mockInvocationContext) Value(any) any                   { return nil }
func (m *mockInvocationContext) ResumedInput(string) (any, bool) { return nil, false }

func TestGenerateRequestConfirmationEvent(t *testing.T) {
	confirmingFunctionCall := &genai.FunctionCall{
		ID:   "call_1",
		Name: "test_tool",
		Args: map[string]any{"arg": "val"},
	}

	tests := []struct {
		name                  string
		invocationContext     agent.InvocationContext
		functionCallEvent     *session.Event
		functionResponseEvent *session.Event
		wantEvent             *session.Event
	}{
		{
			name: "no confirmation requested",
			invocationContext: &mockInvocationContext{
				invocationID: "inv_1",
				agentName:    "agent_1",
			},
			functionCallEvent: &session.Event{
				LLMResponse: model.LLMResponse{
					Content: &genai.Content{
						Parts: []*genai.Part{
							{FunctionCall: confirmingFunctionCall},
						},
					},
				},
			},
			functionResponseEvent: &session.Event{
				Actions: session.EventActions{
					RequestedToolConfirmations: nil,
				},
			},
			wantEvent: nil,
		},
		{
			name: "confirmation requested but no matching function call",
			invocationContext: &mockInvocationContext{
				invocationID: "inv_1",
				agentName:    "agent_1",
			},
			functionCallEvent: &session.Event{
				LLMResponse: model.LLMResponse{
					Content: &genai.Content{
						Parts: []*genai.Part{
							{FunctionCall: &genai.FunctionCall{ID: "other_call"}},
						},
					},
				},
			},
			functionResponseEvent: &session.Event{
				Actions: session.EventActions{
					RequestedToolConfirmations: map[string]toolconfirmation.ToolConfirmation{
						"call_1": {
							Hint: "Are you sure?",
						},
					},
				},
			},
			wantEvent: nil,
		},
		{
			name: "confirmation requested and matching function call",
			invocationContext: &mockInvocationContext{
				invocationID: "inv_1",
				agentName:    "agent_1",
				branch:       "main",
			},
			functionCallEvent: &session.Event{
				LLMResponse: model.LLMResponse{
					Content: &genai.Content{
						Parts: []*genai.Part{
							{FunctionCall: confirmingFunctionCall},
						},
					},
				},
			},
			functionResponseEvent: &session.Event{
				Actions: session.EventActions{
					RequestedToolConfirmations: map[string]toolconfirmation.ToolConfirmation{
						"call_1": {
							Hint: "Are you sure?",
						},
					},
				},
			},
			wantEvent: &session.Event{
				InvocationID: "inv_1",
				Author:       "agent_1",
				Branch:       "main",
				Actions: session.EventActions{
					StateDelta:    map[string]any{},
					ArtifactDelta: map[string]int64{},
				},
				LLMResponse: model.LLMResponse{
					Content: &genai.Content{
						Role: genai.RoleModel,
						Parts: []*genai.Part{
							{
								FunctionCall: &genai.FunctionCall{
									Name: toolconfirmation.FunctionCallName,
									Args: map[string]any{
										"originalFunctionCall": confirmingFunctionCall,
										"toolConfirmation": toolconfirmation.ToolConfirmation{
											Hint: "Are you sure?",
										},
									},
								},
							},
						},
					},
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := generateRequestConfirmationEvent(tt.invocationContext, tt.functionCallEvent, tt.functionResponseEvent)

			if diff := cmp.Diff(tt.wantEvent, got,
				cmpopts.IgnoreFields(session.Event{}, "Timestamp", "LongRunningToolIDs", "ID"),
				cmpopts.IgnoreFields(genai.FunctionCall{}, "ID"), // Ignore generated IDs
			); diff != "" {
				t.Errorf("generateRequestConfirmationEvent() mismatch (-want +got):\n%s", diff)
			}

			if got != nil {
				for _, s := range got.LongRunningToolIDs {
					if s == "" {
						t.Errorf("empty long running tool id")
					}
				}
			}
		})
	}
}

// TestGenerateRequestConfirmationEventHasID verifies that the event returned
// by generateRequestConfirmationEvent always has a non-empty ID.
//
// In Python ADK, every Event gets a UUID via model_post_init:
//
//	def model_post_init(self, __context):
//	    if not self.id:
//	        self.id = Event.new_id()   # str(uuid.uuid4())
//
// In Go ADK, events must be created with session.NewEvent() to get an ID.
// A raw &session.Event{} literal leaves ID as "" which breaks features
// that rely on event IDs (e.g. time-travel restart_from_event_id).
func TestGenerateRequestConfirmationEventHasID(t *testing.T) {
	confirmingFunctionCall := &genai.FunctionCall{
		ID:   "call_1",
		Name: "test_tool",
		Args: map[string]any{"arg": "val"},
	}

	ctx := &mockInvocationContext{
		invocationID: "inv_1",
		agentName:    "agent_1",
		branch:       "main",
	}

	functionCallEvent := &session.Event{
		LLMResponse: model.LLMResponse{
			Content: &genai.Content{
				Parts: []*genai.Part{
					{FunctionCall: confirmingFunctionCall},
				},
			},
		},
	}

	functionResponseEvent := &session.Event{
		Actions: session.EventActions{
			RequestedToolConfirmations: map[string]toolconfirmation.ToolConfirmation{
				"call_1": {
					Hint: "Are you sure?",
				},
			},
		},
	}

	got := generateRequestConfirmationEvent(ctx, functionCallEvent, functionResponseEvent)
	if got == nil {
		t.Fatal("expected non-nil event")
	}

	if got.ID == "" {
		t.Error("event ID is empty; events must have a UUID for time-travel and session lookup")
	}

	if got.InvocationID != "inv_1" {
		t.Errorf("expected InvocationID=\"inv_1\", got %q", got.InvocationID)
	}
}

func TestGenerateRequestConfirmationEventPreservesThoughtSignature(t *testing.T) {
	thoughtSignature := []byte("test-thought-signature")
	ctx := &mockInvocationContext{
		invocationID: "inv_1",
		agentName:    "agent_1",
	}
	functionCallEvent := &session.Event{
		LLMResponse: model.LLMResponse{
			Content: &genai.Content{
				Parts: []*genai.Part{
					{
						ThoughtSignature: thoughtSignature,
						FunctionCall: &genai.FunctionCall{
							ID:   "call_1",
							Name: "test_tool",
							Args: map[string]any{"arg": "val"},
						},
					},
				},
			},
		},
	}
	functionResponseEvent := &session.Event{
		Actions: session.EventActions{
			RequestedToolConfirmations: map[string]toolconfirmation.ToolConfirmation{
				"call_1": {Hint: "Are you sure?"},
			},
		},
	}

	got := generateRequestConfirmationEvent(ctx, functionCallEvent, functionResponseEvent)
	if got == nil || got.Content == nil || len(got.Content.Parts) != 1 {
		t.Fatalf("expected one confirmation part, got %#v", got)
	}
	if diff := cmp.Diff(thoughtSignature, got.Content.Parts[0].ThoughtSignature); diff != "" {
		t.Errorf("ThoughtSignature mismatch (-want +got):\n%s", diff)
	}
}

// TestGenerateRequestConfirmationEventStableOrder verifies that the emitted
// confirmation parts (and the parallel LongRunningToolIDs slice) are ordered by
// funcID, regardless of the (randomized) Go map iteration order of
// RequestedToolConfirmations, and that repeated calls produce identical order.
func TestGenerateRequestConfirmationEventStableOrder(t *testing.T) {
	ctx := &mockInvocationContext{
		invocationID: "inv_1",
		agentName:    "agent_1",
		branch:       "main",
	}

	// funcIDs whose sorted order is call_a, call_b, call_c, call_d, but which
	// are inserted into the map / parts in a different order.
	sortedFuncIDs := []string{"call_a", "call_b", "call_c", "call_d"}
	insertionOrder := []string{"call_c", "call_a", "call_d", "call_b"}

	functionCallParts := make([]*genai.Part, 0, len(insertionOrder))
	requestedConfirmations := make(map[string]toolconfirmation.ToolConfirmation, len(insertionOrder))
	for _, id := range insertionOrder {
		functionCallParts = append(functionCallParts, &genai.Part{
			FunctionCall: &genai.FunctionCall{
				ID:   id,
				Name: "test_tool",
				Args: map[string]any{"arg": id},
			},
		})
		requestedConfirmations[id] = toolconfirmation.ToolConfirmation{
			Hint: "confirm " + id,
		}
	}

	functionCallEvent := &session.Event{
		LLMResponse: model.LLMResponse{
			Content: &genai.Content{
				Parts: functionCallParts,
			},
		},
	}
	functionResponseEvent := &session.Event{
		Actions: session.EventActions{
			RequestedToolConfirmations: requestedConfirmations,
		},
	}

	// orderOf returns the originating funcIDs in the order the parts were
	// emitted, by reading back the originalFunctionCall arg of each generated
	// adk_request_confirmation call.
	orderOf := func(t *testing.T, ev *session.Event) []string {
		t.Helper()
		if ev == nil || ev.Content == nil {
			t.Fatalf("expected non-nil event with content, got %#v", ev)
		}
		if got, want := len(ev.Content.Parts), len(sortedFuncIDs); got != want {
			t.Fatalf("got %d parts, want %d", got, want)
		}
		if got, want := len(ev.LongRunningToolIDs), len(sortedFuncIDs); got != want {
			t.Fatalf("got %d LongRunningToolIDs, want %d", got, want)
		}
		ids := make([]string, 0, len(ev.Content.Parts))
		for i, part := range ev.Content.Parts {
			if part.FunctionCall == nil {
				t.Fatalf("part %d has nil FunctionCall", i)
			}
			if got, want := part.FunctionCall.Name, toolconfirmation.FunctionCallName; got != want {
				t.Fatalf("part %d FunctionCall.Name = %q, want %q", i, got, want)
			}
			orig, ok := part.FunctionCall.Args["originalFunctionCall"].(*genai.FunctionCall)
			if !ok {
				t.Fatalf("part %d missing originalFunctionCall arg, got %#v", i, part.FunctionCall.Args["originalFunctionCall"])
			}
			ids = append(ids, orig.ID)

			// LongRunningToolIDs must be the generated request-confirmation
			// call ID at the same index (parallel slices).
			if got, want := ev.LongRunningToolIDs[i], part.FunctionCall.ID; got != want {
				t.Errorf("LongRunningToolIDs[%d] = %q, want %q (the request-confirmation call ID)", i, got, want)
			}
		}
		return ids
	}

	first := generateRequestConfirmationEvent(ctx, functionCallEvent, functionResponseEvent)
	gotOrder := orderOf(t, first)
	if diff := cmp.Diff(sortedFuncIDs, gotOrder); diff != "" {
		t.Errorf("parts not in sorted funcID order (-want +got):\n%s", diff)
	}

	// Repeated calls must produce an identical ordering.
	for i := 0; i < 10; i++ {
		next := generateRequestConfirmationEvent(ctx, functionCallEvent, functionResponseEvent)
		if diff := cmp.Diff(gotOrder, orderOf(t, next)); diff != "" {
			t.Errorf("ordering not stable on call %d (-first +next):\n%s", i, diff)
		}
	}
}

func TestGenerateRequestConfirmationEventNoThoughtSignature(t *testing.T) {
	ctx := &mockInvocationContext{
		invocationID: "inv_1",
		agentName:    "agent_1",
	}
	functionCallEvent := &session.Event{
		LLMResponse: model.LLMResponse{
			Content: &genai.Content{
				Parts: []*genai.Part{
					{
						FunctionCall: &genai.FunctionCall{
							ID:   "call_1",
							Name: "test_tool",
							Args: map[string]any{"arg": "val"},
						},
					},
				},
			},
		},
	}
	functionResponseEvent := &session.Event{
		Actions: session.EventActions{
			RequestedToolConfirmations: map[string]toolconfirmation.ToolConfirmation{
				"call_1": {Hint: "Are you sure?"},
			},
		},
	}

	got := generateRequestConfirmationEvent(ctx, functionCallEvent, functionResponseEvent)
	if got == nil || got.Content == nil || len(got.Content.Parts) != 1 {
		t.Fatalf("expected one confirmation part, got %#v", got)
	}
	if len(got.Content.Parts[0].ThoughtSignature) != 0 {
		t.Errorf("ThoughtSignature = %q, want empty", got.Content.Parts[0].ThoughtSignature)
	}
}
