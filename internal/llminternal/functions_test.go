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
	"math"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/internal/utils"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool/toolauth"
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

// TestGenerateRequestConfirmationEventResponseOrder verifies that the emitted
// confirmation parts (and the parallel LongRunningToolIDs slice) follow the
// order of the function calls in the model response (Content.Parts), regardless
// of the (randomized) Go map iteration order of RequestedToolConfirmations, and
// that repeated calls produce an identical order.
func TestGenerateRequestConfirmationEventResponseOrder(t *testing.T) {
	ctx := &mockInvocationContext{
		invocationID: "inv_1",
		agentName:    "agent_1",
		branch:       "main",
	}

	// Confirmations are emitted in the order their function calls appear in the
	// model response (Content.Parts), independent of funcID string order or map
	// insertion order. responseOrder is deliberately not sorted.
	responseOrder := []string{"call_c", "call_a", "call_d", "call_b"}

	functionCallParts := make([]*genai.Part, 0, len(responseOrder))
	requestedConfirmations := make(map[string]toolconfirmation.ToolConfirmation, len(responseOrder))
	for _, id := range responseOrder {
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
		if got, want := len(ev.Content.Parts), len(responseOrder); got != want {
			t.Fatalf("got %d parts, want %d", got, want)
		}
		if got, want := len(ev.LongRunningToolIDs), len(responseOrder); got != want {
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
	if diff := cmp.Diff(responseOrder, gotOrder); diff != "" {
		t.Errorf("parts not in model-response order (-want +got):\n%s", diff)
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

// modelCalls returns a model response event that calls a tool once per ID.
func modelCalls(ids ...string) *session.Event {
	parts := make([]*genai.Part, len(ids))
	for i, id := range ids {
		parts[i] = &genai.Part{FunctionCall: &genai.FunctionCall{ID: id, Name: "calendar"}}
	}
	return &session.Event{LLMResponse: model.LLMResponse{Content: &genai.Content{Role: genai.RoleModel, Parts: parts}}}
}

// apiKeyRequest returns a credential request for an API key stored under key.
func apiKeyRequest(key string) toolauth.Config {
	return toolauth.Config{
		AuthScheme:    toolauth.Scheme{Type: toolauth.SchemeTypeAPIKey, In: "header", Name: "X-Key"},
		CredentialKey: key,
	}
}

func TestGenerateAuthEvent(t *testing.T) {
	ctx := &mockInvocationContext{invocationID: "inv_1", agentName: "agent_1", branch: "main"}
	responses := &session.Event{Actions: session.EventActions{RequestedAuthConfigs: map[string]toolauth.Config{
		"fc-3": apiKeyRequest("key-3"),
		"fc-1": apiKeyRequest("key-1"),
	}}}

	got, err := generateAuthEvent(ctx, modelCalls("fc-1", "fc-2", "fc-3"), responses)
	if err != nil {
		t.Fatalf("generateAuthEvent() failed: %v", err)
	}

	scheme := map[string]any{"type": "apiKey", "in": "header", "name": "X-Key"}
	want := &session.Event{
		InvocationID: "inv_1",
		Author:       "agent_1",
		Branch:       "main",
		Actions:      session.EventActions{StateDelta: map[string]any{}, ArtifactDelta: map[string]int64{}},
		LLMResponse: model.LLMResponse{Content: &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{
			{FunctionCall: &genai.FunctionCall{Name: "adk_request_credential", Args: map[string]any{
				"functionCallId": "fc-1",
				"authConfig":     map[string]any{"authScheme": scheme, "credentialKey": "key-1"},
			}}},
			{FunctionCall: &genai.FunctionCall{Name: "adk_request_credential", Args: map[string]any{
				"functionCallId": "fc-3",
				"authConfig":     map[string]any{"authScheme": scheme, "credentialKey": "key-3"},
			}}},
		}}},
	}
	if diff := cmp.Diff(want, got,
		cmpopts.IgnoreFields(session.Event{}, "ID", "Timestamp", "LongRunningToolIDs"),
		cmpopts.IgnoreFields(genai.FunctionCall{}, "ID"),
	); diff != "" {
		t.Errorf("generateAuthEvent() mismatch (-want +got):\n%s", diff)
	}
	if got.ID == "" {
		t.Error("event ID is empty, want one from session.NewEvent")
	}
	callIDs := []string{got.Content.Parts[0].FunctionCall.ID, got.Content.Parts[1].FunctionCall.ID}
	if callIDs[0] == "" || callIDs[0] == callIDs[1] {
		t.Errorf("request call IDs = %q, want two distinct non-empty IDs", callIDs)
	}
	if diff := cmp.Diff(callIDs, got.LongRunningToolIDs); diff != "" {
		t.Errorf("LongRunningToolIDs mismatch (-want +got):\n%s", diff)
	}
}

func TestGenerateAuthEventOneCallPerCredentialKey(t *testing.T) {
	tests := []struct {
		name     string
		requests map[string]toolauth.Config
		want     []any // functionCallId of each request, in order
	}{
		{
			name:     "same key",
			requests: map[string]toolauth.Config{"fc-1": apiKeyRequest("k"), "fc-2": apiKeyRequest("k")},
			want:     []any{"fc-1"},
		},
		{
			name:     "different keys",
			requests: map[string]toolauth.Config{"fc-1": apiKeyRequest("a"), "fc-2": apiKeyRequest("b")},
			want:     []any{"fc-1", "fc-2"},
		},
		{
			name:     "no keys",
			requests: map[string]toolauth.Config{"fc-1": apiKeyRequest(""), "fc-2": apiKeyRequest("")},
			want:     []any{"fc-1", "fc-2"},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ctx := &mockInvocationContext{invocationID: "inv_1", agentName: "agent_1"}
			responses := &session.Event{Actions: session.EventActions{RequestedAuthConfigs: tc.requests}}

			got, err := generateAuthEvent(ctx, modelCalls("fc-1", "fc-2"), responses)
			if err != nil {
				t.Fatalf("generateAuthEvent() failed: %v", err)
			}

			var gotIDs []any
			for _, call := range utils.FunctionCalls(got.Content) {
				gotIDs = append(gotIDs, call.Args["functionCallId"])
			}
			if diff := cmp.Diff(tc.want, gotIDs); diff != "" {
				t.Errorf("requested function call IDs mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestGenerateAuthEventWithoutRequests(t *testing.T) {
	ctx := &mockInvocationContext{invocationID: "inv_1", agentName: "agent_1"}

	got, err := generateAuthEvent(ctx, modelCalls("fc-1"), &session.Event{})

	if got != nil || err != nil {
		t.Errorf("generateAuthEvent() = %v, %v, want nil, nil", got, err)
	}
}

func TestGenerateAuthEventRequestThatDoesNotEncode(t *testing.T) {
	ctx := &mockInvocationContext{invocationID: "inv_1", agentName: "agent_1"}
	req := apiKeyRequest("k")
	req.Extra = map[string]any{"ratio": math.NaN()}
	responses := &session.Event{Actions: session.EventActions{RequestedAuthConfigs: map[string]toolauth.Config{"fc-1": req}}}

	got, err := generateAuthEvent(ctx, modelCalls("fc-1"), responses)

	if err == nil {
		t.Errorf("generateAuthEvent() = %v, nil, want an error for a request that cannot be encoded", got)
	}
}
