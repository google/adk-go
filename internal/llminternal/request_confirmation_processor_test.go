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

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"google.golang.org/genai"
	"google.golang.org/protobuf/testing/protocmp"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/internal/llminternal"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/toolconfirmation"
)

const (
	mockToolName                   = "mock_tool"
	mockFunctionCallID             = "mock_function_call_id"
	mockConfirmationFunctionCallID = "mock_confirmation_function_call_id"
)

type mockTool struct {
	name string
}

func (m *mockTool) Name() string        { return m.name }
func (m *mockTool) Description() string { return "mock tool" }
func (m *mockTool) IsLongRunning() bool { return false }
func (m *mockTool) Declaration() *genai.FunctionDeclaration {
	return &genai.FunctionDeclaration{Name: m.name}
}

func (m *mockTool) Run(ctx agent.Context, args any) (map[string]any, error) {
	if ctx.ToolConfirmation() == nil || !ctx.ToolConfirmation().Confirmed {
		return map[string]any{"error": string("Tool execution not confirmed")}, nil
	}
	return map[string]any{"result": "Mock tool result with test"}, nil
}

func newMockLlmAgent() (agent.Agent, []tool.Tool, error) {
	testModel := &testModel{}
	tools := []tool.Tool{
		&mockTool{name: "mock_tool"},
	}
	agnt, err := llmagent.New(llmagent.Config{
		Name:  "testAgent",
		Model: testModel,
		Tools: tools,
	})
	return agnt, tools, err
}

func createInvocationContext(t *testing.T, agnt agent.Agent, sess session.Session) agent.InvocationContext {
	t.Helper()
	ctx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{
		Agent:   agnt,
		Session: sess,
	})
	return ctx
}

func TestRequestConfirmationRequestProcessor(t *testing.T) {
	// 1. Setup shared data and helpers used across test cases
	originalFunctionCall := &genai.FunctionCall{
		Name: mockToolName,
		Args: map[string]any{"param1": "test"},
		ID:   mockFunctionCallID,
	}

	originalCallMap := map[string]any{
		"name": originalFunctionCall.Name,
		"args": originalFunctionCall.Args,
		"id":   originalFunctionCall.ID,
	}

	// Helper to create input events for the "confirmation" scenarios
	createConfirmationEvents := func(confirmed bool) []*session.Event {
		toolConfirmation := toolconfirmation.ToolConfirmation{Confirmed: false, Hint: "test hint"}
		toolConfirmationArgs := map[string]any{
			"originalFunctionCall": originalCallMap,
			"toolConfirmation":     toolConfirmation,
		}

		userConfirmation := toolconfirmation.ToolConfirmation{Confirmed: confirmed}
		userConfirmationJSON, _ := json.Marshal(userConfirmation) // Ignoring err for brevity in test setup helpers
		userConfirmationResponse := map[string]any{
			"response": string(userConfirmationJSON),
		}

		return []*session.Event{
			{
				Author: "agent",
				LLMResponse: model.LLMResponse{
					Content: &genai.Content{
						Parts: []*genai.Part{
							{
								FunctionCall: &genai.FunctionCall{
									Name: toolconfirmation.FunctionCallName,
									Args: toolConfirmationArgs,
									ID:   mockConfirmationFunctionCallID,
								},
							},
						},
					},
				},
			},
			{
				Author: "user",
				LLMResponse: model.LLMResponse{
					Content: &genai.Content{
						Parts: []*genai.Part{
							{
								FunctionResponse: &genai.FunctionResponse{
									Name:     toolconfirmation.FunctionCallName,
									ID:       mockConfirmationFunctionCallID,
									Response: userConfirmationResponse,
								},
							},
						},
					},
				},
			},
		}
	}

	// 2. Define the test cases
	tests := []struct {
		name       string
		events     []*session.Event
		wantEvents []*session.Event
	}{
		{
			name:       "NoEvents",
			events:     nil,
			wantEvents: nil,
		},
		{
			name: "NoFunctionResponses",
			events: []*session.Event{
				{
					Author: "user",
					LLMResponse: model.LLMResponse{
						Content: &genai.Content{},
					},
				},
			},
			wantEvents: nil,
		},
		{
			name: "NoConfirmationFunctionResponse",
			events: []*session.Event{
				{
					Author: "user",
					LLMResponse: model.LLMResponse{
						Content: &genai.Content{
							Parts: []*genai.Part{
								{
									FunctionResponse: &genai.FunctionResponse{
										Name:     "other_function",
										Response: map[string]any{},
									},
								},
							},
						},
					},
				},
			},
			wantEvents: nil,
		},
		{
			name:   "Success",
			events: createConfirmationEvents(true),
			wantEvents: []*session.Event{
				{
					Author: "testAgent",
					LLMResponse: model.LLMResponse{
						Content: &genai.Content{
							Parts: []*genai.Part{
								{
									FunctionResponse: &genai.FunctionResponse{
										Name:     mockToolName,
										ID:       mockFunctionCallID,
										Response: map[string]any{"result": "Mock tool result with test"},
									},
								},
							},
							Role: "user",
						},
					},
				},
			},
		},
		{
			name:   "ToolNotConfirmed",
			events: createConfirmationEvents(false),
			wantEvents: []*session.Event{
				{
					Author: "testAgent",
					LLMResponse: model.LLMResponse{
						Content: &genai.Content{
							Parts: []*genai.Part{
								{
									FunctionResponse: &genai.FunctionResponse{
										Name:     mockToolName,
										ID:       mockFunctionCallID,
										Response: map[string]any{"error": "Tool execution not confirmed"},
									},
								},
							},
							Role: "user",
						},
					},
				},
			},
		},
	}

	// 3. Execution Loop
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			agnt, tools, err := newMockLlmAgent()
			if err != nil {
				t.Fatalf("error creating mock llmagent: %v", err)
			}

			invocationContext := createInvocationContext(t, agnt, &fakeSession{
				events: tt.events,
			})
			llmRequest := &model.LLMRequest{}

			iter := llminternal.RequestConfirmationRequestProcessor(invocationContext, llmRequest, &llminternal.Flow{Tools: tools})

			var gotEvents []*session.Event
			for event, err := range iter {
				if err != nil {
					t.Fatalf("RequestConfirmationRequestProcessor() unexpected error: %v", err)
				}
				gotEvents = append(gotEvents, event)
			}

			// Validate Count
			if len(gotEvents) != len(tt.wantEvents) {
				t.Errorf("RequestConfirmationRequestProcessor() got %d events, want %d", len(gotEvents), len(tt.wantEvents))
				return
			}

			// Validate Content (only if we expected events)
			if len(tt.wantEvents) > 0 {
				ignoreFields := []cmp.Option{
					protocmp.Transform(),
					cmpopts.IgnoreFields(session.Event{}, "ID"),
					cmpopts.IgnoreFields(session.Event{}, "Timestamp"),
					cmpopts.IgnoreFields(session.Event{}, "InvocationID"),
					cmpopts.IgnoreFields(session.EventActions{}, "StateDelta", "ArtifactDelta"),
				}

				if diff := cmp.Diff(tt.wantEvents, gotEvents, ignoreFields...); diff != "" {
					t.Errorf("RequestConfirmationRequestProcessor() event diff (-want +got):\n%s", diff)
				}
			}
		})
	}
}

// TestRequestConfirmationResumeOrderIsStable verifies that when a user resumes
// several tool confirmations in a single message, the confirmed tool calls are
// re-dispatched in the order their confirmation requests appear in the
// confirmation-request event, deterministically across runs. The IDs are
// deliberately out of sorted order so a sort-based implementation would also
// fail the test.
func TestRequestConfirmationResumeOrderIsStable(t *testing.T) {
	// Request order intentionally differs from sorted order.
	wantOrder := []string{"call_c3", "call_c1", "call_c5", "call_c2", "call_c6", "call_c4"}

	confirmationCallID := func(originalCallID string) string {
		return "confirmation_" + originalCallID
	}

	// One agent event carrying all six confirmation requests.
	var confirmationParts []*genai.Part
	for _, callID := range wantOrder {
		confirmationParts = append(confirmationParts, &genai.Part{
			FunctionCall: &genai.FunctionCall{
				Name: toolconfirmation.FunctionCallName,
				ID:   confirmationCallID(callID),
				Args: map[string]any{
					"originalFunctionCall": map[string]any{
						"name": mockToolName,
						"args": map[string]any{"param1": "test"},
						"id":   callID,
					},
					"toolConfirmation": toolconfirmation.ToolConfirmation{Confirmed: false, Hint: "test hint"},
				},
			},
		})
	}

	// One user event approving all six confirmations in the same message. The
	// responses are assembled in a DIFFERENT order than the requests
	// (responseOrder != wantOrder). The processor keys confirmations by ID (a
	// map lookup) and derives re-dispatch order solely from the
	// confirmation-request event, so the output must track request order
	// regardless of the order the responses arrive in. Scrambling here makes the
	// test fail an implementation that (incorrectly) ordered by response arrival.
	// responseOrder is wantOrder reversed.
	responseOrder := []string{"call_c4", "call_c6", "call_c2", "call_c5", "call_c1", "call_c3"}
	userConfirmationJSON, err := json.Marshal(toolconfirmation.ToolConfirmation{Confirmed: true})
	if err != nil {
		t.Fatalf("error marshalling user confirmation: %v", err)
	}
	var responseParts []*genai.Part
	for _, callID := range responseOrder {
		responseParts = append(responseParts, &genai.Part{
			FunctionResponse: &genai.FunctionResponse{
				Name:     toolconfirmation.FunctionCallName,
				ID:       confirmationCallID(callID),
				Response: map[string]any{"response": string(userConfirmationJSON)},
			},
		})
	}

	events := []*session.Event{
		{
			Author: "agent",
			LLMResponse: model.LLMResponse{
				Content: &genai.Content{Parts: confirmationParts},
			},
		},
		{
			Author: "user",
			LLMResponse: model.LLMResponse{
				Content: &genai.Content{Parts: responseParts},
			},
		},
	}

	// Go randomizes map iteration order, so repeat the run to catch any
	// map-order dependence in the re-dispatch path. Both constants are
	// load-bearing. Go's small-map iteration randomizes a start offset within a
	// bucket group, so the reachable orders are rotations of insertion order, not
	// arbitrary permutations: at six entries the identity (already-sorted) order
	// comes up about 3/8 of the time, so an unordered implementation still
	// diverges the other ~5/8, and 10 iterations shrinks the blind spot to
	// roughly 1 in 18,000. At two tools the identity order would come up ~7/8 per
	// iteration and this same 10-iteration test would pass on the *unfixed* code
	// about 26% of the time — so do not trim this to two calls or a single
	// iteration.
	for run := 0; run < 10; run++ {
		agnt, tools, err := newMockLlmAgent()
		if err != nil {
			t.Fatalf("error creating mock llmagent: %v", err)
		}

		invocationContext := createInvocationContext(t, agnt, &fakeSession{events: events})
		llmRequest := &model.LLMRequest{}

		iter := llminternal.RequestConfirmationRequestProcessor(invocationContext, llmRequest, &llminternal.Flow{Tools: tools})

		var gotEvents []*session.Event
		for event, err := range iter {
			if err != nil {
				t.Fatalf("run %d: RequestConfirmationRequestProcessor() unexpected error: %v", run, err)
			}
			gotEvents = append(gotEvents, event)
		}

		if len(gotEvents) != 1 {
			t.Fatalf("run %d: RequestConfirmationRequestProcessor() got %d events, want 1", run, len(gotEvents))
		}
		if gotEvents[0] == nil || gotEvents[0].Content == nil {
			t.Fatalf("run %d: re-dispatch event or its content is nil", run)
		}

		var gotOrder []string
		for _, part := range gotEvents[0].Content.Parts {
			if part.FunctionResponse != nil {
				gotOrder = append(gotOrder, part.FunctionResponse.ID)
			}
		}
		if diff := cmp.Diff(wantOrder, gotOrder); diff != "" {
			t.Errorf("run %d: re-dispatched function call order diff (-want +got):\n%s", run, diff)
		}
	}
}

// TestRequestConfirmationResumeSkipsAlreadyAnsweredCalls exercises the
// "already answered" skip branch: when the user confirms several tool calls in
// one message but one of those calls has already been answered (a
// FunctionResponse for it exists in a later event), that call must NOT be
// re-dispatched, while the still-pending calls are re-dispatched in request
// order. Without the skip branch this would double-run (or panic on) the
// already-answered call.
func TestRequestConfirmationResumeSkipsAlreadyAnsweredCalls(t *testing.T) {
	// Request order is intentionally unsorted; "call_a" is the one that has
	// already been answered and must be skipped, leaving "call_b" and "call_c".
	requestOrder := []string{"call_b", "call_a", "call_c"}
	const alreadyAnswered = "call_a"
	wantOrder := []string{"call_b", "call_c"}

	confirmationCallID := func(originalCallID string) string {
		return "confirmation_" + originalCallID
	}

	// One agent event carrying all three confirmation requests.
	var confirmationParts []*genai.Part
	for _, callID := range requestOrder {
		confirmationParts = append(confirmationParts, &genai.Part{
			FunctionCall: &genai.FunctionCall{
				Name: toolconfirmation.FunctionCallName,
				ID:   confirmationCallID(callID),
				Args: map[string]any{
					"originalFunctionCall": map[string]any{
						"name": mockToolName,
						"args": map[string]any{"param1": "test"},
						"id":   callID,
					},
					"toolConfirmation": toolconfirmation.ToolConfirmation{Confirmed: false, Hint: "test hint"},
				},
			},
		})
	}

	// One user event approving all three confirmations in the same message.
	userConfirmationJSON, err := json.Marshal(toolconfirmation.ToolConfirmation{Confirmed: true})
	if err != nil {
		t.Fatalf("error marshalling user confirmation: %v", err)
	}
	var responseParts []*genai.Part
	for _, callID := range requestOrder {
		responseParts = append(responseParts, &genai.Part{
			FunctionResponse: &genai.FunctionResponse{
				Name:     toolconfirmation.FunctionCallName,
				ID:       confirmationCallID(callID),
				Response: map[string]any{"response": string(userConfirmationJSON)},
			},
		})
	}

	events := []*session.Event{
		{
			Author: "agent",
			LLMResponse: model.LLMResponse{
				Content: &genai.Content{Parts: confirmationParts},
			},
		},
		{
			Author: "user",
			LLMResponse: model.LLMResponse{
				Content: &genai.Content{Parts: responseParts},
			},
		},
		// A later, agent-authored event that already carries the result for
		// "call_a". Author is not "user", so it is not mistaken for the
		// confirmation-response event, but the delete pass still removes call_a
		// from the resume set — driving the skip branch on re-dispatch.
		{
			Author: "testAgent",
			LLMResponse: model.LLMResponse{
				Content: &genai.Content{
					Parts: []*genai.Part{
						{
							FunctionResponse: &genai.FunctionResponse{
								Name:     mockToolName,
								ID:       alreadyAnswered,
								Response: map[string]any{"result": "already answered"},
							},
						},
					},
				},
			},
		},
	}

	agnt, tools, err := newMockLlmAgent()
	if err != nil {
		t.Fatalf("error creating mock llmagent: %v", err)
	}

	invocationContext := createInvocationContext(t, agnt, &fakeSession{events: events})
	llmRequest := &model.LLMRequest{}

	iter := llminternal.RequestConfirmationRequestProcessor(invocationContext, llmRequest, &llminternal.Flow{Tools: tools})

	var gotEvents []*session.Event
	for event, err := range iter {
		if err != nil {
			t.Fatalf("RequestConfirmationRequestProcessor() unexpected error: %v", err)
		}
		gotEvents = append(gotEvents, event)
	}

	if len(gotEvents) != 1 {
		t.Fatalf("RequestConfirmationRequestProcessor() got %d events, want 1", len(gotEvents))
	}
	if gotEvents[0] == nil || gotEvents[0].Content == nil {
		t.Fatalf("re-dispatch event or its content is nil")
	}

	var gotOrder []string
	for _, part := range gotEvents[0].Content.Parts {
		if part.FunctionResponse != nil {
			gotOrder = append(gotOrder, part.FunctionResponse.ID)
		}
	}
	if diff := cmp.Diff(wantOrder, gotOrder); diff != "" {
		t.Errorf("re-dispatched function call order diff (-want +got):\n%s", diff)
	}
}

// TestRequestConfirmationResumeDedupesDuplicateOriginalID pins the "seen"
// guard: when a single confirmation event carries two confirmation requests
// that resolve to the same originalFunctionCall.ID, the resumed tool must be
// dispatched exactly once, not once per confirmation. Without the guard the ID
// would be appended to resumeOrder twice and the tool would run twice within
// one re-dispatch.
func TestRequestConfirmationResumeDedupesDuplicateOriginalID(t *testing.T) {
	const duplicateCallID = "call_dup"

	// Two distinct confirmation requests (distinct confirmation IDs) that both
	// point at the same original function call.
	confirmationIDs := []string{"confirmation_a", "confirmation_b"}

	var confirmationParts []*genai.Part
	for _, confID := range confirmationIDs {
		confirmationParts = append(confirmationParts, &genai.Part{
			FunctionCall: &genai.FunctionCall{
				Name: toolconfirmation.FunctionCallName,
				ID:   confID,
				Args: map[string]any{
					"originalFunctionCall": map[string]any{
						"name": mockToolName,
						"args": map[string]any{"param1": "test"},
						"id":   duplicateCallID,
					},
					"toolConfirmation": toolconfirmation.ToolConfirmation{Confirmed: false, Hint: "test hint"},
				},
			},
		})
	}

	// One user event approving both confirmations.
	userConfirmationJSON, err := json.Marshal(toolconfirmation.ToolConfirmation{Confirmed: true})
	if err != nil {
		t.Fatalf("error marshalling user confirmation: %v", err)
	}
	var responseParts []*genai.Part
	for _, confID := range confirmationIDs {
		responseParts = append(responseParts, &genai.Part{
			FunctionResponse: &genai.FunctionResponse{
				Name:     toolconfirmation.FunctionCallName,
				ID:       confID,
				Response: map[string]any{"response": string(userConfirmationJSON)},
			},
		})
	}

	events := []*session.Event{
		{
			Author: "agent",
			LLMResponse: model.LLMResponse{
				Content: &genai.Content{Parts: confirmationParts},
			},
		},
		{
			Author: "user",
			LLMResponse: model.LLMResponse{
				Content: &genai.Content{Parts: responseParts},
			},
		},
	}

	agnt, tools, err := newMockLlmAgent()
	if err != nil {
		t.Fatalf("error creating mock llmagent: %v", err)
	}

	invocationContext := createInvocationContext(t, agnt, &fakeSession{events: events})
	llmRequest := &model.LLMRequest{}

	iter := llminternal.RequestConfirmationRequestProcessor(invocationContext, llmRequest, &llminternal.Flow{Tools: tools})

	var gotEvents []*session.Event
	for event, err := range iter {
		if err != nil {
			t.Fatalf("RequestConfirmationRequestProcessor() unexpected error: %v", err)
		}
		gotEvents = append(gotEvents, event)
	}

	if len(gotEvents) != 1 {
		t.Fatalf("RequestConfirmationRequestProcessor() got %d events, want 1", len(gotEvents))
	}
	if gotEvents[0] == nil || gotEvents[0].Content == nil {
		t.Fatalf("re-dispatch event or its content is nil")
	}

	var gotOrder []string
	for _, part := range gotEvents[0].Content.Parts {
		if part.FunctionResponse != nil {
			gotOrder = append(gotOrder, part.FunctionResponse.ID)
		}
	}
	if diff := cmp.Diff([]string{duplicateCallID}, gotOrder); diff != "" {
		t.Errorf("expected the duplicated original call to be dispatched exactly once (-want +got):\n%s", diff)
	}
}
