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
	"context"
	"encoding/json"
	"iter"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/model"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/functiontool"
	"google.golang.org/adk/tool/toolconfirmation"
)

type SecureActionArgs struct {
	ActionName string `json:"action_name"`
}

type SecureActionResult struct {
	Executed bool `json:"executed"`
}

func secureActionFunc(ctx tool.Context, input SecureActionArgs) (SecureActionResult, error) {
	return SecureActionResult{Executed: true}, nil
}

type hitlMockModel struct {
	model.LLM
	Calls int
}

func (m *hitlMockModel) Name() string {
	return "hitl-mock-model"
}

func (m *hitlMockModel) GenerateContent(ctx context.Context, req *model.LLMRequest, useStream bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		m.Calls++
		if m.Calls > 1 {
			// Turn 2 model execution (after user confirms)
			yield(&model.LLMResponse{
				Content: &genai.Content{
					Parts: []*genai.Part{
						genai.NewPartFromText("Successfully processed your request after confirmation."),
					},
					Role: "model",
				},
				Partial: false,
			}, nil)
			return
		}

		// Turn 1: yields 2 parallel function calls requiring confirmation
		yield(&model.LLMResponse{
			Content: &genai.Content{
				Parts: []*genai.Part{
					{
						FunctionCall: &genai.FunctionCall{
							ID:   "secure_call_1",
							Name: "secure_action",
							Args: map[string]any{"action_name": "deploy_prod"},
						},
					},
					{
						FunctionCall: &genai.FunctionCall{
							ID:   "secure_call_2",
							Name: "secure_action",
							Args: map[string]any{"action_name": "delete_db"},
						},
					},
				},
				Role: "model",
			},
			Partial: false,
		}, nil)
	}
}

// TestParallelFunctionCallsWithHITL verifies the end-to-end coordination of multiple parallel
// tool executions that require Human-in-the-Loop (HITL) confirmation.
//
// The test simulates a two-turn interaction:
//  1. Turn 1: The model initiates two parallel tool calls to a sensitive tool. Because no user
//     confirmation is yet present, both tools trigger RequestConfirmation. This sets the
//     SkipSummarization flag, which halts LLM generation immediately after tool responses are returned.
//     The runner yields:
//     a) A model response event containing the two original parallel function calls.
//     b) An aggregated tool confirmation event containing two adk_request_confirmation wrapper calls.
//     c) A merged function response event containing the placeholder (unexecuted) tool responses.
//  2. Turn 2: The client simulates user confirmation by returning confirmation responses matching
//     the unique wrapper call IDs. The RequestConfirmationRequestProcessor detects these, matches
//     them back to the original tools, and concurrently executes the sensitive tools in parallel
//     with the confirmed flag enabled. Finally, the model is called to generate a summary.
func TestParallelFunctionCallsWithHITL(t *testing.T) {
	secureTool, err := functiontool.New(functiontool.Config{
		Name:                "secure_action",
		Description:         "performs a sensitive/secure action",
		RequireConfirmation: true,
	}, secureActionFunc)
	if err != nil {
		t.Fatal(err)
	}

	mockModel := &hitlMockModel{}

	a, err := llmagent.New(llmagent.Config{
		Name:        "hitl_tester",
		Description: "HITL tester agent",
		Instruction: "You are a secure helper.",
		Model:       mockModel,
		Tools: []tool.Tool{
			secureTool,
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	sessionService := session.InMemoryService()
	_, err = sessionService.Create(t.Context(), &session.CreateRequest{
		AppName:   "testApp",
		UserID:    "testUser",
		SessionID: "testSession",
	})
	if err != nil {
		t.Fatal(err)
	}

	r, err := runner.New(runner.Config{
		Agent:          a,
		SessionService: sessionService,
		AppName:        "testApp",
	})
	if err != nil {
		t.Fatal(err)
	}

	// --- TURN 1: Requesting parallel tools requiring confirmation ---
	it := r.Run(t.Context(), "testUser", "testSession", &genai.Content{
		Parts: []*genai.Part{
			genai.NewPartFromText("Please run prod deploy and db delete"),
		},
		Role: "user",
	}, agent.RunConfig{StreamingMode: agent.StreamingModeSSE})

	var turn1Events []*session.Event
	for ev, err := range it {
		if err != nil {
			t.Fatal(err)
		}
		turn1Events = append(turn1Events, ev)
	}

	// Expecting exactly 3 events:
	// 1. ModelResponseEvent (yielding the two function calls secure_call_1 and secure_call_2)
	// 2. ToolConfirmationEvent (yielding two adk_request_confirmation calls)
	// 3. Merged function response event (returning placeholder executed: false responses)
	if len(turn1Events) != 3 {
		t.Fatalf("Expected 3 events in Turn 1, got %d", len(turn1Events))
	}

	// Verify that the model event contains our parallel calls
	modelEvent := turn1Events[0]
	if len(modelEvent.Content.Parts) != 2 {
		t.Errorf("Expected model event to contain 2 parts (function calls), got %d", len(modelEvent.Content.Parts))
	}

	// Verify that the tool confirmation event has the confirmation wrapper calls
	confirmEvent := turn1Events[1]
	if len(confirmEvent.Content.Parts) != 2 {
		t.Errorf("Expected tool confirmation event to contain 2 wrapper function calls, got %d", len(confirmEvent.Content.Parts))
	}

	var confirmCallID1, confirmCallID2 string
	for _, p := range confirmEvent.Content.Parts {
		if p.FunctionCall == nil || p.FunctionCall.Name != toolconfirmation.FunctionCallName {
			t.Errorf("Expected function call name %s, got %v", toolconfirmation.FunctionCallName, p.FunctionCall)
			continue
		}
		origCall, err := toolconfirmation.OriginalCallFrom(p.FunctionCall)
		if err != nil {
			t.Fatalf("Failed to extract original call: %v", err)
		}
		if origCall.ID == "secure_call_1" {
			confirmCallID1 = p.FunctionCall.ID
		} else if origCall.ID == "secure_call_2" {
			confirmCallID2 = p.FunctionCall.ID
		}
	}

	if confirmCallID1 == "" || confirmCallID2 == "" {
		t.Fatalf("Failed to retrieve both confirmation function call IDs from turn 1 events")
	}

	// Verify that the merged function response event contains the placeholder unexecuted responses
	placeholderRespEvent := turn1Events[2]
	if len(placeholderRespEvent.Content.Parts) != 2 {
		t.Errorf("Expected placeholder function response event to contain 2 parts, got %d", len(placeholderRespEvent.Content.Parts))
	}

	expectedPlaceholders := []*genai.Part{
		{
			FunctionResponse: &genai.FunctionResponse{
				Name:     "secure_action",
				ID:       "secure_call_1",
				Response: map[string]any{"error": `error tool "secure_action" requires confirmation, please approve or reject`},
			},
		},
		{
			FunctionResponse: &genai.FunctionResponse{
				Name:     "secure_action",
				ID:       "secure_call_2",
				Response: map[string]any{"error": `error tool "secure_action" requires confirmation, please approve or reject`},
			},
		},
	}

	if diff := cmp.Diff(expectedPlaceholders, placeholderRespEvent.Content.Parts); diff != "" {
		t.Errorf("Mismatch in placeholder tool responses (-want +got):\n%s", diff)
	}

	// --- TURN 2: User confirms the actions ---
	// Build user's response to confirmation requests.
	userConfirmation := toolconfirmation.ToolConfirmation{Confirmed: true}
	userConfirmationJSON, _ := json.Marshal(userConfirmation)
	userConfirmationResponse := map[string]any{
		"response": string(userConfirmationJSON),
	}

	// Run runner again, passing the user confirmation response content directly as the message
	it2 := r.Run(t.Context(), "testUser", "testSession", &genai.Content{
		Role: "user",
		Parts: []*genai.Part{
			{
				FunctionResponse: &genai.FunctionResponse{
					Name:     toolconfirmation.FunctionCallName,
					ID:       confirmCallID1,
					Response: userConfirmationResponse,
				},
			},
			{
				FunctionResponse: &genai.FunctionResponse{
					Name:     toolconfirmation.FunctionCallName,
					ID:       confirmCallID2,
					Response: userConfirmationResponse,
				},
			},
		},
	}, agent.RunConfig{StreamingMode: agent.StreamingModeSSE})

	var turn2Events []*session.Event
	for ev, err := range it2 {
		if err != nil {
			t.Fatal(err)
		}
		turn2Events = append(turn2Events, ev)
	}

	// Turn 2 Events Expectation:
	// 1. Function response event containing executed: true for both calls
	// 2. Final summarizing model response ("Successfully processed your request after confirmation.")
	if len(turn2Events) != 2 {
		t.Fatalf("Expected 2 events in Turn 2, got %d", len(turn2Events))
	}

	funcRespEvent := turn2Events[0]
	if len(funcRespEvent.Content.Parts) != 2 {
		t.Errorf("Expected 2 function responses in Turn 2, got %d", len(funcRespEvent.Content.Parts))
	}

	ignoreFields := []cmp.Option{
		cmpopts.IgnoreFields(genai.FunctionResponse{}, "ID"),
	}

	expectedResponses := []*genai.Part{
		{
			FunctionResponse: &genai.FunctionResponse{
				Name:     "secure_action",
				Response: map[string]any{"executed": true},
			},
		},
		{
			FunctionResponse: &genai.FunctionResponse{
				Name:     "secure_action",
				Response: map[string]any{"executed": true},
			},
		},
	}

	if diff := cmp.Diff(expectedResponses, funcRespEvent.Content.Parts, ignoreFields...); diff != "" {
		t.Errorf("Mismatch in executed tool responses (-want +got):\n%s", diff)
	}

	finalModelEvent := turn2Events[1]
	if finalModelEvent.Content.Parts[0].Text != "Successfully processed your request after confirmation." {
		t.Errorf("Expected summarizing model event, got: %v", finalModelEvent.Content.Parts[0])
	}
}

// TestParallelFunctionCallsWithPartialHITL verifies the behavior when multiple parallel
// tool executions are requested, but only one of them is confirmed by the user while the
// other is rejected/denied.
func TestParallelFunctionCallsWithPartialHITL(t *testing.T) {
	secureTool, err := functiontool.New(functiontool.Config{
		Name:                "secure_action",
		Description:         "performs a sensitive/secure action",
		RequireConfirmation: true,
	}, secureActionFunc)
	if err != nil {
		t.Fatal(err)
	}

	mockModel := &hitlMockModel{}

	a, err := llmagent.New(llmagent.Config{
		Name:        "hitl_tester",
		Description: "HITL tester agent",
		Instruction: "You are a secure helper.",
		Model:       mockModel,
		Tools: []tool.Tool{
			secureTool,
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	sessionService := session.InMemoryService()
	_, err = sessionService.Create(t.Context(), &session.CreateRequest{
		AppName:   "testApp",
		UserID:    "testUser",
		SessionID: "testSession",
	})
	if err != nil {
		t.Fatal(err)
	}

	r, err := runner.New(runner.Config{
		Agent:          a,
		SessionService: sessionService,
		AppName:        "testApp",
	})
	if err != nil {
		t.Fatal(err)
	}

	// --- TURN 1: Requesting parallel tools requiring confirmation ---
	it := r.Run(t.Context(), "testUser", "testSession", &genai.Content{
		Parts: []*genai.Part{
			genai.NewPartFromText("Please run prod deploy and db delete"),
		},
		Role: "user",
	}, agent.RunConfig{StreamingMode: agent.StreamingModeSSE})

	var turn1Events []*session.Event
	for ev, err := range it {
		if err != nil {
			t.Fatal(err)
		}
		turn1Events = append(turn1Events, ev)
	}

	if len(turn1Events) != 3 {
		t.Fatalf("Expected 3 events in Turn 1, got %d", len(turn1Events))
	}

	confirmEvent := turn1Events[1]
	var confirmCallID1, confirmCallID2 string
	for _, p := range confirmEvent.Content.Parts {
		origCall, err := toolconfirmation.OriginalCallFrom(p.FunctionCall)
		if err != nil {
			t.Fatalf("Failed to extract original call: %v", err)
		}
		if origCall.ID == "secure_call_1" {
			confirmCallID1 = p.FunctionCall.ID
		} else if origCall.ID == "secure_call_2" {
			confirmCallID2 = p.FunctionCall.ID
		}
	}

	if confirmCallID1 == "" || confirmCallID2 == "" {
		t.Fatalf("Failed to retrieve both confirmation function call IDs from turn 1 events")
	}

	// --- TURN 2: User partially confirms the actions ---
	// Only confirm secure_call_2 (secure_call_1 remains pending/unconfirmed in this turn).
	confirmedJSON, _ := json.Marshal(toolconfirmation.ToolConfirmation{Confirmed: true})

	it2 := r.Run(t.Context(), "testUser", "testSession", &genai.Content{
		Role: "user",
		Parts: []*genai.Part{
			{
				FunctionResponse: &genai.FunctionResponse{
					Name:     toolconfirmation.FunctionCallName,
					ID:       confirmCallID2,
					Response: map[string]any{"response": string(confirmedJSON)},
				},
			},
		},
	}, agent.RunConfig{StreamingMode: agent.StreamingModeSSE})

	var turn2Events []*session.Event
	for ev, err := range it2 {
		if err != nil {
			t.Fatal(err)
		}
		turn2Events = append(turn2Events, ev)
	}

	if len(turn2Events) != 2 {
		t.Fatalf("Expected 2 events in Turn 2, got %d", len(turn2Events))
	}

	funcRespEvent := turn2Events[0]
	if len(funcRespEvent.Content.Parts) != 1 {
		t.Errorf("Expected 1 function response in Turn 2, got %d", len(funcRespEvent.Content.Parts))
	}

	ignoreFields := []cmp.Option{
		cmpopts.IgnoreFields(genai.FunctionResponse{}, "ID"),
	}

	expectedResponses := []*genai.Part{
		{
			FunctionResponse: &genai.FunctionResponse{
				Name:     "secure_action",
				Response: map[string]any{"executed": true},
			},
		},
	}

	if diff := cmp.Diff(expectedResponses, funcRespEvent.Content.Parts, ignoreFields...); diff != "" {
		t.Errorf("Mismatch in executed tool responses (-want +got):\n%s", diff)
	}

	finalModelEvent := turn2Events[1]
	if finalModelEvent.Content.Parts[0].Text != "Successfully processed your request after confirmation." {
		t.Errorf("Expected summarizing model event, got: %v", finalModelEvent.Content.Parts[0])
	}
}
