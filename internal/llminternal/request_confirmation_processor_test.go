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
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"google.golang.org/genai"
	"google.golang.org/protobuf/testing/protocmp"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	icontext "google.golang.org/adk/internal/context"
	"google.golang.org/adk/internal/llminternal"
	"google.golang.org/adk/internal/toolinternal"
	"google.golang.org/adk/model"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/toolconfirmation"
)

const (
	mockToolName                   = "mock_tool"
	mockFunctionCallID             = "mock_function_call_id"
	mockConfirmationFunctionCallID = "mock_confirmation_function_call_id"
)

type mockTool struct {
	name string
}

var _ toolinternal.FunctionTool = (*mockTool)(nil)

func (m *mockTool) Name() string        { return m.name }
func (m *mockTool) Description() string { return "mock tool" }
func (m *mockTool) IsLongRunning() bool { return false }
func (m *mockTool) Declaration() *genai.FunctionDeclaration {
	return &genai.FunctionDeclaration{Name: m.name}
}

func (m *mockTool) Run(ctx tool.Context, args any) (map[string]any, error) {
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
	ctx := icontext.NewInvocationContext(context.Background(), icontext.InvocationContextParams{
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
					cmpopts.IgnoreFields(session.EventActions{}, "StateDelta"),
				}

				if diff := cmp.Diff(tt.wantEvents, gotEvents, ignoreFields...); diff != "" {
					t.Errorf("RequestConfirmationRequestProcessor() event diff (-want +got):\n%s", diff)
				}
			}
		})
	}
}
