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

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/model"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool/toolconfirmation"
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
