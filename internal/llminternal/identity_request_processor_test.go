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
	"strings"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/internal/utils"
	"google.golang.org/adk/v2/model"
)

func TestIdentityRequestProcessor(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name           string
		agent          agent.Agent
		existingSI     *genai.Content
		wantContains   []string
		wantNotContain []string
		wantNoSI       bool
		wantUnchanged  bool
	}{
		{
			name: "InjectsNameOnly",
			agent: &mockLLMAgent{
				Agent: utils.Must(agent.New(agent.Config{Name: "test_agent"})),
				s:     &State{},
			},
			wantContains: []string{
				"You are an agent.",
				"Your internal name is \"test_agent\".",
			},
			wantNotContain: []string{
				"The description about you is",
			},
		},
		{
			name: "InjectsNameAndDescription",
			agent: &mockLLMAgent{
				Agent: utils.Must(agent.New(agent.Config{
					Name:        "helper_agent",
					Description: "A helpful assistant that answers questions",
				})),
				s: &State{},
			},
			wantContains: []string{
				"You are an agent.",
				"Your internal name is \"helper_agent\".",
				"The description about you is \"A helpful assistant that answers questions\".",
			},
		},
		{
			name:     "NoOpForNonLLMAgent",
			agent:    utils.Must(agent.New(agent.Config{Name: "plain_agent"})),
			wantNoSI: true,
		},
		{
			name: "NoOpForSingleTurnMode",
			agent: &mockLLMAgent{
				Agent: utils.Must(agent.New(agent.Config{Name: "single_turn"})),
				s:     &State{Mode: ModeSingleTurn},
			},
			wantNoSI: true,
		},
		{
			name: "NoOpWhenAllTransfersDisabled",
			agent: &mockLLMAgent{
				Agent: utils.Must(agent.New(agent.Config{Name: "isolated_agent"})),
				s: &State{
					DisallowTransferToParent: true,
					DisallowTransferToPeers:  true,
				},
			},
			wantNoSI: true,
		},
		{
			name: "PreservesInstructionsWhenAllTransfersDisabled",
			agent: &mockLLMAgent{
				Agent: utils.Must(agent.New(agent.Config{
					Name:        "isolated_agent_with_instructions",
					Description: "An isolated assistant.",
				})),
				s: &State{
					DisallowTransferToParent: true,
					DisallowTransferToPeers:  true,
				},
			},
			existingSI:    genai.NewContentFromText("Be concise.", genai.RoleUser),
			wantUnchanged: true,
		},
		{
			name: "InjectsWhenOnlyParentTransferDisabled",
			agent: &mockLLMAgent{
				Agent: utils.Must(agent.New(agent.Config{Name: "peer_agent"})),
				s:     &State{DisallowTransferToParent: true},
			},
			wantContains: []string{"Your internal name is \"peer_agent\"."},
		},
		{
			name: "InjectsWhenOnlyPeerTransferDisabled",
			agent: &mockLLMAgent{
				Agent: utils.Must(agent.New(agent.Config{Name: "parent_agent"})),
				s:     &State{DisallowTransferToPeers: true},
			},
			wantContains: []string{"Your internal name is \"parent_agent\"."},
		},
		{
			name: "InjectsWhenParentAndPeerTransfersDisabledButHasSubAgent",
			agent: &mockLLMAgent{
				Agent: utils.Must(agent.New(agent.Config{
					Name: "coordinator",
					SubAgents: []agent.Agent{
						utils.Must(agent.New(agent.Config{Name: "child_agent"})),
					},
				})),
				s: &State{
					DisallowTransferToParent: true,
					DisallowTransferToPeers:  true,
				},
			},
			wantContains: []string{"Your internal name is \"coordinator\"."},
		},
		{
			name: "EmptyDescription",
			agent: &mockLLMAgent{
				Agent: utils.Must(agent.New(agent.Config{
					Name:        "empty_desc_agent",
					Description: "",
				})),
				s: &State{},
			},
			wantContains: []string{
				"You are an agent.",
				"Your internal name is \"empty_desc_agent\".",
			},
			wantNotContain: []string{
				"The description about you is",
			},
		},
		{
			name: "AppendsToExistingInstructions",
			agent: &mockLLMAgent{
				Agent: utils.Must(agent.New(agent.Config{Name: "append_agent"})),
				s:     &State{},
			},
			existingSI: genai.NewContentFromText("Be concise.", genai.RoleUser),
			wantContains: []string{
				"Be concise.",
				"You are an agent.",
				"Your internal name is \"append_agent\".",
			},
		},
		{
			name: "SkipsForSingleTurnMode",
			agent: &mockLLMAgent{
				Agent: utils.Must(agent.New(agent.Config{
					Name:        "single_turn_agent",
					Description: "A single-shot worker.",
				})),
				s: &State{Mode: ModeSingleTurn},
			},
			wantNoSI: true,
		},
		{
			name: "EmittedForChatMode",
			agent: &mockLLMAgent{
				Agent: utils.Must(agent.New(agent.Config{Name: "chat_agent"})),
				s:     &State{Mode: ModeChat},
			},
			wantContains: []string{
				"You are an agent.",
				"Your internal name is \"chat_agent\".",
			},
		},
		{
			name: "EmittedForTaskMode",
			agent: &mockLLMAgent{
				Agent: utils.Must(agent.New(agent.Config{Name: "task_agent"})),
				s:     &State{Mode: ModeTask},
			},
			wantContains: []string{
				"You are an agent.",
				"Your internal name is \"task_agent\".",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			req := &model.LLMRequest{}
			if tt.existingSI != nil {
				req.Config = &genai.GenerateContentConfig{
					SystemInstruction: tt.existingSI,
				}
			}

			ctx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{
				Agent: tt.agent,
			})

			iter := identityRequestProcessor(ctx, req, &Flow{})

			var eventCount int
			for _, err := range iter {
				if err != nil {
					t.Fatalf("identityRequestProcessor() unexpected error: %v", err)
				}
				eventCount++
			}

			if eventCount > 0 {
				t.Errorf("identityRequestProcessor() yielded %d events, want 0", eventCount)
			}

			if tt.wantNoSI {
				if req.Config != nil && req.Config.SystemInstruction != nil {
					t.Error("identityRequestProcessor() set SystemInstruction, want none")
				}
				return
			}

			if tt.wantUnchanged {
				if req.Config == nil || req.Config.SystemInstruction != tt.existingSI ||
					len(tt.existingSI.Parts) != 1 || tt.existingSI.Parts[0].Text != "Be concise." {
					t.Error("identityRequestProcessor() modified existing instructions")
				}
				return
			}

			if req.Config == nil || req.Config.SystemInstruction == nil {
				t.Fatal("identityRequestProcessor() did not set SystemInstruction")
			}

			si := strings.Join(utils.TextParts(req.Config.SystemInstruction), " ")
			for _, want := range tt.wantContains {
				if !strings.Contains(si, want) {
					t.Errorf("SystemInstruction does not contain %q\ngot: %q", want, si)
				}
			}
			for _, notWant := range tt.wantNotContain {
				if strings.Contains(si, notWant) {
					t.Errorf("SystemInstruction unexpectedly contains %q\ngot: %q", notWant, si)
				}
			}
		})
	}
}
