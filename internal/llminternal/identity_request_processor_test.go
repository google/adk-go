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

	"google.golang.org/adk/agent"
	icontext "google.golang.org/adk/internal/context"
	"google.golang.org/adk/internal/utils"
	"google.golang.org/adk/model"
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
					t.Errorf("identityRequestProcessor() set SystemInstruction on non-LLM agent, got: %v", req.Config.SystemInstruction)
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
