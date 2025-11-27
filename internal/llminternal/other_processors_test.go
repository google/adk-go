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

package llminternal

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	icontext "google.golang.org/adk/internal/context"
	"google.golang.org/adk/internal/utils"
	"google.golang.org/adk/model"
)

func Test_identityRequestProcessor(t *testing.T) {
	testCases := []struct {
		name           string
		agentConfig    agent.Config
		req            *model.LLMRequest
		wantErr        bool
		wantSystemInst string
	}{
		{
			name: "LLM agent with name only - adds name instruction",
			agentConfig: agent.Config{
				Name: "TestAgent",
			},
			req:            &model.LLMRequest{},
			wantErr:        false,
			wantSystemInst: `You are an agent. Your internal name is "TestAgent".`,
		},
		{
			name: "LLM agent with description only - adds description instruction",
			agentConfig: agent.Config{
				Name:        "",
				Description: "A helpful assistant that answers questions",
			},
			req:            &model.LLMRequest{},
			wantErr:        false,
			wantSystemInst: `The description about you is: "A helpful assistant that answers questions".`,
		},
		{
			name: "LLM agent with both name and description - adds both instructions",
			agentConfig: agent.Config{
				Name:        "HelperBot",
				Description: "A friendly assistant that helps users with their tasks",
			},
			req:            &model.LLMRequest{},
			wantErr:        false,
			wantSystemInst: `You are an agent. Your internal name is "HelperBot".` + "\n\n" + `The description about you is: "A friendly assistant that helps users with their tasks".`,
		},
		{
			name: "LLM agent with existing system instruction - appends to existing",
			agentConfig: agent.Config{
				Name:        "ExistingAgent",
				Description: "Agent with existing instructions",
			},
			req: &model.LLMRequest{
				Config: &genai.GenerateContentConfig{
					SystemInstruction: genai.NewContentFromText("Existing system instruction", genai.RoleUser),
				},
			},
			wantErr: false,
			wantSystemInst: "Existing system instruction\n\n" +
				`You are an agent. Your internal name is "ExistingAgent".` + "\n\n" +
				`The description about you is: "Agent with existing instructions".`,
		},
		{
			name: "LLM agent with existing config but no system instruction - creates new",
			agentConfig: agent.Config{
				Name:        "ConfigAgent",
				Description: "Agent with existing config",
			},
			req: &model.LLMRequest{
				Config: &genai.GenerateContentConfig{
					// No SystemInstruction
				},
			},
			wantErr: false,
			wantSystemInst: `You are an agent. Your internal name is "ConfigAgent".` + "\n\n" +
				`The description about you is: "Agent with existing config".`,
		},
		{
			name:           "Non-LLM agent - does nothing and returns no error",
			agentConfig:    agent.Config{},
			req:            &model.LLMRequest{},
			wantErr:        false,
			wantSystemInst: "",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			var testAgent agent.Agent

			// Create a test agent implementing the Agent interface and llmagent state
			// if name or description is empty, we use a nil agent to simulate non-LLM agent.
			if tc.agentConfig.Name != "" || tc.agentConfig.Description != "" {
				testAgent = &struct {
					agent.Agent
					State
				}{
					Agent: utils.Must(agent.New(tc.agentConfig)),
					State: State{},
				}
			}

			// Create real invocation context
			ctx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{
				Agent: testAgent,
			})

			// Call the function under test
			err := identityRequestProcessor(ctx, tc.req)
			if (err != nil) != tc.wantErr {
				t.Fatalf("identityRequestProcessor() error = %v, wantErr %v", err, tc.wantErr)
			}

			// Check the system instruction
			if tc.wantSystemInst != "" {
				if tc.req.Config == nil || tc.req.Config.SystemInstruction == nil {
					t.Errorf("expected system instruction to be set")
					return
				}

				gotText := ""
				for _, part := range tc.req.Config.SystemInstruction.Parts {
					if part.Text != "" {
						if gotText != "" {
							gotText += "\n\n"
						}
						gotText += part.Text
					}
				}

				if diff := cmp.Diff(tc.wantSystemInst, gotText); diff != "" {
					t.Errorf("system instruction mismatch (-want +got):\n%s", diff)
				}
			} else {
				if tc.req.Config != nil && tc.req.Config.SystemInstruction != nil {
					gotText := ""
					for _, part := range tc.req.Config.SystemInstruction.Parts {
						if part.Text != "" {
							if gotText != "" {
								gotText += "\n\n"
							}
							gotText += part.Text
						}
					}
					if gotText != "" {
						t.Errorf("expected no system instruction, got: %s", gotText)
					}
				}
			}
		})
	}
}
