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
	"strings"
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
		agent          agent.Agent
		req            *model.LLMRequest
		wantErr        bool
		wantSystemInst string
	}{
		{
			name: "LLM agent with name only - adds name instruction",
			agent: &struct {
				agent.Agent
				State
			}{
				Agent: utils.Must(agent.New(agent.Config{Name: "TestAgent"})),
				State: State{},
			},
			req:            &model.LLMRequest{},
			wantErr:        false,
			wantSystemInst: `You are an agent. Your internal name is "TestAgent".`,
		},
		{
			name: "LLM agent with description only - adds description instruction",
			agent: &struct {
				agent.Agent
				State
			}{
				Agent: utils.Must(agent.New(agent.Config{Description: "A helpful assistant that answers questions"})),
				State: State{},
			},
			req:            &model.LLMRequest{},
			wantErr:        false,
			wantSystemInst: `The description about you is "A helpful assistant that answers questions".`,
		},
		{
			name: "LLM agent with both name and description - adds both instructions",
			agent: &struct {
				agent.Agent
				State
			}{
				Agent: utils.Must(agent.New(agent.Config{
					Name:        "HelperBot",
					Description: "A friendly assistant that helps users with their tasks",
				})),
				State: State{},
			},
			req:            &model.LLMRequest{},
			wantErr:        false,
			wantSystemInst: `You are an agent. Your internal name is "HelperBot".` + "\n\n" + `The description about you is "A friendly assistant that helps users with their tasks".`,
		},
		{
			name: "LLM agent with special characters in name and description",
			agent: &struct {
				agent.Agent
				State
			}{
				Agent: utils.Must(agent.New(agent.Config{
					Name:        `"Test" Agent`,
					Description: `A "helpful" assistant`,
				})),
				State: State{},
			},
			req:            &model.LLMRequest{},
			wantErr:        false,
			wantSystemInst: `You are an agent. Your internal name is "\"Test\" Agent".` + "\n\n" + `The description about you is "A \"helpful\" assistant".`,
		},
		{
			name: "LLM agent with existing system instruction - appends to existing",
			agent: &struct {
				agent.Agent
				State
			}{
				Agent: utils.Must(agent.New(agent.Config{
					Name:        "ExistingAgent",
					Description: "Agent with existing instructions",
				})),
				State: State{},
			},
			req: &model.LLMRequest{
				Config: &genai.GenerateContentConfig{
					SystemInstruction: genai.NewContentFromText("Existing system instruction", genai.RoleUser),
				},
			},
			wantErr: false,
			wantSystemInst: "Existing system instruction\n\n" +
				`You are an agent. Your internal name is "ExistingAgent".` + "\n\n" +
				`The description about you is "Agent with existing instructions".`,
		},
		{
			name: "LLM agent with existing config but no system instruction - creates new",
			agent: &struct {
				agent.Agent
				State
			}{
				Agent: utils.Must(agent.New(agent.Config{
					Name:        "ConfigAgent",
					Description: "Agent with existing config",
				})),
				State: State{},
			},
			req: &model.LLMRequest{
				Config: &genai.GenerateContentConfig{
					// No SystemInstruction
				},
			},
			wantErr: false,
			wantSystemInst: `You are an agent. Your internal name is "ConfigAgent".` + "\n\n" +
				`The description about you is "Agent with existing config".`,
		},
		{
			name:           "Non-LLM agent with name - does nothing and returns no error",
			agent:          utils.Must(agent.New(agent.Config{Name: "NonLLMAgent"})),
			req:            &model.LLMRequest{},
			wantErr:        false,
			wantSystemInst: "",
		},
		{
			name:           "Nil agent - does nothing and returns no error",
			agent:          nil,
			req:            &model.LLMRequest{},
			wantErr:        false,
			wantSystemInst: "",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Create real invocation context
			ctx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{
				Agent: tc.agent,
			})

			// Call the function under test
			err := identityRequestProcessor(ctx, tc.req)
			if (err != nil) != tc.wantErr {
				t.Fatalf("identityRequestProcessor() error = %v, wantErr %v", err, tc.wantErr)
			}

			// Check the system instruction
			var gotText string
			if tc.req.Config != nil && tc.req.Config.SystemInstruction != nil {
				var texts []string
				for _, part := range tc.req.Config.SystemInstruction.Parts {
					if part.Text != "" {
						texts = append(texts, part.Text)
					}
				}
				gotText = strings.Join(texts, "\n\n")
			}

			if diff := cmp.Diff(tc.wantSystemInst, gotText); diff != "" {
				t.Errorf("system instruction mismatch (-want +got):\n%s", diff)
			}
		})
	}
}
