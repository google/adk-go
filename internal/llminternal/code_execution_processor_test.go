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

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/codeexecution"
	icontext "google.golang.org/adk/internal/context"
	"google.golang.org/adk/internal/utils"
	"google.golang.org/adk/model"
	"google.golang.org/adk/session"
)

func TestCodeExecutionResponseProcessor(t *testing.T) {
	tests := []struct {
		name    string
		agent   agent.Agent
		session session.Session
		req     *model.LLMRequest
		resp    *model.LLMResponse
	}{
		{
			name: "simple",
			agent: &mockLLMAgent{
				Agent: utils.Must(agent.New(agent.Config{
					Name: "test_agent",
				})),
				s: &State{
					CodeExecutor: &mockCodeExecutor{
						stdout: "Hello world!",
					},
				},
			},
			session: &mockSession{
				state: &mockState{
					data: make(map[string]any),
				},
			},
			req: &model.LLMRequest{
				Contents: []*genai.Content{
					{
						Parts: []*genai.Part{
							genai.NewPartFromText("Print 'Hello World!'"),
						},
						Role: genai.RoleUser,
					},
				},
			},
			resp: &model.LLMResponse{
				Content: &genai.Content{
					Parts: []*genai.Part{
						genai.NewPartFromText("Some text before code."),
						genai.NewPartFromText("```python\nprint(\"Hello World!\")\n```"),
						genai.NewPartFromText("Some text after code."),
					},
					Role: genai.RoleModel,
				},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{
				Agent:   tt.agent,
				Session: tt.session,
			})
			iter := codeExecutionResponseProcessor(ctx, tt.req, tt.resp)
			for _, err := range iter {
				if err != nil {
					t.Fatalf("codeExecutionResponseProcessor() unexpected error: %v", err)
				}
			}
		})
	}
}

func TestExtractCodeAndTruncateContent(t *testing.T) {
	tests := []struct {
		name       string
		content    *genai.Content
		delimiters []codeexecution.Delimiter
		want       string
	}{
		{
			name: "simple python code extraction",
			content: &genai.Content{
				Parts: []*genai.Part{
					genai.NewPartFromText("Here is some python code to execute."),
					genai.NewPartFromText("```python\nprint(\"Hello World!\")\n```"),
				},
				Role: genai.RoleModel,
			},
			delimiters: []codeexecution.Delimiter{
				{
					Leading:  "```python\n",
					Trailing: "\n```",
				},
			},
			want: "print(\"Hello World!\")",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := extractCodeAndTruncateContent(tt.content, tt.delimiters)
			if got != tt.want {
				t.Errorf("incorrect code extraction: got %s, expected %s", got, tt.want)
			}
		})
	}
}
