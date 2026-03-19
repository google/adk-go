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

package exampletool

import (
	"context"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/memory"
	"google.golang.org/adk/model"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/toolconfirmation"
)

// --- mockToolContext ---

type mockToolContext struct {
	context.Context
	userContent *genai.Content
}

func (m *mockToolContext) UserContent() *genai.Content {
	return m.userContent
}

// Implement other interface methods with panic or nil as needed for this specific test
func (m *mockToolContext) FunctionCallID() string { return "" }

func (m *mockToolContext) Actions() *session.EventActions {
	return &session.EventActions{}
}

func (m *mockToolContext) SearchMemory(ctx context.Context, query string) (*memory.SearchMemoryResponse, error) {
	return nil, nil
}
func (m *mockToolContext) ToolConfirmation() *toolconfirmation.ToolConfirmation { return nil }
func (m *mockToolContext) RequestConfirmation(hint string, payload any) error   { return nil }
func (m *mockToolContext) AgentName() string                                    { return "mock_agent" }
func (m *mockToolContext) ReadonlyState() session.ReadonlyState                 { return nil }
func (m *mockToolContext) State() session.State                                 { return nil }
func (m *mockToolContext) Artifacts() agent.Artifacts                           { return nil }
func (m *mockToolContext) InvocationID() string                                 { return "mock_invocation" }
func (m *mockToolContext) AppName() string                                      { return "mock_app" }
func (m *mockToolContext) Branch() string                                       { return "mock_branch" }
func (m *mockToolContext) SessionID() string                                    { return "mock_session" }
func (m *mockToolContext) UserID() string                                       { return "mock_user" }

// --- Tests ---

func TestExampleTool_ProcessRequest(t *testing.T) {
	tests := []struct {
		name         string
		examples     []*Example
		userContent  *genai.Content
		model        string
		wantInstruct string
	}{
		{
			name:     "NoUserContent",
			examples: []*Example{{Input: genai.NewContentFromText("hi", "user"), Output: []*genai.Content{genai.NewContentFromText("hello", "model")}}},
			userContent: &genai.Content{
				Parts: []*genai.Part{},
			},
			model:        "gemini-1.5-pro",
			wantInstruct: "",
		},
		{
			name:     "EmptyUserContentString",
			examples: []*Example{{Input: genai.NewContentFromText("hi", "user"), Output: []*genai.Content{genai.NewContentFromText("hello", "model")}}},
			userContent: &genai.Content{
				Parts: []*genai.Part{{Text: ""}},
			},
			model:        "gemini-1.5-pro",
			wantInstruct: "",
		},
		{
			name: "SimpleExample",
			examples: []*Example{
				{
					Input:  genai.NewContentFromText("input1", "user"),
					Output: []*genai.Content{genai.NewContentFromText("output1", "model")},
				},
			},
			userContent: genai.NewContentFromText("user query", "user"),
			model:       "gemini-1.5-pro",
			wantInstruct: `<EXAMPLES>
Begin few-shot
The following are examples of user queries and model responses using the available tools.

EXAMPLE 1:
Begin example
[user]
input1
[model]
output1
End example

End few-shot
<EXAMPLES>`,
		},
		{
			name: "MultipleExamples",
			examples: []*Example{
				{
					Input:  genai.NewContentFromText("in1", "user"),
					Output: []*genai.Content{genai.NewContentFromText("out1", "model")},
				},
				{
					Input:  genai.NewContentFromText("in2", "user"),
					Output: []*genai.Content{genai.NewContentFromText("out2", "model")},
				},
			},
			userContent: genai.NewContentFromText("query", "user"),
			model:       "gemini-1.5-pro",
			wantInstruct: `<EXAMPLES>
Begin few-shot
The following are examples of user queries and model responses using the available tools.

EXAMPLE 1:
Begin example
[user]
in1
[model]
out1
End example

EXAMPLE 2:
Begin example
[user]
in2
[model]
out2
End example

End few-shot
<EXAMPLES>`,
		},
		{
			name: "FunctionCallExample_Gemini1.5",
			examples: []*Example{
				{
					Input: genai.NewContentFromText("call func", "user"),
					Output: []*genai.Content{
						{
							Role: "model",
							Parts: []*genai.Part{
								{
									FunctionCall: &genai.FunctionCall{
										Name: "my_tool",
										Args: map[string]any{"arg1": "val1"},
									},
								},
							},
						},
						{
							Role: "user", // Function response is usually from user/tool role
							Parts: []*genai.Part{
								{
									FunctionResponse: &genai.FunctionResponse{
										Name:     "my_tool",
										Response: map[string]any{"result": "ok"},
									},
								},
							},
						},
					},
				},
			},
			userContent: genai.NewContentFromText("query", "user"),
			model:       "gemini-1.5-pro",
			wantInstruct: `<EXAMPLES>
Begin few-shot
The following are examples of user queries and model responses using the available tools.

EXAMPLE 1:
Begin example
[user]
call func
[model]
` + "```" + `
my_tool(arg1='val1')
` + "```" + `
[user]
` + "```" + `
&{<nil>  []  my_tool map[result:ok]}
` + "```" + `
End example

End few-shot
<EXAMPLES>`,
		},
		{
			name: "FunctionCallExample_Gemini2",
			examples: []*Example{
				{
					Input: genai.NewContentFromText("call func", "user"),
					Output: []*genai.Content{
						{
							Role: "model",
							Parts: []*genai.Part{
								{
									FunctionCall: &genai.FunctionCall{
										Name: "my_tool",
										Args: map[string]any{"arg1": "val1"},
									},
								},
							},
						},
						{
							Role: "user",
							Parts: []*genai.Part{
								{
									FunctionResponse: &genai.FunctionResponse{
										Name:     "my_tool",
										Response: map[string]any{"result": "ok"},
									},
								},
							},
						},
					},
				},
			},
			userContent: genai.NewContentFromText("query", "user"),
			model:       "gemini-2.0-flash",
			wantInstruct: `<EXAMPLES>
Begin few-shot
The following are examples of user queries and model responses using the available tools.

EXAMPLE 1:
Begin example
[user]
call func
[model]
` + "```tool_code" + `
my_tool(arg1='val1')
` + "```" + `
[user]
` + "```tool_outputs" + `
&{<nil>  []  my_tool map[result:ok]}
` + "```" + `
End example

End few-shot
<EXAMPLES>`,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			et, err := New(ExampleToolConfig{Examples: tc.examples})
			if err != nil {
				t.Fatalf("New() error = %v", err)
			}

			req := &model.LLMRequest{
				Model: tc.model,
			}
			ctx := &mockToolContext{
				Context:     context.Background(),
				userContent: tc.userContent,
			}

			err = et.ProcessRequest(ctx, req)
			if err != nil {
				t.Errorf("ProcessRequest() error = %v", err)
			}

			if tc.wantInstruct == "" {
				if req.Config != nil && req.Config.SystemInstruction != nil {
					t.Errorf("ProcessRequest() unexpected system instruction: got %v, want nil/empty", req.Config.SystemInstruction)
				}
				return
			}

			if req.Config == nil || req.Config.SystemInstruction == nil {
				t.Fatal("ProcessRequest() expected system instruction, got nil")
			}

			gotInstruct := req.Config.SystemInstruction.Parts[0].Text
			// Normalize newlines for comparison if needed, though exact match is best
			if diff := cmp.Diff(tc.wantInstruct, gotInstruct); diff != "" {
				t.Errorf("System instruction mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestExampleTool_Interface(t *testing.T) {
	et, _ := New(ExampleToolConfig{})
	var _ tool.Tool = et
	if et.Name() != "example_tool" {
		t.Errorf("Name() = %q, want %q", et.Name(), "example_tool")
	}
	if et.IsLongRunning() {
		t.Error("IsLongRunning() = true, want false")
	}
}
