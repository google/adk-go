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
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/memory"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
)

// --- mockToolContext ---

type mockToolContext struct {
	agent.ContextMock // inherit mocking responses
	context.Context
	userContent *genai.Content
}

// Deadline implements [agent.Context].
func (m *mockToolContext) Deadline() (deadline time.Time, ok bool) {
	return m.Context.Deadline()
}

// Done implements [agent.Context].
func (m *mockToolContext) Done() <-chan struct{} {
	return m.Context.Done()
}

// Err implements [agent.Context].
func (m *mockToolContext) Err() error {
	return m.Context.Err()
}

// Value implements [agent.Context].
func (m *mockToolContext) Value(key any) any {
	return m.Context.Value(key)
}

func (m *mockToolContext) UserContent() *genai.Content {
	return m.userContent
}

// Implement other interface methods with panic or nil as needed for this specific test
func (m *mockToolContext) FunctionCallID() string { return "" }

func (m *mockToolContext) Actions() *session.EventActions {
	return &session.EventActions{}
}

func (m *mockToolContext) SearchMemory(ctx context.Context, query string) (*memory.SearchResponse, error) {
	return nil, nil
}

func (m *mockToolContext) AgentName() string { return "mock_agent" }

func (m *mockToolContext) InvocationID() string { return "mock_invocation" }

func (m *mockToolContext) AppName() string { return "mock_app" }

func (m *mockToolContext) Branch() string { return "mock_branch" }

func (m *mockToolContext) SessionID() string { return "mock_session" }

func (m *mockToolContext) UserID() string                                          { return "mock_user" }
func (m *mockToolContext) WithContext(ctx context.Context) agent.InvocationContext { return m }

var _ agent.Context = (*mockToolContext)(nil)

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
			name:         "NilUserContent",
			examples:     []*Example{{Input: genai.NewContentFromText("hi", "user"), Output: []*genai.Content{genai.NewContentFromText("hello", "model")}}},
			userContent:  nil,
			model:        "gemini-1.5-pro",
			wantInstruct: "",
		},
		{
			name:     "NilFirstPart",
			examples: []*Example{{Input: genai.NewContentFromText("hi", "user"), Output: []*genai.Content{genai.NewContentFromText("hello", "model")}}},
			userContent: &genai.Content{
				Parts: []*genai.Part{nil},
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
			name:     "InlineDataFirstPartThenText",
			examples: []*Example{{Input: genai.NewContentFromText("hi", "user"), Output: []*genai.Content{genai.NewContentFromText("hello", "model")}}},
			userContent: &genai.Content{
				Role: "user",
				Parts: []*genai.Part{
					{InlineData: &genai.Blob{MIMEType: "image/png", Data: []byte("img")}},
					{Text: "follow-up text"},
				},
			},
			model:        "gemini-1.5-pro",
			wantInstruct: "",
		},
		{
			name:     "ThoughtOnlyFirstPart",
			examples: []*Example{{Input: genai.NewContentFromText("hi", "user"), Output: []*genai.Content{genai.NewContentFromText("hello", "model")}}},
			userContent: &genai.Content{
				Role:  "user",
				Parts: []*genai.Part{{Text: "hidden reasoning", Thought: true}},
			},
			model: "gemini-1.5-pro",
			wantInstruct: `<EXAMPLES>
Begin few-shot
The following are examples of user queries and model responses using the available tools.

EXAMPLE 1:
Begin example
[user]
hi
[model]
hello
End example

End few-shot
<EXAMPLES>`,
		},
		{
			// reference: test_convert_examples_handles_content_without_parts
			name: "ContentWithoutParts",
			examples: []*Example{
				{
					Input:  &genai.Content{Role: "user"},
					Output: []*genai.Content{{Role: "model"}},
				},
			},
			userContent: genai.NewContentFromText("query", "user"),
			wantInstruct: `<EXAMPLES>
Begin few-shot
The following are examples of user queries and model responses using the available tools.

EXAMPLE 1:
Begin example
[user]
[model]
End example

End few-shot
<EXAMPLES>`,
		},
		{
			// reference: test_multi_part_text_example_conversion
			name: "MultiPartTextExample",
			examples: []*Example{
				{
					Input: genai.NewContentFromText("test_input", "user"),
					Output: []*genai.Content{
						{
							Role: "model",
							Parts: []*genai.Part{
								{Text: "test_output_1"},
								{Text: "test_output_2"},
								{Text: "test_output_3"},
							},
						},
					},
				},
			},
			userContent: genai.NewContentFromText("query", "user"),
			model:       "gemini-2.5-flash",
			wantInstruct: `<EXAMPLES>
Begin few-shot
The following are examples of user queries and model responses using the available tools.

EXAMPLE 1:
Begin example
[user]
test_input
[model]
test_output_1
test_output_2
test_output_3
End example

End few-shot
<EXAMPLES>`,
		},
		{
			// reference: test_example_conversion_output_clumping
			name: "OutputClumping",
			examples: []*Example{
				{
					Input: genai.NewContentFromText("test_input", "user"),
					Output: []*genai.Content{
						genai.NewContentFromText("test_output_1", "model"),
						genai.NewContentFromText("test_output_2", "model"),
						genai.NewContentFromText("test_output_3", "user"),
						genai.NewContentFromText("test_output_4", "user"),
					},
				},
			},
			userContent: genai.NewContentFromText("query", "user"),
			model:       "gemini-2.5-flash",
			wantInstruct: `<EXAMPLES>
Begin few-shot
The following are examples of user queries and model responses using the available tools.

EXAMPLE 1:
Begin example
[user]
test_input
[model]
test_output_1
test_output_2
[user]
test_output_3
test_output_4
End example

End few-shot
<EXAMPLES>`,
		},
		{
			// reference: test_empty_examples_list_conversion
			name:        "EmptyExamplesList",
			examples:    []*Example{},
			userContent: genai.NewContentFromText("query", "user"),
			model:       "gemini-2.5-flash",
			wantInstruct: `<EXAMPLES>
Begin few-shot
The following are examples of user queries and model responses using the available tools.

End few-shot
<EXAMPLES>`,
		},
		{
			// reference: test_example_conversion_with_text_and_function_call_response
			name: "TextAndFunctionCallResponse",
			examples: []*Example{
				{
					Input: genai.NewContentFromText("test_input", "user"),
					Output: []*genai.Content{
						{
							Role: "model",
							Parts: []*genai.Part{
								{Text: "test_output"},
								{
									FunctionCall: &genai.FunctionCall{
										Name: "my_tool",
										Args: map[string]any{"arg1": "val1"},
									},
								},
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
			model:       "gemini-2.5-flash",
			wantInstruct: `<EXAMPLES>
Begin few-shot
The following are examples of user queries and model responses using the available tools.

EXAMPLE 1:
Begin example
[user]
test_input
[model]
test_output
` + "```" + `
my_tool(arg1='val1')
` + "```" + `
` + "```" + `
&{<nil>  []  my_tool map[result:ok]}
` + "```" + `
End example

End few-shot
<EXAMPLES>`,
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
			name: "FunctionCallExample_NoModel",
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
			name: "FunctionCallExample_NonGemini",
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
			model:       "claude-sonnet-4-5",
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
				Context:     t.Context(),
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

// TestExampleTool_ProcessRequestStableArgumentOrder checks that an example
// function call's arguments are written in the same order on every request, so
// the system instruction does not change from one request to the next.
func TestExampleTool_ProcessRequestStableArgumentOrder(t *testing.T) {
	et, err := New(ExampleToolConfig{Examples: []*Example{{
		Input: genai.NewContentFromText("book a flight", "user"),
		Output: []*genai.Content{{
			Role: "model",
			Parts: []*genai.Part{{
				FunctionCall: &genai.FunctionCall{
					Name: "book_flight",
					Args: map[string]any{
						"origin":      "SFO",
						"destination": "JFK",
						"date":        "2026-11-01",
						"passengers":  2,
						"cabin":       "economy",
						"refundable":  true,
					},
				},
			}},
		}},
	}}})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	const wantCall = "book_flight(cabin='economy', date='2026-11-01', destination='JFK', origin='SFO', passengers=2, refundable=true)"
	// Repeated so that a map order that happens to come out sorted once
	// cannot pass the test.
	for range 20 {
		req := &model.LLMRequest{Model: "gemini-2.5-flash"}
		ctx := &mockToolContext{
			Context:     t.Context(),
			userContent: genai.NewContentFromText("query", "user"),
		}
		if err := et.ProcessRequest(ctx, req); err != nil {
			t.Fatalf("ProcessRequest() error = %v", err)
		}
		if got := req.Config.SystemInstruction.Parts[0].Text; !strings.Contains(got, wantCall) {
			t.Fatalf("system instruction does not contain %q:\n%s", wantCall, got)
		}
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

type fakeProvider struct {
	examples []*Example
	err      error
	gotQuery string
	calls    int
}

func (f *fakeProvider) Examples(_ context.Context, query string) ([]*Example, error) {
	f.calls++
	f.gotQuery = query
	if f.err != nil {
		return nil, f.err
	}
	return f.examples, nil
}

func TestNew_BothExamplesAndProviderError(t *testing.T) {
	_, err := New(ExampleToolConfig{
		Examples: []*Example{},
		Provider: &fakeProvider{},
	})
	if err == nil {
		t.Fatal("New() with both Examples and Provider succeeded, want error")
	}
}

func TestExampleTool_ProcessRequest_Provider(t *testing.T) {
	basicExamples := []*Example{
		{
			Input:  genai.NewContentFromText("test_input", "user"),
			Output: []*genai.Content{genai.NewContentFromText("test_output", "model")},
		},
	}
	wantBasicInstruct := `<EXAMPLES>
Begin few-shot
The following are examples of user queries and model responses using the available tools.

EXAMPLE 1:
Begin example
[user]
test_input
[model]
test_output
End example

End few-shot
<EXAMPLES>`

	// reference: test_building_si_from_base_example_provider
	t.Run("fetches examples from provider and builds instruction", func(t *testing.T) {
		for _, modelName := range []string{"", "gemini-2.5-flash", "google.gemini-2.5-flash", "claude-sonnet-4-5"} {
			fp := &fakeProvider{examples: basicExamples}
			et, err := New(ExampleToolConfig{Provider: fp})
			if err != nil {
				t.Fatalf("New() error = %v", err)
			}
			req := &model.LLMRequest{Model: modelName}
			ctx := &mockToolContext{
				Context:     t.Context(),
				userContent: genai.NewContentFromText("test_query", "user"),
			}
			if err := et.ProcessRequest(ctx, req); err != nil {
				t.Fatalf("ProcessRequest() error = %v", err)
			}
			if fp.calls != 1 {
				t.Fatalf("provider calls = %d, want 1", fp.calls)
			}
			if fp.gotQuery != "test_query" {
				t.Errorf("provider query = %q, want %q", fp.gotQuery, "test_query")
			}
			if req.Config == nil || req.Config.SystemInstruction == nil {
				t.Fatal("ProcessRequest() expected system instruction, got nil")
			}
			gotInstruct := req.Config.SystemInstruction.Parts[0].Text
			if diff := cmp.Diff(wantBasicInstruct, gotInstruct); diff != "" {
				t.Errorf("model %q: System instruction mismatch (-want +got):\n%s", modelName, diff)
			}
		}
	})

	t.Run("joins non-thought text parts and skips non-text first part", func(t *testing.T) {
		fp := &fakeProvider{examples: basicExamples}
		et, err := New(ExampleToolConfig{Provider: fp})
		if err != nil {
			t.Fatalf("New() error = %v", err)
		}
		req := &model.LLMRequest{Model: "gemini-2.5-flash"}
		ctx := &mockToolContext{
			Context: t.Context(),
			userContent: &genai.Content{
				Role: "user",
				Parts: []*genai.Part{
					{InlineData: &genai.Blob{MIMEType: "image/png", Data: []byte("img")}},
					{Text: "first part"},
					nil,
					{Text: "hidden reasoning", Thought: true},
					{Text: "second part"},
				},
			},
		}
		if err := et.ProcessRequest(ctx, req); err != nil {
			t.Fatalf("ProcessRequest() error = %v", err)
		}
		if fp.calls != 1 {
			t.Fatalf("provider calls = %d, want 1", fp.calls)
		}
		if fp.gotQuery != "first part second part" {
			t.Errorf("provider query = %q, want %q", fp.gotQuery, "first part second part")
		}
		if req.Config == nil || req.Config.SystemInstruction == nil {
			t.Fatal("ProcessRequest() expected system instruction, got nil")
		}
	})

	t.Run("skips provider when user content has no text", func(t *testing.T) {
		for _, uc := range []*genai.Content{
			nil,
			{Role: "user", Parts: []*genai.Part{}},
			{Role: "user", Parts: []*genai.Part{nil}},
			{Role: "user", Parts: []*genai.Part{{Text: ""}}},
			{Role: "user", Parts: []*genai.Part{{Text: "hidden reasoning", Thought: true}}},
			{Role: "user", Parts: []*genai.Part{{InlineData: &genai.Blob{MIMEType: "image/png", Data: []byte("img")}}}},
		} {
			fp := &fakeProvider{examples: basicExamples}
			et, err := New(ExampleToolConfig{Provider: fp})
			if err != nil {
				t.Fatalf("New() error = %v", err)
			}
			req := &model.LLMRequest{Model: "gemini-2.5-flash"}
			ctx := &mockToolContext{
				Context:     t.Context(),
				userContent: uc,
			}
			if err := et.ProcessRequest(ctx, req); err != nil {
				t.Fatalf("ProcessRequest() error = %v", err)
			}
			if fp.calls != 0 {
				t.Errorf("provider calls = %d, want 0", fp.calls)
			}
			if req.Config != nil && req.Config.SystemInstruction != nil {
				t.Errorf("unexpected system instruction: %v", req.Config.SystemInstruction)
			}
		}
	})

	t.Run("propagates provider error", func(t *testing.T) {
		wantErr := errors.New("provider failure")
		fp := &fakeProvider{err: wantErr}
		et, err := New(ExampleToolConfig{Provider: fp})
		if err != nil {
			t.Fatalf("New() error = %v", err)
		}
		req := &model.LLMRequest{Model: "gemini-2.5-flash"}
		ctx := &mockToolContext{
			Context:     t.Context(),
			userContent: genai.NewContentFromText("hello", "user"),
		}
		err = et.ProcessRequest(ctx, req)
		if !errors.Is(err, wantErr) {
			t.Fatalf("ProcessRequest() error = %v, want %v", err, wantErr)
		}
	})
}
