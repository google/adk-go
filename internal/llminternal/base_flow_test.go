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
	"errors"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	icontext "google.golang.org/adk/internal/context"
	"google.golang.org/adk/internal/toolinternal"
	"google.golang.org/adk/model"
	"google.golang.org/adk/tool"
)

type mockFunctionTool struct {
	name    string
	runFunc func(tool.Context, map[string]any) (map[string]any, error)
}

func (m *mockFunctionTool) Name() string {
	return m.name
}

func (m *mockFunctionTool) Description() string {
	return "mock tool"
}

func (m *mockFunctionTool) InputSchema() *genai.Schema {
	return nil
}

func (m *mockFunctionTool) OutputSchema() *genai.Schema {
	return nil
}

func (m *mockFunctionTool) IsLongRunning() bool {
	return false
}

func (m *mockFunctionTool) ProcessRequest(ctx tool.Context, req *model.LLMRequest) error {
	return nil
}

func (m *mockFunctionTool) Run(ctx tool.Context, args any) (map[string]any, error) {
	if m.runFunc != nil {
		return m.runFunc(ctx, args.(map[string]any))
	}
	return nil, nil
}

func (m *mockFunctionTool) Declaration() *genai.FunctionDeclaration {
	return nil
}

func TestHandleFunctionCalls(t *testing.T) {
	tests := []struct {
		name                string
		tool                toolinternal.FunctionTool
		args                map[string]any
		beforeToolCallbacks []BeforeToolCallback
		afterToolCallbacks  []AfterToolCallback
		want                map[string]any
	}{
		{
			name: "tool runs successfully",
			tool: &mockFunctionTool{
				name: "testTool",
				runFunc: func(ctx tool.Context, args map[string]any) (map[string]any, error) {
					return map[string]any{"result": "success"}, nil
				},
			},
			args: map[string]any{"key": "value"},
			want: map[string]any{"result": "success"},
		},
		{
			name: "tool error",
			tool: &mockFunctionTool{
				name: "testTool",
				runFunc: func(ctx tool.Context, args map[string]any) (map[string]any, error) {
					return nil, errors.New("tool error")
				},
			},
			args: map[string]any{"key": "value"},
			want: map[string]any{"error": "tool error"},
		},
		{
			name: "before callback returns result",
			tool: &mockFunctionTool{
				name: "testTool",
				runFunc: func(ctx tool.Context, args map[string]any) (map[string]any, error) {
					t.Error("tool should not be called")
					return nil, nil
				},
			},
			beforeToolCallbacks: []BeforeToolCallback{
				func(ctx tool.Context, tool tool.Tool, args map[string]any) (map[string]any, error) {
					return map[string]any{"result": "intercepted"}, nil
				},
				func(ctx tool.Context, tool tool.Tool, args map[string]any) (map[string]any, error) {
					return map[string]any{"result": "2nd callback should not be called"}, nil
				},
			},
			want: map[string]any{"result": "intercepted"},
		},
		{
			name: "before callback returns error",
			tool: &mockFunctionTool{
				name: "testTool",
				runFunc: func(ctx tool.Context, args map[string]any) (map[string]any, error) {
					t.Error("tool should not be called")
					return nil, nil
				},
			},
			beforeToolCallbacks: []BeforeToolCallback{
				func(ctx tool.Context, tool tool.Tool, args map[string]any) (map[string]any, error) {
					return nil, errors.New("before callback error")
				},
				func(ctx tool.Context, tool tool.Tool, args map[string]any) (map[string]any, error) {
					return nil, errors.New("unexpected error")
				},
			},
			want: map[string]any{"error": "before callback error"},
		},
		{
			name: "after callback modifies result",
			tool: &mockFunctionTool{
				name: "testTool",
				runFunc: func(ctx tool.Context, args map[string]any) (map[string]any, error) {
					return map[string]any{"result": "original"}, nil
				},
			},
			afterToolCallbacks: []AfterToolCallback{
				func(ctx tool.Context, tool tool.Tool, args, result map[string]any, err error) (map[string]any, error) {
					return map[string]any{"result": "modified"}, nil
				},
			},
			want: map[string]any{"result": "modified"},
		},
		{
			name: "after callback handles error",
			tool: &mockFunctionTool{
				name: "testTool",
				runFunc: func(ctx tool.Context, args map[string]any) (map[string]any, error) {
					return nil, errors.New("tool error")
				},
			},
			afterToolCallbacks: []AfterToolCallback{
				func(ctx tool.Context, tool tool.Tool, args, result map[string]any, err error) (map[string]any, error) {
					if err != nil {
						return map[string]any{"result": "error handled"}, nil
					}
					return nil, nil
				},
				func(ctx tool.Context, tool tool.Tool, args, result map[string]any, err error) (map[string]any, error) {
					return map[string]any{"result": "unexpected output"}, nil
				},
			},
			want: map[string]any{"result": "error handled"},
		},
		{
			name: "after callback returns error",
			tool: &mockFunctionTool{
				name: "testTool",
				runFunc: func(ctx tool.Context, args map[string]any) (map[string]any, error) {
					return map[string]any{"result": "success"}, nil
				},
			},
			afterToolCallbacks: []AfterToolCallback{
				func(ctx tool.Context, tool tool.Tool, args, result map[string]any, err error) (map[string]any, error) {
					return nil, errors.New("after callback error")
				},
				func(ctx tool.Context, tool tool.Tool, args, result map[string]any, err error) (map[string]any, error) {
					return nil, errors.New("unexpected error")
				},
			},
			want: map[string]any{"error": "after callback error"},
		},
		{
			name: "no-op callbacks return func results",
			tool: &mockFunctionTool{
				name: "testTool",
				runFunc: func(ctx tool.Context, args map[string]any) (map[string]any, error) {
					return map[string]any{"result": "success"}, nil
				},
			},
			beforeToolCallbacks: []BeforeToolCallback{
				func(ctx tool.Context, tool tool.Tool, args map[string]any) (map[string]any, error) {
					return nil, nil
				},
			},
			afterToolCallbacks: []AfterToolCallback{
				func(ctx tool.Context, tool tool.Tool, args, result map[string]any, err error) (map[string]any, error) {
					return nil, nil
				},
			},
			want: map[string]any{"result": "success"},
		},
		{
			name: "before callback result passed to after callback",
			tool: &mockFunctionTool{
				name: "testTool",
				runFunc: func(ctx tool.Context, args map[string]any) (map[string]any, error) {
					t.Error("tool should not be called")
					return nil, nil
				},
			},
			beforeToolCallbacks: []BeforeToolCallback{
				func(ctx tool.Context, tool tool.Tool, args map[string]any) (map[string]any, error) {
					return map[string]any{"result": "from_before"}, nil
				},
			},
			afterToolCallbacks: []AfterToolCallback{
				func(ctx tool.Context, tool tool.Tool, args, result map[string]any, err error) (map[string]any, error) {
					if val, ok := result["result"]; !ok || val != "from_before" {
						return nil, errors.New("unexpected result in after callback")
					}
					return map[string]any{"result": "from_after"}, nil
				},
			},
			want: map[string]any{"result": "from_after"},
		},
		{
			name: "before callback error passed to after callback",
			tool: &mockFunctionTool{
				name: "testTool",
				runFunc: func(ctx tool.Context, args map[string]any) (map[string]any, error) {
					t.Error("tool should not be called")
					return nil, nil
				},
			},
			beforeToolCallbacks: []BeforeToolCallback{
				func(ctx tool.Context, tool tool.Tool, args map[string]any) (map[string]any, error) {
					return nil, errors.New("error_from_before")
				},
			},
			afterToolCallbacks: []AfterToolCallback{
				func(ctx tool.Context, tool tool.Tool, args, result map[string]any, err error) (map[string]any, error) {
					if err == nil || err.Error() != "error_from_before" {
						return nil, errors.New("unexpected error in after callback")
					}
					return map[string]any{"result": "error_handled_in_after"}, nil
				},
			},
			want: map[string]any{"result": "error_handled_in_after"},
		},
	}

	simpleAgent, _ := agent.New(agent.Config{Name: "agentName"})
	ctx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{Agent: simpleAgent})
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			f := &Flow{
				BeforeToolCallbacks: tc.beforeToolCallbacks,
				AfterToolCallbacks:  tc.afterToolCallbacks,
			}

			toolsDict := map[string]tool.Tool{tc.tool.Name(): tc.tool}
			resp := &model.LLMResponse{
				Content: genai.NewContentFromFunctionCall(tc.tool.Name(), tc.args, genai.RoleModel),
			}
			got, err := f.handleFunctionCalls(ctx, toolsDict, resp)
			if err != nil {
				t.Fatalf("encountered unnexpected error: %s", err)
			}
			if got.Content == nil || len(got.Content.Parts) == 0 || got.Content.Parts[0].FunctionResponse == nil {
				t.Errorf("invalid function call result")
				return
			}
			gotFunctionResponse := got.Content.Parts[0].FunctionResponse.Response
			if diff := cmp.Diff(tc.want, gotFunctionResponse); diff != "" {
				t.Errorf("callTool() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestHandleWrongNameFunctionCalls(t *testing.T) {
	tests := []struct {
		name                string
		toolName            string
		tool                toolinternal.FunctionTool
		args                map[string]any
		beforeToolCallbacks []BeforeToolCallback
		afterToolCallbacks  []AfterToolCallback
		want                map[string]any
		wantErr             bool
		wantErrMessage      string
	}{
		{
			name:     "tool run fails with wrong name",
			toolName: "wrongTool",
			tool: &mockFunctionTool{
				name: "testTool",
				runFunc: func(ctx tool.Context, args map[string]any) (map[string]any, error) {
					return map[string]any{"result": "success"}, nil
				},
			},
			args:           map[string]any{"key": "value"},
			wantErr:        true,
			wantErrMessage: `unknown tool: "wrongTool"`,
		},
		{
			name:     "tool fails with wrong name and beforeToolCallback without modify",
			toolName: "wrongTool",
			tool: &mockFunctionTool{
				name: "testTool",
				runFunc: func(ctx tool.Context, args map[string]any) (map[string]any, error) {
					return map[string]any{"result": "success"}, nil
				},
			},
			args: map[string]any{"key": "value"},
			beforeToolCallbacks: []BeforeToolCallback{
				func(ctx tool.Context, tool tool.Tool, args map[string]any) (map[string]any, error) {
					return nil, nil
				},
			},
			wantErr:        true,
			wantErrMessage: `unknown tool: "wrongTool"`,
		},
		{
			name:     "tool doesn't fails with wrong name and beforeToolCallback",
			toolName: "wrongTool",
			tool: &mockFunctionTool{
				name: "testTool",
				runFunc: func(ctx tool.Context, args map[string]any) (map[string]any, error) {
					return map[string]any{"result": "success"}, nil
				},
			},
			args: map[string]any{"key": "value"},
			beforeToolCallbacks: []BeforeToolCallback{
				func(ctx tool.Context, tool tool.Tool, args map[string]any) (map[string]any, error) {
					if tool == nil {
						return map[string]any{"result": "from_before"}, nil
					}
					return nil, nil
				},
			},
			want:    map[string]any{"result": "from_before"},
			wantErr: false,
		},
		{
			name:     "tool doesn't fails with wrong name and beforeToolCallback returning error",
			toolName: "wrongTool",
			tool: &mockFunctionTool{
				name: "testTool",
				runFunc: func(ctx tool.Context, args map[string]any) (map[string]any, error) {
					return map[string]any{"result": "success"}, nil
				},
			},
			args: map[string]any{"key": "value"},
			beforeToolCallbacks: []BeforeToolCallback{
				func(ctx tool.Context, tool tool.Tool, args map[string]any) (map[string]any, error) {
					if tool == nil {
						return nil, errors.New("tool is nil")
					}
					return nil, nil
				},
			},
			want:    map[string]any{"error": "tool is nil"},
			wantErr: false,
		},
		{
			name:     "tool doesn't fails with wrong name and beforeToolCallback, with afterToolCallback also being called",
			toolName: "wrongTool",
			tool: &mockFunctionTool{
				name: "testTool",
				runFunc: func(ctx tool.Context, args map[string]any) (map[string]any, error) {
					return map[string]any{"result": "success"}, nil
				},
			},
			args: map[string]any{"key": "value"},
			beforeToolCallbacks: []BeforeToolCallback{
				func(ctx tool.Context, tool tool.Tool, args map[string]any) (map[string]any, error) {
					if tool == nil {
						return map[string]any{"result": "from_before"}, nil
					}
					return nil, nil
				},
			},
			afterToolCallbacks: []AfterToolCallback{
				func(ctx tool.Context, tool tool.Tool, args, result map[string]any, err error) (map[string]any, error) {
					return map[string]any{"result": "modified"}, nil
				},
			},
			want:    map[string]any{"result": "modified"},
			wantErr: false,
		},
	}

	simpleAgent, _ := agent.New(agent.Config{Name: "agentName"})
	ctx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{Agent: simpleAgent})
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			f := &Flow{
				BeforeToolCallbacks: tc.beforeToolCallbacks,
				AfterToolCallbacks:  tc.afterToolCallbacks,
			}

			toolsDict := map[string]tool.Tool{tc.tool.Name(): tc.tool}
			resp := &model.LLMResponse{
				Content: genai.NewContentFromFunctionCall(tc.toolName, tc.args, genai.RoleModel),
			}
			got, err := f.handleFunctionCalls(ctx, toolsDict, resp)
			if err != nil && !tc.wantErr {
				t.Fatalf("encountered unnexpected error: %s", err)
			}
			if err != nil && tc.wantErr {
				if diff := cmp.Diff(tc.wantErrMessage, err.Error()); diff != "" {
					t.Errorf("callTool() mismatch error message (-want +got):\n%s", diff)
				}
				return
			}
			if got.Content == nil || len(got.Content.Parts) == 0 || got.Content.Parts[0].FunctionResponse == nil {
				t.Errorf("invalid function call result")
				return
			}
			gotFunctionResponse := got.Content.Parts[0].FunctionResponse.Response
			if diff := cmp.Diff(tc.want, gotFunctionResponse); diff != "" {
				t.Errorf("callTool() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}
