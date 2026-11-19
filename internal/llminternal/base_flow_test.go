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
	"bytes"
	"context"
	"errors"
	"fmt"
	"iter"
	"log"
	"maps"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/internal/agent/runconfig"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/internal/toolinternal"
	"google.golang.org/adk/v2/internal/utils"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/platform"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
	"google.golang.org/adk/v2/tool/toolauth"
)

type mockFunctionTool struct {
	name    string
	runFunc func(agent.Context, map[string]any) (map[string]any, error)
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

func (m *mockFunctionTool) ProcessRequest(ctx agent.Context, req *model.LLMRequest) error {
	return nil
}

func (m *mockFunctionTool) Run(ctx agent.Context, args any) (map[string]any, error) {
	if m.runFunc != nil {
		return m.runFunc(ctx, args.(map[string]any))
	}
	return nil, nil
}

func (m *mockFunctionTool) Declaration() *genai.FunctionDeclaration {
	return nil
}

type mockToolset struct {
	name string
}

func (m *mockToolset) Name() string { return m.name }
func (m *mockToolset) Tools(ctx agent.ReadonlyContext) ([]tool.Tool, error) {
	return nil, nil
}

type mockRequestProcessorToolset struct {
	name    string
	process func(ctx agent.Context, req *model.LLMRequest) error
}

func (m *mockRequestProcessorToolset) ProcessRequest(ctx agent.Context, req *model.LLMRequest) error {
	if m.process != nil {
		return m.process(ctx, req)
	}
	return nil
}
func (m *mockRequestProcessorToolset) Name() string { return m.name }
func (m *mockRequestProcessorToolset) Tools(ctx agent.ReadonlyContext) ([]tool.Tool, error) {
	return nil, nil
}

type testCase struct {
	name                 string
	tool                 toolinternal.FunctionTool
	args                 map[string]any
	beforeToolCallbacks  []BeforeToolCallback
	afterToolCallbacks   []AfterToolCallback
	onToolErrorCallbacks []OnToolErrorCallback
	want                 map[string]any
	// check runs after the result has been compared, for a row that asserts on
	// more than the result.
	check func(t *testing.T)
}

func TestCallTool(t *testing.T) {
	panicTool, err := functiontool.New(functiontool.Config{Name: "panicTool"}, func(agent.Context, struct{}) (struct{}, error) {
		panic("boom")
	})
	if err != nil {
		t.Fatal(err)
	}
	// Callbacks run before the flow replaces the panic error with the generic
	// message, so each one records the error it was handed for its own row's
	// check to assert on. Each panic row gets its own map, so that a row only
	// ever reads what its own callbacks recorded.
	panicCallbackErrs := map[string]error{}
	wrappedPanicCallbackErrs := map[string]error{}
	testCases := []testCase{
		{
			name: "tool runs successfully",
			tool: &mockFunctionTool{
				name: "testTool",
				runFunc: func(ctx agent.Context, args map[string]any) (map[string]any, error) {
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
				runFunc: func(ctx agent.Context, args map[string]any) (map[string]any, error) {
					return nil, errors.New("tool error")
				},
			},
			args: map[string]any{"key": "value"},
			want: map[string]any{"error": "tool error"},
		},
		{
			name: "tool panic is hidden from the model but not from callbacks",
			tool: panicTool.(toolinternal.FunctionTool),
			args: map[string]any{},
			onToolErrorCallbacks: []OnToolErrorCallback{
				func(ctx agent.Context, tool tool.Tool, args map[string]any, err error) (map[string]any, error) {
					panicCallbackErrs["OnToolError"] = err
					return nil, nil
				},
			},
			afterToolCallbacks: []AfterToolCallback{
				func(ctx agent.Context, tool tool.Tool, args, result map[string]any, err error) (map[string]any, error) {
					panicCallbackErrs["AfterTool"] = err
					return nil, nil
				},
			},
			want:  map[string]any{"error": "Invoking `panicTool()` failed with an internal error."},
			check: checkPanicCallbackErrs(panicCallbackErrs, "OnToolError", "AfterTool"),
		},
		{
			// An error on its way out of a tool is wrapped more than once:
			// agenttool wraps what the sub-agent run returned, and a callback
			// that handles the error and returns its own is free to wrap it.
			name: "wrapped tool panic is hidden from the model",
			tool: panicTool.(toolinternal.FunctionTool),
			args: map[string]any{},
			onToolErrorCallbacks: []OnToolErrorCallback{
				func(ctx agent.Context, tool tool.Tool, args map[string]any, err error) (map[string]any, error) {
					wrappedPanicCallbackErrs["OnToolError"] = err
					return nil, fmt.Errorf("lookup failed: %w", err)
				},
			},
			want:  map[string]any{"error": "Invoking `panicTool()` failed with an internal error."},
			check: checkPanicCallbackErrs(wrappedPanicCallbackErrs, "OnToolError"),
		},
		{
			name: "before callback returns result",
			tool: &mockFunctionTool{
				name: "testTool",
				runFunc: func(ctx agent.Context, args map[string]any) (map[string]any, error) {
					t.Error("tool should not be called")
					return nil, nil
				},
			},
			beforeToolCallbacks: []BeforeToolCallback{
				func(ctx agent.Context, tool tool.Tool, args map[string]any) (map[string]any, error) {
					return map[string]any{"result": "intercepted"}, nil
				},
				func(ctx agent.Context, tool tool.Tool, args map[string]any) (map[string]any, error) {
					return map[string]any{"result": "2nd callback should not be called"}, nil
				},
			},
			want: map[string]any{"result": "intercepted"},
		},
		{
			name: "before callback returns error",
			tool: &mockFunctionTool{
				name: "testTool",
				runFunc: func(ctx agent.Context, args map[string]any) (map[string]any, error) {
					t.Error("tool should not be called")
					return nil, nil
				},
			},
			beforeToolCallbacks: []BeforeToolCallback{
				func(ctx agent.Context, tool tool.Tool, args map[string]any) (map[string]any, error) {
					return nil, errors.New("before callback error")
				},
				func(ctx agent.Context, tool tool.Tool, args map[string]any) (map[string]any, error) {
					return nil, errors.New("unexpected error")
				},
			},
			want: map[string]any{"error": "before callback error"},
		},
		{
			name: "after callback modifies result",
			tool: &mockFunctionTool{
				name: "testTool",
				runFunc: func(ctx agent.Context, args map[string]any) (map[string]any, error) {
					return map[string]any{"result": "original"}, nil
				},
			},
			afterToolCallbacks: []AfterToolCallback{
				func(ctx agent.Context, tool tool.Tool, args, result map[string]any, err error) (map[string]any, error) {
					return map[string]any{"result": "modified"}, nil
				},
			},
			want: map[string]any{"result": "modified"},
		},
		{
			name: "after callback handles error",
			tool: &mockFunctionTool{
				name: "testTool",
				runFunc: func(ctx agent.Context, args map[string]any) (map[string]any, error) {
					return nil, errors.New("tool error")
				},
			},
			afterToolCallbacks: []AfterToolCallback{
				func(ctx agent.Context, tool tool.Tool, args, result map[string]any, err error) (map[string]any, error) {
					if err != nil {
						return map[string]any{"result": "error handled"}, nil
					}
					return nil, nil
				},
				func(ctx agent.Context, tool tool.Tool, args, result map[string]any, err error) (map[string]any, error) {
					return map[string]any{"result": "unexpected output"}, nil
				},
			},
			want: map[string]any{"result": "error handled"},
		},
		{
			name: "after callback returns error",
			tool: &mockFunctionTool{
				name: "testTool",
				runFunc: func(ctx agent.Context, args map[string]any) (map[string]any, error) {
					return map[string]any{"result": "success"}, nil
				},
			},
			afterToolCallbacks: []AfterToolCallback{
				func(ctx agent.Context, tool tool.Tool, args, result map[string]any, err error) (map[string]any, error) {
					return nil, errors.New("after callback error")
				},
				func(ctx agent.Context, tool tool.Tool, args, result map[string]any, err error) (map[string]any, error) {
					return nil, errors.New("unexpected error")
				},
			},
			want: map[string]any{"error": "after callback error"},
		},
		{
			name: "no-op callbacks return func results",
			tool: &mockFunctionTool{
				name: "testTool",
				runFunc: func(ctx agent.Context, args map[string]any) (map[string]any, error) {
					return map[string]any{"result": "success"}, nil
				},
			},
			beforeToolCallbacks: []BeforeToolCallback{
				func(ctx agent.Context, tool tool.Tool, args map[string]any) (map[string]any, error) {
					return nil, nil
				},
			},
			afterToolCallbacks: []AfterToolCallback{
				func(ctx agent.Context, tool tool.Tool, args, result map[string]any, err error) (map[string]any, error) {
					return nil, nil
				},
			},
			want: map[string]any{"result": "success"},
		},
		{
			name: "before callback result passed to after callback",
			tool: &mockFunctionTool{
				name: "testTool",
				runFunc: func(ctx agent.Context, args map[string]any) (map[string]any, error) {
					t.Error("tool should not be called")
					return nil, nil
				},
			},
			beforeToolCallbacks: []BeforeToolCallback{
				func(ctx agent.Context, tool tool.Tool, args map[string]any) (map[string]any, error) {
					return map[string]any{"result": "from_before"}, nil
				},
			},
			afterToolCallbacks: []AfterToolCallback{
				func(ctx agent.Context, tool tool.Tool, args, result map[string]any, err error) (map[string]any, error) {
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
				runFunc: func(ctx agent.Context, args map[string]any) (map[string]any, error) {
					t.Error("tool should not be called")
					return nil, nil
				},
			},
			beforeToolCallbacks: []BeforeToolCallback{
				func(ctx agent.Context, tool tool.Tool, args map[string]any) (map[string]any, error) {
					return nil, errors.New("error_from_before")
				},
			},
			afterToolCallbacks: []AfterToolCallback{
				func(ctx agent.Context, tool tool.Tool, args, result map[string]any, err error) (map[string]any, error) {
					if err == nil || err.Error() != "error_from_before" {
						return nil, errors.New("unexpected error in after callback")
					}
					return map[string]any{"result": "error_handled_in_after"}, nil
				},
			},
			want: map[string]any{"result": "error_handled_in_after"},
		},
		{
			name: "before callback error passed to on tool error callback",
			tool: &mockFunctionTool{
				name: "testTool",
				runFunc: func(ctx agent.Context, args map[string]any) (map[string]any, error) {
					t.Error("tool should not be called")
					return nil, nil
				},
			},
			beforeToolCallbacks: []BeforeToolCallback{
				func(ctx agent.Context, tool tool.Tool, args map[string]any) (map[string]any, error) {
					return nil, errors.New("error_from_before")
				},
			},
			onToolErrorCallbacks: []OnToolErrorCallback{
				func(ctx agent.Context, tool tool.Tool, args map[string]any, err error) (map[string]any, error) {
					if err == nil || err.Error() != "error_from_before" {
						t.Error("unexpected error in on tool error callback")
						return nil, errors.New("unexpected error in on tool error callback")
					}
					return map[string]any{"result": "error_handled_in_on_tool_error_callback"}, nil
				},
			},
			want: map[string]any{"result": "error_handled_in_on_tool_error_callback"},
		},
		{
			name: "before callback error passed to on tool error callback and after tool called",
			tool: &mockFunctionTool{
				name: "testTool",
				runFunc: func(ctx agent.Context, args map[string]any) (map[string]any, error) {
					t.Error("tool should not be called")
					return nil, nil
				},
			},
			beforeToolCallbacks: []BeforeToolCallback{
				func(ctx agent.Context, tool tool.Tool, args map[string]any) (map[string]any, error) {
					return nil, errors.New("error_from_before")
				},
			},
			onToolErrorCallbacks: []OnToolErrorCallback{
				func(ctx agent.Context, tool tool.Tool, args map[string]any, err error) (map[string]any, error) {
					if err == nil || err.Error() != "error_from_before" {
						t.Error("unexpected error in on tool error callback")
						return nil, errors.New("unexpected error in on tool error callback")
					}
					return map[string]any{"result": "error_handled_in_on_tool_error_callback"}, nil
				},
			},
			afterToolCallbacks: []AfterToolCallback{
				func(ctx agent.Context, tool tool.Tool, args, result map[string]any, err error) (map[string]any, error) {
					if err != nil {
						return nil, errors.New("unexpected error in after callback")
					}
					return map[string]any{"result": "from_after"}, nil
				},
			},
			want: map[string]any{"result": "from_after"},
		},
		{
			name: "before callback error passed to on tool error callback and passed to after tool called",
			tool: &mockFunctionTool{
				name: "testTool",
				runFunc: func(ctx agent.Context, args map[string]any) (map[string]any, error) {
					t.Error("tool should not be called")
					return nil, nil
				},
			},
			beforeToolCallbacks: []BeforeToolCallback{
				func(ctx agent.Context, tool tool.Tool, args map[string]any) (map[string]any, error) {
					return nil, errors.New("error_from_before")
				},
			},
			onToolErrorCallbacks: []OnToolErrorCallback{
				func(ctx agent.Context, tool tool.Tool, args map[string]any, err error) (map[string]any, error) {
					if err == nil || err.Error() != "error_from_before" {
						t.Error("unexpected error in on tool error callback")
						return nil, errors.New("unexpected error in on tool error callback")
					}
					return nil, errors.New("error_from_on_tool_error")
				},
			},
			afterToolCallbacks: []AfterToolCallback{
				func(ctx agent.Context, tool tool.Tool, args, result map[string]any, err error) (map[string]any, error) {
					if err == nil || err.Error() != "error_from_on_tool_error" {
						return nil, errors.New("unexpected error in after callback")
					}
					return nil, errors.New("error_from_after_tool")
				},
			},
			want: map[string]any{"error": "error_from_after_tool"},
		},
		{
			name: "before callback error passed to on tool error callback and passed to after tool called and handled",
			tool: &mockFunctionTool{
				name: "testTool",
				runFunc: func(ctx agent.Context, args map[string]any) (map[string]any, error) {
					t.Error("tool should not be called")
					return nil, nil
				},
			},
			beforeToolCallbacks: []BeforeToolCallback{
				func(ctx agent.Context, tool tool.Tool, args map[string]any) (map[string]any, error) {
					return nil, errors.New("error_from_before")
				},
			},
			onToolErrorCallbacks: []OnToolErrorCallback{
				func(ctx agent.Context, tool tool.Tool, args map[string]any, err error) (map[string]any, error) {
					if err == nil || err.Error() != "error_from_before" {
						t.Error("unexpected error in on tool error callback")
						return nil, errors.New("unexpected error in on tool error callback")
					}
					return nil, errors.New("error_from_on_tool_error")
				},
			},
			afterToolCallbacks: []AfterToolCallback{
				func(ctx agent.Context, tool tool.Tool, args, result map[string]any, err error) (map[string]any, error) {
					if err == nil || err.Error() != "error_from_on_tool_error" {
						return nil, errors.New("unexpected error in after callback")
					}
					return map[string]any{"result": "error_handled_in_on_tool_error_callback"}, nil
				},
			},
			want: map[string]any{"result": "error_handled_in_on_tool_error_callback"},
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			f := &Flow{
				BeforeToolCallbacks:  tc.beforeToolCallbacks,
				AfterToolCallbacks:   tc.afterToolCallbacks,
				OnToolErrorCallbacks: tc.onToolErrorCallbacks,
			}
			ctx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{})
			got := f.callTool(agent.NewToolContext(ctx, "", nil, nil), tc.tool, tc.args)
			if diff := cmp.Diff(tc.want, got); diff != "" {
				t.Errorf("callTool() mismatch (-want +got):\n%s", diff)
			}
			if tc.check != nil {
				tc.check(t)
			}
		})
	}
}

func TestHandleFunctionCalls_CallbackArgEditsDoNotReachTheCallEvent(t *testing.T) {
	echoTool := &mockFunctionTool{
		name: "echo",
		runFunc: func(_ agent.Context, args map[string]any) (map[string]any, error) {
			return map[string]any{"args": args}, nil
		},
	}
	echoArgs := func(_ agent.Context, _ tool.Tool, args map[string]any, _ error) (map[string]any, error) {
		args["input_str"] = "modified_input"
		return map[string]any{"args": args}, nil
	}
	for _, tc := range []struct {
		name         string
		toolName     string
		callArgs     map[string]any
		before       BeforeToolCallback
		onToolError  OnToolErrorCallback
		wantResponse map[string]any
		wantCallArgs map[string]any
	}{
		{
			name:     "before-tool callback adds an argument",
			toolName: "echo",
			callArgs: map[string]any{},
			before: func(_ agent.Context, _ tool.Tool, args map[string]any) (map[string]any, error) {
				args["input_str"] = "modified_input"
				return nil, nil
			},
			wantResponse: map[string]any{"args": map[string]any{"input_str": "modified_input"}},
			wantCallArgs: map[string]any{},
		},
		{
			name:     "before-tool callback adds an argument to a call without arguments",
			toolName: "echo",
			callArgs: nil,
			before: func(_ agent.Context, _ tool.Tool, args map[string]any) (map[string]any, error) {
				args["input_str"] = "modified_input"
				return nil, nil
			},
			wantResponse: map[string]any{"args": map[string]any{"input_str": "modified_input"}},
			wantCallArgs: nil,
		},
		{
			name:     "before-tool callback edits a nested object",
			toolName: "echo",
			callArgs: map[string]any{"filter": map[string]any{"limit": 1}},
			before: func(_ agent.Context, _ tool.Tool, args map[string]any) (map[string]any, error) {
				args["filter"].(map[string]any)["limit"] = 2
				return nil, nil
			},
			wantResponse: map[string]any{"args": map[string]any{"filter": map[string]any{"limit": 2}}},
			wantCallArgs: map[string]any{"filter": map[string]any{"limit": 1}},
		},
		{
			name:     "before-tool callback edits a nested list",
			toolName: "echo",
			callArgs: map[string]any{"tags": []any{"a", map[string]any{"k": "v"}}},
			before: func(_ agent.Context, _ tool.Tool, args map[string]any) (map[string]any, error) {
				tags := args["tags"].([]any)
				tags[0] = "b"
				tags[1].(map[string]any)["k"] = "w"
				return nil, nil
			},
			wantResponse: map[string]any{"args": map[string]any{"tags": []any{"b", map[string]any{"k": "w"}}}},
			wantCallArgs: map[string]any{"tags": []any{"a", map[string]any{"k": "v"}}},
		},
		{
			name:         "nil nested values stay nil",
			toolName:     "echo",
			callArgs:     map[string]any{"filter": map[string]any(nil), "tags": []any(nil)},
			wantResponse: map[string]any{"args": map[string]any{"filter": map[string]any(nil), "tags": []any(nil)}},
			wantCallArgs: map[string]any{"filter": map[string]any(nil), "tags": []any(nil)},
		},
		{
			name:         "on-tool-error callback edits the arguments of an unknown tool",
			toolName:     "missing",
			callArgs:     map[string]any{"input_str": "original"},
			onToolError:  echoArgs,
			wantResponse: map[string]any{"args": map[string]any{"input_str": "modified_input"}},
			wantCallArgs: map[string]any{"input_str": "original"},
		},
		{
			name:         "on-tool-error callback edits the arguments of a tool that cannot run",
			toolName:     "plain",
			callArgs:     map[string]any{"input_str": "original"},
			onToolError:  echoArgs,
			wantResponse: map[string]any{"args": map[string]any{"input_str": "modified_input"}},
			wantCallArgs: map[string]any{"input_str": "original"},
		},
		{
			name:         "streaming tool edits its arguments",
			toolName:     "edit_stream",
			callArgs:     map[string]any{"input_str": "original"},
			wantResponse: map[string]any{"result": "modified_input"},
			wantCallArgs: map[string]any{"input_str": "original"},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			f := &Flow{}
			if tc.before != nil {
				f.BeforeToolCallbacks = []BeforeToolCallback{tc.before}
			}
			if tc.onToolError != nil {
				f.OnToolErrorCallbacks = []OnToolErrorCallback{tc.onToolError}
			}
			call := &genai.FunctionCall{ID: "call_1", Name: tc.toolName, Args: tc.callArgs}
			resp := &model.LLMResponse{Content: &genai.Content{Role: "model", Parts: []*genai.Part{{FunctionCall: call}}}}
			toolsDict := map[string]tool.Tool{"echo": echoTool, "plain": &fakeTool{name: "plain"}, "edit_stream": argEditingStreamTool{}}
			invCtx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{
				InvocationID: "inv_1",
				Agent:        &mockAgent{name: "agent_1"},
			})

			ev, err := f.handleFunctionCalls(invCtx, toolsDict, resp, nil, nil)
			if err != nil {
				t.Fatalf("handleFunctionCalls() error = %v", err)
			}

			if diff := cmp.Diff(tc.wantResponse, ev.LLMResponse.Content.Parts[0].FunctionResponse.Response); diff != "" {
				t.Errorf("function response mismatch (-want +got):\n%s", diff)
			}
			if diff := cmp.Diff(tc.wantCallArgs, call.Args); diff != "" {
				t.Errorf("function call args mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// checkPanicCallbackErrs asserts that a callback of each of the given kinds
// recorded itself in errs, and was handed the panic value and the stack, which
// callbacks need to diagnose a panic the model only sees as a generic message.
func checkPanicCallbackErrs(errs map[string]error, kinds ...string) func(t *testing.T) {
	return func(t *testing.T) {
		t.Helper()
		for _, kind := range kinds {
			err, ok := errs[kind]
			if !ok {
				t.Errorf("%s callback was not called for the panic", kind)
				continue
			}
			if err == nil {
				t.Errorf("%s callback got a nil error for the panic", kind)
				continue
			}
			for _, want := range []string{"boom", "stack:", "runtime/debug.Stack"} {
				if !strings.Contains(err.Error(), want) {
					t.Errorf("%s callback error is missing %q", kind, want)
				}
			}
		}
	}
}

func TestDisplayableToolResultText(t *testing.T) {
	tests := []struct {
		name     string
		result   map[string]any
		wantText string
		wantOK   bool
	}{
		{
			name:   "empty result is not displayable",
			result: map[string]any{},
			wantOK: false,
		},
		{
			name:   "error result is not displayable",
			result: map[string]any{"error": "boom"},
			wantOK: false,
		},
		{
			name:   "empty string result is not displayable",
			result: map[string]any{"result": ""},
			wantOK: false,
		},
		{
			name:   "nil result is not displayable",
			result: map[string]any{"result": nil},
			wantOK: false,
		},
		{
			name:     "single plain-text result key is shown as-is",
			result:   map[string]any{"result": "sub-agent output"},
			wantText: "sub-agent output",
			wantOK:   true,
		},
		{
			name:     "non-string result key is rendered as JSON",
			result:   map[string]any{"result": 42},
			wantText: `{"result":42}`,
			wantOK:   true,
		},
		{
			name:     "structured result is rendered as JSON",
			result:   map[string]any{"status": "ok", "count": 3},
			wantText: `{"count":3,"status":"ok"}`,
			wantOK:   true,
		},
		{
			name:     "JSON output is not HTML-escaped",
			result:   map[string]any{"url": "https://example.com/a&b<c>"},
			wantText: `{"url":"https://example.com/a&b<c>"}`,
			wantOK:   true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			gotText, gotOK := displayableToolResultText(tc.result)
			if gotOK != tc.wantOK {
				t.Fatalf("displayableToolResultText(%v) ok = %v, want %v", tc.result, gotOK, tc.wantOK)
			}
			if gotOK && gotText != tc.wantText {
				t.Errorf("displayableToolResultText(%v) = %q, want %q", tc.result, gotText, tc.wantText)
			}
		})
	}
}

func TestMergeEventActions(t *testing.T) {
	tests := []struct {
		name  string
		base  *session.EventActions
		other *session.EventActions
		want  *session.EventActions
	}{
		{
			name:  "both nil",
			base:  nil,
			other: nil,
			want:  nil,
		},
		{
			name: "other nil returns base",
			base: &session.EventActions{
				StateDelta: map[string]any{"key1": "value1"},
			},
			other: nil,
			want: &session.EventActions{
				StateDelta: map[string]any{"key1": "value1"},
			},
		},
		{
			name: "base nil returns other",
			base: nil,
			other: &session.EventActions{
				StateDelta: map[string]any{"key1": "value1"},
			},
			want: &session.EventActions{
				StateDelta: map[string]any{"key1": "value1"},
			},
		},
		{
			name: "state delta merged with non-overlapping keys",
			base: &session.EventActions{
				StateDelta: map[string]any{"key1": "value1"},
			},
			other: &session.EventActions{
				StateDelta: map[string]any{"key2": "value2"},
			},
			want: &session.EventActions{
				StateDelta: map[string]any{"key1": "value1", "key2": "value2"},
			},
		},
		{
			name: "state delta merged with overlapping keys - later wins",
			base: &session.EventActions{
				StateDelta: map[string]any{"key1": "original"},
			},
			other: &session.EventActions{
				StateDelta: map[string]any{"key1": "overwritten"},
			},
			want: &session.EventActions{
				StateDelta: map[string]any{"key1": "overwritten"},
			},
		},
		{
			name: "state delta merged with nested map values",
			base: &session.EventActions{
				StateDelta: map[string]any{
					"outer": map[string]any{"key1": "value1", "key2": "value2"},
				},
			},
			other: &session.EventActions{
				StateDelta: map[string]any{
					"outer": map[string]any{"key2": "updated", "key3": "value3"},
				},
			},
			want: &session.EventActions{
				StateDelta: map[string]any{
					"outer": map[string]any{"key1": "value1", "key2": "updated", "key3": "value3"},
				},
			},
		},
		{
			name: "state delta merged with multiple keys from multiple tools",
			base: &session.EventActions{
				StateDelta: map[string]any{"tool1_key": "tool1_value"},
			},
			other: &session.EventActions{
				StateDelta: map[string]any{"tool2_key": "tool2_value", "tool3_key": "tool3_value"},
			},
			want: &session.EventActions{
				StateDelta: map[string]any{
					"tool1_key": "tool1_value",
					"tool2_key": "tool2_value",
					"tool3_key": "tool3_value",
				},
			},
		},
		{
			name: "base has nil state delta, other has values",
			base: &session.EventActions{
				SkipSummarization: true,
			},
			other: &session.EventActions{
				StateDelta: map[string]any{"key1": "value1"},
			},
			want: &session.EventActions{
				SkipSummarization: true,
				StateDelta:        map[string]any{"key1": "value1"},
			},
		},
		{
			name: "base has artifact delta, other has nil",
			base: &session.EventActions{
				ArtifactDelta: map[string]int64{"file1": 1, "file2": 2},
			},
			other: &session.EventActions{
				ArtifactDelta: nil,
			},
			want: &session.EventActions{
				ArtifactDelta: map[string]int64{"file1": 1, "file2": 2},
			},
		},
		{
			name: "base has nil artifact delta, other has values",
			base: &session.EventActions{
				ArtifactDelta: nil,
			},
			other: &session.EventActions{
				ArtifactDelta: map[string]int64{"file1": 1, "file2": 2},
			},
			want: &session.EventActions{
				ArtifactDelta: map[string]int64{"file1": 1, "file2": 2},
			},
		},
		{
			name: "artifact delta merged with non-overlapping keys",
			base: &session.EventActions{
				ArtifactDelta: map[string]int64{"file1": 1, "file2": 2},
			},
			other: &session.EventActions{
				ArtifactDelta: map[string]int64{"file3": 3, "file4": 4},
			},
			want: &session.EventActions{
				ArtifactDelta: map[string]int64{"file1": 1, "file2": 2, "file3": 3, "file4": 4},
			},
		},
		{
			name: "artifact delta merged with overlapping keys",
			base: &session.EventActions{
				ArtifactDelta: map[string]int64{"file1": 1, "file2": 100},
			},
			other: &session.EventActions{
				ArtifactDelta: map[string]int64{"file1": 10, "file2": 10},
			},
			want: &session.EventActions{
				ArtifactDelta: map[string]int64{"file1": 10, "file2": 100},
			},
		},
		{
			name: "artifact deltas merged with partially overlapping keys",
			base: &session.EventActions{
				ArtifactDelta: map[string]int64{"file1": 1, "file2": 100},
			},
			other: &session.EventActions{
				ArtifactDelta: map[string]int64{"file2": 2, "file3": 3},
			},
			want: &session.EventActions{
				ArtifactDelta: map[string]int64{"file1": 1, "file2": 100, "file3": 3},
			},
		},
		{
			name: "artifact deltas merged with negative version preserved",
			base: &session.EventActions{
				ArtifactDelta: nil,
			},
			other: &session.EventActions{
				ArtifactDelta: map[string]int64{"file1": -1},
			},
			want: &session.EventActions{
				ArtifactDelta: map[string]int64{"file1": -1},
			},
		},
		{
			name: "skip summarization merging - any true wins",
			base: &session.EventActions{
				SkipSummarization: false,
			},
			other: &session.EventActions{
				SkipSummarization: true,
			},
			want: &session.EventActions{
				SkipSummarization: true,
			},
		},
		{
			name: "escalate merging - any true wins",
			base: &session.EventActions{
				Escalate: false,
			},
			other: &session.EventActions{
				Escalate: true,
			},
			want: &session.EventActions{
				Escalate: true,
			},
		},
		{
			name: "transfer to agent - last wins",
			base: &session.EventActions{
				TransferToAgent: "agent1",
			},
			other: &session.EventActions{
				TransferToAgent: "agent2",
			},
			want: &session.EventActions{
				TransferToAgent: "agent2",
			},
		},
		{
			name: "all fields merged correctly",
			base: &session.EventActions{
				StateDelta:        map[string]any{"key1": "value1"},
				ArtifactDelta:     map[string]int64{"file1": 1},
				SkipSummarization: false,
				TransferToAgent:   "agent1",
				Escalate:          false,
			},
			other: &session.EventActions{
				StateDelta:        map[string]any{"key2": "value2"},
				ArtifactDelta:     map[string]int64{"file2": 2},
				SkipSummarization: true,
				TransferToAgent:   "agent2",
				Escalate:          true,
			},
			want: &session.EventActions{
				StateDelta:        map[string]any{"key1": "value1", "key2": "value2"},
				ArtifactDelta:     map[string]int64{"file1": 1, "file2": 2},
				SkipSummarization: true,
				TransferToAgent:   "agent2",
				Escalate:          true,
			},
		},
		{
			// Each parallel call records its own request under its own call ID,
			// and the user has to answer all of them.
			name: "requested auth configs from both calls kept",
			base: &session.EventActions{
				RequestedAuthConfigs: map[string]toolauth.Config{"call-a": {CredentialKey: "key-a"}},
			},
			other: &session.EventActions{
				RequestedAuthConfigs: map[string]toolauth.Config{"call-b": {CredentialKey: "key-b"}},
			},
			want: &session.EventActions{
				RequestedAuthConfigs: map[string]toolauth.Config{
					"call-a": {CredentialKey: "key-a"},
					"call-b": {CredentialKey: "key-b"},
				},
			},
		},
		{
			name: "requested auth configs kept when only the later call asks",
			base: &session.EventActions{StateDelta: map[string]any{"key1": "value1"}},
			other: &session.EventActions{
				RequestedAuthConfigs: map[string]toolauth.Config{"call-b": {CredentialKey: "key-b"}},
			},
			want: &session.EventActions{
				StateDelta:           map[string]any{"key1": "value1"},
				RequestedAuthConfigs: map[string]toolauth.Config{"call-b": {CredentialKey: "key-b"}},
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := mergeEventActions(tc.base, tc.other)
			if diff := cmp.Diff(tc.want, got); diff != "" {
				t.Errorf("mergeEventActions() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestPreprocess_Toolset(t *testing.T) {
	noOpAgent, err := agent.New(agent.Config{Name: "no-op"})
	if err != nil {
		t.Fatalf("Failed to create agent: %v", err)
	}

	tests := []struct {
		name      string
		agent     agent.Agent
		wantModel string
		wantError bool
	}{
		{
			name:      "agent not llminternal.Agent",
			agent:     noOpAgent,
			wantError: false,
		},
		{
			name:      "agent has no toolsets",
			agent:     &mockLLMAgent{s: &State{}},
			wantError: false,
		},
		{
			name: "toolset implements RequestProcessor, error",
			agent: &mockLLMAgent{
				s: &State{
					Toolsets: []tool.Toolset{&mockRequestProcessorToolset{
						name: "toolset",
						process: func(_ agent.Context, _ *model.LLMRequest) error {
							return errors.New("process error")
						},
					}},
				},
			},
			wantError: true,
		},
		{
			name: "toolsets, success",
			agent: &mockLLMAgent{
				s: &State{
					Toolsets: []tool.Toolset{
						&mockToolset{name: "toolset_without_processor"},
						&mockRequestProcessorToolset{
							name: "toolset_with_processor",
							process: func(_ agent.Context, req *model.LLMRequest) error {
								req.Model = "modified-model"
								return nil
							},
						},
					},
				},
			},
			wantError: false,
			wantModel: "modified-model",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			f := &Flow{}
			ctx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{Agent: tc.agent})
			req := &model.LLMRequest{}

			events := f.preprocess(ctx, req)

			var gotErr error
			for _, err := range events {
				if err != nil {
					gotErr = err
					break
				}
			}
			if (gotErr != nil) != tc.wantError {
				t.Errorf("preprocess() error = %v, wantError %v", gotErr, tc.wantError)
			}
			if req.Model != tc.wantModel {
				t.Errorf("preprocess() model = %s, wantModel %s", req.Model, tc.wantModel)
			}
		})
	}
}

// TestPreprocessStopsWhenTheConsumerDoes covers a request processor that
// yields two events, as the auth preprocessor does when a tool it runs again
// asks for another credential. A consumer that stops after the first event,
// such as a runner whose session store fails to append it, must not be called
// again: Go panics when a range-over-func loop body runs after it returned.
func TestPreprocessStopsWhenTheConsumerDoes(t *testing.T) {
	twoEvents := func(agent.InvocationContext, *model.LLMRequest, *Flow) iter.Seq2[*session.Event, error] {
		return func(yield func(*session.Event, error) bool) {
			if yield(session.NewEvent(t.Context(), "inv"), nil) {
				yield(session.NewEvent(t.Context(), "inv"), nil)
			}
		}
	}
	f := &Flow{RequestProcessors: []func(agent.InvocationContext, *model.LLMRequest, *Flow) iter.Seq2[*session.Event, error]{twoEvents}}
	ctx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{Agent: &mockLLMAgent{s: &State{}}})

	for _, err := range f.preprocess(ctx, &model.LLMRequest{}) {
		if err != nil {
			t.Fatalf("preprocess() error = %v", err)
		}
		break
	}
}

// fnRespEvent builds a function-response event carrying a single text part,
// mirroring what the parallel-call producer emits for a normal tool.
func fnRespEvent(t *testing.T, text string) *session.Event {
	t.Helper()
	ev := session.NewEvent(t.Context(), "inv")
	ev.LLMResponse = model.LLMResponse{
		Content: &genai.Content{
			Role:  "user",
			Parts: []*genai.Part{{Text: text}},
		},
	}
	return ev
}

// TestMergeParallelFunctionResponseEvents_NilEntries guards that nil slots
// (left by long-running/deferred tools that return early) don't panic the
// merge. Pre-fix, a nil events[0] or an all-nil slice dereferenced nil.
func TestMergeParallelFunctionResponseEvents_NilEntries(t *testing.T) {
	t.Run("first entry nil", func(t *testing.T) {
		got, err := mergeParallelFunctionResponseEvents([]*session.Event{nil, fnRespEvent(t, "b")})
		if err != nil {
			t.Fatalf("merge error: %v", err)
		}
		if got == nil || got.LLMResponse.Content == nil {
			t.Fatalf("got nil/empty merged event: %#v", got)
		}
		if n := len(got.LLMResponse.Content.Parts); n != 1 {
			t.Errorf("merged parts = %d, want 1", n)
		}
	})

	t.Run("all entries nil", func(t *testing.T) {
		got, err := mergeParallelFunctionResponseEvents([]*session.Event{nil, nil})
		if err != nil {
			t.Fatalf("merge error: %v", err)
		}
		if got != nil {
			t.Errorf("merged event = %#v, want nil", got)
		}
	})

	t.Run("mixed nil and non-nil", func(t *testing.T) {
		got, err := mergeParallelFunctionResponseEvents([]*session.Event{fnRespEvent(t, "a"), nil, fnRespEvent(t, "c")})
		if err != nil {
			t.Fatalf("merge error: %v", err)
		}
		if got == nil || got.LLMResponse.Content == nil {
			t.Fatalf("got nil/empty merged event: %#v", got)
		}
		if n := len(got.LLMResponse.Content.Parts); n != 2 {
			t.Errorf("merged parts = %d, want 2", n)
		}
	})
}

func TestIsThoughtOnlyTurn(t *testing.T) {
	event := func(partial bool, parts ...*genai.Part) *session.Event {
		var content *genai.Content
		if parts != nil {
			content = &genai.Content{Role: "model", Parts: parts}
		}
		return &session.Event{LLMResponse: model.LLMResponse{Content: content, Partial: partial}}
	}

	tests := []struct {
		name string
		ev   *session.Event
		want bool
	}{
		{"thought_text_only", event(false, &genai.Part{Thought: true, Text: "thinking"}), true},
		{"thought_plus_answer", event(false, &genai.Part{Thought: true, Text: "t"}, &genai.Part{Text: "answer"}), false},
		{"answer_only", event(false, &genai.Part{Text: "answer"}), false},
		{"function_call", event(false, &genai.Part{FunctionCall: &genai.FunctionCall{Name: "f"}}), false},
		{"thought_then_signed_call", event(false, &genai.Part{Thought: true, Text: "t"}, &genai.Part{FunctionCall: &genai.FunctionCall{Name: "f"}, ThoughtSignature: []byte("sig")}), false},
		{"thought_plus_signature_only_part", event(false, &genai.Part{Thought: true, Text: "t"}, &genai.Part{ThoughtSignature: []byte("sig")}), false},
		{"partial_thought", event(true, &genai.Part{Thought: true, Text: "t"}), false},
		{"thought_with_error_code", &session.Event{LLMResponse: model.LLMResponse{
			Content:   &genai.Content{Role: "model", Parts: []*genai.Part{{Thought: true, Text: "t"}}},
			ErrorCode: "MODEL_RETURNED_NO_CONTENT",
		}}, false},
		{"empty_content", event(false), false},
		{"nil_event", nil, false},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := isThoughtOnlyTurn(tc.ev); got != tc.want {
				t.Errorf("isThoughtOnlyTurn = %v, want %v", got, tc.want)
			}
		})
	}
}

// alwaysThinkingModel always returns a completed thought-only ("thinking")
// turn and never surfaces an answer. Real models can get stuck in this state,
// so the flow must not call the model forever waiting for an answer.
type alwaysThinkingModel struct{ calls int }

func (m *alwaysThinkingModel) Name() string { return "always-thinking" }

func (m *alwaysThinkingModel) GenerateContent(ctx context.Context, req *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		m.calls++
		// Number each thought so a test can tell which turn's event the flow
		// left as the result.
		yield(&model.LLMResponse{
			Content: &genai.Content{
				Role:  "model",
				Parts: []*genai.Part{{Text: fmt.Sprintf("thinking %d", m.calls), Thought: true}},
			},
		}, nil)
	}
}

// TestCallLLMStreamingModeFromRunConfig pins how callLLM derives the streaming
// flag, including the nil case (issue #586).
//
// None of these store a runconfig in the Go context, as when agent.Run() is
// invoked directly and the runner is bypassed, so runconfig.FromContext would
// return nil; the streaming mode is read from the invocation context instead.
// An invocation context built outside the runner need not carry a RunConfig at
// all, and that has to resolve to false rather than panic.
func TestCallLLMStreamingModeFromRunConfig(t *testing.T) {
	tests := []struct {
		name      string
		runConfig *agent.RunConfig
		want      bool
	}{
		{
			name:      "SSE streams",
			runConfig: &agent.RunConfig{StreamingMode: agent.StreamingModeSSE},
			want:      true,
		},
		{
			name:      "none does not stream",
			runConfig: &agent.RunConfig{StreamingMode: agent.StreamingModeNone},
			want:      false,
		},
		{
			name:      "unset streaming mode does not stream",
			runConfig: &agent.RunConfig{},
			want:      false,
		},
		{
			name:      "nil run config does not stream",
			runConfig: nil,
			want:      false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var calls int
			var gotStream bool
			m := &mockModelForTest{
				name: "test-model",
				generateContent: func(_ context.Context, _ *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error] {
					calls++
					gotStream = stream
					return func(func(*model.LLMResponse, error) bool) {}
				},
			}
			f := &Flow{Model: m}

			ctx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{
				RunConfig: tc.runConfig,
			})

			req := &model.LLMRequest{}
			for _, err := range f.callLLM(ctx, req, map[string]any{}, map[string]int64{}) {
				if err != nil {
					t.Fatalf("callLLM() error = %v, want nil", err)
				}
			}

			if calls != 1 {
				t.Fatalf("GenerateContent called %d times, want 1", calls)
			}
			if gotStream != tc.want {
				t.Errorf("GenerateContent received stream=%v, want %v", gotStream, tc.want)
			}
		})
	}
}

func TestRun_ThoughtOnlyTurnsTerminate(t *testing.T) {
	m := &alwaysThinkingModel{}
	f := &Flow{Model: m}
	ctx := icontext.NewInvocationContext(
		runconfig.ToContext(t.Context(), &runconfig.RunConfig{}),
		icontext.InvocationContextParams{
			InvocationID: "inv_1",
			Agent:        &mockAgent{name: "agent_1"},
		},
	)

	// Bound the consumer so a regression fails the test instead of hanging it.
	const safetyLimit = 50
	events := 0
	var lastEvent *session.Event
	for ev, err := range f.Run(ctx) {
		if err != nil {
			t.Fatalf("Run() yielded error: %v", err)
		}
		events++
		lastEvent = ev
		if events > safetyLimit {
			break
		}
	}

	if events > safetyLimit {
		t.Fatalf("Run() did not terminate on repeated thought-only turns: yielded >%d events, model called %d times", safetyLimit, m.calls)
	}
	if m.calls != maxConsecutiveThoughtOnlyTurns {
		t.Errorf("model called %d times, want %d (maxConsecutiveThoughtOnlyTurns)", m.calls, maxConsecutiveThoughtOnlyTurns)
	}
	if events != maxConsecutiveThoughtOnlyTurns {
		t.Errorf("Run() yielded %d events, want %d (one per thought-only turn)", events, maxConsecutiveThoughtOnlyTurns)
	}

	// Giving up leaves the last thinking event as the result. The literal
	// "thinking 10" pins both that promise and the value of the bound: it is
	// the tenth turn's thought, so an off-by-one or a changed bound fails here.
	if lastEvent == nil || lastEvent.LLMResponse.Content == nil {
		t.Fatalf("Run() left no content as the result: %#v", lastEvent)
	}
	wantParts := []*genai.Part{{Text: "thinking 10", Thought: true}}
	if diff := cmp.Diff(wantParts, lastEvent.LLMResponse.Content.Parts); diff != "" {
		t.Errorf("last event parts mismatch (-want +got):\n%s", diff)
	}
}

// scriptedThinkingModel replays a fixed sequence of turns and then keeps
// repeating the final one, so a test can interleave thought-only turns with a
// turn that is not a final response.
type scriptedThinkingModel struct {
	turns []*model.LLMResponse
	calls int
}

func (m *scriptedThinkingModel) Name() string { return "scripted-thinking" }

func (m *scriptedThinkingModel) GenerateContent(ctx context.Context, req *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		i := m.calls
		m.calls++
		if i >= len(m.turns) {
			i = len(m.turns) - 1
		}
		yield(m.turns[i], nil)
	}
}

func thoughtTurn() *model.LLMResponse {
	return &model.LLMResponse{Content: &genai.Content{
		Role:  "model",
		Parts: []*genai.Part{{Text: "thinking", Thought: true}},
	}}
}

// TestRun_ThoughtOnlyTurnCountResets pins the reset in Flow.Run: the cap counts
// *consecutive* thought-only turns, so a turn that is not a final response
// (here a function call) must clear the count. Without the reset the cap
// becomes cumulative and the run stops after maxConsecutiveThoughtOnlyTurns
// thought-only turns in total rather than in a row.
func TestRun_ThoughtOnlyTurnCountResets(t *testing.T) {
	const toolName = "noop"
	// Two thoughts, a function call (not a final response, so it resets the
	// count), then thoughts forever.
	m := &scriptedThinkingModel{turns: []*model.LLMResponse{
		thoughtTurn(),
		thoughtTurn(),
		{Content: &genai.Content{
			Role:  "model",
			Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{ID: "fc-1", Name: toolName, Args: map[string]any{}}}},
		}},
		thoughtTurn(),
	}}
	f := &Flow{
		Model: m,
		Tools: []tool.Tool{&mockFunctionTool{name: toolName}},
	}
	ctx := icontext.NewInvocationContext(
		runconfig.ToContext(t.Context(), &runconfig.RunConfig{}),
		icontext.InvocationContextParams{
			InvocationID: "inv_1",
			Agent:        &mockAgent{name: "agent_1"},
		},
	)

	const safetyLimit = 50
	events := 0
	for _, err := range f.Run(ctx) {
		if err != nil {
			t.Fatalf("Run() yielded error: %v", err)
		}
		events++
		if events > safetyLimit {
			break
		}
	}
	if events > safetyLimit {
		t.Fatalf("Run() did not terminate: yielded >%d events, model called %d times", safetyLimit, m.calls)
	}

	// 2 thought-only turns, then the function call resets the count, then a
	// further maxConsecutiveThoughtOnlyTurns thought-only turns.
	want := 2 + 1 + maxConsecutiveThoughtOnlyTurns
	if m.calls != want {
		t.Errorf("model called %d times, want %d; the consecutive-turn count did not reset on the non-final turn", m.calls, want)
	}
}

// A response without content is dropped unless it carries something the
// caller needs, such as grounding metadata from a search tool.
func TestRunKeepsGroundingOnlyResponse(t *testing.T) {
	grounding := &genai.GroundingMetadata{WebSearchQueries: []string{"test query"}}
	tests := []struct {
		name string
		resp *model.LLMResponse
		want []*genai.GroundingMetadata
	}{
		{
			name: "grounding metadata only",
			resp: &model.LLMResponse{GroundingMetadata: grounding},
			want: []*genai.GroundingMetadata{grounding},
		},
		{
			name: "empty response",
			resp: &model.LLMResponse{},
			want: nil,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			m := &mockModelForTest{
				name: "test-model",
				generateContent: func(context.Context, *model.LLMRequest, bool) iter.Seq2[*model.LLMResponse, error] {
					return func(yield func(*model.LLMResponse, error) bool) {
						yield(tc.resp, nil)
					}
				},
			}
			f := &Flow{Model: m}
			ctx := icontext.NewInvocationContext(
				runconfig.ToContext(t.Context(), &runconfig.RunConfig{}),
				icontext.InvocationContextParams{
					InvocationID: "inv_1",
					Agent:        &mockAgent{name: "agent_1"},
				},
			)

			var got []*genai.GroundingMetadata
			for ev, err := range f.Run(ctx) {
				if err != nil {
					t.Fatalf("Run() yielded error: %v", err)
				}
				got = append(got, ev.GroundingMetadata)
			}
			if diff := cmp.Diff(tc.want, got); diff != "" {
				t.Errorf("Run() event grounding metadata mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// TestFlowRunPartialLastEvent verifies that Flow.Run does not return an error
// when the last event from runOneStep has Partial=true.
// This is a regression test for https://github.com/google/adk-go/issues/600.
func TestFlowRunPartialLastEvent(t *testing.T) {
	tests := []struct {
		name              string
		modelResponses    []*model.LLMResponse
		requestProcessors []func(ctx agent.InvocationContext, req *model.LLMRequest, f *Flow) iter.Seq2[*session.Event, error]
		wantTexts         []string
		wantLogAuthor     string
	}{
		{
			name: "single partial response completes without error",
			modelResponses: []*model.LLMResponse{
				{
					Content: genai.NewContentFromText("Hello", genai.RoleModel),
					Partial: true,
				},
			},
			wantTexts:     []string{"Hello"},
			wantLogAuthor: "test-agent",
		},
		{
			name: "multiple partial responses complete without error",
			modelResponses: []*model.LLMResponse{
				{
					Content: genai.NewContentFromText("Hello", genai.RoleModel),
					Partial: true,
				},
				{
					Content: genai.NewContentFromText(" World", genai.RoleModel),
					Partial: true,
				},
			},
			wantTexts:     []string{"Hello", " World"},
			wantLogAuthor: "test-agent",
		},
		{
			name: "trailing partial from sub-agent completes without error",
			requestProcessors: []func(ctx agent.InvocationContext, req *model.LLMRequest, f *Flow) iter.Seq2[*session.Event, error]{
				func(ctx agent.InvocationContext, req *model.LLMRequest, f *Flow) iter.Seq2[*session.Event, error] {
					return func(yield func(*session.Event, error) bool) {
						ev := session.NewEvent(ctx, ctx.InvocationID())
						ev.Author = "sub-agent"
						ev.LLMResponse = model.LLMResponse{
							Content: genai.NewContentFromText("I am a sub-agent", genai.RoleModel),
							Partial: true,
						}
						yield(ev, nil)
					}
				},
			},
			wantTexts:     []string{"I am a sub-agent"},
			wantLogAuthor: "sub-agent",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			mockAgent, err := agent.New(agent.Config{Name: "test-agent"})
			if err != nil {
				t.Fatal(err)
			}

			ctx := runconfig.ToContext(t.Context(), &runconfig.RunConfig{
				StreamingMode: runconfig.StreamingModeSSE,
			})

			invCtx := icontext.NewInvocationContext(ctx, icontext.InvocationContextParams{
				Agent: mockAgent,
			})

			m := &mockModelForTest{
				name: "test-model",
				generateContent: func(_ context.Context, _ *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
					return func(yield func(*model.LLMResponse, error) bool) {
						for _, r := range tc.modelResponses {
							if !yield(r, nil) {
								return
							}
						}
					}
				},
			}

			f := &Flow{
				Model:             m,
				RequestProcessors: tc.requestProcessors,
			}

			var gotTexts []string
			var gotErr error
			// Bound the consumer so a regression fails the test instead of hanging it.
			const safetyLimit = 50
			eventsCount := 0

			logOutput := captureLog(t, func() {
				for ev, err := range f.Run(invCtx) {
					if err != nil {
						gotErr = err
						break
					}
					eventsCount++
					if eventsCount > safetyLimit {
						t.Fatalf("Flow.Run() did not terminate: yielded >%d events", safetyLimit)
					}
					if ev != nil && ev.Content != nil {
						for _, p := range ev.Content.Parts {
							if p.Text != "" {
								gotTexts = append(gotTexts, p.Text)
							}
						}
					}
				}
			})

			if gotErr != nil {
				t.Errorf("Flow.Run() returned unexpected error: %v", gotErr)
			}

			if diff := cmp.Diff(tc.wantTexts, gotTexts); diff != "" {
				t.Errorf("Flow.Run() text mismatch (-want +got):\n%s", diff)
			}

			wantLog := fmt.Sprintf("adk: agent %q (invocation %q): step ended on a partial event from %q; the producer did not close its stream with an aggregated final event, so the turn will not appear in session history\n",
				invCtx.Agent().Name(), invCtx.InvocationID(), tc.wantLogAuthor)
			if logOutput != wantLog {
				t.Errorf("Flow.Run() log mismatch:\nwant: %q\ngot:  %q", wantLog, logOutput)
			}
		})
	}
}

// captureLog redirects the output of the default [log.Logger] for the
// duration of fn and returns everything that was written to it.
func captureLog(t *testing.T, fn func()) string {
	t.Helper()
	var buf bytes.Buffer
	origOut := log.Writer()
	origFlags := log.Flags()
	origPrefix := log.Prefix()
	log.SetOutput(&buf)
	// Reset flags/prefix so the captured output is deterministic and easy to
	// assert against.
	log.SetFlags(0)
	log.SetPrefix("")
	t.Cleanup(func() {
		log.SetOutput(origOut)
		log.SetFlags(origFlags)
		log.SetPrefix(origPrefix)
	})
	fn()
	return buf.String()
}

// dynamicToolset returns different tools on each call to simulate a toolset
// whose output depends on session state written by an earlier tool in the run.
type dynamicToolset struct {
	callCount   int
	toolsByCall map[int][]tool.Tool
}

func (d *dynamicToolset) Name() string { return "dynamic" }
func (d *dynamicToolset) Tools(_ agent.ReadonlyContext) ([]tool.Tool, error) {
	tools := d.toolsByCall[d.callCount]
	d.callCount++
	return tools, nil
}

// TestToolProcessorReEvaluatesToolsetsEachStep verifies that toolProcessor
// calls Toolset.Tools() on every step, not just the first. This ensures that
// tools activated by session-state changes in an earlier step are visible to
// subsequent model calls within the same Runner.Run().
func TestToolProcessorReEvaluatesToolsetsEachStep(t *testing.T) {
	extraTool := &mockFunctionTool{name: "extra_tool"}

	// Call 0: no tools yet. Call 1+: return extraTool.
	ts := &dynamicToolset{
		toolsByCall: map[int][]tool.Tool{
			0: nil,
			1: {extraTool},
		},
	}

	baseTool := &mockFunctionTool{name: "base_tool"}
	agentState := &State{Tools: []tool.Tool{baseTool}, Toolsets: []tool.Toolset{ts}}
	mockAgent := &mockLLMAgent{s: agentState}
	ctx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{Agent: mockAgent})

	f := &Flow{}

	// First call: toolset returns nil, so f.Tools holds only the static tool.
	// A non-nil f.Tools is what would trip a per-run cache guard.
	req1 := &model.LLMRequest{}
	for _, err := range toolProcessor(ctx, req1, f) {
		if err != nil {
			t.Fatalf("toolProcessor call 1 error: %v", err)
		}
	}
	if len(f.Tools) != 1 {
		t.Errorf("after call 1: got %d tools, want 1", len(f.Tools))
	}

	// Second call: toolset now returns extraTool — f.Tools must be updated.
	req2 := &model.LLMRequest{}
	for _, err := range toolProcessor(ctx, req2, f) {
		if err != nil {
			t.Fatalf("toolProcessor call 2 error: %v", err)
		}
	}
	if len(f.Tools) != 2 || f.Tools[1].Name() != extraTool.Name() {
		t.Errorf("after call 2: got tools %v, want [%s %s]", f.Tools, baseTool.Name(), extraTool.Name())
	}

	if ts.callCount != 2 {
		t.Errorf("toolset.Tools() called %d times, want 2", ts.callCount)
	}
}

func TestCallLLMSkipsNilResponseAndContinues(t *testing.T) {
	tests := []struct {
		name          string
		responses     []*model.LLMResponse
		wantResponses int
	}{
		{
			name:          "nil only",
			responses:     []*model.LLMResponse{nil},
			wantResponses: 0,
		},
		{
			name: "nil then final",
			responses: []*model.LLMResponse{
				nil,
				{Content: &genai.Content{Role: "model", Parts: []*genai.Part{{Text: "done"}}}},
			},
			wantResponses: 1,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			m := &mockModelForTest{
				name: "test-model",
				generateContent: func(context.Context, *model.LLMRequest, bool) iter.Seq2[*model.LLMResponse, error] {
					return func(yield func(*model.LLMResponse, error) bool) {
						for _, resp := range tc.responses {
							if !yield(resp, nil) {
								return
							}
						}
					}
				},
			}
			f := &Flow{Model: m}
			ctx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{})

			gotResponses := 0
			for _, err := range f.callLLM(ctx, &model.LLMRequest{}, map[string]any{}, map[string]int64{}) {
				if err != nil {
					t.Fatalf("callLLM() returned unexpected error: %v", err)
				}
				gotResponses++
			}
			if gotResponses != tc.wantResponses {
				t.Fatalf("callLLM() yielded %d responses, want %d", gotResponses, tc.wantResponses)
			}
		})
	}
}

// A model response event's id comes from the UUID provider on the context.
// generateContent draws an id when a response arrives, never ahead of one, and
// draws one id for each distinct event id across the responses it yields.
func TestGenerateContentDrawsOneIDPerEvent(t *testing.T) {
	text := func(s string) *genai.Content { return genai.NewContentFromText(s, genai.RoleModel) }
	type yielded struct {
		resp *model.LLMResponse
		err  error
	}
	boom := errors.New("boom")
	for _, tc := range []struct {
		name      string
		responses []yielded
		wantIDs   []string
		wantErrs  []error
		wantDraws int
	}{
		{
			name:      "no responses draw no ids",
			responses: nil,
			wantIDs:   nil,
			wantErrs:  nil,
			wantDraws: 0,
		},
		{
			name:      "an empty response draws no ids",
			responses: []yielded{{}},
			wantIDs:   nil,
			wantErrs:  nil,
			wantDraws: 0,
		},
		{
			name:      "one complete response",
			responses: []yielded{{resp: &model.LLMResponse{Content: text("done")}}},
			wantIDs:   []string{"id-1"},
			wantErrs:  []error{nil},
			wantDraws: 1,
		},
		{
			name: "partials share the id of the response that completes them",
			responses: []yielded{
				{resp: &model.LLMResponse{Content: text("a"), Partial: true}},
				{resp: &model.LLMResponse{Content: text("b"), Partial: true}},
				{resp: &model.LLMResponse{Content: text("ab")}},
			},
			wantIDs:   []string{"id-1", "id-1", "id-1"},
			wantErrs:  []error{nil, nil, nil},
			wantDraws: 1,
		},
		{
			name: "each complete response gets its own id",
			responses: []yielded{
				{resp: &model.LLMResponse{Content: text("first")}},
				{resp: &model.LLMResponse{Content: text("second")}},
			},
			wantIDs:   []string{"id-1", "id-2"},
			wantErrs:  []error{nil, nil},
			wantDraws: 2,
		},
		{
			name: "an error ends the id",
			responses: []yielded{
				{resp: &model.LLMResponse{Content: text("a"), Partial: true}},
				{resp: &model.LLMResponse{Content: text("b"), Partial: true}, err: boom},
				{resp: &model.LLMResponse{Content: text("after")}},
			},
			wantIDs:   []string{"id-1", "id-1", "id-2"},
			wantErrs:  []error{nil, boom, nil},
			wantDraws: 2,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			draws := 0
			ctx := platform.WithUUIDProvider(t.Context(), func() string {
				draws++
				return fmt.Sprintf("id-%d", draws)
			})
			invCtx := icontext.NewInvocationContext(ctx, icontext.InvocationContextParams{InvocationID: "inv-1"})
			m := &mockModelForTest{
				name: "test-model",
				generateContent: func(context.Context, *model.LLMRequest, bool) iter.Seq2[*model.LLMResponse, error] {
					return func(yield func(*model.LLMResponse, error) bool) {
						for _, r := range tc.responses {
							if !yield(r.resp, r.err) {
								return
							}
						}
					}
				},
			}

			var gotIDs []string
			var gotErrs []error
			for resp, err := range generateContent(invCtx, m, &model.LLMRequest{}, true) {
				gotIDs = append(gotIDs, resp.eventID)
				gotErrs = append(gotErrs, err)
			}

			if diff := cmp.Diff(tc.wantIDs, gotIDs); diff != "" {
				t.Errorf("event ids mismatch (-want +got):\n%s", diff)
			}
			sameErr := cmp.Comparer(func(a, b error) bool { return a == b })
			if diff := cmp.Diff(tc.wantErrs, gotErrs, sameErr); diff != "" {
				t.Errorf("errors mismatch (-want +got):\n%s", diff)
			}
			if draws != tc.wantDraws {
				t.Errorf("drew %d ids from the provider, want %d", draws, tc.wantDraws)
			}
		})
	}
}

const (
	wantNoContentMessage           = "The model returned no content (finish_reason=STOP with empty parts)."
	wantNoActionableContentMessage = "The model returned no actionable content (finish_reason=STOP with thought-only or whitespace-only parts)."
)

// runScriptedFlow runs f.Run with a model that answers its i-th call with
// turns[i], and fails the test if the model is called more often than turns
// allows. It returns the yielded events and the number of model calls.
func runScriptedFlow(t *testing.T, f *Flow, runConfig *agent.RunConfig, turns ...*model.LLMResponse) ([]*session.Event, int) {
	t.Helper()
	calls := 0
	f.Model = &mockModelForTest{
		name: "scripted",
		generateContent: func(context.Context, *model.LLMRequest, bool) iter.Seq2[*model.LLMResponse, error] {
			return func(yield func(*model.LLMResponse, error) bool) {
				calls++
				if calls > len(turns) {
					t.Fatalf("model called %d times, want at most %d", calls, len(turns))
				}
				yield(turns[calls-1], nil)
			}
		},
	}
	ctx := icontext.NewInvocationContext(
		runconfig.ToContext(t.Context(), &runconfig.RunConfig{}),
		icontext.InvocationContextParams{
			InvocationID: "inv_1",
			Agent:        &mockAgent{name: "agent_1"},
			RunConfig:    runConfig,
		},
	)
	var events []*session.Event
	for ev, err := range f.Run(ctx) {
		if err != nil {
			t.Fatalf("Run() yielded error: %v", err)
		}
		events = append(events, ev)
	}
	return events, calls
}

func stopTurn(parts ...*genai.Part) *model.LLMResponse {
	return &model.LLMResponse{
		Content:      &genai.Content{Role: genai.RoleModel, Parts: parts},
		FinishReason: genai.FinishReasonStop,
	}
}

type eventError struct{ Code, Message string }

func errorsOf(events []*session.Event) []eventError {
	var got []eventError
	for _, ev := range events {
		got = append(got, eventError{ev.ErrorCode, ev.ErrorMessage})
	}
	return got
}

// TestHasMeaningfulContent follows adk-python's has_meaningful_content tests
// (flows/llm_flows/core/test_finalizer.py), among them
// test_has_meaningful_content_thought_only and
// test_has_meaningful_content_multiple_thoughts_only.
func TestHasMeaningfulContent(t *testing.T) {
	content := func(parts ...*genai.Part) *model.LLMResponse {
		return &model.LLMResponse{Content: &genai.Content{Role: genai.RoleModel, Parts: parts}}
	}
	tests := []struct {
		name string
		resp *model.LLMResponse
		want bool
	}{
		{"nil_response", nil, false},
		{"nil_content", &model.LLMResponse{}, false},
		{"empty_parts", content(), false},
		{"thought_only", content(&genai.Part{Text: "thinking", Thought: true}), false},
		{"multiple_thoughts_only", content(&genai.Part{Text: "step 1", Thought: true}, &genai.Part{Text: "step 2", Thought: true}), false},
		{"whitespace_only", content(&genai.Part{Text: " \n\t "}), false},
		{"thought_and_whitespace", content(&genai.Part{Text: "thinking", Thought: true}, &genai.Part{Text: "  "}), false},
		{"text", content(&genai.Part{Text: "answer"}), true},
		{"thought_and_text", content(&genai.Part{Text: "thinking", Thought: true}, &genai.Part{Text: "answer"}), true},
		{"function_call", content(&genai.Part{FunctionCall: &genai.FunctionCall{Name: "f"}}), true},
		{"function_response", content(&genai.Part{FunctionResponse: &genai.FunctionResponse{Name: "f"}}), true},
		{"executable_code", content(&genai.Part{ExecutableCode: &genai.ExecutableCode{Code: "x"}}), true},
		{"code_execution_result", content(&genai.Part{CodeExecutionResult: &genai.CodeExecutionResult{Output: "1"}}), true},
		{"inline_data", content(&genai.Part{InlineData: &genai.Blob{MIMEType: "image/png"}}), true},
		{"file_data", content(&genai.Part{FileData: &genai.FileData{FileURI: "gs://b/o"}}), true},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := hasMeaningfulContent(tc.resp); got != tc.want {
				t.Errorf("hasMeaningfulContent() = %t, want %t", got, tc.want)
			}
		})
	}
}

func TestEmptyResponsePolicy_MeaningfulContentParts(t *testing.T) {
	t.Run("function_response_is_meaningful", func(t *testing.T) {
		events, _ := runScriptedFlow(t, &Flow{}, nil,
			stopTurn(&genai.Part{FunctionResponse: &genai.FunctionResponse{ID: "call-1", Name: "tool", Response: map[string]any{"ok": true}}}),
			stopTurn(&genai.Part{Text: "done"}),
		)
		want := []eventError{{}, {}}
		if diff := cmp.Diff(want, errorsOf(events)); diff != "" {
			t.Errorf("event errors mismatch (-want +got):\n%s", diff)
		}
	})
	t.Run("nil_part_element_not_meaningful", func(t *testing.T) {
		events, calls := runScriptedFlow(t, &Flow{}, nil, stopTurn(nil))
		if calls != 1 {
			t.Errorf("model called %d times, want 1", calls)
		}
		want := []eventError{{"MODEL_RETURNED_NO_CONTENT", wantNoActionableContentMessage}}
		if diff := cmp.Diff(want, errorsOf(events)); diff != "" {
			t.Errorf("event errors mismatch (-want +got):\n%s", diff)
		}
	})
}

// TestEmptyResponsePolicy_ThoughtOnlyStopEndsRun follows adk-python's
// test_apply_empty_response_policy_marks_thought_only_stop[StreamingMode.NONE]
// (flows/llm_flows/core/test_model_call.py): a completed STOP turn that holds
// only thoughts becomes an error event, and the model is not called again.
func TestEmptyResponsePolicy_ThoughtOnlyStopEndsRun(t *testing.T) {
	events, calls := runScriptedFlow(t, &Flow{}, nil,
		stopTurn(&genai.Part{Text: "thinking...", Thought: true}),
		stopTurn(&genai.Part{Text: "answer"}),
	)
	if calls != 1 {
		t.Errorf("model called %d times, want 1", calls)
	}
	want := []eventError{{"MODEL_RETURNED_NO_CONTENT", wantNoActionableContentMessage}}
	if diff := cmp.Diff(want, errorsOf(events)); diff != "" {
		t.Errorf("event errors mismatch (-want +got):\n%s", diff)
	}
}

// TestEmptyResponsePolicy_ThoughtOnlyStopUnderSSE follows adk-python's
// test_thought_only_stop_in_sse_streaming_surfaces_error_event
// (flows/llm_flows/test_base_llm_flow.py): under SSE, the partial thought
// chunk carries no error, and the aggregated thought-only STOP response
// becomes the one error event.
func TestEmptyResponsePolicy_ThoughtOnlyStopUnderSSE(t *testing.T) {
	thought := func() *genai.Content {
		return &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{{Text: "Thinking step 1...", Thought: true}}}
	}
	calls := 0
	f := &Flow{Model: &mockModelForTest{
		name: "thought-stream",
		generateContent: func(context.Context, *model.LLMRequest, bool) iter.Seq2[*model.LLMResponse, error] {
			return func(yield func(*model.LLMResponse, error) bool) {
				calls++
				if !yield(&model.LLMResponse{Content: thought(), Partial: true}, nil) {
					return
				}
				yield(&model.LLMResponse{Content: thought(), FinishReason: genai.FinishReasonStop}, nil)
			}
		},
	}}
	ctx := icontext.NewInvocationContext(
		runconfig.ToContext(t.Context(), &runconfig.RunConfig{StreamingMode: runconfig.StreamingModeSSE}),
		icontext.InvocationContextParams{
			InvocationID: "inv_1",
			Agent:        &mockAgent{name: "agent_1"},
			RunConfig:    &agent.RunConfig{StreamingMode: agent.StreamingModeSSE},
		},
	)
	type eventSummary struct {
		Partial bool
		Err     eventError
	}
	var got []eventSummary
	for ev, err := range f.Run(ctx) {
		if err != nil {
			t.Fatalf("Run() yielded error: %v", err)
		}
		got = append(got, eventSummary{Partial: ev.Partial, Err: eventError{ev.ErrorCode, ev.ErrorMessage}})
		if len(got) > 4 {
			break
		}
	}
	want := []eventSummary{
		{Partial: true},
		{Partial: false, Err: eventError{"MODEL_RETURNED_NO_CONTENT", wantNoActionableContentMessage}},
	}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("events mismatch (-want +got):\n%s", diff)
	}
	if calls != 1 {
		t.Errorf("model called %d times, want 1", calls)
	}
}

// TestEmptyResponsePolicy_UnicodeFileInfoRecordUnitSeparators pins Python's
// str.strip whitespace, which also strips U+001C to U+001F; unicode.IsSpace
// does not.
func TestEmptyResponsePolicy_UnicodeFileInfoRecordUnitSeparators(t *testing.T) {
	events, _ := runScriptedFlow(t, &Flow{}, nil, stopTurn(&genai.Part{Text: "\x1c\x1d\x1e\x1f"}))
	want := []eventError{{"MODEL_RETURNED_NO_CONTENT", wantNoActionableContentMessage}}
	if diff := cmp.Diff(want, errorsOf(events)); diff != "" {
		t.Errorf("event errors mismatch (-want +got):\n%s", diff)
	}
}

func TestEmptyResponsePolicy_PreservesPreExistingErrorAndSkipsNonStopOrPartial(t *testing.T) {
	emptyText := func() *genai.Content { return genai.NewContentFromText("", genai.RoleModel) }
	tests := []struct {
		name string
		resp *model.LLMResponse
		want eventError
	}{
		{
			name: "preexisting_message_kept",
			resp: &model.LLMResponse{Content: emptyText(), ErrorMessage: "upstream detail", FinishReason: genai.FinishReasonStop},
			want: eventError{"MODEL_RETURNED_NO_CONTENT", "upstream detail"},
		},
		{
			name: "existing_error_code_untouched",
			resp: &model.LLMResponse{Content: emptyText(), ErrorCode: "SAFETY", ErrorMessage: "blocked", FinishReason: genai.FinishReasonStop},
			want: eventError{"SAFETY", "blocked"},
		},
		{
			name: "non_stop_untouched",
			resp: &model.LLMResponse{Content: emptyText(), FinishReason: genai.FinishReasonMaxTokens},
			want: eventError{},
		},
		{
			name: "partial_untouched",
			resp: &model.LLMResponse{Content: emptyText(), Partial: true, FinishReason: genai.FinishReasonStop},
			want: eventError{},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var events []*session.Event
			captureLog(t, func() { events, _ = runScriptedFlow(t, &Flow{}, nil, tc.resp) })
			if diff := cmp.Diff([]eventError{tc.want}, errorsOf(events)); diff != "" {
				t.Errorf("event errors mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestEmptyResponsePolicy_SSEEmptyContentAndNilContentNotStamped(t *testing.T) {
	sse := &agent.RunConfig{StreamingMode: agent.StreamingModeSSE}
	t.Run("nil_content", func(t *testing.T) {
		events, _ := runScriptedFlow(t, &Flow{}, sse, &model.LLMResponse{FinishReason: genai.FinishReasonStop})
		if len(events) != 0 {
			t.Errorf("Run() yielded %d events, want 0: %+v", len(events), errorsOf(events))
		}
	})
	t.Run("empty_parts", func(t *testing.T) {
		events, _ := runScriptedFlow(t, &Flow{}, sse, stopTurn())
		if diff := cmp.Diff([]eventError{{}}, errorsOf(events)); diff != "" {
			t.Errorf("event errors mismatch (-want +got):\n%s", diff)
		}
	})
}

// TestRunOneStep_AppliesPolicyBeforeProcessors pins that the policy judges what
// the model returned: a response processor that adds content afterwards does
// not hide an empty model turn.
func TestRunOneStep_AppliesPolicyBeforeProcessors(t *testing.T) {
	addPart := func(_ agent.InvocationContext, _ *model.LLMRequest, resp *model.LLMResponse) iter.Seq2[*session.Event, error] {
		return func(func(*session.Event, error) bool) {
			resp.Content.Parts = append(resp.Content.Parts, &genai.Part{Text: "added by processor"})
		}
	}
	f := &Flow{ResponseProcessors: []func(agent.InvocationContext, *model.LLMRequest, *model.LLMResponse) iter.Seq2[*session.Event, error]{addPart}}

	events, _ := runScriptedFlow(t, f, nil, stopTurn())

	if diff := cmp.Diff([]eventError{{"MODEL_RETURNED_NO_CONTENT", wantNoContentMessage}}, errorsOf(events)); diff != "" {
		t.Fatalf("event errors mismatch (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff([]*genai.Part{{Text: "added by processor"}}, events[0].Content.Parts); diff != "" {
		t.Errorf("event parts mismatch (-want +got):\n%s", diff)
	}
}

func TestCallLLM_AgentNameLabelStampingAndPreservation(t *testing.T) {
	type result struct {
		modelCalls  int
		modelLabels map[string]string
	}
	call := func(t *testing.T, a agent.Agent, req *model.LLMRequest, callbacks ...BeforeModelCallback) result {
		t.Helper()
		var res result
		m := &mockModelForTest{
			name: "test-model",
			generateContent: func(_ context.Context, req *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
				res.modelCalls++
				if req.Config != nil {
					res.modelLabels = maps.Clone(req.Config.Labels)
				}
				return func(yield func(*model.LLMResponse, error) bool) {
					yield(&model.LLMResponse{Content: genai.NewContentFromText("ok", genai.RoleModel)}, nil)
				}
			},
		}
		f := &Flow{Model: m, BeforeModelCallbacks: callbacks}
		params := icontext.InvocationContextParams{}
		if a != nil {
			params.Agent = a
		}
		ctx := icontext.NewInvocationContext(t.Context(), params)
		for _, err := range f.callLLM(ctx, req, map[string]any{}, map[string]int64{}) {
			if err != nil {
				t.Fatalf("callLLM() error = %v", err)
			}
		}
		return res
	}

	t.Run("stamps_after_before_model_callback", func(t *testing.T) {
		callbackSawNilConfig := false
		recordConfig := func(_ agent.Context, req *model.LLMRequest) (*model.LLMResponse, error) {
			callbackSawNilConfig = req.Config == nil
			return nil, nil
		}
		res := call(t, &mockAgent{name: "test_agent"}, &model.LLMRequest{}, recordConfig)
		if !callbackSawNilConfig {
			t.Error("before-model callback saw a non-nil config, want the request without the label")
		}
		if diff := cmp.Diff(map[string]string{"adk_agent_name": "test_agent"}, res.modelLabels); diff != "" {
			t.Errorf("model labels mismatch (-want +got):\n%s", diff)
		}
	})
	t.Run("preserves_existing_adk_agent_name_label", func(t *testing.T) {
		req := &model.LLMRequest{Config: &genai.GenerateContentConfig{Labels: map[string]string{"adk_agent_name": "custom_owner", "env": "prod"}}}
		res := call(t, &mockAgent{name: "billing_agent"}, req)
		if diff := cmp.Diff(map[string]string{"adk_agent_name": "custom_owner", "env": "prod"}, res.modelLabels); diff != "" {
			t.Errorf("model labels mismatch (-want +got):\n%s", diff)
		}
	})
	t.Run("callback_map_not_modified", func(t *testing.T) {
		shared := map[string]string{"team": "core"}
		attach := func(_ agent.Context, req *model.LLMRequest) (*model.LLMResponse, error) {
			req.Config = &genai.GenerateContentConfig{Labels: shared}
			return nil, nil
		}
		res := call(t, &mockAgent{name: "test_agent"}, &model.LLMRequest{}, attach)
		if diff := cmp.Diff(map[string]string{"team": "core"}, shared); diff != "" {
			t.Errorf("callback's labels map changed (-want +got):\n%s", diff)
		}
		if diff := cmp.Diff(map[string]string{"adk_agent_name": "test_agent", "team": "core"}, res.modelLabels); diff != "" {
			t.Errorf("model labels mismatch (-want +got):\n%s", diff)
		}
	})
	t.Run("callback_config_not_modified", func(t *testing.T) {
		shared := &genai.GenerateContentConfig{Temperature: genai.Ptr[float32](0.2)}
		attach := func(_ agent.Context, req *model.LLMRequest) (*model.LLMResponse, error) {
			req.Config = shared
			return nil, nil
		}
		res := call(t, &mockAgent{name: "test_agent"}, &model.LLMRequest{}, attach)
		if shared.Labels != nil {
			t.Errorf("callback's config Labels = %v, want nil", shared.Labels)
		}
		if diff := cmp.Diff(map[string]string{"adk_agent_name": "test_agent"}, res.modelLabels); diff != "" {
			t.Errorf("model labels mismatch (-want +got):\n%s", diff)
		}
	})
	t.Run("short_circuit_does_not_stamp", func(t *testing.T) {
		cached := func(agent.Context, *model.LLMRequest) (*model.LLMResponse, error) {
			return &model.LLMResponse{Content: genai.NewContentFromText("cached", genai.RoleModel)}, nil
		}
		req := &model.LLMRequest{}
		res := call(t, &mockAgent{name: "test_agent"}, req, cached)
		if req.Config != nil {
			t.Errorf("request config = %+v after a short-circuit, want nil", req.Config)
		}
		if res.modelCalls != 0 {
			t.Errorf("model called %d times, want 0", res.modelCalls)
		}
	})
	t.Run("nil_agent_does_not_panic_or_stamp", func(t *testing.T) {
		req := &model.LLMRequest{}
		res := call(t, nil, req)
		if req.Config != nil {
			t.Errorf("request config = %+v without an agent, want nil", req.Config)
		}
		if res.modelCalls != 1 {
			t.Errorf("model called %d times, want 1", res.modelCalls)
		}
	})
}

func TestFlowRun_ReplayExecutesToolsetToolsAndTransfersAndFreshIDs(t *testing.T) {
	t.Run("tool from a toolset gets a fresh event ID", func(t *testing.T) {
		var runs atomic.Int32
		tsTool := countedTool(t, "ts_tool", &runs, map[string]any{"ok": true})
		toolset := &dynamicToolset{toolsByCall: map[int][]tool.Tool{0: {tsTool}, 1: {tsTool}}}
		a := &mockLLMAgent{Agent: utils.Must(agent.New(agent.Config{Name: "agent"})), s: &State{Toolsets: []tool.Toolset{toolset}}}
		modelCalls := 0
		f := &Flow{
			Model: replayModel(t, &modelCalls, "after replay"),
			RequestProcessors: []func(agent.InvocationContext, *model.LLMRequest, *Flow) iter.Seq2[*session.Event, error]{
				toolProcessor,
			},
		}
		seeded := callEv("agent", fcall("ts_tool", "c1"))
		seeded.ID = "orig-event-id"
		ctx, sess := replayAgentContext(t, true, a, seeded)

		events := runReplayFlow(t, f, ctx, sess)

		if diff := cmp.Diff([]string{"agent response ts_tool/c1", "agent text after replay"}, describeReplayEvents(events)); diff != "" {
			t.Fatalf("Run() events mismatch (-want +got):\n%s", diff)
		}
		wantResponses := []*genai.FunctionResponse{{ID: "c1", Name: "ts_tool", Response: map[string]any{"ok": true}}}
		if diff := cmp.Diff(wantResponses, utils.FunctionResponses(events[0].Content)); diff != "" {
			t.Errorf("replayed response mismatch (-want +got):\n%s", diff)
		}
		if events[0].ID == "" || events[0].ID == "orig-event-id" {
			t.Errorf("replayed response event ID = %q, want a new non-empty ID", events[0].ID)
		}
	})
	t.Run("unexecuted sibling call of a toolset tool is replayed", func(t *testing.T) {
		// The decision finds the call event only among calls of the tools it
		// knows, so it must see the tools that preprocessing adds.
		var runs atomic.Int32
		tsTool := countedTool(t, "ts_tool", &runs, map[string]any{"ok": true})
		toolset := &dynamicToolset{toolsByCall: map[int][]tool.Tool{0: {tsTool}, 1: {tsTool}}}
		a := &mockLLMAgent{Agent: utils.Must(agent.New(agent.Config{Name: "agent"})), s: &State{Toolsets: []tool.Toolset{toolset}}}
		modelCalls := 0
		f := &Flow{
			Model: replayModel(t, &modelCalls, "done"),
			RequestProcessors: []func(agent.InvocationContext, *model.LLMRequest, *Flow) iter.Seq2[*session.Event, error]{
				toolProcessor,
			},
		}
		ctx, sess := replayAgentContext(t, true, a,
			callEv("agent", fcall("ts_tool", "c1"), fcall("ts_tool", "c2")),
			respEv("user", "ts_tool", "c1"),
		)

		events := runReplayFlow(t, f, ctx, sess)

		if diff := cmp.Diff([]string{"agent response ts_tool/c2", "agent text done"}, describeReplayEvents(events)); diff != "" {
			t.Errorf("Run() events mismatch (-want +got):\n%s", diff)
		}
		if got := runs.Load(); got != 1 {
			t.Errorf("ts_tool ran %d times, want 1", got)
		}
	})
	t.Run("replayed transfer runs the target", func(t *testing.T) {
		sub := utils.Must(agent.New(agent.Config{
			Name: "sub",
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					ev := session.NewEvent(ctx, ctx.InvocationID())
					ev.Author = "sub"
					ev.Content = genai.NewContentFromText("from sub", genai.RoleModel)
					yield(ev, nil)
				}
			},
		}))
		parent := utils.Must(agent.New(agent.Config{Name: "agent", SubAgents: []agent.Agent{sub}}))
		modelCalls := 0
		f := &Flow{
			Model: replayModel(t, &modelCalls),
			Tools: []tool.Tool{&TransferToAgentTool{supportedAgents: []agent.Agent{sub}}},
		}
		ctx, sess := replayAgentContext(t, true, parent,
			callEv("agent", fcallArgs(transferAgentName, "c1", map[string]any{"agent_name": "sub"})))

		events := runReplayFlow(t, f, ctx, sess)

		if diff := cmp.Diff([]string{"agent response transfer_to_agent/c1", "sub text from sub"}, describeReplayEvents(events)); diff != "" {
			t.Errorf("Run() events mismatch (-want +got):\n%s", diff)
		}
		if modelCalls != 0 {
			t.Errorf("model called %d times, want 0", modelCalls)
		}
	})
}

func TestFlowRun_ReplaySameNameCallsWithSkipSummarizationAndCrashRecovery(t *testing.T) {
	t.Run("ID-less response before an explicit ID, SkipSummarization", func(t *testing.T) {
		var mu sync.Mutex
		var ranIndexes []int
		ask, err := functiontool.New(functiontool.Config{Name: "ask", Description: "ask"}, func(ctx agent.Context, args struct {
			Index int `json:"index"`
		},
		) (map[string]any, error) {
			mu.Lock()
			ranIndexes = append(ranIndexes, args.Index)
			mu.Unlock()
			ctx.Actions().SkipSummarization = true
			return map[string]any{"index": args.Index}, nil
		})
		if err != nil {
			t.Fatalf("functiontool.New() error = %v", err)
		}
		modelCalls := 0
		f := &Flow{Model: replayModel(t, &modelCalls), Tools: []tool.Tool{ask}}
		callEvent := callEv("root_agent",
			fcallArgs("ask", "c1", map[string]any{"index": 0}),
			fcallArgs("ask", "c2", map[string]any{"index": 1}),
			fcallArgs("ask", "c3", map[string]any{"index": 2}),
		)
		ctx, sess := replayAgentContext(t, true, &mockAgent{name: "root_agent"},
			textEv("user", "Resume"), callEvent, respEv("user", "ask", ""), respEv("user", "ask", "c1"))

		events := runReplayFlow(t, f, ctx, sess)

		if diff := cmp.Diff([]int{2}, ranIndexes); diff != "" {
			t.Errorf("ask ran with indexes mismatch (-want +got):\n%s", diff)
		}
		if len(events) != 1 {
			t.Fatalf("Run() yielded %v, want one response event", describeReplayEvents(events))
		}
		// functiontool passes results through JSON, so numbers come back as float64.
		wantResponses := []*genai.FunctionResponse{{ID: "c3", Name: "ask", Response: map[string]any{"index": float64(2)}}}
		if diff := cmp.Diff(wantResponses, utils.FunctionResponses(events[0].Content)); diff != "" {
			t.Errorf("replayed response mismatch (-want +got):\n%s", diff)
		}
		if modelCalls != 0 {
			t.Errorf("model called %d times, want 0", modelCalls)
		}
		if got := len(callEvent.Content.Parts); got != 3 {
			t.Errorf("stored call event has %d parts after the run, want 3", got)
		}
	})
	t.Run("crash after the first call ran", func(t *testing.T) {
		var askRuns, fetchRuns atomic.Int32
		modelCalls := 0
		f := &Flow{
			Model: replayModel(t, &modelCalls, "done"),
			Tools: []tool.Tool{
				countedTool(t, "ask", &askRuns, map[string]any{"result": "client"}),
				countedTool(t, "fetch", &fetchRuns, map[string]any{"result": "fetched"}),
			},
		}
		ctx, sess := replayAgentContext(t, true, &mockAgent{name: "root_agent"},
			textEv("user", "query"),
			callEv("root_agent", fcall("ask", "c1"), fcall("fetch", "c2")),
			respEv("user", "ask", "c1"),
		)

		events := runReplayFlow(t, f, ctx, sess)

		if got := fetchRuns.Load(); got != 1 {
			t.Errorf("fetch ran %d times, want 1", got)
		}
		if got := askRuns.Load(); got != 0 {
			t.Errorf("ask ran %d times, want 0", got)
		}
		if diff := cmp.Diff([]string{"root_agent response fetch/c2", "root_agent text done"}, describeReplayEvents(events)); diff != "" {
			t.Errorf("Run() events mismatch (-want +got):\n%s", diff)
		}
	})
}

func TestFlowRun_PreprocessShortCircuitsBeforeResumeDecisionAndPauseEmitsNothing(t *testing.T) {
	t.Run("final event from preprocessing ends the step", func(t *testing.T) {
		var askRuns atomic.Int32
		final := respEv("agent", "confirmed", "x")
		final.Actions.SkipSummarization = true
		modelCalls := 0
		f := &Flow{
			Model: replayModel(t, &modelCalls),
			Tools: []tool.Tool{countedTool(t, "ask", &askRuns, map[string]any{"r": 1})},
			RequestProcessors: []func(agent.InvocationContext, *model.LLMRequest, *Flow) iter.Seq2[*session.Event, error]{
				func(agent.InvocationContext, *model.LLMRequest, *Flow) iter.Seq2[*session.Event, error] {
					return func(yield func(*session.Event, error) bool) { yield(final, nil) }
				},
			},
		}
		ctx, _ := replayAgentContext(t, true, &mockAgent{name: "agent"}, callEv("agent", fcall("ask", "c1")))

		// The yielded events are not stored, so a decision taken at any point
		// of the step sees only the trailing call. A replay would then repeat
		// every step, so the loop stops after a few events.
		var events []*session.Event
		for ev, err := range f.Run(ctx) {
			if err != nil {
				t.Fatalf("Run() error = %v", err)
			}
			events = append(events, ev)
			if len(events) == 3 {
				break
			}
		}

		if diff := cmp.Diff([]string{"agent response confirmed/x"}, describeReplayEvents(events)); diff != "" {
			t.Errorf("Run() events mismatch (-want +got):\n%s", diff)
		}
		if got := askRuns.Load(); got != 0 {
			t.Errorf("ask ran %d times, want 0", got)
		}
		if modelCalls != 0 {
			t.Errorf("model called %d times, want 0", modelCalls)
		}
	})
	t.Run("pause emits nothing", func(t *testing.T) {
		var askRuns atomic.Int32
		modelCalls := 0
		f := &Flow{
			Model: replayModel(t, &modelCalls),
			Tools: []tool.Tool{countedTool(t, "ask", &askRuns, map[string]any{"r": 1})},
		}
		ctx, sess := replayAgentContext(t, true, &mockAgent{name: "agent"},
			withLRO(callEv("agent", fcall("ask", "c1")), "c1"), textEv("user", "later"))

		events := runReplayFlow(t, f, ctx, sess)

		if len(events) != 0 {
			t.Errorf("Run() yielded %v, want nothing", describeReplayEvents(events))
		}
		if modelCalls != 0 {
			t.Errorf("model called %d times, want 0", modelCalls)
		}
	})
}
