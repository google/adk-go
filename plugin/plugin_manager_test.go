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

package plugin_test

import (
	"errors"
	"fmt"
	"net/http"
	"testing"

	"github.com/google/go-cmp/cmp"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/internal/testutil"
	"google.golang.org/adk/model"
	"google.golang.org/adk/plugin"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/functiontool"
)

type testCase struct {
	name                            string
	tool                            func(tool.Context, map[string]any) (map[string]any, error)
	args                            map[string]any
	beforeToolCallbacks             []llmagent.BeforeToolCallback
	afterToolCallbacks              []llmagent.AfterToolCallback
	onToolErrorCallbacks            []llmagent.OnToolErrorCallback
	want                            map[string]any
	dontRunBeforeCanonicalCallback  bool
	dontRunAfterCanonicalCallback   bool
	dontRunOnErrorCanonicalCallback bool
}

func TestCallTool(t *testing.T) {
	testCases := []testCase{
		{
			name: "tool runs successfully",
			tool: func(ctx tool.Context, args map[string]any) (map[string]any, error) {
				return map[string]any{"result": "success"}, nil
			},
			args:                            map[string]any{"key": "value"},
			dontRunOnErrorCanonicalCallback: true,
			want:                            map[string]any{"result": "success"},
		},
		{
			name: "tool error",
			tool: func(ctx tool.Context, args map[string]any) (map[string]any, error) {
				return nil, errors.New("tool error")
			},
			args: map[string]any{"key": "value"},
			want: map[string]any{"error": "tool error"},
		},
		{
			name: "before callback returns result",
			tool: func(ctx tool.Context, args map[string]any) (map[string]any, error) {
				t.Error("tool should not be called")
				return nil, nil
			},
			beforeToolCallbacks: []llmagent.BeforeToolCallback{
				func(ctx tool.Context, tool tool.Tool, args map[string]any) (map[string]any, error) {
					return map[string]any{"result": "intercepted"}, nil
				},
				func(ctx tool.Context, tool tool.Tool, args map[string]any) (map[string]any, error) {
					return map[string]any{"result": "2nd callback should not be called"}, nil
				},
			},
			dontRunOnErrorCanonicalCallback: true,
			dontRunBeforeCanonicalCallback:  true,
			want:                            map[string]any{"result": "intercepted"},
		},
		{
			name: "before callback returns error",
			tool: func(ctx tool.Context, args map[string]any) (map[string]any, error) {
				t.Error("tool should not be called")
				return nil, nil
			},
			beforeToolCallbacks: []llmagent.BeforeToolCallback{
				func(ctx tool.Context, tool tool.Tool, args map[string]any) (map[string]any, error) {
					return nil, errors.New("before callback error")
				},
				func(ctx tool.Context, tool tool.Tool, args map[string]any) (map[string]any, error) {
					return nil, errors.New("unexpected error")
				},
			},
			dontRunBeforeCanonicalCallback: true,
			want:                           map[string]any{"error": "before callback error"},
		},
		{
			name: "after callback modifies result",
			tool: func(ctx tool.Context, args map[string]any) (map[string]any, error) {
				return map[string]any{"result": "original"}, nil
			},
			afterToolCallbacks: []llmagent.AfterToolCallback{
				func(ctx tool.Context, tool tool.Tool, args, result map[string]any, err error) (map[string]any, error) {
					return map[string]any{"result": "modified"}, nil
				},
			},
			dontRunAfterCanonicalCallback:   true,
			dontRunOnErrorCanonicalCallback: true,
			want:                            map[string]any{"result": "modified"},
		},
		{
			name: "after callback handles error and are run in symmetrical order",
			tool: func(ctx tool.Context, args map[string]any) (map[string]any, error) {
				return nil, errors.New("tool error")
			},
			afterToolCallbacks: []llmagent.AfterToolCallback{
				func(ctx tool.Context, tool tool.Tool, args, result map[string]any, err error) (map[string]any, error) {
					t.Errorf("unexpected call to after tool callback")
					return map[string]any{"result": "unexpected output"}, nil
				},
				func(ctx tool.Context, tool tool.Tool, args, result map[string]any, err error) (map[string]any, error) {
					if err != nil {
						return map[string]any{"result": "error handled"}, nil
					}
					return nil, nil
				},
			},
			dontRunAfterCanonicalCallback: true,
			want:                          map[string]any{"result": "error handled"},
		},
		{
			name: "after callback returns error and are run in symmetrical order",
			tool: func(ctx tool.Context, args map[string]any) (map[string]any, error) {
				return map[string]any{"result": "success"}, nil
			},
			afterToolCallbacks: []llmagent.AfterToolCallback{
				func(ctx tool.Context, tool tool.Tool, args, result map[string]any, err error) (map[string]any, error) {
					t.Errorf("unexpected call to after tool callback")
					return nil, errors.New("unexpected error")
				},
				func(ctx tool.Context, tool tool.Tool, args, result map[string]any, err error) (map[string]any, error) {
					return nil, errors.New("after callback error")
				},
			},
			dontRunOnErrorCanonicalCallback: true,
			dontRunAfterCanonicalCallback:   true,
			want:                            map[string]any{"error": "after callback error"},
		},
		{
			name: "no-op callbacks return func results",
			tool: func(ctx tool.Context, args map[string]any) (map[string]any, error) {
				return map[string]any{"result": "success"}, nil
			},
			beforeToolCallbacks: []llmagent.BeforeToolCallback{
				func(ctx tool.Context, tool tool.Tool, args map[string]any) (map[string]any, error) {
					return nil, nil
				},
			},
			afterToolCallbacks: []llmagent.AfterToolCallback{
				func(ctx tool.Context, tool tool.Tool, args, result map[string]any, err error) (map[string]any, error) {
					return nil, nil
				},
			},
			dontRunOnErrorCanonicalCallback: true,
			want:                            map[string]any{"result": "success"},
		},
		{
			name: "before callback result passed to after callback",
			tool: func(ctx tool.Context, args map[string]any) (map[string]any, error) {
				t.Error("tool should not be called")
				return nil, nil
			},
			beforeToolCallbacks: []llmagent.BeforeToolCallback{
				func(ctx tool.Context, tool tool.Tool, args map[string]any) (map[string]any, error) {
					return map[string]any{"result": "from_before"}, nil
				},
			},
			afterToolCallbacks: []llmagent.AfterToolCallback{
				func(ctx tool.Context, tool tool.Tool, args, result map[string]any, err error) (map[string]any, error) {
					if val, ok := result["result"]; !ok || val != "from_before" {
						return nil, errors.New("unexpected result in after callback")
					}
					return map[string]any{"result": "from_after"}, nil
				},
			},
			dontRunOnErrorCanonicalCallback: true,
			dontRunBeforeCanonicalCallback:  true,
			dontRunAfterCanonicalCallback:   true,
			want:                            map[string]any{"result": "from_after"},
		},
		{
			name: "before callback error passed to after callback",
			tool: func(ctx tool.Context, args map[string]any) (map[string]any, error) {
				t.Error("tool should not be called")
				return nil, nil
			},
			beforeToolCallbacks: []llmagent.BeforeToolCallback{
				func(ctx tool.Context, tool tool.Tool, args map[string]any) (map[string]any, error) {
					return nil, errors.New("error_from_before")
				},
			},
			afterToolCallbacks: []llmagent.AfterToolCallback{
				func(ctx tool.Context, tool tool.Tool, args, result map[string]any, err error) (map[string]any, error) {
					if err == nil || err.Error() != "error_from_before" {
						return nil, errors.New("unexpected error in after callback")
					}
					return map[string]any{"result": "error_handled_in_after"}, nil
				},
			},
			dontRunBeforeCanonicalCallback: true,
			dontRunAfterCanonicalCallback:  true,
			want:                           map[string]any{"result": "error_handled_in_after"},
		},
		{
			name: "before callback error passed to on tool error callback",
			tool: func(ctx tool.Context, args map[string]any) (map[string]any, error) {
				t.Error("tool should not be called")
				return nil, nil
			},
			beforeToolCallbacks: []llmagent.BeforeToolCallback{
				func(ctx tool.Context, tool tool.Tool, args map[string]any) (map[string]any, error) {
					return nil, errors.New("error_from_before")
				},
			},
			onToolErrorCallbacks: []llmagent.OnToolErrorCallback{
				func(ctx tool.Context, tool tool.Tool, args map[string]any, err error) (map[string]any, error) {
					if err == nil || err.Error() != "error_from_before" {
						return nil, errors.New("unexpected error in on tool error callback")
					}
					return map[string]any{"result": "error_handled_in_on_tool_error_callback"}, nil
				},
			},
			dontRunBeforeCanonicalCallback:  true,
			dontRunOnErrorCanonicalCallback: true,
			want:                            map[string]any{"result": "error_handled_in_on_tool_error_callback"},
		},
		{
			name: "before callback error passed to on tool error callback and after tool called",
			tool: func(ctx tool.Context, args map[string]any) (map[string]any, error) {
				t.Error("tool should not be called")
				return nil, nil
			},
			beforeToolCallbacks: []llmagent.BeforeToolCallback{
				func(ctx tool.Context, tool tool.Tool, args map[string]any) (map[string]any, error) {
					return nil, errors.New("error_from_before")
				},
			},
			onToolErrorCallbacks: []llmagent.OnToolErrorCallback{
				func(ctx tool.Context, tool tool.Tool, args map[string]any, err error) (map[string]any, error) {
					if err == nil || err.Error() != "error_from_before" {
						return nil, errors.New("unexpected error in on tool error callback")
					}
					return map[string]any{"result": "error_handled_in_on_tool_error_callback"}, nil
				},
			},
			afterToolCallbacks: []llmagent.AfterToolCallback{
				func(ctx tool.Context, tool tool.Tool, args, result map[string]any, err error) (map[string]any, error) {
					if err != nil {
						return nil, errors.New("unexpected error in after callback")
					}
					return map[string]any{"result": "from_after"}, nil
				},
			},
			dontRunAfterCanonicalCallback:   true,
			dontRunBeforeCanonicalCallback:  true,
			dontRunOnErrorCanonicalCallback: true,
			want:                            map[string]any{"result": "from_after"},
		},
		{
			name: "before callback error passed to on tool error callback and passed to after tool called",
			tool: func(ctx tool.Context, args map[string]any) (map[string]any, error) {
				t.Error("tool should not be called")
				return nil, nil
			},
			beforeToolCallbacks: []llmagent.BeforeToolCallback{
				func(ctx tool.Context, tool tool.Tool, args map[string]any) (map[string]any, error) {
					return nil, errors.New("error_from_before")
				},
			},
			onToolErrorCallbacks: []llmagent.OnToolErrorCallback{
				func(ctx tool.Context, tool tool.Tool, args map[string]any, err error) (map[string]any, error) {
					if err == nil || err.Error() != "error_from_before" {
						return nil, errors.New("unexpected error in on tool error callback")
					}
					return nil, errors.New("error_from_on_tool_error")
				},
			},
			afterToolCallbacks: []llmagent.AfterToolCallback{
				func(ctx tool.Context, tool tool.Tool, args, result map[string]any, err error) (map[string]any, error) {
					if err == nil || err.Error() != "error_from_on_tool_error" {
						return nil, errors.New("unexpected error in after callback")
					}
					return nil, errors.New("error_from_after_tool")
				},
			},
			dontRunAfterCanonicalCallback:   true,
			dontRunOnErrorCanonicalCallback: true,
			dontRunBeforeCanonicalCallback:  true,
			want:                            map[string]any{"error": "error_from_after_tool"},
		},
		{
			name: "before callback error passed to on tool error callback and passed to after tool called and handled",
			tool: func(ctx tool.Context, args map[string]any) (map[string]any, error) {
				t.Error("tool should not be called")
				return nil, nil
			},
			beforeToolCallbacks: []llmagent.BeforeToolCallback{
				func(ctx tool.Context, tool tool.Tool, args map[string]any) (map[string]any, error) {
					return nil, errors.New("error_from_before")
				},
			},
			onToolErrorCallbacks: []llmagent.OnToolErrorCallback{
				func(ctx tool.Context, tool tool.Tool, args map[string]any, err error) (map[string]any, error) {
					if err == nil || err.Error() != "error_from_before" {
						return nil, errors.New("unexpected error in on tool error callback")
					}
					return nil, errors.New("error_from_on_tool_error")
				},
			},
			afterToolCallbacks: []llmagent.AfterToolCallback{
				func(ctx tool.Context, tool tool.Tool, args, result map[string]any, err error) (map[string]any, error) {
					if err == nil || err.Error() != "error_from_on_tool_error" {
						return nil, errors.New("unexpected error in after tool callback")
					}
					return map[string]any{"result": "error_handled_in_after_tool_callback"}, nil
				},
			},
			dontRunAfterCanonicalCallback:   true,
			dontRunBeforeCanonicalCallback:  true,
			dontRunOnErrorCanonicalCallback: true,
			want:                            map[string]any{"result": "error_handled_in_after_tool_callback"},
		},
	}
	for _, tc := range testCases {
		t.Run(fmt.Sprintf("%s_plugin", tc.name), func(t *testing.T) {
			maxLen := max(len(tc.beforeToolCallbacks), len(tc.afterToolCallbacks), len(tc.onToolErrorCallbacks))
			var plugins []*plugin.Plugin
			for i := range maxLen {
				var currentBefore llmagent.BeforeToolCallback
				var currentAfter llmagent.AfterToolCallback
				var currentError llmagent.OnToolErrorCallback

				// 2. Bounds checks: Only assign if i is within the slice limits
				if i < len(tc.beforeToolCallbacks) {
					currentBefore = tc.beforeToolCallbacks[i]
				}
				if i < len(tc.afterToolCallbacks) {
					currentAfter = tc.afterToolCallbacks[i]
				}
				if i < len(tc.onToolErrorCallbacks) {
					currentError = tc.onToolErrorCallbacks[i]
				}
				p, err := plugin.New(plugin.Config{
					Name:                fmt.Sprintf("plugin-%d", i),
					BeforeToolCallback:  currentBefore,
					AfterToolCallback:   currentAfter,
					OnToolErrorCallback: currentError,
				})
				if err != nil {
					t.Errorf("failed to initialize plugin: %v", err)
				}
				plugins = append(plugins, p)
			}

			model := &testutil.MockModel{
				Responses: []*genai.Content{
					genai.NewContentFromFunctionCall("testTool", tc.args, genai.RoleModel),
				},
			}

			ft, err := functiontool.New(functiontool.Config{
				Name: "testTool",
			}, tc.tool)
			if err != nil {
				t.Errorf("failed to function tool: %v", err)
			}

			onToolErrorCallbacksCalled := false
			beforeToolCallbacksCalled := false
			afterToolCallbacksCalled := false
			onToolErrorCallbacks := []llmagent.OnToolErrorCallback{
				func(ctx tool.Context, tool tool.Tool, args map[string]any, err error) (map[string]any, error) {
					onToolErrorCallbacksCalled = true
					if tc.dontRunOnErrorCanonicalCallback {
						t.Error("on tool error should not be called")
					}
					return nil, nil
				},
			}
			beforeToolCallbacks := []llmagent.BeforeToolCallback{
				func(ctx tool.Context, tool tool.Tool, args map[string]any) (map[string]any, error) {
					beforeToolCallbacksCalled = true
					if tc.dontRunBeforeCanonicalCallback {
						t.Error("before Tool Callback should not be called")
					}
					return nil, nil
				},
			}
			afterToolCallbacks := []llmagent.AfterToolCallback{
				func(ctx tool.Context, tool tool.Tool, args, result map[string]any, err error) (map[string]any, error) {
					afterToolCallbacksCalled = true
					if tc.dontRunAfterCanonicalCallback {
						t.Error("after Tool Callback should not be called")
					}
					return nil, nil
				},
			}

			agent, err := llmagent.New(llmagent.Config{
				Name:                 "test_agent",
				Model:                model,
				Tools:                []tool.Tool{ft},
				OnToolErrorCallbacks: onToolErrorCallbacks,
				BeforeToolCallbacks:  beforeToolCallbacks,
				AfterToolCallbacks:   afterToolCallbacks,
			})
			if err != nil {
				t.Fatalf("failed to create LLM Agent: %v", err)
			}

			testRunner := testutil.NewTestAgentRunnerWithPluginManager(t, agent, runner.PluginConfig{
				Plugins: plugins,
			})

			stream := testRunner.Run(t, "session", "user input")

			parts, err := testutil.CollectParts(stream)
			if err != nil && err.Error() != "no data" {
				t.Fatalf("agent returned (%v, %v), want result", parts, err)
			}
			var got map[string]any
			for _, part := range parts {
				if part.FunctionResponse != nil {
					got = part.FunctionResponse.Response
				}
			}

			if diff := cmp.Diff(tc.want, got); diff != "" {
				t.Errorf("callTool() mismatch (-want +got):\n%s", diff)
			}

			if onToolErrorCallbacksCalled == false && tc.dontRunOnErrorCanonicalCallback == false {
				t.Error("on tool error should be called")
			}
			if beforeToolCallbacksCalled == false && tc.dontRunBeforeCanonicalCallback == false {
				t.Error("before tool should be called")
			}
			if afterToolCallbacksCalled == false && tc.dontRunAfterCanonicalCallback == false {
				t.Error("after tool should be called")
			}
		})
	}
}

func TestModelCallbacks(t *testing.T) {
	t.Parallel()

	for _, tc := range []struct {
		name                            string
		llmResponses                    []*genai.Content
		beforeModelCallbacks            []llmagent.BeforeModelCallback
		afterModelCallbacks             []llmagent.AfterModelCallback
		onModelErrorCallback            []llmagent.OnModelErrorCallback
		wantTexts                       []string
		wantErr                         error
		dontRunBeforeCanonicalCallback  bool
		dontRunAfterCanonicalCallback   bool
		dontRunOnErrorCanonicalCallback bool
	}{
		{
			name: "before model callback doesn't modify anything",
			beforeModelCallbacks: []llmagent.BeforeModelCallback{
				func(ctx agent.CallbackContext, llmRequest *model.LLMRequest) (*model.LLMResponse, error) {
					return nil, nil
				},
			},
			llmResponses: []*genai.Content{
				genai.NewContentFromText("hello from model", genai.RoleModel),
			},
			dontRunOnErrorCanonicalCallback: true,
			wantTexts: []string{
				"hello from model",
			},
		},
		{
			name: "before model callback returns an error",
			beforeModelCallbacks: []llmagent.BeforeModelCallback{
				func(ctx agent.CallbackContext, llmRequest *model.LLMRequest) (*model.LLMResponse, error) {
					return nil, fmt.Errorf("before_model_callback_error: %w", http.ErrNoCookie)
				},
				func(ctx agent.CallbackContext, llmRequest *model.LLMRequest) (*model.LLMResponse, error) {
					return nil, fmt.Errorf("before_model_callback_error: %w", http.ErrHijacked)
				},
			},
			llmResponses: []*genai.Content{
				genai.NewContentFromText("hello from model", genai.RoleModel),
			},
			dontRunBeforeCanonicalCallback:  true,
			dontRunOnErrorCanonicalCallback: true,
			dontRunAfterCanonicalCallback:   true,
			wantErr:                         http.ErrNoCookie,
		},
		{
			name: "before model callback returns new LLMResponse",
			beforeModelCallbacks: []llmagent.BeforeModelCallback{
				func(ctx agent.CallbackContext, llmRequest *model.LLMRequest) (*model.LLMResponse, error) {
					return &model.LLMResponse{
						Content: genai.NewContentFromText("hello from before_model_callback", genai.RoleModel),
					}, nil
				},
				func(ctx agent.CallbackContext, llmRequest *model.LLMRequest) (*model.LLMResponse, error) {
					return &model.LLMResponse{
						Content: genai.NewContentFromText("unexpected text", genai.RoleModel),
					}, nil
				},
			},
			llmResponses: []*genai.Content{
				genai.NewContentFromText("hello from model", genai.RoleModel),
			},
			dontRunBeforeCanonicalCallback:  true,
			dontRunOnErrorCanonicalCallback: true,
			dontRunAfterCanonicalCallback:   true,
			wantTexts: []string{
				"hello from before_model_callback",
			},
		},
		{
			name: "before model callback returns both new LLMResponse and error",
			beforeModelCallbacks: []llmagent.BeforeModelCallback{
				func(ctx agent.CallbackContext, llmRequest *model.LLMRequest) (*model.LLMResponse, error) {
					return &model.LLMResponse{
						Content: genai.NewContentFromText("hello from before_model_callback", genai.RoleModel),
					}, fmt.Errorf("before_model_callback_error: %w", http.ErrNoCookie)
				},
			},
			llmResponses: []*genai.Content{
				genai.NewContentFromText("hello from model", genai.RoleModel),
			},
			dontRunBeforeCanonicalCallback:  true,
			dontRunOnErrorCanonicalCallback: true,
			dontRunAfterCanonicalCallback:   true,
			wantErr:                         http.ErrNoCookie,
		},
		{
			name: "after model callback doesn't modify anything",
			afterModelCallbacks: []llmagent.AfterModelCallback{
				func(ctx agent.CallbackContext, llmResponse *model.LLMResponse, llmResponseError error) (*model.LLMResponse, error) {
					return nil, nil
				},
			},
			llmResponses: []*genai.Content{
				genai.NewContentFromText("hello from model", genai.RoleModel),
			},
			dontRunOnErrorCanonicalCallback: true,
			wantTexts: []string{
				"hello from model",
			},
		},
		{
			name: "after model callback returns new LLMResponse and are run in symmetrical order",
			afterModelCallbacks: []llmagent.AfterModelCallback{
				func(ctx agent.CallbackContext, llmResponse *model.LLMResponse, llmResponseError error) (*model.LLMResponse, error) {
					return &model.LLMResponse{
						Content: genai.NewContentFromText("unexpected text", genai.RoleModel),
					}, nil
				},
				func(ctx agent.CallbackContext, llmResponse *model.LLMResponse, llmResponseError error) (*model.LLMResponse, error) {
					return &model.LLMResponse{
						Content: genai.NewContentFromText("hello from after_model_callback", genai.RoleModel),
					}, nil
				},
			},
			llmResponses: []*genai.Content{
				genai.NewContentFromText("hello from model", genai.RoleModel),
			},
			dontRunOnErrorCanonicalCallback: true,
			dontRunAfterCanonicalCallback:   true,
			wantTexts: []string{
				"hello from after_model_callback",
			},
		},
		{
			name: "after model callback returns error and are run in symmetrical order",
			afterModelCallbacks: []llmagent.AfterModelCallback{
				func(ctx agent.CallbackContext, llmResponse *model.LLMResponse, llmResponseError error) (*model.LLMResponse, error) {
					return nil, fmt.Errorf("error from after_model_callback: %w", http.ErrHijacked)
				},
				func(ctx agent.CallbackContext, llmResponse *model.LLMResponse, llmResponseError error) (*model.LLMResponse, error) {
					return nil, fmt.Errorf("error from after_model_callback: %w", http.ErrNoCookie)
				},
			},
			llmResponses: []*genai.Content{
				genai.NewContentFromText("hello from model", genai.RoleModel),
			},
			dontRunOnErrorCanonicalCallback: true,
			dontRunAfterCanonicalCallback:   true,
			wantErr:                         http.ErrNoCookie,
		},
		{
			name: "after model callback returns both new LLMResponse and error",
			afterModelCallbacks: []llmagent.AfterModelCallback{
				func(ctx agent.CallbackContext, llmResponse *model.LLMResponse, llmResponseError error) (*model.LLMResponse, error) {
					return &model.LLMResponse{
						Content: genai.NewContentFromText("hello from after_model_callback", genai.RoleModel),
					}, fmt.Errorf("error from after_model_callback: %w", http.ErrNoCookie)
				},
			},
			llmResponses: []*genai.Content{
				genai.NewContentFromText("hello from model", genai.RoleModel),
			},
			dontRunOnErrorCanonicalCallback: true,
			dontRunAfterCanonicalCallback:   true,
			wantErr:                         http.ErrNoCookie,
		},
		{
			name: "on model error callback is not called",
			onModelErrorCallback: []llmagent.OnModelErrorCallback{
				func(ctx agent.CallbackContext, llmRequest *model.LLMRequest, llmError error) (*model.LLMResponse, error) {
					return &model.LLMResponse{
						Content: genai.NewContentFromText("hello from on_model_error_callback", genai.RoleModel),
					}, fmt.Errorf("on_model_error_callback: %w", http.ErrNoCookie)
				},
			},
			llmResponses: []*genai.Content{
				genai.NewContentFromText("hello from model", genai.RoleModel),
			},
			dontRunOnErrorCanonicalCallback: true,
			wantTexts: []string{
				"hello from model",
			},
		},
		{
			name: "on model error callback changes message",
			onModelErrorCallback: []llmagent.OnModelErrorCallback{
				func(ctx agent.CallbackContext, llmRequest *model.LLMRequest, llmError error) (*model.LLMResponse, error) {
					return &model.LLMResponse{
						Content: genai.NewContentFromText("hello from on_model_error_callback", genai.RoleModel),
					}, nil
				},
			},
			llmResponses:                    []*genai.Content{},
			dontRunOnErrorCanonicalCallback: true,
			wantTexts: []string{
				"hello from on_model_error_callback",
			},
		},
		{
			name: "on model error callback changes err",
			onModelErrorCallback: []llmagent.OnModelErrorCallback{
				func(ctx agent.CallbackContext, llmRequest *model.LLMRequest, llmError error) (*model.LLMResponse, error) {
					return &model.LLMResponse{
						Content: genai.NewContentFromText("hello from on_model_error_callback", genai.RoleModel),
					}, fmt.Errorf("error from on_model_error_callback: %w", http.ErrNoCookie)
				},
				func(ctx agent.CallbackContext, llmRequest *model.LLMRequest, llmError error) (*model.LLMResponse, error) {
					return &model.LLMResponse{
						Content: genai.NewContentFromText("hello from on_model_error_callback", genai.RoleModel),
					}, fmt.Errorf("error from on_model_error_callback: %w", http.ErrHijacked)
				},
			},
			llmResponses:                    []*genai.Content{},
			dontRunOnErrorCanonicalCallback: true,
			dontRunAfterCanonicalCallback:   true,
			wantErr:                         http.ErrNoCookie,
		},
		{
			name: "on model error callback returns both new LLMResponse and error",
			onModelErrorCallback: []llmagent.OnModelErrorCallback{
				func(ctx agent.CallbackContext, llmRequest *model.LLMRequest, llmError error) (*model.LLMResponse, error) {
					return &model.LLMResponse{
						Content: genai.NewContentFromText("hello from on_model_error_callback", genai.RoleModel),
					}, fmt.Errorf("error from on_model_error_callback: %w", http.ErrNoCookie)
				},
			},
			llmResponses:                    []*genai.Content{},
			dontRunOnErrorCanonicalCallback: true,
			dontRunAfterCanonicalCallback:   true,
			wantErr:                         http.ErrNoCookie,
		},
		{
			name: "on model error callback does not process before model callback error",
			beforeModelCallbacks: []llmagent.BeforeModelCallback{
				func(ctx agent.CallbackContext, llmRequest *model.LLMRequest) (*model.LLMResponse, error) {
					return nil, fmt.Errorf("before_model_callback_error: %w", http.ErrNoCookie)
				},
			},
			onModelErrorCallback: []llmagent.OnModelErrorCallback{
				func(ctx agent.CallbackContext, llmRequest *model.LLMRequest, llmError error) (*model.LLMResponse, error) {
					return &model.LLMResponse{
						Content: genai.NewContentFromText("hello from on_model_error_callback", genai.RoleModel),
					}, fmt.Errorf("error from on_model_error_callback: %w", http.ErrHijacked)
				},
			},
			llmResponses: []*genai.Content{
				genai.NewContentFromText("hello from model", genai.RoleModel),
			},
			dontRunBeforeCanonicalCallback:  true,
			dontRunOnErrorCanonicalCallback: true,
			dontRunAfterCanonicalCallback:   true,
			wantErr:                         http.ErrNoCookie,
		},
		{
			name: "on model error callback does not process before model callback message",
			beforeModelCallbacks: []llmagent.BeforeModelCallback{
				func(ctx agent.CallbackContext, llmRequest *model.LLMRequest) (*model.LLMResponse, error) {
					return &model.LLMResponse{
						Content: genai.NewContentFromText("hello from before_model_callback", genai.RoleModel),
					}, nil
				},
			},
			onModelErrorCallback: []llmagent.OnModelErrorCallback{
				func(ctx agent.CallbackContext, llmRequest *model.LLMRequest, llmError error) (*model.LLMResponse, error) {
					return &model.LLMResponse{
						Content: genai.NewContentFromText("hello from on_model_error_callback", genai.RoleModel),
					}, fmt.Errorf("error from on_model_error_callback: %w", http.ErrHijacked)
				},
			},
			llmResponses: []*genai.Content{
				genai.NewContentFromText("hello from model", genai.RoleModel),
			},
			dontRunBeforeCanonicalCallback:  true,
			dontRunOnErrorCanonicalCallback: true,
			dontRunAfterCanonicalCallback:   true,
			wantTexts: []string{
				"hello from before_model_callback",
			},
		},
		{
			name: "after error callback process on model error callback message",
			onModelErrorCallback: []llmagent.OnModelErrorCallback{
				func(ctx agent.CallbackContext, llmRequest *model.LLMRequest, llmError error) (*model.LLMResponse, error) {
					return &model.LLMResponse{
						Content: genai.NewContentFromText("hello from on_model_error_callback", genai.RoleModel),
					}, nil
				},
			},
			afterModelCallbacks: []llmagent.AfterModelCallback{
				func(ctx agent.CallbackContext, llmResponse *model.LLMResponse, llmResponseError error) (*model.LLMResponse, error) {
					return &model.LLMResponse{
						Content: genai.NewContentFromText("hello from after_model_callback", genai.RoleModel),
					}, nil
				},
			},
			llmResponses:                    []*genai.Content{},
			dontRunOnErrorCanonicalCallback: true,
			dontRunAfterCanonicalCallback:   true,
			wantTexts: []string{
				"hello from after_model_callback",
			},
		},
		{
			name: "after error callback does not process on model error callback error",
			onModelErrorCallback: []llmagent.OnModelErrorCallback{
				func(ctx agent.CallbackContext, llmRequest *model.LLMRequest, llmError error) (*model.LLMResponse, error) {
					return nil, fmt.Errorf("error from on_model_error_callback: %w", http.ErrNoCookie)
				},
			},
			afterModelCallbacks: []llmagent.AfterModelCallback{
				func(ctx agent.CallbackContext, llmResponse *model.LLMResponse, llmResponseError error) (*model.LLMResponse, error) {
					return nil, fmt.Errorf("error from after_model_callback: %w", http.ErrHijacked)
				},
			},
			llmResponses:                    []*genai.Content{},
			dontRunOnErrorCanonicalCallback: true,
			dontRunAfterCanonicalCallback:   true,
			wantErr:                         http.ErrNoCookie,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			maxLen := max(len(tc.beforeModelCallbacks), len(tc.afterModelCallbacks), len(tc.onModelErrorCallback))
			var plugins []*plugin.Plugin
			for i := range maxLen {
				var currentBefore llmagent.BeforeModelCallback
				var currentAfter llmagent.AfterModelCallback
				var currentError llmagent.OnModelErrorCallback

				// 2. Bounds checks: Only assign if i is within the slice limits
				if i < len(tc.beforeModelCallbacks) {
					currentBefore = tc.beforeModelCallbacks[i]
				}
				if i < len(tc.afterModelCallbacks) {
					currentAfter = tc.afterModelCallbacks[i]
				}
				if i < len(tc.onModelErrorCallback) {
					currentError = tc.onModelErrorCallback[i]
				}
				p, err := plugin.New(plugin.Config{
					Name:                 fmt.Sprintf("plugin-%d", i),
					BeforeModelCallback:  currentBefore,
					AfterModelCallback:   currentAfter,
					OnModelErrorCallback: currentError,
				})
				if err != nil {
					t.Errorf("failed to initialize plugin: %v", err)
				}
				plugins = append(plugins, p)
			}

			onModelErrorCallbacksCalled := false
			beforeModelCallbacksCalled := false
			afterModelCallbacksCalled := false

			onModelErrorCallbacks := []llmagent.OnModelErrorCallback{
				func(ctx agent.CallbackContext, llmRequest *model.LLMRequest, llmError error) (*model.LLMResponse, error) {
					onModelErrorCallbacksCalled = true
					if tc.dontRunOnErrorCanonicalCallback {
						t.Error("on model error should not be called")
					}
					return nil, nil
				},
			}
			beforeModelCallbacks := []llmagent.BeforeModelCallback{
				func(ctx agent.CallbackContext, llmRequest *model.LLMRequest) (*model.LLMResponse, error) {
					beforeModelCallbacksCalled = true
					if tc.dontRunBeforeCanonicalCallback {
						t.Error("before model Callback should not be called")
					}
					return nil, nil
				},
			}
			afterModelCallbacks := []llmagent.AfterModelCallback{
				func(ctx agent.CallbackContext, llmResponse *model.LLMResponse, llmResponseError error) (*model.LLMResponse, error) {
					afterModelCallbacksCalled = true
					if tc.dontRunAfterCanonicalCallback {
						t.Error("after model Callback should not be called")
					}
					return nil, nil
				},
			}

			testLLM := &testutil.MockModel{
				Responses: tc.llmResponses,
			}
			a, err := llmagent.New(llmagent.Config{
				Name:                  "hello_world_agent",
				Model:                 testLLM,
				OnModelErrorCallbacks: onModelErrorCallbacks,
				BeforeModelCallbacks:  beforeModelCallbacks,
				AfterModelCallbacks:   afterModelCallbacks,
			})
			if err != nil {
				t.Fatalf("failed to create llm agent: %v", err)
			}
			runner := testutil.NewTestAgentRunnerWithPluginManager(t, a, runner.PluginConfig{
				Plugins: plugins,
			})
			stream := runner.Run(t, "test_session", "")
			texts, err := testutil.CollectTextParts(stream)
			if tc.wantErr != nil && !errors.Is(err, tc.wantErr) {
				t.Fatalf("stream = (%q, %v), want (_, %v)", texts, err, tc.wantErr)
			}
			if (err != nil) != (tc.wantErr != nil) {
				t.Fatalf("unexpected result from agent, got error: %v, want error: %v", err, tc.wantErr)
			}

			if diff := cmp.Diff(tc.wantTexts, texts); diff != "" {
				t.Fatalf("unexpected result from agent, want: %v, got: %v, diff: %v", tc.wantTexts, texts, diff)
			}

			if onModelErrorCallbacksCalled == false && tc.dontRunOnErrorCanonicalCallback == false {
				t.Error("on model error should be called")
			}
			if beforeModelCallbacksCalled == false && tc.dontRunBeforeCanonicalCallback == false {
				t.Error("before model should be called")
			}
			if afterModelCallbacksCalled == false && tc.dontRunAfterCanonicalCallback == false {
				t.Error("after model should be called")
			}
		})
	}
}
