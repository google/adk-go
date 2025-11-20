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

package llmagent_test

import (
	"errors"
	"fmt"
	"iter"
	"net/http"
	"path/filepath"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/internal/httprr"
	"google.golang.org/adk/internal/testutil"
	"google.golang.org/adk/model"
	"google.golang.org/adk/model/gemini"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/functiontool"
)

const modelName = "gemini-2.0-flash"

//go:generate go test -httprecord=Test

func TestLLMAgent(t *testing.T) {
	errNoNetwork := errors.New("no network")

	for _, tc := range []struct {
		name      string
		transport http.RoundTripper
		wantErr   error
	}{
		{
			name:      "healthy_backend",
			transport: nil, // httprr + http.DefaultTransport
		},
		{
			name:      "broken_backend",
			transport: roundTripperFunc(func(*http.Request) (*http.Response, error) { return nil, errNoNetwork }),
			wantErr:   errNoNetwork,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			model := newGeminiModel(t, modelName, tc.transport)
			a, err := llmagent.New(llmagent.Config{
				Name:                     "hello_world_agent",
				Description:              "hello world agent",
				Model:                    model,
				Instruction:              "Roll the dice and report only the result.",
				GlobalInstruction:        "Answer as precisely as possible.",
				DisallowTransferToParent: true,
				DisallowTransferToPeers:  true,
			})
			if err != nil {
				t.Fatalf("NewLLMAgent failed: %v", err)
			}
			// TODO: set tools, planner.
			runner := testutil.NewTestAgentRunner(t, a)
			stream := runner.Run(t, "test_session", "")
			texts, err := testutil.CollectTextParts(stream)
			if tc.wantErr != nil && !errors.Is(err, tc.wantErr) {
				t.Fatalf("stream = (%q, %v), want (_, %v)", texts, err, tc.wantErr)
			}
			if tc.wantErr == nil && (err != nil || len(texts) != 1) {
				t.Fatalf("stream = (%q, %v), want exactly one text response", texts, err)
			}
		})
	}
}

func TestLLMAgentStreamingModeSSE(t *testing.T) {
	model := newGeminiModel(t, "gemini-2.5-flash", nil)
	a, err := llmagent.New(llmagent.Config{
		Name:                     "calculator",
		Description:              "calculating agent",
		Model:                    model,
		Instruction:              "Think deep. Always double check the answer before making the conclusion.",
		DisallowTransferToParent: true,
		DisallowTransferToPeers:  true,
		GenerateContentConfig: &genai.GenerateContentConfig{
			ThinkingConfig: &genai.ThinkingConfig{
				IncludeThoughts: true, // can trigger multiple message.
			},
		},
	})
	if err != nil {
		t.Fatalf("NewLLMAgent failed: %v", err)
	}
	testRunner := testutil.NewTestAgentRunner(t, a)
	stream := testRunner.RunContentWithConfig(t, "test_session", genai.NewContentFromText("What is the sum of the first 50 prime numbers?", "user"), agent.RunConfig{StreamingMode: agent.StreamingModeSSE})
	events, err := testutil.CollectEvents(stream)
	gotThought := false
	numContents := 0
	for _, e := range events {
		t.Logf("event: %v", e)
		if e.LLMResponse.Content == nil {
			continue
		}
		numContents++
		for _, p := range e.LLMResponse.Content.Parts {
			if p.Thought {
				gotThought = true
			}
		}
	}
	if err != nil {
		t.Fatalf("stream = (_, %v), want (_, nil)", err)
	}
	if numContents <= 1 {
		t.Errorf("stream returned %d events with content, want more than 1 event", numContents)
	}
	if !gotThought {
		t.Error("stream returned no thought, want thought")
	}
}

func TestModelCallbacks(t *testing.T) {
	t.Parallel()

	for _, tc := range []struct {
		name                 string
		llmResponses         []*genai.Content
		beforeModelCallbacks []llmagent.BeforeModelCallback
		afterModelCallbacks  []llmagent.AfterModelCallback
		wantTexts            []string
		wantErr              error
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
			wantErr: http.ErrNoCookie,
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
			wantErr: http.ErrNoCookie,
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
			wantTexts: []string{
				"hello from model",
			},
		},
		{
			name: "after model callback returns new LLMResponse",
			afterModelCallbacks: []llmagent.AfterModelCallback{
				func(ctx agent.CallbackContext, llmResponse *model.LLMResponse, llmResponseError error) (*model.LLMResponse, error) {
					return &model.LLMResponse{
						Content: genai.NewContentFromText("hello from after_model_callback", genai.RoleModel),
					}, nil
				},
				func(ctx agent.CallbackContext, llmResponse *model.LLMResponse, llmResponseError error) (*model.LLMResponse, error) {
					return &model.LLMResponse{
						Content: genai.NewContentFromText("unexpected text", genai.RoleModel),
					}, nil
				},
			},
			llmResponses: []*genai.Content{
				genai.NewContentFromText("hello from model", genai.RoleModel),
			},
			wantTexts: []string{
				"hello from after_model_callback",
			},
		},
		{
			name: "after model callback returns error",
			afterModelCallbacks: []llmagent.AfterModelCallback{
				func(ctx agent.CallbackContext, llmResponse *model.LLMResponse, llmResponseError error) (*model.LLMResponse, error) {
					return nil, fmt.Errorf("error from after_model_callback: %w", http.ErrNoCookie)
				},
				func(ctx agent.CallbackContext, llmResponse *model.LLMResponse, llmResponseError error) (*model.LLMResponse, error) {
					return nil, fmt.Errorf("error from after_model_callback: %w", http.ErrHijacked)
				},
			},
			llmResponses: []*genai.Content{
				genai.NewContentFromText("hello from model", genai.RoleModel),
			},
			wantErr: http.ErrNoCookie,
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
			wantErr: http.ErrNoCookie,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			testLLM := &testutil.MockModel{
				Responses: tc.llmResponses,
			}
			a, err := llmagent.New(llmagent.Config{
				Name:                 "hello_world_agent",
				Model:                testLLM,
				BeforeModelCallbacks: tc.beforeModelCallbacks,
				AfterModelCallbacks:  tc.afterModelCallbacks,
			})
			if err != nil {
				t.Fatalf("failed to create llm agent: %v", err)
			}
			runner := testutil.NewTestAgentRunner(t, a)
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
		})
	}
}

func TestToolCallback(t *testing.T) {
	model := newGeminiModel(t, modelName, nil)

	type Args struct {
		Seed int `json:"seed"`
	}
	type Result struct {
		Number int `json:"number"`
	}

	handler := func(_ tool.Context, input Args) (Result, error) {
		return Result{Number: 1}, nil
	}
	rand, _ := functiontool.New(functiontool.Config{
		Name:        "rand_number",
		Description: "returns random number",
	}, handler)

	t.Run("before_callback_response_used", func(t *testing.T) {
		agent, err := llmagent.New(llmagent.Config{
			Name:                     "agent",
			Description:              "random agent",
			Model:                    model,
			Instruction:              "output ONLY the result computed by the provided function",
			DisallowTransferToParent: true,
			DisallowTransferToPeers:  true,
			Tools:                    []tool.Tool{rand},
			BeforeToolCallbacks: []llmagent.BeforeToolCallback{
				func(ctx tool.Context, tool tool.Tool, args map[string]any) (map[string]any, error) {
					return nil, nil
				},
				func(ctx tool.Context, tool tool.Tool, args map[string]any) (map[string]any, error) {
					return map[string]any{"number": "7"}, nil
				},
			},
		})
		if err != nil {
			t.Fatalf("failed to create LLM Agent: %v", err)
		}

		runner := testutil.NewTestAgentRunner(t, agent)
		stream := runner.Run(t, "session1", "Generate random number with 5 as a seed.")

		ans, err := testutil.CollectTextParts(stream)
		if err != nil || len(ans) == 0 {
			t.Fatalf("agent returned (%v, %v), want result", ans, err)
		}
		if got, want := strings.TrimSpace(ans[len(ans)-1]), "7"; got != want {
			t.Errorf("unexpected result from agent = (%v, %v), want ([%q], nil)", ans, err, want)
		}
	})

	t.Run("extra_before_callback_skipped", func(t *testing.T) {
		agent, err := llmagent.New(llmagent.Config{
			Name:                     "agent",
			Description:              "random agent",
			Model:                    model,
			Instruction:              "output ONLY the result computed by the provided function",
			DisallowTransferToParent: true,
			DisallowTransferToPeers:  true,
			Tools:                    []tool.Tool{rand},
			BeforeToolCallbacks: []llmagent.BeforeToolCallback{
				// Since it retursn non nil, the next callback won't be executed.
				func(ctx tool.Context, tool tool.Tool, args map[string]any) (map[string]any, error) {
					return map[string]any{"number": "3"}, nil
				},
				func(ctx tool.Context, tool tool.Tool, args map[string]any) (map[string]any, error) {
					return map[string]any{"number": "7"}, nil
				},
			},
		})
		if err != nil {
			t.Fatalf("failed to create LLM Agent: %v", err)
		}

		runner := testutil.NewTestAgentRunner(t, agent)
		stream := runner.Run(t, "session1", "Generate random number with 5 as a seed.")

		ans, err := testutil.CollectTextParts(stream)
		if err != nil || len(ans) == 0 {
			t.Fatalf("agent returned (%v, %v), want result", ans, err)
		}
		if got, want := strings.TrimSpace(ans[len(ans)-1]), "3"; got != want {
			t.Errorf("unexpected result from agent = (%v, %v), want ([%q], nil)", ans, err, want)
		}
	})

	t.Run("after_callback_response_used", func(t *testing.T) {
		agent, err := llmagent.New(llmagent.Config{
			Name:                     "agent",
			Description:              "random agent",
			Model:                    model,
			Instruction:              "output ONLY the result computed by the provided function",
			DisallowTransferToParent: true,
			DisallowTransferToPeers:  true,
			Tools:                    []tool.Tool{rand},
			AfterToolCallbacks: []llmagent.AfterToolCallback{
				func(ctx tool.Context, tool tool.Tool, args, result map[string]any, err error) (map[string]any, error) {
					return nil, nil
				},
				func(ctx tool.Context, tool tool.Tool, args, result map[string]any, err error) (map[string]any, error) {
					return map[string]any{"number": "7"}, nil
				},
			},
		})
		if err != nil {
			t.Fatalf("failed to create LLM Agent: %v", err)
		}

		runner := testutil.NewTestAgentRunner(t, agent)
		stream := runner.Run(t, "session1", "Generate random number with 5 as a seed.")

		ans, err := testutil.CollectTextParts(stream)
		if err != nil || len(ans) == 0 {
			t.Fatalf("agent returned (%v, %v), want result", ans, err)
		}
		if got, want := strings.TrimSpace(ans[len(ans)-1]), "7"; got != want {
			t.Errorf("unexpected result from agent = (%v, %v), want ([%q], nil)", ans, err, want)
		}
	})

	t.Run("extra_after_callback_skipped", func(t *testing.T) {
		agent, err := llmagent.New(llmagent.Config{
			Name:                     "agent",
			Description:              "random agent",
			Model:                    model,
			Instruction:              "output ONLY the result computed by the provided function",
			DisallowTransferToParent: true,
			DisallowTransferToPeers:  true,
			Tools:                    []tool.Tool{rand},
			AfterToolCallbacks: []llmagent.AfterToolCallback{
				// Since it retursn non nil, the next callback won't be executed.
				func(ctx tool.Context, tool tool.Tool, args, result map[string]any, err error) (map[string]any, error) {
					return map[string]any{"number": "3"}, nil
				},
				func(ctx tool.Context, tool tool.Tool, args, result map[string]any, err error) (map[string]any, error) {
					return map[string]any{"number": "7"}, nil
				},
			},
		})
		if err != nil {
			t.Fatalf("failed to create LLM Agent: %v", err)
		}

		runner := testutil.NewTestAgentRunner(t, agent)
		stream := runner.Run(t, "session1", "Generate random number with 5 as a seed.")

		ans, err := testutil.CollectTextParts(stream)
		if err != nil || len(ans) == 0 {
			t.Fatalf("agent returned (%v, %v), want result", ans, err)
		}
		if got, want := strings.TrimSpace(ans[len(ans)-1]), "3"; got != want {
			t.Errorf("unexpected result from agent = (%v, %v), want ([%q], nil)", ans, err, want)
		}
	})

	t.Run("after_callback_returned_when_used_with_before_callback", func(t *testing.T) {
		agent, err := llmagent.New(llmagent.Config{
			Name:                     "agent",
			Description:              "random agent",
			Model:                    model,
			Instruction:              "output ONLY the result computed by the provided function",
			DisallowTransferToParent: true,
			DisallowTransferToPeers:  true,
			Tools:                    []tool.Tool{rand},
			BeforeToolCallbacks: []llmagent.BeforeToolCallback{
				func(ctx tool.Context, tool tool.Tool, args map[string]any) (map[string]any, error) {
					return map[string]any{"number": "3"}, nil
				},
			},
			AfterToolCallbacks: []llmagent.AfterToolCallback{
				func(ctx tool.Context, tool tool.Tool, args, result map[string]any, err error) (map[string]any, error) {
					return map[string]any{"number": "7"}, nil
				},
			},
		})
		if err != nil {
			t.Fatalf("failed to create LLM Agent: %v", err)
		}

		runner := testutil.NewTestAgentRunner(t, agent)
		stream := runner.Run(t, "session1", "Generate random number with 5 as a seed.")

		ans, err := testutil.CollectTextParts(stream)
		if err != nil || len(ans) == 0 {
			t.Fatalf("agent returned (%v, %v), want result", ans, err)
		}
		if got, want := strings.TrimSpace(ans[len(ans)-1]), "7"; got != want {
			t.Errorf("unexpected result from agent = (%v, %v), want ([%q], nil)", ans, err, want)
		}
	})

	t.Run("both_callbacks_return_nil_actual_tool_is_executed", func(t *testing.T) {
		agent, err := llmagent.New(llmagent.Config{
			Name:                     "agent",
			Description:              "random agent",
			Model:                    model,
			Instruction:              "output ONLY the result computed by the provided function",
			DisallowTransferToParent: true,
			DisallowTransferToPeers:  true,
			Tools:                    []tool.Tool{rand},
			BeforeToolCallbacks: []llmagent.BeforeToolCallback{
				func(ctx tool.Context, tool tool.Tool, args map[string]any) (map[string]any, error) {
					return nil, nil
				},
			},
			AfterToolCallbacks: []llmagent.AfterToolCallback{
				func(ctx tool.Context, tool tool.Tool, args, result map[string]any, err error) (map[string]any, error) {
					return nil, nil
				},
			},
		})
		if err != nil {
			t.Fatalf("failed to create LLM Agent: %v", err)
		}

		runner := testutil.NewTestAgentRunner(t, agent)
		stream := runner.Run(t, "session1", "Generate random number with 5 as a seed.")

		ans, err := testutil.CollectTextParts(stream)
		if err != nil || len(ans) == 0 {
			t.Fatalf("agent returned (%v, %v), want result", ans, err)
		}
		if got, want := strings.TrimSpace(ans[len(ans)-1]), "1"; got != want {
			t.Errorf("unexpected result from agent = (%v, %v), want ([%q], nil)", ans, err, want)
		}
	})
}

func TestInstructionProvider(t *testing.T) {
	t.Parallel()

	for _, tc := range []struct {
		name              string
		llmagentFunc      func(model.LLM) (agent.Agent, error)
		wantLLMRequests   []*model.LLMRequest
		wantAgentResponse []string
		wantErr           error
	}{
		{
			name: "instruction is evaluated",
			llmagentFunc: func(model model.LLM) (agent.Agent, error) {
				return llmagent.New(llmagent.Config{
					Name:        "test_agent",
					Model:       model,
					Instruction: "instruction {var} test",
				})
			},
			wantLLMRequests: []*model.LLMRequest{
				{
					Contents: []*genai.Content{
						genai.NewContentFromText("user input", genai.RoleUser),
					},
					Config: &genai.GenerateContentConfig{
						SystemInstruction: genai.NewContentFromText("instruction custom_value test", genai.RoleUser),
					},
				},
			},
			wantAgentResponse: []string{
				"llm resp stub",
			},
		},
		{
			name: "instruction provider overrides instruction",
			llmagentFunc: func(model model.LLM) (agent.Agent, error) {
				return llmagent.New(llmagent.Config{
					Name:        "test_agent",
					Model:       model,
					Instruction: "instruction",
					InstructionProvider: func(ctx agent.ReadonlyContext) (string, error) {
						return "instruction provider template {var} not evaluated", nil
					},
				})
			},
			wantLLMRequests: []*model.LLMRequest{
				{
					Contents: []*genai.Content{
						genai.NewContentFromText("user input", genai.RoleUser),
					},
					Config: &genai.GenerateContentConfig{
						SystemInstruction: genai.NewContentFromText("instruction provider template {var} not evaluated", genai.RoleUser),
					},
				},
			},
			wantAgentResponse: []string{
				"llm resp stub",
			},
		},
		{
			name: "global instruction provider overrides global instruction",
			llmagentFunc: func(model model.LLM) (agent.Agent, error) {
				return llmagent.New(llmagent.Config{
					Name:              "test_agent",
					Model:             model,
					GlobalInstruction: "instruction",
					GlobalInstructionProvider: func(ctx agent.ReadonlyContext) (string, error) {
						return "global instruction provider template {var} not evaluated", nil
					},
				})
			},
			wantLLMRequests: []*model.LLMRequest{
				{
					Contents: []*genai.Content{
						genai.NewContentFromText("user input", genai.RoleUser),
					},
					Config: &genai.GenerateContentConfig{
						SystemInstruction: genai.NewContentFromText("global instruction provider template {var} not evaluated", genai.RoleUser),
					},
				},
			},
			wantAgentResponse: []string{
				"llm resp stub",
			},
		},
		{
			name: "global instruction provider merged with instruction provider",
			llmagentFunc: func(model model.LLM) (agent.Agent, error) {
				return llmagent.New(llmagent.Config{
					Name:  "test_agent",
					Model: model,
					InstructionProvider: func(ctx agent.ReadonlyContext) (string, error) {
						return "instruction provider {var}", nil
					},
					GlobalInstructionProvider: func(ctx agent.ReadonlyContext) (string, error) {
						return "global instruction provider {var}", nil
					},
				})
			},
			wantLLMRequests: []*model.LLMRequest{
				{
					Contents: []*genai.Content{
						genai.NewContentFromText("user input", genai.RoleUser),
					},
					Config: &genai.GenerateContentConfig{
						SystemInstruction: &genai.Content{
							Parts: []*genai.Part{
								genai.NewPartFromText("global instruction provider {var}"),
								genai.NewPartFromText("instruction provider {var}"),
							},
							Role: genai.RoleUser,
						},
					},
				},
			},
			wantAgentResponse: []string{
				"llm resp stub",
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			model := &testutil.MockModel{
				Responses: []*genai.Content{
					genai.NewContentFromText("llm resp stub", genai.RoleModel),
				},
			}

			agent, err := tc.llmagentFunc(model)
			if err != nil {
				t.Fatalf("failed to create LLM Agent: %v", err)
			}

			testRunner := testutil.NewTestAgentRunner(t, agent)
			testRunner.SetInitSessionState(map[string]any{"var": "custom_value"})

			stream := testRunner.Run(t, "session", "user input")

			gotResp, err := testutil.CollectTextParts(stream)
			if err != nil {
				t.Fatalf("agent returned (%v, %v), want result", gotResp, err)
			}

			if diff := cmp.Diff(tc.wantLLMRequests, model.Requests); diff != "" {
				t.Errorf("unexpected LLM requests, want: %v, got: %v, diff: %v", tc.wantLLMRequests, model.Requests, diff)
			}

			if diff := cmp.Diff(tc.wantAgentResponse, gotResp); diff != "" {
				t.Errorf("unexpected agent response, want: %v, got: %v, diff: %v", tc.wantAgentResponse, gotResp, diff)
			}
		})
	}
}

func TestFunctionTool(t *testing.T) {
	model := newGeminiModel(t, modelName, nil)

	type Args struct {
		A int `json:"a"`
		B int `json:"b"`
	}
	type Result struct {
		Sum int `json:"sum"`
	}

	prompt := "what is the sum of 1 + 2?"
	handler := func(_ tool.Context, input Args) (Result, error) {
		if input.A != 1 || input.B != 2 {
			t.Errorf("handler received %+v, want {a: 1, b: 2}", input)
		}
		return Result{Sum: input.A + input.B}, nil
	}
	rand, _ := functiontool.New(functiontool.Config{
		Name:        "sum",
		Description: "computes the sum of two numbers",
	}, handler)

	agent, err := llmagent.New(llmagent.Config{
		Name:        "agent",
		Description: "math agent",
		Model:       model,
		Instruction: "output ONLY the result computed by the provided function",
		// TODO(hakim): set to false when autoflow is implemented.
		DisallowTransferToParent: true,
		DisallowTransferToPeers:  true,
		Tools:                    []tool.Tool{rand},
	})
	if err != nil {
		t.Fatalf("failed to create LLM Agent: %v", err)
	}

	runner := testutil.NewTestAgentRunner(t, agent)
	stream := runner.Run(t, "session1", prompt)

	ans, err := testutil.CollectTextParts(stream)
	if err != nil || len(ans) == 0 {
		t.Fatalf("agent returned (%v, %v), want result", ans, err)
	}
	if got, want := strings.TrimSpace(ans[len(ans)-1]), "3"; got != want {
		t.Errorf("unexpected result from agent = (%v, %v), want ([%q], nil)", ans, err, want)
	}
}

func TestAgentTransfer(t *testing.T) {
	// Helpers to create genai.Content conveniently.
	transferCall := func(agentName string) *genai.Content {
		return genai.NewContentFromFunctionCall(
			"transfer_to_agent",
			map[string]any{"agent_name": agentName},
			"model",
		)
	}
	transferResponse := func() *genai.Content {
		return genai.NewContentFromFunctionResponse(
			"transfer_to_agent", map[string]any{}, "user")
	}
	text := func(text string) *genai.Content {
		return genai.NewContentFromText(
			text,
			"model",
		)
	}
	// returns a model that returns the prepopulated resp one by one.
	testModel := func(resp ...*genai.Content) model.LLM {
		return &testutil.MockModel{Responses: resp}
	}

	type content struct {
		Author string
		Parts  []*genai.Part
	}
	// contents returns (Author, Parts) stream extracted from the event stream.
	contents := func(stream iter.Seq2[*session.Event, error]) ([]content, error) {
		var ret []content
		for ev, err := range stream {
			if err != nil {
				return nil, err
			}
			if ev.LLMResponse.Content == nil {
				return nil, fmt.Errorf("unexpected event: %v", ev)
			}
			for _, p := range ev.LLMResponse.Content.Parts {
				if p.FunctionCall != nil {
					p.FunctionCall.ID = ""
				}
				if p.FunctionResponse != nil {
					p.FunctionResponse.ID = ""
				}
			}
			ret = append(ret, content{Author: ev.Author, Parts: ev.LLMResponse.Content.Parts})
		}
		return ret, nil
	}

	check := func(t *testing.T, rootAgent agent.Agent, wants [][]content) {
		runner := testutil.NewTestAgentRunner(t, rootAgent)
		for i := range len(wants) {
			got, err := contents(runner.Run(t, "session_id", fmt.Sprintf("round %d", i)))
			if err != nil {
				t.Fatalf("[round $d]: stream ended with an error: %v", err)
			}
			if diff := cmp.Diff(wants[i], got); diff != "" {
				t.Errorf("[round %d] events diff (-want, +got) = %v", i, diff)
			}
		}
	}

	t.Run("auto_to_auto", func(t *testing.T) {
		// root_agent -- sub_agent_1
		model := testModel(
			transferCall("sub_agent_1"),
			text("response1"),
			text("response2"))

		subAgent1, err := llmagent.New(llmagent.Config{
			Name:  "sub_agent_1",
			Model: model,
		})
		if err != nil {
			t.Fatalf("failed to create subAgent1: %v", err)
		}

		rootAgent, err := llmagent.New(llmagent.Config{
			Name:      "root_agent",
			Model:     model,
			SubAgents: []agent.Agent{subAgent1},
		})
		if err != nil {
			t.Fatalf("failed to create rootAgent: %v", err)
		}

		check(t, rootAgent, [][]content{
			0: {
				{"root_agent", transferCall("sub_agent_1").Parts},
				{"root_agent", transferResponse().Parts},
				{"sub_agent_1", text("response1").Parts},
			},
			1: { // rootAgent should still be the current agent.
				{"sub_agent_1", text("response2").Parts},
			},
		})
	})

	t.Run("auto_to_single", func(t *testing.T) {
		// root_agent -- sub_agent_1 (single)
		model := testModel(
			transferCall("sub_agent_1"),
			text("response1"),
			text("response2"))

		subAgent1, err := llmagent.New(llmagent.Config{
			Name:                     "sub_agent_1",
			Model:                    model,
			DisallowTransferToParent: true,
			DisallowTransferToPeers:  true,
		})
		if err != nil {
			t.Fatalf("failed to create subAgent1: %v", err)
		}

		rootAgent, err := llmagent.New(llmagent.Config{
			Name:      "root_agent",
			Model:     model,
			SubAgents: []agent.Agent{subAgent1},
		})
		if err != nil {
			t.Fatalf("failed to create rootAgent: %v", err)
		}

		check(t, rootAgent, [][]content{
			0: {
				{"root_agent", transferCall("sub_agent_1").Parts},
				{"root_agent", transferResponse().Parts},
				{"sub_agent_1", text("response1").Parts},
			},
			1: { // rootAgent should still be the current agent.
				{"root_agent", text("response2").Parts},
			},
		})
	})

	t.Run("auto_to_auto_to_single", func(t *testing.T) {
		// root_agent -- sub_agent_1 -- sub_agent_1_1
		model := testModel(
			transferCall("sub_agent_1"),
			transferCall("sub_agent_1_1"),
			text("response1"),
			text("response2"))

		subAgent1_1, err := llmagent.New(llmagent.Config{
			Name:                     "sub_agent_1_1",
			Model:                    model,
			DisallowTransferToParent: true,
			DisallowTransferToPeers:  true,
		})
		if err != nil {
			t.Fatalf("failed to create subAgent1_1: %v", err)
		}

		subAgent1, err := llmagent.New(llmagent.Config{
			Name:      "sub_agent_1",
			Model:     model,
			SubAgents: []agent.Agent{subAgent1_1},
		})
		if err != nil {
			t.Fatalf("failed to create subAgent1: %v", err)
		}

		rootAgent, err := llmagent.New(llmagent.Config{
			Name:      "root_agent",
			Model:     model,
			SubAgents: []agent.Agent{subAgent1},
		})
		if err != nil {
			t.Fatalf("failed to create rootAgent: %v", err)
		}

		check(t, rootAgent, [][]content{
			0: {
				{"root_agent", transferCall("sub_agent_1").Parts},
				{"root_agent", transferResponse().Parts},
				{"sub_agent_1", transferCall("sub_agent_1_1").Parts},
				{"sub_agent_1", transferResponse().Parts},
				{"sub_agent_1_1", text("response1").Parts},
			},
			1: {
				// sub_agent_1 should still be the current agent.
				// sub_agent_1_1 is single, so it should not be the current agent.
				// Otherwise, the conversation will be tied to sub_agent_1_1 forever.
				{"sub_agent_1", text("response2").Parts},
			},
		})
	})

	// TODO: cover cases similar to adk-python's
	// tests/unittests/flows/llm_flows/test_agent_transfer.py
	//   - test_auto_to_sequential
	//   - test_auto_to_sequential_to_auto
	//   - test_auto_to_loop
}

func newGeminiModel(t *testing.T, modelName string, transport http.RoundTripper) model.LLM {
	apiKey := "fakeKey"
	if transport == nil { // use httprr
		trace := filepath.Join("testdata", strings.ReplaceAll(t.Name()+".httprr", "/", "_"))
		recording := false
		transport, recording = newGeminiTestClientConfig(t, trace)
		if recording { // if we are recording httprr trace, don't use the fakeKey.
			apiKey = ""
		}
	}
	model, err := gemini.NewModel(t.Context(), modelName, &genai.ClientConfig{
		HTTPClient: &http.Client{Transport: transport},
		APIKey:     apiKey,
	})
	if err != nil {
		t.Fatalf("failed to create model: %v", err)
	}
	return model
}

func newGeminiTestClientConfig(t *testing.T, rrfile string) (http.RoundTripper, bool) {
	t.Helper()
	rr, err := testutil.NewGeminiTransport(rrfile)
	if err != nil {
		t.Fatal(err)
	}
	recording, _ := httprr.Recording(rrfile)
	return rr, recording
}

type roundTripperFunc func(*http.Request) (*http.Response, error)

// RoundTrip implements http.RoundTripper.
func (fn roundTripperFunc) RoundTrip(req *http.Request) (*http.Response, error) {
	return fn(req)
}
