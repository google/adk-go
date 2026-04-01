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

package llminternal_test

import (
	"net/http"
	"path/filepath"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/internal/httprr"
	"google.golang.org/adk/internal/testutil"
	"google.golang.org/adk/model"
	"google.golang.org/adk/model/gemini"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/functiontool"
)

type SumArgs struct {
	A int `json:"a"` // an integer to sum
	B int `json:"b"` // another integer to sum
}
type SumResult struct {
	Sum int `json:"sum"` // the sum of two integers
}

func sumFunc(ctx tool.Context, input SumArgs) (SumResult, error) {
	return SumResult{Sum: input.A + input.B}, nil
}

var expectedNonPartialLLMResponse25Flash = []*model.LLMResponse{
	{
		Partial: false,
		Content: &genai.Content{
			Parts: []*genai.Part{
				genai.NewPartFromFunctionCall("sum", map[string]any{
					"a": 2.0,
					"b": 3.0,
				}),
				genai.NewPartFromFunctionCall("sum", map[string]any{
					"a": 4.0,
					"b": 5.0,
				}),
				genai.NewPartFromFunctionCall("sum", map[string]any{
					"a": 6.0,
					"b": 7.0,
				}),
			},
			Role: "model",
		},
	},
	{
		Partial: false,
		Content: &genai.Content{
			Parts: []*genai.Part{
				genai.NewPartFromFunctionResponse("sum", map[string]any{
					"sum": 5.0,
				}),
				genai.NewPartFromFunctionResponse("sum", map[string]any{
					"sum": 9.0,
				}),
				genai.NewPartFromFunctionResponse("sum", map[string]any{
					"sum": 13.0,
				}),
			},
			Role: "user",
		},
	},
	{
		Partial: false,
		Content: &genai.Content{
			Parts: []*genai.Part{
				genai.NewPartFromText("The sum of 2 and 3 is 5.\nThe sum of 4 and 5 is 9.\nThe sum of 6 and 7 is 13."),
			},
			Role: "model",
		},
	},
	{
		Partial: false,
		Content: &genai.Content{
			Parts: []*genai.Part{
				genai.NewPartFromFunctionCall("sum", map[string]any{
					"a": 10.0,
					"b": 20.0,
				}),
				genai.NewPartFromFunctionCall("sum", map[string]any{
					"a": 40.0,
					"b": 50.0,
				}),
				genai.NewPartFromFunctionCall("sum", map[string]any{
					"a": 60.0,
					"b": 70.0,
				}),
			},
			Role: "model",
		},
	},
	{
		Partial: false,
		Content: &genai.Content{
			Parts: []*genai.Part{
				genai.NewPartFromFunctionResponse("sum", map[string]any{
					"sum": 30.0,
				}),
				genai.NewPartFromFunctionResponse("sum", map[string]any{
					"sum": 90.0,
				}),
				genai.NewPartFromFunctionResponse("sum", map[string]any{
					"sum": 130.0,
				}),
			},
			Role: "user",
		},
	},
	{
		Partial: false,
		Content: &genai.Content{
			Parts: []*genai.Part{
				genai.NewPartFromText("The sum of 10 and 20 is 30.\nThe sum of 40 and 50 is 90.\nThe sum of 60 and 70 is 130."),
			},
			Role: "model",
		},
	},
}

var expectedNonPartialLLMResponse3FlashPreview = []*model.LLMResponse{
	{
		Partial: false,
		Content: &genai.Content{
			Parts: []*genai.Part{
				genai.NewPartFromFunctionCall("sum", map[string]any{
					"a": 2.0,
					"b": 3.0,
				}),
				genai.NewPartFromFunctionCall("sum", map[string]any{
					"a": 4.0,
					"b": 5.0,
				}),
				genai.NewPartFromFunctionCall("sum", map[string]any{
					"a": 6.0,
					"b": 7.0,
				}),
			},
			Role: "model",
		},
	},
	{
		Partial: false,
		Content: &genai.Content{
			Parts: []*genai.Part{
				genai.NewPartFromFunctionResponse("sum", map[string]any{
					"sum": 5.0,
				}),
				genai.NewPartFromFunctionResponse("sum", map[string]any{
					"sum": 9.0,
				}),
				genai.NewPartFromFunctionResponse("sum", map[string]any{
					"sum": 13.0,
				}),
			},
			Role: "user",
		},
	},
	{
		Partial: false,
		Content: &genai.Content{
			Parts: []*genai.Part{
				genai.NewPartFromText("The sum of 2 and 3 is 5, the sum of 4 and 5 is 9, and the sum of 6 and 7 is 13."),
			},
			Role: "model",
		},
	},
	{
		Partial: false,
		Content: &genai.Content{
			Parts: []*genai.Part{
				genai.NewPartFromFunctionCall("sum", map[string]any{
					"a": 10.0,
					"b": 20.0,
				}),
				genai.NewPartFromFunctionCall("sum", map[string]any{
					"a": 40.0,
					"b": 50.0,
				}),
				genai.NewPartFromFunctionCall("sum", map[string]any{
					"a": 60.0,
					"b": 70.0,
				}),
			},
			Role: "model",
		},
	},
	{
		Partial: false,
		Content: &genai.Content{
			Parts: []*genai.Part{
				genai.NewPartFromFunctionResponse("sum", map[string]any{
					"sum": 30.0,
				}),
				genai.NewPartFromFunctionResponse("sum", map[string]any{
					"sum": 90.0,
				}),
				genai.NewPartFromFunctionResponse("sum", map[string]any{
					"sum": 130.0,
				}),
			},
			Role: "user",
		},
	},
	{
		Partial: false,
		Content: &genai.Content{
			Parts: []*genai.Part{
				genai.NewPartFromText("The sum of 10 and 20 is 30, the sum of 40 and 50 is 90, and the sum of 60 and 70 is 130."),
			},
			Role: "model",
		},
	},
}

var expectedNonPartialLLMResponse3ProPreview = []*model.LLMResponse{
	{
		Partial: false,
		Content: &genai.Content{
			Parts: []*genai.Part{
				genai.NewPartFromFunctionCall("sum", map[string]any{
					"a": 2.0,
					"b": 3.0,
				}),
				genai.NewPartFromFunctionCall("sum", map[string]any{
					"a": 4.0,
					"b": 5.0,
				}),
				genai.NewPartFromFunctionCall("sum", map[string]any{
					"a": 6.0,
					"b": 7.0,
				}),
			},
			Role: "model",
		},
	},
	{
		Partial: false,
		Content: &genai.Content{
			Parts: []*genai.Part{
				genai.NewPartFromFunctionResponse("sum", map[string]any{
					"sum": 5.0,
				}),
				genai.NewPartFromFunctionResponse("sum", map[string]any{
					"sum": 9.0,
				}),
				genai.NewPartFromFunctionResponse("sum", map[string]any{
					"sum": 13.0,
				}),
			},
			Role: "user",
		},
	},
	{
		Partial: false,
		Content: &genai.Content{
			Parts: []*genai.Part{
				genai.NewPartFromText("Here are the results of your additions:\n* 2 + 3 = 5\n* 4 + 5 = 9\n* 6 + 7 = 13"),
				{}, // empty part with thought signature
			},
			Role: "model",
		},
	},
	{
		Partial: false,
		Content: &genai.Content{
			Parts: []*genai.Part{
				genai.NewPartFromFunctionCall("sum", map[string]any{
					"a": 10.0,
					"b": 20.0,
				}),
				genai.NewPartFromFunctionCall("sum", map[string]any{
					"a": 40.0,
					"b": 50.0,
				}),
				genai.NewPartFromFunctionCall("sum", map[string]any{
					"a": 60.0,
					"b": 70.0,
				}),
			},
			Role: "model",
		},
	},
	{
		Partial: false,
		Content: &genai.Content{
			Parts: []*genai.Part{
				genai.NewPartFromFunctionResponse("sum", map[string]any{
					"sum": 30.0,
				}),
				genai.NewPartFromFunctionResponse("sum", map[string]any{
					"sum": 90.0,
				}),
				genai.NewPartFromFunctionResponse("sum", map[string]any{
					"sum": 130.0,
				}),
			},
			Role: "user",
		},
	},
	{
		Partial: false,
		Content: &genai.Content{
			Parts: []*genai.Part{
				genai.NewPartFromText("Here are the results for those additions:\n* 10 + 20 = 30\n* 40 + 50 = 90\n* 60 + 70 = 130"),
				{}, // empty part with thought signature
			},
			Role: "model",
		},
	},
}

func TestParallelFunctionCalls(t *testing.T) {
	tests := []struct {
		name            string
		modelName       string
		wantLLMResponse []*model.LLMResponse
	}{
		{"gemini-2.5-flash", "gemini-2.5-flash", expectedNonPartialLLMResponse25Flash},
		{"gemini-3-flash-preview", "gemini-3-flash-preview", expectedNonPartialLLMResponse3FlashPreview},
		{"gemini-3.1-pro-preview", "gemini-3.1-pro-preview", expectedNonPartialLLMResponse3ProPreview},
	}
	for _, tt := range tests {
		t.Run("test_parallel_function_calls_"+tt.name, func(t *testing.T) {
			httpRecordFilename := filepath.Join("testdata", strings.ReplaceAll(t.Name(), "/", "_")+".httprr")

			baseTransport, err := testutil.NewGeminiTransport(httpRecordFilename)
			if err != nil {
				t.Fatal(err)
			}

			apiKey := ""
			if recording, _ := httprr.Recording(httpRecordFilename); !recording {
				apiKey = "fakekey"
			}

			cfg := &genai.ClientConfig{
				HTTPClient: &http.Client{Transport: baseTransport},
				APIKey:     apiKey,
			}

			geminiModel, err := gemini.NewModel(t.Context(), tt.modelName, cfg)
			if err != nil {
				t.Fatal(err)
			}

			sumTool, err := functiontool.New(functiontool.Config{
				Name:        "sum",
				Description: "sums two integers",
			}, sumFunc)
			if err != nil {
				t.Fatal(err)
			}

			a, err := llmagent.New(llmagent.Config{
				Name:        "calculator",
				Description: "A calculator that can add two integers",
				Instruction: "You are a calculator assistant. You will recieve requests to add two integers. Respond with the sum of the two integers and you must use the sum tool to calculate the sum.",
				Model:       geminiModel,
				Tools: []tool.Tool{
					sumTool,
				},
			})
			if err != nil {
				t.Fatal(err)
			}

			sessionService := session.InMemoryService()
			_, err = sessionService.Create(t.Context(), &session.CreateRequest{
				AppName:   "testApp",
				UserID:    "testUser",
				SessionID: "testSession",
			})
			if err != nil {
				t.Fatal(err)
			}

			r, err := runner.New(runner.Config{
				Agent:          a,
				SessionService: sessionService,
				AppName:        "testApp",
			})
			if err != nil {
				t.Fatal(err)
			}

			it := r.Run(t.Context(), "testUser", "testSession", &genai.Content{
				Parts: []*genai.Part{
					genai.NewPartFromText("Can you add 2 and 3? Also 4 and 5? And 6 and 7?"),
				},
				Role: "user",
			}, agent.RunConfig{StreamingMode: agent.StreamingModeSSE})

			functionCalls := make([]*genai.FunctionCall, 0)
			functionResponses := make([]*genai.FunctionResponse, 0)
			functionCallsPartial := make([]*genai.FunctionCall, 0)
			functionResponsesPartial := make([]*genai.FunctionResponse, 0)
			nonPartialEvents := make([]*model.LLMResponse, 0)

			handleLoop := func(ev *session.Event) {
				if !ev.Partial {
					nonPartialEvents = append(nonPartialEvents, &ev.LLMResponse)
				}
				if ev.Content != nil {
					for _, part := range ev.Content.Parts {
						if part.FunctionCall != nil {
							if ev.Partial {
								functionCallsPartial = append(functionCallsPartial, part.FunctionCall)
							} else {
								functionCalls = append(functionCalls, part.FunctionCall)
							}
						}
						if part.FunctionResponse != nil {
							if ev.Partial {
								functionResponsesPartial = append(functionResponsesPartial, part.FunctionResponse)
							} else {
								functionResponses = append(functionResponses, part.FunctionResponse)
							}
						}
					}
				}
			}

			for ev, err := range it {
				if err != nil {
					t.Fatal(err)
				}
				handleLoop(ev)
			}

			ignoreFields := []cmp.Option{
				cmpopts.IgnoreFields(genai.FunctionCall{}, "ID"),
				cmpopts.IgnoreFields(genai.Part{}, "ThoughtSignature"),
				cmpopts.IgnoreFields(genai.FunctionResponse{}, "ID"),
				cmpopts.IgnoreFields(model.LLMResponse{}, "UsageMetadata"),
			}

			if len(functionCalls) != 3 || len(functionResponses) != 3 {
				t.Errorf("expected 3 function calls and 3 function responses, got %d function calls and %d function responses", len(functionCalls), len(functionResponses))
			}
			if len(functionCallsPartial) != 3 || len(functionResponsesPartial) != 0 {
				t.Errorf("expected 3 partial function calls and 0 partial function responses, got %d partial function calls and %d partial function responses", len(functionCallsPartial), len(functionResponsesPartial))
			}

			it = r.Run(t.Context(), "testUser", "testSession", &genai.Content{
				Parts: []*genai.Part{
					genai.NewPartFromText("Great, now can you add 10 and 20? Also 40 and 50? And 60 and 70?"),
				},
				Role: "user",
			}, agent.RunConfig{StreamingMode: agent.StreamingModeSSE})
			for ev, err := range it {
				if err != nil {
					t.Fatal(err)
				}
				handleLoop(ev)
			}

			if len(functionCalls) != 6 || len(functionResponses) != 6 {
				t.Errorf("expected 6 function calls and 6 function responses, got %d function calls and %d function responses", len(functionCalls), len(functionResponses))
			}
			if len(functionCallsPartial) != 6 || len(functionResponsesPartial) != 0 {
				t.Errorf("expected 6 partial function calls and 0 partial function responses, got %d partial function calls and %d partial function responses", len(functionCallsPartial), len(functionResponsesPartial))
			}

			for i, ev := range nonPartialEvents {
				if diff := cmp.Diff(tt.wantLLMResponse[i], ev, ignoreFields...); diff != "" {
					t.Errorf("diff in the events: got event[%d]: %v, want: %v, diff: %v", i, ev, tt.wantLLMResponse[i], diff)
				}
				if i == 0 || i == 3 {
					if len(ev.Content.Parts[0].ThoughtSignature) == 0 {
						t.Errorf("expected non-empty thought signature, got empty")
					}
				}
			}
		})
	}
}
