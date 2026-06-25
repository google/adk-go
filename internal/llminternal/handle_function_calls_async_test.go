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
	"context"
	"iter"
	"testing"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/internal/toolinternal/toolutils"
	"google.golang.org/adk/model"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/functiontool"
)

type SleepArgs struct {
	DurationMS int `json:"duration_ms"`
}
type SleepResult struct {
	Success bool `json:"success"`
}

func sleepFunc(_ context.Context, _ agent.Context, input SleepArgs) (SleepResult, error) {
	time.Sleep(time.Duration(input.DurationMS) * time.Millisecond)
	return SleepResult{Success: true}, nil
}

// mockModel is a simple mock model that returns parallel tool calls.
type mockModel struct {
	model.LLM
	Calls int
}

func (m *mockModel) Name() string {
	return "mock-model"
}

func (m *mockModel) GenerateContent(ctx context.Context, req *model.LLMRequest, useStream bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		m.Calls++
		if m.Calls > 1 {
			// Second call should be the final response after tool execution.
			// Or we just return a final response if we don't want to loop.
			// For this test, we just need to trigger the tool calls once.
			yield(&model.LLMResponse{
				Content: &genai.Content{
					Parts: []*genai.Part{
						genai.NewPartFromText("I am done."),
					},
					Role: "model",
				},
				Partial: false,
			}, nil)
			return
		}

		// First call returns parallel tool calls.
		yield(&model.LLMResponse{
			Content: &genai.Content{
				Parts: []*genai.Part{
					{
						FunctionCall: &genai.FunctionCall{
							ID:   "call_1",
							Name: "sleep",
							Args: map[string]any{"duration_ms": 500},
						},
					},
					{
						FunctionCall: &genai.FunctionCall{
							ID:   "call_2",
							Name: "sleep",
							Args: map[string]any{"duration_ms": 500},
						},
					},
					{
						FunctionCall: &genai.FunctionCall{
							ID:   "call_3",
							Name: "sleep",
							Args: map[string]any{"duration_ms": 500},
						},
					},
				},
				Role: "model",
			},
			Partial: false,
		}, nil)
	}
}

func TestHandleFunctionCallsAsync(t *testing.T) {
	sleepTool, err := functiontool.New(functiontool.Config{
		Name:        "sleep",
		Description: "sleeps for a duration",
	}, sleepFunc)
	if err != nil {
		t.Fatal(err)
	}

	model := &mockModel{}

	a, err := llmagent.New(llmagent.Config{
		Name:        "tester",
		Description: "Tester agent",
		Instruction: "You are a tester agent.",
		Model:       model,
		Tools: []tool.Tool{
			sleepTool,
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

	startTime := time.Now()

	it := r.Run(t.Context(), "testUser", "testSession", &genai.Content{
		Parts: []*genai.Part{
			genai.NewPartFromText("Test sleep"),
		},
		Role: "user",
	}, agent.RunConfig{StreamingMode: agent.StreamingModeSSE})

	events := []*session.Event{}
	for ev, err := range it {
		if err != nil {
			t.Fatal(err)
		}
		events = append(events, ev)
	}
	if len(events) != 3 {
		t.Errorf("Expected 3 events, got %d", len(events))
	}

	elapsed := time.Since(startTime)
	t.Logf("Elapsed time: %v", elapsed)

	if len(events[0].Content.Parts) != 3 {
		t.Errorf("Expected first event to have 3 function calls, got %d", len(events[0].Content.Parts))
	}
	if len(events[1].Content.Parts) != 3 {
		t.Errorf("Expected second event to have 3 function responses, got %d", len(events[1].Content.Parts))
	}
	if len(events[2].Content.Parts) != 1 {
		t.Errorf("Expected third event to have 1 text part got %d", len(events[2].Content.Parts))
	}

	// Since we are calling sleep 3 times for 500ms each, synchronous execution would take
	// ~1500ms, while asynchronous execution should take ~500ms.
	// We assert that the time is significantly less than 1500ms to verify async.
	// We also assert it's at least 500ms.

	if elapsed < 500*time.Millisecond {
		t.Errorf("Elapsed time %v is less than expected 500ms", elapsed)
	}

	if elapsed > 1000*time.Millisecond {
		t.Errorf("Elapsed time %v is greater than expected 1000ms for async execution", elapsed)
	}
}

type deferringTool struct {
	name        string
	description string
	result      map[string]any
	defers      bool
	longRunning bool
	runCount    int
}

func (t *deferringTool) Name() string        { return t.name }
func (t *deferringTool) Description() string { return t.description }
func (t *deferringTool) IsLongRunning() bool { return t.longRunning }
func (t *deferringTool) DefersResponse() bool {
	return t.defers
}

func (t *deferringTool) Declaration() *genai.FunctionDeclaration {
	return &genai.FunctionDeclaration{
		Name:        t.name,
		Description: t.description,
		Parameters: &genai.Schema{
			Type: genai.TypeObject,
		},
	}
}

func (t *deferringTool) Run(ctx context.Context, invCleanCtx agent.Context, args any) (map[string]any, error) {
	t.runCount++
	return t.result, nil
}

func (t *deferringTool) ProcessRequest(ctx context.Context, invCleanCtx agent.Context, req *model.LLMRequest) error {
	return toolutils.PackTool(req, t)
}

type singleCallMockModel struct {
	model.LLM
	toolName string
	fcID     string
	calls    int
}

func (m *singleCallMockModel) Name() string { return "single-call-mock" }

func (m *singleCallMockModel) GenerateContent(ctx context.Context, req *model.LLMRequest, useStream bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		m.calls++
		if m.calls > 1 {
			yield(&model.LLMResponse{
				Content: &genai.Content{
					Parts: []*genai.Part{genai.NewPartFromText("done")},
					Role:  "model",
				},
			}, nil)
			return
		}
		yield(&model.LLMResponse{
			Content: &genai.Content{
				Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{
					ID:   m.fcID,
					Name: m.toolName,
					Args: map[string]any{},
				}}},
				Role: "model",
			},
		}, nil)
	}
}

func TestHandleFunctionCalls_FRSkipGates(t *testing.T) {
	cases := []struct {
		name             string
		defers           bool
		longRunning      bool
		toolResult       map[string]any
		wantEventCount   int
		wantFREventIndex int // -1 if no FR event expected
	}{
		{
			name:             "deferred + nil: gate fires, no FR event",
			defers:           true,
			toolResult:       nil,
			wantEventCount:   2, // FC + final text only
			wantFREventIndex: -1,
		},
		{
			// generate FC only, no subsequent LLM call
			name:             "long-running + nil: gate fires, no FR; node parks after FC only",
			longRunning:      true,
			toolResult:       nil,
			wantEventCount:   1,
			wantFREventIndex: -1,
		},
		{
			name:             "deferred + long-running + nil: gate fires, no FR; node parks after FC only",
			defers:           true,
			longRunning:      true,
			toolResult:       nil,
			wantEventCount:   1,
			wantFREventIndex: -1,
		},
		{
			name:             "plain + nil: no gate, FR(nil) still emitted (no-op tool legacy)",
			toolResult:       nil,
			wantEventCount:   3, // FC + FR(nil) + text
			wantFREventIndex: 1,
		},
		{
			name:             "deferred + non-nil: FR event still emitted (gate is nil-only)",
			defers:           true,
			toolResult:       map[string]any{"answer": "ok"},
			wantEventCount:   3, // FC + FR + text
			wantFREventIndex: 1,
		},
		{
			name:             "long-running + non-nil (pending dict): FR event still emitted",
			longRunning:      true,
			toolResult:       map[string]any{"status": "pending"},
			wantEventCount:   3, // FC + FR(pending) + text — the canonical HITL pattern
			wantFREventIndex: 1,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			const toolName = "deferring"
			const fcID = "fc-x"

			dtool := &deferringTool{
				name:        toolName,
				description: "test deferring tool",
				result:      tc.toolResult,
				defers:      tc.defers,
				longRunning: tc.longRunning,
			}
			mockLLM := &singleCallMockModel{toolName: toolName, fcID: fcID}

			a, err := llmagent.New(llmagent.Config{
				Name:        "tester",
				Description: "tester",
				Model:       mockLLM,
				Tools:       []tool.Tool{dtool},
			})
			if err != nil {
				t.Fatal(err)
			}

			svc := session.InMemoryService()
			if _, err := svc.Create(t.Context(), &session.CreateRequest{
				AppName: "app", UserID: "u", SessionID: "s",
			}); err != nil {
				t.Fatal(err)
			}
			r, err := runner.New(runner.Config{Agent: a, SessionService: svc, AppName: "app"})
			if err != nil {
				t.Fatal(err)
			}

			events := []*session.Event{}
			for ev, err := range r.Run(t.Context(), "u", "s",
				&genai.Content{Parts: []*genai.Part{genai.NewPartFromText("go")}, Role: "user"},
				agent.RunConfig{StreamingMode: agent.StreamingModeSSE}) {
				if err != nil {
					t.Fatal(err)
				}
				events = append(events, ev)
			}

			if dtool.runCount != 1 {
				t.Errorf("tool.Run called %d times, want 1", dtool.runCount)
			}
			if len(events) != tc.wantEventCount {
				t.Fatalf("got %d events, want %d:\n%v",
					len(events), tc.wantEventCount, events)
			}
			if tc.wantFREventIndex >= 0 {
				fr := events[tc.wantFREventIndex]
				if fr.Content == nil || len(fr.Content.Parts) == 0 ||
					fr.Content.Parts[0].FunctionResponse == nil {
					t.Errorf("event[%d] should be a FunctionResponse, got:\n%v",
						tc.wantFREventIndex, events)
				}
			} else {
				// Assert NO event carries a FunctionResponse part.
				for i, ev := range events {
					if ev.Content == nil {
						continue
					}
					for _, p := range ev.Content.Parts {
						if p.FunctionResponse != nil {
							t.Errorf("event[%d] unexpectedly carries a FunctionResponse "+
								"(deferred tool's FR should have been skipped):\n%v",
								i, events)
						}
					}
				}
			}
		})
	}
}
