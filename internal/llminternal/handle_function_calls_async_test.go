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

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/internal/testutil"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
	"google.golang.org/adk/v2/tool/toolutils"
)

type SleepArgs struct {
	DurationMS int `json:"duration_ms"`
}
type SleepResult struct {
	Success bool `json:"success"`
}

func sleepFunc(ctx agent.Context, input SleepArgs) (SleepResult, error) {
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

func (t *deferringTool) Run(ctx agent.Context, args any) (map[string]any, error) {
	t.runCount++
	return t.result, nil
}

func (t *deferringTool) ProcessRequest(ctx agent.Context, req *model.LLMRequest) error {
	return toolutils.PackTool(req, t)
}

type singleCallMockModel struct {
	model.LLM
	toolName string
	fcID     string
	// args is passed as the function call's Args; a plain empty map is used
	// when unset, which is fine for tools that ignore their arguments.
	args  map[string]any
	calls int
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
		args := m.args
		if args == nil {
			args = map[string]any{}
		}
		yield(&model.LLMResponse{
			Content: &genai.Content{
				Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{
					ID:   m.fcID,
					Name: m.toolName,
					Args: args,
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
		wantResponse     map[string]any
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
			name:             "plain + nil: no gate, FR wraps nil as {result: nil}",
			toolResult:       nil,
			wantEventCount:   3, // FC + FR + text
			wantFREventIndex: 1,
			wantResponse:     map[string]any{"result": nil},
		},
		{
			name:             "deferred + non-nil: FR event still emitted (gate is nil-only)",
			defers:           true,
			toolResult:       map[string]any{"answer": "ok"},
			wantEventCount:   3, // FC + FR + text
			wantFREventIndex: 1,
			wantResponse:     map[string]any{"answer": "ok"},
		},
		{
			name:             "long-running + non-nil (pending dict): FR event still emitted",
			longRunning:      true,
			toolResult:       map[string]any{"status": "pending"},
			wantEventCount:   3, // FC + FR(pending) + text — the canonical HITL pattern
			wantFREventIndex: 1,
			wantResponse:     map[string]any{"status": "pending"},
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
					t.Fatalf("event[%d] should be a FunctionResponse, got:\n%v",
						tc.wantFREventIndex, events)
				}
				if diff := cmp.Diff(tc.wantResponse, fr.Content.Parts[0].FunctionResponse.Response); diff != "" {
					t.Errorf("FunctionResponse.Response mismatch (-want +got):\n%s", diff)
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

// noArgsCallModel calls toolName once with args (nil by default), then
// answers "done".
type noArgsCallModel struct {
	toolName string
	args     map[string]any
	calls    int
}

func (m *noArgsCallModel) Name() string                       { return "gemini-2.5-flash" }
func (m *noArgsCallModel) GetGoogleLLMVariant() genai.Backend { return genai.BackendGeminiAPI }

func (m *noArgsCallModel) GenerateContent(context.Context, *model.LLMRequest, bool) iter.Seq2[*model.LLMResponse, error] {
	m.calls++
	part := &genai.Part{FunctionCall: &genai.FunctionCall{ID: "fc-1", Name: m.toolName, Args: m.args}}
	if m.calls > 1 {
		part = genai.NewPartFromText("done")
	}
	return func(yield func(*model.LLMResponse, error) bool) {
		yield(&model.LLMResponse{Content: &genai.Content{Role: "model", Parts: []*genai.Part{part}}}, nil)
	}
}

// argsRecordingTool records the args it is run with and returns no result.
type argsRecordingTool struct {
	gotArgs any
}

func (t *argsRecordingTool) Name() string        { return "record" }
func (t *argsRecordingTool) Description() string { return "records its args" }
func (t *argsRecordingTool) IsLongRunning() bool { return false }

func (t *argsRecordingTool) Declaration() *genai.FunctionDeclaration {
	return &genai.FunctionDeclaration{Name: t.Name(), Description: t.Description()}
}

func (t *argsRecordingTool) Run(_ agent.Context, args any) (map[string]any, error) {
	t.gotArgs = args
	return nil, nil
}

func (t *argsRecordingTool) ProcessRequest(_ agent.Context, req *model.LLMRequest) error {
	return toolutils.PackTool(req, t)
}

// A function call without args reaches the tool and its callbacks as an empty
// map, so neither has to tell a nil map from an empty one.
func TestHandleFunctionCalls_NoArgsReachToolAsEmptyMap(t *testing.T) {
	rec := &argsRecordingTool{}
	var beforeArgs, afterArgs map[string]any
	a, err := llmagent.New(llmagent.Config{
		Name:  "tester",
		Model: &noArgsCallModel{toolName: rec.Name()},
		Tools: []tool.Tool{rec},
		BeforeToolCallbacks: []llmagent.BeforeToolCallback{
			func(_ agent.Context, _ tool.Tool, args map[string]any) (map[string]any, error) {
				beforeArgs = args
				return nil, nil
			},
		},
		AfterToolCallbacks: []llmagent.AfterToolCallback{
			func(_ agent.Context, _ tool.Tool, args, _ map[string]any, _ error) (map[string]any, error) {
				afterArgs = args
				return nil, nil
			},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	for _, err := range testutil.NewTestAgentRunner(t, a).Run(t, "s", "go") {
		if err != nil {
			t.Fatal(err)
		}
	}

	if diff := cmp.Diff(any(map[string]any{}), rec.gotArgs); diff != "" {
		t.Errorf("tool args mismatch (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff(map[string]any{}, beforeArgs); diff != "" {
		t.Errorf("before-tool callback args mismatch (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff(map[string]any{}, afterArgs); diff != "" {
		t.Errorf("after-tool callback args mismatch (-want +got):\n%s", diff)
	}
}

type argsRecordingStreamTool struct{ gotArgs any }

func (t *argsRecordingStreamTool) Name() string        { return "stream_record" }
func (t *argsRecordingStreamTool) Description() string { return "records its args" }
func (t *argsRecordingStreamTool) IsLongRunning() bool { return false }

func (t *argsRecordingStreamTool) Declaration() *genai.FunctionDeclaration {
	return &genai.FunctionDeclaration{Name: t.Name(), Description: t.Description()}
}

func (t *argsRecordingStreamTool) RunStream(_ agent.Context, args any) iter.Seq2[string, error] {
	t.gotArgs = args
	return func(yield func(string, error) bool) { yield("ok", nil) }
}

func (t *argsRecordingStreamTool) ProcessRequest(_ agent.Context, req *model.LLMRequest) error {
	return toolutils.PackTool(req, t)
}

func TestHandleFunctionCalls_NoArgsReachStreamingToolAsEmptyMap(t *testing.T) {
	st := &argsRecordingStreamTool{}
	a, err := llmagent.New(llmagent.Config{
		Name:  "tester",
		Model: &noArgsCallModel{toolName: st.Name()},
		Tools: []tool.Tool{st},
	})
	if err != nil {
		t.Fatal(err)
	}
	for _, err := range testutil.NewTestAgentRunner(t, a).Run(t, "s", "go") {
		if err != nil {
			t.Fatal(err)
		}
	}
	if diff := cmp.Diff(any(map[string]any{}), st.gotArgs); diff != "" {
		t.Errorf("streaming tool args mismatch (-want +got):\n%s", diff)
	}
}

func TestHandleFunctionCalls_NoArgsReachOnToolErrorForUnknownTool(t *testing.T) {
	var gotArgs map[string]any
	called := false
	a, err := llmagent.New(llmagent.Config{
		Name:  "tester",
		Model: &noArgsCallModel{toolName: "missing"},
		Tools: []tool.Tool{&argsRecordingTool{}},
		OnToolErrorCallbacks: []llmagent.OnToolErrorCallback{
			func(_ agent.Context, _ tool.Tool, args map[string]any, _ error) (map[string]any, error) {
				called, gotArgs = true, args
				return map[string]any{"handled": true}, nil
			},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	for _, err := range testutil.NewTestAgentRunner(t, a).Run(t, "s", "go") {
		if err != nil {
			t.Fatal(err)
		}
	}
	if !called {
		t.Fatal("on-tool-error callback not called")
	}
	if diff := cmp.Diff(map[string]any{}, gotArgs); diff != "" {
		t.Errorf("on-tool-error args mismatch (-want +got):\n%s", diff)
	}
}

// A set_model_response call without args gives the same structured output as a
// call with empty args, the empty object, rather than the {"result": null}
// sent for a tool that returns nothing.
func TestHandleFunctionCalls_SetModelResponseWithoutArgs(t *testing.T) {
	cases := []struct {
		name string
		args map[string]any
	}{
		{name: "nil args", args: nil},
		{name: "empty args", args: map[string]any{}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			a, err := llmagent.New(llmagent.Config{
				Name:      "tester",
				Model:     &noArgsCallModel{toolName: "set_model_response", args: tc.args},
				Tools:     []tool.Tool{&argsRecordingTool{}},
				OutputKey: "out",
				OutputSchema: &genai.Schema{
					Type:       genai.TypeObject,
					Properties: map[string]*genai.Schema{"note": {Type: genai.TypeString}},
				},
			})
			if err != nil {
				t.Fatal(err)
			}

			var lastText string
			var gotOut any
			for ev, err := range testutil.NewTestAgentRunner(t, a).Run(t, "s", "go") {
				if err != nil {
					t.Fatal(err)
				}
				if ev.Content != nil && len(ev.Content.Parts) > 0 && ev.Content.Parts[0].Text != "" {
					lastText = ev.Content.Parts[0].Text
				}
				if v, ok := ev.Actions.StateDelta["out"]; ok {
					gotOut = v
				}
			}

			if lastText != "{}" {
				t.Errorf("structured output = %q, want %q", lastText, "{}")
			}
			if gotOut != "{}" {
				t.Errorf("state[out] = %v, want %q", gotOut, "{}")
			}
		})
	}
}
