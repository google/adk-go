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

package llminternal

import (
	"context"
	"errors"
	"fmt"
	"iter"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/gorilla/websocket"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/codeexecutor"
	"google.golang.org/adk/v2/internal/agent/runconfig"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/internal/utils"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
)

// cfcModel serves live calls through client and records the unary calls it
// gets, which a CFC model call never makes.
type cfcModel struct {
	name   string
	client *genai.Client

	mu    sync.Mutex
	unary []*model.LLMRequest
}

func (m *cfcModel) Name() string { return m.name }

func (m *cfcModel) Client() *genai.Client { return m.client }

func (m *cfcModel) GenerateContent(_ context.Context, req *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		m.mu.Lock()
		m.unary = append(m.unary, req)
		m.mu.Unlock()
		yield(&model.LLMResponse{Content: genai.NewContentFromText("unary reply", genai.RoleModel)}, nil)
	}
}

func (m *cfcModel) unaryCalls() []*model.LLMRequest {
	m.mu.Lock()
	defer m.mu.Unlock()
	return append([]*model.LLMRequest(nil), m.unary...)
}

// unaryOnlyModel answers unary calls and has no live connection.
type unaryOnlyModel struct{}

func (unaryOnlyModel) Name() string { return "gemini-2.0-flash" }

func (unaryOnlyModel) GenerateContent(context.Context, *model.LLMRequest, bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		yield(&model.LLMResponse{Content: genai.NewContentFromText("unary reply", genai.RoleModel)}, nil)
	}
}

// cfcRun describes one run of a flow under test.
type cfcRun struct {
	agent       agent.Agent
	runConfig   agent.RunConfig
	maxLLMCalls int
	branch      string
}

// newCFCContext returns the invocation context of run, with a fresh session
// and the runner's run config in the Go context, as Runner.Run sets them up.
func newCFCContext(t *testing.T, run cfcRun) agent.InvocationContext {
	t.Helper()
	created, err := session.InMemoryService().Create(t.Context(), &session.CreateRequest{AppName: "app", UserID: "user"})
	if err != nil {
		t.Fatalf("session Create failed: %v", err)
	}
	rc := run.runConfig
	return icontext.NewInvocationContext(
		runconfig.ToContext(t.Context(), &runconfig.RunConfig{MaxLLMCalls: run.maxLLMCalls}),
		icontext.InvocationContextParams{Agent: run.agent, Session: created.Session, RunConfig: &rc, Branch: run.branch},
	)
}

func newCFCAgent(t *testing.T, name string, s *State) agent.Agent {
	t.Helper()
	base, err := agent.New(agent.Config{Name: name})
	if err != nil {
		t.Fatalf("agent.New failed: %v", err)
	}
	return &mockLLMAgent{Agent: base, s: s}
}

// collectEvents ranges seq to its end, or until the deadline, and returns its
// non-nil events and its errors.
func collectEvents(t *testing.T, seq iter.Seq2[*session.Event, error]) ([]*session.Event, []error) {
	t.Helper()
	type result struct {
		events []*session.Event
		errs   []error
	}
	done := make(chan result, 1)
	go func() {
		var r result
		for ev, err := range seq {
			if err != nil {
				r.errs = append(r.errs, err)
				continue
			}
			if ev != nil {
				r.events = append(r.events, ev)
			}
		}
		done <- r
	}()
	select {
	case r := <-done:
		return r.events, r.errs
	case <-time.After(10 * time.Second):
		t.Fatal("the run did not end within 10s")
		return nil, nil
	}
}

// describeCFCEvent names ev by its author and kind, with its partial and turn
// complete flags, so that an event sequence reads as one literal.
func describeCFCEvent(ev *session.Event) string {
	kind := "empty"
	switch {
	case ev.Content != nil && len(ev.Content.Parts) > 0 && ev.Content.Parts[0].FunctionCall != nil:
		kind = "call " + ev.Content.Parts[0].FunctionCall.Name
	case ev.Content != nil && len(ev.Content.Parts) > 0 && ev.Content.Parts[0].FunctionResponse != nil:
		kind = "response " + ev.Content.Parts[0].FunctionResponse.Name
	case ev.Content != nil && len(ev.Content.Parts) > 0:
		kind = "text " + ev.Content.Parts[0].Text
	case ev.OutputTranscription != nil:
		kind = "transcription " + ev.OutputTranscription.Text
	}
	s := ev.Author + " " + kind
	if ev.Partial {
		s += " (partial)"
	}
	if ev.TurnComplete {
		s += " (turn complete)"
	}
	return s
}

func describeCFCEvents(evs []*session.Event) []string {
	var out []string
	for _, ev := range evs {
		out = append(out, describeCFCEvent(ev))
	}
	return out
}

// writeFrames writes each frame to conn, reporting a failure on t.
func writeFrames(t *testing.T, conn *websocket.Conn, frames ...string) bool {
	for _, f := range frames {
		if err := conn.WriteMessage(websocket.TextMessage, []byte(f)); err != nil {
			t.Errorf("writing frame failed: %v", err)
			return false
		}
	}
	return true
}

// awaitFunctionResponse reads client frames until one carries a function
// response and returns it; ok is false when the client closed first.
func awaitFunctionResponse(conn *websocket.Conn) (frame string, ok bool) {
	for {
		_, data, err := conn.ReadMessage()
		if err != nil {
			return "", false
		}
		if strings.Contains(string(data), "functionResponse") {
			return string(data), true
		}
	}
}

const (
	addNumbersCall    = `{"toolCall":{"functionCalls":[{"id":"call_1","name":"add_numbers","args":{"a":3,"b":4}}]}}`
	cfcTextChunk      = `{"serverContent":{"modelTurn":{"parts":[{"text":"sum "}],"role":"model"}}}`
	cfcTranscription  = `{"serverContent":{"outputTranscription":{"text":"sum is 7"}}}`
	cfcTurnComplete   = `{"serverContent":{"modelTurn":{"parts":[{"text":"sum is 7"}],"role":"model"},"turnComplete":true}}`
	cfcTrailingTurn   = `{"serverContent":{"modelTurn":{"parts":[{"text":"trailing after turn_complete"}],"role":"model"},"turnComplete":true}}`
	cfcRawTurn        = `{"serverContent":{"modelTurn":{"parts":[{"text":"raw text"}],"role":"model"},"turnComplete":true}}`
	cfcOpenTurn       = `{"serverContent":{"modelTurn":{"parts":[{"text":"still talking"}],"role":"model"}}}`
	cfcOKTurn         = `{"serverContent":{"modelTurn":{"parts":[{"text":"ok"}],"role":"model"},"turnComplete":true}}`
	cfcDoneTurn       = `{"serverContent":{"modelTurn":{"parts":[{"text":"done"}],"role":"model"},"turnComplete":true}}`
	transferCall      = `{"toolCall":{"functionCalls":[{"id":"call_t","name":"transfer_to_agent","args":{"agent_name":"sub_agent"}}]}}`
	afterTransferTurn = `{"serverContent":{"modelTurn":{"parts":[{"text":"after transfer"}],"role":"model"},"turnComplete":true}}`
	taskCompletedCall = `{"toolCall":{"functionCalls":[{"id":"call_c","name":"task_completed","args":{}}]}}`
)

// liveOnlyProcessors is the smallest request processor list a live call
// needs.
var liveOnlyProcessors = []func(ctx agent.InvocationContext, req *model.LLMRequest, f *Flow) iter.Seq2[*session.Event, error]{liveConfigProcessor}

// A CFC model call runs over the live connection: before-model callbacks and
// the call count apply once, after-model callbacks apply to every live event,
// the live loop runs the tool once and sends its response back, the setup
// response is not yielded, a partial event is yielded only in SSE mode, every
// event carries the step's branch, and the call ends at the first turn
// complete event.
func TestCFCModelCall_LiveLifecycleToolsStreamingAndCallbacks(t *testing.T) {
	tests := []struct {
		name string
		mode agent.StreamingMode
		want []string
	}{
		{
			name: "no streaming drops the partial event",
			mode: agent.StreamingModeNone,
			want: []string{
				"cfc_agent call add_numbers",
				"cfc_agent response add_numbers",
				"cfc_agent text sum ",
				"cfc_agent transcription sum is 7",
				"cfc_agent text sum is 7 (turn complete)",
			},
		},
		{
			name: "SSE keeps the partial event",
			mode: agent.StreamingModeSSE,
			want: []string{
				"cfc_agent call add_numbers",
				"cfc_agent response add_numbers",
				"cfc_agent text sum ",
				"cfc_agent transcription sum is 7 (partial)",
				"cfc_agent transcription sum is 7",
				"cfc_agent text sum is 7 (turn complete)",
			},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			baseline, _ := runLiveStacks()
			defer assertNoRunLiveLeak(t, baseline)
			var mu sync.Mutex
			var responseFrames []string
			client, connCount := startFakeLiveServer(t, func(connNum int, conn *websocket.Conn) {
				if !writeFrames(t, conn, addNumbersCall) {
					return
				}
				frame, ok := awaitFunctionResponse(conn)
				if !ok {
					return
				}
				mu.Lock()
				responseFrames = append(responseFrames, frame)
				mu.Unlock()
				if !writeFrames(t, conn, cfcTextChunk, cfcTranscription, cfcTurnComplete, cfcTrailingTurn) {
					return
				}
				blockUntilClientCloses(conn)
			})
			var toolCalls, beforeCount, afterCount atomic.Int32
			addNumbers := &mockFunctionTool{name: "add_numbers", runFunc: func(agent.Context, map[string]any) (map[string]any, error) {
				toolCalls.Add(1)
				return map[string]any{"sum": 7}, nil
			}}
			m := &cfcModel{name: "gemini-2.0-flash", client: client}
			f := &Flow{
				Model:             m,
				Tools:             []tool.Tool{addNumbers},
				RequestProcessors: liveOnlyProcessors,
				BeforeModelCallbacks: []BeforeModelCallback{func(agent.Context, *model.LLMRequest) (*model.LLMResponse, error) {
					beforeCount.Add(1)
					return nil, nil
				}},
				AfterModelCallbacks: []AfterModelCallback{func(agent.Context, *model.LLMResponse, error) (*model.LLMResponse, error) {
					afterCount.Add(1)
					return nil, nil
				}},
			}
			ctx := newCFCContext(t, cfcRun{
				agent:       newCFCAgent(t, "cfc_agent", &State{}),
				runConfig:   agent.RunConfig{SupportCFC: true, StreamingMode: tc.mode},
				maxLLMCalls: 1,
				branch:      "main.sub",
			})

			events, errs := collectEvents(t, f.Run(ctx))

			if len(errs) != 0 {
				t.Fatalf("Run() errors = %v, want none", errs)
			}
			if diff := cmp.Diff(tc.want, describeCFCEvents(events)); diff != "" {
				t.Errorf("events mismatch (-want +got):\n%s", diff)
			}
			for _, ev := range events {
				if ev.Branch != "main.sub" {
					t.Errorf("event %q has Branch %q, want %q", describeCFCEvent(ev), ev.Branch, "main.sub")
				}
			}
			if got := beforeCount.Load(); got != 1 {
				t.Errorf("before-model callbacks ran %d times, want 1", got)
			}
			// Every live event the call received: the call, its response, the
			// text chunk, the partial transcription, the aggregated
			// transcription and the turn complete event.
			if got := afterCount.Load(); got != 6 {
				t.Errorf("after-model callbacks ran %d times, want 6", got)
			}
			if got := toolCalls.Load(); got != 1 {
				t.Errorf("add_numbers ran %d times, want 1", got)
			}
			if got := connCount.Load(); got != 1 {
				t.Errorf("live connections = %d, want 1", got)
			}
			if got := len(m.unaryCalls()); got != 0 {
				t.Errorf("unary model calls = %d, want 0", got)
			}
			mu.Lock()
			frames := append([]string(nil), responseFrames...)
			mu.Unlock()
			if len(frames) != 1 || !strings.Contains(frames[0], `"sum":7`) {
				t.Errorf("function response frames = %v, want one carrying \"sum\":7", frames)
			}
		})
	}
}

// describeControlEvent is describeCFCEvent with the control field named for
// an event that has no content, no output transcription and no turn
// complete flag.
func describeControlEvent(ev *session.Event) string {
	switch {
	case ev.Content != nil || ev.OutputTranscription != nil || ev.TurnComplete:
		return describeCFCEvent(ev)
	case ev.InputTranscription != nil:
		return ev.Author + " input transcription " + ev.InputTranscription.Text
	case ev.Interrupted:
		return ev.Author + " interrupted"
	case ev.SessionResumptionHandle != "":
		return ev.Author + " resumption handle " + ev.SessionResumptionHandle
	case ev.UsageMetadata != nil:
		return fmt.Sprintf("%s usage %d tokens", ev.Author, ev.UsageMetadata.TotalTokenCount)
	}
	return describeCFCEvent(ev)
}

// A CFC model call skips only a live response with no content and no
// control field (adk-python live/_live_llm_flow.py:408-419). A frame that
// carries one control field alone is yielded as an event, and a turn
// complete frame without model content ends the call, which is how the Live
// API ends a turn.
func TestCFCModelCall_KeepsControlOnlyLiveFrames(t *testing.T) {
	tests := []struct {
		name   string
		frames []string
		want   []string
	}{
		{
			name:   "turn complete alone ends the call",
			frames: []string{cfcTextChunk, `{"serverContent":{"turnComplete":true}}`},
			want:   []string{"cfc_agent text sum ", "cfc_agent empty (turn complete)"},
		},
		{
			name:   "interrupted",
			frames: []string{`{"serverContent":{"interrupted":true}}`, cfcTurnComplete},
			want:   []string{"cfc_agent interrupted", "cfc_agent text sum is 7 (turn complete)"},
		},
		{
			// The chunk is partial and so not yielded outside SSE; the
			// aggregated transcription the turn complete frame releases is
			// another response with only the input transcription.
			name:   "input transcription",
			frames: []string{`{"serverContent":{"inputTranscription":{"text":"add 3 and 4"}}}`, cfcTurnComplete},
			want:   []string{"cfc_agent input transcription add 3 and 4", "cfc_agent text sum is 7 (turn complete)"},
		},
		{
			name:   "session resumption handle",
			frames: []string{`{"sessionResumptionUpdate":{"newHandle":"handle-1","resumable":true}}`, cfcTurnComplete},
			want:   []string{"cfc_agent resumption handle handle-1", "cfc_agent text sum is 7 (turn complete)"},
		},
		{
			name:   "usage metadata",
			frames: []string{`{"usageMetadata":{"promptTokenCount":3,"responseTokenCount":2,"totalTokenCount":5}}`, cfcTurnComplete},
			want:   []string{"cfc_agent usage 5 tokens", "cfc_agent text sum is 7 (turn complete)"},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			client, connCount := startFakeLiveServer(t, func(_ int, conn *websocket.Conn) {
				if !writeFrames(t, conn, tc.frames...) {
					return
				}
				blockUntilClientCloses(conn)
			})
			f := &Flow{Model: &cfcModel{name: "gemini-2.0-flash", client: client}, RequestProcessors: liveOnlyProcessors}
			ctx := newCFCContext(t, cfcRun{
				agent:       newCFCAgent(t, "cfc_agent", &State{}),
				runConfig:   agent.RunConfig{SupportCFC: true},
				maxLLMCalls: 1,
			})

			events, errs := collectEvents(t, f.Run(ctx))

			if len(errs) != 0 {
				t.Fatalf("Run() errors = %v, want none", errs)
			}
			var got []string
			for _, ev := range events {
				got = append(got, describeControlEvent(ev))
			}
			if diff := cmp.Diff(tc.want, got); diff != "" {
				t.Errorf("events mismatch (-want +got):\n%s", diff)
			}
			if got := connCount.Load(); got != 1 {
				t.Errorf("live connections = %d, want 1", got)
			}
		})
	}
}

// LiveConnection.Recv sets neither an error code nor grounding metadata, but
// the response processors run before the skip and may set them; a response
// that carries only one of the two is not skipped, as in adk-python
// (live/_live_llm_flow.py:408-419).
func TestCFCEmptyLiveResponse_ErrorCodeOrGroundingMetadataAlone(t *testing.T) {
	tests := []struct {
		name string
		resp *model.LLMResponse
	}{
		{name: "error code", resp: &model.LLMResponse{ErrorCode: "RESOURCE_EXHAUSTED"}},
		{name: "grounding metadata", resp: &model.LLMResponse{GroundingMetadata: &genai.GroundingMetadata{WebSearchQueries: []string{"sum of 3 and 4"}}}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if isEmptyLiveResponse(tc.resp) {
				t.Errorf("isEmptyLiveResponse(%s alone) = true, want false", tc.name)
			}
		})
	}
}

// An agent without a built-in executor gets one for a CFC invocation only:
// the live connection carries the built-in code execution tool and no
// fenced-code instruction, the agent's own executor is not called, and a
// later invocation without CFC makes a unary call with the agent's own
// executor in force.
func TestCFCScopesBuiltInExecutorToInvocation_CustomExecutor(t *testing.T) {
	baseline, _ := runLiveStacks()
	defer assertNoRunLiveLeak(t, baseline)
	setupFrames := make(chan string, 4)
	client, connCount, _ := startLiveServerWithSetup(t, func(_ int, frame []byte) {
		setupFrames <- string(frame)
	}, func(_ int, conn *websocket.Conn) {
		if !writeFrames(t, conn, cfcOKTurn) {
			return
		}
		blockUntilClientCloses(conn)
	})
	exec := &fencedCountingExecutor{}
	m := &cfcModel{name: "gemini-2.0-flash", client: client}
	a := newCFCAgent(t, "test_agent", &State{Model: m, CodeExecutor: exec})
	newFlow := func() *Flow {
		return &Flow{Model: m, RequestProcessors: DefaultRequestProcessors, ResponseProcessors: DefaultResponseProcessors}
	}

	_, errs := collectEvents(t, newFlow().Run(newCFCContext(t, cfcRun{agent: a, runConfig: agent.RunConfig{SupportCFC: true}})))
	if len(errs) != 0 {
		t.Fatalf("CFC run errors = %v, want none", errs)
	}
	var setup string
	select {
	case setup = <-setupFrames:
	default:
		t.Fatal("the CFC run sent no setup frame")
	}
	if !strings.Contains(setup, `"tools":[{"codeExecution":{}}]`) {
		t.Errorf("setup frame = %s, want the tools [{\"codeExecution\":{}}]", setup)
	}
	if strings.Contains(setup, "CRITICAL: Code execution format") {
		t.Errorf("setup frame carries the fenced-code instruction, want none")
	}
	if got := len(m.unaryCalls()); got != 0 {
		t.Errorf("CFC run made %d unary calls, want 0", got)
	}

	_, errs = collectEvents(t, newFlow().Run(newCFCContext(t, cfcRun{agent: a})))
	if len(errs) != 0 {
		t.Fatalf("second run errors = %v, want none", errs)
	}
	if got := connCount.Load(); got != 1 {
		t.Errorf("live connections after both runs = %d, want 1", got)
	}
	unary := m.unaryCalls()
	if len(unary) != 1 {
		t.Fatalf("unary calls after both runs = %d, want 1", len(unary))
	}
	for _, tl := range unary[0].Config.Tools {
		if tl.CodeExecution != nil {
			t.Errorf("second run's request has a code execution tool, want none")
		}
	}
	si := unary[0].Config.SystemInstruction
	if si == nil || len(si.Parts) == 0 || !strings.Contains(si.Parts[0].Text, "# CRITICAL: Code execution format") {
		t.Errorf("second run's system instruction = %v, want the fenced-code instruction of the agent's executor", si)
	}
	if got := exec.calls.Load(); got != 0 {
		t.Errorf("agent's executor ran %d times, want 0", got)
	}
}

// fencedCountingExecutor is a non-built-in executor with OptimizeDataFile
// on, so that the request processor gives the model the fenced-code
// instruction. It counts its calls.
type fencedCountingExecutor struct{ calls atomic.Int32 }

func (*fencedCountingExecutor) Settings() codeexecutor.Settings {
	s := codeexecutor.DefaultSettings()
	s.OptimizeDataFile = true
	return s
}

func (e *fencedCountingExecutor) ExecuteCode(context.Context, agent.InvocationContext, *codeexecutor.Input) (*codeexecutor.Result, error) {
	e.calls.Add(1)
	return &codeexecutor.Result{Stdout: "ok"}, nil
}

func TestCFCModelCall_ErrorsAndEarlyConsumerStop(t *testing.T) {
	t.Run("model without a live connection", func(t *testing.T) {
		f := &Flow{Model: unaryOnlyModel{}, RequestProcessors: liveOnlyProcessors}
		ctx := newCFCContext(t, cfcRun{agent: newCFCAgent(t, "cfc_agent", &State{}), runConfig: agent.RunConfig{SupportCFC: true}})

		events, errs := collectEvents(t, f.Run(ctx))

		if len(events) != 0 {
			t.Errorf("Run() events = %v, want none", describeCFCEvents(events))
		}
		if len(errs) != 1 || errs[0].Error() != "model does not support live connection" {
			t.Errorf("Run() errors = %v, want [model does not support live connection]", errs)
		}
	})

	t.Run("model without a live connection, replaced by an on-model-error callback", func(t *testing.T) {
		var seen error
		f := &Flow{
			Model:             unaryOnlyModel{},
			RequestProcessors: liveOnlyProcessors,
			OnModelErrorCallbacks: []OnModelErrorCallback{func(_ agent.Context, _ *model.LLMRequest, err error) (*model.LLMResponse, error) {
				seen = err
				return &model.LLMResponse{Content: genai.NewContentFromText("fallback", genai.RoleModel)}, nil
			}},
		}
		ctx := newCFCContext(t, cfcRun{agent: newCFCAgent(t, "cfc_agent", &State{}), runConfig: agent.RunConfig{SupportCFC: true}})

		events, errs := collectEvents(t, f.Run(ctx))

		if len(errs) != 0 {
			t.Fatalf("Run() errors = %v, want none", errs)
		}
		if diff := cmp.Diff([]string{"cfc_agent text fallback"}, describeCFCEvents(events)); diff != "" {
			t.Errorf("events mismatch (-want +got):\n%s", diff)
		}
		if seen == nil || seen.Error() != "model does not support live connection" {
			t.Errorf("on-model-error callback got %v, want the live connection error", seen)
		}
	})

	t.Run("the call count is enforced before the next connection", func(t *testing.T) {
		baseline, _ := runLiveStacks()
		defer assertNoRunLiveLeak(t, baseline)
		client, connCount := startFakeLiveServer(t, func(_ int, conn *websocket.Conn) {
			if !writeFrames(t, conn, taskCompletedCall) {
				return
			}
			blockUntilClientCloses(conn)
		})
		taskCompleted := &mockFunctionTool{name: "task_completed", runFunc: func(agent.Context, map[string]any) (map[string]any, error) {
			return map[string]any{}, nil
		}}
		f := &Flow{Model: &cfcModel{name: "gemini-2.0-flash", client: client}, Tools: []tool.Tool{taskCompleted}, RequestProcessors: liveOnlyProcessors}
		ctx := newCFCContext(t, cfcRun{agent: newCFCAgent(t, "cfc_agent", &State{}), runConfig: agent.RunConfig{SupportCFC: true}, maxLLMCalls: 1})

		events, errs := collectEvents(t, f.Run(ctx))

		// task_completed ends the live call without a turn complete event, so
		// its function response is the step's last event and the flow takes a
		// second step, whose model call is over the budget.
		if diff := cmp.Diff([]string{"cfc_agent call task_completed", "cfc_agent response task_completed"}, describeCFCEvents(events)); diff != "" {
			t.Errorf("events mismatch (-want +got):\n%s", diff)
		}
		if len(errs) != 1 || !errors.Is(errs[0], agent.ErrLLMCallsLimitExceeded) {
			t.Errorf("Run() errors = %v, want one wrapping ErrLLMCallsLimitExceeded", errs)
		}
		if got := connCount.Load(); got != 1 {
			t.Errorf("live connections = %d, want 1", got)
		}
	})

	t.Run("an after-model callback error ends the call", func(t *testing.T) {
		baseline, _ := runLiveStacks()
		defer assertNoRunLiveLeak(t, baseline)
		client, _ := startFakeLiveServer(t, func(_ int, conn *websocket.Conn) {
			if !writeFrames(t, conn, cfcOpenTurn) {
				return
			}
			blockUntilClientCloses(conn)
		})
		f := &Flow{
			Model:             &cfcModel{name: "gemini-2.0-flash", client: client},
			RequestProcessors: liveOnlyProcessors,
			AfterModelCallbacks: []AfterModelCallback{func(agent.Context, *model.LLMResponse, error) (*model.LLMResponse, error) {
				return nil, errors.New("policy boom")
			}},
		}
		ctx := newCFCContext(t, cfcRun{agent: newCFCAgent(t, "cfc_agent", &State{}), runConfig: agent.RunConfig{SupportCFC: true}})

		events, errs := collectEvents(t, f.Run(ctx))

		if len(events) != 0 {
			t.Errorf("Run() events = %v, want none", describeCFCEvents(events))
		}
		if len(errs) != 1 || errs[0].Error() != "policy boom" {
			t.Errorf("Run() errors = %v, want [policy boom]", errs)
		}
	})

	t.Run("a consumer that stops early closes the connection", func(t *testing.T) {
		baseline, _ := runLiveStacks()
		defer assertNoRunLiveLeak(t, baseline)
		closed := make(chan struct{})
		client, _ := startFakeLiveServer(t, func(_ int, conn *websocket.Conn) {
			defer close(closed)
			if !writeFrames(t, conn, cfcOpenTurn) {
				return
			}
			blockUntilClientCloses(conn)
		})
		f := &Flow{Model: &cfcModel{name: "gemini-2.0-flash", client: client}, RequestProcessors: liveOnlyProcessors}
		ctx := newCFCContext(t, cfcRun{agent: newCFCAgent(t, "cfc_agent", &State{}), runConfig: agent.RunConfig{SupportCFC: true}})

		var first string
		for ev, err := range f.Run(ctx) {
			if err != nil {
				t.Fatalf("Run() error = %v", err)
			}
			first = describeCFCEvent(ev)
			break
		}

		if first != "cfc_agent text still talking" {
			t.Errorf("first event = %q, want %q", first, "cfc_agent text still talking")
		}
		select {
		case <-closed:
		case <-time.After(10 * time.Second):
			t.Error("the server never saw the connection close")
		}
	})
}

// An after-model callback's replacement of a live event keeps the event: its
// author, id, invocation, branch and turn complete flag, so the call ends.
func TestCFCAfterModelCallbackReplacementKeepsEventAndTurnComplete(t *testing.T) {
	baseline, _ := runLiveStacks()
	defer assertNoRunLiveLeak(t, baseline)
	client, _ := startFakeLiveServer(t, func(_ int, conn *websocket.Conn) {
		if !writeFrames(t, conn, cfcRawTurn) {
			return
		}
		blockUntilClientCloses(conn)
	})
	f := &Flow{
		Model:             &cfcModel{name: "gemini-2.0-flash", client: client},
		RequestProcessors: liveOnlyProcessors,
		AfterModelCallbacks: []AfterModelCallback{func(_ agent.Context, resp *model.LLMResponse, _ error) (*model.LLMResponse, error) {
			if resp.Content == nil {
				return nil, nil
			}
			return &model.LLMResponse{Content: genai.NewContentFromText("scrubbed", genai.RoleModel)}, nil
		}},
	}
	ctx := newCFCContext(t, cfcRun{agent: newCFCAgent(t, "test-live-agent", &State{}), runConfig: agent.RunConfig{SupportCFC: true}, branch: "main"})

	events, errs := collectEvents(t, f.Run(ctx))

	if len(errs) != 0 {
		t.Fatalf("Run() errors = %v, want none", errs)
	}
	if len(events) != 1 {
		t.Fatalf("Run() events = %v, want one", describeCFCEvents(events))
	}
	ev := events[0]
	if ev.Author != "test-live-agent" || ev.ID == "" || ev.InvocationID != ctx.InvocationID() || ev.Branch != "main" || !ev.TurnComplete {
		t.Errorf("event Author %q, ID %q, InvocationID %q, Branch %q, TurnComplete %v; want %q, non-empty, %q, %q, true",
			ev.Author, ev.ID, ev.InvocationID, ev.Branch, ev.TurnComplete, "test-live-agent", ctx.InvocationID(), "main")
	}
	if diff := cmp.Diff([]*genai.Part{{Text: "scrubbed"}}, ev.Content.Parts); diff != "" {
		t.Errorf("event parts mismatch (-want +got):\n%s", diff)
	}
}

// A CFC model event carries the live connection's session id. An after-model
// callback's replacement does not inherit it, as adk-python's
// _inherit_unset_model_fields fills only partial, turn_complete and
// usage_metadata (flows/llm_flows/core/_finalizer.py:212-216); a replacement
// that sets its own id keeps it.
func TestCFCModelEventLiveSessionID(t *testing.T) {
	replaceWith := func(liveSessionID string) AfterModelCallback {
		return func(_ agent.Context, resp *model.LLMResponse, _ error) (*model.LLMResponse, error) {
			if resp.Content == nil {
				return nil, nil
			}
			return &model.LLMResponse{Content: genai.NewContentFromText("scrubbed", genai.RoleModel), LiveSessionID: liveSessionID}, nil
		}
	}
	tests := []struct {
		name              string
		callbacks         []AfterModelCallback
		wantLiveSessionID string
	}{
		{name: "no callback", wantLiveSessionID: "cfc-live-1"},
		{name: "replacement without an id", callbacks: []AfterModelCallback{replaceWith("")}, wantLiveSessionID: ""},
		{name: "replacement with its own id", callbacks: []AfterModelCallback{replaceWith("from-callback")}, wantLiveSessionID: "from-callback"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			baseline, _ := runLiveStacks()
			defer assertNoRunLiveLeak(t, baseline)
			client := startLiveServerWithSetupComplete(t, func(int) string { return `{"setupComplete":{"sessionId":"cfc-live-1"}}` }, func(_ int, conn *websocket.Conn) {
				if !writeFrames(t, conn, cfcRawTurn) {
					return
				}
				blockUntilClientCloses(conn)
			})
			f := &Flow{
				Model:               &cfcModel{name: "gemini-2.0-flash", client: client},
				RequestProcessors:   liveOnlyProcessors,
				AfterModelCallbacks: tc.callbacks,
			}
			ctx := newCFCContext(t, cfcRun{agent: newCFCAgent(t, "test-live-agent", &State{}), runConfig: agent.RunConfig{SupportCFC: true}})

			events, errs := collectEvents(t, f.Run(ctx))

			if len(errs) != 0 {
				t.Fatalf("Run() errors = %v, want none", errs)
			}
			if len(events) != 1 {
				t.Fatalf("Run() events = %v, want one", describeCFCEvents(events))
			}
			if got := events[0].LiveSessionID; got != tc.wantLiveSessionID {
				t.Errorf("event LiveSessionID = %q, want %q", got, tc.wantLiveSessionID)
			}
			if !events[0].TurnComplete {
				t.Error("event TurnComplete = false, want true")
			}
		})
	}
}

// An after-model callback that edits a live function call edits the event
// the consumer gets, not the call the live loop runs, and does not race with
// it (run with -race).
func TestCFCAfterModelCallbackEditDoesNotReachLiveTool(t *testing.T) {
	baseline, _ := runLiveStacks()
	defer assertNoRunLiveLeak(t, baseline)
	client, _ := startFakeLiveServer(t, func(_ int, conn *websocket.Conn) {
		if !writeFrames(t, conn, addNumbersCall) {
			return
		}
		if _, ok := awaitFunctionResponse(conn); !ok {
			return
		}
		if !writeFrames(t, conn, cfcDoneTurn) {
			return
		}
		blockUntilClientCloses(conn)
	})
	var mu sync.Mutex
	var toolArgs []map[string]any
	addNumbers := &mockFunctionTool{name: "add_numbers", runFunc: func(_ agent.Context, args map[string]any) (map[string]any, error) {
		mu.Lock()
		toolArgs = append(toolArgs, args)
		mu.Unlock()
		return map[string]any{"sum": 7}, nil
	}}
	f := &Flow{
		Model:             &cfcModel{name: "gemini-2.0-flash", client: client},
		Tools:             []tool.Tool{addNumbers},
		RequestProcessors: liveOnlyProcessors,
		AfterModelCallbacks: []AfterModelCallback{func(_ agent.Context, resp *model.LLMResponse, _ error) (*model.LLMResponse, error) {
			if resp.Content == nil {
				return nil, nil
			}
			for _, p := range resp.Content.Parts {
				if p.FunctionCall != nil {
					p.FunctionCall.Args["a"] = 100
					p.FunctionCall.Name = "renamed"
				}
			}
			return nil, nil
		}},
	}
	ctx := newCFCContext(t, cfcRun{agent: newCFCAgent(t, "cfc_agent", &State{}), runConfig: agent.RunConfig{SupportCFC: true}})

	events, errs := collectEvents(t, f.Run(ctx))

	if len(errs) != 0 {
		t.Fatalf("Run() errors = %v, want none", errs)
	}
	want := []string{"cfc_agent call renamed", "cfc_agent response add_numbers", "cfc_agent text done (turn complete)"}
	if diff := cmp.Diff(want, describeCFCEvents(events)); diff != "" {
		t.Errorf("events mismatch (-want +got):\n%s", diff)
	}
	mu.Lock()
	got := append([]map[string]any(nil), toolArgs...)
	mu.Unlock()
	if diff := cmp.Diff([]map[string]any{{"a": float64(3), "b": float64(4)}}, got); diff != "" {
		t.Errorf("tool args mismatch (-want +got):\n%s", diff)
	}
}

// A transfer that the model makes during a CFC model call runs its target
// once the call has ended: the transferring agent's live connection is
// closed first and its next live event is never yielded, the target answers
// over a connection of its own, and the run ends with the target's answer.
func TestCFCTransferRunsTarget(t *testing.T) {
	baseline, _ := runLiveStacks()
	defer assertNoRunLiveLeak(t, baseline)
	rootClosed := make(chan struct{})
	client, connCount := startFakeLiveServer(t, func(connNum int, conn *websocket.Conn) {
		if connNum > 1 {
			select {
			case <-rootClosed:
			case <-time.After(2 * time.Second):
				t.Error("live connection 2 opened while the transferring agent's connection was still open")
			}
			if !writeFrames(t, conn, `{"serverContent":{"modelTurn":{"parts":[{"text":"from sub"}],"role":"model"},"turnComplete":true}}`) {
				return
			}
			blockUntilClientCloses(conn)
			return
		}
		defer close(rootClosed)
		if !writeFrames(t, conn, transferCall) {
			return
		}
		// The client may close before it sends the function response.
		if _, ok := awaitFunctionResponse(conn); !ok {
			return
		}
		if !writeFrames(t, conn, afterTransferTurn) {
			return
		}
		blockUntilClientCloses(conn)
	})
	sub, err := agent.New(agent.Config{
		Name: "sub_agent",
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return (&Flow{Model: &cfcModel{name: "gemini-2.0-flash", client: client}, RequestProcessors: liveOnlyProcessors}).Run(ctx)
		},
	})
	if err != nil {
		t.Fatalf("agent.New(sub_agent) failed: %v", err)
	}
	rootBase, err := agent.New(agent.Config{Name: "cfc_agent", SubAgents: []agent.Agent{sub}})
	if err != nil {
		t.Fatalf("agent.New(cfc_agent) failed: %v", err)
	}
	transfer := &mockFunctionTool{name: "transfer_to_agent", runFunc: func(ctx agent.Context, _ map[string]any) (map[string]any, error) {
		ctx.Actions().TransferToAgent = "sub_agent"
		return map[string]any{}, nil
	}}
	f := &Flow{Model: &cfcModel{name: "gemini-2.0-flash", client: client}, Tools: []tool.Tool{transfer}, RequestProcessors: liveOnlyProcessors}
	ctx := newCFCContext(t, cfcRun{agent: &mockLLMAgent{Agent: rootBase, s: &State{}}, runConfig: agent.RunConfig{SupportCFC: true}})

	events, errs := collectEvents(t, f.Run(ctx))

	if len(errs) != 0 {
		t.Fatalf("Run() errors = %v, want none", errs)
	}
	want := []string{"cfc_agent call transfer_to_agent", "cfc_agent response transfer_to_agent", "sub_agent text from sub (turn complete)"}
	if diff := cmp.Diff(want, describeCFCEvents(events)); diff != "" {
		t.Errorf("events mismatch (-want +got):\n%s", diff)
	}
	if got := connCount.Load(); got != 2 {
		t.Errorf("live connections = %d, want 2", got)
	}
}

// The target of a transfer made during a CFC model call answers through its
// own after-model callbacks only: the transferring agent's callbacks, which
// replace every one of its own responses, do not see the target's answer.
// adk-python runs the target inside the root's live call, so the root's
// callbacks rewrite it (flows/llm_flows/core/_model_call.py); Go does not
// (ledger: Fixed bug, ruling R17).
func TestCFCTransferTargetSkipsTransferringAgentAfterModelCallbacks(t *testing.T) {
	baseline, _ := runLiveStacks()
	defer assertNoRunLiveLeak(t, baseline)
	client, _ := startFakeLiveServer(t, func(connNum int, conn *websocket.Conn) {
		if connNum > 1 {
			if !writeFrames(t, conn, `{"serverContent":{"modelTurn":{"parts":[{"text":"from sub"}],"role":"model"},"turnComplete":true}}`) {
				return
			}
			blockUntilClientCloses(conn)
			return
		}
		if !writeFrames(t, conn, transferCall) {
			return
		}
		blockUntilClientCloses(conn)
	})
	subCallback := func(_ agent.Context, resp *model.LLMResponse, _ error) (*model.LLMResponse, error) {
		if resp.Content == nil || len(resp.Content.Parts) == 0 {
			return nil, nil
		}
		return &model.LLMResponse{Content: genai.NewContentFromText(resp.Content.Parts[0].Text+" (checked by sub)", genai.RoleModel)}, nil
	}
	sub, err := agent.New(agent.Config{
		Name: "sub_agent",
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return (&Flow{
				Model:               &cfcModel{name: "gemini-2.0-flash", client: client},
				RequestProcessors:   liveOnlyProcessors,
				AfterModelCallbacks: []AfterModelCallback{subCallback},
			}).Run(ctx)
		},
	})
	if err != nil {
		t.Fatalf("agent.New(sub_agent) failed: %v", err)
	}
	rootBase, err := agent.New(agent.Config{Name: "cfc_agent", SubAgents: []agent.Agent{sub}})
	if err != nil {
		t.Fatalf("agent.New(cfc_agent) failed: %v", err)
	}
	transfer := &mockFunctionTool{name: "transfer_to_agent", runFunc: func(ctx agent.Context, _ map[string]any) (map[string]any, error) {
		ctx.Actions().TransferToAgent = "sub_agent"
		return map[string]any{}, nil
	}}
	f := &Flow{
		Model:             &cfcModel{name: "gemini-2.0-flash", client: client},
		Tools:             []tool.Tool{transfer},
		RequestProcessors: liveOnlyProcessors,
		AfterModelCallbacks: []AfterModelCallback{func(_ agent.Context, resp *model.LLMResponse, _ error) (*model.LLMResponse, error) {
			if resp.Content == nil {
				return nil, nil
			}
			return &model.LLMResponse{Content: genai.NewContentFromText("rewritten by root", genai.RoleModel)}, nil
		}},
	}
	ctx := newCFCContext(t, cfcRun{agent: &mockLLMAgent{Agent: rootBase, s: &State{}}, runConfig: agent.RunConfig{SupportCFC: true}})

	events, errs := collectEvents(t, f.Run(ctx))

	if len(errs) != 0 {
		t.Fatalf("Run() errors = %v, want none", errs)
	}
	want := []string{
		"cfc_agent text rewritten by root",
		"cfc_agent text rewritten by root",
		"sub_agent text from sub (checked by sub) (turn complete)",
	}
	if diff := cmp.Diff(want, describeCFCEvents(events)); diff != "" {
		t.Errorf("events mismatch (-want +got):\n%s", diff)
	}
}

// A request processor's final-looking event without a function response,
// such as the data-file code event, does not end the step: the model is
// still called.
func TestRunOneStep_PreprocessCodeEventDoesNotSkipModelCall(t *testing.T) {
	codeEventProcessor := func(ctx agent.InvocationContext, req *model.LLMRequest, f *Flow) iter.Seq2[*session.Event, error] {
		return func(yield func(*session.Event, error) bool) {
			ev := session.NewEvent(ctx, ctx.InvocationID())
			ev.Author = ctx.Agent().Name()
			ev.Content = &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{
				{Text: "Processing input file: `data_1_1.csv`"},
				genai.NewPartFromExecutableCode("x = 1", genai.LanguagePython),
			}}
			yield(ev, nil)
		}
	}
	m := &cfcModel{name: "gemini-2.0-flash"}
	f := &Flow{
		Model: m,
		RequestProcessors: []func(ctx agent.InvocationContext, req *model.LLMRequest, f *Flow) iter.Seq2[*session.Event, error]{
			liveConfigProcessor, codeEventProcessor,
		},
	}
	ctx := newCFCContext(t, cfcRun{agent: newCFCAgent(t, "root_agent", &State{})})

	events, errs := collectEvents(t, f.runOneStep(ctx))

	if len(errs) != 0 {
		t.Fatalf("runOneStep() errors = %v, want none", errs)
	}
	want := []string{"root_agent text Processing input file: `data_1_1.csv`", "root_agent text unary reply"}
	if diff := cmp.Diff(want, describeCFCEvents(events)); diff != "" {
		t.Errorf("events mismatch (-want +got):\n%s", diff)
	}
	if got := len(m.unaryCalls()); got != 1 {
		t.Errorf("model calls = %d, want 1", got)
	}
}

// When an after-model callback fails on a CFC function call event, or when
// the consumer stops ranging at that event, RunLive must not execute the tool.
func TestCFCDoesNotRunToolBeforeConsumerAcceptsCallEvent(t *testing.T) {
	t.Run("after_model_callback_error", func(t *testing.T) {
		baseline, _ := runLiveStacks()
		defer assertNoRunLiveLeak(t, baseline)
		client, _ := startFakeLiveServer(t, func(_ int, conn *websocket.Conn) {
			if !writeFrames(t, conn, addNumbersCall) {
				return
			}
			blockUntilClientCloses(conn)
		})
		var toolRuns atomic.Int32
		addNumbers := &mockFunctionTool{name: "add_numbers", runFunc: func(_ agent.Context, _ map[string]any) (map[string]any, error) {
			toolRuns.Add(1)
			return map[string]any{"sum": 7}, nil
		}}
		wantErr := errors.New("rejected by after-model callback")
		f := &Flow{
			Model:             &cfcModel{name: "gemini-2.0-flash", client: client},
			Tools:             []tool.Tool{addNumbers},
			RequestProcessors: liveOnlyProcessors,
			AfterModelCallbacks: []AfterModelCallback{func(_ agent.Context, resp *model.LLMResponse, _ error) (*model.LLMResponse, error) {
				if len(utils.FunctionCalls(resp.Content)) > 0 {
					time.Sleep(15 * time.Millisecond)
					return nil, wantErr
				}
				return nil, nil
			}},
		}
		ctx := newCFCContext(t, cfcRun{agent: newCFCAgent(t, "cfc_agent", &State{}), runConfig: agent.RunConfig{SupportCFC: true}})

		_, errs := collectEvents(t, f.Run(ctx))

		if len(errs) != 1 || !errors.Is(errs[0], wantErr) {
			t.Fatalf("Run() errors = %v, want [%v]", errs, wantErr)
		}
		if got := toolRuns.Load(); got != 0 {
			t.Errorf("tool runs = %d, want 0 when AfterModelCallback rejects the call", got)
		}
	})

	t.Run("consumer_waits_then_stops_at_call_event", func(t *testing.T) {
		baseline, _ := runLiveStacks()
		defer assertNoRunLiveLeak(t, baseline)
		client, _ := startFakeLiveServer(t, func(_ int, conn *websocket.Conn) {
			if !writeFrames(t, conn, addNumbersCall) {
				return
			}
			blockUntilClientCloses(conn)
		})
		var toolRuns atomic.Int32
		addNumbers := &mockFunctionTool{name: "add_numbers", runFunc: func(_ agent.Context, _ map[string]any) (map[string]any, error) {
			toolRuns.Add(1)
			return map[string]any{"sum": 7}, nil
		}}
		f := &Flow{
			Model:             &cfcModel{name: "gemini-2.0-flash", client: client},
			Tools:             []tool.Tool{addNumbers},
			RequestProcessors: liveOnlyProcessors,
		}
		ctx := newCFCContext(t, cfcRun{agent: newCFCAgent(t, "cfc_agent", &State{}), runConfig: agent.RunConfig{SupportCFC: true}})

		sawCall := false
		for ev, err := range f.Run(ctx) {
			if err != nil {
				t.Fatalf("Run() error = %v", err)
			}
			if ev != nil && len(utils.FunctionCalls(ev.LLMResponse.Content)) > 0 {
				sawCall = true
				time.Sleep(15 * time.Millisecond)
				if got := toolRuns.Load(); got != 0 {
					t.Errorf("tool runs while consumer is handling call event = %d, want 0", got)
				}
				break
			}
		}
		if !sawCall {
			t.Fatal("Run() ended without yielding the function call event")
		}
		if got := toolRuns.Load(); got != 0 {
			t.Errorf("tool runs after consumer stopped at call event = %d, want 0", got)
		}
	})
}
