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

package llmagent_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"iter"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"
	"gorm.io/gorm"
	"gorm.io/gorm/logger"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/agent/workflowagents/parallelagent"
	"google.golang.org/adk/v2/agent/workflowagents/sequentialagent"
	"google.golang.org/adk/v2/artifact"
	"google.golang.org/adk/v2/codeexecutor"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/platform"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/session/database"
)

var (
	errExecutorScriptExhausted = errors.New("flow executor: no step left")
	errBarrierTimeout          = errors.New("barrier: timed out")
)

// The bookkeeping keys of the code execution flow, as adk-python names them
// (code_executors/code_executor_context.py).
const (
	codeExecutionContextKey = "_code_execution_context"
	inputFilesKey           = "_code_executor_input_files"
	errorCountsKey          = "_code_executor_error_counts"
	executionResultsKey     = "_code_execution_results"
)

// flowStep is one scripted outcome of flowExecutor. When run is set, it
// replaces res and err.
type flowStep struct {
	res *codeexecutor.Result
	err error
	run func(ctx context.Context) (*codeexecutor.Result, error)
}

// flowCall is what flowExecutor saw at one ExecuteCode call.
type flowCall struct {
	code          string
	inputFiles    []*codeexecutor.File
	executionID   string
	invocationID  string
	sessionEvents int
	// stateJSON holds the JSON of each bookkeeping key present in the
	// session state at the call.
	stateJSON map[string]string
}

// flowExecutor is a codeexecutor.CodeExecutor that is safe for concurrent use.
// It returns its steps in order, records every call, and fails once the steps
// run out, so a surplus call is visible.
type flowExecutor struct {
	settings codeexecutor.Settings

	mu    sync.Mutex
	steps []flowStep
	calls []flowCall
}

func newFlowExecutor(settings codeexecutor.Settings, steps ...flowStep) *flowExecutor {
	return &flowExecutor{settings: settings, steps: steps}
}

func (e *flowExecutor) Settings() codeexecutor.Settings { return e.settings }

func (e *flowExecutor) ExecuteCode(ctx context.Context, ic agent.InvocationContext, in *codeexecutor.Input) (*codeexecutor.Result, error) {
	call := flowCall{
		code:          in.Code,
		inputFiles:    in.InputFiles,
		executionID:   in.ExecutionID,
		invocationID:  ic.InvocationID(),
		sessionEvents: ic.Session().Events().Len(),
		stateJSON:     map[string]string{},
	}
	for _, key := range []string{codeExecutionContextKey, inputFilesKey, errorCountsKey, executionResultsKey} {
		v, err := ic.Session().State().Get(key)
		if err != nil {
			continue
		}
		b, err := json.Marshal(v)
		if err != nil {
			return nil, fmt.Errorf("marshal state key %s: %w", key, err)
		}
		call.stateJSON[key] = string(b)
	}
	e.mu.Lock()
	e.calls = append(e.calls, call)
	if len(e.steps) == 0 {
		e.mu.Unlock()
		return nil, errExecutorScriptExhausted
	}
	step := e.steps[0]
	e.steps = e.steps[1:]
	e.mu.Unlock()
	if step.run != nil {
		return step.run(ctx)
	}
	return step.res, step.err
}

func (e *flowExecutor) recorded() []flowCall {
	e.mu.Lock()
	defer e.mu.Unlock()
	return slices.Clone(e.calls)
}

// savedArtifact is one Save that reached flowArtifacts.
type savedArtifact struct {
	name     string
	data     string
	mimeType string
}

// flowArtifacts is an in-memory artifact.Service that records every Save and
// fails the Save of failName with failErr.
type flowArtifacts struct {
	artifact.Service
	failName string
	failErr  error

	mu    sync.Mutex
	saves []savedArtifact
}

func newFlowArtifacts() *flowArtifacts {
	return &flowArtifacts{Service: artifact.InMemoryService()}
}

func (a *flowArtifacts) Save(ctx context.Context, req *artifact.SaveRequest) (*artifact.SaveResponse, error) {
	if a.failName != "" && req.FileName == a.failName {
		return nil, a.failErr
	}
	resp, err := a.Service.Save(ctx, req)
	if err != nil {
		return nil, err
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	saved := savedArtifact{name: req.FileName}
	if req.Part.InlineData != nil {
		saved.data = string(req.Part.InlineData.Data)
		saved.mimeType = req.Part.InlineData.MIMEType
	}
	a.saves = append(a.saves, saved)
	return resp, nil
}

func (a *flowArtifacts) recorded() []savedArtifact {
	a.mu.Lock()
	defer a.mu.Unlock()
	return slices.Clone(a.saves)
}

// flowOpts changes how runFlow runs an agent.
type flowOpts struct {
	// noArtifacts runs without an artifact service.
	noArtifacts bool
	// artifacts replaces the default flowArtifacts.
	artifacts *flowArtifacts
	// sessions replaces session.InMemoryService().
	sessions session.Service
	// state is the state session s1 is created with.
	state map[string]any
	// history is appended to session s1 before the run.
	history []*session.Event
	// ctx replaces t.Context() as the run's context.
	ctx context.Context
}

type flowRun struct {
	events    []*session.Event
	err       error
	sess      session.Session
	artifacts *flowArtifacts
}

// runFlow runs a with the user message "go" in session s1 and collects the
// events until the first error.
func runFlow(t *testing.T, a agent.Agent, opts flowOpts) flowRun {
	t.Helper()
	ctx := opts.ctx
	if ctx == nil {
		ctx = t.Context()
	}
	svc := opts.sessions
	if svc == nil {
		svc = session.InMemoryService()
	}
	key := session.GetRequest{AppName: "test_app", UserID: "test_user", SessionID: "s1"}
	created, err := svc.Create(t.Context(), &session.CreateRequest{AppName: key.AppName, UserID: key.UserID, SessionID: key.SessionID, State: opts.state})
	if err != nil {
		t.Fatalf("session Create: %v", err)
	}
	for _, ev := range opts.history {
		if err := svc.AppendEvent(t.Context(), created.Session, ev); err != nil {
			t.Fatalf("session AppendEvent: %v", err)
		}
	}
	cfg := runner.Config{AppName: key.AppName, Agent: a, SessionService: svc}
	run := flowRun{artifacts: opts.artifacts}
	if !opts.noArtifacts {
		if run.artifacts == nil {
			run.artifacts = newFlowArtifacts()
		}
		cfg.ArtifactService = run.artifacts
	}
	r, err := runner.New(cfg)
	if err != nil {
		t.Fatalf("runner.New: %v", err)
	}
	for ev, err := range r.Run(ctx, key.UserID, key.SessionID, genai.NewContentFromText("go", genai.RoleUser), agent.RunConfig{}) {
		if err != nil {
			run.err = err
			break
		}
		run.events = append(run.events, ev)
	}
	got, err := svc.Get(t.Context(), &key)
	if err != nil {
		t.Fatalf("session Get: %v", err)
	}
	run.sess = got.Session
	return run
}

func newFlowAgent(t *testing.T, llm model.LLM, exec codeexecutor.CodeExecutor) agent.Agent {
	t.Helper()
	return newPlannerAgent(t, llmagent.Config{Name: "test_agent", Model: llm, CodeExecutor: exec})
}

func partsReply(parts ...*genai.Part) *model.LLMResponse {
	return &model.LLMResponse{
		Content:      &genai.Content{Role: genai.RoleModel, Parts: parts},
		FinishReason: genai.FinishReasonStop,
	}
}

func execPart(code string) *genai.Part {
	return genai.NewPartFromExecutableCode(code, genai.LanguagePython)
}

// roleText is a content as its role and the texts of its parts.
type roleText struct {
	Role  string
	Texts []string
}

func roleTexts(contents []*genai.Content) []roleText {
	var out []roleText
	for _, c := range contents {
		rt := roleText{Role: c.Role}
		for _, p := range c.Parts {
			rt.Texts = append(rt.Texts, p.Text)
		}
		out = append(out, rt)
	}
	return out
}

func eventParts(t *testing.T, ev *session.Event) []*genai.Part {
	t.Helper()
	if ev.LLMResponse.Content == nil {
		t.Fatalf("event has no content")
	}
	return ev.LLMResponse.Content.Parts
}

func resultPart(t *testing.T, ev *session.Event) *genai.CodeExecutionResult {
	t.Helper()
	parts := eventParts(t, ev)
	if len(parts) != 1 || parts[0].CodeExecutionResult == nil {
		t.Fatalf("event parts = %+v, want one code execution result part", parts)
	}
	return parts[0].CodeExecutionResult
}

func TestLLMAgent_CodeExecutionEmitsCodeResultThenAnswerEvents(t *testing.T) {
	llm := &scriptedModel{responses: []*model.LLMResponse{
		textResponse("```python\nprint(6 * 7)\n```"),
		textResponse("The answer is 42."),
	}}
	exec := newFlowExecutor(codeexecutor.DefaultSettings(), flowStep{res: &codeexecutor.Result{Stdout: "42\n"}})

	run := runFlow(t, newFlowAgent(t, llm, exec), flowOpts{})

	if run.err != nil {
		t.Fatalf("run error: %v", run.err)
	}
	if len(run.events) != 3 {
		t.Fatalf("run yielded %d events, want 3", len(run.events))
	}
	codeEv, resultEv, answerEv := run.events[0], run.events[1], run.events[2]
	if codeEv.Author != "test_agent" {
		t.Errorf("code event Author = %q, want %q", codeEv.Author, "test_agent")
	}
	if got := codeEv.LLMResponse.Content.Role; got != genai.RoleModel {
		t.Errorf("code event Role = %q, want %q", got, genai.RoleModel)
	}
	if diff := cmp.Diff([]*genai.Part{execPart("print(6 * 7)")}, eventParts(t, codeEv)); diff != "" {
		t.Errorf("code event parts mismatch (-want +got):\n%s", diff)
	}
	if len(codeEv.Actions.StateDelta) != 0 || len(codeEv.Actions.ArtifactDelta) != 0 {
		t.Errorf("code event deltas = %v, %v, want empty", codeEv.Actions.StateDelta, codeEv.Actions.ArtifactDelta)
	}
	if got := resultEv.LLMResponse.Content.Role; got != genai.RoleModel {
		t.Errorf("result event Role = %q, want %q", got, genai.RoleModel)
	}
	resultPart(t, resultEv)
	if diff := cmp.Diff(map[string]any{codeExecutionContextKey: map[string]any{}}, resultEv.Actions.StateDelta); diff != "" {
		t.Errorf("result event StateDelta mismatch (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff([]*genai.Part{{Text: "The answer is 42."}}, eventParts(t, answerEv)); diff != "" {
		t.Errorf("answer event parts mismatch (-want +got):\n%s", diff)
	}
	if id := codeEv.InvocationID; id == "" || resultEv.InvocationID != id || answerEv.InvocationID != id {
		t.Errorf("InvocationIDs = %q, %q, %q, want equal and non-empty", codeEv.InvocationID, resultEv.InvocationID, answerEv.InvocationID)
	}
	calls := exec.recorded()
	if len(calls) != 1 {
		t.Fatalf("executor calls = %d, want 1", len(calls))
	}
	if calls[0].code != "print(6 * 7)" {
		t.Errorf("executor Code = %q, want %q", calls[0].code, "print(6 * 7)")
	}
	// The user event and the code event are stored before the code runs.
	if calls[0].sessionEvents != 2 {
		t.Errorf("session events when the executor ran = %d, want 2", calls[0].sessionEvents)
	}
	if got := len(llm.recorded()); got != 2 {
		t.Errorf("model calls = %d, want 2", got)
	}
}

// The code and result events carry the agent's branch, so the history of a
// sibling branch leaves them out.
func TestLLMAgent_CodeExecutionEventsCarryBranch(t *testing.T) {
	llm := &scriptedModel{responses: []*model.LLMResponse{textResponse("```python\na_code()\n```"), textResponse("a done")}}
	exec := newFlowExecutor(codeexecutor.DefaultSettings(), flowStep{res: &codeexecutor.Result{Stdout: "1"}})
	parallel, err := parallelagent.New(parallelagent.Config{AgentConfig: agent.Config{
		Name:      "par",
		SubAgents: []agent.Agent{newPlannerAgent(t, llmagent.Config{Name: "a", Model: llm, CodeExecutor: exec})},
	}})
	if err != nil {
		t.Fatalf("parallelagent.New: %v", err)
	}

	run := runFlow(t, parallel, flowOpts{})

	if run.err != nil {
		t.Fatalf("run error: %v", run.err)
	}
	if len(run.events) != 3 {
		t.Fatalf("run yielded %d events, want 3", len(run.events))
	}
	codeEv, resultEv := run.events[0], run.events[1]
	resultPart(t, resultEv)
	if codeEv.Branch != "par.a" || resultEv.Branch != "par.a" {
		t.Errorf("code and result event Branch = %q, %q, want %q", codeEv.Branch, resultEv.Branch, "par.a")
	}
}

func TestLLMAgent_CodeExecutionExtractsCode(t *testing.T) {
	tests := []struct {
		name      string
		settings  codeexecutor.Settings
		reply     []*genai.Part
		wantCode  string // "" when nothing runs
		wantParts []*genai.Part
	}{
		{
			name:      "prefix_text_kept_and_suffix_dropped",
			reply:     []*genai.Part{{Text: "I will compute.\n```python\nx = 1\n```\nDone."}},
			wantCode:  "x = 1",
			wantParts: []*genai.Part{{Text: "I will compute.\n"}, execPart("x = 1")},
		},
		{
			name:      "prefix_part_copies_first_non_empty_text_part",
			reply:     []*genai.Part{{Text: ""}, {Text: "I will compute.\n```python\nx = 1\n```\nDone.", ThoughtSignature: []byte("sig")}},
			wantCode:  "x = 1",
			wantParts: []*genai.Part{{Text: "I will compute.\n", ThoughtSignature: []byte("sig")}, execPart("x = 1")},
		},
		{
			name:      "earliest_block_wins_over_first_pair",
			reply:     []*genai.Part{{Text: "A\n```python\nfirst()\n```\nB\n```tool_code\nsecond()\n```"}},
			wantCode:  "first()",
			wantParts: []*genai.Part{{Text: "A\n"}, execPart("first()")},
		},
		{
			name:      "first_pair_wins_a_tie",
			settings:  codeexecutor.Settings{CodeBlockDelimiters: [][2]string{{"```", "```"}, {"```python\n", "\n```"}}},
			reply:     []*genai.Part{{Text: "```python\nprint(1)\n```"}},
			wantCode:  "python\nprint(1)\n",
			wantParts: []*genai.Part{execPart("python\nprint(1)\n")},
		},
		{
			name:      "first_pair_wins_a_tie_with_other_closing",
			settings:  codeexecutor.Settings{CodeBlockDelimiters: [][2]string{{"```", "```"}, {"```", "</end>"}}},
			reply:     []*genai.Part{{Text: "prefix ```x = 1```</end>"}},
			wantCode:  "x = 1",
			wantParts: []*genai.Part{{Text: "prefix "}, execPart("x = 1")},
		},
		{
			name:      "pair_without_closing_is_skipped",
			settings:  codeexecutor.Settings{CodeBlockDelimiters: [][2]string{{"<a>", "</a>"}, {"<b>", "</b>"}}},
			reply:     []*genai.Part{{Text: "<a>x <b>y</b>"}},
			wantCode:  "y",
			wantParts: []*genai.Part{{Text: "<a>x "}, execPart("y")},
		},
		{
			name:      "text_parts_joined_with_newline",
			reply:     []*genai.Part{{Text: "```python\nx"}, {Text: "y\n```"}},
			wantCode:  "x\ny",
			wantParts: []*genai.Part{execPart("x\ny")},
		},
		{
			name:      "executable_code_part_keeps_parts_before_it",
			reply:     []*genai.Part{{Text: "hi"}, execPart("print(2)"), {Text: "later"}},
			wantCode:  "print(2)",
			wantParts: []*genai.Part{{Text: "hi"}, execPart("print(2)")},
		},
		{
			name: "executable_code_followed_by_result_is_skipped",
			reply: []*genai.Part{
				execPart("old()"),
				genai.NewPartFromCodeExecutionResult(genai.OutcomeOK, "old"),
				{Text: "```python\nnew()\n```"},
			},
			wantCode:  "new()",
			wantParts: []*genai.Part{execPart("new()")},
		},
		{
			name:  "empty_block_runs_nothing",
			reply: []*genai.Part{{Text: "```python\n\n```"}},
		},
		{
			name:     "empty_delimiter_list_runs_nothing",
			settings: codeexecutor.Settings{CodeBlockDelimiters: [][2]string{}},
			reply:    []*genai.Part{{Text: "```python\nx\n```"}},
		},
		{
			name:  "opening_without_closing_runs_nothing",
			reply: []*genai.Part{{Text: "```python\nprint(1)"}},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			responses := []*model.LLMResponse{partsReply(tc.reply...)}
			var steps []flowStep
			if tc.wantCode != "" {
				responses = append(responses, textResponse("done"))
				steps = append(steps, flowStep{res: &codeexecutor.Result{Stdout: "ok\n"}})
			}
			llm := &scriptedModel{responses: responses}
			exec := newFlowExecutor(tc.settings, steps...)
			cfg := llmagent.Config{Name: "test_agent", Model: llm, CodeExecutor: exec}
			// The reply's parts may be edited by the flow; compare with a copy.
			wantReply := cloneParts(tc.reply)

			run := runFlow(t, newPlannerAgent(t, cfg), flowOpts{})

			if run.err != nil {
				t.Fatalf("run error: %v", run.err)
			}
			calls := exec.recorded()
			if tc.wantCode == "" {
				if len(calls) != 0 {
					t.Errorf("executor calls = %d, want 0", len(calls))
				}
				if len(run.events) != 1 {
					t.Fatalf("run yielded %d events, want 1", len(run.events))
				}
				if diff := cmp.Diff(wantReply, eventParts(t, run.events[0])); diff != "" {
					t.Errorf("model event parts mismatch (-want +got):\n%s", diff)
				}
				return
			}
			if len(calls) != 1 || calls[0].code != tc.wantCode {
				t.Fatalf("executor calls = %+v, want one with Code %q", calls, tc.wantCode)
			}
			if len(run.events) != 3 {
				t.Fatalf("run yielded %d events, want 3", len(run.events))
			}
			if diff := cmp.Diff(tc.wantParts, eventParts(t, run.events[0])); diff != "" {
				t.Errorf("code event parts mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func cloneParts(parts []*genai.Part) []*genai.Part {
	out := make([]*genai.Part, len(parts))
	for i, p := range parts {
		c := *p
		out[i] = &c
	}
	return out
}

// TestLLMAgent_CodeExecutionEmptyExecutableCodeKeepsResponse is the regression
// test of a bug adk-python v2.12.0 has: there an empty executable-code part
// truncates the response to that part (code_execution_utils.py,
// extract_code_and_truncate_content). The empty part also ends the search: a
// code block in a later text part does not run.
func TestLLMAgent_CodeExecutionEmptyExecutableCodeKeepsResponse(t *testing.T) {
	tests := []struct {
		name  string
		reply func() []*genai.Part
	}{
		{
			name:  "plain_text_after",
			reply: func() []*genai.Part { return []*genai.Part{execPart(""), {Text: "answer"}} },
		},
		{
			name:  "code_block_after",
			reply: func() []*genai.Part { return []*genai.Part{execPart(""), {Text: "```python\nanswer()\n```"}} },
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			llm := &scriptedModel{responses: []*model.LLMResponse{partsReply(tc.reply()...)}}
			exec := newFlowExecutor(codeexecutor.DefaultSettings())

			run := runFlow(t, newFlowAgent(t, llm, exec), flowOpts{})

			if run.err != nil {
				t.Fatalf("run error: %v", run.err)
			}
			if len(run.events) != 1 {
				t.Fatalf("run yielded %d events, want 1", len(run.events))
			}
			if diff := cmp.Diff(tc.reply(), eventParts(t, run.events[0])); diff != "" {
				t.Errorf("model event parts mismatch (-want +got):\n%s", diff)
			}
			if got := len(exec.recorded()); got != 0 {
				t.Errorf("executor calls = %d, want 0", got)
			}
			if got := len(llm.recorded()); got != 1 {
				t.Errorf("model calls = %d, want 1", got)
			}
		})
	}
}

func TestLLMAgent_CodeExecutionMovesThoughtsInFrontOfCode(t *testing.T) {
	llm := &scriptedModel{responses: []*model.LLMResponse{
		partsReply(&genai.Part{Text: "```python\nx = 1\n```"}, &genai.Part{Text: "late thought", Thought: true, ThoughtSignature: []byte("s2")}),
		textResponse("done"),
	}}
	exec := newFlowExecutor(codeexecutor.DefaultSettings(), flowStep{res: &codeexecutor.Result{Stdout: "ok\n"}})

	run := runFlow(t, newFlowAgent(t, llm, exec), flowOpts{})

	if run.err != nil {
		t.Fatalf("run error: %v", run.err)
	}
	if len(run.events) != 3 {
		t.Fatalf("run yielded %d events, want 3", len(run.events))
	}
	want := []*genai.Part{{Text: "late thought", Thought: true, ThoughtSignature: []byte("s2")}, execPart("x = 1")}
	if diff := cmp.Diff(want, eventParts(t, run.events[0])); diff != "" {
		t.Errorf("code event parts mismatch (-want +got):\n%s", diff)
	}
	if calls := exec.recorded(); len(calls) != 1 || calls[0].code != "x = 1" {
		t.Errorf("executor calls = %+v, want one with Code %q", calls, "x = 1")
	}
}

func TestLLMAgent_CodeExecutionLeavesThoughtResponseUnchanged(t *testing.T) {
	reply := func() []*genai.Part {
		return []*genai.Part{
			{Text: "```python\nx = 1\n```", Thought: true, ThoughtSignature: []byte("sig")},
			{Text: "```python\nunsigned_thought()\n```", Thought: true},
			{Text: "The answer is 720."},
		}
	}
	llm := &scriptedModel{responses: []*model.LLMResponse{partsReply(reply()...)}}
	exec := newFlowExecutor(codeexecutor.DefaultSettings())

	run := runFlow(t, newFlowAgent(t, llm, exec), flowOpts{})

	if run.err != nil {
		t.Fatalf("run error: %v", run.err)
	}
	if len(run.events) != 1 {
		t.Fatalf("run yielded %d events, want 1", len(run.events))
	}
	if diff := cmp.Diff(reply(), eventParts(t, run.events[0])); diff != "" {
		t.Errorf("model event parts mismatch (-want +got):\n%s", diff)
	}
	if got := len(exec.recorded()); got != 0 {
		t.Errorf("executor calls = %d, want 0", got)
	}
}

func TestLLMAgent_CodeExecutionResultText(t *testing.T) {
	aTxt := &codeexecutor.File{Name: "a.txt", Content: []byte("a")}
	bCsv := &codeexecutor.File{Name: "b.csv", Content: []byte("b")}
	tests := []struct {
		name        string
		res         *codeexecutor.Result
		wantOutcome genai.Outcome
		wantOutput  string
	}{
		{name: "stdout", res: &codeexecutor.Result{Stdout: "42\n"}, wantOutcome: genai.OutcomeOK, wantOutput: "Code execution result:\n42\n\n"},
		{name: "stderr_wins", res: &codeexecutor.Result{Stdout: "ignored\n", Stderr: "boom"}, wantOutcome: genai.OutcomeFailed, wantOutput: "boom"},
		{name: "files_only", res: &codeexecutor.Result{OutputFiles: []*codeexecutor.File{aTxt}}, wantOutcome: genai.OutcomeOK, wantOutput: "Saved artifacts:\n`a.txt`"},
		{
			name:        "stdout_and_files",
			res:         &codeexecutor.Result{Stdout: "hi\n", OutputFiles: []*codeexecutor.File{aTxt, bCsv}},
			wantOutcome: genai.OutcomeOK,
			wantOutput:  "Code execution result:\nhi\n\n\n\nSaved artifacts:\n`a.txt`,`b.csv`",
		},
		{name: "empty", res: &codeexecutor.Result{}, wantOutcome: genai.OutcomeOK, wantOutput: "Code execution result:\n\n"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			llm := &scriptedModel{responses: []*model.LLMResponse{textResponse("```python\nx = 1\n```"), textResponse("done")}}
			exec := newFlowExecutor(codeexecutor.DefaultSettings(), flowStep{res: tc.res})

			run := runFlow(t, newFlowAgent(t, llm, exec), flowOpts{})

			if run.err != nil {
				t.Fatalf("run error: %v", run.err)
			}
			if len(run.events) != 3 {
				t.Fatalf("run yielded %d events, want 3", len(run.events))
			}
			want := &genai.CodeExecutionResult{Outcome: tc.wantOutcome, Output: tc.wantOutput}
			if diff := cmp.Diff(want, resultPart(t, run.events[1])); diff != "" {
				t.Errorf("result part mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestLLMAgent_CodeExecutionSavesOutputFiles(t *testing.T) {
	llm := &scriptedModel{responses: []*model.LLMResponse{
		textResponse("```python\nx1()\n```"),
		textResponse("```python\nx2()\n```"),
		textResponse("```python\nx3()\n```"),
		textResponse("done"),
	}}
	exec := newFlowExecutor(codeexecutor.DefaultSettings(),
		flowStep{res: &codeexecutor.Result{OutputFiles: []*codeexecutor.File{
			{Name: "p.png", Content: []byte{0x89, 0x50}, MimeType: "image/png"},
			{Name: "t.txt", Content: []byte("hi")},
		}}},
		flowStep{res: &codeexecutor.Result{OutputFiles: []*codeexecutor.File{{Name: "p.png", Content: []byte("v2"), MimeType: "image/png"}}}},
		flowStep{res: &codeexecutor.Result{Stderr: "boom", OutputFiles: []*codeexecutor.File{{Name: "e.txt", Content: []byte("x"), MimeType: "text/plain"}}}},
	)

	run := runFlow(t, newFlowAgent(t, llm, exec), flowOpts{})

	if run.err != nil {
		t.Fatalf("run error: %v", run.err)
	}
	if len(run.events) != 7 {
		t.Fatalf("run yielded %d events, want 7", len(run.events))
	}
	// adk-go's in-memory artifact service numbers versions from 1.
	wantDeltas := []map[string]int64{{"p.png": 1, "t.txt": 1}, {"p.png": 2}, {"e.txt": 1}}
	for i, want := range wantDeltas {
		if diff := cmp.Diff(want, run.events[2*i+1].Actions.ArtifactDelta); diff != "" {
			t.Errorf("turn %d result event ArtifactDelta mismatch (-want +got):\n%s", i+1, diff)
		}
	}
	wantResult := &genai.CodeExecutionResult{Outcome: genai.OutcomeFailed, Output: "boom"}
	if diff := cmp.Diff(wantResult, resultPart(t, run.events[5])); diff != "" {
		t.Errorf("turn 3 result part mismatch (-want +got):\n%s", diff)
	}
	wantSaves := []savedArtifact{
		{name: "p.png", data: "\x89\x50", mimeType: "image/png"},
		{name: "t.txt", data: "hi", mimeType: "text/plain"},
		{name: "p.png", data: "v2", mimeType: "image/png"},
		{name: "e.txt", data: "x", mimeType: "text/plain"},
	}
	if diff := cmp.Diff(wantSaves, run.artifacts.recorded(), cmp.AllowUnexported(savedArtifact{})); diff != "" {
		t.Errorf("saved artifacts mismatch (-want +got):\n%s", diff)
	}
}

func TestLLMAgent_CodeExecutionOutputFileErrors(t *testing.T) {
	errSave := errors.New("save failed")
	tests := []struct {
		name      string
		files     []*codeexecutor.File
		failName  string
		wantErrIs error
		wantSaves []savedArtifact
	}{
		{name: "nil_file", files: []*codeexecutor.File{nil}},
		{
			name:      "failing_save",
			files:     []*codeexecutor.File{{Name: "ok.txt", Content: []byte("ok")}, {Name: "bad.txt", Content: []byte("bad")}},
			failName:  "bad.txt",
			wantErrIs: errSave,
			wantSaves: []savedArtifact{{name: "ok.txt", data: "ok", mimeType: "text/plain"}},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			llm := &scriptedModel{responses: []*model.LLMResponse{textResponse("```python\nx = 1\n```"), textResponse("done")}}
			exec := newFlowExecutor(codeexecutor.DefaultSettings(), flowStep{res: &codeexecutor.Result{OutputFiles: tc.files}})
			arts := newFlowArtifacts()
			arts.failName, arts.failErr = tc.failName, errSave

			run := runFlow(t, newFlowAgent(t, llm, exec), flowOpts{artifacts: arts})

			if run.err == nil {
				t.Fatalf("run error = nil, want an error")
			}
			if tc.wantErrIs != nil && !errors.Is(run.err, tc.wantErrIs) {
				t.Errorf("run error = %v, want one wrapping %v", run.err, tc.wantErrIs)
			}
			if len(run.events) != 1 || run.events[0].LLMResponse.Content == nil || run.events[0].LLMResponse.Content.Parts[0].ExecutableCode == nil {
				t.Errorf("run yielded %d events, want only the code event", len(run.events))
			}
			if diff := cmp.Diff(tc.wantSaves, arts.recorded(), cmp.AllowUnexported(savedArtifact{})); diff != "" {
				t.Errorf("saved artifacts mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestLLMAgent_CodeExecutionRetryGate(t *testing.T) {
	fail := flowStep{res: &codeexecutor.Result{Stderr: "boom"}}
	succeed := flowStep{res: &codeexecutor.Result{Stdout: "ok\n"}}
	reply := func(n int) string { return fmt.Sprintf("pre\n```python\nx%d\n```\npost", n) }
	tests := []struct {
		name       string
		attempts   *int
		steps      []flowStep
		replies    int
		wantEvents int
	}{
		{name: "default_closes_after_two_failures", steps: []flowStep{fail, fail}, replies: 3, wantEvents: 5},
		{name: "three_attempts", attempts: new(3), steps: []flowStep{fail, fail, fail}, replies: 4, wantEvents: 7},
		{name: "success_resets_count", steps: []flowStep{fail, succeed, fail, fail}, replies: 5, wantEvents: 9},
		{name: "zero_attempts_never_runs", attempts: new(0), replies: 1, wantEvents: 1},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var responses []*model.LLMResponse
			for n := 1; n <= tc.replies; n++ {
				responses = append(responses, textResponse(reply(n)))
			}
			llm := &scriptedModel{responses: responses}
			exec := newFlowExecutor(codeexecutor.Settings{ErrorRetryAttempts: tc.attempts}, tc.steps...)

			run := runFlow(t, newFlowAgent(t, llm, exec), flowOpts{})

			if run.err != nil {
				t.Fatalf("run error: %v", run.err)
			}
			if got := len(exec.recorded()); got != len(tc.steps) {
				t.Errorf("executor calls = %d, want %d", got, len(tc.steps))
			}
			if len(run.events) != tc.wantEvents {
				t.Fatalf("run yielded %d events, want %d", len(run.events), tc.wantEvents)
			}
			want := []*genai.Part{{Text: reply(tc.replies)}}
			if diff := cmp.Diff(want, eventParts(t, run.events[len(run.events)-1])); diff != "" {
				t.Errorf("last event parts mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func newSQLiteSessions(t *testing.T) session.Service {
	t.Helper()
	svc, err := database.NewSessionServiceFromURL("sqlite:///:memory:", &gorm.Config{Logger: logger.Discard})
	if err != nil {
		t.Fatalf("database.NewSessionServiceFromURL: %v", err)
	}
	t.Cleanup(func() {
		if err := svc.(io.Closer).Close(); err != nil {
			t.Errorf("close session service: %v", err)
		}
	})
	if err := database.AutoMigrate(svc); err != nil {
		t.Fatalf("database.AutoMigrate: %v", err)
	}
	return svc
}

func TestLLMAgent_CodeExecutionExecutionID(t *testing.T) {
	csv := &codeexecutor.File{Name: "data.csv", Content: []byte("a,b\n1,2\n"), MimeType: "text/csv"}
	tests := []struct {
		name           string
		stateful       bool
		state          map[string]any
		sqlite         bool
		wantID         string
		wantDelta      map[string]any
		wantInputFiles []*codeexecutor.File
	}{
		{
			name:      "stateful_uses_session_id",
			stateful:  true,
			wantID:    "s1",
			wantDelta: map[string]any{codeExecutionContextKey: map[string]any{"execution_session_id": "s1"}},
		},
		{
			name:      "not_stateful",
			wantID:    "",
			wantDelta: map[string]any{codeExecutionContextKey: map[string]any{}},
		},
		{
			name:      "stored_id_wins",
			stateful:  true,
			state:     map[string]any{codeExecutionContextKey: map[string]any{"execution_session_id": "custom"}},
			wantID:    "custom",
			wantDelta: map[string]any{codeExecutionContextKey: map[string]any{"execution_session_id": "custom"}},
		},
		{
			name:      "stored_empty_id_gives_session_id",
			stateful:  true,
			state:     map[string]any{codeExecutionContextKey: map[string]any{"execution_session_id": ""}},
			wantID:    "s1",
			wantDelta: map[string]any{codeExecutionContextKey: map[string]any{"execution_session_id": "s1"}},
		},
		{
			name:      "stored_id_wins_read_back_from_sql",
			stateful:  true,
			state:     map[string]any{codeExecutionContextKey: map[string]any{"execution_session_id": "custom"}},
			sqlite:    true,
			wantID:    "custom",
			wantDelta: map[string]any{codeExecutionContextKey: map[string]any{"execution_session_id": "custom"}},
		},
		{
			name:           "stored_input_files_reach_executor",
			state:          map[string]any{inputFilesKey: []*codeexecutor.File{csv}},
			wantID:         "",
			wantDelta:      map[string]any{codeExecutionContextKey: map[string]any{}},
			wantInputFiles: []*codeexecutor.File{{Name: "data.csv", Content: []byte("a,b\n1,2\n"), MimeType: "text/csv"}},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			llm := &scriptedModel{responses: []*model.LLMResponse{textResponse("```python\nx = 1\n```"), textResponse("done")}}
			exec := newFlowExecutor(codeexecutor.Settings{Stateful: tc.stateful}, flowStep{res: &codeexecutor.Result{Stdout: "ok\n"}})
			opts := flowOpts{state: tc.state}
			if tc.sqlite {
				opts.sessions = newSQLiteSessions(t)
			}

			run := runFlow(t, newFlowAgent(t, llm, exec), opts)

			if run.err != nil {
				t.Fatalf("run error: %v", run.err)
			}
			calls := exec.recorded()
			if len(calls) != 1 {
				t.Fatalf("executor calls = %d, want 1", len(calls))
			}
			if calls[0].executionID != tc.wantID {
				t.Errorf("executor ExecutionID = %q, want %q", calls[0].executionID, tc.wantID)
			}
			if diff := cmp.Diff(tc.wantInputFiles, calls[0].inputFiles); diff != "" {
				t.Errorf("executor InputFiles mismatch (-want +got):\n%s", diff)
			}
			if len(run.events) != 3 {
				t.Fatalf("run yielded %d events, want 3", len(run.events))
			}
			if diff := cmp.Diff(tc.wantDelta, run.events[1].Actions.StateDelta); diff != "" {
				t.Errorf("result event StateDelta mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestLLMAgent_CodeExecutionBookkeepingStaysOutOfDeltas(t *testing.T) {
	llm := &scriptedModel{responses: []*model.LLMResponse{
		textResponse("```python\nx=1\n```"),
		textResponse("```python\nx=2\n```"),
		textResponse("```python\nx=3\n```"),
		textResponse("done"),
	}}
	exec := newFlowExecutor(codeexecutor.DefaultSettings(),
		flowStep{res: &codeexecutor.Result{Stderr: "boom"}},
		flowStep{res: &codeexecutor.Result{Stdout: "ok"}},
		flowStep{res: &codeexecutor.Result{Stdout: "ok"}},
	)
	fixed := time.Date(2025, 1, 1, 12, 0, 0, 0, time.UTC)
	ctx := platform.WithTimeProvider(t.Context(), func() time.Time { return fixed })

	run := runFlow(t, newFlowAgent(t, llm, exec), flowOpts{ctx: ctx})

	if run.err != nil {
		t.Fatalf("run error: %v", run.err)
	}
	calls := exec.recorded()
	if len(calls) != 3 {
		t.Fatalf("executor calls = %d, want 3", len(calls))
	}
	inv := calls[0].invocationID
	want2 := map[string]string{
		codeExecutionContextKey: `{}`,
		executionResultsKey:     `{"` + inv + `":[{"code":"x=1","result_stderr":"boom","result_stdout":"","timestamp":1735732800}]}`,
		errorCountsKey:          `{"` + inv + `":1}`,
	}
	if diff := cmp.Diff(want2, calls[1].stateJSON); diff != "" {
		t.Errorf("state at executor call 2 mismatch (-want +got):\n%s", diff)
	}
	want3 := map[string]string{
		codeExecutionContextKey: `{}`,
		executionResultsKey: `{"` + inv + `":[{"code":"x=1","result_stderr":"boom","result_stdout":"","timestamp":1735732800},` +
			`{"code":"x=2","result_stderr":"","result_stdout":"ok","timestamp":1735732800}]}`,
		errorCountsKey: `{}`,
	}
	if diff := cmp.Diff(want3, calls[2].stateJSON); diff != "" {
		t.Errorf("state at executor call 3 mismatch (-want +got):\n%s", diff)
	}
	var storedKeys []string
	for k := range run.sess.State().All() {
		storedKeys = append(storedKeys, k)
	}
	if diff := cmp.Diff([]string{codeExecutionContextKey}, storedKeys); diff != "" {
		t.Errorf("stored session state keys mismatch (-want +got):\n%s", diff)
	}
}

func TestLLMAgent_CodeExecutionExecutorFailureEndsRun(t *testing.T) {
	errBoom := errors.New("boom")
	tests := []struct {
		name    string
		step    flowStep
		wantIs  error
		wantMsg string
	}{
		{name: "executor_error", step: flowStep{err: errBoom}, wantIs: errBoom},
		{name: "nil_result", step: flowStep{}, wantMsg: "execute code: code executor returned no result"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			llm := &scriptedModel{responses: []*model.LLMResponse{textResponse("```python\nsecret_marker()\n```"), textResponse("done")}}
			exec := newFlowExecutor(codeexecutor.DefaultSettings(), tc.step)

			run := runFlow(t, newFlowAgent(t, llm, exec), flowOpts{})

			if run.err == nil {
				t.Fatalf("run error = nil, want an error")
			}
			if tc.wantIs != nil && !errors.Is(run.err, tc.wantIs) {
				t.Errorf("run error = %v, want one wrapping %v", run.err, tc.wantIs)
			}
			if tc.wantMsg != "" && run.err.Error() != tc.wantMsg {
				t.Errorf("run error = %q, want %q", run.err.Error(), tc.wantMsg)
			}
			if strings.Contains(run.err.Error(), "secret_marker") {
				t.Errorf("run error names the executed code")
			}
			if len(run.events) != 1 || run.events[0].LLMResponse.Content.Parts[0].ExecutableCode == nil {
				t.Errorf("run yielded %d events, want only the code event", len(run.events))
			}
			if got := len(exec.recorded()); got != 1 {
				t.Errorf("executor calls = %d, want 1", got)
			}
			if got := len(llm.recorded()); got != 1 {
				t.Errorf("model calls = %d, want 1", got)
			}
		})
	}
}

func TestLLMAgent_CodeExecutionCancellation(t *testing.T) {
	errNotCancelled := errors.New("executor context was not cancelled")
	entered := make(chan struct{})
	exec := newFlowExecutor(codeexecutor.DefaultSettings(), flowStep{run: func(ctx context.Context) (*codeexecutor.Result, error) {
		close(entered)
		select {
		case <-ctx.Done():
			return nil, fmt.Errorf("stop: %w", ctx.Err())
		case <-time.After(10 * time.Second):
			return nil, errNotCancelled
		}
	}})
	llm := &scriptedModel{responses: []*model.LLMResponse{textResponse("```python\nx = 1\n```"), textResponse("done")}}
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	go func() {
		select {
		case <-entered:
			cancel()
		case <-ctx.Done():
		}
	}()

	run := runFlow(t, newFlowAgent(t, llm, exec), flowOpts{ctx: ctx})

	if !errors.Is(run.err, context.Canceled) {
		t.Errorf("run error = %v, want one wrapping context.Canceled", run.err)
	}
	if errors.Is(run.err, errNotCancelled) {
		t.Errorf("run error = %v: the executor did not see the cancellation", run.err)
	}
	if len(run.events) != 1 {
		t.Errorf("run yielded %d events, want only the code event", len(run.events))
	}
}

func TestLLMAgent_CodeExecutionWithoutArtifactService(t *testing.T) {
	const wantErr = "artifact service is not initialized"
	t.Run("non_built_in_without_output_files", func(t *testing.T) {
		llm := &scriptedModel{responses: []*model.LLMResponse{textResponse("```python\nx = 1\n```"), textResponse("done")}}
		exec := newFlowExecutor(codeexecutor.DefaultSettings(), flowStep{res: &codeexecutor.Result{Stdout: "42\n"}})

		run := runFlow(t, newFlowAgent(t, llm, exec), flowOpts{noArtifacts: true})

		if run.err == nil || run.err.Error() != wantErr {
			t.Errorf("run error = %v, want %q", run.err, wantErr)
		}
		if len(run.events) != 1 || run.events[0].LLMResponse.Content.Parts[0].ExecutableCode == nil {
			t.Errorf("run yielded %d events, want only the code event", len(run.events))
		}
		if got := len(exec.recorded()); got != 1 {
			t.Errorf("executor calls = %d, want 1", got)
		}
	})
	t.Run("built_in_text_only", func(t *testing.T) {
		llm := newNamedModel("gemini-2.0-flash", textResponse("just text"))

		run := runFlow(t, newFlowAgent(t, llm, codeexecutor.NewBuiltInExecutor()), flowOpts{noArtifacts: true})

		if run.err != nil {
			t.Fatalf("run error: %v", run.err)
		}
		if len(run.events) != 2 {
			t.Fatalf("run yielded %d events, want 2", len(run.events))
		}
		if run.events[0].LLMResponse.Content != nil {
			t.Errorf("first event Content = %v, want nil", run.events[0].LLMResponse.Content)
		}
		if diff := cmp.Diff([]*genai.Part{{Text: "just text"}}, eventParts(t, run.events[1])); diff != "" {
			t.Errorf("model event parts mismatch (-want +got):\n%s", diff)
		}
	})
	t.Run("built_in_with_image", func(t *testing.T) {
		llm := newNamedModel("gemini-2.0-flash", partsReply(&genai.Part{InlineData: &genai.Blob{Data: []byte("png"), MIMEType: "image/png"}}))

		run := runFlow(t, newFlowAgent(t, llm, codeexecutor.NewBuiltInExecutor()), flowOpts{noArtifacts: true})

		if run.err == nil || run.err.Error() != wantErr {
			t.Errorf("run error = %v, want %q", run.err, wantErr)
		}
		if len(run.events) != 0 {
			t.Errorf("run yielded %d events, want 0", len(run.events))
		}
	})
}

func TestLLMAgent_CodeExecutionHistoryAsFencedText(t *testing.T) {
	llm := &scriptedModel{responses: []*model.LLMResponse{
		textResponse("```python\nprint(6 * 7)\n```"),
		textResponse("The answer is 42."),
	}}
	exec := newFlowExecutor(codeexecutor.DefaultSettings(), flowStep{res: &codeexecutor.Result{Stdout: "42\n"}})

	run := runFlow(t, newFlowAgent(t, llm, exec), flowOpts{})

	if run.err != nil {
		t.Fatalf("run error: %v", run.err)
	}
	reqs := llm.recorded()
	if len(reqs) != 2 {
		t.Fatalf("model calls = %d, want 2", len(reqs))
	}
	want := []roleText{
		{Role: genai.RoleUser, Texts: []string{"go"}},
		{Role: genai.RoleModel, Texts: []string{"```tool_code\nprint(6 * 7)\n```"}},
		{Role: genai.RoleUser, Texts: []string{"```tool_output\nCode execution result:\n42\n\n\n```"}},
	}
	contents := reqs[1].Contents
	if len(contents) < len(want) {
		t.Fatalf("second request has %d contents, want at least %d", len(contents), len(want))
	}
	// Only the history is compared: the contents processor may append its
	// continuation turn after it, which is not part of the conversion.
	if diff := cmp.Diff(want, roleTexts(contents[:len(want)])); diff != "" {
		t.Errorf("second request contents mismatch (-want +got):\n%s", diff)
	}
	// Stored history keeps the code execution parts.
	var stored []*session.Event
	for ev := range run.sess.Events().All() {
		if ev.Author == "test_agent" {
			stored = append(stored, ev)
		}
	}
	if len(stored) != 3 {
		t.Fatalf("stored agent events = %d, want 3", len(stored))
	}
	if diff := cmp.Diff([]*genai.Part{execPart("print(6 * 7)")}, eventParts(t, stored[0])); diff != "" {
		t.Errorf("stored code event parts mismatch (-want +got):\n%s", diff)
	}
	if got := stored[1].LLMResponse.Content.Role; got != genai.RoleModel {
		t.Errorf("stored result event Role = %q, want %q", got, genai.RoleModel)
	}
	resultPart(t, stored[1])
}

func TestLLMAgent_CodeExecutionHistoryConversionCases(t *testing.T) {
	resultOK := func(out string) *genai.Part { return genai.NewPartFromCodeExecutionResult(genai.OutcomeOK, out) }
	tests := []struct {
		name     string
		builtIn  bool
		settings codeexecutor.Settings
		seed     []*genai.Part
		want     *genai.Content
	}{
		{
			name:     "first_pair_fences_code",
			settings: codeexecutor.Settings{CodeBlockDelimiters: [][2]string{{"<code>", "</code>"}, {"```python\n", "\n```"}}},
			seed:     []*genai.Part{execPart("x()")},
			want:     genai.NewContentFromText("<code>x()</code>", genai.RoleModel),
		},
		{
			name:     "empty_delimiter_list_leaves_code_bare",
			settings: codeexecutor.Settings{CodeBlockDelimiters: [][2]string{}},
			seed:     []*genai.Part{execPart("x()")},
			want:     genai.NewContentFromText("x()", genai.RoleModel),
		},
		{
			name:     "result_delimiters",
			settings: codeexecutor.Settings{ExecutionResultDelimiters: &[2]string{"<out>", "</out>"}},
			seed:     []*genai.Part{resultOK("42")},
			want:     genai.NewContentFromText("<out>42</out>", genai.RoleUser),
		},
		{
			name: "empty_output_gives_empty_text",
			seed: []*genai.Part{resultOK("")},
			want: &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{{Text: ""}}},
		},
		{
			name: "several_parts_ending_in_result_unchanged",
			seed: []*genai.Part{{Text: "t"}, resultOK("r")},
			want: &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{{Text: "t"}, resultOK("r")}},
		},
		{
			name: "trailing_code_after_text",
			seed: []*genai.Part{{Text: "t"}, execPart("y()")},
			want: &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{{Text: "t"}, {Text: "```tool_code\ny()\n```"}}},
		},
		{
			name:    "built_in_executor_converts_too",
			builtIn: true,
			seed:    []*genai.Part{execPart("z()")},
			want:    genai.NewContentFromText("```tool_code\nz()\n```", genai.RoleModel),
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			seed := session.NewEvent(t.Context(), "inv-old")
			seed.Author = "test_agent"
			seed.LLMResponse = model.LLMResponse{Content: &genai.Content{Role: genai.RoleModel, Parts: tc.seed}}
			var llm interface {
				model.LLM
				recorded() []*model.LLMRequest
			}
			var exec codeexecutor.CodeExecutor
			if tc.builtIn {
				llm = newNamedModel("gemini-2.0-flash", textResponse("done"))
				exec = codeexecutor.NewBuiltInExecutor()
			} else {
				llm = &scriptedModel{responses: []*model.LLMResponse{textResponse("done")}}
				exec = newFlowExecutor(tc.settings)
			}

			run := runFlow(t, newFlowAgent(t, llm, exec), flowOpts{history: []*session.Event{seed}})

			if run.err != nil {
				t.Fatalf("run error: %v", run.err)
			}
			reqs := llm.recorded()
			if len(reqs) != 1 || len(reqs[0].Contents) == 0 {
				t.Fatalf("model requests = %+v, want one with contents", reqs)
			}
			if diff := cmp.Diff(tc.want, reqs[0].Contents[0]); diff != "" {
				t.Errorf("first request content mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// Agent b's request relays agent a's code turn as a user content that ends in
// the code; the conversion fences the code and keeps role user (adk-python
// flows/llm_flows/context/_fencing.py, code_execution_utils.py,
// convert_code_execution_parts).
func TestLLMAgent_CodeExecutionHistoryKeepsRelayedCodeTurnRole(t *testing.T) {
	aModel := &scriptedModel{responses: []*model.LLMResponse{textResponse("```python\na_code()\n```"), textResponse("a done")}}
	aExec := newFlowExecutor(codeexecutor.DefaultSettings(), flowStep{res: &codeexecutor.Result{Stdout: "1"}})
	bModel := &scriptedModel{responses: []*model.LLMResponse{textResponse("b done")}}
	seq, err := sequentialagent.New(sequentialagent.Config{AgentConfig: agent.Config{Name: "seq", SubAgents: []agent.Agent{
		newPlannerAgent(t, llmagent.Config{Name: "a", Model: aModel, CodeExecutor: aExec}),
		newPlannerAgent(t, llmagent.Config{Name: "b", Model: bModel, CodeExecutor: newFlowExecutor(codeexecutor.DefaultSettings())}),
	}}})
	if err != nil {
		t.Fatalf("sequentialagent.New: %v", err)
	}

	run := runFlow(t, seq, flowOpts{})

	if run.err != nil {
		t.Fatalf("run error: %v", run.err)
	}
	reqs := bModel.recorded()
	if len(reqs) != 1 {
		t.Fatalf("b model calls = %d, want 1", len(reqs))
	}
	var roles []string
	for _, rt := range roleTexts(reqs[0].Contents) {
		if n := len(rt.Texts); n > 1 && rt.Texts[n-1] == "```tool_code\na_code()\n```" {
			roles = append(roles, rt.Role)
		}
	}
	if diff := cmp.Diff([]string{genai.RoleUser}, roles); diff != "" {
		t.Errorf("roles of the relayed code turn in b's request (-want +got):\n%s", diff)
	}
}

// flowBarrier lets n goroutines wait until all of them have arrived, or until
// 10 s have passed.
type flowBarrier struct {
	mu      sync.Mutex
	pending int
	all     chan struct{}
}

func newFlowBarrier(n int) *flowBarrier { return &flowBarrier{pending: n, all: make(chan struct{})} }

func (b *flowBarrier) arrive() error {
	b.mu.Lock()
	b.pending--
	if b.pending == 0 {
		close(b.all)
	}
	b.mu.Unlock()
	select {
	case <-b.all:
		return nil
	case <-time.After(10 * time.Second):
		return errBarrierTimeout
	}
}

// gatedModel is a scriptedModel whose second call waits at gate.
type gatedModel struct {
	*scriptedModel
	gate  *flowBarrier
	calls atomic.Int32
}

func (m *gatedModel) GenerateContent(ctx context.Context, req *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error] {
	if m.calls.Add(1) == 2 {
		if err := m.gate.arrive(); err != nil {
			return func(yield func(*model.LLMResponse, error) bool) { yield(nil, err) }
		}
	}
	return m.scriptedModel.GenerateContent(ctx, req, stream)
}

// newFailingParallelAgent returns a parallel agent of two agents, "a" and "b",
// and their executors. Each agent answers with a code block, which its
// executor runs as fail, and then, at modelGate, with a second one. The
// executors have the default limit of two failures and one step each, so a
// second execution fails the run.
func newFailingParallelAgent(t *testing.T, fail flowStep, modelGate *flowBarrier) (agent.Agent, []*flowExecutor) {
	t.Helper()
	var execs []*flowExecutor
	var branches []agent.Agent
	for _, name := range []string{"a", "b"} {
		llm := &gatedModel{gate: modelGate, scriptedModel: &scriptedModel{responses: []*model.LLMResponse{
			textResponse("```python\n" + name + "_code()\n```"),
			textResponse("```python\n" + name + "_again()\n```"),
		}}}
		exec := newFlowExecutor(codeexecutor.DefaultSettings(), fail)
		execs = append(execs, exec)
		branches = append(branches, newPlannerAgent(t, llmagent.Config{Name: name, Model: llm, CodeExecutor: exec}))
	}
	parallel, err := parallelagent.New(parallelagent.Config{AgentConfig: agent.Config{Name: "parallel", SubAgents: branches}})
	if err != nil {
		t.Fatalf("parallelagent.New: %v", err)
	}
	return parallel, execs
}

func TestLLMAgent_CodeExecutionParallelAgentsShareBookkeeping(t *testing.T) {
	for i := range 50 {
		t.Run(fmt.Sprintf("run_%d", i), func(t *testing.T) {
			execGate, modelGate := newFlowBarrier(2), newFlowBarrier(2)
			failAtGate := flowStep{run: func(context.Context) (*codeexecutor.Result, error) {
				if err := execGate.arrive(); err != nil {
					return nil, err
				}
				return &codeexecutor.Result{Stderr: "boom"}, nil
			}}
			parallel, execs := newFailingParallelAgent(t, failAtGate, modelGate)
			var probed any
			probe, err := agent.New(agent.Config{Name: "probe", Run: func(ic agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(func(*session.Event, error) bool) {
					probed, _ = ic.Session().State().Get(executionResultsKey)
				}
			}})
			if err != nil {
				t.Fatalf("agent.New: %v", err)
			}
			root, err := sequentialagent.New(sequentialagent.Config{AgentConfig: agent.Config{Name: "root", SubAgents: []agent.Agent{parallel, probe}}})
			if err != nil {
				t.Fatalf("sequentialagent.New: %v", err)
			}

			run := runFlow(t, root, flowOpts{})

			if run.err != nil {
				t.Fatalf("run error: %v", run.err)
			}
			for i, exec := range execs {
				if got := len(exec.recorded()); got != 1 {
					t.Errorf("executor %d calls = %d, want 1", i, got)
				}
			}
			if len(run.events) == 0 {
				t.Fatalf("run yielded no event")
			}
			records, ok := probed.(map[string][]map[string]any)
			if !ok {
				t.Fatalf("probed %s = %T, want map[string][]map[string]any", executionResultsKey, probed)
			}
			var codes []string
			for _, rec := range records[run.events[0].InvocationID] {
				codes = append(codes, fmt.Sprint(rec["code"]))
			}
			slices.Sort(codes)
			if diff := cmp.Diff([]string{"a_code()", "b_code()"}, codes); diff != "" {
				t.Errorf("recorded codes mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// pairedReadWait bounds how long readPairer holds the first read. A read
// cannot tell a second reader that waits for the bookkeeping lock from one
// that is slow, so the hold ends after this time; a second reader that does
// not wait for a lock arrives within microseconds.
const pairedReadWait = time.Second

// readPairer holds the first read of the error counts that comes after arm
// until a second such read has also read the stored value, or until
// pairedReadWait has passed. Two read-modify-writes that do not exclude each
// other then both read the same value.
type readPairer struct {
	armed  atomic.Bool
	mu     sync.Mutex
	reads  int
	paired chan struct{}
}

func newReadPairer() *readPairer { return &readPairer{paired: make(chan struct{})} }

func (p *readPairer) arm() { p.armed.Store(true) }

// afterRead is called after each read of key, once its value is read.
func (p *readPairer) afterRead(key string) {
	if key != errorCountsKey || !p.armed.Load() {
		return
	}
	p.mu.Lock()
	p.reads++
	n := p.reads
	if n == 2 {
		close(p.paired)
	}
	p.mu.Unlock()
	if n != 1 {
		return
	}
	select {
	case <-p.paired:
	case <-time.After(pairedReadWait):
	}
}

// pairingState is a session.State whose reads go through a readPairer.
type pairingState struct {
	session.State
	pairer *readPairer
}

func (s *pairingState) Get(key string) (any, error) {
	v, err := s.State.Get(key)
	s.pairer.afterRead(key)
	return v, err
}

// pairingSession is a session.Session whose State is a pairingState.
type pairingSession struct {
	session.Session
	state *pairingState
}

func (s *pairingSession) State() session.State { return s.state }

// pairingSessions is a session.Service whose Get returns a pairingSession.
// AppendEvent passes the session the pairingSession wraps to the wrapped
// service, so the events and state deltas reach it.
type pairingSessions struct {
	session.Service
	pairer *readPairer
}

func (s *pairingSessions) Get(ctx context.Context, req *session.GetRequest) (*session.GetResponse, error) {
	resp, err := s.Service.Get(ctx, req)
	if err != nil {
		return nil, err
	}
	wrapped := &pairingSession{Session: resp.Session, state: &pairingState{State: resp.Session.State(), pairer: s.pairer}}
	return &session.GetResponse{Session: wrapped}, nil
}

func (s *pairingSessions) AppendEvent(ctx context.Context, sess session.Session, ev *session.Event) error {
	if w, ok := sess.(*pairingSession); ok {
		sess = w.Session
	}
	return s.Service.AppendEvent(ctx, sess, ev)
}

func TestLLMAgent_CodeExecutionParallelFailuresAllCount(t *testing.T) {
	pairer := newReadPairer()
	execGate, modelGate := newFlowBarrier(2), newFlowBarrier(2)
	failAtGate := flowStep{run: func(context.Context) (*codeexecutor.Result, error) {
		if err := execGate.arrive(); err != nil {
			return nil, err
		}
		pairer.arm()
		return &codeexecutor.Result{Stderr: "boom"}, nil
	}}
	parallel, execs := newFailingParallelAgent(t, failAtGate, modelGate)

	run := runFlow(t, parallel, flowOpts{sessions: &pairingSessions{Service: session.InMemoryService(), pairer: pairer}})

	if run.err != nil {
		t.Fatalf("run error: %v", run.err)
	}
	for i, exec := range execs {
		if got := len(exec.recorded()); got != 1 {
			t.Errorf("executor %d calls = %d, want 1", i, got)
		}
	}
}

// setLocal sets time.Local for the test and restores it afterwards. A test
// that calls it must not run in parallel.
func setLocal(t *testing.T, loc *time.Location) {
	t.Helper()
	old := time.Local
	time.Local = loc
	t.Cleanup(func() { time.Local = old })
}

func TestLLMAgent_BuiltInCodeExecutionSavesImages(t *testing.T) {
	setLocal(t, time.FixedZone("test", 3600))
	fixed := time.Date(2025, 1, 1, 12, 0, 0, 0, time.UTC)
	ctx := platform.WithTimeProvider(t.Context(), func() time.Time { return fixed })
	saved := func(name string) *genai.Part { return &genai.Part{Text: "Saved as artifact: " + name + ". "} }

	t.Run("names_zone_and_non_images", func(t *testing.T) {
		pdf := &genai.Part{InlineData: &genai.Blob{Data: []byte("d3"), MIMEType: "application/pdf", DisplayName: "doc.pdf"}}
		llm := newNamedModel("gemini-2.0-flash", partsReply(
			&genai.Part{InlineData: &genai.Blob{Data: []byte("d1"), MIMEType: "image/png"}},
			&genai.Part{InlineData: &genai.Blob{Data: []byte("d2"), MIMEType: "image/svg+xml"}},
			pdf,
			&genai.Part{InlineData: &genai.Blob{Data: []byte("d4"), MIMEType: "image/png", DisplayName: "chart.png"}},
		))

		run := runFlow(t, newFlowAgent(t, llm, codeexecutor.NewBuiltInExecutor()), flowOpts{ctx: ctx})

		if run.err != nil {
			t.Fatalf("run error: %v", run.err)
		}
		if len(run.events) != 2 {
			t.Fatalf("run yielded %d events, want 2", len(run.events))
		}
		if run.events[0].LLMResponse.Content != nil {
			t.Errorf("image event Content = %v, want nil", run.events[0].LLMResponse.Content)
		}
		wantDelta := map[string]int64{"20250101_130000.png": 1, "20250101_130000.svg+xml": 1, "chart.png": 1}
		if diff := cmp.Diff(wantDelta, run.events[0].Actions.ArtifactDelta); diff != "" {
			t.Errorf("image event ArtifactDelta mismatch (-want +got):\n%s", diff)
		}
		wantParts := []*genai.Part{
			saved("20250101_130000.png"),
			saved("20250101_130000.svg+xml"),
			{InlineData: &genai.Blob{Data: []byte("d3"), MIMEType: "application/pdf", DisplayName: "doc.pdf"}},
			saved("chart.png"),
		}
		if diff := cmp.Diff(wantParts, eventParts(t, run.events[1])); diff != "" {
			t.Errorf("model event parts mismatch (-want +got):\n%s", diff)
		}
		wantSaves := []savedArtifact{
			{name: "20250101_130000.png", data: "d1", mimeType: "image/png"},
			{name: "20250101_130000.svg+xml", data: "d2", mimeType: "image/svg+xml"},
			{name: "chart.png", data: "d4", mimeType: "image/png"},
		}
		if diff := cmp.Diff(wantSaves, run.artifacts.recorded(), cmp.AllowUnexported(savedArtifact{})); diff != "" {
			t.Errorf("saved artifacts mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("unnamed_images_in_one_second_share_a_name", func(t *testing.T) {
		llm := newNamedModel("gemini-2.0-flash", partsReply(
			&genai.Part{InlineData: &genai.Blob{Data: []byte("first"), MIMEType: "image/png"}},
			&genai.Part{InlineData: &genai.Blob{Data: []byte("second"), MIMEType: "image/png"}},
		))

		run := runFlow(t, newFlowAgent(t, llm, codeexecutor.NewBuiltInExecutor()), flowOpts{ctx: ctx})

		if run.err != nil {
			t.Fatalf("run error: %v", run.err)
		}
		if len(run.events) != 2 {
			t.Fatalf("run yielded %d events, want 2", len(run.events))
		}
		if diff := cmp.Diff(map[string]int64{"20250101_130000.png": 2}, run.events[0].Actions.ArtifactDelta); diff != "" {
			t.Errorf("image event ArtifactDelta mismatch (-want +got):\n%s", diff)
		}
		wantParts := []*genai.Part{saved("20250101_130000.png"), saved("20250101_130000.png")}
		if diff := cmp.Diff(wantParts, eventParts(t, run.events[1])); diff != "" {
			t.Errorf("model event parts mismatch (-want +got):\n%s", diff)
		}
		wantSaves := []savedArtifact{
			{name: "20250101_130000.png", data: "first", mimeType: "image/png"},
			{name: "20250101_130000.png", data: "second", mimeType: "image/png"},
		}
		if diff := cmp.Diff(wantSaves, run.artifacts.recorded(), cmp.AllowUnexported(savedArtifact{})); diff != "" {
			t.Errorf("saved artifacts mismatch (-want +got):\n%s", diff)
		}
	})
}

func TestLLMAgent_BuiltInCodeExecutionEmitsEventForTextOnlyResponse(t *testing.T) {
	llm := newNamedModel("gemini-2.0-flash", textResponse("just text"))

	run := runFlow(t, newFlowAgent(t, llm, codeexecutor.NewBuiltInExecutor()), flowOpts{})

	if run.err != nil {
		t.Fatalf("run error: %v", run.err)
	}
	if len(run.events) != 2 {
		t.Fatalf("run yielded %d events, want 2", len(run.events))
	}
	ev, modelEv := run.events[0], run.events[1]
	if ev.LLMResponse.Content != nil {
		t.Errorf("first event Content = %v, want nil", ev.LLMResponse.Content)
	}
	if ev.Author != "test_agent" {
		t.Errorf("first event Author = %q, want %q", ev.Author, "test_agent")
	}
	if ev.InvocationID == "" || ev.InvocationID != modelEv.InvocationID {
		t.Errorf("InvocationIDs = %q, %q, want equal and non-empty", ev.InvocationID, modelEv.InvocationID)
	}
	if len(ev.Actions.StateDelta) != 0 || len(ev.Actions.ArtifactDelta) != 0 {
		t.Errorf("first event deltas = %v, %v, want empty", ev.Actions.StateDelta, ev.Actions.ArtifactDelta)
	}
	if diff := cmp.Diff([]*genai.Part{{Text: "just text"}}, eventParts(t, modelEv)); diff != "" {
		t.Errorf("model event parts mismatch (-want +got):\n%s", diff)
	}
}

func TestLLMAgent_BuiltInCodeExecutionImageErrors(t *testing.T) {
	errSave := errors.New("save failed")
	tests := []struct {
		name      string
		parts     []*genai.Part
		failName  string
		wantErrIs error
		wantSaves []savedArtifact
	}{
		{
			name:  "image_without_data",
			parts: []*genai.Part{{InlineData: &genai.Blob{MIMEType: "image/png", DisplayName: "empty.png"}}},
		},
		{
			name: "failing_save",
			parts: []*genai.Part{
				{InlineData: &genai.Blob{Data: []byte("one"), MIMEType: "image/png", DisplayName: "first.png"}},
				{InlineData: &genai.Blob{Data: []byte("two"), MIMEType: "image/png", DisplayName: "second.png"}},
			},
			failName:  "second.png",
			wantErrIs: errSave,
			wantSaves: []savedArtifact{{name: "first.png", data: "one", mimeType: "image/png"}},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			llm := newNamedModel("gemini-2.0-flash", partsReply(tc.parts...))
			arts := newFlowArtifacts()
			arts.failName, arts.failErr = tc.failName, errSave

			run := runFlow(t, newFlowAgent(t, llm, codeexecutor.NewBuiltInExecutor()), flowOpts{artifacts: arts})

			if run.err == nil {
				t.Fatalf("run error = nil, want an error")
			}
			if tc.wantErrIs != nil && !errors.Is(run.err, tc.wantErrIs) {
				t.Errorf("run error = %v, want one wrapping %v", run.err, tc.wantErrIs)
			}
			if len(run.events) != 0 {
				t.Errorf("run yielded %d events, want 0", len(run.events))
			}
			if diff := cmp.Diff(tc.wantSaves, arts.recorded(), cmp.AllowUnexported(savedArtifact{})); diff != "" {
				t.Errorf("saved artifacts mismatch (-want +got):\n%s", diff)
			}
		})
	}
}
