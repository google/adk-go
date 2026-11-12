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
	"errors"
	"iter"
	"strings"
	"sync"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/agent/workflowagents/parallelagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/planner"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
)

var errScriptExhausted = errors.New("scripted model: no response left")

// scriptedModel is a model.LLM double that returns its responses in order,
// records every request it receives, and fails once the script runs out.
type scriptedModel struct {
	mu        sync.Mutex
	responses []*model.LLMResponse
	requests  []*model.LLMRequest
}

func (m *scriptedModel) Name() string { return "scripted" }

func (m *scriptedModel) GenerateContent(ctx context.Context, req *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		m.mu.Lock()
		m.requests = append(m.requests, req)
		if len(m.responses) == 0 {
			m.mu.Unlock()
			yield(nil, errScriptExhausted)
			return
		}
		resp := m.responses[0]
		m.responses = m.responses[1:]
		m.mu.Unlock()
		yield(resp, nil)
	}
}

func (m *scriptedModel) recorded() []*model.LLMRequest {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.requests
}

func textResponse(text string) *model.LLMResponse {
	return &model.LLMResponse{
		Content:      genai.NewContentFromText(text, genai.RoleModel),
		FinishReason: genai.FinishReasonStop,
	}
}

// instructionPlanner implements only planner.Planner.
type instructionPlanner struct {
	instruction string
	err         error
}

func (p *instructionPlanner) BuildPlanningInstruction(agent.ReadonlyContext, *model.LLMRequest) (string, error) {
	return p.instruction, p.err
}

// rewritingPlanner implements planner.RewritingPlanner. Its response step
// writes state (when stateKey is set), counts its calls, and returns parts and
// processErr.
type rewritingPlanner struct {
	instructionPlanner
	stateKey, stateVal string
	parts              []*genai.Part
	processErr         error

	mu    sync.Mutex
	calls int
}

func (p *rewritingPlanner) ProcessPlanningResponse(ctx agent.Context, _ []*genai.Part) ([]*genai.Part, error) {
	p.mu.Lock()
	p.calls++
	p.mu.Unlock()
	if p.stateKey != "" {
		if err := ctx.State().Set(p.stateKey, p.stateVal); err != nil {
			return nil, err
		}
	}
	return p.parts, p.processErr
}

func (p *rewritingPlanner) callCount() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.calls
}

// overridingPlanner embeds *planner.BuiltInPlanner and adds its own response
// step, as a Python subclass of BuiltInPlanner can override
// process_planning_response.
type overridingPlanner struct {
	*planner.BuiltInPlanner
}

func (p overridingPlanner) ProcessPlanningResponse(agent.Context, []*genai.Part) ([]*genai.Part, error) {
	return []*genai.Part{{Text: "rewritten by subclass"}}, nil
}

type plannerRun struct {
	events []*session.Event
	err    error
	// sess is the stored session, read back after the run.
	sess session.Session
}

// runPlannerAgent runs a once on a new in-memory session that holds history,
// and collects the events until the first error.
func runPlannerAgent(t *testing.T, a agent.Agent, history ...*session.Event) plannerRun {
	t.Helper()
	ctx := t.Context()
	svc := session.InMemoryService()
	key := session.GetRequest{AppName: "test_app", UserID: "test_user", SessionID: "s1"}
	created, err := svc.Create(ctx, &session.CreateRequest{AppName: key.AppName, UserID: key.UserID, SessionID: key.SessionID})
	if err != nil {
		t.Fatalf("session Create: %v", err)
	}
	for _, ev := range history {
		if err := svc.AppendEvent(ctx, created.Session, ev); err != nil {
			t.Fatalf("session AppendEvent: %v", err)
		}
	}
	r, err := runner.New(runner.Config{AppName: key.AppName, Agent: a, SessionService: svc})
	if err != nil {
		t.Fatalf("runner.New: %v", err)
	}
	var run plannerRun
	for ev, err := range r.Run(ctx, key.UserID, key.SessionID, genai.NewContentFromText("go", genai.RoleUser), agent.RunConfig{}) {
		if err != nil {
			run.err = err
			break
		}
		run.events = append(run.events, ev)
	}
	got, err := svc.Get(ctx, &key)
	if err != nil {
		t.Fatalf("session Get: %v", err)
	}
	run.sess = got.Session
	return run
}

// thoughtHistory is a user turn followed by a model turn of test_agent that
// holds a thought part and a plain part.
func thoughtHistory(t *testing.T) []*session.Event {
	t.Helper()
	user := session.NewEvent(t.Context(), "inv-old")
	user.Author = "user"
	user.LLMResponse = model.LLMResponse{Content: genai.NewContentFromText("earlier question", genai.RoleUser)}
	reply := session.NewEvent(t.Context(), "inv-old")
	reply.Author = "test_agent"
	reply.LLMResponse = model.LLMResponse{Content: &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{
		{Text: "stored thought", Thought: true},
		{Text: "stored plain"},
	}}}
	return []*session.Event{user, reply}
}

func newPlannerAgent(t *testing.T, cfg llmagent.Config) agent.Agent {
	t.Helper()
	a, err := llmagent.New(cfg)
	if err != nil {
		t.Fatalf("llmagent.New: %v", err)
	}
	return a
}

func systemInstructionText(t *testing.T, req *model.LLMRequest) string {
	t.Helper()
	if req.Config == nil || req.Config.SystemInstruction == nil || len(req.Config.SystemInstruction.Parts) == 0 {
		t.Fatalf("request has no system instruction")
	}
	return req.Config.SystemInstruction.Parts[0].Text
}

func modelEventParts(t *testing.T, events []*session.Event) []*genai.Part {
	t.Helper()
	if len(events) == 0 {
		t.Fatalf("run yielded no event")
	}
	last := events[len(events)-1]
	if last.LLMResponse.Content == nil {
		t.Fatalf("last event has no content")
	}
	return last.LLMResponse.Content.Parts
}

func hasText(events []*session.Event, text string) bool {
	for _, ev := range events {
		if ev.LLMResponse.Content == nil {
			continue
		}
		for _, p := range ev.LLMResponse.Content.Parts {
			if p.Text == text {
				return true
			}
		}
	}
	return false
}

func TestLLMAgent_PlannerInstructionEmptyAndMultiStep(t *testing.T) {
	t.Run("empty_instruction_noop", func(t *testing.T) {
		llm := &scriptedModel{responses: []*model.LLMResponse{textResponse("done")}}
		a := newPlannerAgent(t, llmagent.Config{
			Name:        "test_agent",
			Model:       llm,
			Instruction: "Base.",
			Planner:     &instructionPlanner{},
		})

		run := runPlannerAgent(t, a)

		if run.err != nil {
			t.Fatalf("run error: %v", run.err)
		}
		reqs := llm.recorded()
		if len(reqs) != 1 {
			t.Fatalf("model calls = %d, want 1", len(reqs))
		}
		want := "Base.\n\nYou are an agent. Your internal name is \"test_agent\"."
		if got := systemInstructionText(t, reqs[0]); got != want {
			t.Errorf("system instruction = %q, want %q", got, want)
		}
	})

	t.Run("appended_once_per_model_call_across_tool_turn", func(t *testing.T) {
		noop, err := functiontool.New(functiontool.Config{Name: "noop", Description: "does nothing"},
			func(agent.Context, struct{}) (struct{}, error) { return struct{}{}, nil })
		if err != nil {
			t.Fatalf("functiontool.New: %v", err)
		}
		llm := &scriptedModel{responses: []*model.LLMResponse{
			{Content: &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{{
				FunctionCall: &genai.FunctionCall{ID: "fc-1", Name: "noop", Args: map[string]any{}},
			}}}},
			textResponse("done"),
		}}
		a := newPlannerAgent(t, llmagent.Config{
			Name:    "test_agent",
			Model:   llm,
			Tools:   []tool.Tool{noop},
			Planner: &instructionPlanner{instruction: "Plan instruction."},
		})

		run := runPlannerAgent(t, a)

		if run.err != nil {
			t.Fatalf("run error: %v", run.err)
		}
		reqs := llm.recorded()
		if len(reqs) != 2 {
			t.Fatalf("model calls = %d, want 2", len(reqs))
		}
		want := "You are an agent. Your internal name is \"test_agent\".\n\nPlan instruction."
		for i, req := range reqs {
			got := systemInstructionText(t, req)
			if got != want {
				t.Errorf("call %d: system instruction = %q, want %q", i+1, got, want)
			}
			if n := strings.Count(got, "Plan instruction."); n != 1 {
				t.Errorf("call %d: planning instruction appears %d times, want 1", i+1, n)
			}
		}
	})
}

func TestLLMAgent_PlannerRemovesRequestThoughtsPreservingSessionHistory(t *testing.T) {
	llm := &scriptedModel{responses: []*model.LLMResponse{textResponse("done")}}
	a := newPlannerAgent(t, llmagent.Config{
		Name:    "test_agent",
		Model:   llm,
		Planner: &instructionPlanner{},
	})

	run := runPlannerAgent(t, a, thoughtHistory(t)...)

	if run.err != nil {
		t.Fatalf("run error: %v", run.err)
	}
	reqs := llm.recorded()
	if len(reqs) != 1 {
		t.Fatalf("model calls = %d, want 1", len(reqs))
	}
	wantSent := []*genai.Part{{Text: "stored thought"}, {Text: "stored plain"}}
	if diff := cmp.Diff(wantSent, reqs[0].Contents[1].Parts); diff != "" {
		t.Errorf("request history parts mismatch (-want +got):\n%s", diff)
	}
	stored := run.sess.Events().At(1).LLMResponse.Content.Parts[0]
	if !stored.Thought {
		t.Errorf("stored history part Thought = false, want true")
	}
}

func TestLLMAgent_BuiltInPlannerThinkingConfigPrecedenceAndNoRewrite(t *testing.T) {
	t.Run("overrides_agent_thinking_config_and_preserves_thoughts", func(t *testing.T) {
		llm := &scriptedModel{responses: []*model.LLMResponse{textResponse("/*PLANNING*/raw")}}
		a := newPlannerAgent(t, llmagent.Config{
			Name:  "test_agent",
			Model: llm,
			GenerateContentConfig: &genai.GenerateContentConfig{
				ThinkingConfig: &genai.ThinkingConfig{ThinkingBudget: genai.Ptr[int32](128)},
			},
			Planner: planner.NewBuiltInPlanner(&genai.ThinkingConfig{IncludeThoughts: true, ThinkingBudget: genai.Ptr[int32](1024)}),
		})

		run := runPlannerAgent(t, a, thoughtHistory(t)...)

		if run.err != nil {
			t.Fatalf("run error: %v", run.err)
		}
		reqs := llm.recorded()
		if len(reqs) != 1 {
			t.Fatalf("model calls = %d, want 1", len(reqs))
		}
		wantThinking := &genai.ThinkingConfig{IncludeThoughts: true, ThinkingBudget: genai.Ptr[int32](1024)}
		if diff := cmp.Diff(wantThinking, reqs[0].Config.ThinkingConfig); diff != "" {
			t.Errorf("request ThinkingConfig mismatch (-want +got):\n%s", diff)
		}
		if !reqs[0].Contents[1].Parts[0].Thought {
			t.Errorf("request history part Thought = false, want true")
		}
		wantInst := "You are an agent. Your internal name is \"test_agent\"."
		if got := systemInstructionText(t, reqs[0]); got != wantInst {
			t.Errorf("system instruction = %q, want %q", got, wantInst)
		}
		wantParts := []*genai.Part{{Text: "/*PLANNING*/raw"}}
		if diff := cmp.Diff(wantParts, modelEventParts(t, run.events)); diff != "" {
			t.Errorf("model event parts mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("nil_generate_content_config", func(t *testing.T) {
		llm := &scriptedModel{responses: []*model.LLMResponse{textResponse("done")}}
		a := newPlannerAgent(t, llmagent.Config{
			Name:    "test_agent",
			Model:   llm,
			Planner: planner.NewBuiltInPlanner(&genai.ThinkingConfig{IncludeThoughts: true}),
		})

		run := runPlannerAgent(t, a)

		if run.err != nil {
			t.Fatalf("run error: %v", run.err)
		}
		reqs := llm.recorded()
		if len(reqs) != 1 {
			t.Fatalf("model calls = %d, want 1", len(reqs))
		}
		if diff := cmp.Diff(&genai.ThinkingConfig{IncludeThoughts: true}, reqs[0].Config.ThinkingConfig); diff != "" {
			t.Errorf("request ThinkingConfig mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("apply_thinking_config_allocates_nil_request_config", func(t *testing.T) {
		req := &model.LLMRequest{}

		planner.NewBuiltInPlanner(&genai.ThinkingConfig{IncludeThoughts: true}).ApplyThinkingConfig(req)

		want := &genai.GenerateContentConfig{ThinkingConfig: &genai.ThinkingConfig{IncludeThoughts: true}}
		if diff := cmp.Diff(want, req.Config); diff != "" {
			t.Errorf("request Config mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("nil_planner_thinking_config_preserves_agent_config", func(t *testing.T) {
		llm := &scriptedModel{responses: []*model.LLMResponse{textResponse("done")}}
		a := newPlannerAgent(t, llmagent.Config{
			Name:  "test_agent",
			Model: llm,
			GenerateContentConfig: &genai.GenerateContentConfig{
				ThinkingConfig: &genai.ThinkingConfig{ThinkingBudget: genai.Ptr[int32](256)},
			},
			Planner: planner.NewBuiltInPlanner(nil),
		})

		run := runPlannerAgent(t, a)

		if run.err != nil {
			t.Fatalf("run error: %v", run.err)
		}
		reqs := llm.recorded()
		if len(reqs) != 1 {
			t.Fatalf("model calls = %d, want 1", len(reqs))
		}
		if diff := cmp.Diff(&genai.ThinkingConfig{ThinkingBudget: genai.Ptr[int32](256)}, reqs[0].Config.ThinkingConfig); diff != "" {
			t.Errorf("request ThinkingConfig mismatch (-want +got):\n%s", diff)
		}
	})
}

func TestLLMAgent_RewritingPlannerEmitsSeparateStateEventBeforeModelEvent(t *testing.T) {
	llm := &scriptedModel{responses: []*model.LLMResponse{textResponse("raw answer")}}
	a := newPlannerAgent(t, llmagent.Config{
		Name:  "planner_agent",
		Model: llm,
		Planner: &rewritingPlanner{
			stateKey: "plan_key",
			stateVal: "plan_val",
			parts:    []*genai.Part{{Text: "clean answer"}},
		},
	})
	// A parallel parent gives the agent a non-empty branch, so the test sees
	// whether the state event carries it.
	parent, err := parallelagent.New(parallelagent.Config{AgentConfig: agent.Config{Name: "parallel", SubAgents: []agent.Agent{a}}})
	if err != nil {
		t.Fatalf("parallelagent.New: %v", err)
	}

	run := runPlannerAgent(t, parent)

	if run.err != nil {
		t.Fatalf("run error: %v", run.err)
	}
	if len(run.events) != 2 {
		t.Fatalf("run yielded %d events, want 2", len(run.events))
	}
	stateEv, modelEv := run.events[0], run.events[1]
	if stateEv.Author != "planner_agent" {
		t.Errorf("state event Author = %q, want %q", stateEv.Author, "planner_agent")
	}
	if stateEv.ID == "" {
		t.Errorf("state event ID is empty")
	}
	if stateEv.InvocationID == "" || stateEv.InvocationID != modelEv.InvocationID {
		t.Errorf("state event InvocationID = %q, model event InvocationID = %q, want equal and non-empty", stateEv.InvocationID, modelEv.InvocationID)
	}
	if stateEv.Branch != "parallel.planner_agent" || modelEv.Branch != "parallel.planner_agent" {
		t.Errorf("state event Branch = %q, model event Branch = %q, want both %q", stateEv.Branch, modelEv.Branch, "parallel.planner_agent")
	}
	if stateEv.LLMResponse.Content != nil {
		t.Errorf("state event Content = %v, want nil", stateEv.LLMResponse.Content)
	}
	if diff := cmp.Diff(map[string]any{"plan_key": "plan_val"}, stateEv.Actions.StateDelta); diff != "" {
		t.Errorf("state event StateDelta mismatch (-want +got):\n%s", diff)
	}
	if modelEv.Author != "planner_agent" {
		t.Errorf("model event Author = %q, want %q", modelEv.Author, "planner_agent")
	}
	if diff := cmp.Diff([]*genai.Part{{Text: "clean answer"}}, modelEv.LLMResponse.Content.Parts); diff != "" {
		t.Errorf("model event parts mismatch (-want +got):\n%s", diff)
	}
	if len(modelEv.Actions.StateDelta) != 0 {
		t.Errorf("model event StateDelta = %v, want empty", modelEv.Actions.StateDelta)
	}
	got, err := run.sess.State().Get("plan_key")
	if err != nil || got != "plan_val" {
		t.Errorf("session state plan_key = %v (err %v), want %q", got, err, "plan_val")
	}
}

func TestLLMAgent_RewritingPlannerEdgeCasesAndBuiltInOverride(t *testing.T) {
	t.Run("empty_slice_keeps_original_parts_F3", func(t *testing.T) {
		llm := &scriptedModel{responses: []*model.LLMResponse{textResponse("kept original")}}
		p := &rewritingPlanner{parts: []*genai.Part{}}
		a := newPlannerAgent(t, llmagent.Config{Name: "test_agent", Model: llm, Planner: p})

		run := runPlannerAgent(t, a)

		if run.err != nil {
			t.Fatalf("run error: %v", run.err)
		}
		if p.callCount() != 1 {
			t.Fatalf("ProcessPlanningResponse calls = %d, want 1", p.callCount())
		}
		if len(run.events) != 1 {
			t.Fatalf("run yielded %d events, want 1", len(run.events))
		}
		if diff := cmp.Diff([]*genai.Part{{Text: "kept original"}}, modelEventParts(t, run.events)); diff != "" {
			t.Errorf("model event parts mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("empty_model_response_skips_planner", func(t *testing.T) {
		llm := &scriptedModel{responses: []*model.LLMResponse{{Content: nil, ErrorCode: "BLOCKED"}}}
		p := &rewritingPlanner{parts: []*genai.Part{{Text: "unused"}}}
		a := newPlannerAgent(t, llmagent.Config{Name: "test_agent", Model: llm, Planner: p})

		run := runPlannerAgent(t, a)

		if run.err != nil {
			t.Fatalf("run error: %v", run.err)
		}
		if n := len(llm.recorded()); n != 1 {
			t.Fatalf("model calls = %d, want 1", n)
		}
		if p.callCount() != 0 {
			t.Errorf("ProcessPlanningResponse calls = %d, want 0", p.callCount())
		}
	})

	t.Run("response_without_parts_skips_planner", func(t *testing.T) {
		llm := &scriptedModel{responses: []*model.LLMResponse{{
			Content:      &genai.Content{Role: genai.RoleModel},
			FinishReason: genai.FinishReasonStop,
		}}}
		p := &rewritingPlanner{parts: []*genai.Part{{Text: "must not be called"}}}
		a := newPlannerAgent(t, llmagent.Config{Name: "test_agent", Model: llm, Planner: p})

		run := runPlannerAgent(t, a)

		if run.err != nil {
			t.Fatalf("run error: %v", run.err)
		}
		if p.callCount() != 0 {
			t.Errorf("ProcessPlanningResponse calls = %d, want 0", p.callCount())
		}
	})

	t.Run("embedded_builtin_planner_with_override_F4", func(t *testing.T) {
		llm := &scriptedModel{responses: []*model.LLMResponse{textResponse("raw")}}
		a := newPlannerAgent(t, llmagent.Config{
			Name:    "test_agent",
			Model:   llm,
			Planner: overridingPlanner{planner.NewBuiltInPlanner(&genai.ThinkingConfig{ThinkingBudget: genai.Ptr[int32](512)})},
		})

		run := runPlannerAgent(t, a)

		if run.err != nil {
			t.Fatalf("run error: %v", run.err)
		}
		reqs := llm.recorded()
		if len(reqs) != 1 {
			t.Fatalf("model calls = %d, want 1", len(reqs))
		}
		if diff := cmp.Diff(&genai.ThinkingConfig{ThinkingBudget: genai.Ptr[int32](512)}, reqs[0].Config.ThinkingConfig); diff != "" {
			t.Errorf("request ThinkingConfig mismatch (-want +got):\n%s", diff)
		}
		if diff := cmp.Diff([]*genai.Part{{Text: "rewritten by subclass"}}, modelEventParts(t, run.events)); diff != "" {
			t.Errorf("model event parts mismatch (-want +got):\n%s", diff)
		}
	})
}

func TestLLMAgent_PlanReActThoughtOnlyTurnEndsStepWithFinishReasonStop(t *testing.T) {
	t.Run("plan_re_act_planner_ends_step", func(t *testing.T) {
		llm := &scriptedModel{responses: []*model.LLMResponse{{
			Content:      &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{{Text: "/*PLANNING*/Step 1"}}},
			FinishReason: genai.FinishReasonStop,
		}}}
		a := newPlannerAgent(t, llmagent.Config{
			Name:    "test_agent",
			Model:   llm,
			Planner: planner.NewPlanReActPlanner(),
		})

		run := runPlannerAgent(t, a)

		if run.err != nil {
			t.Fatalf("run error: %v", run.err)
		}
		if n := len(llm.recorded()); n != 1 {
			t.Errorf("model calls = %d, want 1", n)
		}
		if len(run.events) != 1 {
			t.Fatalf("run yielded %d events, want 1", len(run.events))
		}
		if diff := cmp.Diff([]*genai.Part{{Text: "Step 1", Thought: true}}, modelEventParts(t, run.events)); diff != "" {
			t.Errorf("model event parts mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("instruction_only_planner_keeps_recall", func(t *testing.T) {
		llm := &scriptedModel{responses: []*model.LLMResponse{
			{Content: &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{{Text: "thinking", Thought: true}}}, FinishReason: genai.FinishReasonStop},
			textResponse("answer"),
		}}
		a := newPlannerAgent(t, llmagent.Config{Name: "test_agent", Model: llm, Planner: &instructionPlanner{}})

		run := runPlannerAgent(t, a)

		if run.err != nil {
			t.Fatalf("run error: %v", run.err)
		}
		if n := len(llm.recorded()); n != 2 {
			t.Errorf("model calls = %d, want 2", n)
		}
		if len(run.events) != 2 || !hasText(run.events[1:], "answer") {
			t.Errorf("yielded %d events, want 2 ending with the answer", len(run.events))
		}
	})
}

func TestLLMAgent_PlannerBuildInstructionErrorStopsRun(t *testing.T) {
	errPlan := errors.New("plan failed")
	llm := &scriptedModel{responses: []*model.LLMResponse{textResponse("unused")}}
	a := newPlannerAgent(t, llmagent.Config{
		Name:    "test_agent",
		Model:   llm,
		Planner: &instructionPlanner{err: errPlan},
	})

	run := runPlannerAgent(t, a)

	if !errors.Is(run.err, errPlan) {
		t.Errorf("run error = %v, want one wrapping %v", run.err, errPlan)
	}
	if n := len(llm.recorded()); n != 0 {
		t.Errorf("model calls = %d, want 0", n)
	}
	if hasText(run.events, "unused") {
		t.Errorf("run yielded the model response, want none")
	}
}

func TestLLMAgent_RewritingPlannerProcessResponseErrorStopsRun(t *testing.T) {
	errRewrite := errors.New("rewrite failed")
	llm := &scriptedModel{responses: []*model.LLMResponse{textResponse("raw answer")}}
	a := newPlannerAgent(t, llmagent.Config{
		Name:  "planner_agent",
		Model: llm,
		Planner: &rewritingPlanner{
			stateKey:   "plan_key",
			stateVal:   "plan_val",
			processErr: errRewrite,
		},
	})

	run := runPlannerAgent(t, a)

	if !errors.Is(run.err, errRewrite) {
		t.Errorf("run error = %v, want one wrapping %v", run.err, errRewrite)
	}
	for _, ev := range run.events {
		if _, ok := ev.Actions.StateDelta["plan_key"]; ok {
			t.Errorf("run yielded an event with state delta plan_key, want none")
		}
	}
	if hasText(run.events, "raw answer") {
		t.Errorf("run yielded the model response, want none")
	}
}
