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
	"iter"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/internal/workflowinternal"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
)

// TestDispatchTaskFC_IsolationScope exercises the full
// chat-coordinator → task-delegation pipeline: a chat-mode root
// LlmAgent calls the TaskAgentTool that wraps a task-mode sub-agent,
// and we assert the sub-agent's emitted events carry the FC id as
// their IsolationScope.
func TestDispatchTaskFC_IsolationScope(t *testing.T) {
	t.Parallel()

	const (
		coordName = "coordinator"
		taskName  = "doer"
		fcID      = "fc-delegation-1"
	)

	// Task sub-agent: a task-mode LlmAgent driven by a scripted LLM
	// that immediately calls finish_task("done") and exits.
	taskLLM := &scriptedLLM{
		turns: []*model.LLMResponse{
			{
				Content: &genai.Content{
					Role: "model",
					Parts: []*genai.Part{{
						FunctionCall: &genai.FunctionCall{
							ID:   "finish-1",
							Name: "finish_task",
							Args: map[string]any{"result": "done"},
						},
					}},
				},
			},
		},
	}
	taskAgent, err := llmagent.New(llmagent.Config{
		Name:        taskName,
		Description: "Solves a delegated task.",
		Model:       taskLLM,
		Mode:        llmagent.ModeTask,
	})
	if err != nil {
		t.Fatalf("llmagent.New(task): %v", err)
	}

	// Coordinator: a chat-mode LlmAgent that on its first turn emits
	// a TaskAgentTool FC to delegate to the task sub-agent, then on
	// its second turn (after the wrapper synthesises the task's FR)
	// emits a final text "delegation complete" and exits.
	coordLLM := &scriptedLLM{
		turns: []*model.LLMResponse{
			{
				Content: &genai.Content{
					Role: "model",
					Parts: []*genai.Part{{
						FunctionCall: &genai.FunctionCall{
							ID:   fcID,
							Name: taskName,
							Args: map[string]any{"request": "do the thing"},
						},
					}},
				},
			},
			{
				Content: &genai.Content{
					Role:  "model",
					Parts: []*genai.Part{genai.NewPartFromText("delegation complete")},
				},
			},
		},
	}
	coord, err := llmagent.New(llmagent.Config{
		Name:      coordName,
		Model:     coordLLM,
		Mode:      llmagent.ModeChat,
		SubAgents: []agent.Agent{taskAgent},
	})
	if err != nil {
		t.Fatalf("llmagent.New(coord): %v", err)
	}

	// Runner setup.
	svc := session.InMemoryService()
	if _, err := svc.Create(t.Context(), &session.CreateRequest{
		AppName: "app", UserID: "u", SessionID: "s",
	}); err != nil {
		t.Fatalf("session.Create: %v", err)
	}
	r, err := runner.New(runner.Config{Agent: coord, SessionService: svc, AppName: "app"})
	if err != nil {
		t.Fatalf("runner.New: %v", err)
	}

	// Drive one user turn.
	events := []*session.Event{}
	for ev, err := range r.Run(t.Context(), "u", "s",
		&genai.Content{Parts: []*genai.Part{genai.NewPartFromText("go")}, Role: "user"},
		agent.RunConfig{StreamingMode: agent.StreamingModeSSE}) {
		if err != nil {
			t.Fatalf("runner.Run: %v", err)
		}
		events = append(events, ev)
	}

	// We require that AT LEAST ONE event authored by the task agent
	// carries IsolationScope == fcID. Without the wiring in
	// dispatchTaskFC, the task agent would inherit the coordinator's
	// scope (empty), so this assertion would fail.
	//
	// We do NOT pin the exact event count or shape — the test is
	// scoped to the IsolationScope invariant; the wider chat-loop
	// behavior is covered by other tests.
	var seenTaskEventInScope bool
	var taskEventScopes []string // for failure diagnostics
	for _, ev := range events {
		if ev == nil || ev.Author != taskName {
			continue
		}
		taskEventScopes = append(taskEventScopes, ev.IsolationScope)
		if ev.IsolationScope == fcID {
			seenTaskEventInScope = true
		}
	}
	if !seenTaskEventInScope {
		t.Errorf("expected at least one event authored by task agent %q to carry "+
			"IsolationScope == %q (i.e. dispatchTaskFC passes WithIsolationScope(fc.ID)); "+
			"observed task-event scopes: %v",
			taskName, fcID, taskEventScopes)
	}
}

// scriptedLLM is a model.LLM that yields a fixed sequence of
// LLMResponses, one per call to GenerateContent. After exhausting the
// script, subsequent calls yield a terminal "done" text — this lets
// the runner's outer turn loop exit cleanly without hanging.
type scriptedLLM struct {
	turns    []*model.LLMResponse
	callIdx  int
	doneText string // override for the post-script fallback
}

func (s *scriptedLLM) Name() string { return "scripted-mock" }

func (s *scriptedLLM) GenerateContent(_ context.Context, _ *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	idx := s.callIdx
	s.callIdx++
	return func(yield func(*model.LLMResponse, error) bool) {
		if idx < len(s.turns) {
			yield(s.turns[idx], nil)
			return
		}
		// Post-script terminal turn so the runner exits cleanly even
		// if more LLM calls happen than scripted.
		text := s.doneText
		if text == "" {
			text = "done"
		}
		yield(&model.LLMResponse{
			Content: &genai.Content{
				Role:  "model",
				Parts: []*genai.Part{genai.NewPartFromText(text)},
			},
		}, nil)
	}
}

var _ model.LLM = (*scriptedLLM)(nil)

func newStubNodeContext(t *testing.T, a agent.Agent, isolationScope string) agent.Context {
	t.Helper()
	ic := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{
		Agent:          a,
		IsolationScope: isolationScope,
		InvocationID:   "inv-test",
	})
	return agent.NewContext(ic)
}

func makeLLMAgent(t *testing.T, name string, opts ...func(*llmagent.Config)) agent.Agent {
	t.Helper()
	cfg := llmagent.Config{
		Name:        name,
		Description: "test agent",
		Model:       &scriptedLLM{},
	}
	for _, opt := range opts {
		opt(&cfg)
	}
	a, err := llmagent.New(cfg)
	if err != nil {
		t.Fatalf("llmagent.New(%q): %v", name, err)
	}
	return a
}

func withMode(m llmagent.Mode) func(*llmagent.Config) {
	return func(c *llmagent.Config) { c.Mode = m }
}

func withOutputSchema(s *genai.Schema) func(*llmagent.Config) {
	return func(c *llmagent.Config) { c.OutputSchema = s }
}

func withOutputKey(k string) func(*llmagent.Config) {
	return func(c *llmagent.Config) { c.OutputKey = k }
}

// =============================================================================
// llmagent.PrepareLLMAgentInput
// =============================================================================

// TestPrepareLLMAgentInput pins the seeded-input contract: only
// single_turn agents with a non-nil nodeInput produce a seed event;
// everything else returns nil.
func TestPrepareLLMAgentInput(t *testing.T) {
	t.Parallel()

	t.Run("nil nodeInput returns nil", func(t *testing.T) {
		t.Parallel()
		a := makeLLMAgent(t, "x", withMode(llmagent.ModeSingleTurn))
		ctx := newStubNodeContext(t, a, "")
		if got := llmagent.PrepareLLMAgentInput(a, ctx, nil); got != nil {
			t.Errorf("llmagent.PrepareLLMAgentInput(nil) = %v, want nil", got)
		}
	})

	t.Run("non-LlmAgent returns nil", func(t *testing.T) {
		t.Parallel()
		a, err := agent.New(agent.Config{Name: "plain"})
		if err != nil {
			t.Fatal(err)
		}
		ctx := newStubNodeContext(t, a, "")
		if got := llmagent.PrepareLLMAgentInput(a, ctx, "ignored"); got != nil {
			t.Errorf("llmagent.PrepareLLMAgentInput on non-LlmAgent = %v, want nil", got)
		}
	})

	t.Run("chat mode returns nil (no seeding)", func(t *testing.T) {
		t.Parallel()
		a := makeLLMAgent(t, "chat", withMode(llmagent.ModeChat))
		ctx := newStubNodeContext(t, a, "")
		if got := llmagent.PrepareLLMAgentInput(a, ctx, "input"); got != nil {
			t.Errorf("llmagent.PrepareLLMAgentInput on chat-mode = %v, want nil", got)
		}
	})

	t.Run("task mode returns nil (FC-driven, no seeding)", func(t *testing.T) {
		t.Parallel()
		a := makeLLMAgent(t, "task", withMode(llmagent.ModeTask))
		ctx := newStubNodeContext(t, a, "")
		if got := llmagent.PrepareLLMAgentInput(a, ctx, "input"); got != nil {
			t.Errorf("llmagent.PrepareLLMAgentInput on task-mode = %v, want nil", got)
		}
	})

	t.Run("single_turn + string input yields user-role text part", func(t *testing.T) {
		t.Parallel()
		a := makeLLMAgent(t, "st", withMode(llmagent.ModeSingleTurn))
		ctx := newStubNodeContext(t, a, "")
		got := llmagent.PrepareLLMAgentInput(a, ctx, "hello there")
		if got == nil {
			t.Fatal("llmagent.PrepareLLMAgentInput returned nil; want non-nil event")
		}
		if got.Author != "user" {
			t.Errorf("Author = %q, want %q", got.Author, "user")
		}
		if got.Content == nil || got.Content.Role != genai.RoleUser {
			t.Fatalf("Content/Role mismatch: %+v", got.Content)
		}
		if len(got.Content.Parts) != 1 || got.Content.Parts[0].Text != "hello there" {
			t.Errorf("Parts = %+v, want one text part 'hello there'", got.Content.Parts)
		}
	})

	t.Run("single_turn + *genai.Content reuses parts and forces role=user", func(t *testing.T) {
		t.Parallel()
		a := makeLLMAgent(t, "st", withMode(llmagent.ModeSingleTurn))
		ctx := newStubNodeContext(t, a, "")
		input := &genai.Content{
			Role:  "model", // deliberately wrong; must be forced to user
			Parts: []*genai.Part{{Text: "part one"}, {Text: "part two"}},
		}
		got := llmagent.PrepareLLMAgentInput(a, ctx, input)
		if got == nil {
			t.Fatal("expected non-nil event")
		}
		if got.Content.Role != genai.RoleUser {
			t.Errorf("Role = %q, want %q (must be forced to user)", got.Content.Role, genai.RoleUser)
		}
		if diff := cmp.Diff(input.Parts, got.Content.Parts); diff != "" {
			t.Errorf("Parts mismatch (-input +got):\n%s", diff)
		}
	})

	t.Run("single_turn + struct input is JSON-marshalled", func(t *testing.T) {
		t.Parallel()
		a := makeLLMAgent(t, "st", withMode(llmagent.ModeSingleTurn))
		ctx := newStubNodeContext(t, a, "")
		input := map[string]any{"task": "summarize", "limit": 5}
		got := llmagent.PrepareLLMAgentInput(a, ctx, input)
		if got == nil {
			t.Fatal("expected non-nil event")
		}
		if len(got.Content.Parts) != 1 {
			t.Fatalf("Parts count = %d, want 1", len(got.Content.Parts))
		}
		text := got.Content.Parts[0].Text
		// JSON map iteration order is non-deterministic; check both keys present.
		if !strings.Contains(text, `"task":"summarize"`) ||
			!strings.Contains(text, `"limit":5`) {
			t.Errorf("text %q does not look like marshalled %+v", text, input)
		}
	})

	t.Run("single_turn + IsolationScope on ctx propagates to event", func(t *testing.T) {
		t.Parallel()
		a := makeLLMAgent(t, "st", withMode(llmagent.ModeSingleTurn))
		const scope = "iso-scope-xyz"
		ctx := newStubNodeContext(t, a, scope)
		got := llmagent.PrepareLLMAgentInput(a, ctx, "hello")
		if got == nil {
			t.Fatal("expected non-nil event")
		}
		if got.IsolationScope != scope {
			t.Errorf("IsolationScope = %q, want %q", got.IsolationScope, scope)
		}
	})

	t.Run("single_turn + empty IsolationScope leaves event scope empty", func(t *testing.T) {
		t.Parallel()
		a := makeLLMAgent(t, "st", withMode(llmagent.ModeSingleTurn))
		ctx := newStubNodeContext(t, a, "")
		got := llmagent.PrepareLLMAgentInput(a, ctx, "hello")
		if got == nil {
			t.Fatal("expected non-nil event")
		}
		if got.IsolationScope != "" {
			t.Errorf("IsolationScope = %q, want empty", got.IsolationScope)
		}
	})
}

func TestProcessLLMAgentOutput(t *testing.T) {
	t.Parallel()

	t.Run("nil event is no-op", func(t *testing.T) {
		t.Parallel()
		a := makeLLMAgent(t, "x")
		if err := llmagent.ProcessLLMAgentOutput(a, nil); err != nil {
			t.Errorf("err = %v, want nil", err)
		}
	})

	t.Run("event with FunctionCall is skipped", func(t *testing.T) {
		t.Parallel()
		a := makeLLMAgent(t, "x")
		ev := &session.Event{
			LLMResponse: model.LLMResponse{
				Content: &genai.Content{
					Role: "model",
					Parts: []*genai.Part{{
						FunctionCall: &genai.FunctionCall{ID: "fc-1", Name: "f"},
					}},
				},
			},
		}
		if err := llmagent.ProcessLLMAgentOutput(a, ev); err != nil {
			t.Fatal(err)
		}
		if ev.Output != nil {
			t.Errorf("Output = %v on FC event, want nil (skipped)", ev.Output)
		}
		if ev.NodeInfo != nil && ev.NodeInfo.MessageAsOutput {
			t.Errorf("MessageAsOutput = true on FC event, want unset")
		}
	})

	t.Run("partial event is skipped", func(t *testing.T) {
		t.Parallel()
		a := makeLLMAgent(t, "x")
		ev := &session.Event{
			LLMResponse: model.LLMResponse{
				Partial: true,
				Content: &genai.Content{
					Role:  "model",
					Parts: []*genai.Part{{Text: "partial"}},
				},
			},
		}
		if err := llmagent.ProcessLLMAgentOutput(a, ev); err != nil {
			t.Fatal(err)
		}
		if ev.Output != nil {
			t.Errorf("Output = %v on partial event, want nil", ev.Output)
		}
	})

	t.Run("non-model role is skipped", func(t *testing.T) {
		t.Parallel()
		a := makeLLMAgent(t, "x")
		ev := &session.Event{
			LLMResponse: model.LLMResponse{
				Content: &genai.Content{
					Role:  genai.RoleUser,
					Parts: []*genai.Part{{Text: "from user"}},
				},
			},
		}
		if err := llmagent.ProcessLLMAgentOutput(a, ev); err != nil {
			t.Fatal(err)
		}
		if ev.Output != nil {
			t.Errorf("Output = %v on non-model event, want nil", ev.Output)
		}
	})

	t.Run("non-LlmAgent is skipped", func(t *testing.T) {
		t.Parallel()
		a, err := agent.New(agent.Config{Name: "plain"})
		if err != nil {
			t.Fatal(err)
		}
		ev := &session.Event{
			LLMResponse: model.LLMResponse{
				Content: &genai.Content{Role: "model", Parts: []*genai.Part{{Text: "hi"}}},
			},
		}
		if err := llmagent.ProcessLLMAgentOutput(a, ev); err != nil {
			t.Fatal(err)
		}
		if ev.Output != nil {
			t.Errorf("Output = %v on non-LlmAgent event, want nil", ev.Output)
		}
	})

	t.Run("plain text reply: Output=text, MessageAsOutput=true, no state_delta", func(t *testing.T) {
		t.Parallel()
		a := makeLLMAgent(t, "x")
		ev := &session.Event{
			LLMResponse: model.LLMResponse{
				Content: &genai.Content{Role: "model", Parts: []*genai.Part{{Text: "the answer"}}},
			},
		}
		if err := llmagent.ProcessLLMAgentOutput(a, ev); err != nil {
			t.Fatal(err)
		}
		if got, want := ev.Output, any("the answer"); got != want {
			t.Errorf("Output = %v, want %v", got, want)
		}
		if ev.NodeInfo == nil || !ev.NodeInfo.MessageAsOutput {
			t.Errorf("MessageAsOutput not set; got NodeInfo = %+v", ev.NodeInfo)
		}
		if len(ev.Actions.StateDelta) != 0 {
			t.Errorf("StateDelta = %v, want empty (no OutputKey configured)", ev.Actions.StateDelta)
		}
	})

	t.Run("multi-part text concatenation skips Thought parts", func(t *testing.T) {
		t.Parallel()
		a := makeLLMAgent(t, "x")
		ev := &session.Event{
			LLMResponse: model.LLMResponse{
				Content: &genai.Content{
					Role: "model",
					Parts: []*genai.Part{
						{Text: "thinking out loud", Thought: true},
						{Text: "real answer "},
						{Text: "continued"},
					},
				},
			},
		}
		if err := llmagent.ProcessLLMAgentOutput(a, ev); err != nil {
			t.Fatal(err)
		}
		if got, want := ev.Output, any("real answer continued"); got != want {
			t.Errorf("Output = %v, want %v (thought parts must be excluded)", got, want)
		}
	})

	t.Run("OutputKey populates Actions.StateDelta", func(t *testing.T) {
		t.Parallel()
		const key = "my_key"
		a := makeLLMAgent(t, "x", withOutputKey(key))
		ev := &session.Event{
			LLMResponse: model.LLMResponse{
				Content: &genai.Content{Role: "model", Parts: []*genai.Part{{Text: "value"}}},
			},
		}
		if err := llmagent.ProcessLLMAgentOutput(a, ev); err != nil {
			t.Fatal(err)
		}
		if got, want := ev.Actions.StateDelta[key], any("value"); got != want {
			t.Errorf("StateDelta[%q] = %v, want %v", key, got, want)
		}
	})

	t.Run("OutputSchema valid JSON: Output is parsed", func(t *testing.T) {
		t.Parallel()
		schema := &genai.Schema{
			Type: genai.TypeObject,
			Properties: map[string]*genai.Schema{
				"name": {Type: genai.TypeString},
			},
		}
		a := makeLLMAgent(t, "x", withOutputSchema(schema))
		ev := &session.Event{
			LLMResponse: model.LLMResponse{
				Content: &genai.Content{
					Role:  "model",
					Parts: []*genai.Part{{Text: `{"name":"alice"}`}},
				},
			},
		}
		if err := llmagent.ProcessLLMAgentOutput(a, ev); err != nil {
			t.Fatalf("err = %v, want nil", err)
		}
		got, ok := ev.Output.(map[string]any)
		if !ok {
			t.Fatalf("Output type = %T, want map[string]any (parsed object)", ev.Output)
		}
		if got["name"] != "alice" {
			t.Errorf("Output[\"name\"] = %v, want \"alice\"", got["name"])
		}
	})

	t.Run("OutputSchema invalid JSON: returns validation error", func(t *testing.T) {
		t.Parallel()
		schema := &genai.Schema{
			Type: genai.TypeObject,
			Properties: map[string]*genai.Schema{
				"age": {Type: genai.TypeInteger},
			},
			Required: []string{"age"},
		}
		a := makeLLMAgent(t, "x", withOutputSchema(schema))
		const raw = `{"wrong":"shape"}`
		ev := &session.Event{
			LLMResponse: model.LLMResponse{
				Content: &genai.Content{
					Role:  "model",
					Parts: []*genai.Part{{Text: raw}},
				},
			},
		}
		if err := llmagent.ProcessLLMAgentOutput(a, ev); err == nil {
			t.Fatalf("err = nil, want a schema-validation error for raw=%q", raw)
		}
	})

	t.Run("OutputSchema + whitespace-only text: Output stays nil, no error", func(t *testing.T) {
		t.Parallel()
		schema := &genai.Schema{Type: genai.TypeObject}
		a := makeLLMAgent(t, "x", withOutputSchema(schema))
		const raw = "   " // whitespace-only
		ev := &session.Event{
			LLMResponse: model.LLMResponse{
				Content: &genai.Content{
					Role:  "model",
					Parts: []*genai.Part{{Text: raw}},
				},
			},
		}
		if err := llmagent.ProcessLLMAgentOutput(a, ev); err != nil {
			t.Fatalf("err = %v, want nil (whitespace short-circuits to nil)", err)
		}
		if ev.Output != nil {
			t.Errorf("Output = %v, want nil (whitespace text must NOT be returned as the output)", ev.Output)
		}
		if ev.NodeInfo == nil || !ev.NodeInfo.MessageAsOutput {
			t.Errorf("MessageAsOutput should still be set")
		}
	})

	t.Run("pre-existing NodeInfo is preserved", func(t *testing.T) {
		t.Parallel()
		a := makeLLMAgent(t, "x")
		ev := &session.Event{
			NodeInfo: &session.NodeInfo{Path: "outer/inner"},
			LLMResponse: model.LLMResponse{
				Content: &genai.Content{Role: "model", Parts: []*genai.Part{{Text: "ok"}}},
			},
		}
		if err := llmagent.ProcessLLMAgentOutput(a, ev); err != nil {
			t.Fatal(err)
		}
		if ev.NodeInfo.Path != "outer/inner" {
			t.Errorf("NodeInfo.Path = %q, want preserved %q", ev.NodeInfo.Path, "outer/inner")
		}
		if !ev.NodeInfo.MessageAsOutput {
			t.Errorf("MessageAsOutput not set after preserving Path")
		}
	})
}

// TestRunLLMAgentAsNode_NonLLMAgent_Errors covers the early-out:
// passing a non-LlmAgent yields a single error event and terminates.
func TestRunLLMAgentAsNode_NonLLMAgent_Errors(t *testing.T) {
	t.Parallel()
	a, err := agent.New(agent.Config{Name: "plain"})
	if err != nil {
		t.Fatal(err)
	}
	ctx := newStubNodeContext(t, a, "")

	var gotErr error
	var gotEvents int
	for ev, err := range llmagent.RunLLMAgentAsNode(a, ctx, nil) {
		if err != nil {
			gotErr = err
		}
		if ev != nil {
			gotEvents++
		}
	}
	if gotErr == nil {
		t.Fatal("expected an error for non-LlmAgent; got nil")
	}
	if !strings.Contains(gotErr.Error(), "not an LlmAgent") {
		t.Errorf("err = %q, want it to mention 'not an LlmAgent'", gotErr.Error())
	}
	if gotEvents != 0 {
		t.Errorf("got %d events, want 0 (only an error)", gotEvents)
	}
}

// TestRunLLMAgentAsNode_UnsupportedMode_Errors covers mode value validation.
func TestRunLLMAgentAsNode_UnsupportedMode_Errors(t *testing.T) {
	t.Parallel()
	a := makeLLMAgent(t, "x", withMode(llmagent.Mode("bogus")))
	ctx := newStubNodeContext(t, a, "")

	var gotErr error
	for _, err := range llmagent.RunLLMAgentAsNode(a, ctx, nil) {
		if err != nil {
			gotErr = err
		}
	}
	if gotErr == nil {
		t.Fatal("expected an error for unsupported mode; got nil")
	}
	if !strings.Contains(gotErr.Error(), "task, single_turn, and chat") {
		t.Errorf("err = %q, want it to enumerate the supported modes", gotErr.Error())
	}
}

// TestRunLLMAgentAsNode_Task_HappyPath drives a task-mode agent
// end-to-end via the runner. The scripted LLM emits finish_task
// immediately; the wrapper must promote the args under the wrapper
// key as the task's terminal Output. The task agent's events are
// scoped (see TestDispatchTaskFC_IsolationScope) so we verify the
// FC and the success FR appear in the per-task event stream.
func TestRunLLMAgentAsNode_Task_HappyPath(t *testing.T) {
	t.Parallel()
	const (
		coordName    = "coord"
		taskName     = "doer"
		fcID         = "fc-happy-task"
		expectedTask = "summarize"
	)
	taskLLM := &scriptedLLM{
		turns: []*model.LLMResponse{{
			Content: &genai.Content{
				Role: "model",
				Parts: []*genai.Part{{
					FunctionCall: &genai.FunctionCall{
						ID:   "finish-1",
						Name: "finish_task",
						Args: map[string]any{"result": expectedTask},
					},
				}},
			},
		}},
	}
	taskAgent := makeLLMAgent(t, taskName,
		withMode(llmagent.ModeTask),
		func(c *llmagent.Config) { c.Model = taskLLM },
	)
	coordLLM := &scriptedLLM{
		turns: []*model.LLMResponse{
			{
				Content: &genai.Content{
					Role: "model",
					Parts: []*genai.Part{{
						FunctionCall: &genai.FunctionCall{
							ID:   fcID,
							Name: taskName,
							Args: map[string]any{"request": "summarize the corpus"},
						},
					}},
				},
			},
			{
				Content: &genai.Content{
					Role:  "model",
					Parts: []*genai.Part{genai.NewPartFromText("delegation complete")},
				},
			},
		},
	}
	coord, err := llmagent.New(llmagent.Config{
		Name:      coordName,
		Model:     coordLLM,
		Mode:      llmagent.ModeChat,
		SubAgents: []agent.Agent{taskAgent},
	})
	if err != nil {
		t.Fatalf("llmagent.New(coord): %v", err)
	}

	svc := session.InMemoryService()
	if _, err := svc.Create(t.Context(), &session.CreateRequest{
		AppName: "app", UserID: "u", SessionID: "s",
	}); err != nil {
		t.Fatal(err)
	}
	r, err := runner.New(runner.Config{Agent: coord, SessionService: svc, AppName: "app"})
	if err != nil {
		t.Fatal(err)
	}

	var sawFinishFC bool
	var sawSuccessFR bool
	for ev, err := range r.Run(t.Context(), "u", "s",
		&genai.Content{Parts: []*genai.Part{genai.NewPartFromText("go")}, Role: "user"},
		agent.RunConfig{StreamingMode: agent.StreamingModeSSE}) {
		if err != nil {
			t.Fatalf("runner.Run: %v", err)
		}
		if ev == nil || ev.Author != taskName {
			continue
		}
		for _, p := range ev.Content.Parts {
			if p == nil {
				continue
			}
			if p.FunctionCall != nil && p.FunctionCall.Name == "finish_task" {
				sawFinishFC = true
			}
			if p.FunctionResponse != nil && p.FunctionResponse.Name == "finish_task" {
				sawSuccessFR = true
			}
		}
	}
	if !sawFinishFC {
		t.Error("expected the task agent to emit a finish_task FunctionCall event")
	}
	if !sawSuccessFR {
		t.Error("expected the task agent to receive a finish_task FunctionResponse event")
	}
}

func runChatCoordinatorOneTurn(t *testing.T, coord agent.Agent, userText string) []*session.Event {
	t.Helper()
	svc := session.InMemoryService()
	if _, err := svc.Create(t.Context(), &session.CreateRequest{
		AppName: "app", UserID: "u", SessionID: "s",
	}); err != nil {
		t.Fatalf("session.Create: %v", err)
	}
	r, err := runner.New(runner.Config{Agent: coord, SessionService: svc, AppName: "app"})
	if err != nil {
		t.Fatalf("runner.New: %v", err)
	}
	events := []*session.Event{}
	for ev, err := range r.Run(t.Context(), "u", "s",
		&genai.Content{Parts: []*genai.Part{genai.NewPartFromText(userText)}, Role: "user"},
		agent.RunConfig{StreamingMode: agent.StreamingModeSSE}) {
		if err != nil {
			t.Fatalf("runner.Run: %v", err)
		}
		if ev != nil {
			events = append(events, ev)
		}
	}
	return events
}

// TestChatRoot_TwoTaskSubAgents_Sequential verifies that
// a chat coordinator emits two task-delegation FCs sequentially: one
// to sub_a, then (after sub_a's FR arrives) one to sub_b. Both task
// agents must run to completion (finish_task → success FR), and
// the coordinator must see both FRs before producing its final reply.
//
// Note on chat-loop semantics: the coordinator's first LLM turn emits
// ONE task FC (to sub_a); the wrapper dispatches it and synthesises
// the FR; the loop re-enters Agent.Run; the second LLM turn emits the
// FC for sub_b; same dispatch; loop re-enters once more; the third
// LLM turn emits the final coordinator reply. So this test pins
// runChat's sequential dispatch-and-re-enter behaviour.
func TestChatRoot_TwoTaskSubAgents_Sequential(t *testing.T) {
	t.Parallel()
	const (
		coordName = "coord"
		nameA     = "sub_a"
		nameB     = "sub_b"
		fcIDA     = "fc-a"
		fcIDB     = "fc-b"
	)

	makeTaskAgent := func(name string) agent.Agent {
		llm := &scriptedLLM{
			turns: []*model.LLMResponse{{
				Content: &genai.Content{
					Role: "model",
					Parts: []*genai.Part{{
						FunctionCall: &genai.FunctionCall{
							ID:   "finish-" + name,
							Name: "finish_task",
							Args: map[string]any{"result": "done-" + name},
						},
					}},
				},
			}},
		}
		return makeLLMAgent(t, name,
			withMode(llmagent.ModeTask),
			func(c *llmagent.Config) { c.Model = llm },
		)
	}
	subA := makeTaskAgent(nameA)
	subB := makeTaskAgent(nameB)

	// Coordinator emits FC-A first, then FC-B (after seeing FR-A
	// synthesised into session), then a final "all done" reply.
	coordLLM := &scriptedLLM{
		turns: []*model.LLMResponse{
			{Content: fcContent(fcIDA, nameA, map[string]any{"q": "first"})},
			{Content: fcContent(fcIDB, nameB, map[string]any{"q": "second"})},
			{Content: &genai.Content{
				Role:  "model",
				Parts: []*genai.Part{genai.NewPartFromText("all done")},
			}},
		},
	}
	coord, err := llmagent.New(llmagent.Config{
		Name:      coordName,
		Model:     coordLLM,
		Mode:      llmagent.ModeChat,
		SubAgents: []agent.Agent{subA, subB},
	})
	if err != nil {
		t.Fatal(err)
	}

	events := runChatCoordinatorOneTurn(t, coord, "go")

	// Both task agents must have completed: each should have emitted
	// a finish_task FC AND received the success FR.
	sawFinishFC := map[string]bool{}
	sawSuccessFR := map[string]bool{}
	for _, ev := range events {
		if ev.Author != nameA && ev.Author != nameB {
			continue
		}
		for _, p := range ev.Content.Parts {
			if p == nil {
				continue
			}
			if p.FunctionCall != nil && p.FunctionCall.Name == "finish_task" {
				sawFinishFC[ev.Author] = true
			}
			if p.FunctionResponse != nil && p.FunctionResponse.Name == "finish_task" {
				sawSuccessFR[ev.Author] = true
			}
		}
	}
	for _, name := range []string{nameA, nameB} {
		if !sawFinishFC[name] {
			t.Errorf("task agent %q did not emit finish_task FC", name)
		}
		if !sawSuccessFR[name] {
			t.Errorf("task agent %q did not receive finish_task FR", name)
		}
	}
}

// TestChatTask_ValidationErrorDrivesRetry verifies that
// a task agent's first finish_task call has invalid args (missing
// required field per the output schema). The FinishTaskTool returns
// an error-dict FR which the LLM sees on the next round; the LLM
// then retries finish_task with valid args. The test asserts both
// attempts were issued and the second succeeded.
func TestChatTask_ValidationErrorDrivesRetry(t *testing.T) {
	t.Parallel()
	const (
		coordName = "coord"
		taskName  = "doer"
		fcID      = "fc-retry-task"
	)
	schema := &genai.Schema{
		Type:       genai.TypeObject,
		Properties: map[string]*genai.Schema{"answer": {Type: genai.TypeString}},
		Required:   []string{"answer"},
	}

	// Task LLM: first turn issues a finish_task with the wrong field
	// (`wrong_field` instead of `answer`). After the FinishTaskTool's
	// error FR appears in session, the second turn issues a corrected
	// finish_task with `answer`. A safety third turn emits text "done"
	// in case the script is over-consumed.
	taskLLM := &scriptedLLM{
		turns: []*model.LLMResponse{
			{Content: fcContent("ft-bad", "finish_task",
				map[string]any{"wrong_field": "oops"})},
			{Content: fcContent("ft-good", "finish_task",
				map[string]any{"answer": "42"})},
		},
		doneText: "should not be reached",
	}
	taskAgent := makeLLMAgent(t, taskName,
		withMode(llmagent.ModeTask),
		withOutputSchema(schema),
		func(c *llmagent.Config) { c.Model = taskLLM },
	)

	coordLLM := &scriptedLLM{
		turns: []*model.LLMResponse{
			{Content: fcContent(fcID, taskName, map[string]any{"request": "what is x?"})},
			{Content: &genai.Content{
				Role:  "model",
				Parts: []*genai.Part{genai.NewPartFromText("delegation complete")},
			}},
		},
	}
	coord, err := llmagent.New(llmagent.Config{
		Name:      coordName,
		Model:     coordLLM,
		Mode:      llmagent.ModeChat,
		SubAgents: []agent.Agent{taskAgent},
	})
	if err != nil {
		t.Fatal(err)
	}

	events := runChatCoordinatorOneTurn(t, coord, "go")

	// We expect two finish_task FCs from the task agent (the bad one,
	// then the corrected one), and the second-attempt FR should be
	// the success message.
	var finishFCs int
	var sawErrorFR bool
	var sawSuccessFR bool
	for _, ev := range events {
		if ev.Author != taskName {
			continue
		}
		for _, p := range ev.Content.Parts {
			if p == nil {
				continue
			}
			if p.FunctionCall != nil && p.FunctionCall.Name == "finish_task" {
				finishFCs++
			}
			if p.FunctionResponse != nil && p.FunctionResponse.Name == "finish_task" {
				if _, hasErr := p.FunctionResponse.Response["error"]; hasErr {
					sawErrorFR = true
				}
				if v, ok := p.FunctionResponse.Response["result"].(string); ok &&
					v == workflowinternal.FinishTaskSuccessResult {
					sawSuccessFR = true
				}
			}
		}
	}
	if finishFCs != 2 {
		t.Errorf("expected 2 finish_task FCs (bad then good), got %d", finishFCs)
	}
	if !sawErrorFR {
		t.Error("expected the FinishTaskTool to return an error FR for the bad finish_task args")
	}
	if !sawSuccessFR {
		t.Error("expected the FinishTaskTool to return a success FR for the corrected finish_task args")
	}
}

// TestChatCoordinator_ResumesUnresolvedTaskFC verifies that on
// the first user turn, the coordinator emits a task FC but no
// matching FR is produced (we simulate this by stopping the runner
// before the FR can be synthesised — in this Go variant we instead
// rely on the fact that a fresh runner.Run on a SECOND user turn
// triggers the wrapper's findUnresolvedTaskDelegations pre-LLM
// scan, which re-dispatches the still-pending FC before letting the
// coordinator's LLM run again).
//
// The minimal pre-condition the test asserts: when a second user
// turn fires and the session already contains an unresolved task FC
// from the coordinator, the task agent runs (its events appear in
// the second turn's stream) BEFORE the coordinator's LLM produces
// any new output.
func TestChatCoordinator_ResumesUnresolvedTaskFC(t *testing.T) {
	t.Parallel()
	const (
		coordName = "coord"
		taskName  = "doer"
		fcID      = "fc-resume-1"
	)

	// Task LLM: when finally invoked (on the second user turn's
	// pre-LLM scan), immediately finishes with "answered".
	taskLLM := &scriptedLLM{
		turns: []*model.LLMResponse{{
			Content: fcContent("finish-1", "finish_task",
				map[string]any{"result": "answered"}),
		}},
	}
	taskAgent := makeLLMAgent(t, taskName,
		withMode(llmagent.ModeTask),
		func(c *llmagent.Config) { c.Model = taskLLM },
	)

	// Coordinator LLM:
	//   turn 0 (user "go") → emit task FC; the scripted LLM stops
	//     here. The runner's chat wrapper would normally dispatch the
	//     FC and re-enter, but to simulate "the session has an
	//     unresolved FC from a prior turn" we have the coordinator's
	//     second-turn script emit a terminal text "first done" — this
	//     happens AFTER the wrapper synthesises the FR on its first
	//     re-entry, so by the time the second user turn fires below,
	//     the FC is resolved already.
	//
	// To genuinely test resume-on-pending-FC we would need to inject
	// a pending FC into session BEFORE the runner starts. That
	// requires either (a) raw session.AppendEvent before runner.Run,
	// or (b) two runner.Run calls with the script crafted so the
	// first leaves a pending FC.
	//
	// (a) is the cleanest. Let's do that.
	coordLLM := &scriptedLLM{
		// First user turn's LLM call: coordinator emits final text.
		// The pending FC injected into session below is what should
		// be re-dispatched BEFORE this LLM call.
		turns: []*model.LLMResponse{
			{Content: &genai.Content{
				Role:  "model",
				Parts: []*genai.Part{genai.NewPartFromText("post-resume reply")},
			}},
		},
	}
	coord, err := llmagent.New(llmagent.Config{
		Name:      coordName,
		Model:     coordLLM,
		Mode:      llmagent.ModeChat,
		SubAgents: []agent.Agent{taskAgent},
	})
	if err != nil {
		t.Fatal(err)
	}

	// Pre-seed the session with an unresolved task FC authored by
	// the coordinator.
	svc := session.InMemoryService()
	createResp, err := svc.Create(t.Context(), &session.CreateRequest{
		AppName: "app", UserID: "u", SessionID: "s",
	})
	if err != nil {
		t.Fatal(err)
	}
	pendingFCEvent := session.NewEvent(t.Context(), "inv-prior")
	pendingFCEvent.Author = coordName
	pendingFCEvent.LLMResponse = model.LLMResponse{
		Content: fcContent(fcID, taskName, map[string]any{"q": "leftover"}),
	}
	if err := svc.AppendEvent(t.Context(), createResp.Session, pendingFCEvent); err != nil {
		t.Fatalf("AppendEvent(pending FC): %v", err)
	}

	r, err := runner.New(runner.Config{Agent: coord, SessionService: svc, AppName: "app"})
	if err != nil {
		t.Fatal(err)
	}

	// Now drive a user turn. The wrapper's pre-LLM scan must find
	// the pending FC, dispatch it, and synthesise an FR — all
	// BEFORE the coordinator's LLM is called.
	events := []*session.Event{}
	for ev, err := range r.Run(t.Context(), "u", "s",
		&genai.Content{Parts: []*genai.Part{genai.NewPartFromText("now resume")}, Role: "user"},
		agent.RunConfig{StreamingMode: agent.StreamingModeSSE}) {
		if err != nil {
			t.Fatalf("runner.Run: %v", err)
		}
		if ev != nil {
			events = append(events, ev)
		}
	}

	// The task agent's events must appear in this turn's stream
	// (it must have actually run during pre-LLM scan).
	var sawTaskFinishFC bool
	var sawTaskSuccessFR bool
	for _, ev := range events {
		if ev.Author != taskName {
			continue
		}
		for _, p := range ev.Content.Parts {
			if p == nil {
				continue
			}
			if p.FunctionCall != nil && p.FunctionCall.Name == "finish_task" {
				sawTaskFinishFC = true
			}
			if p.FunctionResponse != nil && p.FunctionResponse.Name == "finish_task" {
				sawTaskSuccessFR = true
			}
		}
	}
	if !sawTaskFinishFC {
		t.Error("expected the task agent's finish_task FC to appear during the resume turn")
	}
	if !sawTaskSuccessFR {
		t.Error("expected the task agent's finish_task success FR to appear during the resume turn")
	}
}

func fcContent(id, name string, args map[string]any) *genai.Content {
	return &genai.Content{
		Role: "model",
		Parts: []*genai.Part{{
			FunctionCall: &genai.FunctionCall{ID: id, Name: name, Args: args},
		}},
	}
}
