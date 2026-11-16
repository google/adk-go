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

package loopagent_test

import (
	"context"
	"fmt"
	"iter"
	"slices"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/agent/workflowagents/loopagent"
	"google.golang.org/adk/v2/agent/workflowagents/parallelagent"
	"google.golang.org/adk/v2/internal/agent/runconfig"
	"google.golang.org/adk/v2/internal/testutil"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
)

const (
	resumeApp     = "resume_app"
	resumeUser    = "user_id"
	resumeSession = "session_id"
	resumeInv     = "inv-1"
)

// resumeDescribe renders ev as "<author>:END" for an end-of-agent event;
// "<author>:STATE(<current_sub_agent>,<times_looped>)" for a loop checkpoint
// and "<author>:STATE({})" for an empty one; otherwise "<author>:" followed by
// "FC(name)" for each function call and the text of each text part.
func resumeDescribe(ev *session.Event) string {
	if ev.Actions.EndOfAgent != nil && *ev.Actions.EndOfAgent {
		return ev.Author + ":END"
	}
	if st := ev.Actions.AgentState; st != nil {
		if len(st) == 0 {
			return ev.Author + ":STATE({})"
		}
		return fmt.Sprintf("%s:STATE(%v,%v)", ev.Author, st["current_sub_agent"], st["times_looped"])
	}
	var b strings.Builder
	if ev.Content != nil {
		for _, p := range ev.Content.Parts {
			if p.FunctionCall != nil {
				fmt.Fprintf(&b, "FC(%s)", p.FunctionCall.Name)
				continue
			}
			b.WriteString(p.Text)
		}
	}
	return ev.Author + ":" + b.String()
}

// resumeStep is one text event a resumeAgent yields.
type resumeStep struct {
	text     string
	escalate bool
}

// resumeAgent returns an agent named name whose n-th run, counting from 1,
// yields the events steps(n). runs counts its runs.
func resumeAgent(t *testing.T, name string, runs *atomic.Int32, steps func(run int32) []resumeStep) agent.Agent {
	t.Helper()
	a, err := agent.New(agent.Config{Name: name, Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
		return func(yield func(*session.Event, error) bool) {
			for _, step := range steps(runs.Add(1)) {
				ev := session.NewEvent(ctx, ctx.InvocationID())
				ev.Author = name
				ev.Content = genai.NewContentFromText(step.text, genai.RoleModel)
				ev.Actions.Escalate = step.escalate
				if !yield(ev, nil) {
					return
				}
			}
		}
	}})
	if err != nil {
		t.Fatal(err)
	}
	return a
}

// runCount is the steps of a resumeAgent that yields its run number.
func runCount(run int32) []resumeStep { return []resumeStep{{text: fmt.Sprint(run)}} }

func resumeLoop(t *testing.T, name string, maxIterations uint, subAgents ...agent.Agent) agent.Agent {
	t.Helper()
	loop, err := loopagent.New(loopagent.Config{
		AgentConfig:   agent.Config{Name: name, SubAgents: subAgents},
		MaxIterations: maxIterations,
	})
	if err != nil {
		t.Fatal(err)
	}
	return loop
}

// resumeSessionService returns a session service holding the test session
// with evs appended in order.
func resumeSessionService(t *testing.T, evs ...*session.Event) session.Service {
	t.Helper()
	svc := session.InMemoryService()
	created, err := svc.Create(t.Context(), &session.CreateRequest{AppName: resumeApp, UserID: resumeUser, SessionID: resumeSession})
	if err != nil {
		t.Fatal(err)
	}
	for _, ev := range evs {
		if err := svc.AppendEvent(t.Context(), created.Session, ev); err != nil {
			t.Fatal(err)
		}
	}
	return svc
}

// resumeCollect runs one turn of root on a resumable runner over svc and
// returns the described events and the errors, reading on after an error as
// the runner's callers do.
func resumeCollect(t *testing.T, root agent.Agent, svc session.Service, msg *genai.Content, opts ...runner.RunOption) ([]string, []string) {
	t.Helper()
	r, err := runner.New(runner.Config{AppName: resumeApp, Agent: root, SessionService: svc, Resumable: true})
	if err != nil {
		t.Fatal(err)
	}
	var events, errs []string
	for ev, err := range r.Run(t.Context(), resumeUser, resumeSession, msg, agent.RunConfig{}, opts...) {
		if err != nil {
			errs = append(errs, err.Error())
			continue
		}
		events = append(events, resumeDescribe(ev))
	}
	return events, errs
}

// resumeFresh runs root for one turn on a resumable runner.
func resumeFresh(t *testing.T, root agent.Agent) ([]string, []string) {
	t.Helper()
	return resumeCollect(t, root, resumeSessionService(t), genai.NewContentFromText("go", genai.RoleUser))
}

// resumeFromState resumes invocation resumeInv of root on a resumable runner,
// from a session that holds the invocation's user message and root's agent
// state state.
func resumeFromState(t *testing.T, root agent.Agent, state map[string]any) ([]string, []string) {
	t.Helper()
	user := session.NewEvent(context.Background(), resumeInv)
	user.Author = "user"
	user.Content = genai.NewContentFromText("go", genai.RoleUser)
	checkpoint := session.NewEvent(context.Background(), resumeInv)
	checkpoint.Author = root.Name()
	checkpoint.Actions.AgentState = state
	return resumeCollect(t, root, resumeSessionService(t, user, checkpoint), nil, runner.WithInvocationID(resumeInv))
}

// TestLoopAgent_ResumableCheckpointsAndEscalation checks the events of a
// resumable loop (agents/loop_agent.py, _run_async_impl): nothing for an
// empty loop; a checkpoint {current_sub_agent, times_looped} before each
// sub-agent with times_looped counting the completed passes; the end event
// after the last pass and after an escalation, which still lets the
// escalating sub-agent finish; and an escalation from a nested loop ends the
// outer loop too (ported flaw PARALLEL-LOOP-ESCALATE-FLAW-1). It fails if
// times_looped counts from 1, if the end event is missing after an
// escalation, or if the loop checks the escalating event's author.
func TestLoopAgent_ResumableCheckpointsAndEscalation(t *testing.T) {
	t.Run("empty resumable loop", func(t *testing.T) {
		events, errs := resumeFresh(t, resumeLoop(t, "loop", 2))

		if diff := cmp.Diff([]string(nil), events); diff != "" {
			t.Errorf("events (-want +got):\n%s", diff)
		}
		if len(errs) != 0 {
			t.Errorf("errors = %v, want none", errs)
		}
	})
	t.Run("two passes checkpoints and end", func(t *testing.T) {
		var a1Runs, a2Runs atomic.Int32
		loop := resumeLoop(t, "loop", 2, resumeAgent(t, "a1", &a1Runs, runCount), resumeAgent(t, "a2", &a2Runs, runCount))

		events, errs := resumeFresh(t, loop)

		want := []string{
			"loop:STATE(a1,0)", "a1:1", "loop:STATE(a2,0)", "a2:1",
			"loop:STATE(a1,1)", "a1:2", "loop:STATE(a2,1)", "a2:2",
			"loop:END",
		}
		if diff := cmp.Diff(want, events); diff != "" {
			t.Errorf("events (-want +got):\n%s", diff)
		}
		if len(errs) != 0 {
			t.Errorf("errors = %v, want none", errs)
		}
	})
	t.Run("escalation drains sub-agent and ends loop", func(t *testing.T) {
		var a1Runs, a2Runs atomic.Int32
		a1 := resumeAgent(t, "a1", &a1Runs, func(int32) []resumeStep {
			return []resumeStep{{text: "esc", escalate: true}, {text: "followup"}}
		})
		loop := resumeLoop(t, "loop", 3, a1, resumeAgent(t, "a2", &a2Runs, runCount))

		events, errs := resumeFresh(t, loop)

		if diff := cmp.Diff([]string{"loop:STATE(a1,0)", "a1:esc", "a1:followup", "loop:END"}, events); diff != "" {
			t.Errorf("events (-want +got):\n%s", diff)
		}
		if len(errs) != 0 {
			t.Errorf("errors = %v, want none", errs)
		}
		if got := a2Runs.Load(); got != 0 {
			t.Errorf("a2 ran %d times, want 0", got)
		}
	})
	t.Run("nested loop escalation ends outer loop (ported flaw 1)", func(t *testing.T) {
		var aRuns atomic.Int32
		a := resumeAgent(t, "a", &aRuns, func(int32) []resumeStep { return []resumeStep{{text: "esc", escalate: true}} })
		outer := resumeLoop(t, "outer", 3, resumeLoop(t, "inner", 3, a))

		events, errs := resumeFresh(t, outer)

		want := []string{"outer:STATE(inner,0)", "inner:STATE(a,0)", "a:esc", "inner:END", "outer:END"}
		if diff := cmp.Diff(want, events); diff != "" {
			t.Errorf("events (-want +got):\n%s", diff)
		}
		if len(errs) != 0 {
			t.Errorf("errors = %v, want none", errs)
		}
		if got := aRuns.Load(); got != 1 {
			t.Errorf("a ran %d times, want 1", got)
		}
	})
}

// TestLoopAgent_ResumeEdgeCases checks where a resumed loop restarts
// (agents/loop_agent.py, _get_start_state): a current_sub_agent that is no
// longer a sub-agent restarts the pass at 0 without writing its checkpoint
// again and keeps times_looped; a times_looped at MaxIterations runs no pass;
// a stored state of the wrong shape fails the run; graph narrowing
// (a function response to a call in a graph workflow) overrides the stored
// sub-agent and writes the checkpoint of the one it restarts at. It fails if
// an unknown name resets times_looped, if the loop runs a pass before checking
// the count, if a decode error is ignored, or if graph narrowing leaves the
// loop treating its restart checkpoint as already stored.
func TestLoopAgent_ResumeEdgeCases(t *testing.T) {
	newLoop := func(t *testing.T, a1Runs, a2Runs *atomic.Int32) agent.Agent {
		return resumeLoop(t, "loop", 2, resumeAgent(t, "a1", a1Runs, runCount), resumeAgent(t, "a2", a2Runs, runCount))
	}

	t.Run("unknown sub-agent keeps times_looped and restarts pass at 0", func(t *testing.T) {
		var a1Runs, a2Runs atomic.Int32

		events, errs := resumeFromState(t, newLoop(t, &a1Runs, &a2Runs), map[string]any{"current_sub_agent": "gone", "times_looped": 1})

		if diff := cmp.Diff([]string{"a1:1", "loop:STATE(a2,1)", "a2:1", "loop:END"}, events); diff != "" {
			t.Errorf("events (-want +got):\n%s", diff)
		}
		if len(errs) != 0 {
			t.Errorf("errors = %v, want none", errs)
		}
	})
	t.Run("times_looped at max_iterations emits only end", func(t *testing.T) {
		var a1Runs, a2Runs atomic.Int32

		events, errs := resumeFromState(t, newLoop(t, &a1Runs, &a2Runs), map[string]any{"current_sub_agent": "a1", "times_looped": 2})

		if diff := cmp.Diff([]string{"loop:END"}, events); diff != "" {
			t.Errorf("events (-want +got):\n%s", diff)
		}
		if len(errs) != 0 {
			t.Errorf("errors = %v, want none", errs)
		}
		if got := a1Runs.Load(); got != 0 {
			t.Errorf("a1 ran %d times, want 0", got)
		}
	})
	for _, state := range []map[string]any{
		{"current_sub_agent": "a1", "times_looped": -1},
		{"current_sub_agent": "a1", "times_looped": "1"},
		{"unknown": "x"},
	} {
		t.Run(fmt.Sprintf("malformed state %v returns error", state), func(t *testing.T) {
			var a1Runs, a2Runs atomic.Int32

			events, errs := resumeFromState(t, newLoop(t, &a1Runs, &a2Runs), state)

			if len(events) != 0 {
				t.Errorf("events = %v, want none", events)
			}
			if len(errs) != 1 || !strings.Contains(errs[0], `loop agent "loop": stored agent state`) {
				t.Errorf("errors = %q, want one containing %q", errs, `loop agent "loop": stored agent state`)
			}
			if a1, a2 := a1Runs.Load(), a2Runs.Load(); a1 != 0 || a2 != 0 {
				t.Errorf("a1 ran %d times and a2 %d times, want 0 and 0", a1, a2)
			}
		})
	}
	t.Run("graph resume index overrides stored state", func(t *testing.T) {
		// Turn 1 pauses in node b of the graph workflow wf. A checkpoint that
		// names a1 is then appended to the invocation, so the stored state
		// and the answered call point at different sub-agents.
		var a1Runs atomic.Int32
		loop := resumeLoop(t, "loop", 1, resumeAgent(t, "a1", &a1Runs, runCount), testutil.NewInputWorkflow(t, "wf"))
		svc := resumeSessionService(t)
		turn1, errs := resumeCollect(t, loop, svc, genai.NewContentFromText("go", genai.RoleUser))
		if len(errs) != 0 {
			t.Fatalf("turn 1 errors = %v", errs)
		}
		if !slices.Contains(turn1, "loop:STATE(wf,0)") {
			t.Fatalf("turn 1 events = %v, want loop's checkpoint for wf", turn1)
		}
		got, err := svc.Get(t.Context(), &session.GetRequest{AppName: resumeApp, UserID: resumeUser, SessionID: resumeSession})
		if err != nil {
			t.Fatal(err)
		}
		stale := session.NewEvent(context.Background(), got.Session.Events().At(got.Session.Events().Len()-1).InvocationID)
		stale.Author = "loop"
		stale.Actions.AgentState = map[string]any{"current_sub_agent": "a1", "times_looped": 0}
		if err := svc.AppendEvent(t.Context(), got.Session, stale); err != nil {
			t.Fatal(err)
		}

		turn2, errs := resumeCollect(t, loop, svc, testutil.InputResponse("b", "approved"))

		if len(errs) != 0 {
			t.Errorf("turn 2 errors = %v, want none", errs)
		}
		var loopEvents []string
		for _, ev := range turn2 {
			if strings.HasPrefix(ev, "loop:") {
				loopEvents = append(loopEvents, ev)
			}
		}
		if diff := cmp.Diff([]string{"loop:STATE(wf,0)", "loop:END"}, loopEvents); diff != "" {
			t.Errorf("loop's turn 2 events (-want +got):\n%s", diff)
		}
		if !slices.Contains(turn2, "d:D") {
			t.Errorf("turn 2 events = %v, want d's output", turn2)
		}
		if got := a1Runs.Load(); got != 1 {
			t.Errorf("a1 ran %d times over both turns, want 1", got)
		}
	})
}

// resumeReplyModel answers every request with the text reply.
type resumeReplyModel struct{ reply string }

func (resumeReplyModel) Name() string { return "resume-reply" }

func (m resumeReplyModel) GenerateContent(context.Context, *model.LLMRequest, bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		yield(&model.LLMResponse{Content: genai.NewContentFromText(m.reply, genai.RoleModel), FinishReason: genai.FinishReasonStop}, nil)
	}
}

// TestLoopAgent_ResetsDescendantsBetweenPassesAndPreservesOnPause checks
// that a resumable loop clears the state and the end mark of every agent below
// it after each completed pass, so a nested parallel agent runs its custom
// sub-agent again in the next pass (agents/loop_agent.py,
// ctx.reset_sub_agent_states), and that a pass that pauses clears nothing
// (tests/unittests/agents/test_loop_agent.py, the pause test). It fails if the
// loop does not reset its descendants, resets only its direct sub-agents, or
// resets before it returns on a pause.
func TestLoopAgent_ResetsDescendantsBetweenPassesAndPreservesOnPause(t *testing.T) {
	t.Run("resets custom ended sub-agent inside nested parallel across passes", func(t *testing.T) {
		var customRuns atomic.Int32
		custom, err := agent.New(agent.Config{Name: "custom", Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				ev := session.NewEvent(ctx, ctx.InvocationID())
				ev.Author = "custom"
				ev.Content = genai.NewContentFromText(fmt.Sprint(customRuns.Add(1)), genai.RoleModel)
				if !yield(ev, nil) {
					return
				}
				agent.RecordEndOfAgent(ctx)
			}
		}})
		if err != nil {
			t.Fatal(err)
		}
		par, err := parallelagent.New(parallelagent.Config{AgentConfig: agent.Config{Name: "par", SubAgents: []agent.Agent{custom}}})
		if err != nil {
			t.Fatal(err)
		}

		events, errs := resumeFresh(t, resumeLoop(t, "loop", 2, par))

		want := []string{
			"loop:STATE(par,0)", "par:STATE({})", "custom:1", "par:END",
			"loop:STATE(par,1)", "par:STATE({})", "custom:2", "par:END",
			"loop:END",
		}
		if diff := cmp.Diff(want, events); diff != "" {
			t.Errorf("events (-want +got):\n%s", diff)
		}
		if len(errs) != 0 {
			t.Errorf("errors = %v, want none", errs)
		}
		if got := customRuns.Load(); got != 2 {
			t.Errorf("custom ran %d times, want 2", got)
		}
	})
	t.Run("pause preserves descendant state and end marks", func(t *testing.T) {
		var rc *runconfig.RunConfig
		probe, err := agent.New(agent.Config{Name: "probe", Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(func(*session.Event, error) bool) { rc = runconfig.FromContext(ctx) }
		}})
		if err != nil {
			t.Fatal(err)
		}
		done, err := llmagent.New(llmagent.Config{Name: "done_llm", Model: resumeReplyModel{reply: "done"}})
		if err != nil {
			t.Fatal(err)
		}
		pauser, _ := newPausingAgent(t, "pauser", 0)

		events, errs := resumeFresh(t, resumeLoop(t, "loop", 2, probe, done, pauser))

		want := []string{
			"loop:STATE(probe,0)", "loop:STATE(done_llm,0)", "done_llm:done", "done_llm:END",
			"loop:STATE(pauser,0)", "pauser:FC(ask)",
		}
		if diff := cmp.Diff(want, events); diff != "" {
			t.Errorf("events (-want +got):\n%s", diff)
		}
		if len(errs) != 0 {
			t.Errorf("errors = %v, want none", errs)
		}
		if rc == nil {
			t.Fatal("probe saw no run config")
		}
		if !rc.AgentEnded("done_llm") {
			t.Error("AgentEnded(done_llm) = false after the pause; want true (a paused pass resets nothing)")
		}
	})
}
