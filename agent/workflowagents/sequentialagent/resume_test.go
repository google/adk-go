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

package sequentialagent_test

import (
	"context"
	"errors"
	"fmt"
	"iter"
	"maps"
	"slices"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/internal/testutil"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
)

const (
	resumeApp     = "resume_app"
	resumeUser    = "user_id"
	resumeSession = "session_id"
	resumeInv     = "inv-1"
)

// resumeDescribe renders ev as "<author>:END" for an end-of-agent event,
// "<author>:STATE(k=v,...)" with sorted keys for an agent-state event, and
// otherwise "<author>:" followed by "FC(name)" for each function call and the
// text of each text part, with "[ESC]" appended when ev escalates.
func resumeDescribe(ev *session.Event) string {
	if ev.Actions.EndOfAgent != nil && *ev.Actions.EndOfAgent {
		return ev.Author + ":END"
	}
	if ev.Actions.AgentState != nil {
		var kv []string
		for _, k := range slices.Sorted(maps.Keys(ev.Actions.AgentState)) {
			kv = append(kv, fmt.Sprintf("%s=%v", k, ev.Actions.AgentState[k]))
		}
		return ev.Author + ":STATE(" + strings.Join(kv, ",") + ")"
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
	if ev.Actions.Escalate {
		b.WriteString("[ESC]")
	}
	return ev.Author + ":" + b.String()
}

// resumeAgent returns an agent named name that yields one text event per
// entry of texts, escalating on each when escalate is set, and then the error
// err when it is not nil. runs counts its runs.
func resumeAgent(t *testing.T, name string, runs *atomic.Int32, escalate bool, err error, texts ...string) agent.Agent {
	t.Helper()
	return mustAgent(agent.New(agent.Config{Name: name, Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
		return func(yield func(*session.Event, error) bool) {
			runs.Add(1)
			for _, text := range texts {
				ev := textEvent(ctx, name, text)
				ev.Actions.Escalate = escalate
				if !yield(ev, nil) {
					return
				}
			}
			if err != nil {
				yield(nil, err)
			}
		}
	}}))
}

func resumeRunner(t *testing.T, root agent.Agent, svc session.Service) *runner.Runner {
	t.Helper()
	r, err := runner.New(runner.Config{AppName: resumeApp, Agent: root, SessionService: svc, Resumable: true})
	if err != nil {
		t.Fatal(err)
	}
	return r
}

// resumeCollect runs one turn of r with msg and returns the described events
// and the errors, reading on after an error as the runner's callers do.
func resumeCollect(t *testing.T, r *runner.Runner, msg *genai.Content, opts ...runner.RunOption) ([]string, []string) {
	t.Helper()
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

func resumeSeedEvent(author string, content *genai.Content, state map[string]any) *session.Event {
	ev := session.NewEvent(context.Background(), resumeInv)
	ev.Author = author
	ev.Content = content
	ev.Actions.AgentState = state
	return ev
}

// resumeFromState resumes invocation resumeInv of root on a resumable runner,
// from a session that holds the invocation's user message and root's agent
// state state.
func resumeFromState(t *testing.T, root agent.Agent, state map[string]any) ([]string, []string) {
	t.Helper()
	svc := resumeSessionService(t,
		resumeSeedEvent("user", genai.NewContentFromText("go", genai.RoleUser), nil),
		resumeSeedEvent(root.Name(), nil, state),
	)
	return resumeCollect(t, resumeRunner(t, root, svc), nil, runner.WithInvocationID(resumeInv))
}

// TestSequentialAgent_ResumableEmptyAndEscalatePassthrough checks, in a
// resumable run, that an empty sequential agent yields nothing, not even its
// end event (agents/sequential_agent.py, `if not self.sub_agents: return`),
// and that an escalation does not stop the sequence: adk-python's
// SequentialAgent has no escalation check. It fails if Run yields the end
// event before checking for sub-agents, or if Run stops on Actions.Escalate.
func TestSequentialAgent_ResumableEmptyAndEscalatePassthrough(t *testing.T) {
	t.Run("empty sequential agent", func(t *testing.T) {
		seq := newSequentialAgent(t, nil, "seq")

		events, errs := resumeCollect(t, resumeRunner(t, seq, resumeSessionService(t)), genai.NewContentFromText("go", genai.RoleUser))

		if diff := cmp.Diff([]string(nil), events); diff != "" {
			t.Errorf("events (-want +got):\n%s", diff)
		}
		if len(errs) != 0 {
			t.Errorf("errors = %v, want none", errs)
		}
	})
	t.Run("escalation does not stop the sequence", func(t *testing.T) {
		var escRuns, bRuns atomic.Int32
		seq := newSequentialAgent(t, []agent.Agent{
			resumeAgent(t, "esc_a", &escRuns, true, nil, "a"),
			resumeAgent(t, "b", &bRuns, false, nil, "b"),
		}, "seq")

		events, errs := resumeCollect(t, resumeRunner(t, seq, resumeSessionService(t)), genai.NewContentFromText("go", genai.RoleUser))

		want := []string{"seq:STATE(current_sub_agent=esc_a)", "esc_a:a[ESC]", "seq:STATE(current_sub_agent=b)", "b:b", "seq:END"}
		if diff := cmp.Diff(want, events); diff != "" {
			t.Errorf("events (-want +got):\n%s", diff)
		}
		if len(errs) != 0 {
			t.Errorf("errors = %v, want none", errs)
		}
		if got := bRuns.Load(); got != 1 {
			t.Errorf("b ran %d times, want 1", got)
		}
	})
}

// TestSequentialAgent_ResumeEdgeCases checks where a resumed sequential agent
// restarts (agents/sequential_agent.py, _get_start_index): an empty
// current_sub_agent means the sequence finished; a name that is no longer a
// sub-agent restarts at the first sub-agent without writing its checkpoint
// again; a stored state of the wrong shape fails the run; and graph narrowing
// overrides the stored state. It fails if the empty name restarts at index 0,
// if an unknown name writes the first checkpoint again, if a decode error is
// ignored, or if graph narrowing does not win.
func TestSequentialAgent_ResumeEdgeCases(t *testing.T) {
	newSeq := func(t *testing.T, a1Runs, a2Runs *atomic.Int32) agent.Agent {
		return newSequentialAgent(t, []agent.Agent{textAgent(t, "a1", a1Runs), textAgent(t, "a2", a2Runs)}, "seq")
	}

	t.Run("empty current_sub_agent means finished", func(t *testing.T) {
		var a1Runs, a2Runs atomic.Int32

		events, errs := resumeFromState(t, newSeq(t, &a1Runs, &a2Runs), map[string]any{"current_sub_agent": ""})

		if diff := cmp.Diff([]string{"seq:END"}, events); diff != "" {
			t.Errorf("events (-want +got):\n%s", diff)
		}
		if len(errs) != 0 {
			t.Errorf("errors = %v, want none", errs)
		}
		if a1, a2 := a1Runs.Load(), a2Runs.Load(); a1 != 0 || a2 != 0 {
			t.Errorf("a1 ran %d times and a2 %d times, want 0 and 0", a1, a2)
		}
	})
	t.Run("unknown current_sub_agent restarts at 0 without duplicate state event", func(t *testing.T) {
		var a1Runs, a2Runs atomic.Int32

		events, errs := resumeFromState(t, newSeq(t, &a1Runs, &a2Runs), map[string]any{"current_sub_agent": "removed_agent"})

		want := []string{"a1:a1 text", "seq:STATE(current_sub_agent=a2)", "a2:a2 text", "seq:END"}
		if diff := cmp.Diff(want, events); diff != "" {
			t.Errorf("events (-want +got):\n%s", diff)
		}
		if len(errs) != 0 {
			t.Errorf("errors = %v, want none", errs)
		}
	})
	for _, state := range []map[string]any{
		{"current_sub_agent": 42},
		{"unknown": "x"},
	} {
		t.Run(fmt.Sprintf("malformed stored state %v returns error", state), func(t *testing.T) {
			var a1Runs, a2Runs atomic.Int32

			events, errs := resumeFromState(t, newSeq(t, &a1Runs, &a2Runs), state)

			if len(events) != 0 {
				t.Errorf("events = %v, want none", events)
			}
			if len(errs) != 1 || !strings.Contains(errs[0], `sequential agent "seq": stored agent state`) {
				t.Errorf("errors = %q, want one containing %q", errs, `sequential agent "seq": stored agent state`)
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
		seq := newSequentialAgent(t, []agent.Agent{textAgent(t, "a1", &a1Runs), testutil.NewInputWorkflow(t, "wf")}, "seq")
		svc := resumeSessionService(t)
		r := resumeRunner(t, seq, svc)
		turn1, errs := resumeCollect(t, r, genai.NewContentFromText("go", genai.RoleUser))
		if len(errs) != 0 {
			t.Fatalf("turn 1 errors = %v", errs)
		}
		if !slices.Contains(turn1, "seq:STATE(current_sub_agent=wf)") {
			t.Fatalf("turn 1 events = %v, want seq's checkpoint for wf", turn1)
		}
		got, err := svc.Get(t.Context(), &session.GetRequest{AppName: resumeApp, UserID: resumeUser, SessionID: resumeSession})
		if err != nil {
			t.Fatal(err)
		}
		inv := got.Session.Events().At(got.Session.Events().Len() - 1).InvocationID
		stale := session.NewEvent(context.Background(), inv)
		stale.Author = "seq"
		stale.Actions.AgentState = map[string]any{"current_sub_agent": "a1"}
		if err := svc.AppendEvent(t.Context(), got.Session, stale); err != nil {
			t.Fatal(err)
		}

		turn2, errs := resumeCollect(t, r, testutil.InputResponse("b", "approved"))

		if len(errs) != 0 {
			t.Errorf("turn 2 errors = %v, want none", errs)
		}
		var seqEvents []string
		for _, ev := range turn2 {
			if strings.HasPrefix(ev, "seq:") {
				seqEvents = append(seqEvents, ev)
			}
		}
		if diff := cmp.Diff([]string{"seq:STATE(current_sub_agent=wf)", "seq:END"}, seqEvents); diff != "" {
			t.Errorf("seq's turn 2 events (-want +got):\n%s", diff)
		}
		if !slices.Contains(turn2, "d:D") {
			t.Errorf("turn 2 events = %v, want d's output", turn2)
		}
		if got := a1Runs.Load(); got != 1 {
			t.Errorf("a1 ran %d times over both turns, want 1", got)
		}
	})
}

// TestSequentialAgent_ResumableSubAgentErrorOrEarlyStopWithholdsEndEvent
// checks that a resumable sequential agent writes no end event when a
// sub-agent fails, as adk-python's exception skips it, or when the consumer
// stops. It fails if the end event is yielded on those paths (for example from
// a defer), or if Run goes on after the consumer refused a checkpoint.
func TestSequentialAgent_ResumableSubAgentErrorOrEarlyStopWithholdsEndEvent(t *testing.T) {
	t.Run("sub-agent error", func(t *testing.T) {
		var a1Runs, a2Runs, a3Runs atomic.Int32
		seq := newSequentialAgent(t, []agent.Agent{
			resumeAgent(t, "a1", &a1Runs, false, nil, "ok"),
			resumeAgent(t, "a2", &a2Runs, false, errors.New("a2 boom")),
			resumeAgent(t, "a3", &a3Runs, false, nil, "late"),
		}, "seq")

		events, errs := resumeCollect(t, resumeRunner(t, seq, resumeSessionService(t)), genai.NewContentFromText("go", genai.RoleUser))

		want := []string{"seq:STATE(current_sub_agent=a1)", "a1:ok", "seq:STATE(current_sub_agent=a2)"}
		if diff := cmp.Diff(want, events); diff != "" {
			t.Errorf("events (-want +got):\n%s", diff)
		}
		if diff := cmp.Diff([]string{"a2 boom"}, errs); diff != "" {
			t.Errorf("errors (-want +got):\n%s", diff)
		}
		if got := a3Runs.Load(); got != 0 {
			t.Errorf("a3 ran %d times, want 0", got)
		}
	})
	t.Run("consumer stops at first checkpoint", func(t *testing.T) {
		var a1Runs atomic.Int32
		seq := newSequentialAgent(t, []agent.Agent{resumeAgent(t, "a1", &a1Runs, false, nil, "ok")}, "seq")
		r := resumeRunner(t, seq, resumeSessionService(t))

		var events []string
		for ev, err := range r.Run(t.Context(), resumeUser, resumeSession, genai.NewContentFromText("go", genai.RoleUser), agent.RunConfig{}) {
			if err != nil {
				t.Fatal(err)
			}
			events = append(events, resumeDescribe(ev))
			break
		}

		if diff := cmp.Diff([]string{"seq:STATE(current_sub_agent=a1)"}, events); diff != "" {
			t.Errorf("events (-want +got):\n%s", diff)
		}
		if got := a1Runs.Load(); got != 0 {
			t.Errorf("a1 ran %d times, want 0", got)
		}
	})
}
