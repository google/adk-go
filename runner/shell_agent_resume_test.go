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

package runner_test

import (
	"fmt"
	"iter"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/internal/testutil"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
)

// shellDescribe renders ev as "<author>:END" for an end-of-agent event;
// "<author>:STATE({})" for an empty agent state, "<author>:STATE(<sub>)" for a
// sequential checkpoint and "<author>:STATE(<sub>,<times_looped>)" for a loop
// checkpoint; otherwise "<author>:" followed by "FC(name)" for each function
// call and the text of each text part.
func shellDescribe(ev *session.Event) string {
	if ev.Actions.EndOfAgent != nil && *ev.Actions.EndOfAgent {
		return ev.Author + ":END"
	}
	if st := ev.Actions.AgentState; st != nil {
		switch {
		case len(st) == 0:
			return ev.Author + ":STATE({})"
		case st["times_looped"] != nil:
			return fmt.Sprintf("%s:STATE(%v,%v)", ev.Author, st["current_sub_agent"], st["times_looped"])
		default:
			return fmt.Sprintf("%s:STATE(%v)", ev.Author, st["current_sub_agent"])
		}
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

func shellDescribeAll(evs []*session.Event) []string {
	var out []string
	for _, ev := range evs {
		out = append(out, shellDescribe(ev))
	}
	return out
}

// shellLLM returns an LLM agent named name whose model answers with replies in
// order, and the model, whose Requests count its calls. A reply that starts
// with "FC:" is a call to the long-running tool "approve" with the ID after
// the prefix; any other reply is text.
func shellLLM(t *testing.T, name string, replies ...string) (agent.Agent, *testutil.MockModel) {
	t.Helper()
	m := &testutil.MockModel{}
	for _, r := range replies {
		if id, ok := strings.CutPrefix(r, "FC:"); ok {
			m.Responses = append(m.Responses, fnCallContent(id, "approve", map[string]any{}))
			continue
		}
		m.Responses = append(m.Responses, genai.NewContentFromText(r, genai.RoleModel))
	}
	approve, err := functiontool.New(functiontool.Config{Name: "approve", Description: "asks for approval", IsLongRunning: true},
		func(agent.Context, struct{}) (map[string]any, error) { return nil, nil })
	if err != nil {
		t.Fatal(err)
	}
	a, err := llmagent.New(llmagent.Config{Name: name, Model: m, Tools: []tool.Tool{approve}})
	if err != nil {
		t.Fatal(err)
	}
	return a, m
}

// shellTextAgent returns an agent named name that yields "<name> text" and
// counts its runs in runs.
func shellTextAgent(t *testing.T, name string, runs *atomic.Int32) agent.Agent {
	t.Helper()
	a, err := agent.New(agent.Config{Name: name, Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
		return func(yield func(*session.Event, error) bool) {
			runs.Add(1)
			ev := session.NewEvent(ctx, ctx.InvocationID())
			ev.Author = name
			ev.Content = genai.NewContentFromText(name+" text", genai.RoleModel)
			yield(ev, nil)
		}
	}})
	if err != nil {
		t.Fatal(err)
	}
	return a
}

// shellTurns runs root on a resumable runner for a first turn with the text
// "go" and a second turn that answers the call "call-1" to "approve", and
// returns the described events of each turn.
func shellTurns(t *testing.T, root agent.Agent) (turn1, turn2 []string) {
	t.Helper()
	svc := session.InMemoryService()
	seedResumeSession(t, svc)
	r := newResumeRunner(t, root, svc, true)
	turn1 = shellDescribeAll(runTurn(t, r, genai.NewContentFromText("go", genai.RoleUser)))
	turn2 = shellDescribeAll(runTurn(t, r, responseContent("call-1", "approve", map[string]any{"ok": true})))
	return turn1, turn2
}

// TestRun_ResumableSequentialPauseAndResumeContinues checks that answering a
// paused call resumes a sequential agent where it paused: the caller
// finishes, the sub-agents after it run in the same turn, and a sub-agent that
// ran before the pause does not run again (agents/sequential_agent.py with
// runners.py, the resume of a paused invocation). It fails if the runner does
// not seed the stored states, if the sequential agent ignores its checkpoint,
// or if the pause carries over into the resume turn.
func TestRun_ResumableSequentialPauseAndResumeContinues(t *testing.T) {
	tests := []struct {
		name                  string
		callerFirst           bool
		wantTurn1, wantTurn2  []string
		wantOtherCallsInTurn1 int
		wantOtherCallsTotal   int
	}{
		{
			name:        "caller first",
			callerFirst: true,
			wantTurn1:   []string{"root:STATE(caller)", "caller:FC(approve)"},
			wantTurn2: []string{
				"caller:approved", "caller:END",
				"root:STATE(other)", "other:other answer", "other:END",
				"root:END",
			},
			wantOtherCallsInTurn1: 0,
			wantOtherCallsTotal:   1,
		},
		{
			name:        "other first",
			callerFirst: false,
			wantTurn1: []string{
				"root:STATE(other)", "other:other answer", "other:END",
				"root:STATE(caller)", "caller:FC(approve)",
			},
			wantTurn2:             []string{"caller:approved", "caller:END", "root:END"},
			wantOtherCallsInTurn1: 1,
			wantOtherCallsTotal:   1,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			caller, _ := shellLLM(t, "caller", "FC:call-1", "approved")
			other, otherModel := shellLLM(t, "other", "other answer", "other answer 2")
			subAgents := []agent.Agent{caller, other}
			if !tc.callerFirst {
				subAgents = []agent.Agent{other, caller}
			}
			svc := session.InMemoryService()
			seedResumeSession(t, svc)
			r := newResumeRunner(t, mustSequential(t, "root", subAgents...), svc, true)

			turn1 := shellDescribeAll(runTurn(t, r, genai.NewContentFromText("go", genai.RoleUser)))
			otherCallsInTurn1 := len(otherModel.Requests)
			turn2 := shellDescribeAll(runTurn(t, r, responseContent("call-1", "approve", map[string]any{"ok": true})))

			if diff := cmp.Diff(tc.wantTurn1, turn1); diff != "" {
				t.Errorf("turn 1 events (-want +got):\n%s", diff)
			}
			if diff := cmp.Diff(tc.wantTurn2, turn2); diff != "" {
				t.Errorf("turn 2 events (-want +got):\n%s", diff)
			}
			if otherCallsInTurn1 != tc.wantOtherCallsInTurn1 {
				t.Errorf("other made %d model calls in turn 1, want %d", otherCallsInTurn1, tc.wantOtherCallsInTurn1)
			}
			if got := len(otherModel.Requests); got != tc.wantOtherCallsTotal {
				t.Errorf("other made %d model calls in total, want %d", got, tc.wantOtherCallsTotal)
			}
		})
	}
}

// TestRun_ResumableLoopPauseAndResumeContinuesPasses checks that answering a
// call that paused a loop finishes the paused pass from the caller on, without
// running the sub-agents before it again, and then runs the remaining passes
// (agents/loop_agent.py, _get_start_state). It fails if the loop does not
// restore its pass count or its start index.
func TestRun_ResumableLoopPauseAndResumeContinuesPasses(t *testing.T) {
	first, firstModel := shellLLM(t, "first", "f1", "f2")
	caller, _ := shellLLM(t, "caller", "FC:call-1", "approved", "c2")
	last, lastModel := shellLLM(t, "last", "l1", "l2")

	turn1, turn2 := shellTurns(t, mustLoop(t, "loop", 2, first, caller, last))

	wantTurn1 := []string{"loop:STATE(first,0)", "first:f1", "first:END", "loop:STATE(caller,0)", "caller:FC(approve)"}
	if diff := cmp.Diff(wantTurn1, turn1); diff != "" {
		t.Errorf("turn 1 events (-want +got):\n%s", diff)
	}
	wantTurn2 := []string{
		"caller:approved", "caller:END",
		"loop:STATE(last,0)", "last:l1", "last:END",
		"loop:STATE(first,1)", "first:f2", "first:END",
		"loop:STATE(caller,1)", "caller:c2", "caller:END",
		"loop:STATE(last,1)", "last:l2", "last:END",
		"loop:END",
	}
	if diff := cmp.Diff(wantTurn2, turn2); diff != "" {
		t.Errorf("turn 2 events (-want +got):\n%s", diff)
	}
	if got := len(firstModel.Requests); got != 2 {
		t.Errorf("first made %d model calls, want 2", got)
	}
	if got := len(lastModel.Requests); got != 2 {
		t.Errorf("last made %d model calls, want 2", got)
	}
}

// TestRun_ResumableParallelSkipsEndedBranchesOnResume checks that answering
// a call that paused one branch of a parallel agent runs only the branches
// that have not ended, writes no second {} checkpoint, resumes the paused
// sequential branch at the caller, and then ends the parallel agent
// (agents/parallel_agent.py, the end_of_agents checks). It fails if the
// parallel agent runs an ended branch again, writes its checkpoint again, or
// if the end of an LLM agent is not recorded.
func TestRun_ResumableParallelSkipsEndedBranchesOnResume(t *testing.T) {
	a1, a1Model := shellLLM(t, "a1", "a1 text")
	a2, a2Model := shellLLM(t, "a2", "a2 text")
	a3, a3Model := shellLLM(t, "a3", "FC:call-1", "approved")
	par := mustParallel(t, "par", mustSequential(t, "seq", a2, a3), a1)

	turn1, turn2 := shellTurns(t, par)

	if len(turn1) == 0 || turn1[0] != "par:STATE({})" {
		t.Errorf("turn 1 events = %v, want par:STATE({}) first", turn1)
	}
	if diff := cmp.Diff([]string{"a3:approved", "a3:END", "seq:END", "par:END"}, turn2); diff != "" {
		t.Errorf("turn 2 events (-want +got):\n%s", diff)
	}
	if got := len(a1Model.Requests); got != 1 {
		t.Errorf("a1 made %d model calls, want 1", got)
	}
	if got := len(a2Model.Requests); got != 1 {
		t.Errorf("a2 made %d model calls, want 1", got)
	}
	if got := len(a3Model.Requests); got != 2 {
		t.Errorf("a3 made %d model calls, want 2", got)
	}
}

// TestRun_ResumedLoopOverParallelOverSequential_Race checks, under the race
// detector, a loop that resumes from seeded states over a parallel agent whose
// branches are sequential agents: in the first pass seqA resumes at a2 while
// seqB runs from its start, and after that pass the loop clears every
// descendant, so in the second pass both sequential agents run in full
// (agents/loop_agent.py, ctx.reset_sub_agent_states). It fails if the loop
// does not reset its descendants, resets only its direct sub-agents, or races
// with the parallel agent's branches.
func TestRun_ResumedLoopOverParallelOverSequential_Race(t *testing.T) {
	var aRuns, bRuns atomic.Int32
	seqA := mustSequential(t, "seqA", shellTextAgent(t, "a1", &aRuns), shellTextAgent(t, "a2", &aRuns))
	seqB := mustSequential(t, "seqB", shellTextAgent(t, "b1", &bRuns), shellTextAgent(t, "b2", &bRuns))
	loop := mustLoop(t, "loop", 2, mustParallel(t, "par", seqA, seqB))
	svc := session.InMemoryService()
	seedResumeSession(t, svc,
		seedUser("inv-1", "go"),
		seedAgentState("inv-1", "loop", map[string]any{"current_sub_agent": "par", "times_looped": 0}),
		seedAgentState("inv-1", "par", map[string]any{}),
		seedAgentState("inv-1", "seqA", map[string]any{"current_sub_agent": "a2"}),
		seedEnd("inv-1", "a1"),
	)

	events := shellDescribeAll(runTurn(t, newResumeRunner(t, loop, svc, true), nil, runner.WithInvocationID("inv-1")))

	if a, b := aRuns.Load(), bRuns.Load(); a != 3 || b != 4 {
		t.Errorf("seqA's agents ran %d times and seqB's %d times, want 3 and 4", a, b)
	}
	if len(events) == 0 || events[len(events)-1] != "loop:END" {
		t.Errorf("events = %v, want loop:END last", events)
	}
}
