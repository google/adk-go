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

package workflow

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/session"
)

// branchTestEvent is a non-partial event of invocation "inv" at node path
// path and branch branch, raising the given interrupts.
func branchTestEvent(path, branch string, interrupts ...string) *session.Event {
	return &session.Event{
		InvocationID:       "inv",
		Branch:             branch,
		NodeInfo:           &session.NodeInfo{Path: path},
		LongRunningToolIDs: interrupts,
	}
}

// TestScanHistory_NodeBranchComesFromItsOwnEvents covers the branch rule of
// scanHistory: a node's branch is the branch of its own first event of the
// invocation, and a descendant's interrupt supplies it only for a node that
// has no event of its own.
//
// Fails if a descendant's interrupt overrides the branch of a node that has
// an own event (case A), if an own event does not set the branch (case B),
// or if the fallback to the descendant's branch is removed (case C).
func TestScanHistory_NodeBranchComesFromItsOwnEvents(t *testing.T) {
	nodesByName := map[string]Node{"root_agent": newDummyNode("root_agent")}
	tests := []struct {
		name   string
		events sliceEvents
		want   string
	}{
		{
			name: "A_own_call_event_then_tool_interrupt",
			events: sliceEvents{
				branchTestEvent("root_agent", ""),
				branchTestEvent("root_agent/ask@call_1", "ask@call_1", "x"),
			},
			want: "",
		},
		{
			name: "B_tool_interrupt_then_own_interrupt",
			events: sliceEvents{
				branchTestEvent("root_agent/ask@call_1", "ask@call_1", "x"),
				branchTestEvent("root_agent", "root.a", "y"),
			},
			want: "root.a",
		},
		{
			name: "C_no_own_event",
			events: sliceEvents{
				branchTestEvent("root_agent/ask@call_1", "ask@call_1", "x"),
			},
			want: "ask@call_1",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			scans := scanHistory(tt.events, nodesByName, "inv", "")

			scan := scans["root_agent"]
			if scan == nil {
				t.Fatalf("scanHistory returned no scan for root_agent; scans = %v", scans)
			}
			if scan.branch != tt.want {
				t.Errorf("root_agent branch = %q, want %q", scan.branch, tt.want)
			}
		})
	}
}

// branchAskChild returns a dynamic node "child" that pauses on interrupt
// "q1" and, once answered, returns the answer. Each activation logs the
// branch it runs on and the answer it got (nil before the answer).
func branchAskChild(log *faninLog) Node {
	return NewDynamicNode("child", func(ctx agent.Context, _ any, emit func(*session.Event) error) (any, error) {
		reply, err := ResumeOrRequestInput(ctx, emit, session.RequestInput{InterruptID: "q1", Message: "pick"})
		log.add(faninObs{Branch: ctx.Branch(), Value: reply})
		return reply, err
	}, NodeConfig{})
}

// branchOrchestrator returns a dynamic node "orch" that emits one event of
// its own when emitOwn is set, then runs child with opts and returns the
// child's output.
func branchOrchestrator(child Node, emitOwn bool, opts ...RunNodeOption) Node {
	return NewDynamicNode("orch", func(ctx agent.Context, _ any, emit func(*session.Event) error) (any, error) {
		if emitOwn {
			ev := session.NewEvent(ctx, ctx.InvocationID())
			ev.Content = genai.NewContentFromText("orch started", genai.RoleModel)
			if err := emit(ev); err != nil {
				return nil, err
			}
		}
		return RunNode[any](ctx, child, nil, opts...)
	}, NodeConfig{})
}

// branchPausedRun runs wf until it pauses and returns the context it ran
// under, with a session holding the run's history, and the run state
// ReconstructRunState rebuilds from that history.
func branchPausedRun(t *testing.T, wf *Workflow) (*MockInvocationContext, *RunState) {
	t.Helper()
	ctx := newMockCtx(t)
	var history sliceEvents
	for ev, err := range wf.Run(ctx) {
		if err != nil {
			t.Fatalf("turn 1 Run error = %v", err)
		}
		history = append(history, ev)
	}
	ctx.sess = &eventsSession{events: history}
	state, err := wf.ReconstructRunState(ctx.sess, ctx.InvocationID())
	if err != nil {
		t.Fatalf("ReconstructRunState error = %v", err)
	}
	return ctx, state
}

// branchResume resumes wf at path "<name>@1", the path Run gave it, with
// responses, and returns the events of the resume turn.
func branchResume(t *testing.T, ctx *MockInvocationContext, wf *Workflow, state *RunState, responses map[string]any) []*session.Event {
	t.Helper()
	path := wf.name + "@1"
	resumeCtx := agent.PromoteWithDelta(ctx, &agent.CommonContextDelta{Path: &path})
	var events []*session.Event
	for ev, err := range wf.Resume(resumeCtx, state, responses) {
		if err != nil {
			t.Fatalf("Resume error = %v", err)
		}
		events = append(events, ev)
	}
	return events
}

// branchObs is the branch and output of one event of a node.
type branchObs struct {
	Branch string
	Output any
}

// branchEventsOf returns the branch and output of every event of node
// name, in stream order.
func branchEventsOf(events []*session.Event, name string) []branchObs {
	var out []branchObs
	for _, ev := range events {
		if ev.NodeInfo.Name() != name {
			continue
		}
		out = append(out, branchObs{Branch: ev.Branch, Output: ev.Output})
	}
	return out
}

// TestReconstructRunState_ChildOnSubBranchPausesAndResumes covers a dynamic
// orchestrator on the root branch whose child, on its own sub-branch,
// pauses and is answered: the resumed orchestrator stays on the root branch
// and the resumed child runs on its sub-branch again.
//
// Fails if the branch rule of scanHistory gives the orchestrator its
// child's sub-branch (the orchestrator would resume on "child@1" and the
// child on "child@1.child@1").
func TestReconstructRunState_ChildOnSubBranchPausesAndResumes(t *testing.T) {
	var childRuns faninLog
	orch := branchOrchestrator(branchAskChild(&childRuns), true, WithUseSubBranch())
	wf, err := New("wf", []Edge{{From: Start, To: orch}})
	if err != nil {
		t.Fatal(err)
	}
	ctx, state := branchPausedRun(t, wf)

	if got := nodeState(t, state, "orch").Branch; got != "" {
		t.Errorf("rebuilt orch branch = %q, want %q", got, "")
	}
	events := branchResume(t, ctx, wf, state, map[string]any{"q1": "A"})

	wantOrch := []branchObs{{Branch: ""}, {Branch: "", Output: "A"}}
	if diff := cmp.Diff(wantOrch, branchEventsOf(events, "orch")); diff != "" {
		t.Errorf("resume turn orch events mismatch (-want +got):\n%s", diff)
	}
	wantChild := []faninObs{{Branch: "child@1"}, {Branch: "child@1", Value: "A"}}
	if diff := cmp.Diff(wantChild, childRuns.get()); diff != "" {
		t.Errorf("child activations mismatch (-want +got):\n%s", diff)
	}
}

// TestResume_FanOutMemberKeepsItsOwnBranch covers a node that a fan-out puts
// on its own branch ("orch@1"), emits an event of its own and runs a child
// that pauses: the node is rebuilt on its own branch, whether the child
// paused on that branch or on a branch of its own, as a tool call does.
//
// Fails if a descendant's interrupt replaces the branch of a node that has
// an own event, or if a node's first own event does not set its branch.
func TestResume_FanOutMemberKeepsItsOwnBranch(t *testing.T) {
	// other yields nothing, so orch stays the workflow's only terminal output.
	other := newFaninNode("other", NodeConfig{}, func(agent.Context, any, func(*session.Event, error) bool) {})

	t.Run("child_on_orch_branch", func(t *testing.T) {
		orch := branchOrchestrator(branchAskChild(&faninLog{}), true)
		wf, err := New("wf", []Edge{{From: Start, To: orch}, {From: Start, To: other}})
		if err != nil {
			t.Fatal(err)
		}
		ctx, state := branchPausedRun(t, wf)

		if got := nodeState(t, state, "orch").Branch; got != "orch@1" {
			t.Errorf("rebuilt orch branch = %q, want %q", got, "orch@1")
		}
		events := branchResume(t, ctx, wf, state, map[string]any{"q1": "A"})
		want := []branchObs{{Branch: "orch@1", Output: "A"}}
		if diff := cmp.Diff(want, branchEventsOf(events, "child")); diff != "" {
			t.Errorf("resume turn child events mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("child_on_override_branch_with_fan_out", func(t *testing.T) {
		orch := branchOrchestrator(branchAskChild(&faninLog{}), true, WithOverrideBranch("ask@call_1"), WithRunID("call_1"))
		wf, err := New("wf", []Edge{{From: Start, To: orch}, {From: Start, To: other}})
		if err != nil {
			t.Fatal(err)
		}
		_, state := branchPausedRun(t, wf)

		if got := nodeState(t, state, "orch").Branch; got != "orch@1" {
			t.Errorf("rebuilt orch branch = %q, want %q", got, "orch@1")
		}
	})

	t.Run("child_on_override_branch_without_fan_out", func(t *testing.T) {
		orch := branchOrchestrator(branchAskChild(&faninLog{}), true, WithOverrideBranch("ask@call_1"), WithRunID("call_1"))
		wf, err := New("wf", []Edge{{From: Start, To: orch}})
		if err != nil {
			t.Fatal(err)
		}
		_, state := branchPausedRun(t, wf)

		if got := nodeState(t, state, "orch").Branch; got != "" {
			t.Errorf("rebuilt orch branch = %q, want %q", got, "")
		}
	})
}
