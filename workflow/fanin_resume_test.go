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
	"errors"
	"fmt"
	"slices"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/session"
)

// faninInvocation is the invocation ID of every event a run under
// newMockCtx records.
const faninInvocation = "test-invocation-id"

// faninAsk yields a request for input id from ctx's activation, which
// parks the activation until a response arrives.
func faninAsk(ctx agent.Context, yield func(*session.Event, error) bool, id string) {
	yield(NewRequestInputEvent(ctx, session.RequestInput{InterruptID: id}), nil)
}

// faninResumeCtx returns a context with path "wf@1", the path Run gave
// the workflow on the first turn, so a resumed activation's events carry
// "wf@1/<name>@<run id>".
func faninResumeCtx(t *testing.T) agent.Context {
	t.Helper()
	ctx, _ := faninCtx(t, faninBudget)
	wfPath := "wf@1"
	return agent.PromoteWithDelta(ctx, &agent.CommonContextDelta{Path: &wfPath})
}

// faninReconstruct rebuilds the run state from history and fails the test
// when there is nothing to resume.
func faninReconstruct(t *testing.T, w *Workflow, history []*session.Event) *RunState {
	t.Helper()
	state, err := w.ReconstructRunState(fakeSession{events: sliceEvents(history)}, faninInvocation)
	if err != nil {
		t.Fatalf("ReconstructRunState: %v", err)
	}
	if state == nil {
		t.Fatal("ReconstructRunState returned no state")
	}
	return state
}

// faninUserResponse is the user event that answers interrupt id with v,
// as the runner persists it at the start of a resume turn.
func faninUserResponse(id string, v any) *session.Event {
	ev := &session.Event{Author: "user", InvocationID: faninInvocation}
	ev.Content = genai.NewContentFromFunctionResponse(WorkflowInputFunctionCallName, map[string]any{"response": v}, genai.RoleUser)
	ev.Content.Parts[0].FunctionResponse.ID = id
	return ev
}

// faninStripBranches clears Branch, which the resume tests do not check.
func faninStripBranches(obs []faninObs) []faninObs {
	for i := range obs {
		obs[i].Branch = ""
	}
	return obs
}

// TestPersistence_ResumeRunCounter covers B13 and ruling 6a: a node that
// paused on its second activation resumes as loop@2, and its next
// activation in the resumed run is loop@3.
//
// Fails if ReconstructRunState does not restore the run counter from
// history (the resumed node would run as loop@1), or if the resumed
// activation allocates a new run id.
func TestPersistence_ResumeRunCounter(t *testing.T) {
	rerun := true
	var calls atomic.Int32
	loop := newFaninNode("loop", NodeConfig{RerunOnResume: &rerun}, func(ctx agent.Context, _ any, yield func(*session.Event, error) bool) {
		if v, ok := ctx.ResumedInput("ask-2"); ok {
			faninEmit(ctx, yield, fmt.Sprintf("iter-2:%v", v), "again")
			return
		}
		switch calls.Add(1) {
		case 1:
			faninEmit(ctx, yield, "iter-1", "again")
		case 2:
			faninAsk(ctx, yield, "ask-2")
		default:
			faninEmit(ctx, yield, "iter-3:ok", "done")
		}
	})
	w, err := New("wf", []Edge{
		{From: Start, To: loop},
		{From: loop, To: loop, Route: StringRoute("again")},
	})
	if err != nil {
		t.Fatal(err)
	}
	ctx, _ := faninCtx(t, faninBudget)
	turn1 := faninRun(t, w.Run(ctx))
	if turn1.err != nil {
		t.Fatalf("turn 1 Run error = %v, want nil", turn1.err)
	}
	state := faninReconstruct(t, w, turn1.events)

	turn2 := faninRun(t, w.Resume(faninResumeCtx(t), state, map[string]any{"ask-2": "ok"}))

	if turn2.err != nil {
		t.Errorf("Resume error = %v, want nil", turn2.err)
	}
	wantPaths := []string{"wf@1/loop@2", "wf@1/loop@3"}
	if diff := cmp.Diff(wantPaths, faninPaths(faninActivations(turn2.events, "loop"))); diff != "" {
		t.Errorf("resume turn loop paths mismatch (-want +got):\n%s", diff)
	}
	outs := faninOutputs(turn2.events, "loop")
	if len(outs) == 0 || outs[len(outs)-1].Value != "iter-3:ok" {
		t.Errorf("resume turn loop outputs = %v, want the last to be iter-3:ok", faninValues(outs))
	}
}

// TestPersistence_RestoresQueuedTriggers covers B13 and ruling 6b: when
// c pauses on c@1 while b's trigger for c waits behind it, the resume
// finishes c@1 and then runs the waiting trigger as c@2 with b's output,
// whatever the order of b's output and c's pause in history.
//
// Fails if ReconstructRunState drops the trigger that had not started, or
// if Resume does not return it to the trigger buffer.
func TestPersistence_RestoresQueuedTriggers(t *testing.T) {
	rerun := true
	cStarted := make(chan struct{})
	var cOnce sync.Once
	a := newFaninNode("a", NodeConfig{}, faninEmitter("from-A", "r"))
	b := newFaninNode("b", NodeConfig{}, func(ctx agent.Context, _ any, yield func(*session.Event, error) bool) {
		faninWait(cStarted, faninBudget)
		faninEmit(ctx, yield, "from-B", "r")
	})
	c := newFaninNode("c", NodeConfig{RerunOnResume: &rerun}, func(ctx agent.Context, in any, yield func(*session.Event, error) bool) {
		cOnce.Do(func() { close(cStarted) })
		if in != "from-A" {
			faninEmit(ctx, yield, fmt.Sprintf("c:%v:no-interrupt", in))
			return
		}
		if v, ok := ctx.ResumedInput("ask-c"); ok {
			faninEmit(ctx, yield, fmt.Sprintf("c:%v:%v", in, v))
			return
		}
		faninAsk(ctx, yield, "ask-c")
	})
	w, err := New("wf", []Edge{
		{From: Start, To: a},
		{From: Start, To: b},
		{From: a, To: c, Route: StringRoute("r")},
		{From: b, To: c, Route: StringRoute("r")},
	})
	if err != nil {
		t.Fatal(err)
	}
	ctx, _ := faninCtx(t, faninBudget)
	turn1 := faninRun(t, w.Run(ctx))
	if turn1.err != nil {
		t.Fatalf("turn 1 Run error = %v, want nil", turn1.err)
	}
	var bOutput *session.Event
	var rest []*session.Event
	for _, ev := range turn1.events {
		if ev.Output != nil && ev.NodeInfo.Name() == "b" {
			bOutput = ev
			continue
		}
		rest = append(rest, ev)
	}
	pause := slices.IndexFunc(rest, func(ev *session.Event) bool {
		return ev.NodeInfo.Name() == "c" && len(ev.LongRunningToolIDs) > 0
	})
	if bOutput == nil || pause < 0 {
		t.Fatal("turn 1 history lacks b's output or c's pause")
	}
	want := []faninObs{
		{Path: "wf@1/c@1", Value: "c:from-A:approved"},
		{Path: "wf@1/c@2", Value: "c:from-B:no-interrupt"},
	}
	for _, tc := range []struct {
		name string
		at   int // where b's output goes in the history
	}{
		{name: "b_output_before_interrupt", at: pause},
		{name: "b_output_after_interrupt", at: pause + 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			history := slices.Insert(slices.Clone(rest), tc.at, bOutput)
			state := faninReconstruct(t, w, history)

			turn2 := faninRun(t, w.Resume(faninResumeCtx(t), state, map[string]any{"ask-c": "approved"}))

			if turn2.err != nil {
				t.Errorf("Resume error = %v, want nil", turn2.err)
			}
			if diff := cmp.Diff(want, faninStripBranches(faninOutputs(turn2.events, "c"))); diff != "" {
				t.Errorf("resume turn c outputs mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// TestPersistence_MatchesPausedActivationInput covers B13 and ruling 6c:
// when c completed c@1 with a's output and paused on c@2, triggered by b,
// the resume re-enters c@2 with b's output, not with the first
// predecessor's in edge order.
//
// Fails if ReconstructRunState takes the input from predecessorInput (the
// first predecessor with an output) instead of the replayed trigger of the
// paused activation.
func TestPersistence_MatchesPausedActivationInput(t *testing.T) {
	rerun := true
	cDone := make(chan struct{})
	var cOnce sync.Once
	a := newFaninNode("a", NodeConfig{}, faninEmitter("from-A", "r"))
	b := newFaninNode("b", NodeConfig{}, func(ctx agent.Context, _ any, yield func(*session.Event, error) bool) {
		faninWait(cDone, faninBudget)
		time.Sleep(50 * time.Millisecond) // let c@1's completion land first
		faninEmit(ctx, yield, "from-B", "r")
	})
	c := newFaninNode("c", NodeConfig{RerunOnResume: &rerun}, func(ctx agent.Context, in any, yield func(*session.Event, error) bool) {
		if in == "from-A" {
			faninEmit(ctx, yield, fmt.Sprintf("done:%v", in))
			cOnce.Do(func() { close(cDone) })
			return
		}
		if v, ok := ctx.ResumedInput("ask-c2"); ok {
			faninEmit(ctx, yield, fmt.Sprintf("%v:%v", in, v))
			return
		}
		faninAsk(ctx, yield, "ask-c2")
	})
	w, err := New("wf", []Edge{
		{From: Start, To: a},
		{From: Start, To: b},
		{From: a, To: c, Route: StringRoute("r")},
		{From: b, To: c, Route: StringRoute("r")},
	})
	if err != nil {
		t.Fatal(err)
	}
	ctx, _ := faninCtx(t, faninBudget)
	turn1 := faninRun(t, w.Run(ctx))
	if turn1.err != nil {
		t.Fatalf("turn 1 Run error = %v, want nil", turn1.err)
	}
	state := faninReconstruct(t, w, turn1.events)

	turn2 := faninRun(t, w.Resume(faninResumeCtx(t), state, map[string]any{"ask-c2": "go"}))

	if turn2.err != nil {
		t.Errorf("Resume error = %v, want nil", turn2.err)
	}
	want := []faninObs{{Path: "wf@1/c@2", Value: "from-B:go"}}
	if diff := cmp.Diff(want, faninStripBranches(faninOutputs(turn2.events, "c"))); diff != "" {
		t.Errorf("resume turn c outputs mismatch (-want +got):\n%s", diff)
	}
}

// TestResume_TwoHandoffAskersFeedingOneJoin_RunsJoinOnce covers amendment
// 6: one Resume that answers two handoff askers feeding a JoinNode runs
// the join once, with both responses.
//
// Fails if Resume's second pass buffers the join once per answered asker.
func TestResume_TwoHandoffAskersFeedingOneJoin_RunsJoinOnce(t *testing.T) {
	handoff := false
	h1 := newFaninNode("h1", NodeConfig{RerunOnResume: &handoff}, func(ctx agent.Context, _ any, yield func(*session.Event, error) bool) {
		faninAsk(ctx, yield, "ask-h1")
	})
	h2 := newFaninNode("h2", NodeConfig{RerunOnResume: &handoff}, func(ctx agent.Context, _ any, yield func(*session.Event, error) bool) {
		faninAsk(ctx, yield, "ask-h2")
	})
	join := NewJoinNode("J")
	w, err := New("wf", []Edge{
		{From: Start, To: h1},
		{From: Start, To: h2},
		{From: h1, To: join},
		{From: h2, To: join},
	})
	if err != nil {
		t.Fatal(err)
	}
	ctx, _ := faninCtx(t, faninBudget)
	turn1 := faninRun(t, w.Run(ctx))
	if turn1.err != nil {
		t.Fatalf("turn 1 Run error = %v, want nil", turn1.err)
	}
	state := faninReconstruct(t, w, turn1.events)

	// A slow consumer: a second activation of J, if one was started,
	// yields its output while the first one's is being consumed.
	slowOnJ := func(ev *session.Event) {
		if ev.NodeInfo.Name() == "J" {
			time.Sleep(50 * time.Millisecond)
		}
	}
	turn2 := faninRunObserved(t, w.Resume(faninResumeCtx(t), state, map[string]any{"ask-h1": "r1", "ask-h2": "r2"}), slowOnJ)

	if turn2.err != nil {
		t.Errorf("Resume error = %v, want nil", turn2.err)
	}
	want := []faninObs{{Path: "wf@1/J@1", Value: map[string]any{"h1": "r1", "h2": "r2"}}}
	if diff := cmp.Diff(want, faninStripBranches(faninOutputs(turn2.events, "J"))); diff != "" {
		t.Errorf("resume turn J outputs mismatch (-want +got):\n%s", diff)
	}
}

// TestResume_HandoffSuccessorRunsAfterOtherPredecessorRanEarlier covers
// amendment 7: when a handoff asker is answered, its successor runs with
// the response even though another predecessor already ran that
// successor on the first turn. Python runs wf@1/x@2 with {"v": "H"}.
//
// Fails if Resume skips a handoff successor because the successor's name
// completed on an earlier turn, instead of deciding by whether the asker
// was answered on an earlier turn.
func TestResume_HandoffSuccessorRunsAfterOtherPredecessorRanEarlier(t *testing.T) {
	handoff := false
	var xLog faninLog
	a := newFaninNode("a", NodeConfig{}, faninEmitter("A", "r"))
	h := newFaninNode("h", NodeConfig{RerunOnResume: &handoff}, func(ctx agent.Context, _ any, yield func(*session.Event, error) bool) {
		faninAsk(ctx, yield, "ask-h")
	})
	x := newFaninNode("x", NodeConfig{}, func(ctx agent.Context, in any, yield func(*session.Event, error) bool) {
		xLog.add(faninObs{Path: ctx.Path(), Value: in})
		faninEmit(ctx, yield, "x-done")
	})
	w, err := New("wf", []Edge{
		{From: Start, To: a},
		{From: Start, To: h},
		{From: a, To: x, Route: StringRoute("r")},
		{From: h, To: x},
	})
	if err != nil {
		t.Fatal(err)
	}
	ctx, _ := faninCtx(t, faninBudget)
	turn1 := faninRun(t, w.Run(ctx))
	if turn1.err != nil {
		t.Fatalf("turn 1 Run error = %v, want nil", turn1.err)
	}
	ranInTurn1 := len(xLog.get())
	state := faninReconstruct(t, w, turn1.events)

	turn2 := faninRun(t, w.Resume(faninResumeCtx(t), state, map[string]any{"ask-h": map[string]any{"v": "H"}}))

	if turn2.err != nil {
		t.Errorf("Resume error = %v, want nil", turn2.err)
	}
	want := []faninObs{{Path: "wf@1/x@2", Value: map[string]any{"v": "H"}}}
	if diff := cmp.Diff(want, xLog.get()[ranInTurn1:]); diff != "" {
		t.Errorf("resume turn x activations mismatch (-want +got):\n%s", diff)
	}
}

// TestPersistence_UnansweredInterruptKeepsActivationInput covers
// amendment 8, first case: when the run state is rebuilt before the
// response is in history (a direct Resume call), the paused node keeps
// its activation's input.
//
// Fails if inferNodeState leaves Input unset for a node with unanswered
// interrupts.
func TestPersistence_UnansweredInterruptKeepsActivationInput(t *testing.T) {
	rerun := true
	var cLog faninLog
	p := newFaninNode("p", NodeConfig{}, faninEmitter("P"))
	c := newFaninNode("c", NodeConfig{RerunOnResume: &rerun}, func(ctx agent.Context, in any, yield func(*session.Event, error) bool) {
		if _, ok := ctx.ResumedInput("ask"); ok {
			cLog.add(faninObs{Path: ctx.Path(), Value: in})
			faninEmit(ctx, yield, "c-done")
			return
		}
		faninAsk(ctx, yield, "ask")
	})
	w, err := New("wf", []Edge{{From: Start, To: p}, {From: p, To: c}})
	if err != nil {
		t.Fatal(err)
	}
	ctx, _ := faninCtx(t, faninBudget)
	turn1 := faninRun(t, w.Run(ctx))
	if turn1.err != nil {
		t.Fatalf("turn 1 Run error = %v, want nil", turn1.err)
	}
	state := faninReconstruct(t, w, turn1.events)

	turn2 := faninRun(t, w.Resume(faninResumeCtx(t), state, map[string]any{"ask": "yes"}))

	if turn2.err != nil {
		t.Errorf("Resume error = %v, want nil", turn2.err)
	}
	want := []faninObs{{Path: "wf@1/c@1", Value: "P"}}
	if diff := cmp.Diff(want, cLog.get()); diff != "" {
		t.Errorf("resumed c activations mismatch (-want +got):\n%s", diff)
	}
}

// TestPersistence_SegmentReplaysAsLatestTerminalEvent covers amendment 8,
// second case: a node activation's outcome in history is its latest
// terminal event. An activation that failed and then succeeded on retry
// produced a trigger; one that yielded an output and then failed did not.
//
// Fails if the history replay takes an activation's first terminal event
// as its outcome.
func TestPersistence_SegmentReplaysAsLatestTerminalEvent(t *testing.T) {
	t.Run("failed_then_succeeded", func(t *testing.T) {
		// b@1 fails, then succeeds on retry and triggers c@2, which pauses.
		rerun := true
		cDone := make(chan struct{})
		var cOnce sync.Once
		var bCalls atomic.Int32
		a := newFaninNode("a", NodeConfig{}, faninEmitter("from-A", "r"))
		b := newFaninNode("b", NodeConfig{RetryConfig: retryAlways(2)}, func(ctx agent.Context, _ any, yield func(*session.Event, error) bool) {
			if bCalls.Add(1) == 1 {
				faninWait(cDone, faninBudget)
				time.Sleep(50 * time.Millisecond) // let c@1's completion land first
				yield(nil, errors.New("transient"))
				return
			}
			faninEmit(ctx, yield, "from-B", "r")
		})
		c := newFaninNode("c", NodeConfig{RerunOnResume: &rerun}, func(ctx agent.Context, in any, yield func(*session.Event, error) bool) {
			if in == "from-A" {
				faninEmit(ctx, yield, fmt.Sprintf("done:%v", in))
				cOnce.Do(func() { close(cDone) })
				return
			}
			if v, ok := ctx.ResumedInput("ask-c2"); ok {
				faninEmit(ctx, yield, fmt.Sprintf("%v:%v", in, v))
				return
			}
			faninAsk(ctx, yield, "ask-c2")
		})
		w, err := New("wf", []Edge{
			{From: Start, To: a},
			{From: Start, To: b},
			{From: a, To: c, Route: StringRoute("r")},
			{From: b, To: c, Route: StringRoute("r")},
		})
		if err != nil {
			t.Fatal(err)
		}
		ctx, _ := faninCtx(t, faninBudget)
		turn1 := faninRun(t, w.Run(ctx))
		if turn1.err != nil {
			t.Fatalf("turn 1 Run error = %v, want nil", turn1.err)
		}
		state := faninReconstruct(t, w, turn1.events)

		turn2 := faninRun(t, w.Resume(faninResumeCtx(t), state, map[string]any{"ask-c2": "go"}))

		if turn2.err != nil {
			t.Errorf("Resume error = %v, want nil", turn2.err)
		}
		want := []faninObs{{Path: "wf@1/c@2", Value: "from-B:go"}}
		if diff := cmp.Diff(want, faninStripBranches(faninOutputs(turn2.events, "c"))); diff != "" {
			t.Errorf("resume turn c outputs mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("output_then_failure", func(t *testing.T) {
		// c@1 pauses; b@1 yields an output and then fails, so it never
		// triggered c. The resume re-runs b, whose new output triggers c@2.
		rerun := true
		cPaused := make(chan struct{})
		var cOnce sync.Once
		var bCalls atomic.Int32
		a := newFaninNode("a", NodeConfig{}, faninEmitter("from-A", "r"))
		b := newFaninNode("b", NodeConfig{}, func(ctx agent.Context, _ any, yield func(*session.Event, error) bool) {
			if bCalls.Add(1) > 1 {
				faninEmit(ctx, yield, "B2", "r")
				return
			}
			faninWait(cPaused, faninBudget)
			time.Sleep(50 * time.Millisecond) // let c@1's pause land first
			if faninEmit(ctx, yield, "from-B", "r") {
				yield(nil, errors.New("b failed"))
			}
		})
		c := newFaninNode("c", NodeConfig{RerunOnResume: &rerun}, func(ctx agent.Context, in any, yield func(*session.Event, error) bool) {
			if in != "from-A" {
				faninEmit(ctx, yield, fmt.Sprintf("c:%v:no-interrupt", in))
				return
			}
			if v, ok := ctx.ResumedInput("ask-c"); ok {
				faninEmit(ctx, yield, fmt.Sprintf("c:%v:%v", in, v))
				return
			}
			faninAsk(ctx, yield, "ask-c")
			cOnce.Do(func() { close(cPaused) })
		})
		w, err := New("wf", []Edge{
			{From: Start, To: a},
			{From: Start, To: b},
			{From: a, To: c, Route: StringRoute("r")},
			{From: b, To: c, Route: StringRoute("r")},
		})
		if err != nil {
			t.Fatal(err)
		}
		ctx, _ := faninCtx(t, faninBudget)
		turn1 := faninRun(t, w.Run(ctx))
		if turn1.err == nil {
			t.Fatal("turn 1 Run error = nil, want b's failure")
		}
		state := faninReconstruct(t, w, turn1.events)

		turn2 := faninRun(t, w.Resume(faninResumeCtx(t), state, map[string]any{"ask-c": "ok"}))

		if turn2.err != nil {
			t.Errorf("Resume error = %v, want nil", turn2.err)
		}
		want := []faninObs{
			{Path: "wf@1/c@1", Value: "c:from-A:ok"},
			{Path: "wf@1/c@2", Value: "c:B2:no-interrupt"},
		}
		if diff := cmp.Diff(want, faninStripBranches(faninOutputs(turn2.events, "c"))); diff != "" {
			t.Errorf("resume turn c outputs mismatch (-want +got):\n%s", diff)
		}
	})
}

// TestPersistence_EarlierActivationResponseDoesNotLeak covers amendment
// 8, third case: after loop@1 paused, was answered and completed, and
// loop@2 paused, the rebuilt state holds only loop@2's pause, so the
// resumed loop@2 does not see loop@1's response.
//
// Fails if scanHistory folds an earlier activation's interrupts and
// responses into a later activation of the same node.
func TestPersistence_EarlierActivationResponseDoesNotLeak(t *testing.T) {
	rerun := true
	var sawAsk1 atomic.Bool
	loop := newFaninNode("loop", NodeConfig{RerunOnResume: &rerun}, func(ctx agent.Context, in any, yield func(*session.Event, error) bool) {
		if v, ok := ctx.ResumedInput("ask-2"); ok {
			if _, leaked := ctx.ResumedInput("ask-1"); leaked {
				sawAsk1.Store(true)
			}
			faninEmit(ctx, yield, fmt.Sprintf("iter-2:%v", v), "done")
			return
		}
		if v, ok := ctx.ResumedInput("ask-1"); ok {
			faninEmit(ctx, yield, fmt.Sprintf("iter-1:%v", v), "again")
			return
		}
		if in == "iter-1:ok1" {
			faninAsk(ctx, yield, "ask-2")
			return
		}
		faninAsk(ctx, yield, "ask-1")
	})
	w, err := New("wf", []Edge{
		{From: Start, To: loop},
		{From: loop, To: loop, Route: StringRoute("again")},
	})
	if err != nil {
		t.Fatal(err)
	}
	ctx, _ := faninCtx(t, faninBudget)
	turn1 := faninRun(t, w.Run(ctx))
	if turn1.err != nil {
		t.Fatalf("turn 1 Run error = %v, want nil", turn1.err)
	}
	turn2 := faninRun(t, w.Resume(faninResumeCtx(t), faninReconstruct(t, w, turn1.events), map[string]any{"ask-1": "ok1"}))
	if turn2.err != nil {
		t.Fatalf("turn 2 Resume error = %v, want nil", turn2.err)
	}
	history := slices.Concat(turn1.events, []*session.Event{faninUserResponse("ask-1", "ok1")}, turn2.events)
	state := faninReconstruct(t, w, history)

	turn3 := faninRun(t, w.Resume(faninResumeCtx(t), state, map[string]any{"ask-2": "ok2"}))

	if turn3.err != nil {
		t.Errorf("turn 3 Resume error = %v, want nil", turn3.err)
	}
	if sawAsk1.Load() {
		t.Error("loop@2 saw the response to loop@1's interrupt ask-1")
	}
	want := []faninObs{{Path: "wf@1/loop@2", Value: "iter-2:ok2"}}
	if diff := cmp.Diff(want, faninStripBranches(faninOutputs(turn3.events, "loop"))); diff != "" {
		t.Errorf("turn 3 loop outputs mismatch (-want +got):\n%s", diff)
	}
}
