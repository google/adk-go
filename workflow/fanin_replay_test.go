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
	"maps"
	"slices"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/google/jsonschema-go/jsonschema"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/session"
)

// faninConv drives a workflow over several turns the way the runner does:
// each reply is recorded as a user event in the history, the run state is
// rebuilt from that history, and Resume gets the same responses.
type faninConv struct {
	t       *testing.T
	w       *Workflow
	history []*session.Event
}

// startFaninConv runs turn 1, which must end without error.
func startFaninConv(t *testing.T, w *Workflow) *faninConv {
	t.Helper()
	ctx, _ := faninCtx(t, faninBudget)
	res := faninRun(t, w.Run(ctx))
	if res.err != nil {
		t.Fatalf("turn 1 Run error = %v, want nil", res.err)
	}
	return &faninConv{t: t, w: w, history: res.events}
}

// reply answers interrupts with answers. The result carries the error of
// rebuilding the run state, as the runner yields it for a response the
// request's schema rejects. When history holds nothing to resume, as for
// a duplicate answer, it yields nothing, as the runner does.
func (c *faninConv) reply(answers map[string]any) faninResult {
	c.t.Helper()
	ev := &session.Event{Author: "user", InvocationID: faninInvocation}
	ev.Content = &genai.Content{Role: genai.RoleUser}
	for _, id := range slices.Sorted(maps.Keys(answers)) {
		p := genai.NewPartFromFunctionResponse(WorkflowInputFunctionCallName, map[string]any{"response": answers[id]})
		p.FunctionResponse.ID = id
		ev.Content.Parts = append(ev.Content.Parts, p)
	}
	c.history = append(c.history, ev)
	state, err := c.w.ReconstructRunState(fakeSession{events: sliceEvents(c.history)}, faninInvocation)
	if err != nil {
		return faninResult{err: err}
	}
	if state == nil {
		return faninResult{}
	}
	res := faninRun(c.t, c.w.Resume(faninResumeCtx(c.t), state, answers))
	c.history = append(c.history, res.events...)
	return res
}

// faninApprovalSchema accepts {"approved": <bool>} only.
func faninApprovalSchema() *jsonschema.Schema {
	return &jsonschema.Schema{
		Type:       "object",
		Properties: map[string]*jsonschema.Schema{"approved": {Type: "boolean"}},
		Required:   []string{"approved"},
	}
}

// faninAsker is a node that asks for input id (with the approval schema
// when schema is set) and, as handoff or re-entry, is answered by it.
func faninAsker(name, id string, rerun, schema bool) *faninNode {
	return newFaninNode(name, NodeConfig{RerunOnResume: &rerun}, func(ctx agent.Context, _ any, yield func(*session.Event, error) bool) {
		ri := session.RequestInput{InterruptID: id}
		if schema {
			ri.ResponseSchema = faninApprovalSchema()
		}
		yield(NewRequestInputEvent(ctx, ri), nil)
	})
}

// faninPausingX is a re-entry node that asks "ask-x" and logs the path and
// input of the activation that the answer resumes.
func faninPausingX(runs *faninLog) *faninNode {
	rerun := true
	return newFaninNode("x", NodeConfig{RerunOnResume: &rerun}, func(ctx agent.Context, in any, yield func(*session.Event, error) bool) {
		if _, ok := ctx.ResumedInput("ask-x"); ok {
			runs.add(faninObs{Path: ctx.Path(), Value: in})
			faninEmit(ctx, yield, "x-done")
			return
		}
		faninAsk(ctx, yield, "ask-x")
	})
}

// TestResume_RejectedThenValidAnswer_RunsSuccessorThatRanEarlier covers
// amendment 7 together with a response its schema rejects: h's first
// answer fails its schema and does not count, so the valid answer is the
// first one, and x runs again with it although a already ran x on turn 1.
//
// Fails if scanHistory counts a rejected response as consumed: the valid
// answer then reads as an earlier turn's, and Resume skips x.
func TestResume_RejectedThenValidAnswer_RunsSuccessorThatRanEarlier(t *testing.T) {
	var xLog faninLog
	a := newFaninNode("a", NodeConfig{}, faninEmitter("A", "r"))
	h := faninAsker("h", "ask-h", false, true)
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
	conv := startFaninConv(t, w)

	rejected := conv.reply(map[string]any{"ask-h": "not an object"})
	accepted := conv.reply(map[string]any{"ask-h": map[string]any{"approved": true}})

	if !errors.Is(rejected.err, ErrInvalidResumeResponse) {
		t.Errorf("rejected answer: error = %v, want ErrInvalidResumeResponse", rejected.err)
	}
	if accepted.err != nil {
		t.Errorf("valid answer: Resume error = %v, want nil", accepted.err)
	}
	want := []faninObs{
		{Path: "wf@1/x@1", Value: "A"},
		{Path: "wf@1/x@2", Value: map[string]any{"approved": true}},
	}
	if diff := cmp.Diff(want, xLog.get()); diff != "" {
		t.Errorf("x activations mismatch (-want +got):\n%s", diff)
	}
}

// TestPersistence_AnsweredActivationTriggersNextOne covers the replay of
// an answered re-entry activation: loop@1 asked, was answered and
// completed with route "again", which triggered loop@2, and loop@2 paused.
// On the next resume loop@2 re-enters with loop@1's output, the trigger it
// consumed, and not with p's output.
//
// Fails if the replay leaves loop@1's interrupt open after its answer: the
// activation never completes in the replay, loop@2's trigger is missing,
// and loop@2 re-enters with the first predecessor's output "P".
func TestPersistence_AnsweredActivationTriggersNextOne(t *testing.T) {
	rerun := true
	var resumedIn faninLog
	p := newFaninNode("p", NodeConfig{}, faninEmitter("P"))
	loop := newFaninNode("loop", NodeConfig{RerunOnResume: &rerun}, func(ctx agent.Context, in any, yield func(*session.Event, error) bool) {
		if v, ok := ctx.ResumedInput("ask-2"); ok {
			resumedIn.add(faninObs{Path: ctx.Path(), Value: in})
			faninEmit(ctx, yield, fmt.Sprintf("%v:%v", in, v), "done")
			return
		}
		if v, ok := ctx.ResumedInput("ask-1"); ok {
			faninEmit(ctx, yield, fmt.Sprintf("iter-1:%v", v), "again")
			return
		}
		if in == "P" {
			faninAsk(ctx, yield, "ask-1")
			return
		}
		faninAsk(ctx, yield, "ask-2")
	})
	w, err := New("wf", []Edge{
		{From: Start, To: p},
		{From: p, To: loop},
		{From: loop, To: loop, Route: StringRoute("again")},
	})
	if err != nil {
		t.Fatal(err)
	}
	conv := startFaninConv(t, w)
	conv.reply(map[string]any{"ask-1": "ok1"})

	conv.reply(map[string]any{"ask-2": "ok2"})

	want := []faninObs{{Path: "wf@1/loop@2", Value: "iter-1:ok1"}}
	if diff := cmp.Diff(want, resumedIn.get()); diff != "" {
		t.Errorf("resumed loop@2 mismatch (-want +got):\n%s", diff)
	}
}

// TestPersistence_ReplayOrdersByLatestTerminalEvent covers the order in
// which the replay hands out triggers: activations are ordered by their
// latest terminal event, not by their first event. a emits a progress
// message before b's output but its own output after c@1 finished, so c@1
// consumed b's trigger and the paused c@2 consumed a's.
//
// Fails if the replay orders activations by the position of their first
// event: a then replays first, c@1 is given a's trigger, and the paused
// c@2 re-enters with b's output.
func TestPersistence_ReplayOrdersByLatestTerminalEvent(t *testing.T) {
	rerun := true
	var resumedIn faninLog
	c1Done := make(chan struct{})
	var once sync.Once
	a := newFaninNode("a", NodeConfig{}, func(ctx agent.Context, _ any, yield func(*session.Event, error) bool) {
		ev := session.NewEvent(ctx, ctx.InvocationID())
		ev.Content = genai.NewContentFromText("working", genai.RoleModel)
		yield(ev, nil)
		faninWait(c1Done, faninBudget)
		time.Sleep(50 * time.Millisecond)
		faninEmit(ctx, yield, "from-A", "r")
	})
	b := newFaninNode("b", NodeConfig{}, func(ctx agent.Context, _ any, yield func(*session.Event, error) bool) {
		time.Sleep(50 * time.Millisecond)
		faninEmit(ctx, yield, "from-B", "r")
	})
	c := newFaninNode("c", NodeConfig{RerunOnResume: &rerun}, func(ctx agent.Context, in any, yield func(*session.Event, error) bool) {
		if in == "from-B" {
			faninEmit(ctx, yield, "done:from-B")
			once.Do(func() { close(c1Done) })
			return
		}
		if v, ok := ctx.ResumedInput("ask-c2"); ok {
			resumedIn.add(faninObs{Path: ctx.Path(), Value: in})
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
	conv := startFaninConv(t, w)

	conv.reply(map[string]any{"ask-c2": "go"})

	want := []faninObs{{Path: "wf@1/c@2", Value: "from-A"}}
	if diff := cmp.Diff(want, resumedIn.get()); diff != "" {
		t.Errorf("resumed c activations mismatch (-want +got):\n%s", diff)
	}
}

// TestPersistence_SilentActivationConsumesItsTrigger covers an activation
// that left no event: gate@1 (WaitForOutput) yields nothing, so history
// has no record of it, and the paused gate@2 consumed the second trigger,
// b's.
//
// Fails if activation k takes the oldest trigger instead of the k-th: the
// paused gate@2 then re-enters with a's output.
func TestPersistence_SilentActivationConsumesItsTrigger(t *testing.T) {
	yes := true
	var resumedIn faninLog
	g1Done := make(chan struct{})
	var once sync.Once
	a := newFaninNode("a", NodeConfig{}, faninEmitter("from-A", "r"))
	b := newFaninNode("b", NodeConfig{}, func(ctx agent.Context, _ any, yield func(*session.Event, error) bool) {
		faninWait(g1Done, faninBudget)
		time.Sleep(50 * time.Millisecond)
		faninEmit(ctx, yield, "from-B", "r")
	})
	gate := newFaninNode("gate", NodeConfig{RerunOnResume: &yes, WaitForOutput: &yes}, func(ctx agent.Context, in any, yield func(*session.Event, error) bool) {
		if in == "from-A" {
			once.Do(func() { close(g1Done) })
			return
		}
		if v, ok := ctx.ResumedInput("ask-g"); ok {
			resumedIn.add(faninObs{Path: ctx.Path(), Value: in})
			faninEmit(ctx, yield, fmt.Sprintf("%v:%v", in, v))
			return
		}
		faninAsk(ctx, yield, "ask-g")
	})
	w, err := New("wf", []Edge{
		{From: Start, To: a},
		{From: Start, To: b},
		{From: a, To: gate, Route: StringRoute("r")},
		{From: b, To: gate, Route: StringRoute("r")},
	})
	if err != nil {
		t.Fatal(err)
	}
	conv := startFaninConv(t, w)

	conv.reply(map[string]any{"ask-g": "go"})

	want := []faninObs{{Path: "wf@1/gate@2", Value: "from-B"}}
	if diff := cmp.Diff(want, resumedIn.get()); diff != "" {
		t.Errorf("resumed gate activations mismatch (-want +got):\n%s", diff)
	}
}

// TestPersistence_UnreplayableNodeGetsNoQueuedTriggers covers history
// that does not replay: c@1's trigger has no producer event in the
// invocation, and b's output, recorded after c@1 paused, triggers c. The
// output is consumed once: c@1 re-enters with it, and no second activation
// runs it again.
//
// Fails if Resume restores the queued triggers of a node whose history
// does not replay: c then runs a second time, as c@2, with the same output.
func TestPersistence_UnreplayableNodeGetsNoQueuedTriggers(t *testing.T) {
	yes := true
	var runs faninLog
	a := newFaninNode("a", NodeConfig{}, func(agent.Context, any, func(*session.Event, error) bool) {})
	b := newFaninNode("b", NodeConfig{}, func(agent.Context, any, func(*session.Event, error) bool) {})
	c := newFaninNode("c", NodeConfig{RerunOnResume: &yes}, func(ctx agent.Context, in any, yield func(*session.Event, error) bool) {
		runs.add(faninObs{Path: ctx.Path(), Value: in})
		faninEmit(ctx, yield, "c-done")
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
	pause := &session.Event{InvocationID: faninInvocation, NodeInfo: &session.NodeInfo{Path: "wf@1/c@1"}, LongRunningToolIDs: []string{"ask-c"}}
	bOut := &session.Event{InvocationID: faninInvocation, NodeInfo: &session.NodeInfo{Path: "wf@1/b@1"}}
	bOut.Output = "from-B"
	bOut.Routes = []string{"r"}
	conv := &faninConv{t: t, w: w, history: []*session.Event{pause, bOut}}

	conv.reply(map[string]any{"ask-c": "ok"})

	want := []faninObs{{Path: "wf@1/c@1", Value: "from-B"}}
	if diff := cmp.Diff(want, runs.get()); diff != "" {
		t.Errorf("c activations mismatch (-want +got):\n%s", diff)
	}
}

// TestPersistence_HandoffAnswerTriggersSuccessorBehindPausedNode covers an
// answered handoff asker as a producer of triggers: h (handoff) and r
// (re-entry) both feed c, and one message answers both. c@1 runs with h's
// answer and pauses, while r's output queues c@2. The next resume finishes
// c@1 with h's answer and runs c@2 with r's output.
//
// Fails if the replay does not treat h's answer as h's completion: c@1
// consumes r's trigger in the replay, re-enters with "R", and h's answer is
// lost.
func TestPersistence_HandoffAnswerTriggersSuccessorBehindPausedNode(t *testing.T) {
	for _, rSlow := range []bool{false, true} {
		t.Run(fmt.Sprintf("r_slow=%v", rSlow), func(t *testing.T) {
			no, yes := false, true
			var runs faninLog
			h := faninAsker("h", "ask-h", no, false)
			r := newFaninNode("r", NodeConfig{RerunOnResume: &yes}, func(ctx agent.Context, _ any, yield func(*session.Event, error) bool) {
				if _, ok := ctx.ResumedInput("ask-r"); ok {
					if rSlow {
						time.Sleep(100 * time.Millisecond)
					}
					faninEmit(ctx, yield, "R", "x")
					return
				}
				faninAsk(ctx, yield, "ask-r")
			})
			c := newFaninNode("c", NodeConfig{RerunOnResume: &yes}, func(ctx agent.Context, in any, yield func(*session.Event, error) bool) {
				if in == "R" {
					runs.add(faninObs{Path: ctx.Path(), Value: in})
					faninEmit(ctx, yield, "c:R")
					return
				}
				if v, ok := ctx.ResumedInput("ask-c"); ok {
					runs.add(faninObs{Path: ctx.Path(), Value: fmt.Sprintf("%v:%v", in, v)})
					faninEmit(ctx, yield, "c-done")
					return
				}
				if !rSlow {
					time.Sleep(100 * time.Millisecond)
				}
				faninAsk(ctx, yield, "ask-c")
			})
			w, err := New("wf", []Edge{
				{From: Start, To: h},
				{From: Start, To: r},
				{From: h, To: c},
				{From: r, To: c, Route: StringRoute("x")},
			})
			if err != nil {
				t.Fatal(err)
			}
			conv := startFaninConv(t, w)
			conv.reply(map[string]any{"ask-h": "ok", "ask-r": "ok"})
			runs = faninLog{}

			conv.reply(map[string]any{"ask-c": "ok"})

			want := []faninObs{
				{Path: "wf@1/c@1", Value: "ok:ok"},
				{Path: "wf@1/c@2", Value: "R"},
			}
			if diff := cmp.Diff(want, runs.get()); diff != "" {
				t.Errorf("turn 3 c activations mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// TestPersistence_HandoffTriggerOrderedByItsAnswer covers where an answered
// handoff asker's trigger falls in the replay: at its answer, not at its
// interrupt. h asks on turn 1, then b's output triggers c@1, which pauses.
// h's answer on turn 2 queues c behind the paused c@1. Turn 3 re-enters c@1
// with b's output and then runs c@2 with h's answer.
//
// Fails if the replay orders h by its interrupt: h's trigger then comes
// before b's, c@1 re-enters with h's answer, and c@2 gets b's output.
func TestPersistence_HandoffTriggerOrderedByItsAnswer(t *testing.T) {
	yes := true
	var runs faninLog
	h := faninAsker("h", "ask-h", false, false)
	b := newFaninNode("b", NodeConfig{}, func(ctx agent.Context, _ any, yield func(*session.Event, error) bool) {
		time.Sleep(50 * time.Millisecond) // h's interrupt lands first
		faninEmit(ctx, yield, "from-B", "r")
	})
	c := newFaninNode("c", NodeConfig{RerunOnResume: &yes}, func(ctx agent.Context, in any, yield func(*session.Event, error) bool) {
		if _, ok := ctx.ResumedInput("ask-c"); ok || in != "from-B" {
			runs.add(faninObs{Path: ctx.Path(), Value: in})
			faninEmit(ctx, yield, "c-done")
			return
		}
		faninAsk(ctx, yield, "ask-c")
	})
	w, err := New("wf", []Edge{
		{From: Start, To: h},
		{From: Start, To: b},
		{From: h, To: c},
		{From: b, To: c, Route: StringRoute("r")},
	})
	if err != nil {
		t.Fatal(err)
	}
	conv := startFaninConv(t, w)
	conv.reply(map[string]any{"ask-h": "H"})

	conv.reply(map[string]any{"ask-c": "ok"})

	want := []faninObs{
		{Path: "wf@1/c@1", Value: "from-B"},
		{Path: "wf@1/c@2", Value: "H"},
	}
	if diff := cmp.Diff(want, runs.get()); diff != "" {
		t.Errorf("c activations mismatch (-want +got):\n%s", diff)
	}
}

// TestPersistence_RejectedAnswerTriggersNothing covers the replay of a
// response its schema rejects: h's first answer fails, the second is
// accepted and triggers x@1, which pauses. x@1 re-enters with the accepted
// answer.
//
// Fails if the replay consumes the rejected answer: h's activation then
// completes with it, x@1's trigger carries the rejected answer, and the
// paused x@1 re-enters with "not an object".
func TestPersistence_RejectedAnswerTriggersNothing(t *testing.T) {
	var runs faninLog
	h := faninAsker("h", "ask-h", false, true)
	w, err := New("wf", []Edge{{From: Start, To: h}, {From: h, To: faninPausingX(&runs)}})
	if err != nil {
		t.Fatal(err)
	}
	conv := startFaninConv(t, w)
	rejected := conv.reply(map[string]any{"ask-h": "not an object"})
	conv.reply(map[string]any{"ask-h": map[string]any{"approved": true}})

	conv.reply(map[string]any{"ask-x": "ok"})

	if !errors.Is(rejected.err, ErrInvalidResumeResponse) {
		t.Errorf("rejected answer: error = %v, want ErrInvalidResumeResponse", rejected.err)
	}
	want := []faninObs{{Path: "wf@1/x@1", Value: map[string]any{"approved": true}}}
	if diff := cmp.Diff(want, runs.get()); diff != "" {
		t.Errorf("x activations mismatch (-want +got):\n%s", diff)
	}
}

// TestPersistence_RepeatedAnswerTriggersNothing covers a duplicate answer:
// h is answered twice, the second answer triggers nothing, so answering
// the paused x@1 runs x once.
//
// Fails if the replay lets a repeated answer reopen and complete h again:
// x then has a second trigger and runs again as x@2.
func TestPersistence_RepeatedAnswerTriggersNothing(t *testing.T) {
	var runs faninLog
	h := faninAsker("h", "ask-h", false, false)
	w, err := New("wf", []Edge{{From: Start, To: h}, {From: h, To: faninPausingX(&runs)}})
	if err != nil {
		t.Fatal(err)
	}
	conv := startFaninConv(t, w)
	conv.reply(map[string]any{"ask-h": "H"})
	conv.reply(map[string]any{"ask-h": "H"})

	conv.reply(map[string]any{"ask-x": "ok"})

	want := []faninObs{{Path: "wf@1/x@1", Value: "H"}}
	if diff := cmp.Diff(want, runs.get()); diff != "" {
		t.Errorf("x activations mismatch (-want +got):\n%s", diff)
	}
}

// TestPersistence_LateDuplicateOfEarlierActivationResponseDoesNotLeak
// covers amendment 8, third case, for a response that arrives after the
// later activation started: loop@1 asked ask-1 and was answered, loop@2
// asks ask-2, and the next message repeats the answer to ask-1 along with
// the answer to ask-2. loop@2 sees only its own response.
//
// Fails if the scan keeps loop@1's interrupt owners when loop@2 starts: the
// repeated ask-1 response is then credited to loop@2, which sees it as
// resumed input.
func TestPersistence_LateDuplicateOfEarlierActivationResponseDoesNotLeak(t *testing.T) {
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
	conv := startFaninConv(t, w)
	conv.reply(map[string]any{"ask-1": "ok1"})

	turn3 := conv.reply(map[string]any{"ask-1": "ok1", "ask-2": "ok2"})

	if turn3.err != nil {
		t.Errorf("turn 3 Resume error = %v, want nil", turn3.err)
	}
	if sawAsk1.Load() {
		t.Error("loop@2 saw the repeated response to loop@1's interrupt ask-1")
	}
	want := []faninObs{{Path: "wf@1/loop@2", Value: "iter-2:ok2"}}
	if diff := cmp.Diff(want, faninStripBranches(faninOutputs(turn3.events, "loop"))); diff != "" {
		t.Errorf("turn 3 loop outputs mismatch (-want +got):\n%s", diff)
	}
}

// TestResume_ParkedWaitForOutputActivationAcceptsQueuedTrigger covers a
// WaitForOutput node that was resumed from an interrupt and then parked
// without output: gate@1 asked and b's trigger queued behind it; on a
// direct Resume call gate@1 re-enters, yields nothing and parks, and the
// queued trigger then runs as gate@2.
//
// Fails if a parked activation keeps the interrupts of the pause it was
// resumed from: gate stays busy, and gate@2 never starts.
func TestResume_ParkedWaitForOutputActivationAcceptsQueuedTrigger(t *testing.T) {
	yes := true
	var runs faninLog
	gateAsked := make(chan struct{})
	var once sync.Once
	a := newFaninNode("a", NodeConfig{}, faninEmitter("from-A", "r"))
	b := newFaninNode("b", NodeConfig{}, func(ctx agent.Context, _ any, yield func(*session.Event, error) bool) {
		faninWait(gateAsked, faninBudget)
		time.Sleep(50 * time.Millisecond) // gate@1's pause lands first
		faninEmit(ctx, yield, "from-B", "r")
	})
	gate := newFaninNode("gate", NodeConfig{RerunOnResume: &yes, WaitForOutput: &yes}, func(ctx agent.Context, in any, yield func(*session.Event, error) bool) {
		if _, ok := ctx.ResumedInput("ask-g"); ok {
			return
		}
		if in == "from-A" {
			faninAsk(ctx, yield, "ask-g")
			once.Do(func() { close(gateAsked) })
			return
		}
		runs.add(faninObs{Path: ctx.Path(), Value: in})
		faninEmit(ctx, yield, "g2:"+in.(string))
	})
	w, err := New("wf", []Edge{
		{From: Start, To: a},
		{From: Start, To: b},
		{From: a, To: gate, Route: StringRoute("r")},
		{From: b, To: gate, Route: StringRoute("r")},
	})
	if err != nil {
		t.Fatal(err)
	}
	conv := startFaninConv(t, w)
	// A direct Resume call: the response is not in history, so the rebuilt
	// state still lists ask-g as an open interrupt of gate.
	state := faninReconstruct(t, w, conv.history)

	turn2 := faninRun(t, w.Resume(faninResumeCtx(t), state, map[string]any{"ask-g": "go"}))

	if turn2.err != nil {
		t.Errorf("Resume error = %v, want nil", turn2.err)
	}
	want := []faninObs{{Path: "wf@1/gate@2", Value: "from-B"}}
	if diff := cmp.Diff(want, runs.get()); diff != "" {
		t.Errorf("gate activations that ran with output mismatch (-want +got):\n%s", diff)
	}
}

// TestResume_TerminalOutputOfEarlierTurnCountsAtFinalize covers B7 across
// turns: t1 produced its output on turn 1 and t2 pauses; when t2 produces
// output on the resume turn, the run has two terminal outputs, as Python's
// rerun counts them.
//
// Fails if the resumed run forgets the terminal output of an earlier turn:
// Resume would end without ErrMultipleTerminalOutputs.
func TestResume_TerminalOutputOfEarlierTurnCountsAtFinalize(t *testing.T) {
	rerun := true
	t1 := newFaninNode("t1", NodeConfig{}, faninEmitter("T1"))
	t2 := newFaninNode("t2", NodeConfig{RerunOnResume: &rerun}, func(ctx agent.Context, _ any, yield func(*session.Event, error) bool) {
		if _, ok := ctx.ResumedInput("ask-t2"); ok {
			faninEmit(ctx, yield, "T2")
			return
		}
		faninAsk(ctx, yield, "ask-t2")
	})
	w, err := New("wf", []Edge{{From: Start, To: t1}, {From: Start, To: t2}})
	if err != nil {
		t.Fatal(err)
	}
	conv := startFaninConv(t, w)

	turn2 := conv.reply(map[string]any{"ask-t2": "go"})

	if !errors.Is(turn2.err, ErrMultipleTerminalOutputs) {
		t.Errorf("Resume error = %v, want ErrMultipleTerminalOutputs", turn2.err)
	}
}

func TestReplayHistory_PrePauseOutputResetWhenInterruptAnswered(t *testing.T) {
	c := newFaninNode("c", NodeConfig{RerunOnResume: new(true)}, func(ctx agent.Context, _ any, yield func(*session.Event, error) bool) {
		faninEmit(ctx, yield, "draft", "done")
		faninAsk(ctx, yield, "ask-1")
	})
	done := NewFunctionNode("done", func(_ agent.Context, in any) (string, error) {
		return fmt.Sprintf("done:%v", in), nil
	}, NodeConfig{})
	w, err := New("wf", []Edge{
		{From: Start, To: c},
		{From: c, To: c, Route: Default},
		{From: c, To: done, Route: StringRoute("done")},
	})
	if err != nil {
		t.Fatal(err)
	}
	conv := startFaninConv(t, w)
	userAns := faninUserResponse("ask-1", "ok1")
	state := faninReconstruct(t, w, append(slices.Clone(conv.history), userAns))
	if got := len(state.queued["c"]); got != 0 {
		t.Fatalf("state.queued[c] len = %d, want 0 (pre-pause output left st.terminal == NodeCompleted)", got)
	}
}

func TestResume_EarlierHandoffDoesNotDoubleTriggerFailedLoopSuccessor(t *testing.T) {
	var xCalls atomic.Int32
	var xInputs []any
	var mu sync.Mutex
	h := newFaninNode("h", NodeConfig{}, func(ctx agent.Context, _ any, yield func(*session.Event, error) bool) {
		faninAsk(ctx, yield, "ask-h")
	})
	p := newFaninNode("p", NodeConfig{}, func(ctx agent.Context, _ any, yield func(*session.Event, error) bool) {
		faninAsk(ctx, yield, "ask-p")
	})
	x := newFaninNode("x", NodeConfig{}, func(ctx agent.Context, in any, yield func(*session.Event, error) bool) {
		n := xCalls.Add(1)
		mu.Lock()
		xInputs = append(xInputs, in)
		mu.Unlock()
		switch n {
		case 1:
			faninEmit(ctx, yield, "iter1", "again")
		case 2:
			yield(nil, errors.New("x@2 failed"))
		default:
			faninEmit(ctx, yield, fmt.Sprintf("iter2:%v", in))
		}
	})
	w, err := New("wf", []Edge{
		{From: Start, To: h},
		{From: Start, To: p},
		{From: h, To: x},
		{From: x, To: x, Route: StringRoute("again")},
	})
	if err != nil {
		t.Fatal(err)
	}
	conv := startFaninConv(t, w)

	// Turn 2: answer h; x@1 loops to x@2, which fails.
	turn2 := conv.reply(map[string]any{"ask-h": "from-h"})
	if turn2.err == nil {
		t.Fatal("turn 2 error = nil, want x@2 failure")
	}
	mu.Lock()
	xInputs = nil
	mu.Unlock()

	// Turn 3: user sends BOTH ask-h (repeated from Turn 2) and ask-p (new);
	// x@2 must retry once with "iter1", not also get re-triggered by h's repeated "from-h".
	turn3 := conv.reply(map[string]any{"ask-h": "from-h", "ask-p": "from-p"})
	if turn3.err != nil {
		t.Fatalf("turn 3 error = %v, want nil", turn3.err)
	}
	if diff := cmp.Diff([]any{"iter1"}, xInputs); diff != "" {
		t.Errorf("turn 3 x inputs mismatch (-want +got):\n%s", diff)
	}
}
