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
	"fmt"
	"sync/atomic"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/session"
)

// faninJoinOutputs returns path and output of every output event of the
// join named name, in stream order. A join's branch is not under test
// here, so it is left out.
func faninJoinOutputs(events []*session.Event, name string) []faninObs {
	out := faninOutputs(events, name)
	for i := range out {
		out[i].Branch = ""
	}
	return out
}

// TestJoinNode_WaitsForQueuedPredecessorTrigger is the regression test
// for P2 (ruling 9): a JoinNode does not fire on a predecessor's earlier
// output while that predecessor has a trigger waiting to start.
//
// Fails if the barrier's settled predicate ignores buffered triggers, or
// if bufferSuccessors judges join barriers before it buffers the
// completion's other targets.
func TestJoinNode_WaitsForQueuedPredecessorTrigger(t *testing.T) {
	want := []faninObs{{Path: "wf@1/j@1", Value: map[string]any{"a": "A1", "b": "B2"}}}

	t.Run("predecessor_trigger_queued_behind_cap", func(t *testing.T) {
		// With one slot, b@2's trigger waits behind a's, so a completes
		// while b has a trigger buffered and b@1's output is stale.
		var bCalls atomic.Int32
		a := newFaninNode("a", NodeConfig{}, faninEmitter("A1"))
		b := newFaninNode("b", NodeConfig{}, func(ctx agent.Context, _ any, yield func(*session.Event, error) bool) {
			if bCalls.Add(1) == 1 {
				faninEmit(ctx, yield, "B1", "again")
				return
			}
			faninEmit(ctx, yield, "B2", "done")
		})
		j := NewJoinNode("j")
		w, err := New("wf", []Edge{
			{From: Start, To: b},
			{From: Start, To: a},
			{From: b, To: b, Route: StringRoute("again")},
			{From: a, To: j},
			{From: b, To: j},
		}, WithMaxConcurrency(1))
		if err != nil {
			t.Fatal(err)
		}
		ctx, _ := faninCtx(t, faninBudget)

		res := faninRun(t, w.Run(ctx))

		if res.err != nil {
			t.Errorf("Run error = %v, want nil", res.err)
		}
		if diff := cmp.Diff(want, faninJoinOutputs(res.events, "j")); diff != "" {
			t.Errorf("j outputs mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("completion_triggers_predecessor_and_join", func(t *testing.T) {
		// a's one completion triggers b@2 and evaluates j; j is listed
		// first among a's edges.
		var bCalls atomic.Int32
		bDone := make(chan struct{})
		a := newFaninNode("a", NodeConfig{}, func(ctx agent.Context, _ any, yield func(*session.Event, error) bool) {
			faninWait(bDone, faninBudget)
			time.Sleep(50 * time.Millisecond) // let b@1's completion land first
			faninEmit(ctx, yield, "A1", "r")
		})
		b := newFaninNode("b", NodeConfig{}, func(ctx agent.Context, _ any, yield func(*session.Event, error) bool) {
			if bCalls.Add(1) == 1 {
				faninEmit(ctx, yield, "B1")
				close(bDone)
				return
			}
			// A join that fired on b@1's output ends before b@2 does.
			time.Sleep(100 * time.Millisecond)
			faninEmit(ctx, yield, "B2")
		})
		j := NewJoinNode("j")
		w, err := New("wf", []Edge{
			{From: Start, To: a},
			{From: Start, To: b},
			{From: a, To: j},
			{From: a, To: b, Route: StringRoute("r")},
			{From: b, To: j},
		})
		if err != nil {
			t.Fatal(err)
		}
		ctx, _ := faninCtx(t, faninBudget)

		res := faninRun(t, w.Run(ctx))

		if res.err != nil {
			t.Errorf("Run error = %v, want nil", res.err)
		}
		if diff := cmp.Diff(want, faninJoinOutputs(res.events, "j")); diff != "" {
			t.Errorf("j outputs mismatch (-want +got):\n%s", diff)
		}
	})
}

// TestJoinNode_LoopingPredecessorRefiresJoin covers B9 and amendment 3:
// a predecessor that loops counts as settled at its own completion, so
// each of its activations fires the join again, under the join's next run
// id, with that activation's output (Python's join_loop graph).
//
// Fails if the barrier treats the completing predecessor as unsettled
// because its own next trigger is already buffered.
func TestJoinNode_LoopingPredecessorRefiresJoin(t *testing.T) {
	var loopCalls atomic.Int32
	aDone := make(chan struct{})
	a := newFaninNode("a", NodeConfig{}, func(ctx agent.Context, _ any, yield func(*session.Event, error) bool) {
		faninEmit(ctx, yield, "A")
		close(aDone)
	})
	loop := newFaninNode("loop", NodeConfig{}, func(ctx agent.Context, _ any, yield func(*session.Event, error) bool) {
		k := loopCalls.Add(1)
		if k == 1 {
			faninWait(aDone, faninBudget)
			time.Sleep(50 * time.Millisecond) // let a's completion land first
		}
		var routes []string
		if k < 3 {
			routes = []string{"again"}
		}
		faninEmit(ctx, yield, fmt.Sprintf("iter-%d", k), routes...)
	})
	join := NewJoinNode("J")
	w, err := New("wf", []Edge{
		{From: Start, To: a},
		{From: Start, To: loop},
		{From: a, To: join},
		{From: loop, To: join},
		{From: loop, To: loop, Route: StringRoute("again")},
	})
	if err != nil {
		t.Fatal(err)
	}
	ctx, _ := faninCtx(t, faninBudget)

	res := faninRun(t, w.Run(ctx))

	if res.err != nil {
		t.Errorf("Run error = %v, want nil", res.err)
	}
	want := []faninObs{
		{Path: "wf@1/J@1", Value: map[string]any{"a": "A", "loop": "iter-1"}},
		{Path: "wf@1/J@2", Value: map[string]any{"a": "A", "loop": "iter-2"}},
		{Path: "wf@1/J@3", Value: map[string]any{"a": "A", "loop": "iter-3"}},
	}
	if diff := cmp.Diff(want, faninJoinOutputs(res.events, "J")); diff != "" {
		t.Errorf("J outputs mismatch (-want +got):\n%s", diff)
	}
}

// TestJoinNode_UsesLastOutputWhenPredecessorActivationYieldsNone covers
// amendment 5: a join receives each predecessor's last non-nil output, so
// a predecessor activation that yields no output leaves its earlier
// output in the join's input, as in Python's node_outputs.
//
// Fails if the barrier reads the predecessor's NodeState.Output, which a
// new activation resets, instead of the scheduler's last outputs.
func TestJoinNode_UsesLastOutputWhenPredecessorActivationYieldsNone(t *testing.T) {
	var pCalls atomic.Int32
	p := newFaninNode("p", NodeConfig{}, func(ctx agent.Context, _ any, yield func(*session.Event, error) bool) {
		if pCalls.Add(1) == 1 {
			faninEmit(ctx, yield, "P1", "again")
			return
		}
		faninEmit(ctx, yield, nil)
	})
	j := NewJoinNode("j")
	w, err := New("wf", []Edge{
		{From: Start, To: p},
		{From: p, To: p, Route: StringRoute("again")},
		{From: p, To: j},
	})
	if err != nil {
		t.Fatal(err)
	}
	ctx, _ := faninCtx(t, faninBudget)

	res := faninRun(t, w.Run(ctx))

	if res.err != nil {
		t.Errorf("Run error = %v, want nil", res.err)
	}
	want := []faninObs{
		{Path: "wf@1/j@1", Value: map[string]any{"p": "P1"}},
		{Path: "wf@1/j@2", Value: map[string]any{"p": "P1"}},
	}
	if diff := cmp.Diff(want, faninJoinOutputs(res.events, "j")); diff != "" {
		t.Errorf("j outputs mismatch (-want +got):\n%s", diff)
	}
}
