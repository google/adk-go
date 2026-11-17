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
	"sync/atomic"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/session"
)

// TestScheduler_MaxConcurrency_QueuedTriggerDoesNotHoldSlotAndPreservesOrder
// covers B10 and ruling 7: a trigger waiting behind a running activation
// of its own node takes no slot, so a later trigger of another node
// starts in the slot that frees up, and the waiting trigger starts when
// its node is free.
//
// Fails if a buffered trigger counts against the cap.
// TestScheduler_QueuedTriggerOfBusyNodeDoesNotBlockLaterNodes covers a
// scan that stops at the busy node's trigger instead of skipping it.
func TestScheduler_MaxConcurrency_QueuedTriggerDoesNotHoldSlotAndPreservesOrder(t *testing.T) {
	var inFlight, peak, cCalls atomic.Int32
	var starts faninLog
	cStarted, eStarted := make(chan struct{}), make(chan struct{})
	a := newFaninNode("a", NodeConfig{}, func(ctx agent.Context, _ any, yield func(*session.Event, error) bool) {
		defer bumpPeak(&inFlight, &peak)()
		faninEmit(ctx, yield, "A", "r")
	})
	b := newFaninNode("b", NodeConfig{}, func(ctx agent.Context, _ any, yield func(*session.Event, error) bool) {
		defer bumpPeak(&inFlight, &peak)()
		faninWait(cStarted, faninBudget)
		faninEmit(ctx, yield, "B", "r")
	})
	c := newFaninNode("c", NodeConfig{}, func(ctx agent.Context, _ any, yield func(*session.Event, error) bool) {
		defer bumpPeak(&inFlight, &peak)()
		starts.add(faninObs{Path: ctx.Path()})
		if cCalls.Add(1) == 1 {
			close(cStarted)
			faninWait(eStarted, time.Second) // c@1 runs until e@1 has started
		}
		faninEmit(ctx, yield, nil)
	})
	d := newFaninNode("d", NodeConfig{}, func(ctx agent.Context, _ any, yield func(*session.Event, error) bool) {
		defer bumpPeak(&inFlight, &peak)()
		starts.add(faninObs{Path: ctx.Path()})
		faninEmit(ctx, yield, "D")
	})
	e := newFaninNode("e", NodeConfig{}, func(ctx agent.Context, _ any, yield func(*session.Event, error) bool) {
		defer bumpPeak(&inFlight, &peak)()
		starts.add(faninObs{Path: ctx.Path()})
		close(eStarted)
		faninEmit(ctx, yield, nil)
	})
	w, err := New("wf", []Edge{
		{From: Start, To: a},
		{From: Start, To: b},
		{From: a, To: c, Route: StringRoute("r")},
		{From: a, To: d, Route: StringRoute("r")},
		{From: b, To: c, Route: StringRoute("r")},
		{From: d, To: e},
	}, WithMaxConcurrency(2))
	if err != nil {
		t.Fatal(err)
	}
	ctx, _ := faninCtx(t, faninBudget)

	res := faninRun(t, w.Run(ctx))

	if res.err != nil {
		t.Errorf("Run error = %v, want nil", res.err)
	}
	want := []faninObs{{Path: "wf@1/c@1"}, {Path: "wf@1/d@1"}, {Path: "wf@1/e@1"}, {Path: "wf@1/c@2"}}
	if diff := cmp.Diff(want, starts.get()); diff != "" {
		t.Errorf("start order mismatch (-want +got):\n%s", diff)
	}
	if got := peak.Load(); got != 2 {
		t.Errorf("peak running activations = %d, want 2", got)
	}
}

// TestScheduler_MaxConcurrency_StartsTriggersInKeyInsertionOrder covers
// amendment 4: the trigger buffer keeps one FIFO bucket per node in the
// order the node first got a trigger, so a node's second trigger starts
// before a node whose first trigger arrived after the first node's
// (Python's trigger_buffer, _workflow.py:550-560).
//
// Fails if the buffer is one flat FIFO across nodes, which would start
// d@1 before c@2.
func TestScheduler_MaxConcurrency_StartsTriggersInKeyInsertionOrder(t *testing.T) {
	var starts faninLog
	record := func(out any, routes ...string) func(agent.Context, any, func(*session.Event, error) bool) {
		return func(ctx agent.Context, in any, yield func(*session.Event, error) bool) {
			starts.add(faninObs{Path: ctx.Path(), Value: in})
			faninEmit(ctx, yield, out, routes...)
		}
	}
	x := newFaninNode("x", NodeConfig{}, record("x-out", "r"))
	c := newFaninNode("c", NodeConfig{}, record(nil))
	d := newFaninNode("d", NodeConfig{}, record(nil))
	w, err := New("wf", []Edge{
		{From: Start, To: x},
		{From: Start, To: c},
		{From: Start, To: d},
		{From: x, To: c, Route: StringRoute("r")},
	}, WithMaxConcurrency(1))
	if err != nil {
		t.Fatal(err)
	}
	ctx, _ := faninCtx(t, faninBudget)

	res := faninRun(t, w.Run(ctx))

	if res.err != nil {
		t.Errorf("Run error = %v, want nil", res.err)
	}
	want := []faninObs{
		{Path: "wf@1/x@1", Value: "seed"},
		{Path: "wf@1/c@1", Value: "seed"},
		{Path: "wf@1/c@2", Value: "x-out"},
		{Path: "wf@1/d@1", Value: "seed"},
	}
	if diff := cmp.Diff(want, starts.get()); diff != "" {
		t.Errorf("start order mismatch (-want +got):\n%s", diff)
	}
}
