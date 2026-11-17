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
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/session"
)

// TestScheduler_WaitForOutput_ParksAndAcceptsSecondTrigger covers B3 and
// ruling 5: a WaitForOutput node whose activation ends with neither output
// nor route parks without triggering its successor, and its next trigger
// starts a new activation whose output does trigger it.
//
// Fails if handleCompletion completes such an activation instead of
// parking it, or if a node parked without interrupts counts as busy.
func TestScheduler_WaitForOutput_ParksAndAcceptsSecondTrigger(t *testing.T) {
	var gateCalls atomic.Int32
	var afterLog faninLog
	wait := true
	a := newFaninNode("a", NodeConfig{}, faninEmitter("A", "route1"))
	b := newFaninNode("b", NodeConfig{}, faninEmitter("B", "route1"))
	gate := newFaninNode("gate", NodeConfig{WaitForOutput: &wait}, func(ctx agent.Context, _ any, yield func(*session.Event, error) bool) {
		if gateCalls.Add(1) == 1 {
			faninEmit(ctx, yield, nil)
			return
		}
		faninEmit(ctx, yield, "gate-open")
	})
	after := newFaninNode("after", NodeConfig{}, func(ctx agent.Context, in any, yield func(*session.Event, error) bool) {
		afterLog.add(faninObs{Path: ctx.Path(), Value: in})
		faninEmit(ctx, yield, fmt.Sprintf("after:%v", in))
	})
	w, err := New("wf", []Edge{
		{From: Start, To: a},
		{From: Start, To: b},
		{From: a, To: gate, Route: StringRoute("route1")},
		{From: b, To: gate, Route: StringRoute("route1")},
		{From: gate, To: after},
	})
	if err != nil {
		t.Fatal(err)
	}
	ctx, _ := faninCtx(t, faninBudget)

	res := faninRun(t, w.Run(ctx))

	if res.err != nil {
		t.Errorf("Run error = %v, want nil", res.err)
	}
	wantAfter := []faninObs{{Path: "wf@1/after@1", Value: "gate-open"}}
	if diff := cmp.Diff(wantAfter, afterLog.get()); diff != "" {
		t.Errorf("after activations mismatch (-want +got):\n%s", diff)
	}
	wantGate := []faninObs{{Path: "wf@1/gate@1"}, {Path: "wf@1/gate@2"}}
	gotGate := faninActivations(res.events, "gate")
	for i := range gotGate {
		gotGate[i].Branch = ""
	}
	if diff := cmp.Diff(wantGate, gotGate); diff != "" {
		t.Errorf("gate activations mismatch (-want +got):\n%s", diff)
	}
}

// TestScheduler_Finalize_CompletesWhenWaitForOutputNodeNeverFires covers
// B7 and rulings 5 and 8: a WaitForOutput node left parked without
// interrupts does not hold finalize back, so two terminal outputs still
// fail the run, and the parked node's successor never runs.
//
// Fails if finalize returns early for any NodeWaiting node rather than
// only for one with interrupts, or if the waiter is completed instead of
// parked.
func TestScheduler_Finalize_CompletesWhenWaitForOutputNodeNeverFires(t *testing.T) {
	var downstreamRan atomic.Bool
	wait := true
	waiter := newFaninNode("waiter", NodeConfig{WaitForOutput: &wait}, faninEmitter(nil))
	term1 := newFaninNode("term1", NodeConfig{}, faninEmitter("T1"))
	term2 := newFaninNode("term2", NodeConfig{}, faninEmitter("T2"))
	blockedDownstream := newFaninNode("blockedDownstream", NodeConfig{}, func(ctx agent.Context, _ any, yield func(*session.Event, error) bool) {
		downstreamRan.Store(true)
		faninEmit(ctx, yield, nil)
	})
	w, err := New("wf", []Edge{
		{From: Start, To: waiter},
		{From: Start, To: term1},
		{From: Start, To: term2},
		{From: waiter, To: blockedDownstream},
	})
	if err != nil {
		t.Fatal(err)
	}
	ctx, _ := faninCtx(t, faninBudget)

	res := faninRun(t, w.Run(ctx))

	if downstreamRan.Load() {
		t.Error("blockedDownstream ran, want the waiter's successor never to run")
	}
	if !errors.Is(res.err, ErrMultipleTerminalOutputs) || !strings.Contains(res.err.Error(), "term1, term2") {
		t.Errorf("Run error = %v, want ErrMultipleTerminalOutputs naming term1, term2", res.err)
	}
}

// TestScheduler_Finalize_KeepsLastNonNilOutputAcrossActivations covers B7,
// ruling 8 and the ported flaw P1: every activation of a terminal node
// stamps OutputFor with its own path, and a terminal node whose later
// activation yields no output still counts as an output producer.
//
// Fails if handleCompletion stops recording the last non-nil output per
// node (lastOutputs), or if OutputFor is not stamped per activation.
func TestScheduler_Finalize_KeepsLastNonNilOutputAcrossActivations(t *testing.T) {
	t.Run("single_terminal_multiple_outputs", func(t *testing.T) {
		var calls atomic.Int32
		a := newFaninNode("a", NodeConfig{}, faninEmitter("A", "r"))
		b := newFaninNode("b", NodeConfig{}, faninEmitter("B", "r"))
		term := newFaninNode("term", NodeConfig{}, func(ctx agent.Context, _ any, yield func(*session.Event, error) bool) {
			faninEmit(ctx, yield, fmt.Sprintf("out-%d", calls.Add(1)))
		})
		w, err := New("wf", []Edge{
			{From: Start, To: a},
			{From: Start, To: b},
			{From: a, To: term, Route: StringRoute("r")},
			{From: b, To: term, Route: StringRoute("r")},
		})
		if err != nil {
			t.Fatal(err)
		}
		ctx, _ := faninCtx(t, faninBudget)

		res := faninRun(t, w.Run(ctx))

		if res.err != nil {
			t.Errorf("Run error = %v, want nil", res.err)
		}
		var got [][]string
		for _, ev := range res.events {
			if ev.Output != nil && ev.NodeInfo.Name() == "term" {
				got = append(got, ev.NodeInfo.OutputFor)
			}
		}
		want := [][]string{{"wf@1/term@1", "wf@1"}, {"wf@1/term@2", "wf@1"}}
		if diff := cmp.Diff(want, got); diff != "" {
			t.Errorf("term OutputFor mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("second_activation_nil_output_keeps_first", func(t *testing.T) {
		var calls atomic.Int32
		termDone := make(chan struct{})
		a := newFaninNode("a", NodeConfig{}, faninEmitter("A", "r"))
		b := newFaninNode("b", NodeConfig{}, func(ctx agent.Context, _ any, yield func(*session.Event, error) bool) {
			// Trigger term only after its first activation has ended.
			faninWait(termDone, faninBudget)
			time.Sleep(100 * time.Millisecond)
			faninEmit(ctx, yield, "B", "r")
		})
		term := newFaninNode("term", NodeConfig{}, func(ctx agent.Context, _ any, yield func(*session.Event, error) bool) {
			if calls.Add(1) == 1 {
				faninEmit(ctx, yield, "first")
				close(termDone)
				return
			}
			faninEmit(ctx, yield, nil)
		})
		other := newFaninNode("other", NodeConfig{}, faninEmitter("second"))
		w, err := New("wf", []Edge{
			{From: Start, To: a},
			{From: Start, To: b},
			{From: Start, To: other},
			{From: a, To: term, Route: StringRoute("r")},
			{From: b, To: term, Route: StringRoute("r")},
		})
		if err != nil {
			t.Fatal(err)
		}
		ctx, _ := faninCtx(t, faninBudget)

		res := faninRun(t, w.Run(ctx))

		if got := calls.Load(); got != 2 {
			t.Fatalf("activations of term = %d, want 2", got)
		}
		if !errors.Is(res.err, ErrMultipleTerminalOutputs) || !strings.Contains(res.err.Error(), "other, term") {
			t.Errorf("Run error = %v, want ErrMultipleTerminalOutputs naming other, term", res.err)
		}
	})
}

// TestScheduler_WaitForOutput_AttemptBudgetResetsPerActivation covers
// amendment 1: each new activation of a node starts with a full retry
// budget, even when the previous activation used a retry and then parked
// for WaitForOutput. Python retries gate@2 and runs after with open:B.
//
// Fails if startNode does not reset NodeState.Attempt for a new
// activation, so gate@2's first failure exhausts MaxAttempts.
func TestScheduler_WaitForOutput_AttemptBudgetResetsPerActivation(t *testing.T) {
	var gateCalls atomic.Int32
	var afterLog faninLog
	gateStarted := make(chan struct{})
	wait := true
	cfg := NodeConfig{
		WaitForOutput: &wait,
		RetryConfig: &RetryConfig{
			MaxAttempts:   2,
			InitialDelay:  time.Millisecond,
			BackoffFactor: 1,
			ShouldRetry:   func(error) bool { return true },
		},
	}
	a := newFaninNode("a", NodeConfig{}, faninEmitter("A", "r"))
	b := newFaninNode("b", NodeConfig{}, func(ctx agent.Context, _ any, yield func(*session.Event, error) bool) {
		faninWait(gateStarted, faninBudget)
		faninEmit(ctx, yield, "B", "r")
	})
	gate := newFaninNode("gate", cfg, func(ctx agent.Context, in any, yield func(*session.Event, error) bool) {
		switch gateCalls.Add(1) {
		case 1:
			close(gateStarted)
			yield(nil, errors.New("transient"))
		case 2:
			faninEmit(ctx, yield, nil) // gate@1 parks
		case 3:
			yield(nil, errors.New("transient"))
		default:
			faninEmit(ctx, yield, fmt.Sprintf("open:%v", in))
		}
	})
	after := newFaninNode("after", NodeConfig{}, func(ctx agent.Context, in any, yield func(*session.Event, error) bool) {
		afterLog.add(faninObs{Path: ctx.Path(), Value: in})
		faninEmit(ctx, yield, fmt.Sprintf("after:%v", in))
	})
	w, err := New("wf", []Edge{
		{From: Start, To: a},
		{From: Start, To: b},
		{From: a, To: gate, Route: StringRoute("r")},
		{From: b, To: gate, Route: StringRoute("r")},
		{From: gate, To: after},
	})
	if err != nil {
		t.Fatal(err)
	}
	ctx, _ := faninCtx(t, faninBudget)

	res := faninRun(t, w.Run(ctx))

	if res.err != nil {
		t.Errorf("Run error = %v, want nil", res.err)
	}
	if got := gateCalls.Load(); got != 4 {
		t.Errorf("attempts of gate = %d, want 4", got)
	}
	want := []faninObs{{Path: "wf@1/after@1", Value: "open:B"}}
	if diff := cmp.Diff(want, afterLog.get()); diff != "" {
		t.Errorf("after activations mismatch (-want +got):\n%s", diff)
	}
}
