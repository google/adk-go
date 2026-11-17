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
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/session"
)

// TestScheduler_RepeatedActivation_FanOutSubBranchesUseRunID covers B1,
// B4 and B6: each activation of a looping node gets the next run id, and
// when an activation fans out, each target's sub-branch names the run id
// of the activation it starts.
//
// Fails if startNode derives the sub-branch with "@1" instead of the
// allocated run id, or allocates no new run id per activation.
func TestScheduler_RepeatedActivation_FanOutSubBranchesUseRunID(t *testing.T) {
	var calls atomic.Int32
	loop := newFaninNode("loop", NodeConfig{}, func(ctx agent.Context, _ any, yield func(*session.Event, error) bool) {
		k := calls.Add(1)
		route := "again"
		if k == 3 {
			route = "exit"
		}
		faninEmit(ctx, yield, fmt.Sprintf("iter-%d", k), route)
	})
	side := newFaninNode("side", NodeConfig{}, faninEmitter(nil))
	done := newFaninNode("done", NodeConfig{}, faninEmitter(nil))
	w, err := New("wf", []Edge{
		{From: Start, To: loop},
		{From: loop, To: loop, Route: StringRoute("again")},
		{From: loop, To: side, Route: StringRoute("again")},
		{From: loop, To: done, Route: StringRoute("exit")},
	})
	if err != nil {
		t.Fatal(err)
	}
	ctx, _ := faninCtx(t, faninBudget)

	res := faninRun(t, w.Run(ctx))

	if res.err != nil {
		t.Fatalf("Run error = %v, want nil", res.err)
	}
	for _, tc := range []struct {
		node string
		want []faninObs
	}{
		{node: "loop", want: []faninObs{
			{Path: "wf@1/loop@1", Branch: ""},
			{Path: "wf@1/loop@2", Branch: "loop@2"},
			{Path: "wf@1/loop@3", Branch: "loop@2.loop@3"},
		}},
		{node: "side", want: []faninObs{
			{Path: "wf@1/side@1", Branch: "side@1"},
			{Path: "wf@1/side@2", Branch: "loop@2.side@2"},
		}},
		{node: "done", want: []faninObs{
			{Path: "wf@1/done@1", Branch: "loop@2.loop@3"},
		}},
	} {
		if diff := cmp.Diff(tc.want, faninActivations(res.events, tc.node)); diff != "" {
			t.Errorf("%s activations mismatch (-want +got):\n%s", tc.node, diff)
		}
	}
}

// TestScheduler_RetryPreservesActivationRunID covers B4 and ruling 3:
// every retry attempt of an activation runs under that activation's run
// id and path, and the next activation takes the next run id.
//
// Fails if a retry allocates a new run id (scheduleRetry dropping the
// activation's runID), or if activations do not get sequential run ids.
func TestScheduler_RetryPreservesActivationRunID(t *testing.T) {
	var calls atomic.Int32
	var mu sync.Mutex
	var attempts []faninObs
	cfg := NodeConfig{RetryConfig: &RetryConfig{
		MaxAttempts:   2,
		InitialDelay:  time.Millisecond,
		BackoffFactor: 1,
		ShouldRetry:   func(error) bool { return true },
	}}
	flaky := newFaninNode("flaky", cfg, func(ctx agent.Context, _ any, yield func(*session.Event, error) bool) {
		mu.Lock()
		attempts = append(attempts, faninObs{Path: ctx.Path(), Value: ctx.RunID()})
		mu.Unlock()
		switch calls.Add(1) {
		case 1, 3:
			yield(nil, errors.New("transient"))
		case 2:
			faninEmit(ctx, yield, "out-1", "again")
		default:
			faninEmit(ctx, yield, "out-2", "done")
		}
	})
	w, err := New("wf", []Edge{
		{From: Start, To: flaky},
		{From: flaky, To: flaky, Route: StringRoute("again")},
	})
	if err != nil {
		t.Fatal(err)
	}
	ctx, _ := faninCtx(t, faninBudget)

	res := faninRun(t, w.Run(ctx))

	if res.err != nil {
		t.Fatalf("Run error = %v, want nil", res.err)
	}
	wantAttempts := []faninObs{
		{Path: "wf@1/flaky@1", Value: "1"},
		{Path: "wf@1/flaky@1", Value: "1"},
		{Path: "wf@1/flaky@2", Value: "2"},
		{Path: "wf@1/flaky@2", Value: "2"},
	}
	mu.Lock()
	defer mu.Unlock()
	if diff := cmp.Diff(wantAttempts, attempts); diff != "" {
		t.Errorf("attempts (path, run id) mismatch (-want +got):\n%s", diff)
	}
	wantOutputs := []faninObs{
		{Path: "wf@1/flaky@1", Value: "out-1"},
		{Path: "wf@1/flaky@2", Value: "out-2"},
	}
	if diff := cmp.Diff(wantOutputs, faninOutputs(res.events, "flaky")); diff != "" {
		t.Errorf("flaky outputs mismatch (-want +got):\n%s", diff)
	}
}

// TestScheduler_FanOutWithBlockedJoin_UsesSubBranch covers B6: a
// completion with two targets gives the non-join target a sub-branch even
// when the other target is a join whose barrier is still closed, as
// Python counts every target (_workflow.py:853-857).
//
// Fails if the sub-branch decision counts only the targets that start.
func TestScheduler_FanOutWithBlockedJoin_UsesSubBranch(t *testing.T) {
	xStarted := make(chan struct{})
	a := newFaninNode("a", NodeConfig{}, faninEmitter("A"))
	b := newFaninNode("b", NodeConfig{}, func(ctx agent.Context, _ any, yield func(*session.Event, error) bool) {
		faninWait(xStarted, faninBudget)
		faninEmit(ctx, yield, "B")
	})
	x := newFaninNode("x", NodeConfig{}, func(ctx agent.Context, _ any, yield func(*session.Event, error) bool) {
		close(xStarted)
		faninEmit(ctx, yield, nil)
	})
	j := NewJoinNode("j")
	w, err := New("wf", []Edge{
		{From: Start, To: a},
		{From: Start, To: b},
		{From: a, To: x},
		{From: a, To: j},
		{From: b, To: j},
	})
	if err != nil {
		t.Fatal(err)
	}
	ctx, _ := faninCtx(t, faninBudget)

	res := faninRun(t, w.Run(ctx))

	if res.err != nil {
		t.Fatalf("Run error = %v, want nil", res.err)
	}
	want := []faninObs{{Path: "wf@1/x@1", Branch: "a@1.x@1"}}
	if diff := cmp.Diff(want, faninActivations(res.events, "x")); diff != "" {
		t.Errorf("x activations mismatch (-want +got):\n%s", diff)
	}
}
