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
	"context"
	"errors"
	"sync/atomic"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/session"
)

// TestScheduler_QueuedTriggerOfBusyNodeDoesNotBlockLaterNodes covers
// B3 and B10: a trigger that waits behind a running activation of its own
// node is skipped, not a stopping point, so a trigger buffered after it
// starts at once. c@1 runs until e@1 has started; a completes while c@1
// runs and buffers c@2 first and e@1 second.
//
// Fails if scheduleReady stops its scan at the busy node's trigger: e@1
// then waits for c@1, c@1 waits for e@1, and the wait times out.
func TestScheduler_QueuedTriggerOfBusyNodeDoesNotBlockLaterNodes(t *testing.T) {
	var cCalls atomic.Int32
	var eStartedWhileC1Ran atomic.Bool
	var starts faninLog
	cStarted, eStarted := make(chan struct{}), make(chan struct{})
	a := newFaninNode("a", NodeConfig{}, func(ctx agent.Context, _ any, yield func(*session.Event, error) bool) {
		faninWait(cStarted, faninBudget)
		faninEmit(ctx, yield, "A", "r")
	})
	c := newFaninNode("c", NodeConfig{}, func(ctx agent.Context, _ any, yield func(*session.Event, error) bool) {
		starts.add(faninObs{Path: ctx.Path()})
		if cCalls.Add(1) == 1 {
			close(cStarted)
			eStartedWhileC1Ran.Store(faninWait(eStarted, faninBudget/2))
		}
		faninEmit(ctx, yield, nil)
	})
	e := newFaninNode("e", NodeConfig{}, func(ctx agent.Context, _ any, yield func(*session.Event, error) bool) {
		starts.add(faninObs{Path: ctx.Path()})
		close(eStarted)
		faninEmit(ctx, yield, nil)
	})
	w, err := New("wf", []Edge{
		{From: Start, To: c},
		{From: Start, To: a},
		{From: a, To: c, Route: StringRoute("r")},
		{From: a, To: e, Route: StringRoute("r")},
	})
	if err != nil {
		t.Fatal(err)
	}
	ctx, _ := faninCtx(t, faninBudget)

	res := faninRun(t, w.Run(ctx))

	if res.err != nil {
		t.Errorf("Run error = %v, want nil", res.err)
	}
	if !eStartedWhileC1Ran.Load() {
		t.Error("e@1 did not start while c@1 was running")
	}
	want := []faninObs{{Path: "wf@1/c@1"}, {Path: "wf@1/e@1"}, {Path: "wf@1/c@2"}}
	if diff := cmp.Diff(want, starts.get()); diff != "" {
		t.Errorf("start order mismatch (-want +got):\n%s", diff)
	}
}

// TestScheduler_MaxConcurrency_RetriggeredNodeStartsAfterBufferedNodes
// covers amendment 4: a node whose bucket emptied goes to the end of the
// buffer when it is triggered again. x@1 and y@1 are buffered by START
// with a cap of one; x@1 runs and routes back to x, and y@1, buffered
// before that second trigger, starts first (Python deletes an emptied
// bucket, _workflow.py:550-560).
//
// Fails if an emptied bucket keeps its place in the order: x@2 then starts
// before y@1.
func TestScheduler_MaxConcurrency_RetriggeredNodeStartsAfterBufferedNodes(t *testing.T) {
	var xCalls atomic.Int32
	var starts faninLog
	x := newFaninNode("x", NodeConfig{}, func(ctx agent.Context, _ any, yield func(*session.Event, error) bool) {
		starts.add(faninObs{Path: ctx.Path()})
		if xCalls.Add(1) == 1 {
			faninEmit(ctx, yield, nil, "again")
			return
		}
		faninEmit(ctx, yield, nil)
	})
	y := newFaninNode("y", NodeConfig{}, func(ctx agent.Context, _ any, yield func(*session.Event, error) bool) {
		starts.add(faninObs{Path: ctx.Path()})
		faninEmit(ctx, yield, nil)
	})
	w, err := New("wf", []Edge{
		{From: Start, To: x},
		{From: Start, To: y},
		{From: x, To: x, Route: StringRoute("again")},
	}, WithMaxConcurrency(1))
	if err != nil {
		t.Fatal(err)
	}
	ctx, _ := faninCtx(t, faninBudget)

	res := faninRun(t, w.Run(ctx))

	if res.err != nil {
		t.Errorf("Run error = %v, want nil", res.err)
	}
	want := []faninObs{{Path: "wf@1/x@1"}, {Path: "wf@1/y@1"}, {Path: "wf@1/x@2"}}
	if diff := cmp.Diff(want, starts.get()); diff != "" {
		t.Errorf("start order mismatch (-want +got):\n%s", diff)
	}
}

// TestDynamicNode_RootWrapperRetryAfterChildFailure_KeepsStaticRunIDs
// covers B4 and B12 under a root wrapper, where a static node's path is
// empty and so is the path the static run counters are keyed under. probe@1
// routes to loop, whose child fails on the first attempt; the retry
// restarts the child's run ids, and loop routes back to probe, whose second
// activation must still get run id 2.
//
// Fails if resetting the run counters of an empty path clears the whole
// table: probe would get run id 1 again.
func TestDynamicNode_RootWrapperRetryAfterChildFailure_KeepsStaticRunIDs(t *testing.T) {
	var childCalls, probeCalls atomic.Int32
	var runIDs faninLog
	probe := newFaninNode("probe", NodeConfig{}, func(ctx agent.Context, _ any, yield func(*session.Event, error) bool) {
		runIDs.add(faninObs{Value: ctx.RunID()})
		if probeCalls.Add(1) == 1 {
			faninEmit(ctx, yield, nil, "go")
			return
		}
		faninEmit(ctx, yield, nil)
	})
	child := NewFunctionNode("child", func(agent.Context, any) (string, error) {
		if childCalls.Add(1) == 1 {
			return "", errors.New("child failed")
		}
		return "child-ok", nil
	}, NodeConfig{})
	loop := NewDynamicNode("loop", func(ctx agent.Context, _ any, emit func(*session.Event) error) (string, error) {
		out, err := RunNode[string](ctx, child, nil)
		if err != nil {
			return "", err
		}
		ev := session.NewEvent(ctx, ctx.InvocationID())
		ev.Routes = []string{"back"}
		if err := emit(ev); err != nil {
			return "", err
		}
		return out, nil
	}, NodeConfig{RetryConfig: retryAlways(2)})
	w, err := New("wf", []Edge{
		{From: Start, To: probe},
		{From: probe, To: loop, Route: StringRoute("go")},
		{From: loop, To: probe, Route: StringRoute("back")},
	}, WithRootWrapper())
	if err != nil {
		t.Fatal(err)
	}

	res := faninRunWithSession(t, w)

	if res.err != nil {
		t.Errorf("Run error = %v, want nil", res.err)
	}
	if diff := cmp.Diff([]any{"1", "2"}, faninValues(runIDs.get())); diff != "" {
		t.Errorf("run ids of probe's activations mismatch (-want +got):\n%s", diff)
	}
}

// TestPersistence_HistoryWithoutRunIDsKeepsPredecessorInput covers history
// whose paths carry no run id ("wf@1/p", as events recorded before run ids
// existed): there is nothing to replay, so the paused node re-enters with
// its predecessor's last output.
//
// Fails if a node the replay knows nothing about gets no input: c would
// re-enter with nil.
func TestPersistence_HistoryWithoutRunIDsKeepsPredecessorInput(t *testing.T) {
	rerun := true
	var runs faninLog
	p := newFaninNode("p", NodeConfig{}, faninEmitter("P"))
	c := newFaninNode("c", NodeConfig{RerunOnResume: &rerun}, func(ctx agent.Context, in any, yield func(*session.Event, error) bool) {
		if _, ok := ctx.ResumedInput("ask"); ok {
			runs.add(faninObs{Path: ctx.Path(), Value: in})
			faninEmit(ctx, yield, "c-done")
			return
		}
		faninAsk(ctx, yield, "ask")
	})
	w, err := New("wf", []Edge{{From: Start, To: p}, {From: p, To: c}})
	if err != nil {
		t.Fatal(err)
	}
	pOut := &session.Event{InvocationID: faninInvocation, NodeInfo: &session.NodeInfo{Path: "wf@1/p"}}
	pOut.Output = "P"
	cPause := &session.Event{InvocationID: faninInvocation, NodeInfo: &session.NodeInfo{Path: "wf@1/c"}, LongRunningToolIDs: []string{"ask"}}
	conv := &faninConv{t: t, w: w, history: []*session.Event{pOut, cPause}}

	conv.reply(map[string]any{"ask": "yes"})

	want := []faninObs{{Path: "wf@1/c@1", Value: "P"}}
	if diff := cmp.Diff(want, runs.get()); diff != "" {
		t.Errorf("c activations mismatch (-want +got):\n%s", diff)
	}
}

// TestResume_DuplicateHandoffAnswerDoesNotRerunSuccessor covers amendment
// 7's idempotency rule: h was answered on turn 2 and its successor x ran.
// A later message repeats h's answer and answers g. x must not run again;
// g re-enters with its answer.
//
// Fails if Resume re-triggers the successors of a handoff asker whose
// answer is already in history: x would run a second time, as x@2.
func TestResume_DuplicateHandoffAnswerDoesNotRerunSuccessor(t *testing.T) {
	var xLog, gLog faninLog
	h := faninAsker("h", "ask-h", false, false)
	rerun := true
	g := newFaninNode("g", NodeConfig{RerunOnResume: &rerun}, func(ctx agent.Context, _ any, yield func(*session.Event, error) bool) {
		if v, ok := ctx.ResumedInput("ask-g"); ok {
			gLog.add(faninObs{Path: ctx.Path(), Value: v})
			faninEmit(ctx, yield, nil)
			return
		}
		faninAsk(ctx, yield, "ask-g")
	})
	x := newFaninNode("x", NodeConfig{}, func(ctx agent.Context, in any, yield func(*session.Event, error) bool) {
		xLog.add(faninObs{Path: ctx.Path(), Value: in})
		faninEmit(ctx, yield, "x-done")
	})
	w, err := New("wf", []Edge{
		{From: Start, To: h},
		{From: Start, To: g},
		{From: h, To: x},
	})
	if err != nil {
		t.Fatal(err)
	}
	conv := startFaninConv(t, w)
	conv.reply(map[string]any{"ask-h": "H"})

	turn3 := conv.reply(map[string]any{"ask-h": "H", "ask-g": "G"})

	if turn3.err != nil {
		t.Errorf("turn 3 Resume error = %v, want nil", turn3.err)
	}
	want := []faninObs{{Path: "wf@1/x@1", Value: "H"}}
	if diff := cmp.Diff(want, xLog.get()); diff != "" {
		t.Errorf("x activations mismatch (-want +got):\n%s", diff)
	}
	wantG := []faninObs{{Path: "wf@1/g@1", Value: "G"}}
	if diff := cmp.Diff(wantG, gLog.get()); diff != "" {
		t.Errorf("g activations mismatch (-want +got):\n%s", diff)
	}
}

// TestDynamicNode_ParentRetryAfterChildFailure_RestartsNestedChildRunIDs
// covers B12 for a child that has children of its own: when attempt 1
// fails because grandchild failed, the retry restarts the run ids under
// every nested path, so attempt 2 runs child@1 and grandchild@1 again.
//
// Fails if the reset forgets only the counters directly under the
// activation's path: grandchild would run as grandchild@2 in attempt 2.
func TestDynamicNode_ParentRetryAfterChildFailure_RestartsNestedChildRunIDs(t *testing.T) {
	var grandchildCalls atomic.Int32
	var grandchildPaths faninLog
	grandchild := NewFunctionNode("grandchild", func(ctx agent.Context, _ any) (string, error) {
		grandchildPaths.add(faninObs{Path: ctx.Path()})
		if grandchildCalls.Add(1) == 1 {
			return "", errors.New("grandchild failed")
		}
		return "ok", nil
	}, NodeConfig{})
	child := NewDynamicNode("child", func(ctx agent.Context, _ any, _ func(*session.Event) error) (string, error) {
		return RunNode[string](ctx, grandchild, nil)
	}, NodeConfig{})
	parent := NewDynamicNode("parent", func(ctx agent.Context, _ any, _ func(*session.Event) error) (string, error) {
		return RunNode[string](ctx, child, nil)
	}, NodeConfig{RetryConfig: retryAlways(2)})
	w, err := New("wf", []Edge{{From: Start, To: parent}})
	if err != nil {
		t.Fatal(err)
	}

	res := faninRunWithSession(t, w)

	if res.err != nil {
		t.Errorf("Run error = %v, want nil", res.err)
	}
	want := []faninObs{
		{Path: "wf@1/parent@1/child@1/grandchild@1"},
		{Path: "wf@1/parent@1/child@1/grandchild@1"},
	}
	if diff := cmp.Diff(want, grandchildPaths.get()); diff != "" {
		t.Errorf("grandchild attempt paths mismatch (-want +got):\n%s", diff)
	}
}

// TestScheduler_FailureUnderCancellation_RetryLeftEndsCanceled covers a node
// attempt that fails while the invocation context is already dead. With a
// retry left, the failure is not the node's final outcome: the run ends with
// the cancellation and the attempt does not run again, as in adk-python,
// which decides the retry right after the failure and is canceled in the
// backoff sleep (workflow/_node_runner.py, _attempt_retry). Without one, the
// node's own error is the outcome and the run keeps it. The node cancels the
// context before it fails, so the scheduler classifies the completion under a
// dead context on every run.
//
// Fails if handleCompletion reports a dead context's failed completion
// without asking whether the attempt would be retried (the first case ends
// with "attempt failed"), or turns every failure of a node with a RetryConfig
// into a cancellation (the second case ends with context.Canceled).
func TestScheduler_FailureUnderCancellation_RetryLeftEndsCanceled(t *testing.T) {
	nodeErr := errors.New("attempt failed")
	tests := []struct {
		name        string
		shouldRetry bool
		want        error
	}{
		{name: "retry left", shouldRetry: true, want: context.Canceled},
		{name: "no retry left", shouldRetry: false, want: nodeErr},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ctx, cancel := faninCtx(t, faninBudget)
			var calls atomic.Int32
			n := newFaninNode("n", NodeConfig{RetryConfig: &RetryConfig{
				MaxAttempts:   3,
				InitialDelay:  time.Millisecond,
				BackoffFactor: 1.0,
				ShouldRetry:   func(error) bool { return tc.shouldRetry },
			}}, func(_ agent.Context, _ any, yield func(*session.Event, error) bool) {
				calls.Add(1)
				cancel()
				yield(nil, nodeErr)
			})
			w, err := New("wf", []Edge{{From: Start, To: n}})
			if err != nil {
				t.Fatal(err)
			}

			res := faninRun(t, w.Run(ctx))

			if !errors.Is(res.err, tc.want) {
				t.Errorf("Run error = %v, want %v", res.err, tc.want)
			}
			if got := calls.Load(); got != 1 {
				t.Errorf("node attempts = %d, want 1", got)
			}
		})
	}
}

// TestScheduler_LastAttemptFailsUnderCancellation_KeepsNodeError covers the
// attempt-count boundary of the same rule: the last attempt RetryConfig
// allows fails while the invocation context is dead. No retry is left, so
// the failure is the node's outcome and the run keeps the node's own error,
// as adk-python's _should_retry_node returns False at
// attempt_count >= max_attempts (workflow/_node_runner.py). The first attempt
// fails under a live context and is retried; the second cancels the context
// and then fails.
//
// Fails if handleCompletion asks whether the attempt would be retried with
// the count of earlier failed attempts instead of the count including this
// one (the run ends with context.Canceled).
func TestScheduler_LastAttemptFailsUnderCancellation_KeepsNodeError(t *testing.T) {
	nodeErr := errors.New("attempt failed")
	ctx, cancel := faninCtx(t, faninBudget)
	var calls atomic.Int32
	n := newFaninNode("n", NodeConfig{RetryConfig: &RetryConfig{
		MaxAttempts:   2,
		InitialDelay:  time.Millisecond,
		BackoffFactor: 1.0,
		ShouldRetry:   func(error) bool { return true },
	}}, func(_ agent.Context, _ any, yield func(*session.Event, error) bool) {
		if calls.Add(1) == 2 {
			cancel()
		}
		yield(nil, nodeErr)
	})
	w, err := New("wf", []Edge{{From: Start, To: n}})
	if err != nil {
		t.Fatal(err)
	}

	res := faninRun(t, w.Run(ctx))

	if !errors.Is(res.err, nodeErr) {
		t.Errorf("Run error = %v, want %v", res.err, nodeErr)
	}
	if got := calls.Load(); got != 2 {
		t.Errorf("node attempts = %d, want 2", got)
	}
}
