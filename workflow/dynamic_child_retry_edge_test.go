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
	"testing"
	"testing/synctest"

	"github.com/google/go-cmp/cmp"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/session"
)

// TestRunNode_Child_ConcurrentCallsShareAllAttempts checks that a caller
// that joins a run of a child with the same WithRunID waits through every
// attempt of that run and receives the output of the attempt that succeeds.
//
// Fails if the in-flight slot is released between attempts, so the waiting
// caller receives a failed attempt's error or starts a run of its own.
func TestRunNode_Child_ConcurrentCallsShareAllAttempts(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		var log childAttemptLog
		release := make(chan struct{})
		child := NewFunctionNode("child", func(ctx agent.Context, _ any) (string, error) {
			n := log.add(ctx)
			if n == 1 {
				<-release
			}
			if n < 3 {
				return "", errAttempt
			}
			return "shared", nil
		}, NodeConfig{RetryConfig: retryNoDelay(3)})
		var leader, waiter callResult
		parent := NewDynamicNode("parent", func(ctx agent.Context, _ any, _ func(*session.Event) error) (any, error) {
			var wg sync.WaitGroup
			wg.Go(func() {
				out, err := RunNode[string](ctx, child, nil, WithRunID("k"))
				leader = callResult{out: out, err: err}
			})
			synctest.Wait() // the leader's first attempt waits on release
			wg.Go(func() {
				out, err := RunNode[string](ctx, child, nil, WithRunID("k"))
				waiter = callResult{out: out, err: err}
			})
			synctest.Wait() // the second caller waits on the leader's run
			close(release)
			wg.Wait()
			return nil, nil
		}, NodeConfig{})

		res := runParent(t, parent)

		if res.err != nil {
			t.Fatalf("Run error = %v, want nil", res.err)
		}
		if leader.out != "shared" || leader.err != nil {
			t.Errorf("leader's RunNode = (%v, %v), want (shared, nil)", leader.out, leader.err)
		}
		if waiter.out != "shared" || waiter.err != nil {
			t.Errorf("waiter's RunNode = (%v, %v), want (shared, nil)", waiter.out, waiter.err)
		}
		if diff := cmp.Diff([]int{1, 2, 3}, log.attempts()); diff != "" {
			t.Errorf("attempt numbers mismatch (-want +got):\n%s", diff)
		}
		if diff := cmp.Diff([]string{"failed", "failed", "output=shared"}, describeAt(res.events, "wf@1/parent@1/child@k")); diff != "" {
			t.Errorf("events at the child's path mismatch (-want +got):\n%s", diff)
		}
	})
}

// TestRunNode_Child_RetriedAcrossSubSchedulerKinds checks that a child is
// retried the same way when RunNode is called from a tool context inside a
// dynamic node, which gets a scheduler without replay, and from a context
// that carries no scheduler at all.
//
// Fails if the retry loop serves only the replaying scheduler of a dynamic
// node.
func TestRunNode_Child_RetriedAcrossSubSchedulerKinds(t *testing.T) {
	// flaky fails its first two attempts, writes its attempt number to
	// state on every attempt, and returns "ok" on the third.
	flaky := func(log *childAttemptLog) Node {
		return NewFunctionNode("child", func(ctx agent.Context, _ any) (string, error) {
			n := log.add(ctx)
			if err := ctx.State().Set("attempt", n); err != nil {
				return "", err
			}
			if n < 3 {
				return "", errAttempt
			}
			return "ok", nil
		}, NodeConfig{RetryConfig: retryNoDelay(3)})
	}

	t.Run("tool_call_in_dynamic_node", func(t *testing.T) {
		var log childAttemptLog
		child := flaky(&log)
		var got callResult
		parent := NewDynamicNode("parent", func(ctx agent.Context, _ any, _ func(*session.Event) error) (any, error) {
			out, err := RunNode[string](agent.NewToolContext(ctx, "fc-1", nil, nil), child, nil)
			got = callResult{out: out, err: err}
			return nil, nil
		}, NodeConfig{})

		res := runParent(t, parent)

		if res.err != nil {
			t.Fatalf("Run error = %v, want nil", res.err)
		}
		if got.out != "ok" || got.err != nil {
			t.Errorf("RunNode = (%v, %v), want (ok, nil)", got.out, got.err)
		}
		want := []childAttempt{
			{Path: childPathUnderParent, RunID: "1", Attempt: 1},
			{Path: childPathUnderParent, RunID: "1", Attempt: 2},
			{Path: childPathUnderParent, RunID: "1", Attempt: 3},
		}
		if diff := cmp.Diff(want, log.get()); diff != "" {
			t.Errorf("child runs mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("context_without_scheduler", func(t *testing.T) {
		var log childAttemptLog
		tc := sessionToolCtx(t)

		out, err := RunNode[string](tc, flaky(&log), nil)

		if out != "ok" || err != nil {
			t.Errorf("RunNode = (%v, %v), want (ok, nil)", out, err)
		}
		want := []childAttempt{
			{Path: "child@1", RunID: "1", Attempt: 1},
			{Path: "child@1", RunID: "1", Attempt: 2},
			{Path: "child@1", RunID: "1", Attempt: 3},
		}
		if diff := cmp.Diff(want, log.get()); diff != "" {
			t.Errorf("child runs mismatch (-want +got):\n%s", diff)
		}
		if got := tc.Actions().StateDelta["attempt"]; got != 3 {
			t.Errorf("caller's StateDelta[attempt] = %v, want 3", got)
		}
	})
}

// TestRunNode_Child_PauseIsNotRetried checks that an attempt that pauses,
// for input or as a WaitForOutput child without output, ends the call with
// the pause even when the RetryConfig allows more attempts, and leaves no
// failure record.
//
// Fails if a pause gets a retry decision or a failure record.
func TestRunNode_Child_PauseIsNotRetried(t *testing.T) {
	tests := []struct {
		name string
		cfg  NodeConfig
		// pause is what attempts after the first do: the first fails.
		pause   func(ctx agent.Context, yield func(*session.Event, error) bool)
		wantErr error
	}{
		{
			name: "request_input",
			cfg:  NodeConfig{RetryConfig: retryNoDelay(3)},
			pause: func(ctx agent.Context, yield func(*session.Event, error) bool) {
				yield(NewRequestInputEvent(ctx, session.RequestInput{InterruptID: "q1", Message: "?"}), nil)
			},
			wantErr: ErrNodeInterrupted,
		},
		{
			name:    "wait_for_output",
			cfg:     NodeConfig{RetryConfig: retryNoDelay(3), WaitForOutput: new(true)},
			pause:   func(agent.Context, func(*session.Event, error) bool) {},
			wantErr: ErrNodeWaitingForOutput,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var log childAttemptLog
			child := newFaninNode("child", tc.cfg, func(ctx agent.Context, _ any, yield func(*session.Event, error) bool) {
				if log.add(ctx) == 1 {
					yield(nil, errAttempt)
					return
				}
				tc.pause(ctx, yield)
			})
			parent, got := callingParent(child)

			res := runParent(t, parent)

			if err := receive(t, got).err; !errors.Is(err, tc.wantErr) {
				t.Errorf("RunNode error = %v, want %v", err, tc.wantErr)
			}
			if diff := cmp.Diff([]int{1, 2}, log.attempts()); diff != "" {
				t.Errorf("attempt numbers mismatch (-want +got):\n%s", diff)
			}
			var records int
			for _, ev := range res.events {
				if ev.NodeInfo != nil && ev.NodeInfo.Path == childPathUnderParent && workflowNodeOutcome(ev) == workflowNodeFailureOutcome {
					records++
				}
			}
			if records != 1 {
				t.Errorf("failure records at the child's path = %d, want 1", records)
			}
		})
	}
}

// TestRunNode_Child_ConsumerStoppedBetweenAttemptsDoesNotRetry checks that a
// child is not run again when the consumer refused the failed attempt's
// failure record, or refused one of its events, while the dynamic node's
// context is still live.
//
// Fails if retryChild retries without its failure record having reached the
// consumer, or if a refused event becomes a failure that RetryConfig may
// retry.
func TestRunNode_Child_ConsumerStoppedBetweenAttemptsDoesNotRetry(t *testing.T) {
	errStopped := errors.New("consumer stopped")
	isRecord := func(ev *session.Event) bool { return workflowNodeOutcome(ev) == workflowNodeFailureOutcome }
	tests := []struct {
		name string
		// refuse reports whether the consumer refuses ev.
		refuse       func(ev *session.Event) bool
		wantAttempts []int
		wantOut      any
		wantErr      error
	}{
		{
			name:         "every_event_taken",
			refuse:       func(*session.Event) bool { return false },
			wantAttempts: []int{1, 2, 3},
			wantOut:      "ok",
		},
		{
			name:         "failure_record_refused",
			refuse:       isRecord,
			wantAttempts: []int{1},
			wantErr:      errAttempt,
		},
		{
			name:         "event_refused",
			refuse:       func(ev *session.Event) bool { return !isRecord(ev) },
			wantAttempts: []int{1},
			wantErr:      errStopped,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var log childAttemptLog
			child := newFaninNode("child", NodeConfig{RetryConfig: retryNoDelay(3)}, func(ctx agent.Context, _ any, yield func(*session.Event, error) bool) {
				n := log.add(ctx)
				progress := session.NewEvent(ctx, ctx.InvocationID())
				progress.Actions.StateDelta = map[string]any{"progress": n}
				if !yield(progress, nil) {
					return
				}
				if n < 3 {
					yield(nil, errAttempt)
					return
				}
				faninEmit(ctx, yield, "ok")
			})
			sub := newDynamicSubScheduler(newTopLevelCtx(t), "wf@1/parent@1", func(ev *session.Event) error {
				if tc.refuse(ev) {
					return errStopped
				}
				return nil
			})

			out, err := sub.RunNode(child, nil, runNodeOptions{})

			if tc.wantErr == nil && (err != nil || out != tc.wantOut) {
				t.Errorf("RunNode = (%v, %v), want (%v, nil)", out, err, tc.wantOut)
			}
			if tc.wantErr != nil && !errors.Is(err, tc.wantErr) {
				t.Errorf("RunNode error = %v, want %v", err, tc.wantErr)
			}
			if diff := cmp.Diff(tc.wantAttempts, log.attempts()); diff != "" {
				t.Errorf("attempt numbers mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// TestRunNode_Child_PanicOnSecondAttemptDoesNotLeakFirstAttemptState checks
// that when a retried child's later attempt unwinds by a panic, the call's
// outcome is not taken from the earlier attempt that failed normally: the
// child's path stays in the dynamic node's failed-children inventory.
//
// Fails if the attempt loop keeps the earlier attempt's result while the
// next attempt runs.
func TestRunNode_Child_PanicOnSecondAttemptDoesNotLeakFirstAttemptState(t *testing.T) {
	child := NewDynamicNode("child", func(ctx agent.Context, _ any, _ func(*session.Event) error) (any, error) {
		if agent.AttemptCount(ctx) == 1 {
			return nil, errAttempt
		}
		panic("boom")
	}, NodeConfig{RetryConfig: retryNoDelay(2)})
	parent := NewDynamicNode("parent", func(ctx agent.Context, _ any, _ func(*session.Event) error) (out any, err error) {
		defer func() {
			if r := recover(); r != nil {
				err = fmt.Errorf("child panicked: %v", r)
			}
		}()
		return RunNode[any](ctx, child, nil)
	}, NodeConfig{})

	res := runParent(t, parent)

	if res.err == nil {
		t.Fatal("Run error = nil, want the parent's failure")
	}
	var failedPaths []any
	found := false
	for _, ev := range res.events {
		if ev.NodeInfo != nil && ev.NodeInfo.Path == "wf@1/parent@1" && workflowNodeOutcome(ev) == workflowNodeFailureOutcome {
			failedPaths = toAnySlice(ev.CustomMetadata[workflowFailedChildPathsKey])
			found = true
		}
	}
	if !found {
		t.Fatal("no failure record at wf@1/parent@1")
	}
	if diff := cmp.Diff([]any{childPathUnderParent}, failedPaths); diff != "" {
		t.Errorf("failed child paths on the parent's failure record mismatch (-want +got):\n%s", diff)
	}
}

// TestRunNode_WorkflowNodeChild_RetryRerunsCompletedInnerNodes pins what a
// WorkflowNode child retried by its own RetryConfig does with its inner
// nodes: the retry starts the inner graph from Start, so an inner node that
// completed in the failed attempt runs again. adk-python runs it once.
//
// Fails if RunNode does not retry a WorkflowNode child, or once a retried
// WorkflowNode continues its inner graph instead of starting it again.
func TestRunNode_WorkflowNodeChild_RetryRerunsCompletedInnerNodes(t *testing.T) {
	var okLog, flakyLog childAttemptLog
	innerOK := NewFunctionNode("inner_ok", func(ctx agent.Context, _ any) (string, error) {
		okLog.add(ctx)
		return "ok", nil
	}, NodeConfig{})
	var flakyCalls int
	innerFlaky := NewFunctionNode("inner_flaky", func(ctx agent.Context, in string) (string, error) {
		flakyLog.add(ctx)
		flakyCalls++
		if flakyCalls == 1 {
			return "", errAttempt
		}
		return in + "+recovered", nil
	}, NodeConfig{})
	sub, err := NewWorkflowNode("sub_wf", Chain(Start, innerOK, innerFlaky))
	if err != nil {
		t.Fatal(err)
	}
	sub.BaseNode = NewBaseNode("sub_wf", "", NodeConfig{RetryConfig: retryNoDelay(2)})
	parent, got := callingParent(sub)

	runParent(t, parent)

	if r := receive(t, got); r.out != "ok+recovered" || r.err != nil {
		t.Errorf("RunNode = (%v, %v), want (ok+recovered, nil)", r.out, r.err)
	}
	if diff := cmp.Diff([]int{1, 1}, flakyLog.attempts()); diff != "" {
		t.Errorf("inner_flaky attempt numbers mismatch (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff([]int{1, 1}, okLog.attempts()); diff != "" {
		t.Errorf("inner_ok attempt numbers mismatch (-want +got):\n%s", diff)
	}
}
