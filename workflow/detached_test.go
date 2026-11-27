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
	"iter"
	"runtime"
	"strings"
	"testing"
	"testing/synctest"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/session"
)

// The tests in this file port adk-python's detached dynamic node tests
// (tests/unittests/workflow/test_workflow_dynamic_nodes.py:1095-1230 at
// v2.12.0). A parent dynamic node starts a child with RunNode on a
// goroutine it does not wait for, then returns; the child finishes
// after a delay, so it is still running when the graph finishes.
//
// adk-python's parent awaits asyncio.sleep(0) so that the detached
// run starts before the parent returns: only a started run is tracked.
// Here the parent waits until the child is running.

// detachedParent returns a dynamic node that starts child on a
// goroutine it does not wait for, with input, and returns "parent
// done" once the child is running, which child reports by closing
// started. The child's RunNode result goes to result, when non-nil.
func detachedParent(child Node, started <-chan struct{}, input any, result chan<- error) Node {
	return NewDynamicNode("parent", func(ctx agent.Context, _ any, _ func(*session.Event) error) (string, error) {
		go func() {
			_, err := RunNode[any](ctx, child, input)
			if result != nil {
				result <- err
			}
		}()
		<-started
		return "parent done", nil
	}, NodeConfig{})
}

// delayedNode returns a node that closes the returned channel when it
// starts, waits delay, then emits ev(ctx) or fails with err.
func delayedNode(name string, delay time.Duration, ev func(agent.Context) *session.Event, err error) (Node, <-chan struct{}) {
	started := make(chan struct{})
	return newCustomNode(name, func(ctx agent.Context, _ any, yield func(*session.Event, error) bool) {
		close(started)
		time.Sleep(delay)
		if err != nil {
			yield(nil, err)
			return
		}
		yield(ev(ctx), nil)
	}), started
}

func outputEvent(out any) func(agent.Context) *session.Event {
	return func(ctx agent.Context) *session.Event {
		ev := session.NewEvent(ctx, ctx.InvocationID())
		ev.Output = out
		return ev
	}
}

// runDetached runs a workflow START -> parent and returns its outputs
// and first error.
func runDetached(t *testing.T, parent Node) ([]string, error) {
	t.Helper()
	w := mustNew(t, []Edge{{From: Start, To: parent}})
	var events []*session.Event
	var runErr error
	for ev, err := range w.Run(newSeededMockCtx(t)) {
		if err != nil {
			if runErr == nil {
				runErr = err
			}
			continue
		}
		events = append(events, ev)
	}
	return outputsOf(events), runErr
}

func TestDetachedDynamicNode_FailureSurfaces(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		boom := errors.New("detached boom")
		child, started := delayedNode("failing_child", 50*time.Millisecond, nil, boom)

		_, err := runDetached(t, detachedParent(child, started, "fail", nil))

		if !errors.Is(err, boom) {
			t.Errorf("workflow error = %v, want %v", err, boom)
		}
	})
}

func TestDetachedDynamicNode_InterruptSurfaces(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		child, started := delayedNode("interrupting_child", 50*time.Millisecond, func(ctx agent.Context) *session.Event {
			ev := session.NewEvent(ctx, ctx.InvocationID())
			ev.Content = &genai.Content{Parts: []*genai.Part{
				{FunctionCall: &genai.FunctionCall{Name: "approve", ID: "fc-det"}},
			}}
			ev.LongRunningToolIDs = []string{"fc-det"}
			return ev
		}, nil)

		_, err := runDetached(t, detachedParent(child, started, nil, nil))

		if !errors.Is(err, errDetachedInterrupted) {
			t.Errorf("workflow error = %v, want errDetachedInterrupted", err)
		}
	})
}

func TestDetachedDynamicNode_SuccessKeepsWorkflowSucceeding(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		result := make(chan error, 1)
		child, started := delayedNode("clean_child", 50*time.Millisecond, outputEvent("child done"), nil)

		outputs, err := runDetached(t, detachedParent(child, started, nil, result))
		if err != nil {
			t.Fatalf("workflow error: %v", err)
		}

		// adk-python yields the detached child's events as well, on the
		// invocation's event queue.
		if diff := cmp.Diff([]string{"child done", "parent done"}, outputs); diff != "" {
			t.Errorf("workflow outputs mismatch (-want +got):\n%s", diff)
		}
		if err := <-result; err != nil {
			t.Errorf("detached RunNode error: %v", err)
		}
	})
}

// A child that finishes before its parent body returns was never
// detached: its failure goes to the body, which here ignores it, and
// the workflow succeeds. adk-python's version of this test pins a wider
// limitation, that no detached run finished before the graph ends is
// inspected; Go reports a detached child's failure whenever it happens.
func TestDetachedDynamicNode_AwaitedChildFailureIsHandledByBody(t *testing.T) {
	boom := errors.New("early boom")
	child, _ := delayedNode("early_failing_child", 0, nil, boom)
	parent := NewDynamicNode("parent", func(ctx agent.Context, _ any, _ func(*session.Event) error) (string, error) {
		done := make(chan error, 1)
		go func() {
			_, err := RunNode[any](ctx, child, "fail")
			done <- err
		}()
		// Let the child finish while the graph is still running.
		if err := <-done; !errors.Is(err, boom) {
			return "", err
		}
		return "parent done", nil
	}, NodeConfig{})

	outputs, err := runDetached(t, parent)
	if err != nil {
		t.Fatalf("workflow error: %v", err)
	}
	if diff := cmp.Diff([]string{"parent done"}, outputs); diff != "" {
		t.Errorf("workflow outputs mismatch (-want +got):\n%s", diff)
	}
}

// A detached child that waits for output and finishes without one is
// not a failure: adk-python checks only a detached child's error and
// interrupt IDs (Workflow._surface_detached_dynamic_outcome).
func TestDetachedDynamicNode_WaitForOutputWithoutOutputIsNotAnError(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		started := make(chan struct{})
		child := &customNodeFunc{
			BaseNode: NewBaseNode("waiting_child", "", NodeConfig{WaitForOutput: new(true)}),
			run: func(agent.Context, any, func(*session.Event, error) bool) {
				close(started)
				time.Sleep(50 * time.Millisecond)
			},
		}

		outputs, err := runDetached(t, detachedParent(child, started, nil, nil))
		if err != nil {
			t.Fatalf("workflow error: %v", err)
		}
		if diff := cmp.Diff([]string{"parent done"}, outputs); diff != "" {
			t.Errorf("workflow outputs mismatch (-want +got):\n%s", diff)
		}
	})
}

// A RunNode started after its parent body returned, while the workflow
// still runs, is detached from the start: the workflow waits for it and
// fails with its error. Here a second node keeps the workflow running
// until the late child has started.
func TestDetachedDynamicNode_StartedAfterBodyReturnsIsTracked(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		boom := errors.New("late boom")
		child, childStarted := delayedNode("late_child", 50*time.Millisecond, nil, boom)
		release := make(chan struct{})
		parent := NewDynamicNode("parent", func(ctx agent.Context, _ any, _ func(*session.Event) error) (string, error) {
			go func() {
				<-release
				_, _ = RunNode[any](ctx, child, nil)
			}()
			return "parent done", nil
		}, NodeConfig{})
		// next releases the late RunNode once the parent has completed,
		// and returns only after the child has started.
		next := newFnNode("next", func(agent.Context) (any, error) {
			close(release)
			<-childStarted
			return "next done", nil
		})
		w := mustNew(t, []Edge{{From: Start, To: parent}, {From: parent, To: next}})

		var runErr error
		for _, err := range w.Run(newSeededMockCtx(t)) {
			if err != nil && runErr == nil {
				runErr = err
			}
		}
		if !errors.Is(runErr, boom) {
			t.Errorf("workflow error = %v, want %v", runErr, boom)
		}
	})
}

// A child that fails while its parent is yielding its own output is
// detached by then: the parent body has returned. Its failure must
// still fail the workflow, not be lost between the body returning and
// the node's iterator ending.
func TestDetachedDynamicNode_ChildFailingDuringParentYieldSurfaces(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		boom := errors.New("boom during yield")
		failNow := make(chan struct{})
		child := newCustomNode("child", func(_ agent.Context, _ any, yield func(*session.Event, error) bool) {
			<-failNow
			yield(nil, boom)
		})
		parent := NewDynamicNode("parent", func(ctx agent.Context, _ any, _ func(*session.Event) error) (string, error) {
			go func() { _, _ = RunNode[any](ctx, child, nil) }()
			synctest.Wait() // the child is running, blocked on failNow
			return "parent done", nil
		}, NodeConfig{})
		w := mustNew(t, []Edge{{From: Start, To: parent}})

		var runErr error
		for ev, err := range w.Run(newSeededMockCtx(t)) {
			if err != nil && runErr == nil {
				runErr = err
			}
			if ev != nil && ev.Output == "parent done" {
				// The parent's yield waits for this loop body: let the
				// child fail and finish before the parent's iterator ends.
				close(failNow)
				synctest.Wait()
			}
		}
		if !errors.Is(runErr, boom) {
			t.Errorf("workflow error = %v, want %v", runErr, boom)
		}
	})
}

// An event on its way to the consumer keeps the tracker open until the
// consumer has taken it, so the send cannot block on a queue nobody
// reads any more.
func TestDetachedRuns_InFlightSendKeepsTrackerOpen(t *testing.T) {
	queue := make(chan queueItem) // unbuffered: the send blocks until read
	d := &detachedRuns{queue: queue}
	sent := make(chan error, 1)
	go func() { sent <- d.send(&session.Event{}, "") }()

	for !func() bool { d.mu.Lock(); defer d.mu.Unlock(); return d.sending > 0 }() {
		runtime.Gosched()
	}
	if d.closeIfIdle() {
		t.Fatal("closeIfIdle() = true while an event is being sent, want false")
	}
	item := (<-queue).(detachedEventItem)
	d.handled(false)
	close(item.processed)
	if err := <-sent; err != nil {
		t.Errorf("send error: %v", err)
	}
	if !d.closeIfIdle() {
		t.Error("closeIfIdle() = false once the event was handled, want true")
	}
	if err := d.send(&session.Event{}, ""); !errors.Is(err, errDetachedUndeliverable) {
		t.Errorf("send after close error = %v, want errDetachedUndeliverable", err)
	}
}

// Once the workflow fails, it cancels its detached children, and an
// event emitted through a completed node after that is refused rather
// than yielded. The detached child holds the workflow open after the
// cancellation until the parent's own emit has been called late.
func TestDetachedDynamicNode_EmitAfterWorkflowFailedIsRefused(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		boom := errors.New("sibling boom")
		var parentEmit func(*session.Event) error
		var lateErr error
		child := newCustomNode("child", func(ctx agent.Context, _ any, _ func(*session.Event, error) bool) {
			<-ctx.Done() // the workflow failed and canceled its nodes
			lateErr = parentEmit(outputEvent("too late")(ctx))
		})
		parent := NewDynamicNode("parent", func(ctx agent.Context, _ any, emit func(*session.Event) error) (string, error) {
			parentEmit = emit
			go func() { _, _ = RunNode[any](ctx, child, nil) }()
			synctest.Wait() // the child is running
			return "parent done", nil
		}, NodeConfig{})
		failing := newFnNode("failing", func(agent.Context) (any, error) { return nil, boom })
		w := mustNew(t, []Edge{{From: Start, To: parent}, {From: parent, To: failing}})

		var outputs []any
		var runErr error
		for ev, err := range w.Run(newSeededMockCtx(t)) {
			if err != nil && runErr == nil {
				runErr = err
			}
			if ev != nil && ev.Output != nil {
				outputs = append(outputs, ev.Output)
			}
		}

		if !errors.Is(runErr, boom) {
			t.Errorf("workflow error = %v, want %v", runErr, boom)
		}
		if !errors.Is(lateErr, context.Canceled) {
			t.Errorf("late emit error = %v, want context.Canceled", lateErr)
		}
		if diff := cmp.Diff([]any{"parent done"}, outputs); diff != "" {
			t.Errorf("workflow outputs mismatch (-want +got):\n%s", diff)
		}
	})
}

// When one detached child fails, the workflow cancels the others
// rather than waiting for them to finish on their own.
func TestDetachedDynamicNode_FailureCancelsOtherDetachedChildren(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		boom := errors.New("detached boom")
		failing, failingStarted := delayedNode("failing_child", 50*time.Millisecond, nil, boom)
		blockedErr := make(chan error, 1)
		blocked := newCustomNode("blocked_child", func(ctx agent.Context, _ any, _ func(*session.Event, error) bool) {
			<-ctx.Done()
			blockedErr <- ctx.Err()
		})
		parent := NewDynamicNode("parent", func(ctx agent.Context, _ any, _ func(*session.Event) error) (string, error) {
			go func() { _, _ = RunNode[any](ctx, blocked, nil) }()
			go func() { _, _ = RunNode[any](ctx, failing, nil) }()
			<-failingStarted
			return "parent done", nil
		}, NodeConfig{})

		_, err := runDetached(t, parent)

		if !errors.Is(err, boom) {
			t.Errorf("workflow error = %v, want %v", err, boom)
		}
		if err := <-blockedErr; !errors.Is(err, context.Canceled) {
			t.Errorf("blocked child's context error = %v, want context.Canceled", err)
		}
	})
}

// An invocation deadline that expires while a detached child runs is
// reported as the deadline, as for any other node, not wrapped in the
// child's echo of it.
func TestDetachedDynamicNode_InvocationDeadlineIsReportedAsIs(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		baseCtx, cancel := context.WithTimeout(t.Context(), time.Second)
		defer cancel()
		child := newCustomNode("child", func(ctx agent.Context, _ any, yield func(*session.Event, error) bool) {
			<-ctx.Done()
			yield(nil, ctx.Err())
		})
		parent := NewDynamicNode("parent", func(ctx agent.Context, _ any, _ func(*session.Event) error) (string, error) {
			go func() { _, _ = RunNode[any](ctx, child, nil) }()
			synctest.Wait() // the child is running
			return "parent done", nil
		}, NodeConfig{})
		w := mustNew(t, []Edge{{From: Start, To: parent}})
		ictx := newSeededMockCtx(t)
		ictx.Context = baseCtx

		var runErr error
		for _, err := range w.Run(ictx) {
			if err != nil && runErr == nil {
				runErr = err
			}
		}
		if runErr != context.DeadlineExceeded { //nolint:errorlint // the deadline itself, not an error wrapping it
			t.Errorf("workflow error = %v, want context.DeadlineExceeded itself", runErr)
		}
	})
}

// Two overlapping calls sharing one WithRunID each lead a run of the
// same child path when the first fails: the second does not share a
// failure, and runs the child again. Each detached run must still
// report its own outcome, or the workflow waits forever.
func TestDetachedDynamicNode_SharedRunIDRunsEachReportOutcome(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		boom := errors.New("first attempt")
		var attempts int
		child := newCustomNode("child", func(ctx agent.Context, _ any, yield func(*session.Event, error) bool) {
			attempts++
			time.Sleep(50 * time.Millisecond)
			if attempts == 1 {
				yield(nil, boom)
				return
			}
			yield(outputEvent("second")(ctx), nil)
		})
		parent := NewDynamicNode("parent", func(ctx agent.Context, _ any, _ func(*session.Event) error) (string, error) {
			for range 2 {
				go func() {
					if _, err := RunNode[any](ctx, child, nil, WithRunID("x")); err != nil {
						_, _ = RunNode[any](ctx, child, nil, WithRunID("x"))
					}
				}()
			}
			synctest.Wait() // the first run leads; the other caller waits on it
			return "parent done", nil
		}, NodeConfig{})

		_, err := runDetached(t, parent)

		if !errors.Is(err, boom) {
			t.Errorf("workflow error = %v, want %v", err, boom)
		}
	})
}

// After the consumer breaks, a detached child's event must not be
// yielded to it again: calling yield after the consumer's range loop
// ended panics. A sibling failing first makes the workflow drain while
// the child's event is still on its way; the consumer has broken by
// then. The event is taken but not yielded.
func TestDetachedDynamicNode_DrainingStopsYielding(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		send := make(chan struct{})
		child := newCustomNode("child", func(ctx agent.Context, _ any, yield func(*session.Event, error) bool) {
			<-send
			yield(outputEvent("while draining")(ctx), nil)
		})
		parent := NewDynamicNode("parent", func(ctx agent.Context, _ any, _ func(*session.Event) error) (string, error) {
			go func() { _, _ = RunNode[any](ctx, child, nil) }()
			synctest.Wait() // the child is running
			return "parent done", nil
		}, NodeConfig{})
		w := mustNew(t, []Edge{{From: Start, To: parent}})

		for ev, err := range w.Run(newSeededMockCtx(t)) {
			if err != nil {
				t.Fatalf("workflow error: %v", err)
			}
			if ev.Output == "parent done" {
				close(send)
				synctest.Wait() // the child's event waits on the queue
				break
			}
		}
	})
}

// A panicking emitting function node closes its emit too: an event its
// goroutine emits afterwards must not run the finished iterator's loop
// body, where it would be credited to the node's retry and fail it with
// ErrMultipleOutputs.
func TestEmittingFunctionNode_PanicClosesEmit(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		var attempts int
		lateErr := make(chan error, 1)
		node := NewEmittingFunctionNode("flaky", func(ctx agent.Context, _ any, emit func(*session.Event) error) (string, error) {
			attempts++
			if attempts == 1 {
				go func() {
					time.Sleep(time.Second) // the retry is running
					lateErr <- emit(outputEvent("late")(ctx))
				}()
				panic("first attempt")
			}
			time.Sleep(2 * time.Second)
			return "retried", nil
		}, NodeConfig{RetryConfig: &RetryConfig{MaxAttempts: 2, InitialDelay: time.Millisecond, MaxDelay: time.Millisecond, BackoffFactor: 1}})

		outputs, err := runDetached(t, node)
		if err != nil {
			t.Fatalf("workflow error: %v", err)
		}
		if diff := cmp.Diff([]string{"retried"}, outputs); diff != "" {
			t.Errorf("workflow outputs mismatch (-want +got):\n%s", diff)
		}
		if err := <-lateErr; !errors.Is(err, errEmitAfterReturn) {
			t.Errorf("late emit error = %v, want errEmitAfterReturn", err)
		}
	})
}

// A ParallelWorker cancels its context once its items are done, which
// cancels a child the wrapped dynamic node detached. That is not a
// failure: adk-python skips a canceled detached task.
func TestDetachedDynamicNode_CanceledByParallelWorkerIsNotAnError(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		child := newCustomNode("child", func(ctx agent.Context, _ any, yield func(*session.Event, error) bool) {
			select {
			case <-ctx.Done():
				yield(nil, ctx.Err())
			case <-time.After(50 * time.Millisecond):
				yield(outputEvent("child done")(ctx), nil)
			}
		})
		dyn := NewDynamicNode("dyn", func(ctx agent.Context, in any, _ func(*session.Event) error) (any, error) {
			go func() { _, _ = RunNode[any](ctx, child, nil) }()
			synctest.Wait() // the child is running
			return in, nil
		}, NodeConfig{})
		seed := newFnNode("seed", func(agent.Context) (any, error) { return []any{1}, nil })
		pw, err := NewParallelWorker("pw", dyn, 0, NodeConfig{})
		if err != nil {
			t.Fatalf("NewParallelWorker: %v", err)
		}
		w := mustNew(t, []Edge{{From: Start, To: seed}, {From: seed, To: pw}})

		for _, err := range w.Run(newSeededMockCtx(t)) {
			if err != nil {
				t.Errorf("workflow error: %v", err)
			}
		}
	})
}

// TestDetachedEmitAfterBodyReturns hands emit, or RunNode, to a
// goroutine the node body does not wait for, and returns or panics. The
// goroutine calls it only after the workflow has finished. That event
// used to reach the node's yield after the scheduler's range loop over
// the node had ended, which panicked the goroutine ("range function
// continued iteration after whole loop exit") and with it the process.
// The call now fails instead: a node's emit always refuses an event
// once its body has returned; a RunNode started then is refused once
// no workflow can take it.
func TestDetachedEmitAfterBodyReturns(t *testing.T) {
	child := newFnNode("child", func(agent.Context) (any, error) { return "child out", nil })
	for _, tc := range []struct {
		name string
		// late runs on a goroutine of the body's, once the workflow has
		// finished.
		late    func(ctx agent.Context, emit func(*session.Event) error) error
		panics  bool
		wantErr error
	}{
		{
			name: "RunNode",
			late: func(ctx agent.Context, _ func(*session.Event) error) error {
				_, err := RunNode[any](ctx, child, nil)
				return err
			},
			wantErr: errDetachedUndeliverable,
		},
		{
			name: "RunNode, body panics",
			late: func(ctx agent.Context, _ func(*session.Event) error) error {
				_, err := RunNode[any](ctx, child, nil)
				return err
			},
			panics:  true,
			wantErr: errDetachedUndeliverable,
		},
		{
			name: "emit",
			late: func(_ agent.Context, emit func(*session.Event) error) error {
				return emit(&session.Event{Output: "late"})
			},
			wantErr: errEmitAfterReturn,
		},
		{
			name: "emit, body panics",
			late: func(_ agent.Context, emit func(*session.Event) error) error {
				return emit(&session.Event{Output: "late"})
			},
			panics: true,
			// The failed node's context is canceled.
			wantErr: context.Canceled,
		},
	} {
		for _, kind := range []string{"dynamic node", "emitting function node"} {
			if kind == "emitting function node" && strings.HasPrefix(tc.name, "RunNode") {
				continue // RunNode needs a dynamic node's context
			}
			t.Run(kind+", "+tc.name, func(t *testing.T) {
				release := make(chan struct{})
				lateErr := make(chan error, 1)
				body := func(ctx agent.Context, _ any, emit func(*session.Event) error) (string, error) {
					go func() {
						<-release
						lateErr <- tc.late(ctx, emit)
					}()
					if tc.panics {
						panic("body failed")
					}
					return "parent done", nil
				}
				parent := NewDynamicNode("parent", body, NodeConfig{})
				if kind == "emitting function node" {
					parent = NewEmittingFunctionNode("parent", body, NodeConfig{})
				}

				outputs, runErr := runDetached(t, parent)
				close(release)

				if (runErr != nil) != tc.panics {
					t.Errorf("workflow error = %v, want error: %t", runErr, tc.panics)
				}
				if err := <-lateErr; !errors.Is(err, tc.wantErr) {
					t.Errorf("late call error = %v, want %v", err, tc.wantErr)
				}
				var want []string
				if !tc.panics {
					want = []string{"parent done"}
				}
				if diff := cmp.Diff(want, outputs, cmpopts.EquateEmpty()); diff != "" {
					t.Errorf("workflow outputs mismatch (-want +got):\n%s", diff)
				}
			})
		}
	}
}

// TestDetachedChildEventsAreYielded has a detached child emit an event
// mid-run and then an output, while the workflow waits for it. Both
// are yielded, after the parent's output, and the child's RunNode
// succeeds.
func TestDetachedChildEventsAreYielded(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		result := make(chan error, 1)
		child := NewEmittingFunctionNode("child", func(ctx agent.Context, _ any, emit func(*session.Event) error) (string, error) {
			time.Sleep(50 * time.Millisecond)
			if err := emit(outputEvent("progress")(ctx)); err != nil {
				return "", err
			}
			return "child out", nil
		}, NodeConfig{})
		parent := NewDynamicNode("parent", func(ctx agent.Context, _ any, _ func(*session.Event) error) (string, error) {
			go func() {
				_, err := RunNode[any](ctx, child, nil)
				result <- err
			}()
			synctest.Wait() // the child is running, asleep
			return "parent done", nil
		}, NodeConfig{})

		w, err := New("wf", []Edge{{From: Start, To: parent}})
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		var outputs []any
		for ev, err := range w.Run(newSeededMockCtx(t)) {
			if err != nil {
				t.Fatalf("workflow error: %v", err)
			}
			if ev.Output == nil {
				continue
			}
			outputs = append(outputs, ev.Output)
			// The workflow stamps a detached child's events as it does
			// every event it yields.
			if _, ok := ev.CustomMetadata[workflowBasePathPrefix+"wf"]; !ok {
				t.Errorf("event with output %v has no workflow base path; metadata %v", ev.Output, ev.CustomMetadata)
			}
		}

		if diff := cmp.Diff([]any{"parent done", "progress", "child out"}, outputs); diff != "" {
			t.Errorf("workflow outputs in order mismatch (-want +got):\n%s", diff)
		}
		if err := <-result; err != nil {
			t.Errorf("detached RunNode error: %v", err)
		}
	})
}

// customNodeFunc adapts a function to a Node, for nodes whose events a
// test controls directly.
type customNodeFunc struct {
	BaseNode
	run func(ctx agent.Context, input any, yield func(*session.Event, error) bool)
}

func newCustomNode(name string, run func(ctx agent.Context, input any, yield func(*session.Event, error) bool)) Node {
	return &customNodeFunc{BaseNode: NewBaseNode(name, "", NodeConfig{}), run: run}
}

func (n *customNodeFunc) Run(ctx agent.Context, input any) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) { n.run(ctx, input, yield) }
}
