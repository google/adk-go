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
	"iter"
	"sync"
	"testing"
	"testing/synctest"
	"time"

	"github.com/google/go-cmp/cmp"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/session"
)

// trackedNode records the input of every activation, in start order,
// and the most activations that were ever in flight at once. Each
// activation first waits on the gate returned by gateFor (nil: no
// wait), then emits the output and routes returned by result.
type trackedNode struct {
	BaseNode
	gateFor func(call int) <-chan struct{}
	result  func(input any) (out any, routes []string, err error)

	mu       sync.Mutex
	inputs   []any
	inFlight int
	peak     int
}

func newTrackedNode(name string, cfg NodeConfig, result func(input any) (any, []string, error)) *trackedNode {
	return &trackedNode{BaseNode: NewBaseNode(name, "", cfg), result: result}
}

func (n *trackedNode) Run(ctx agent.Context, input any) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		n.mu.Lock()
		n.inputs = append(n.inputs, input)
		call := len(n.inputs)
		n.inFlight++
		n.peak = max(n.peak, n.inFlight)
		n.mu.Unlock()
		defer func() {
			n.mu.Lock()
			n.inFlight--
			n.mu.Unlock()
		}()

		if n.gateFor != nil {
			if gate := n.gateFor(call); gate != nil {
				<-gate
			}
		}
		out, routes, err := n.result(input)
		if err != nil {
			yield(nil, err)
			return
		}
		ev := session.NewEvent(ctx, ctx.InvocationID())
		ev.Output = out
		ev.Routes = routes
		yield(ev, nil)
	}
}

func (n *trackedNode) snapshot() (inputs []any, peak int) {
	n.mu.Lock()
	defer n.mu.Unlock()
	return append([]any(nil), n.inputs...), n.peak
}

// gateFirstCall returns a gateFor that holds only the first
// activation, until release is closed.
func gateFirstCall(release chan struct{}) func(int) <-chan struct{} {
	return func(call int) <-chan struct{} {
		if call == 1 {
			return release
		}
		return nil
	}
}

func fixedOutput(out any, routes ...string) func(any) (any, []string, error) {
	return func(any) (any, []string, error) { return out, routes, nil }
}

func echoInput(name string) func(any) (any, []string, error) {
	return func(in any) (any, []string, error) { return fmt.Sprintf("%v:%s", in, name), nil, nil }
}

// startWorkflow drains w in a goroutine of the synctest bubble and
// returns a function that waits for the run to end and returns the
// outputs it yielded, sorted, and its first error.
func startWorkflow(t *testing.T, w *Workflow) func() ([]string, error) {
	t.Helper()
	var (
		events []*session.Event
		runErr error
		done   = make(chan struct{})
	)
	go func() {
		defer close(done)
		for ev, err := range w.Run(newSeededMockCtx(t)) {
			if err != nil && runErr == nil {
				runErr = err
			}
			events = append(events, ev)
		}
	}()
	return func() ([]string, error) {
		<-done
		return outputsOf(events), runErr
	}
}

// TestScheduler_FanInWithRoute ports adk-python's
// test_fan_in_with_route (tests/unittests/workflow/test_workflow_routes.py):
// two parallel sources route to the same target, which runs once for
// each. The two activations of c are not ordered, so the test runs
// the graph repeatedly; before activations of one node were
// serialized, some runs hung or lost one of c's outputs.
func TestScheduler_FanInWithRoute(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		for i := range 50 {
			a := newTrackedNode("a", NodeConfig{}, fixedOutput("A", "route1"))
			b := newTrackedNode("b", NodeConfig{}, fixedOutput("B", "route1"))
			c := newTrackedNode("c", NodeConfig{}, echoInput("c"))
			w := mustNew(t, []Edge{
				{From: Start, To: a},
				{From: Start, To: b},
				{From: a, To: c, Route: StringRoute("route1")},
				{From: b, To: c, Route: StringRoute("route1")},
			})

			got, err := startWorkflow(t, w)()
			if err != nil {
				t.Fatalf("run %d: unexpected error: %v", i, err)
			}
			want := []string{"A", "A:c", "B", "B:c"}
			if diff := cmp.Diff(want, got); diff != "" {
				t.Fatalf("run %d: outputs mismatch (-want +got):\n%s", i, diff)
			}
		}
	})
}

// TestScheduler_SerializesActivationsOfOneNode holds c's first
// activation until both of its predecessors have completed, so the
// second activation arrives while the first is running. It must wait
// for the first to finish rather than run beside it.
func TestScheduler_SerializesActivationsOfOneNode(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		releaseC := make(chan struct{})
		a := newTrackedNode("a", NodeConfig{}, fixedOutput("A", "go"))
		b := newTrackedNode("b", NodeConfig{}, fixedOutput("B", "go"))
		c := newTrackedNode("c", NodeConfig{}, echoInput("c"))
		c.gateFor = gateFirstCall(releaseC)
		w := mustNew(t, []Edge{
			{From: Start, To: a},
			{From: Start, To: b},
			{From: a, To: c, Route: StringRoute("go")},
			{From: b, To: c, Route: StringRoute("go")},
		})

		wait := startWorkflow(t, w)
		synctest.Wait() // a and b are done; c's first activation is parked on releaseC
		close(releaseC)
		got, err := wait()
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if diff := cmp.Diff([]string{"A", "A:c", "B", "B:c"}, got); diff != "" {
			t.Errorf("outputs mismatch (-want +got):\n%s", diff)
		}
		inputs, peak := c.snapshot()
		if peak != 1 {
			t.Errorf("c had %d activations in flight at once, want 1", peak)
		}
		if len(inputs) != 2 {
			t.Errorf("c ran %d times (inputs %v), want 2", len(inputs), inputs)
		}
	})
}

// TestScheduler_DeferredActivationWaitsForRetry triggers c a second
// time around a failed first attempt. The retry belongs to the first
// activation, so it runs before the second, whether the second
// arrives while the retry waits or was already deferred when the
// first attempt failed.
func TestScheduler_DeferredActivationWaitsForRetry(t *testing.T) {
	for _, tc := range []struct {
		name string
		// deferBeforeFailure holds c's first attempt until b has
		// triggered c again; otherwise b triggers c only once the
		// retry is waiting.
		deferBeforeFailure bool
	}{
		{name: "triggered while the retry waits"},
		{name: "deferred before the attempt fails", deferBeforeFailure: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				releaseB, releaseC := make(chan struct{}), make(chan struct{})
				a := newTrackedNode("a", NodeConfig{}, fixedOutput("A", "go"))
				b := newTrackedNode("b", NodeConfig{}, fixedOutput("B", "go"))
				b.gateFor = gateFirstCall(releaseB)
				calls := 0
				c := newTrackedNode("c", NodeConfig{RetryConfig: &RetryConfig{
					MaxAttempts:   2,
					InitialDelay:  time.Second,
					MaxDelay:      time.Second,
					BackoffFactor: 1,
				}}, func(in any) (any, []string, error) {
					calls++
					if calls == 1 {
						return nil, nil, errors.New("transient")
					}
					return fmt.Sprintf("%v:c", in), nil, nil
				})
				if tc.deferBeforeFailure {
					c.gateFor = gateFirstCall(releaseC)
				}
				w := mustNew(t, []Edge{
					{From: Start, To: a},
					{From: Start, To: b},
					{From: a, To: c, Route: StringRoute("go")},
					{From: b, To: c, Route: StringRoute("go")},
				})

				wait := startWorkflow(t, w)
				synctest.Wait()
				close(releaseB) // b completes and triggers c again
				synctest.Wait()
				close(releaseC)
				if _, err := wait(); err != nil {
					t.Fatalf("unexpected error: %v", err)
				}

				inputs, peak := c.snapshot()
				if diff := cmp.Diff([]any{"A", "A", "B"}, inputs); diff != "" {
					t.Errorf("c's inputs in start order mismatch (-want +got):\n%s", diff)
				}
				if peak != 1 {
					t.Errorf("c had %d activations in flight at once, want 1", peak)
				}
			})
		})
	}
}

// TestScheduler_DeferredActivationWaitsForConcurrencySlot triggers c
// a second time while its first activation is still queued behind
// the concurrency cap. The queued activation runs first.
func TestScheduler_DeferredActivationWaitsForConcurrencySlot(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		a := newTrackedNode("a", NodeConfig{}, fixedOutput("A", "go"))
		d := newTrackedNode("d", NodeConfig{}, fixedOutput("D", "go"))
		c := newTrackedNode("c", NodeConfig{}, echoInput("c"))
		// With a cap of 1, a's completion starts d and queues c; d's
		// completion then triggers c again while c is still queued.
		w, err := New("", []Edge{
			{From: Start, To: a},
			{From: a, To: d, Route: StringRoute("go")},
			{From: a, To: c, Route: StringRoute("go")},
			{From: d, To: c, Route: StringRoute("go")},
		}, WithMaxConcurrency(1))
		if err != nil {
			t.Fatalf("New: %v", err)
		}

		if _, err := startWorkflow(t, w)(); err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		inputs, _ := c.snapshot()
		if diff := cmp.Diff([]any{"A", "D"}, inputs); diff != "" {
			t.Errorf("c's inputs in start order mismatch (-want +got):\n%s", diff)
		}
	})
}

// TestScheduler_DeferredActivationsKeepArrivalOrder makes c route
// back to itself while an earlier activation of c is already
// deferred. As in adk-python's per-node trigger buffer, the
// self-loop's activation queues behind the earlier one.
func TestScheduler_DeferredActivationsKeepArrivalOrder(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		releaseB, releaseC := make(chan struct{}), make(chan struct{})
		a := newTrackedNode("a", NodeConfig{}, fixedOutput("A", "go"))
		b := newTrackedNode("b", NodeConfig{}, fixedOutput("B", "go"))
		b.gateFor = gateFirstCall(releaseB)
		c := newTrackedNode("c", NodeConfig{}, func(in any) (any, []string, error) {
			out := fmt.Sprintf("%v:c", in)
			if in == "A" {
				return out, []string{"loop"}, nil
			}
			return out, nil, nil
		})
		c.gateFor = gateFirstCall(releaseC)
		w := mustNew(t, []Edge{
			{From: Start, To: a},
			{From: Start, To: b},
			{From: a, To: c, Route: StringRoute("go")},
			{From: b, To: c, Route: StringRoute("go")},
			{From: c, To: c, Route: StringRoute("loop")},
		})

		wait := startWorkflow(t, w)
		synctest.Wait() // c("A") is parked on releaseC; b on releaseB
		close(releaseB) // c("B") is deferred behind c("A")
		synctest.Wait()
		close(releaseC) // c("A") completes and loops back to c
		if _, err := wait(); err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		inputs, _ := c.snapshot()
		if diff := cmp.Diff([]any{"A", "B", "A:c"}, inputs); diff != "" {
			t.Errorf("c's inputs in start order mismatch (-want +got):\n%s", diff)
		}
	})
}
