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
	"fmt"
	"iter"
	"sort"
	"sync/atomic"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/session"
)

// TestScheduler_LinearChain verifies that a linear graph
// Start → A → B → C runs to completion in the expected order and
// that each node's lifecycle ends at NodeCompleted.
func TestScheduler_LinearChain(t *testing.T) {
	mockCtx := newSeededMockCtx(t)

	a := newRecordingNode("A")
	b := newRecordingNode("B")
	c := newRecordingNode("C")
	a.release()
	b.release()
	c.release()

	w := mustNew(t, Chain(Start, a, b, c))

	gotEvents := drain(t, w.Run(mockCtx))

	// Each recording node yields exactly one event with output =
	// "<input>:<name>". The chain accumulates: A sees "seed", B sees
	// "seed:A", C sees "seed:A:B".
	wantOutputs := []string{"seed:A", "seed:A:B", "seed:A:B:C"}
	gotOutputs := outputsOf(gotEvents)
	if diff := cmp.Diff(wantOutputs, gotOutputs); diff != "" {
		t.Errorf("outputs mismatch (-want +got):\n%s", diff)
	}
}

// TestScheduler_FanOutConcurrency verifies that three nodes
// downstream of START are mid-Run simultaneously, not serialised by
// the legacy BFS. Each node blocks on its release channel until the
// test signals.
func TestScheduler_FanOutConcurrency(t *testing.T) {
	const fanOut = 3

	mockCtx := newSeededMockCtx(t)

	var concurrent atomic.Int32
	var peakConcurrent atomic.Int32

	// Make N nodes that each bump a "currently running" counter on
	// entry and decrement on exit. Track the peak. They block on a
	// shared barrier so they can all be observed mid-flight together.
	barrier := make(chan struct{})
	nodes := make([]Node, fanOut)
	for i := range fanOut {
		name := fmt.Sprintf("N%d", i)
		nodes[i] = newBlockingNode(name, func() {
			nowConcurrent := concurrent.Add(1)
			for {
				peak := peakConcurrent.Load()
				if nowConcurrent <= peak || peakConcurrent.CompareAndSwap(peak, nowConcurrent) {
					break
				}
			}
			<-barrier
			concurrent.Add(-1)
		})
	}

	edges := make([]Edge, 0, fanOut)
	for _, n := range nodes {
		edges = append(edges, Edge{From: Start, To: n})
	}
	w := mustNew(t, edges)

	// Release the barrier once we observe peak == fanOut, or fail
	// after a generous timeout.
	done := make(chan struct{})
	go func() {
		defer close(done)
		drain(t, w.Run(mockCtx))
	}()

	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if peakConcurrent.Load() == fanOut {
			break
		}
		time.Sleep(time.Millisecond)
	}

	close(barrier)
	<-done

	if got := peakConcurrent.Load(); got != fanOut {
		t.Errorf("peak concurrent nodes = %d, want %d (fan-out children should run in parallel)", got, fanOut)
	}
}

// TestScheduler_FailedSiblingsCancelled verifies that when one
// node returns an error, the consumer cancels every sibling, the
// final yielded error is the failing node's own error, and the
// lifecycle map ends up with the right mix of statuses.
func TestScheduler_FailedSiblingsCancelled(t *testing.T) {
	mockCtx := newSeededMockCtx(t)

	failErr := errors.New("B failed intentionally")

	// A and C block until their context is cancelled. B errors
	// immediately. After B's failure, A and C should observe ctx.Done.
	a := newCancelObservingNode("A")
	b := newErroringNode("B", failErr)
	c := newCancelObservingNode("C")

	edges := []Edge{
		{From: Start, To: a},
		{From: Start, To: b},
		{From: Start, To: c},
	}
	w := mustNew(t, edges)

	gotErr := drainErr(t, w.Run(mockCtx))
	if !errors.Is(gotErr, failErr) {
		t.Errorf("Run error = %v, want it to wrap %v", gotErr, failErr)
	}
	if got := a.cancelObserved.Load(); !got {
		t.Errorf("node A: ctx.Done() not observed (sibling cancellation broken)")
	}
	if got := c.cancelObserved.Load(); !got {
		t.Errorf("node C: ctx.Done() not observed (sibling cancellation broken)")
	}
}

// TestScheduler_CallerBreakStopsScheduling verifies that when the
// caller breaks out of the event range loop, the scheduler stops
// dispatching new successor nodes — not just stops yielding events.
// Without this guarantee, an early caller break would leave the rest
// of the workflow running silently in the background.
func TestScheduler_CallerBreakStopsScheduling(t *testing.T) {
	mockCtx := newSeededMockCtx(t)

	var bRan, cRan atomic.Bool

	a := NewFunctionNode("A", func(ctx agent.InvocationContext, input any) (string, error) {
		return "a-out", nil
	}, defaultNodeConfig)
	b := NewFunctionNode("B", func(ctx agent.InvocationContext, input any) (string, error) {
		bRan.Store(true)
		return "b-out", nil
	}, defaultNodeConfig)
	c := NewFunctionNode("C", func(ctx agent.InvocationContext, input any) (string, error) {
		cRan.Store(true)
		return "c-out", nil
	}, defaultNodeConfig)

	w := mustNew(t, []Edge{
		{From: Start, To: a},
		{From: a, To: b},
		{From: b, To: c},
	})

	// Caller breaks immediately after the first event.
	for range w.Run(mockCtx) {
		break
	}

	if bRan.Load() {
		t.Error("node B ran after caller break; draining should stop further scheduling")
	}
	if cRan.Load() {
		t.Error("node C ran after caller break; draining should stop further scheduling")
	}
}

// TestScheduler_NodeTimeout verifies that a node with
// Config().Timeout set is killed after the timeout and the resulting
// completion is treated as a failure (deadline exceeded). Sibling
// nodes without a timeout are not affected.
func TestScheduler_NodeTimeout(t *testing.T) {
	mockCtx := newSeededMockCtx(t)

	slow := newCancelObservingNode("slow")
	slow.cfg = NodeConfig{Timeout: 50 * time.Millisecond}

	w := mustNew(t, []Edge{{From: Start, To: slow}})

	gotErr := drainErr(t, w.Run(mockCtx))
	if !errors.Is(gotErr, context.DeadlineExceeded) {
		t.Errorf("Run error = %v, want context.DeadlineExceeded", gotErr)
	}
	if !slow.cancelObserved.Load() {
		t.Error("slow node: ctx.Done() not observed (timeout cancellation broken)")
	}
}

// TestScheduler_MultipleOutputsFailNode verifies that a node which
// yields more than one event carrying StateDelta["output"] fails
// the workflow with ErrMultipleOutputs (matching adk-python's
// "single output per node" contract). The first output value is
// preserved; subsequent ones trip the accumulator error.
func TestScheduler_MultipleOutputsFailNode(t *testing.T) {
	mockCtx := newSeededMockCtx(t)

	bad := newMultiOutputNode("bad", []string{"first", "second"})
	w := mustNew(t, []Edge{{From: Start, To: bad}})

	gotErr := drainErr(t, w.Run(mockCtx))
	if !errors.Is(gotErr, ErrMultipleOutputs) {
		t.Errorf("Run error = %v, want it to wrap ErrMultipleOutputs", gotErr)
	}
}

// TestScheduler_ProgressEventsThenSingleOutputSucceed verifies
// that events with no StateDelta["output"] (progress / status
// events) do not consume the single-output budget. A node may
// yield many such events followed by exactly one output event
// without violating the contract.
func TestScheduler_ProgressEventsThenSingleOutputSucceed(t *testing.T) {
	mockCtx := newSeededMockCtx(t)

	n := newProgressThenOutputNode("n", 3 /*progress events*/, "final")
	w := mustNew(t, []Edge{{From: Start, To: n}})

	events := drain(t, w.Run(mockCtx))

	// Expect 3 progress events + 1 output event = 4 events total,
	// and the last one carries the output value.
	if got, want := len(events), 4; got != want {
		t.Fatalf("event count = %d, want %d", got, want)
	}
	last := events[len(events)-1]
	if last.Actions.StateDelta == nil {
		t.Fatal("last event has nil StateDelta")
	}
	if got, want := fmt.Sprint(last.Actions.StateDelta["output"]), "final"; got != want {
		t.Errorf("last event output = %q, want %q", got, want)
	}
}

// --- test fixtures: helper nodes and drain helpers below this line ---

// drain consumes all events from an iter.Seq2 and returns the
// successful events. The first error fails the test.
func drain(t *testing.T, seq iter.Seq2[*session.Event, error]) []*session.Event {
	t.Helper()
	var out []*session.Event
	for ev, err := range seq {
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		out = append(out, ev)
	}
	return out
}

// drainErr consumes all events and returns the first error. Fails
// the test if no error was produced.
func drainErr(t *testing.T, seq iter.Seq2[*session.Event, error]) error {
	t.Helper()
	var firstErr error
	for _, err := range seq {
		if err != nil && firstErr == nil {
			firstErr = err
		}
	}
	if firstErr == nil {
		t.Fatal("expected an error from Run, got none")
	}
	return firstErr
}

// outputsOf extracts the StateDelta["output"] string from each event,
// sorted to make assertions order-independent across concurrent runs.
func outputsOf(events []*session.Event) []string {
	out := make([]string, 0, len(events))
	for _, ev := range events {
		if ev == nil || ev.Actions.StateDelta == nil {
			continue
		}
		if v, ok := ev.Actions.StateDelta["output"]; ok {
			out = append(out, fmt.Sprint(v))
		}
	}
	sort.Strings(out)
	return out
}

// recordingNode emits one event with output = "<input>:<name>".
// Used in chains to verify per-step input propagation.
type recordingNode struct {
	BaseNode
	released chan struct{}
}

func newRecordingNode(name string) *recordingNode {
	return &recordingNode{
		BaseNode: NewBaseNode(name, "", NodeConfig{}),
		released: make(chan struct{}),
	}
}

func (n *recordingNode) release() { close(n.released) }

func (n *recordingNode) Run(ctx agent.InvocationContext, input any) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		<-n.released
		ev := session.NewEvent(ctx.InvocationID())
		out := fmt.Sprintf("%v:%s", input, n.Name())
		ev.Actions.StateDelta["output"] = out
		yield(ev, nil)
	}
}

// blockingNode runs the supplied work function (which typically
// touches an external counter and waits on a barrier). Yields one
// empty event after the work returns so the scheduler can record
// completion.
type blockingNode struct {
	BaseNode
	work func()
}

func newBlockingNode(name string, work func()) *blockingNode {
	return &blockingNode{
		BaseNode: NewBaseNode(name, "", NodeConfig{}),
		work:     work,
	}
}

func (n *blockingNode) Run(ctx agent.InvocationContext, _ any) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		n.work()
		ev := session.NewEvent(ctx.InvocationID())
		ev.Actions.StateDelta["output"] = n.Name()
		yield(ev, nil)
	}
}

// erroringNode returns the supplied error immediately.
type erroringNode struct {
	BaseNode
	err error
}

func newErroringNode(name string, err error) *erroringNode {
	return &erroringNode{
		BaseNode: NewBaseNode(name, "", NodeConfig{}),
		err:      err,
	}
}

func (n *erroringNode) Run(_ agent.InvocationContext, _ any) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		yield(nil, n.err)
	}
}

// cancelObservingNode blocks until its context is cancelled, then
// records that fact via cancelObserved. Used to verify sibling
// cancellation and timeout behaviour.
type cancelObservingNode struct {
	BaseNode
	cfg            NodeConfig
	cancelObserved atomic.Bool
}

func newCancelObservingNode(name string) *cancelObservingNode {
	return &cancelObservingNode{BaseNode: NewBaseNode(name, "", NodeConfig{})}
}

// Config returns n.cfg, which tests may mutate after construction.
func (n *cancelObservingNode) Config() NodeConfig { return n.cfg }

func (n *cancelObservingNode) Run(ctx agent.InvocationContext, _ any) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		<-ctx.Done()
		n.cancelObserved.Store(true)
	}
}

// multiOutputNode yields one event per supplied output value, each
// carrying StateDelta["output"]. Used to exercise the
// single-output-per-node constraint.
type multiOutputNode struct {
	BaseNode
	outputs []string
}

func newMultiOutputNode(name string, outputs []string) *multiOutputNode {
	return &multiOutputNode{
		BaseNode: NewBaseNode(name, "", NodeConfig{}),
		outputs:  outputs,
	}
}

func (n *multiOutputNode) Run(ctx agent.InvocationContext, _ any) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		for _, out := range n.outputs {
			ev := session.NewEvent(ctx.InvocationID())
			ev.Actions.StateDelta["output"] = out
			if !yield(ev, nil) {
				return
			}
		}
	}
}

// progressThenOutputNode yields N events with no output (progress
// events) followed by exactly one output event. Used to verify
// that progress events do not consume the single-output budget.
type progressThenOutputNode struct {
	BaseNode
	progressCount int
	finalOutput   string
}

func newProgressThenOutputNode(name string, progressCount int, finalOutput string) *progressThenOutputNode {
	return &progressThenOutputNode{
		BaseNode:      NewBaseNode(name, "", NodeConfig{}),
		progressCount: progressCount,
		finalOutput:   finalOutput,
	}
}

func (n *progressThenOutputNode) Run(ctx agent.InvocationContext, _ any) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		for range n.progressCount {
			ev := session.NewEvent(ctx.InvocationID())
			// progress event: no StateDelta["output"]
			if !yield(ev, nil) {
				return
			}
		}
		final := session.NewEvent(ctx.InvocationID())
		final.Actions.StateDelta["output"] = n.finalOutput
		yield(final, nil)
	}
}
