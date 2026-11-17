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
	"slices"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/session"
)

// faninBudget bounds one workflow run in the fan-in tests. A run that
// does not end within it fails the test instead of hanging the binary.
const faninBudget = 5 * time.Second

// faninGrace is how long an activation waits for a concurrent second
// activation of its own node. A scheduler that serializes activations
// never starts one, so on correct code the wait simply times out.
const faninGrace = 200 * time.Millisecond

// faninNode is a static node whose body each test supplies.
type faninNode struct {
	BaseNode
	run func(ctx agent.Context, input any, yield func(*session.Event, error) bool)
}

func newFaninNode(name string, cfg NodeConfig, run func(ctx agent.Context, input any, yield func(*session.Event, error) bool)) *faninNode {
	return &faninNode{BaseNode: NewBaseNode(name, "", cfg), run: run}
}

func (n *faninNode) Run(ctx agent.Context, input any) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		n.run(ctx, input, yield)
	}
}

// faninEmit yields one event with output out (none when nil) and the
// given routes.
func faninEmit(ctx agent.Context, yield func(*session.Event, error) bool, out any, routes ...string) bool {
	ev := session.NewEvent(ctx, ctx.InvocationID())
	ev.Output = out
	ev.Routes = routes
	return yield(ev, nil)
}

// faninEmitter returns a node body that yields one event with output out
// and the given routes.
func faninEmitter(out any, routes ...string) func(agent.Context, any, func(*session.Event, error) bool) {
	return func(ctx agent.Context, _ any, yield func(*session.Event, error) bool) {
		faninEmit(ctx, yield, out, routes...)
	}
}

// faninWait blocks until ch is closed or d elapses, and reports whether
// ch was closed.
func faninWait(ch <-chan struct{}, d time.Duration) bool {
	select {
	case <-ch:
		return true
	case <-time.After(d):
		return false
	}
}

// faninCtx returns a seeded invocation context that is canceled after d,
// which releases the node goroutines of a run that never finishes.
func faninCtx(t *testing.T, d time.Duration) (*MockInvocationContext, context.CancelFunc) {
	t.Helper()
	ctx := newSeededMockCtx(t)
	c, cancel := context.WithTimeout(ctx.Context, d)
	t.Cleanup(cancel)
	ctx.Context = c
	return ctx, cancel
}

// faninResult is what one drained run yielded.
type faninResult struct {
	events []*session.Event
	err    error // the first error the run yielded
}

// faninDrain consumes seq on its own goroutine and reports false when the
// run has not ended after limit. onEvent, when non-nil, sees each event as
// it is yielded. The goroutine never touches a testing.T, so a run that
// outlives its test cannot panic the binary.
func faninDrain(seq iter.Seq2[*session.Event, error], limit time.Duration, onEvent func(*session.Event)) (faninResult, bool) {
	done := make(chan faninResult, 1)
	go func() {
		var res faninResult
		for ev, err := range seq {
			if err != nil && res.err == nil {
				res.err = err
			}
			if ev != nil {
				res.events = append(res.events, ev)
				if onEvent != nil {
					onEvent(ev)
				}
			}
		}
		done <- res
	}()
	select {
	case res := <-done:
		return res, true
	case <-time.After(limit):
		return faninResult{}, false
	}
}

// faninRun runs seq, whose context expires after faninBudget, to its end.
// It fails the test if the run does not end, or ends only because its
// context expired: no run in these tests should hang until its deadline.
func faninRun(t *testing.T, seq iter.Seq2[*session.Event, error]) faninResult {
	t.Helper()
	return faninRunObserved(t, seq, nil)
}

// faninRunObserved is faninRun with onEvent called for each event as the
// run yields it.
func faninRunObserved(t *testing.T, seq iter.Seq2[*session.Event, error], onEvent func(*session.Event)) faninResult {
	t.Helper()
	start := time.Now()
	res, ok := faninDrain(seq, faninBudget+time.Second, onEvent)
	if !ok {
		t.Fatalf("run did not end within %v", faninBudget)
	}
	if elapsed := time.Since(start); elapsed >= faninBudget {
		t.Errorf("run ended only after its %v deadline (took %v)", faninBudget, elapsed)
	}
	return res
}

// faninObs is one observation of a node activation.
type faninObs struct {
	Path   string
	Branch string
	Value  any
}

// faninLog collects observations from node goroutines.
type faninLog struct {
	mu    sync.Mutex
	items []faninObs
}

func (l *faninLog) add(o faninObs) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.items = append(l.items, o)
}

func (l *faninLog) get() []faninObs {
	l.mu.Lock()
	defer l.mu.Unlock()
	return slices.Clone(l.items)
}

// faninOutputs returns path, branch and output of every event of node
// name that carries an output, in stream order.
func faninOutputs(events []*session.Event, name string) []faninObs {
	var out []faninObs
	for _, ev := range events {
		if ev.Output == nil || ev.NodeInfo.Name() != name {
			continue
		}
		out = append(out, faninObs{Path: ev.NodeInfo.Path, Branch: ev.Branch, Value: ev.Output})
	}
	return out
}

// faninActivations returns path and branch of node name's activations,
// one per distinct event path, in order of first appearance.
func faninActivations(events []*session.Event, name string) []faninObs {
	var out []faninObs
	seen := map[string]bool{}
	for _, ev := range events {
		if ev.NodeInfo.Name() != name || seen[ev.NodeInfo.Path] {
			continue
		}
		seen[ev.NodeInfo.Path] = true
		out = append(out, faninObs{Path: ev.NodeInfo.Path, Branch: ev.Branch})
	}
	return out
}

// TestScheduler_RoutedFanIn_SerializesAndPropagatesInputs covers B2 and
// B3: routed fan-in to c runs c, and its successor z, once per
// predecessor completion, one activation at a time, each with that
// predecessor's output.
//
// Fails if scheduleReady starts a trigger whose node is busy (a second
// live activation of c), or if a completion does not start the node's
// next buffered trigger.
func TestScheduler_RoutedFanIn_SerializesAndPropagatesInputs(t *testing.T) {
	var inFlight, peak, cCalls atomic.Int32
	cStarted, secondC := make(chan struct{}), make(chan struct{})
	a := newFaninNode("a", NodeConfig{}, faninEmitter("from-A", "route1"))
	b := newFaninNode("b", NodeConfig{}, func(ctx agent.Context, _ any, yield func(*session.Event, error) bool) {
		faninWait(cStarted, faninBudget)
		faninEmit(ctx, yield, "from-B", "route1")
	})
	c := newFaninNode("c", NodeConfig{}, func(ctx agent.Context, in any, yield func(*session.Event, error) bool) {
		defer bumpPeak(&inFlight, &peak)()
		switch cCalls.Add(1) {
		case 1:
			close(cStarted)
			faninWait(secondC, faninGrace)
		case 2:
			close(secondC)
		}
		faninEmit(ctx, yield, fmt.Sprintf("c:%v", in))
	})
	z := newFaninNode("z", NodeConfig{}, func(ctx agent.Context, in any, yield func(*session.Event, error) bool) {
		faninEmit(ctx, yield, fmt.Sprintf("z:%v", in))
	})
	w, err := New("wf", []Edge{
		{From: Start, To: a},
		{From: Start, To: b},
		{From: a, To: c, Route: StringRoute("route1")},
		{From: b, To: c, Route: StringRoute("route1")},
		{From: c, To: z},
	})
	if err != nil {
		t.Fatal(err)
	}
	ctx, _ := faninCtx(t, faninBudget)

	res := faninRun(t, w.Run(ctx))

	if got := peak.Load(); got != 1 {
		t.Errorf("peak in-flight activations of c = %d, want 1", got)
	}
	if res.err != nil {
		t.Errorf("Run error = %v, want nil", res.err)
	}
	wantC := []faninObs{
		{Path: "wf@1/c@1", Branch: "a@1", Value: "c:from-A"},
		{Path: "wf@1/c@2", Branch: "b@1", Value: "c:from-B"},
	}
	if diff := cmp.Diff(wantC, faninOutputs(res.events, "c")); diff != "" {
		t.Errorf("c outputs mismatch (-want +got):\n%s", diff)
	}
	wantZ := []faninObs{
		{Path: "wf@1/z@1", Branch: "a@1", Value: "z:c:from-A"},
		{Path: "wf@1/z@2", Branch: "b@1", Value: "z:c:from-B"},
	}
	if diff := cmp.Diff(wantZ, faninOutputs(res.events, "z")); diff != "" {
		t.Errorf("z outputs mismatch (-want +got):\n%s", diff)
	}
}

// TestScheduler_CancellationAndError_DiscardQueuedTriggers covers B3 and
// the error handling plan: when the running activation of c fails, or the
// invocation is canceled, the trigger buffered behind it never starts.
//
// Fails if cancelAll does not clear the trigger buffer, or if a
// completion starts buffered triggers while the run is draining.
func TestScheduler_CancellationAndError_DiscardQueuedTriggers(t *testing.T) {
	errBoom := errors.New("boom")
	for _, tc := range []struct {
		name    string
		cancel  bool
		wantErr error
	}{
		{name: "activation_error", wantErr: errBoom},
		{name: "context_cancel", cancel: true, wantErr: context.Canceled},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx, cancel := faninCtx(t, faninBudget)
			var cCalls atomic.Int32
			secondC := make(chan struct{})
			a := newFaninNode("a", NodeConfig{}, faninEmitter("from-A", "r"))
			b := newFaninNode("b", NodeConfig{}, faninEmitter("from-B", "r"))
			c := newFaninNode("c", NodeConfig{}, func(cctx agent.Context, in any, yield func(*session.Event, error) bool) {
				if cCalls.Add(1) > 1 {
					close(secondC)
					faninEmit(cctx, yield, fmt.Sprintf("c:%v", in))
					return
				}
				// Both predecessors complete during this wait, so the
				// second trigger of c exists before this activation ends.
				faninWait(secondC, faninGrace)
				if tc.cancel {
					cancel()
					<-cctx.Done()
					return
				}
				yield(nil, errBoom)
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

			res := faninRun(t, w.Run(ctx))

			if got := cCalls.Load(); got != 1 {
				t.Errorf("activations of c = %d, want 1", got)
			}
			if !errors.Is(res.err, tc.wantErr) {
				t.Errorf("Run error = %v, want %v", res.err, tc.wantErr)
			}
		})
	}
}

// TestScheduler_RoutedFanInOut_NoHangOrLostOutput is the regression test
// for u1101: routed fan-in and fan-out to two terminal nodes started a
// second live activation of a node and kept its bookkeeping by name,
// which hung the run, lost an output or reported ErrMultipleOutputs.
// Every iteration must end with ErrMultipleTerminalOutputs and all four
// terminal outputs, each under its own run id.
//
// Fails if scheduleReady starts a trigger whose node is busy, so that two
// activations of c or d run at once.
func TestScheduler_RoutedFanInOut_NoHangOrLostOutput(t *testing.T) {
	a := newFaninNode("a", NodeConfig{}, faninEmitter("A", "route1"))
	b := newFaninNode("b", NodeConfig{}, faninEmitter("B", "route1"))
	c := newFaninNode("c", NodeConfig{}, faninEmitter("C"))
	d := newFaninNode("d", NodeConfig{}, faninEmitter("D"))
	w, err := New("wf", []Edge{
		{From: Start, To: a},
		{From: Start, To: b},
		{From: a, To: c, Route: StringRoute("route1")},
		{From: a, To: d, Route: StringRoute("route1")},
		{From: b, To: c, Route: StringRoute("route1")},
		{From: b, To: d, Route: StringRoute("route1")},
	})
	if err != nil {
		t.Fatal(err)
	}
	want := []string{"wf@1/c@1", "wf@1/c@2", "wf@1/d@1", "wf@1/d@2"}
	for i := range 250 {
		ctx, cancel := faninCtx(t, 2*time.Second)
		res, ok := faninDrain(w.Run(ctx), 3*time.Second, nil)
		cancel()
		if !ok {
			t.Fatalf("iteration %d: run did not end within its 2 s deadline", i)
		}
		if !errors.Is(res.err, ErrMultipleTerminalOutputs) {
			t.Fatalf("iteration %d: Run error = %v, want ErrMultipleTerminalOutputs", i, res.err)
		}
		var got []string
		for _, ev := range res.events {
			if ev.Output != nil && (ev.NodeInfo.Name() == "c" || ev.NodeInfo.Name() == "d") {
				got = append(got, ev.NodeInfo.Path)
			}
		}
		slices.Sort(got)
		if diff := cmp.Diff(want, got); diff != "" {
			t.Fatalf("iteration %d: terminal output paths mismatch (-want +got):\n%s", i, diff)
		}
	}
}

// TestScheduler_TriggerDuringRetryBackoff_WaitsForRetry covers a trigger
// that arrives while its node waits out a retry backoff: it starts only
// after the retry succeeded, as the node's next activation.
//
// Fails if busy ignores retryTimers, so the trigger starts a second
// activation during the backoff and the timer then starts a third.
func TestScheduler_TriggerDuringRetryBackoff_WaitsForRetry(t *testing.T) {
	var inFlight, peak, calls atomic.Int32
	var log faninLog
	cFailed := make(chan struct{})
	cfg := NodeConfig{RetryConfig: &RetryConfig{
		MaxAttempts:   2,
		InitialDelay:  50 * time.Millisecond,
		BackoffFactor: 1,
		ShouldRetry:   func(error) bool { return true },
	}}
	a := newFaninNode("a", NodeConfig{}, faninEmitter("from-A", "r"))
	b := newFaninNode("b", NodeConfig{}, func(ctx agent.Context, _ any, yield func(*session.Event, error) bool) {
		faninWait(cFailed, faninBudget)
		time.Sleep(10 * time.Millisecond) // inside c's 50 ms backoff
		faninEmit(ctx, yield, "from-B", "r")
	})
	c := newFaninNode("c", cfg, func(ctx agent.Context, in any, yield func(*session.Event, error) bool) {
		defer bumpPeak(&inFlight, &peak)()
		log.add(faninObs{Path: ctx.Path(), Value: in})
		if calls.Add(1) == 1 {
			close(cFailed)
			yield(nil, errors.New("transient"))
			return
		}
		faninEmit(ctx, yield, fmt.Sprintf("c:%v", in))
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
	ctx, _ := faninCtx(t, faninBudget)

	res := faninRun(t, w.Run(ctx))

	if res.err != nil {
		t.Fatalf("Run error = %v, want nil", res.err)
	}
	want := []faninObs{
		{Path: "wf@1/c@1", Value: "from-A"},
		{Path: "wf@1/c@1", Value: "from-A"},
		{Path: "wf@1/c@2", Value: "from-B"},
	}
	if diff := cmp.Diff(want, log.get()); diff != "" {
		t.Errorf("attempts of c mismatch (-want +got):\n%s", diff)
	}
	if got := peak.Load(); got != 1 {
		t.Errorf("peak in-flight attempts of c = %d, want 1", got)
	}
}
