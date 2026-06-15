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

// FINDING L1 — the node retry path is not coordinated with cancelAll /
// draining.
//
// CONTEXT (the originally-hypothesized symptom — a goroutine leak):
// scheduleRetry (scheduler.go) spawns an UNTRACKED goroutine (not in the
// scheduler WaitGroup) that blocks on `eventQueue <-` or `parentCtx.Done()`.
// cancelAll() does not cancel parentCtx, so a retry sender that fired into a
// narrow window with a full event queue could block forever. That leak is
// extremely timing-sensitive: the consumer keeps draining eventQueue until
// runsByName hits zero, which frees slots for any blocked sender, and in
// tests parentCtx (t.Context()) is cancelled on cleanup, releasing it. I
// could not make the leak fail deterministically.
//
// THE REAL, DETERMINISTIC BUG (same root cause): cancelAll() sets
// `s.retryTimers = nil` while the retry SCHEDULING path remains live.
// handleCompletion calls scheduleRetry for any retryable node that completes
// with a retryable error, and it does so BEFORE the scheduleSuccessors gate
// (scheduler.go) — i.e. even while draining. scheduleRetry then executes
// `s.retryTimers[n.Name()] = timer` on the nil map, panicking with
// "assignment to entry in nil map". The panic happens on the consumer
// goroutine (Workflow.Run's run loop), taking down the whole run.
//
// EXPECTED: A node that fails (retryably) after the run has begun draining
// (caller broke the range / a sibling failed / the context was cancelled)
// must be handled cleanly — no panic, no leaked goroutine. cancelAll and the
// retry path must agree on lifecycle.
//
// HOW THIS TEST DEMONSTRATES IT: An emitter node yields one event so the
// caller can break the range loop, which makes the scheduler call cancelAll
// (nilling retryTimers). A second node with a RetryConfig sleeps briefly and
// then fails AFTER the break, so its completion is processed during draining
// and drives handleCompletion -> scheduleRetry -> nil-map write. The panic
// propagates out of Workflow.Run on the consumer goroutine; the test
// recover()s it and reports via t.Errorf. FAILS deterministically (no -race
// needed); run with -count to confirm.

package workflow

import (
	"errors"
	"iter"
	"testing"
	"time"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/session"
)

// vbugL1EmitNode yields exactly one output event so the consumer has
// something to forward to the caller (whose break triggers cancelAll).
type vbugL1EmitNode struct{ BaseNode }

func (n *vbugL1EmitNode) Run(ctx agent.Context, _ any) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		ev := session.NewEvent(ctx.InvocationID())
		ev.Output = "emit"
		yield(ev, nil)
	}
}

// vbugL1DelayedFailNode fails with a retryable error after a delay, so its
// completion is processed AFTER the caller's break has triggered cancelAll
// (which nils retryTimers). The sleep ignores ctx cancellation and the
// yielded error is a plain (non-context) error so handleCompletion takes the
// retry branch rather than the cancelled branch.
type vbugL1DelayedFailNode struct{ BaseNode }

func (n *vbugL1DelayedFailNode) Run(_ agent.Context, _ any) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		time.Sleep(30 * time.Millisecond)
		yield(nil, errors.New("boom"))
	}
}

func TestVbugL1_RetryAfterCancelAllPanicsOnNilRetryTimers(t *testing.T) {
	var recovered any
	func() {
		defer func() { recovered = recover() }()

		rc := DefaultRetryConfig()
		rc.MaxAttempts = 3 // ShouldRetry true on the first failure
		rc.InitialDelay = time.Millisecond

		emit := &vbugL1EmitNode{BaseNode: NewBaseNode("emit", "", NodeConfig{})}
		retryNode := &vbugL1DelayedFailNode{
			BaseNode: NewBaseNode("retry_node", "", NodeConfig{RetryConfig: rc}),
		}

		w, err := New("", []Edge{
			{From: Start, To: emit},
			{From: Start, To: retryNode},
		})
		if err != nil {
			t.Fatalf("New: %v", err)
		}

		// Break on the first event: this makes the run loop call cancelAll
		// (consumerGone path), nilling retryTimers. The run loop then keeps
		// draining until runsByName is empty, so it is still executing on
		// this goroutine when retryNode fails ~30ms later and drives
		// scheduleRetry into the nil map.
		for _, runErr := range w.Run(newMockCtx(t)) {
			_ = runErr
			break
		}
	}()

	if recovered != nil {
		t.Errorf("L1: a retryable node completing during cancelAll/draining panicked "+
			"in scheduleRetry (nil retryTimers): %v. cancelAll() nils retryTimers but "+
			"handleCompletion still calls scheduleRetry for a node that fails while "+
			"draining; the retry path is not coordinated with cancellation.", recovered)
	}
}
