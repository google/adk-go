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

// FINDING M3 — NodeConfig.WaitForOutput is a silent no-op.
//
// Bug: WaitForOutput is documented to keep a node in NodeWaiting (not
// finalized, successors not scheduled) until it actually yields an
// output-bearing event, instead of moving it to NodeCompleted on first
// clean return. But the scheduler never reads the field: handleCompletion
// unconditionally takes the NodeCompleted happy path on a clean return
// (when there is no error and no open interrupt), so WaitForOutput has no
// effect whatsoever.
//
// Expected correct behavior: a node configured with WaitForOutput=true
// that returns WITHOUT emitting an output-bearing event must remain in
// NodeWaiting; its successors must NOT be scheduled.
//
// Observability note: Workflow.Run does not expose the internal RunState
// to callers (the scheduler and its node statuses are created and
// discarded inside Run), so this test observes the contract indirectly,
// through a successor: with WaitForOutput respected, the successor M must
// not run. The bug schedules M anyway.
//
// This test currently FAILS, demonstrating the bug: N is treated as
// completed and its successor M runs even though N never produced output.

package workflow

import (
	"iter"
	"sync/atomic"
	"testing"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/session"
)

// vbugM3True returns a pointer to true for the tri-state WaitForOutput
// field.
func vbugM3True() *bool { b := true; return &b }

// vbugM3WaitNode is configured with WaitForOutput=true but deliberately
// returns after emitting only an output-LESS (progress) event. Per the
// documented contract it should therefore park in NodeWaiting.
type vbugM3WaitNode struct {
	BaseNode
	ran atomic.Bool
}

func newVbugM3WaitNode(name string) *vbugM3WaitNode {
	return &vbugM3WaitNode{
		BaseNode: NewBaseNode(name, "", NodeConfig{WaitForOutput: vbugM3True()}),
	}
}

func (n *vbugM3WaitNode) Run(ctx agent.Context, _ any) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		n.ran.Store(true)
		// A progress event carries no Output. The node returns without
		// ever yielding an output-bearing event.
		yield(session.NewEvent(ctx.InvocationID()), nil)
	}
}

// TestVbugM3_WaitForOutputHoldsSuccessor runs Start -> N -> M, where N
// has WaitForOutput=true and never emits output. M must not run.
func TestVbugM3_WaitForOutputHoldsSuccessor(t *testing.T) {
	mockCtx := newSeededMockCtx(t)

	n := newVbugM3WaitNode("N")

	var mRan atomic.Bool
	m := NewFunctionNode("M",
		func(agent.Context, any) (string, error) {
			mRan.Store(true)
			return "m-out", nil
		}, defaultNodeConfig)

	w := mustNew(t, Chain(Start, n, m))

	var runErr error
	for _, err := range w.Run(mockCtx) {
		if err != nil && runErr == nil {
			runErr = err
		}
	}
	if runErr != nil {
		t.Fatalf("unexpected error from Run: %v", runErr)
	}

	// Sanity: N must have actually executed; otherwise the assertion
	// below would be vacuous.
	if !n.ran.Load() {
		t.Fatal("node N did not run; test fixture is broken")
	}

	if mRan.Load() {
		t.Errorf("successor M ran, but N (WaitForOutput=true) emitted no output; " +
			"N should have stayed NodeWaiting and not scheduled successors. " +
			"WaitForOutput is being ignored by the scheduler")
	}
}
