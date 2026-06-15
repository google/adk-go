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

// FINDING H3 — WorkflowNode.Run re-invokes its iterator yield after it
// returned false, panicking the range-over-func machinery.
//
// Bug: WorkflowNode.Run wraps a nested sub-workflow. When its consumer
// stops consuming mid-stream — which is exactly what the parent
// scheduler does when a sibling node fails and the WorkflowNode's
// context is cancelled, making the consuming runNode return — the outer
// yield returns false. Instead of stopping, WorkflowNode.Run does
// "cancel(); continue" and then calls yield AGAIN on the next inner
// event. Calling a range-over-func yield after it has returned false
// panics ("range function continued iteration after loop body returned
// false"). The scheduler's own consumer loop guards against this with a
// consumerGone flag; WorkflowNode.Run has no such guard.
//
// Expected correct behavior: once its consumer stops, WorkflowNode.Run
// must stop calling yield (drain/cancel without re-yielding), so a
// mid-stream break unwinds cleanly with no panic.
//
// This test reproduces the exact buggy path directly: it consumes one
// event from WorkflowNode.Run, then breaks — modelling the scheduler
// abandoning the node mid-stream. A correct implementation never
// panics; the buggy one does.
//
// Timing note (BEST-EFFORT): after the break, WorkflowNode.Run cancels
// the sub-workflow and continues. Whether the sub-scheduler yields one
// more event (triggering the re-yield panic) before it observes the
// cancellation is a scheduling race. To make the bug reliably
// observable in a single run, this test repeats the break scenario many
// times and fails if ANY attempt panics. If you want to confirm
// independently, it can also be run with: go test ./workflow/
// -run TestVbugH3 -count=10
//
// This test currently FAILS, demonstrating the bug.

package workflow

import (
	"iter"
	"testing"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/session"
)

// vbugH3BurstNode yields n output-less ("partial") events. Partial
// events are fire-and-forget (no back-pressure handshake), so they
// buffer in the sub-scheduler's event queue — which keeps a follow-up
// event ready to be (incorrectly) re-yielded the instant the consumer
// breaks.
type vbugH3BurstNode struct {
	BaseNode
	n int
}

func newVbugH3BurstNode(name string, n int) *vbugH3BurstNode {
	return &vbugH3BurstNode{BaseNode: NewBaseNode(name, "", NodeConfig{}), n: n}
}

func (b *vbugH3BurstNode) Run(ctx agent.Context, _ any) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		for range b.n {
			ev := session.NewEvent(ctx.InvocationID())
			ev.LLMResponse.Partial = true
			if !yield(ev, nil) {
				return
			}
		}
	}
}

// vbugH3RunOnce builds a fresh nested WorkflowNode wrapping a
// multi-event sub-workflow, consumes exactly one event, then breaks —
// modelling the parent scheduler abandoning the WorkflowNode mid-stream
// when a sibling fails. It returns the recovered panic value, or nil if
// the break unwound cleanly.
func vbugH3RunOnce(t *testing.T) (recovered any) {
	t.Helper()

	inner := newVbugH3BurstNode("vbugH3_inner", 32)
	wfNode, err := NewWorkflowNode("vbugH3_nested", Chain(Start, inner))
	if err != nil {
		t.Fatalf("NewWorkflowNode: %v", err)
	}
	exCtx := agent.NewNodeContext(newSeededMockCtx(t), nil)

	defer func() {
		if r := recover(); r != nil {
			recovered = r
		}
	}()

	count := 0
	for range wfNode.Run(exCtx, "seed") {
		count++
		if count == 1 {
			break // abandon the WorkflowNode mid-stream
		}
	}
	return nil
}

// TestVbugH3_NestedWorkflowMidStreamBreakNoPanic asserts that breaking
// out of a nested WorkflowNode's event stream never panics.
func TestVbugH3_NestedWorkflowMidStreamBreakNoPanic(t *testing.T) {
	const attempts = 300

	panicCount := 0
	var lastPanic any
	for range attempts {
		if r := vbugH3RunOnce(t); r != nil {
			panicCount++
			lastPanic = r
		}
	}

	if panicCount > 0 {
		t.Errorf("WorkflowNode.Run panicked on %d/%d mid-stream breaks; last panic: %v\n"+
			"WorkflowNode.Run calls yield again after yield returned false "+
			"(it does cancel();continue instead of stopping), which is illegal for a "+
			"range-over-func iterator. It needs the consumer-gone guard the scheduler's "+
			"own loop has.", panicCount, attempts, lastPanic)
	}
}
