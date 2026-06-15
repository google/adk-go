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

// FINDING H6 — a plain (non-Join) node with multiple predecessors is
// scheduled (and executed) more than once.
//
// Bug: the scheduler dedups successor dispatch only WITHIN a single
// activation; there is no cross-activation dedup and no "already
// running/completed" guard in startNode. In a diamond Start->A,
// A->B, A->C, B->D, C->D where D is a PLAIN node (not a JoinNode), D is
// scheduled once when B completes and AGAIN when C completes. D then
// runs twice. Depending on interleaving this either yields D's output
// twice or trips the per-name accumulator's single-output guard
// (ErrMultipleOutputs), corrupting the run. Only JoinNode predecessors
// are barrier-gated; a plain fan-in node has no such protection.
//
// Expected correct behavior: a plain node with several predecessors
// should execute exactly once per run (the engine should converge the
// branches, e.g. via a barrier or a running/completed guard), and the
// workflow should complete cleanly.
//
// This test currently FAILS, demonstrating the bug: D executes twice
// (count == 2) and/or the run returns an error.

package workflow

import (
	"fmt"
	"iter"
	"sync/atomic"
	"testing"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/session"
)

// vbugH6CountingNode mirrors the package's recording node but counts
// how many times it actually executes (its iterator body runs). It
// yields a single output-bearing event per execution.
type vbugH6CountingNode struct {
	BaseNode
	count atomic.Int32
}

func newVbugH6CountingNode(name string) *vbugH6CountingNode {
	return &vbugH6CountingNode{BaseNode: NewBaseNode(name, "", NodeConfig{})}
}

func (n *vbugH6CountingNode) Run(ctx agent.Context, input any) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		n.count.Add(1)
		ev := session.NewEvent(ctx.InvocationID())
		ev.Output = fmt.Sprintf("%v:%s", input, n.Name())
		yield(ev, nil)
	}
}

// TestVbugH6_PlainFanInRunsOnce builds a diamond converging on a plain
// counting node D and asserts D executes exactly once and the run is
// error-free.
func TestVbugH6_PlainFanInRunsOnce(t *testing.T) {
	mockCtx := newSeededMockCtx(t)

	a := NewFunctionNode("A",
		func(agent.Context, any) (string, error) { return "a", nil },
		defaultNodeConfig)
	b := NewFunctionNode("B",
		func(agent.Context, any) (string, error) { return "b", nil },
		defaultNodeConfig)
	c := NewFunctionNode("C",
		func(agent.Context, any) (string, error) { return "c", nil },
		defaultNodeConfig)
	d := newVbugH6CountingNode("D")

	// Diamond: Start -> A; A fans out to B and C; B and C both feed
	// the plain node D.
	edges := []Edge{
		{From: Start, To: a},
		{From: a, To: b},
		{From: a, To: c},
		{From: b, To: d},
		{From: c, To: d},
	}
	w := mustNew(t, edges)

	// Consume the whole stream manually (do not use drain, which would
	// t.Fatalf on the ErrMultipleOutputs interleaving and mask the
	// count assertion).
	var runErr error
	for _, err := range w.Run(mockCtx) {
		if err != nil && runErr == nil {
			runErr = err
		}
	}

	if got := d.count.Load(); got != 1 {
		t.Errorf("plain fan-in node D executed %d times, want exactly 1 "+
			"(diamond converging on a non-Join node is double-scheduled)", got)
	}
	if runErr != nil {
		t.Errorf("workflow returned error %v, want clean completion "+
			"(double-scheduling D collides its single-output accumulator)", runErr)
	}
}
