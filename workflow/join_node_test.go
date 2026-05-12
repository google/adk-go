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
	"sync"
	"sync/atomic"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"

	"google.golang.org/adk/agent"
)

// TestNewJoinNode_ForcesWaitForOutput verifies that the constructor
// hard-codes Config().WaitForOutput=true regardless of what the
// caller passes — JoinNode's contract depends on it.
func TestNewJoinNode_ForcesWaitForOutput(t *testing.T) {
	t.Run("nil WaitForOutput is forced to true", func(t *testing.T) {
		jn := NewJoinNode("J", NodeConfig{})
		got := jn.Config().WaitForOutput
		if got == nil || !*got {
			t.Errorf("Config().WaitForOutput = %v, want non-nil pointer to true", got)
		}
	})
	t.Run("explicit false WaitForOutput is overridden to true", func(t *testing.T) {
		f := false
		jn := NewJoinNode("J", NodeConfig{WaitForOutput: &f})
		got := jn.Config().WaitForOutput
		if got == nil || !*got {
			t.Errorf("Config().WaitForOutput = %v, want non-nil pointer to true (constructor must override)", got)
		}
	})
}

// TestJoinNode_E2E_FanInTwoBranches exercises the canonical fan-in
// graph:
//
//	Start ──► A ──► J ──► D
//	  │             ▲
//	  └────► B ─────┘
//
// The JoinNode J should fire its successor D exactly once, with an
// aggregated map[string]any input keyed by predecessor name.
func TestJoinNode_E2E_FanInTwoBranches(t *testing.T) {
	mockCtx := newSeededMockCtx(t)

	a := NewFunctionNode("A", func(ctx agent.InvocationContext, input string) (string, error) {
		return "outA", nil
	}, defaultNodeConfig)
	b := NewFunctionNode("B", func(ctx agent.InvocationContext, input string) (string, error) {
		return "outB", nil
	}, defaultNodeConfig)
	j := NewJoinNode("J", NodeConfig{})

	var dActivations atomic.Int32
	var dInputMu sync.Mutex
	var dInput map[string]any

	d := NewFunctionNode("D", func(ctx agent.InvocationContext, input map[string]any) (string, error) {
		dActivations.Add(1)
		dInputMu.Lock()
		dInput = input
		dInputMu.Unlock()
		return "outD", nil
	}, defaultNodeConfig)

	w := mustNew(t, []Edge{
		{From: Start, To: a},
		{From: Start, To: b},
		{From: a, To: j},
		{From: b, To: j},
		{From: j, To: d},
	})

	drain(t, w.Run(mockCtx))

	if got, want := dActivations.Load(), int32(1); got != want {
		t.Errorf("D activation count = %d, want %d (JoinNode must fire downstream exactly once after both predecessors)", got, want)
	}

	dInputMu.Lock()
	defer dInputMu.Unlock()
	want := map[string]any{"A": "outA", "B": "outB"}
	if diff := cmp.Diff(want, dInput); diff != "" {
		t.Errorf("D's input from JoinNode (-want +got):\n%s", diff)
	}
}

// TestJoinNode_E2E_FanInThreeBranches exercises the three-way fan-in
// example from the API design doc:
//
//	Start ──► A ──► J ──► D
//	  ├────► B ─────┤
//	  └────► C ─────┘
//
// Verifies the JoinNode's behaviour scales beyond two predecessors
// and that the merged map contains every contributing branch's
// output.
func TestJoinNode_E2E_FanInThreeBranches(t *testing.T) {
	mockCtx := newSeededMockCtx(t)

	a := NewFunctionNode("A", func(ctx agent.InvocationContext, input string) (string, error) {
		return "outA", nil
	}, defaultNodeConfig)
	b := NewFunctionNode("B", func(ctx agent.InvocationContext, input string) (string, error) {
		return "outB", nil
	}, defaultNodeConfig)
	c := NewFunctionNode("C", func(ctx agent.InvocationContext, input string) (string, error) {
		return "outC", nil
	}, defaultNodeConfig)
	j := NewJoinNode("J", NodeConfig{})

	var dActivations atomic.Int32
	var dInputMu sync.Mutex
	var dInput map[string]any

	d := NewFunctionNode("D", func(ctx agent.InvocationContext, input map[string]any) (string, error) {
		dActivations.Add(1)
		dInputMu.Lock()
		dInput = input
		dInputMu.Unlock()
		return "outD", nil
	}, defaultNodeConfig)

	w := mustNew(t, []Edge{
		{From: Start, To: a},
		{From: Start, To: b},
		{From: Start, To: c},
		{From: a, To: j},
		{From: b, To: j},
		{From: c, To: j},
		{From: j, To: d},
	})

	drain(t, w.Run(mockCtx))

	if got, want := dActivations.Load(), int32(1); got != want {
		t.Errorf("D activation count = %d, want %d", got, want)
	}

	dInputMu.Lock()
	defer dInputMu.Unlock()
	want := map[string]any{"A": "outA", "B": "outB", "C": "outC"}
	if diff := cmp.Diff(want, dInput); diff != "" {
		t.Errorf("D's input from JoinNode (-want +got):\n%s", diff)
	}
}

// TestJoinNode_Run_DetachedYieldsDegenerateOutput verifies that a
// JoinNode invoked outside the workflow engine (no joinAccumulator
// on the context) emits a single-key map for the current trigger
// rather than failing or hanging. This keeps the node usable in
// unit tests that exercise Run directly.
func TestJoinNode_Run_DetachedYieldsDegenerateOutput(t *testing.T) {
	jn := NewJoinNode("J", NodeConfig{})
	ctx := newMockCtx(t)

	events := drain(t, jn.Run(ctx, "lonely-input"))

	if got, want := len(events), 1; got != want {
		t.Fatalf("event count = %d, want %d", got, want)
	}
	got, ok := events[0].Actions.StateDelta["output"].(map[string]any)
	if !ok {
		t.Fatalf("output value type = %T, want map[string]any", events[0].Actions.StateDelta["output"])
	}
	// The detached path keys the map on ctx.TriggeredBy(), which is
	// "" for MockInvocationContext — that's the documented degenerate
	// behaviour.
	want := map[string]any{"": "lonely-input"}
	if diff := cmp.Diff(want, got, cmpopts.EquateEmpty()); diff != "" {
		t.Errorf("detached JoinNode output (-want +got):\n%s", diff)
	}
}

// TestJoinNode_Run_RejectsUnknownPredecessor verifies that a
// JoinNode whose nodeContext claims a triggering predecessor
// outside its declared in-nodes set surfaces an error rather than
// silently corrupting the accumulator.
func TestJoinNode_Run_RejectsUnknownPredecessor(t *testing.T) {
	jn := NewJoinNode("J", NodeConfig{})
	parent := newMockCtx(t)
	ctx := newNodeContext(parent, "Z" /*not in InNodes*/, map[string]struct{}{"A": {}, "B": {}}, &joinAccumulator{})

	gotErr := drainErr(t, jn.Run(ctx, "x"))
	if gotErr == nil {
		t.Fatal("expected an error for unknown-predecessor trigger, got none")
	}
}
