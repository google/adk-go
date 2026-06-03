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
	"reflect"
	"sort"
	"testing"

	"google.golang.org/adk/agent"
)

// TestJoinNode_E2E_FanInTwoBranches verifies that with two
// parallel predecessors the JoinNode is activated once with the
// aggregated map; the downstream consumer observes it verbatim.
func TestJoinNode_E2E_FanInTwoBranches(t *testing.T) {
	mockCtx := newSeededMockCtx(t)

	branchA := constNode("branchA", "A-result")
	branchB := constNode("branchB", "B-result")
	join := NewJoinNode("join")
	var seen any
	handler := inputRecorder("handler", &seen)

	w := mustNew(t, []Edge{
		{From: Start, To: branchA},
		{From: Start, To: branchB},
		{From: branchA, To: join},
		{From: branchB, To: join},
		{From: join, To: handler},
	})

	drain(t, w.Run(mockCtx))

	gotMap, ok := seen.(map[string]any)
	if !ok {
		t.Fatalf("handler input = %T %#v, want map[string]any", seen, seen)
	}
	wantMap := map[string]any{
		"branchA": "A-result",
		"branchB": "B-result",
	}
	if !reflect.DeepEqual(gotMap, wantMap) {
		t.Errorf("handler input = %#v, want %#v", gotMap, wantMap)
	}
}

// TestJoinNode_E2E_FanInThreeBranches scales the canonical case
// to three predecessors to verify the dispatch list does not
// special-case the two-predecessor count.
func TestJoinNode_E2E_FanInThreeBranches(t *testing.T) {
	mockCtx := newSeededMockCtx(t)

	a := constNode("a", "va")
	b := constNode("b", "vb")
	c := constNode("c", "vc")
	join := NewJoinNode("join")
	var seen any
	handler := inputRecorder("handler", &seen)

	w := mustNew(t, []Edge{
		{From: Start, To: a},
		{From: Start, To: b},
		{From: Start, To: c},
		{From: a, To: join},
		{From: b, To: join},
		{From: c, To: join},
		{From: join, To: handler},
	})

	drain(t, w.Run(mockCtx))

	m, _ := seen.(map[string]any)
	var seenKeys []string
	for k := range m {
		seenKeys = append(seenKeys, k)
	}
	sort.Strings(seenKeys)
	if want := []string{"a", "b", "c"}; !reflect.DeepEqual(seenKeys, want) {
		t.Errorf("handler input keys = %v, want %v", seenKeys, want)
	}
}

// TestJoinNode_BarrierSkipsUntilAllPredecessorsComplete pins the
// barrier semantics directly. With branchA fast and branchB slow
// (a fresh-channel block lifted by a separate goroutine), the
// handler must not see partial aggregation: the join is not
// triggered when only A has completed.
func TestJoinNode_BarrierSkipsUntilAllPredecessorsComplete(t *testing.T) {
	mockCtx := newSeededMockCtx(t)

	bBlocker := make(chan struct{})
	branchA := NewFunctionNode("branchA",
		func(ctx agent.InvocationContext, _ any) (string, error) {
			// Release B once A's function body returns. The
			// barrier must hold until B also completes, so the
			// handler runs exactly once with both outputs.
			close(bBlocker)
			return "a", nil
		}, defaultNodeConfig)
	branchB := NewFunctionNode("branchB",
		func(ctx agent.InvocationContext, _ any) (string, error) {
			<-bBlocker
			return "b", nil
		}, defaultNodeConfig)
	join := NewJoinNode("join")

	handlerCalls := 0
	var handlerInput any
	handler := NewFunctionNode("handler",
		func(ctx agent.InvocationContext, input any) (string, error) {
			handlerCalls++
			handlerInput = input
			return "ok", nil
		}, defaultNodeConfig)

	w := mustNew(t, []Edge{
		{From: Start, To: branchA},
		{From: Start, To: branchB},
		{From: branchA, To: join},
		{From: branchB, To: join},
		{From: join, To: handler},
	})

	drain(t, w.Run(mockCtx))

	if handlerCalls != 1 {
		t.Fatalf("handler ran %d times, want exactly 1", handlerCalls)
	}
	got, _ := handlerInput.(map[string]any)
	want := map[string]any{"branchA": "a", "branchB": "b"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("handler input = %#v, want %#v", got, want)
	}
}

// TestJoinNode_RunIsPassthrough pins the JoinNode contract
// independently of the engine: given an aggregated input map,
// the node emits exactly one event whose Output
// equals the input. This is the property the orchestrator relies
// on when treating JoinNode as a no-op aggregator.
func TestJoinNode_RunIsPassthrough(t *testing.T) {
	mockCtx := newSeededMockCtx(t)
	n := NewJoinNode("join")

	input := map[string]any{"a": "x", "b": 42}
	events := drain(t, n.Run(mockCtx, input))

	if len(events) != 1 {
		t.Fatalf("expected 1 event, got %d", len(events))
	}
	got := events[0].Output
	if !reflect.DeepEqual(got, input) {
		t.Errorf("Output = %#v, want %#v (verbatim)", got, input)
	}
}

// TestJoinNode_PredecessorWithNilOutput pins the aggregation
// contract for a predecessor whose recorded Output is nil: the
// aggregated map carries the predecessor's name with a nil value,
// and the join activates normally — the barrier counts
// completions, not outputs.
func TestJoinNode_PredecessorWithNilOutput(t *testing.T) {
	mockCtx := newSeededMockCtx(t)

	branchA := constNode("branchA", "A-result")
	branchNil := NewFunctionNode("branchNil",
		func(ctx agent.InvocationContext, _ any) (any, error) {
			return nil, nil
		}, defaultNodeConfig)
	join := NewJoinNode("join")
	var seen any
	handler := inputRecorder("handler", &seen)

	w := mustNew(t, []Edge{
		{From: Start, To: branchA},
		{From: Start, To: branchNil},
		{From: branchA, To: join},
		{From: branchNil, To: join},
		{From: join, To: handler},
	})

	drain(t, w.Run(mockCtx))

	gotMap, ok := seen.(map[string]any)
	if !ok {
		t.Fatalf("handler input = %T %#v, want map[string]any", seen, seen)
	}
	wantMap := map[string]any{
		"branchA":   "A-result",
		"branchNil": nil,
	}
	if !reflect.DeepEqual(gotMap, wantMap) {
		t.Errorf("handler input = %#v, want %#v", gotMap, wantMap)
	}
}

// =============================================================================
// Test fixtures and helpers
// =============================================================================

// constNode returns a FunctionNode that ignores its input and
// yields the given string value.
func constNode(name, value string) *FunctionNode {
	return NewFunctionNode(name,
		func(agent.InvocationContext, any) (string, error) { return value, nil },
		defaultNodeConfig)
}

// inputRecorder returns a FunctionNode that stores its input in
// *seen and yields "done". Use it for the terminal consumer of a
// graph when the assertion only inspects the input it observed.
func inputRecorder(name string, seen *any) *FunctionNode {
	return NewFunctionNode(name,
		func(_ agent.InvocationContext, input any) (string, error) {
			*seen = input
			return "done", nil
		}, defaultNodeConfig)
}
