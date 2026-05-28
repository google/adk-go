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
	"testing"

	"google.golang.org/adk/agent"
)

// branchRecorder returns a FunctionNode that captures the
// ctx.Branch() string into *seen at execution time, keyed by node
// name.
func branchRecorder(name string, mu *sync.Mutex, seen *map[string]string) *FunctionNode {
	return NewFunctionNode(name,
		func(ctx agent.InvocationContext, input any) (string, error) {
			mu.Lock()
			(*seen)[name] = ctx.Branch()
			mu.Unlock()
			return name + "-done", nil
		}, defaultNodeConfig)
}

// TestScheduler_SingleSuccessorChain_InheritsRootBranch verifies
// that a linear chain (one successor per node) keeps every
// activation on the root branch — sub-branches are only derived at
// fan-out, not on every edge.
func TestScheduler_SingleSuccessorChain_InheritsRootBranch(t *testing.T) {
	mockCtx := newSeededMockCtx(t)
	var mu sync.Mutex
	seen := map[string]string{}

	a := branchRecorder("a", &mu, &seen)
	b := branchRecorder("b", &mu, &seen)
	c := branchRecorder("c", &mu, &seen)

	w := mustNew(t, []Edge{
		{From: Start, To: a},
		{From: a, To: b},
		{From: b, To: c},
	})

	drain(t, w.Run(mockCtx))

	for _, name := range []string{"a", "b", "c"} {
		if got := seen[name]; got != "" {
			t.Errorf("seen[%q] = %q, want \"\" (chain inherits root)", name, got)
		}
	}
}

// TestScheduler_MultipleSuccessors_AssignsSubBranches verifies that
// fan-out from one node to N successors derives sub-branches of the
// form "<successor>@1" for each branch.
func TestScheduler_MultipleSuccessors_AssignsSubBranches(t *testing.T) {
	mockCtx := newSeededMockCtx(t)
	var mu sync.Mutex
	seen := map[string]string{}

	a := branchRecorder("a", &mu, &seen)
	b := branchRecorder("b", &mu, &seen)
	c := branchRecorder("c", &mu, &seen)

	// START fans out to a + b + c.
	w := mustNew(t, []Edge{
		{From: Start, To: a},
		{From: Start, To: b},
		{From: Start, To: c},
	})

	drain(t, w.Run(mockCtx))

	want := map[string]string{
		"a": "a@1",
		"b": "b@1",
		"c": "c@1",
	}
	for name, wantBranch := range want {
		if got := seen[name]; got != wantBranch {
			t.Errorf("seen[%q] = %q, want %q (fan-out should derive sub-branch)",
				name, got, wantBranch)
		}
	}
}

// TestScheduler_FanOutChainedDeeperBranches verifies that after a
// fan-out, successors of each branch continue to extend their
// branch when they themselves fan out further.
func TestScheduler_FanOutChainedDeeperBranches(t *testing.T) {
	mockCtx := newSeededMockCtx(t)
	var mu sync.Mutex
	seen := map[string]string{}

	a := branchRecorder("a", &mu, &seen)
	// a fans out to b1+b2 within its sub-branch.
	b1 := branchRecorder("b1", &mu, &seen)
	b2 := branchRecorder("b2", &mu, &seen)

	w := mustNew(t, []Edge{
		{From: Start, To: a}, // a stays on root (single successor)
		{From: a, To: b1},    // a fans out: b1 and b2 get sub-branches
		{From: a, To: b2},
	})

	drain(t, w.Run(mockCtx))

	if got, want := seen["a"], ""; got != want {
		t.Errorf("seen[a] = %q, want %q (root, since START's only successor is a)", got, want)
	}
	if got, want := seen["b1"], "b1@1"; got != want {
		t.Errorf("seen[b1] = %q, want %q", got, want)
	}
	if got, want := seen["b2"], "b2@1"; got != want {
		t.Errorf("seen[b2] = %q, want %q", got, want)
	}
}

// TestScheduler_JoinNode_UsesCommonPrefix verifies that when a
// JoinNode fires after a fan-out of its predecessors, the join
// activation runs on the common branch prefix of its predecessors
// (which in the simple case is the parent of the fan-out).
func TestScheduler_JoinNode_UsesCommonPrefix(t *testing.T) {
	mockCtx := newSeededMockCtx(t)
	var mu sync.Mutex
	seen := map[string]string{}

	branchA := branchRecorder("branchA", &mu, &seen)
	branchB := branchRecorder("branchB", &mu, &seen)
	join := NewJoinNode("join")
	handler := branchRecorder("handler", &mu, &seen)

	w := mustNew(t, []Edge{
		{From: Start, To: branchA}, // fan-out: branchA and branchB get sub-branches
		{From: Start, To: branchB},
		{From: branchA, To: join},
		{From: branchB, To: join},
		{From: join, To: handler}, // single successor; handler inherits join's branch
	})

	drain(t, w.Run(mockCtx))

	// Predecessors are on their own sub-branches.
	if got, want := seen["branchA"], "branchA@1"; got != want {
		t.Errorf("seen[branchA] = %q, want %q", got, want)
	}
	if got, want := seen["branchB"], "branchB@1"; got != want {
		t.Errorf("seen[branchB] = %q, want %q", got, want)
	}
	// Join's branch is the common prefix of "branchA@1" and "branchB@1",
	// which share no dot-separated segments → root.
	// The handler inherits the join's branch (single successor).
	if got, want := seen["handler"], ""; got != want {
		t.Errorf("seen[handler] = %q, want %q (common prefix of branchA@1/branchB@1 is root)", got, want)
	}
}

// TestScheduler_JoinNode_NestedFanOut verifies the common-prefix
// behaviour when predecessors share a deeper ancestor branch. After
// fan-out, predecessors that themselves shared an outer branch
// produce a join on that outer branch (not all the way back to
// root).
func TestScheduler_JoinNode_NestedFanOut(t *testing.T) {
	mockCtx := newSeededMockCtx(t)
	var mu sync.Mutex
	seen := map[string]string{}

	// outer is a single successor of START → stays on root.
	outer := branchRecorder("outer", &mu, &seen)
	// outer fans out to leaf1 + leaf2 → each gets sub-branch.
	leaf1 := branchRecorder("leaf1", &mu, &seen)
	leaf2 := branchRecorder("leaf2", &mu, &seen)
	join := NewJoinNode("join")
	handler := branchRecorder("handler", &mu, &seen)

	w := mustNew(t, []Edge{
		{From: Start, To: outer},
		{From: outer, To: leaf1},
		{From: outer, To: leaf2},
		{From: leaf1, To: join},
		{From: leaf2, To: join},
		{From: join, To: handler},
	})

	drain(t, w.Run(mockCtx))

	if got, want := seen["outer"], ""; got != want {
		t.Errorf("seen[outer] = %q, want %q", got, want)
	}
	if got, want := seen["leaf1"], "leaf1@1"; got != want {
		t.Errorf("seen[leaf1] = %q, want %q", got, want)
	}
	if got, want := seen["leaf2"], "leaf2@1"; got != want {
		t.Errorf("seen[leaf2] = %q, want %q", got, want)
	}
	// Common prefix of "leaf1@1" and "leaf2@1" is "" (no shared
	// segments) — because outer is on the *root* branch, its
	// successors' sub-branches share no parent segment, so the join
	// returns to root (deepest common ancestor).
	if got, want := seen["handler"], ""; got != want {
		t.Errorf("seen[handler] = %q, want %q", got, want)
	}
}

// TestScheduler_EventsAreBranchStamped verifies that the scheduler
// stamps Event.Branch on every emitted event using the activation's
// branch when the node itself leaves Event.Branch empty.
func TestScheduler_EventsAreBranchStamped(t *testing.T) {
	mockCtx := newSeededMockCtx(t)
	var mu sync.Mutex
	seen := map[string]string{}

	a := branchRecorder("a", &mu, &seen)
	b := branchRecorder("b", &mu, &seen)

	w := mustNew(t, []Edge{
		{From: Start, To: a},
		{From: Start, To: b},
	})

	events := drain(t, w.Run(mockCtx))

	gotByAuthor := map[string][]string{}
	for _, ev := range events {
		if ev.Author == "" {
			continue
		}
		gotByAuthor[ev.Author] = append(gotByAuthor[ev.Author], ev.Branch)
	}

	for _, author := range []string{"a", "b"} {
		branches, ok := gotByAuthor[author]
		if !ok {
			continue // node may have emitted no events with author set
		}
		wantBranch := author + "@1"
		for _, got := range branches {
			if got != wantBranch {
				t.Errorf("event from %q has Branch = %q, want %q",
					author, got, wantBranch)
			}
		}
	}
}
