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
	"fmt"
	"sort"
	"testing"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/session"
)

// TestRunNode_SequentialFanOut_WithUseSubBranch_DistinctBranches
// verifies that opting into WithUseSubBranch() at each RunNode call
// produces distinct per-child branches.
//
// Why sequential (not errgroup): per req §5.1 D-Emit-Sequential,
// emit / RunNode are single-goroutine only. Running fan-out via
// errgroup violates the iter.Seq2 contract (concurrent yield calls
// from multiple goroutines) and races on the event-collection slice
// inside drainDynamicWithErr.
func TestRunNode_SequentialFanOut_WithUseSubBranch_DistinctBranches(t *testing.T) {
	const (
		childName = "peeker"
		nItems    = 3
	)

	var seenBranches []string
	peekerNode := NewFunctionNode(
		childName,
		func(ctx agent.InvocationContext, input string) (string, error) {
			seenBranches = append(seenBranches, ctx.Branch())
			return input, nil
		},
		NodeConfig{},
	)

	orch := NewDynamicNode[any, []string](
		"orch",
		func(ctx NodeContext, _ any, _ func(*session.Event) error) ([]string, error) {
			items := []string{"a", "b", "c"}
			results := make([]string, len(items))
			for i, item := range items {
				out, err := RunNode[string](ctx, peekerNode, item,
					WithUseSubBranch(),
					WithRunID(fmt.Sprintf("custom-%d", i+1)))
				if err != nil {
					return nil, err
				}
				results[i] = out
			}
			return results, nil
		},
		NodeConfig{},
	)

	_, err := drainDynamicWithErr(t, orch, nil)
	if err != nil {
		t.Fatalf("orchestrator: %v", err)
	}

	if got, want := len(seenBranches), nItems; got != want {
		t.Fatalf("got %d peeker invocations, want %d", got, want)
	}

	// Parent (orchestrator) runs on root branch "" (mock ctx); each
	// child sub-branch is bare "<childName>@<customRunID>".
	sort.Strings(seenBranches)
	wantBranches := []string{
		childName + "@custom-1",
		childName + "@custom-2",
		childName + "@custom-3",
	}
	for i, want := range wantBranches {
		if seenBranches[i] != want {
			t.Errorf("seenBranches[%d] = %q, want %q",
				i, seenBranches[i], want)
		}
	}
}

// TestRunNode_SequentialFanOut_NoOption_StillSharesBranch pins the
// default behaviour: without the opt-in, children inherit the
// orchestrator's branch verbatim. This preserves backward
// compatibility for code that relies on inherited branches.
func TestRunNode_SequentialFanOut_NoOption_StillSharesBranch(t *testing.T) {
	const (
		childName = "peeker"
		nItems    = 3
	)

	var seenBranches []string
	peekerNode := NewFunctionNode(
		childName,
		func(ctx agent.InvocationContext, input string) (string, error) {
			seenBranches = append(seenBranches, ctx.Branch())
			return input, nil
		},
		NodeConfig{},
	)

	orch := NewDynamicNode[any, []string](
		"orch",
		func(ctx NodeContext, _ any, _ func(*session.Event) error) ([]string, error) {
			items := []string{"a", "b", "c"}
			results := make([]string, len(items))
			for i, item := range items {
				out, err := RunNode[string](ctx, peekerNode, item,
					WithRunID(fmt.Sprintf("custom-%d", i+1)))
				if err != nil {
					return nil, err
				}
				results[i] = out
			}
			return results, nil
		},
		NodeConfig{},
	)

	_, err := drainDynamicWithErr(t, orch, nil)
	if err != nil {
		t.Fatalf("orchestrator: %v", err)
	}

	if got, want := len(seenBranches), nItems; got != want {
		t.Fatalf("got %d peeker invocations, want %d", got, want)
	}
	for i, b := range seenBranches {
		if b != "" {
			t.Errorf("seenBranches[%d] = %q, want \"\" "+
				"(without WithUseSubBranch the child should inherit "+
				"the orchestrator's root branch)", i, b)
		}
	}
}
