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
	"fmt"
	"slices"
	"sort"
	"testing"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/session"
)

// TestRunNode_SequentialFanOut_BranchesFromOptions drives a
// dynamic orchestrator that sequentially calls RunNode three times
// on the same child and verifies the per-call Branch() observed by
// the child as a function of the RunNode options passed.
//
// Why sequential (not errgroup): per req §5.1 D-Emit-Sequential,
// emit / RunNode are single-goroutine only. Running fan-out via
// errgroup violates the iter.Seq2 contract (concurrent yield calls
// from multiple goroutines) and races on the event-collection slice
// inside drainDynamicWithErr.
func TestRunNode_SequentialFanOut_BranchesFromOptions(t *testing.T) {
	const childName = "peeker"

	tests := []struct {
		name string
		// extraOpts are appended after the per-call WithRunID; nil
		// or empty means "inherit parent branch" (no opt-in).
		extraOpts []RunNodeOption
		// wantBranches is the sorted list of Branch() values the
		// child should observe across the three sequential calls.
		wantBranches []string
	}{
		{
			// Opt-in: each child sees a distinct sub-branch of the
			// form "<childName>@<customRunID>" off the
			// orchestrator's root branch.
			name:      "WithUseSubBranch_DistinctBranches",
			extraOpts: []RunNodeOption{WithUseSubBranch()},
			wantBranches: []string{
				childName + "@custom-1",
				childName + "@custom-2",
				childName + "@custom-3",
			},
		},
		{
			// Default behaviour: without the opt-in, children
			// inherit the orchestrator's branch verbatim
			// (backward-compat contract for callers relying on
			// inherited branches).
			name:         "NoOption_StillSharesBranch",
			extraOpts:    nil,
			wantBranches: []string{"", "", ""},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			seenBranches := runSequentialFanOut(t, childName, tc.extraOpts)
			sort.Strings(seenBranches)
			if !slices.Equal(seenBranches, tc.wantBranches) {
				t.Errorf("seenBranches = %q, want %q",
					seenBranches, tc.wantBranches)
			}
		})
	}
}

// runSequentialFanOut drives a dynamic orchestrator that sequentially
// invokes a peeker child three times (items "a"/"b"/"c"), passing
// WithRunID("custom-N") plus the caller-supplied extraOpts on each
// call, and returns the Branch() values the child observed in call
// order. Fails the test if the orchestrator errors or invokes the
// child a wrong number of times.
func runSequentialFanOut(t *testing.T, childName string, extraOpts []RunNodeOption) []string {
	t.Helper()
	const nItems = 3

	var seenBranches []string
	peekerNode := NewFunctionNode(
		childName,
		func(_ context.Context, ctx agent.Context, input string) (string, error) {
			seenBranches = append(seenBranches, ctx.Branch())
			return input, nil
		},
		NodeConfig{},
	)

	orch := NewDynamicNode[any, []string](
		"orch",
		func(ctx context.Context, invCleanCtx NodeContext, _ any, _ func(*session.Event) error) ([]string, error) {
			items := []string{"a", "b", "c"}
			results := make([]string, len(items))
			for i, item := range items {
				opts := append([]RunNodeOption{
					WithRunID(fmt.Sprintf("custom-%d", i+1)),
				}, extraOpts...)
				out, err := RunNode[string](ctx, invCleanCtx, peekerNode, item, opts...)
				if err != nil {
					return nil, err
				}
				results[i] = out
			}
			return results, nil
		},
		NodeConfig{},
	)

	if _, err := drainDynamicWithErr(t, orch, nil); err != nil {
		t.Fatalf("orchestrator: %v", err)
	}
	if got, want := len(seenBranches), nItems; got != want {
		t.Fatalf("got %d peeker invocations, want %d", got, want)
	}
	return seenBranches
}
