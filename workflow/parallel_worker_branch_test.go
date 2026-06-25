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
	"errors"
	"sort"
	"sync"
	"testing"

	"google.golang.org/adk/agent"
)

// TestParallelWorker_PerItemSubBranch verifies that ParallelWorker
// gives each of its N concurrent workers a distinct sub-branch of
// the form "<wrapped>@<i+1>", so a downstream LlmAgent reading the
// session event log under the LLM flow's branch-prefix filter no
// longer sees sibling-worker events in its prompt history.
func TestParallelWorker_PerItemSubBranch(t *testing.T) {
	const wrappedName = "summarize"

	var (
		mu           sync.Mutex
		seenBranches []string
	)
	wrapped := NewFunctionNode(wrappedName,
		func(ctx agent.Context, input string) (string, error) {
			mu.Lock()
			seenBranches = append(seenBranches, ctx.Branch())
			mu.Unlock()
			return input + "-done", nil
		}, defaultNodeConfig)

	pw, err := NewParallelWorker("pw", wrapped, 0, defaultNodeConfig)
	if err != nil {
		t.Fatal(err)
	}

	mockCtx := newMockCtx(t)
	exCtx := agent.NewContext(mockCtx)

	events := pw.Run(exCtx, []any{"a", "b", "c"})
	for _, err := range events {
		if err != nil {
			t.Fatalf("Run error: %v", err)
		}
	}

	if got, want := len(seenBranches), 3; got != want {
		t.Fatalf("seenBranches len = %d, want %d", got, want)
	}

	// Parent (mockCtx) is on root branch "" — each worker's
	// sub-branch is the bare "<wrappedName>@<i+1>".
	sort.Strings(seenBranches)
	wantBranches := []string{
		wrappedName + "@1",
		wrappedName + "@2",
		wrappedName + "@3",
	}
	for i, want := range wantBranches {
		if seenBranches[i] != want {
			t.Errorf("seenBranches[%d] = %q, want %q",
				i, seenBranches[i], want)
		}
	}
}

// TestParallelWorker_SubBranchUnderNonRootParent verifies that
// ParallelWorker correctly appends the per-item segment to a non-
// empty parent branch (e.g. when ParallelWorker itself sits inside
// another fan-out and runs under a sub-branch).
func TestParallelWorker_SubBranchUnderNonRootParent(t *testing.T) {
	const wrappedName = "worker"

	var (
		mu           sync.Mutex
		seenBranches []string
	)
	wrapped := NewFunctionNode(wrappedName,
		func(ctx agent.Context, input string) (string, error) {
			mu.Lock()
			seenBranches = append(seenBranches, ctx.Branch())
			mu.Unlock()
			return input, nil
		}, defaultNodeConfig)

	pw, err := NewParallelWorker("pw", wrapped, 0, defaultNodeConfig)
	if err != nil {
		t.Fatal(err)
	}

	// Simulate ParallelWorker running under a parent branch
	// "outer@1" (e.g. one branch of a static fan-out).
	parentCtx := withBranch(newMockCtx(t), "outer@1")
	exCtx := agent.NewContext(parentCtx)

	events := pw.Run(exCtx, []any{"x", "y"})
	for _, err := range events {
		if err != nil {
			t.Fatalf("Run error: %v", err)
		}
	}

	if got, want := len(seenBranches), 2; got != want {
		t.Fatalf("seenBranches len = %d, want %d", got, want)
	}
	sort.Strings(seenBranches)
	wantBranches := []string{
		"outer@1." + wrappedName + "@1",
		"outer@1." + wrappedName + "@2",
	}
	for i, want := range wantBranches {
		if seenBranches[i] != want {
			t.Errorf("seenBranches[%d] = %q, want %q",
				i, seenBranches[i], want)
		}
	}
}

// TestParallelWorker_RetryKeepsSameBranch verifies that a retried
// item runs under the same sub-branch as the original attempt — the
// branch is captured into the per-item context once before the
// retry loop in runWorker.
func TestParallelWorker_RetryKeepsSameBranch(t *testing.T) {
	const wrappedName = "flaky"

	errAlwaysFail := errors.New("force retry")
	var (
		mu               sync.Mutex
		attemptsByBranch = map[string]int{}
	)
	wrapped := NewFunctionNode(wrappedName,
		func(ctx agent.Context, input string) (string, error) {
			mu.Lock()
			attemptsByBranch[ctx.Branch()]++
			mu.Unlock()
			return "", errAlwaysFail
		}, defaultNodeConfig)

	cfg := NodeConfig{RetryConfig: &RetryConfig{MaxAttempts: 3}}
	pw, err := NewParallelWorker("pw", wrapped, 0, cfg)
	if err != nil {
		t.Fatal(err)
	}

	mockCtx := newMockCtx(t)
	exCtx := agent.NewContext(mockCtx)

	var gotErr error
	for _, runErr := range pw.Run(exCtx, []any{"a"}) {
		if runErr != nil {
			gotErr = runErr
		}
	}
	if gotErr == nil {
		t.Fatal("expected wrapped node to ultimately fail after retries, got nil error")
	}

	if got := len(attemptsByBranch); got != 1 {
		t.Fatalf("attemptsByBranch keys = %d, want 1 "+
			"(retries must share one branch); got %+v", got, attemptsByBranch)
	}
	for branch, count := range attemptsByBranch {
		if branch != wrappedName+"@1" {
			t.Errorf("retry branch = %q, want %q", branch, wrappedName+"@1")
		}
		if count < 2 {
			t.Errorf("retry attempts under %q = %d, want ≥2", branch, count)
		}
	}
}
