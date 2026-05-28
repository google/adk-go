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
	"strings"

	"google.golang.org/adk/agent"
	icontext "google.golang.org/adk/internal/context"
)

// Branch composition helpers for the static, parallel, and dynamic
// schedulers. Branches are dot-separated strings identifying the
// position of an in-flight node within an invocation's parallel
// execution tree. The LLM history filter
// (internal/llminternal/contents_processor.go:eventBelongsToBranch)
// uses prefix matching with explicit dot delimiter to scope events:
// an agent on branch "a.b" sees events on branches "" (root), "a"
// (ancestor), and "a.b" (self) but not "a.c" (sibling).
//
// Mirrors adk-python's logic in:
//   - src/google/adk/workflow/_node_runner.py:198-209 (sub-branch derivation)
//   - src/google/adk/workflow/_workflow.py:59-71 (common-prefix)
//   - src/google/adk/flows/llm_flows/contents.py:885-900 (matching)

// deriveSubBranch appends segment to parent with the dot delimiter.
// An empty parent yields the bare segment (root + segment), keeping
// the resulting string non-dot-prefixed; an empty segment is a
// no-op returning parent unchanged.
//
// segment is expected to be a stable identifier — Python uses
// "<node_name>@<run_id>" — but this helper does not impose a shape.
// Callers that need uniqueness across replays must supply a stable
// run id (auto-counter or WithRunID for dynamic; index+1 for
// ParallelWorker; "<successor>@1" for static fan-out).
func deriveSubBranch(parent, segment string) string {
	if segment == "" {
		return parent
	}
	if parent == "" {
		return segment
	}
	return parent + "." + segment
}

// withBranch returns ctx wrapped with branch as its Branch().
// Today the only implementer of agent.InvocationContext is
// *icontext.InvocationContext, so the common path is a single type
// assertion + WithBranch. Anything else falls back to a small
// adapter that overrides only Branch() — sufficient for tests that
// supply a hand-rolled MockInvocationContext.
func withBranch(ctx agent.InvocationContext, branch string) agent.InvocationContext {
	if ctx.Branch() == branch {
		return ctx
	}
	if ic, ok := ctx.(*icontext.InvocationContext); ok {
		return ic.WithBranch(branch)
	}
	return &branchOverride{InvocationContext: ctx, branch: branch}
}

// branchOverride wraps any InvocationContext and overrides Branch().
// Used as a fallback when ctx is not the canonical
// *icontext.InvocationContext implementation (currently only test
// mocks). All other interface methods delegate to the embedded
// value.
//
// WithContext is overridden so the branch survives a subsequent
// context-cancellation wrap (e.g. ParallelWorker calls
// ctx.WithContext(cancelCtx) on its input, and the resulting ctx
// must still carry the override when callers later read Branch()).
type branchOverride struct {
	agent.InvocationContext
	branch string
}

func (b *branchOverride) Branch() string { return b.branch }

// WithContext returns a branchOverride wrapping the inner
// InvocationContext's WithContext result so the branch override is
// preserved through context-cancellation wrapping.
func (b *branchOverride) WithContext(ctx context.Context) agent.InvocationContext {
	return &branchOverride{
		InvocationContext: b.InvocationContext.WithContext(ctx),
		branch:            b.branch,
	}
}

// deriveChildBranch composes the branch for a dynamically-scheduled
// child given the parent's branch and the RunNode options. Mirrors
// adk-python's logic in _node_runner._create_child_context
// (_node_runner.py:198-209).
//
// Algorithm:
//   - base = overrideBranch if non-empty, else parentBranch
//   - useSubBranch=true → base.<name>@<runID> (or bare <name>@<runID>
//     when base is empty)
//   - useSubBranch=false → base unchanged
//
// Note: overrideBranch="" is treated as "no override" (Go does not
// distinguish nil from empty string the way Python distinguishes
// None from ""); see WithOverrideBranch godoc for the rationale.
func deriveChildBranch(parentBranch, name, runID string, useSubBranch bool, overrideBranch string) string {
	base := parentBranch
	if overrideBranch != "" {
		base = overrideBranch
	}
	if useSubBranch {
		return deriveSubBranch(base, name+"@"+runID)
	}
	return base
}

// commonBranchPrefix returns the longest dot-delimited prefix shared
// by all input branches. Used by JoinNode to derive its own branch
// from the branches of its predecessors so the join "returns up the
// tree" to the deepest common ancestor.
//
// Behavior:
//   - Empty input → "" (root).
//   - Any input branch == "" → "" (root contains everything).
//   - All inputs identical → that exact value.
//   - Otherwise the longest shared dot-segment prefix.
//
// Note that segment-aware comparison is intentional: branches "a"
// and "ab" share no prefix (zero common segments), not "a"-as-string.
// This matches Python's segment-split behaviour at _workflow.py:63-71.
func commonBranchPrefix(branches []string) string {
	if len(branches) == 0 {
		return ""
	}
	splits := make([][]string, len(branches))
	minLen := -1
	for i, b := range branches {
		if b == "" {
			return ""
		}
		segs := strings.Split(b, ".")
		splits[i] = segs
		if minLen < 0 || len(segs) < minLen {
			minLen = len(segs)
		}
	}
	commonCount := 0
	for i := 0; i < minLen; i++ {
		seg := splits[0][i]
		for _, s := range splits[1:] {
			if s[i] != seg {
				return strings.Join(splits[0][:commonCount], ".")
			}
		}
		commonCount++
	}
	return strings.Join(splits[0][:commonCount], ".")
}
