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

import "fmt"

// RunNodeOption configures a single RunNode call.
type RunNodeOption func(*runNodeOptions)

type runNodeOptions struct {
	customRunID    string
	useSubBranch   bool
	overrideBranch string
	useAsOutput    bool
}

// WithRunID overrides the auto-generated counter with a stable
// user-supplied identifier — useful for reorderable lists keyed by
// e.g. an order id. id must be non-empty, contain at least one
// non-digit character (purely numeric ids collide with the
// auto-counter), and exclude the composite-path separators '/' and
// '@'. Violations surface as ErrInvalidRunID from RunNode.
//
// Mirrors adk-python's run_id kwarg
// (https://adk.dev/graphs/dynamic/#custom-execution-ids).
func WithRunID(id string) RunNodeOption {
	return func(o *runNodeOptions) { o.customRunID = id }
}

// WithUseSubBranch derives a per-child sub-branch of the form
// "<parentBranch>.<childName>@<runID>" (or just "<childName>@<runID>"
// at root) for the child activation.
//
// Use this when the caller runs multiple concurrent or independent
// children that should not see each other's events in their LLM
// prompt history. Without it, every RunNode child inherits the
// orchestrator's branch and an LlmAgent child would see sibling
// events through the LLM flow's history filter.
//
// Combinable with WithOverrideBranch: the override sets the base,
// and the sub-branch segment is appended to it.
func WithUseSubBranch() RunNodeOption {
	return func(o *runNodeOptions) { o.useSubBranch = true }
}

// WithUseAsOutput promotes this child's output to the parent
// dynamic node's terminal Output, discarding the value returned by
// the orchestrator body. At most one delegating child per parent
// activation; a second one fails with ErrOutputAlreadyDelegated.
func WithUseAsOutput() RunNodeOption {
	return func(o *runNodeOptions) { o.useAsOutput = true }
}

// WithOverrideBranch replaces the inherited branch verbatim,
// regardless of WithUseSubBranch. Useful for nested dispatch
// patterns (e.g. chat coordinator → task agent) where the parent
// assigns a specific branch label by convention.
//
// Combinable with WithUseSubBranch: the override sets the base,
// and the sub-branch segment "<childName>@<runID>" is appended.
//
// Empty branch is treated as "no override". To force root, pass
// WithUseSubBranch() alone, which derives a fresh sub-branch off
// root.
func WithOverrideBranch(branch string) RunNodeOption {
	return func(o *runNodeOptions) { o.overrideBranch = branch }
}

// RunNode schedules child as a sub-node of the currently-executing
// dynamic node and returns its typed output. ctx must be the
// NodeContext passed into the enclosing dynamic node's body.
//
// On failure:
//   - errors.Is(err, ErrNodeInterrupted): child paused for HITL.
//   - errors.Is(err, ErrNodeFailed): child errored after retries;
//     errors.As recovers *NodeRunError with diagnostics.
//   - ErrInvalidRunNodeContext, ErrInvalidRunID: misuse.
//   - ctx.Err(): parent cancellation.
func RunNode[OUT any](ctx NodeContext, child Node, input any, opts ...RunNodeOption) (OUT, error) {
	var zero OUT

	nc, ok := ctx.(*nodeContext)
	if !ok || nc.subScheduler == nil {
		return zero, ErrInvalidRunNodeContext
	}

	var o runNodeOptions
	for _, opt := range opts {
		opt(&o)
	}

	rawOut, err := nc.subScheduler.runNode(child, input, o)
	if err != nil {
		return zero, err
	}
	if rawOut == nil {
		return zero, nil
	}
	typed, ok := rawOut.(OUT)
	if !ok {
		return zero, fmt.Errorf("workflow.RunNode: child %q output type %T does not satisfy expected %T",
			child.Name(), rawOut, zero)
	}
	return typed, nil
}
