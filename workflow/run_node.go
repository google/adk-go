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
	"fmt"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/internal/toolinternal"
)

// RunNodeOption configures a single RunNode call.
type RunNodeOption func(*runNodeOptions)

type runNodeOptions struct {
	customRunID            string
	useSubBranch           bool
	overrideBranch         string
	useAsOutput            bool
	overrideIsolationScope string
	scopeFromNodePath      bool
	raiseOnWait            bool
	// outcome, when non-nil, receives what the child's activation hands
	// back to the caller besides its output.
	outcome *runNodeOutcome
}

// runNodeOutcome is what a child leaves on its own context for the caller of
// RunNode.
type runNodeOutcome struct {
	skipSummarization bool
}

// WithRunID overrides the auto-generated counter with a stable
// user-supplied identifier — useful for reorderable lists keyed by
// e.g. an order id. id must be non-empty, contain at least one
// non-digit character (purely numeric ids collide with the
// auto-counter), and exclude the composite-path separators '/' and
// '@'. Violations surface as ErrInvalidRunID from RunNode. A call that
// runs without replay (from a tool context, or on a context without a
// sub-scheduler; see RunNode) also accepts a purely numeric id.
//
// A stable id deduplicates calls within one activation of the enclosing
// dynamic node: a repeat returns the first run's output without
// re-running the child, and calls that overlap in time share one
// execution and its outcome. A failed or interrupted run is not
// retained, so a later call runs the child again. Do not rely on this
// across a HITL pause: the resume turn re-executes the dynamic node's
// body, and a child that completed before the pause can run again.
//
// A child that is not a dynamic node must not call RunNode with its own
// name and id from inside its own run. The inner call resolves to the run
// already in progress, waits for it to finish, and so waits on itself
// until the invocation is canceled. A dynamic node can recurse this way,
// because its body schedules children under its own path.
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

// WithIsolationScope tags the child and its emitted events with scope,
// restricting the child's LLM history to matching events (see
// session.Event.IsolationScope). Empty means no scope.
func WithIsolationScope(scope string) RunNodeOption {
	return func(o *runNodeOptions) { o.overrideIsolationScope = scope }
}

// WithIsolationScopeFromNodePath scopes the child under its own node
// path. The full path (not just "<name>@<run_id>") keeps scopes unique
// across nested workflows and reused node names. This is the task-mode
// LlmAgent case. WithIsolationScope, if also set, takes precedence.
func WithIsolationScopeFromNodePath() RunNodeOption {
	return func(o *runNodeOptions) { o.scopeFromNodePath = true }
}

// WithRaiseOnWait makes RunNode treat a child that finishes its
// iteration with an unresolved long-running tool call as an interruption rather than a
// normal completion.
//
// Use this when the caller (e.g. a chat-mode coordinator
// dispatching a task sub-agent via a FunctionCall) MUST distinguish
// "child paused waiting for HITL" from "child finished cleanly
// with no output".
func WithRaiseOnWait() RunNodeOption {
	return func(o *runNodeOptions) { o.raiseOnWait = true }
}

// RunNode runs child as a sub-node of the node, tool or callback that ctx
// belongs to and returns its typed output.
//
// When ctx carries a sub-scheduler (the body of a dynamic node, the agent of
// an agent node and its tools), the child runs on it and its events leave
// through the node's iterator. A call from a tool context
// (ctx.FunctionCallID() != "") never replays: without WithRunID the child
// runs under run id "1", a numeric WithRunID is accepted, and no cached
// output or run in flight is shared with another call.
//
// When ctx carries none, the child runs once under a scheduler built for this
// call and not attached to ctx: run id "1" or the WithRunID value (numeric
// ids accepted), node path ctx.Path() + "/<child>@<id>" (no leading "/" when
// ctx.Path() is empty), no cached output. The child's events cannot reach the
// user: the state and artifact deltas of its non-partial events are merged
// into ctx.Actions() when it is non-nil, and a child that asks for input
// fails with an error wrapping ErrInvalidRunNodeContext. This serves a tool
// or callback context and a context outside any workflow that has a session.
// RunNode refuses, with ErrInvalidRunNodeContext and without running the
// child, the context a workflow node that is not a dynamic node runs under,
// and a context that is neither a tool context nor carries a session.
//
// When the child set Actions().SkipSummarization on its own context, RunNode
// sets it on ctx.Actions() before it returns, whether the child succeeded or
// failed.
//
// On failure:
//   - errors.Is(err, ErrNodeInterrupted): child paused for HITL. A tool
//     returns this error as it is, so its flow ends the step.
//   - errors.Is(err, ErrNodeFailed): child errored after retries;
//     errors.As recovers *NodeRunError with diagnostics (*NodeTimeoutError
//     for a timeout).
//   - ErrInputValidation inside a *NodeRunError: input violates the
//     child's input schema; the child does not run.
//   - ErrInvalidRunNodeContext, ErrInvalidRunID: misuse.
//   - ctx.Err(): parent cancellation.
//
// RunNode is safe for concurrent calls on one context.
func RunNode[OUT any](ctx agent.Context, child Node, input any, opts ...RunNodeOption) (OUT, error) {
	var zero OUT

	ss := ctx.SubScheduler()
	switch {
	case ss != nil:
		// A callback context, which logs when asked for FunctionCallID,
		// carries no sub-scheduler and so never reaches this check.
		if d, ok := ss.(*dynamicSubScheduler); ok && d.replay && ctx.FunctionCallID() != "" {
			ss = d.forToolCall()
		}
	case isActivationContext(ctx), isSessionless(ctx):
		return zero, ErrInvalidRunNodeContext
	default:
		ss = newFallbackSubScheduler(ctx)
	}

	var o runNodeOptions
	for _, opt := range opts {
		opt(&o)
	}

	var outcome runNodeOutcome
	o.outcome = &outcome
	rawOut, err := ss.RunNode(child, input, o)
	if outcome.skipSummarization {
		if actions := ctx.Actions(); actions != nil {
			callerActionsMu.Lock()
			actions.SkipSummarization = true
			callerActionsMu.Unlock()
		}
	}
	if err != nil {
		if errors.Is(err, ErrNodeInterrupted) {
			// A tool's flow ends its step on a pause rather than answering
			// the model with an error.
			return zero, &toolinternal.PausedError{Err: err}
		}
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
