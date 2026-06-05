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

	"google.golang.org/adk/agent"
)

// NodeContext is the per-node context seen inside Node.Run and inside
// dynamic-node orchestrator bodies. It extends agent.InvocationContext
// with workflow-specific accessors.
//
// TODO(wolo): unify with the in-flight context-unification work
// (CallbackContext / ToolContext series).
type NodeContext interface {
	agent.InvocationContext

	// ResumedInput returns the response payload for a re-entry resume
	// activation keyed by InterruptID, or (nil, false) otherwise.
	ResumedInput(interruptID string) (any, bool)

	// Path returns the composite path of the currently-executing node.
	// Empty for top-level static nodes; "<parent_path>/<child_name>@<run_id>"
	// for dynamic children.
	Path() string

	// RunID returns the per-invocation identifier. Empty for top-level
	// static nodes; auto-counter or user-supplied via WithRunID for
	// dynamic children.
	RunID() string

	// WithBranch returns a NodeContext whose Branch() returns the
	// given value; all other fields (path, runID, subScheduler,
	// resumeInputs, embedded InvocationContext) are preserved.
	WithBranch(branch string) NodeContext
}

// nodeContext is the unexported NodeContext implementation.
type nodeContext struct {
	agent.InvocationContext

	// resumeInputs are keyed by InterruptID. Nil on fresh activations
	// and on handoff resume.
	resumeInputs map[string]any

	// path and runID are populated for dynamic children, empty for
	// top-level static activations.
	path  string
	runID string

	// subScheduler is non-nil only when this context belongs to a
	// dynamic-node activation; RunNode uses it to schedule children.
	subScheduler *dynamicSubScheduler

	// outputForAncestors are the extra node paths this activation's
	// output counts for, set when this activation is a WithUseAsOutput
	// child so its own delegating children can extend the chain.
	outputForAncestors []string
}

// Compile-time: *nodeContext implements NodeContext.
var _ NodeContext = (*nodeContext)(nil)

// newNodeContext wraps parent for a top-level (static) activation.
func newNodeContext(parent agent.InvocationContext, resumeInputs map[string]any) *nodeContext {
	return &nodeContext{
		InvocationContext: parent,
		resumeInputs:      resumeInputs,
	}
}

// newDynamicNodeContext wraps parent for either a dynamic-node
// activation or one of its children, attaching path, runID, and the
// sub-scheduler RunNode reaches from the orchestrator body. Children
// pass the sub-scheduler's counter (or WithRunID) value as runID; a
// dynamic node's own activation passes runID="" — it is not itself a
// sub-scheduler child. Child inherits resumeInputs so HITL responses
// reach dynamic children.
func newDynamicNodeContext(parent NodeContext, path, runID string, sub *dynamicSubScheduler, outputForAncestors []string) *nodeContext {
	var inherited map[string]any
	if p, ok := parent.(*nodeContext); ok {
		inherited = p.resumeInputs
	}
	return &nodeContext{
		InvocationContext:  parent,
		resumeInputs:       inherited,
		path:               path,
		runID:              runID,
		subScheduler:       sub,
		outputForAncestors: outputForAncestors,
	}
}

func (c *nodeContext) ResumedInput(interruptID string) (any, bool) {
	if c.resumeInputs == nil {
		return nil, false
	}
	v, ok := c.resumeInputs[interruptID]
	return v, ok
}

func (c *nodeContext) Path() string  { return c.path }
func (c *nodeContext) RunID() string { return c.runID }

func (c *nodeContext) WithBranch(branch string) NodeContext {
	// Reuse the package-level withBranch helper to swap Branch on
	// the underlying InvocationContext; preserve the NodeContext
	// envelope (path, runID, resumeInputs, subScheduler) unchanged.
	return &nodeContext{
		InvocationContext:  withBranch(c.InvocationContext, branch),
		resumeInputs:       c.resumeInputs,
		path:               c.path,
		runID:              c.runID,
		subScheduler:       c.subScheduler,
		outputForAncestors: c.outputForAncestors,
	}
}

// WithContext preserves the nodeContext wrapper when callers derive
// a new context from this one (e.g. when the scheduler attaches an
// OpenTelemetry span context). Without this override, the base
// invocationContext.WithContext would return a *invocationContext
// and silently drop the resumeInputs map, breaking re-entry resume
// activations and any other workflow-specific accessors.
func (c *nodeContext) WithContext(ctx context.Context) agent.InvocationContext {
	return &nodeContext{
		c.InvocationContext.WithContext(ctx),
		c.resumeInputs,
		c.path,
		c.runID,
		c.subScheduler,
		c.outputForAncestors,
	}
}
