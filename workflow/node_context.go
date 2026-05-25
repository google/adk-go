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

// NodeContext is the per-node context seen inside Node.Run, and (in the
// upcoming dynamic-workflows API) inside dynamic-node orchestrator
// function bodies. It extends agent.InvocationContext with
// workflow-specific accessors.
//
// The interface shape is forward-compatible with the in-flight
// context-unification work (see adk-go PRs in the ToolContext /
// CallbackContext series): like those, NodeContext is an interface
// whose concrete implementation stays unexported, so that future
// unification can swap the backing type without breaking users.
type NodeContext interface {
	agent.InvocationContext

	// ResumedInput returns the response payload associated with the
	// given InterruptID for a re-entry resume activation. Returns
	// (nil, false) on fresh activations, on handoff resume, and when
	// the InterruptID does not match any payload supplied by the
	// resuming caller.
	ResumedInput(interruptID string) (any, bool)

	// Path returns the composite path of the currently-executing
	// node. For top-level static nodes it is the empty string. For
	// dynamic children scheduled via the upcoming workflow.RunNode
	// API it will be "<parent_path>/<child_name>@<run_id>"; until
	// that API lands, Path always returns "".
	Path() string

	// RunID returns the per-invocation identifier of the current
	// node. Empty for top-level static activations. Populated by the
	// dynamic sub-scheduler when this is a dynamic child.
	RunID() string

	// WithGoCtx returns a NodeContext whose underlying
	// context.Context is replaced by ctx. Session, invocation
	// metadata, and resume inputs are preserved. Used to integrate
	// with errgroup, http request scopes, and other code that
	// expects to drive cancellation via a *context.Context.
	WithGoCtx(ctx context.Context) NodeContext
}

// nodeContext is the unexported NodeContext implementation. It wraps
// the workflow's incoming agent.InvocationContext and carries the
// resume payloads the scheduler injected for a re-entry activation.
type nodeContext struct {
	agent.InvocationContext

	// resumeInputs carries the user-supplied response payloads for
	// a re-entry resume activation, keyed by InterruptID. Nil on
	// fresh activations and on handoff resume (where the response
	// flows to the successor as its input rather than back to the
	// asker via this map).
	resumeInputs map[string]any
}

// Compile-time assertion that *nodeContext satisfies NodeContext.
var _ NodeContext = (*nodeContext)(nil)

// newNodeContext returns a nodeContext wrapping parent. resumeInputs
// is nil for non-resume activations.
func newNodeContext(parent agent.InvocationContext, resumeInputs map[string]any) *nodeContext {
	return &nodeContext{
		InvocationContext: parent,
		resumeInputs:      resumeInputs,
	}
}

// ResumedInput returns the response payload associated with the
// given InterruptID for a re-entry resume activation. Returns
// (nil, false) on fresh activations, on handoff resume, and when
// the InterruptID does not match any payload supplied by the
// resuming caller.
func (c *nodeContext) ResumedInput(interruptID string) (any, bool) {
	if c.resumeInputs == nil {
		return nil, false
	}
	v, ok := c.resumeInputs[interruptID]
	return v, ok
}

// Path implements NodeContext. Returns "" today; populated by the
// dynamic sub-scheduler in a future change.
func (c *nodeContext) Path() string {
	return ""
}

// RunID implements NodeContext. Returns "" today; populated by the
// dynamic sub-scheduler in a future change.
func (c *nodeContext) RunID() string {
	return ""
}

// WithGoCtx returns a NodeContext whose underlying context.Context is
// replaced by ctx, delegating to InvocationContext.WithContext and
// preserving resumeInputs.
func (c *nodeContext) WithGoCtx(ctx context.Context) NodeContext {
	return &nodeContext{
		InvocationContext: c.InvocationContext.WithContext(ctx),
		resumeInputs:      c.resumeInputs,
	}
}
