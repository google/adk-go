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
	"google.golang.org/adk/agent"
	"google.golang.org/adk/session"
)

// NodeContext is the per-node execution context. It embeds
// agent.InvocationContext so callers have access to session, services,
// branch, and user content, and adds workflow-specific state.
//
// Phase 2A ships only the type so the engine, NodeRunner, and dynamic
// scheduler can reference it. Methods land alongside the orchestration
// loop in 2C-2D.
type NodeContext struct {
	agent.InvocationContext

	// nodePath is the hierarchical address of this node, e.g.
	// "wf@1/classify@1". Empty for the workflow's own RunImpl.
	nodePath string

	// runID is the per-name run counter ("1", "2", …). Combined with the
	// node's name to form the trailing segment of nodePath.
	runID string

	// resumeInputs maps interrupt IDs to user responses, populated when
	// the workflow is being resumed from a HITL interrupt. (Phase 2 doesn't
	// implement resume; field reserved for Phase 4-5.)
	resumeInputs map[string]any

	// actions accumulates state and artifact deltas emitted via the
	// EventEmitter; the engine flushes them onto outgoing events.
	actions *session.EventActions

	// scheduler is the dynamic-node scheduler for this context. It is
	// created lazily on the first RunNode call.
	scheduler *dynamicScheduler

	// parentEmitter is the EventEmitter the orchestrator handed to the
	// surrounding RunImpl. RunNode forwards child events through it so
	// they reach the workflow's iter.Seq2.
	parentEmitter EventEmitter
}

// NodePath returns the node's hierarchical address.
func (c *NodeContext) NodePath() string { return c.nodePath }

// RunID returns the per-name run counter as a string.
func (c *NodeContext) RunID() string { return c.runID }

// Actions returns the mutable EventActions accumulator. Direct mutation is
// allowed; the EventEmitter wraps these for convenience.
func (c *NodeContext) Actions() *session.EventActions {
	if c.actions == nil {
		c.actions = &session.EventActions{
			StateDelta:    map[string]any{},
			ArtifactDelta: map[string]int64{},
		}
	}
	return c.actions
}

// ResumeInput returns the user's response to a prior RequestInput, keyed
// by interrupt ID. Returns (nil, false) when no response is available
// (the node was never interrupted with that ID, or this is the first run).
//
// Mirrors adk-python's ctx.resume_inputs[interrupt_id] access pattern.
func (c *NodeContext) ResumeInput(interruptID string) (any, bool) {
	if c.resumeInputs == nil {
		return nil, false
	}
	v, ok := c.resumeInputs[interruptID]
	return v, ok
}

// RequestInput represents a human-in-the-loop interrupt yielded from a
// node. The engine converts it into a session.Event carrying a
// FunctionCall named "adk_request_input" and pauses the node until the
// user supplies a matching FunctionResponse.
//
// Phase 2A only declares the shape so callers can reference it; the
// engine-side conversion lands in Phase 5.
type RequestInput struct {
	// Prompt is the message shown to the user.
	Prompt string

	// ResponseSchema, when non-nil, validates the user's response on
	// resume. Must be JSON-serializable so the schema can travel through
	// session events.
	ResponseSchema Schema

	// InterruptID uniquely identifies this interrupt. Defaults to a fresh
	// UUID when left empty.
	InterruptID string
}
