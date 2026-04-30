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
	"time"
	"unicode"

	"google.golang.org/adk/session"
)

// Node is the unit of execution inside a Workflow graph. Every wrapper
// (FunctionNode, JoinNode, ToolNode, ParallelWorker, LlmAgentNode, and
// Workflow itself) implements this interface.
//
// External users typically construct nodes via the helper constructors
// (Func, FuncStream, Join, Tool, Parallel, FromAgent) rather than
// implementing this interface directly.
//
// The interface is sealed via baseTag(): only types embedding Base can
// satisfy it. This avoids accidental "any value with the right method set"
// implementations and keeps node serialization predictable.
type Node interface {
	Name() string
	Description() string

	// Spec returns the static configuration of the node — schemas, retry,
	// timeout, resume / wait-for-output flags, predecessor requirement.
	Spec() NodeSpec

	// RunImpl executes the node. The engine wraps the call in a per-node
	// goroutine and forwards every event pushed through em into the
	// workflow's iter.Seq2.
	//
	// Implementations push results via:
	//   - em.Event(*session.Event)   — raw event passthrough
	//   - em.Output(any)             — produces an event whose NodeInfo.Output is set
	//   - em.RequestInput(...)       — pauses the node for a HITL response
	//   - em.StateDelta(...) / em.ArtifactDelta(...) — state/artifact mutations
	//
	// Returning a non-nil error marks the node FAILED; the engine applies
	// the node's RetryConfig before propagating the error to the workflow.
	//
	// The ctx field NodeContext embeds agent.InvocationContext so callers
	// have access to session, services, branch, and user content.
	RunImpl(ctx *NodeContext, input any, em EventEmitter) error

	// baseTag is an internal seal. Only types embedding Base satisfy it.
	baseTag() *Base
}

// EventEmitter is what RunImpl uses to push results into the parent
// workflow's event stream.
type EventEmitter interface {
	Event(*session.Event) error
	Output(value any) error
	RequestInput(req RequestInput) error
	StateDelta(delta map[string]any) error
	ArtifactDelta(delta map[string]int64) error
}

// NodeSpec captures the static configuration of a node.
//
// All fields are optional; the zero value is valid (no schemas, no retries,
// no timeout, eager / non-resuming behavior).
type NodeSpec struct {
	// InputSchema, when non-nil, is consulted by the node runner before
	// dispatching input to RunImpl. Validation failures mark the node FAILED.
	InputSchema Schema

	// OutputSchema, when non-nil, validates whatever the node yields via
	// em.Output(...). Validation failures mark the node FAILED.
	OutputSchema Schema

	// StateSchema, when non-nil, is propagated to child contexts and
	// validated on every state mutation. Prefixed keys ("app:", "user:",
	// "temp:") bypass validation, mirroring adk-python.
	StateSchema Schema

	// RetryConfig, when non-nil, configures retries around RunImpl.
	RetryConfig *RetryConfig

	// Timeout, when > 0, caps a single attempt's runtime. Zero means no
	// timeout.
	Timeout time.Duration

	// RerunOnResume controls behavior when resuming after an interrupt.
	// true: rerun the node from scratch.
	// false: complete immediately using the user's resume input as output.
	RerunOnResume bool

	// WaitForOutput, when true, leaves the node WAITING after RunImpl
	// returns until em.Output is called. Useful for nodes (e.g. JoinNode)
	// that must run multiple times before producing a final output.
	WaitForOutput bool

	// RequiresAllPredecessors, when true, instructs the engine to wait for
	// every predecessor edge to fire before running this node (fan-in
	// JoinNode semantic).
	RequiresAllPredecessors bool
}

// Base is the embeddable struct used by every node wrapper.
//
// Embed Base in a struct to satisfy the Node interface. The wrapper still
// has to implement RunImpl explicitly; Base provides the boilerplate
// metadata accessors.
//
//	type myNode struct {
//	    workflow.Base
//	    // …extra fields…
//	}
//	func (m *myNode) RunImpl(ctx *workflow.NodeContext, in any, em workflow.EventEmitter) error { … }
type Base struct {
	name        string
	description string
	spec        NodeSpec
}

// SetMetadata initializes Base. Call from the wrapper's constructor before
// the node is exposed to the engine.
func (b *Base) SetMetadata(name, description string, spec NodeSpec) error {
	if err := validateNodeName(name); err != nil {
		return err
	}
	b.name = name
	b.description = description
	b.spec = spec
	return nil
}

// Name implements Node.
func (b *Base) Name() string { return b.name }

// Description implements Node.
func (b *Base) Description() string { return b.description }

// Spec implements Node.
func (b *Base) Spec() NodeSpec { return b.spec }

// baseTag implements the seal.
func (b *Base) baseTag() *Base { return b }

// validateNodeName mirrors adk-python's BaseNode.name validator: must be a
// valid Go identifier (letter / underscore start, then letters/digits/
// underscores). Rejects empty names.
func validateNodeName(name string) error {
	if name == "" {
		return errors.New("workflow: node name must not be empty")
	}
	for i, r := range name {
		if i == 0 {
			if !unicode.IsLetter(r) && r != '_' {
				return fmt.Errorf("workflow: node name %q must start with a letter or underscore", name)
			}
			continue
		}
		if !unicode.IsLetter(r) && !unicode.IsDigit(r) && r != '_' {
			return fmt.Errorf("workflow: node name %q must contain only letters, digits, and underscores", name)
		}
	}
	return nil
}

// NodeOpt customizes node construction via functional options. Used by
// every wrapper constructor (Func, Join, Tool, Parallel, FromAgent) so
// fields can be set without bloating each constructor's signature.
type NodeOpt func(*nodeOpts)

type nodeOpts struct {
	description             string
	inputSchema             Schema
	outputSchema            Schema
	stateSchema             Schema
	retryConfig             *RetryConfig
	timeout                 time.Duration
	rerunOnResume           bool
	waitForOutput           bool
	requiresAllPredecessors bool
}

func applyOpts(opts []NodeOpt) nodeOpts {
	var o nodeOpts
	for _, opt := range opts {
		opt(&o)
	}
	return o
}

func (o nodeOpts) toSpec() NodeSpec {
	return NodeSpec{
		InputSchema:             o.inputSchema,
		OutputSchema:            o.outputSchema,
		StateSchema:             o.stateSchema,
		RetryConfig:             o.retryConfig,
		Timeout:                 o.timeout,
		RerunOnResume:           o.rerunOnResume,
		WaitForOutput:           o.waitForOutput,
		RequiresAllPredecessors: o.requiresAllPredecessors,
	}
}

// WithDescription sets the node's human-readable description.
func WithDescription(desc string) NodeOpt {
	return func(o *nodeOpts) { o.description = desc }
}

// WithInputSchema attaches an input schema. The engine validates input
// data against this schema before invoking the node.
func WithInputSchema(s Schema) NodeOpt {
	return func(o *nodeOpts) { o.inputSchema = s }
}

// WithOutputSchema attaches an output schema. The engine validates the
// node's output before forwarding it to downstream nodes.
func WithOutputSchema(s Schema) NodeOpt {
	return func(o *nodeOpts) { o.outputSchema = s }
}

// WithStateSchema attaches a state schema. State mutations made through
// the node's NodeContext are validated against this schema.
func WithStateSchema(s Schema) NodeOpt {
	return func(o *nodeOpts) { o.stateSchema = s }
}

// WithRetry sets a retry policy for the node.
func WithRetry(cfg *RetryConfig) NodeOpt {
	return func(o *nodeOpts) { o.retryConfig = cfg }
}

// WithTimeout caps a single attempt's runtime.
func WithTimeout(d time.Duration) NodeOpt {
	return func(o *nodeOpts) { o.timeout = d }
}

// WithRerunOnResume sets the rerun-on-resume flag (default false).
func WithRerunOnResume(v bool) NodeOpt {
	return func(o *nodeOpts) { o.rerunOnResume = v }
}

// WithWaitForOutput sets the wait-for-output flag (default false).
func WithWaitForOutput(v bool) NodeOpt {
	return func(o *nodeOpts) { o.waitForOutput = v }
}
