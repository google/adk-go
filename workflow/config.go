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
	"time"

	"github.com/google/jsonschema-go/jsonschema"
)

// defaultRetryConfig is the default retry configuration for a node.
var defaultRetryConfig = RetryConfig{
	MaxAttempts:   5,
	InitialDelay:  time.Second,
	MaxDelay:      60 * time.Second,
	BackoffFactor: 2.0,
	Jitter:        1.0,
	ShouldRetry:   defaultShouldRetry,
}

func defaultShouldRetry(err error) bool {
	return !errors.Is(err, ErrInputValidation)
}

// DefaultRetryConfig returns a copy of the default retry policy
// (5 attempts, 1s initial delay, 60s cap, 2x backoff, full jitter).
// Every error is retried except ErrInputValidation, which a retry
// cannot fix because the input is deterministic. Override fields on
// the returned value to customize:
//
//	rc := workflow.DefaultRetryConfig()
//	rc.MaxAttempts = 10
//	cfg := workflow.NodeConfig{RetryConfig: rc}
func DefaultRetryConfig() *RetryConfig {
	return new(defaultRetryConfig)
}

// NodeConfig defines the configuration for a node.
//
// All fields are optional. The pointer-typed fields (RerunOnResume,
// WaitForOutput) are tri-state so node implementations can
// distinguish "user explicitly opted out" (&false) from "user did
// not configure" (nil).
type NodeConfig struct {
	// ParallelWorker, when true, runs the node concurrently for each
	// item of a list-typed input. The engine collects per-item
	// outputs and emits a single aggregate output event.
	ParallelWorker bool

	// RerunOnResume controls human-in-the-loop resume behavior:
	// &true re-runs the interrupted node from scratch on resume
	// (re-entry mode), &false routes the resume payload to the
	// node's successor as input (handoff mode), and nil defers to
	// the engine. The engine currently treats nil as handoff.
	RerunOnResume *bool

	// WaitForOutput, when true, parks the parent (ErrNodeInterrupted) to
	// re-run instead of completing it when a child dispatched via RunNode
	// finishes without setting an event Output — letting a multi-turn
	// child be retried rather than falsely completed with the zero value.
	// Only honored on the RunNode (dynamic sub-scheduler) path today. nil
	// means "use the engine default" — false for most node kinds.
	WaitForOutput *bool

	// RetryConfig, when non-nil, retries a failed attempt of this node
	// per the policy, in a graph and as a child started by RunNode (the
	// child's attempts share its path and run id; see RunNode). A node
	// whose output its output schema refuses fails that attempt like an
	// error the node returned, so the policy may run it again; the refused
	// output is never sent, and what the attempt wrote to state is. A
	// refusal of the node's input is not retried: the node does not start.
	// nil means "no retries".
	RetryConfig *RetryConfig

	// Timeout, when > 0, bounds each attempt of the node: when it
	// elapses, the attempt's context is done and the attempt fails
	// with a *NodeTimeoutError that names the node. Each attempt of a
	// node retried per RetryConfig gets the whole Timeout. It applies
	// wherever the node runs: in a graph, as a child started by
	// RunNode, and to each item of a ParallelWorker that wraps the
	// node. Zero (the default) or a negative value
	// means the node is bounded only by the parent context's deadline,
	// if any.
	Timeout time.Duration

	// EmitsOwnSpan, when true, tells the scheduler not to wrap the node
	// in an "invoke_node" telemetry span because the node body already
	// starts its own span (e.g. an LlmAgent node whose wrapped agent
	// emits "invoke_agent"). This keeps the span tree consistent with
	// the direct agent path and adk-python.
	EmitsOwnSpan bool

	// ContentInput, when true, declares that the node takes the user
	// message as a *genai.Content when it is a direct successor of
	// [Start]: [Workflow.Run] passes it the invocation's user content,
	// and [Workflow.Resume] passes it the invocation's first user
	// message again when it re-runs the node. When false (the zero
	// value), such a node receives the concatenated text of the
	// message's parts as a string, or nil when there is no user content.
	// The field has no effect on input from any other predecessor.
	//
	// [NewFunctionNode], [NewFunctionNodeWithSchema],
	// [NewEmittingFunctionNode], [NewEmittingFunctionNodeWithSchema],
	// [NewDynamicNode] and [NewDynamicNodeWithSchema] set ContentInput
	// when their input type is *genai.Content or genai.Content, and
	// [NewFunctionNodeFromState] sets it when the node-input field has
	// one of those types. A custom node that embeds [BaseNode] and reads
	// a *genai.Content from [Start] sets it in its NodeConfig.
	//
	// A [WorkflowNode] ignores ContentInput: as a successor of [Start] it
	// receives the seed unchanged, and its inner [Start] applies this
	// rule to its own successors.
	ContentInput bool

	// StateSchema, when non-nil, is the state schema of this node. While the node runs, a write through its
	// context's State().Set and the state delta of each event the node yields are checked against this schema
	// and no other: it replaces the schema of the enclosing workflow (see [WithStateSchema]) for this node,
	// and the two are never merged. A key declared only by the enclosing workflow's schema is undeclared here.
	// The checks are those of [WithStateSchema]: one entry at a time, a key that contains ':' is never
	// refused, a missing required field is never refused, and a refused write returns a [*StateSchemaError]
	// and writes nothing. Reads are never validated. An agent node's OutputKey is written on the node's event
	// and checked like any other key, so a schema that does not declare it fails that node, where adk-python
	// writes it unchecked.
	//
	// Every context derived from the node's context uses the same schema unless it belongs to a node that
	// carries its own: the dynamic children the node starts with RunNode, the nodes of a nested workflow that
	// has no schema of its own (a nested workflow with one, see [NewWorkflowNodeWithStateSchema], uses it), and
	// the items of a [ParallelWorker] (whose wrapped node's own schema wins over the worker's). The nodes that
	// follow this one in the graph are not derived from its context and are not affected. nil, the default,
	// means the node uses the schema it inherits, and no check at all when nothing above it has one.
	//
	// The schema must be resolved, its references must stay inside its own document, and it must not be
	// modified once any node uses it. One schema may be shared by any number of nodes and workflows. A zero
	// [jsonschema.Resolved] is a schema that declares no field: it refuses every key without a colon. The
	// node's Config must return the same StateSchema on every call. A schema that cannot be checked at run
	// time is reported as an error that matches [ErrStateSchemaUnusable] with errors.Is, and it is never
	// ignored or replaced by another schema: [New] returns it, naming the node, for a node of the graph;
	// [NewParallelWorker] returns it for its wrapped node; a dynamic child with such a schema fails when it
	// starts, with a [*NodeRunError] that wraps it; a node given to the runner as its root
	// (runner.Config.Node) is built into its workflow when the runner runs, so the error is the first Run's,
	// after the user message and before the node starts.
	StateSchema *jsonschema.Resolved
}

// RetryConfig defines the parameters for retrying a failed node.
//
// Recommended construction is via DefaultRetryConfig and override
// the fields you want to customize:
//
//	rc := workflow.DefaultRetryConfig()
//	rc.MaxAttempts = 10
//	cfg := workflow.NodeConfig{RetryConfig: rc}
//
// Constructing via struct literal (RetryConfig{...}) is permitted
// but discouraged: any unset field defaults to its zero value, not
// to DefaultRetryConfig's value. The zero RetryConfig is a valid
// "no retry, no backoff, no jitter" policy.
//
// A child started by RunNode reads the same fields in the same way as a
// node in a graph.
//
// The struct shape is deliberately a flat set of plain values: no
// dependency on cenkalti/backoff/v5.
type RetryConfig struct {
	// MaxAttempts is the maximum number of attempts, the first one
	// included. Zero, one or a negative value means a single attempt
	// and no retry, so the zero value never retries.
	// DefaultRetryConfig sets 5.
	MaxAttempts int

	// InitialDelay is the delay before the first retry; BackoffFactor
	// scales it for each later retry (see CalculateDelay). Zero means
	// no wait between attempts. DefaultRetryConfig sets 1 second.
	InitialDelay time.Duration

	// MaxDelay bounds every delay between attempts, jitter included.
	// Zero or a negative value means no bound. DefaultRetryConfig sets
	// 60 seconds.
	MaxDelay time.Duration

	// BackoffFactor is the per-attempt multiplier applied to the
	// delay. A factor of 1.0 means a constant InitialDelay between
	// retries; 2.0 means classic exponential backoff. Values between
	// 0 and 1.0 shrink the delay each attempt (rare but permitted).
	// Zero or a negative value counts as 1.0, so the zero value gives
	// a constant delay. DefaultRetryConfig sets 2.0.
	BackoffFactor float64

	// Jitter is a randomness factor in [0.0, 1.0]. The actual delay
	// is sampled from delay * (1 ± Jitter); with a MaxDelay the sample
	// never exceeds it, because the delay is capped at
	// MaxDelay / (1 + Jitter) before it is sampled. Zero means
	// deterministic delays. DefaultRetryConfig sets 1.0.
	Jitter float64

	// ShouldRetry reports whether a failed attempt is retried (true
	// means retry). It is called only while attempts remain under
	// MaxAttempts. nil retries every error except one that matches
	// ErrInputValidation (errors.Is), which a retry cannot fix;
	// DefaultRetryConfig sets the same predicate.
	ShouldRetry func(error) bool
}
