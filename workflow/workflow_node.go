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
	"fmt"
	"iter"
	"maps"
	"slices"
	"strings"

	"github.com/google/jsonschema-go/jsonschema"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/session"
)

// WorkflowNode wraps a Workflow to be used as a Node.
type WorkflowNode struct {
	BaseNode
	subWorkflow *Workflow
}

// NewWorkflowNode creates a new node that runs a nested workflow.
// It uses the same arguments as New to construct the inner workflow, and
// returns the errors New returns. An empty edge list gives a node that
// runs nothing and produces no output.
func NewWorkflowNode(name string, edges []Edge) (*WorkflowNode, error) {
	return newWorkflowNode(name, edges, nil)
}

// NewWorkflowNodeWithStateSchema is NewWorkflowNode whose nested workflow has the state schema schema, as
// if it were built with [WithStateSchema](schema): the nodes of the nested workflow, and the dynamic
// children and parallel items below them, are checked against schema unless a node carries its own
// [NodeConfig.StateSchema]. The enclosing node's schema is not consulted inside the nested workflow.
//
// It returns the errors of New(name, edges, WithStateSchema(schema)), unwrapped: a graph error, a
// state-bound FunctionNode parameter that schema does not declare, and, matching [ErrStateSchemaUnusable],
// a schema that cannot be checked at run time. A nil schema gives the node NewWorkflowNode returns.
//
// The node takes no input or output schema from this constructor. To give it both, set them on the node it
// returns, as NewWorkflowNodeWithSchemas does for its own: n.BaseNode = NewBaseNodeWithSchemas(name, "",
// NodeConfig{}, inputSchema, outputSchema). The state schema lives on the nested workflow and is not
// touched by that assignment. [NewWorkflowNodeWithOptions] takes the state schema together with the
// concurrency cap and the input and output schemas.
func NewWorkflowNodeWithStateSchema(name string, edges []Edge, schema *jsonschema.Resolved) (*WorkflowNode, error) {
	return newWorkflowNode(name, edges, schema)
}

// NewWorkflowNodeWithOptions is NewWorkflowNode whose nested workflow is built with the options opts
// takes in New, so a workflow nested in another one, or run as runner.Config.Node, takes the same
// options as a root workflow. With no options it returns the node NewWorkflowNode returns.
//
// [WithMaxConcurrency] caps the graph-scheduled nodes of the nested workflow: the cap counts that
// workflow's own nodes, the enclosing workflow counts the node as one of its nodes, and the enclosing
// workflow's cap does not reach the nested workflow's nodes. [WithStateSchema] checks the writes of the
// nested workflow's nodes and replaces the enclosing workflow's schema for them; without it they are
// checked against the enclosing workflow's schema. Both hold on every turn of a resumed run.
//
// [WithInputSchema] and [WithOutputSchema] become the node's schemas, as NewWorkflowNodeWithSchemas sets
// them: the enclosing workflow validates the node's input before the nested workflow starts and its
// single terminal output, with defaults filled, before the next node receives the coerced value. The
// nested workflow does not validate them again.
//
// The node resumes a paused or failed run of its nested workflow when it is itself runner.Config.Node or
// nested in a workflow under workflowagent.New. Under a root built with agent.New over [Workflow.Run],
// every turn starts the graph from Start.
//
// WithRootWrapper and WithNodeRootWrapper are for the workflows the runner builds: given either, in any
// position, it returns an error that names the node and builds nothing. Otherwise it returns the errors
// New returns, unwrapped. The returned node has an empty description and RerunOnResume set to true;
// assigning n.BaseNode = NewBaseNodeWithSchemas(name, description, cfg, inputSchema, outputSchema) sets
// them and leaves the nested workflow's cap and state schema as they are.
func NewWorkflowNodeWithOptions(name string, edges []Edge, opts ...Option) (*WorkflowNode, error) {
	var o workflowOptions
	for _, opt := range opts {
		opt(&o)
	}
	if o.isRootWrapper || o.isNodeRootWrapper {
		return nil, fmt.Errorf("workflow: node %q: WithRootWrapper and WithNodeRootWrapper are for the runner and do not apply to a WorkflowNode", name)
	}
	// The input and output schemas are validated by the enclosing workflow on the node, so that the
	// next node receives the coerced output; the nested workflow is built without them.
	inner := append(slices.Clone(opts), WithInputSchema(nil), WithOutputSchema(nil))
	wf, err := New(name, edges, inner...)
	if err != nil {
		return nil, err
	}
	return &WorkflowNode{
		BaseNode:    NewBaseNodeWithSchemas(name, "", NodeConfig{RerunOnResume: new(true)}, o.inputSchema, o.outputSchema),
		subWorkflow: wf,
	}, nil
}

// newWorkflowNode builds the nested workflow with stateSchema, which may be nil, and wraps it in a node
// with an empty NodeConfig.
func newWorkflowNode(name string, edges []Edge, stateSchema *jsonschema.Resolved) (*WorkflowNode, error) {
	wf, err := New(name, edges, WithStateSchema(stateSchema))
	if err != nil {
		return nil, err
	}
	rerun := true
	return &WorkflowNode{
		BaseNode:    NewBaseNode(name, "", NodeConfig{RerunOnResume: &rerun}),
		subWorkflow: wf,
	}, nil
}

// NewWorkflowNodeWithSchemas is NewWorkflowNode with an input and an output schema on the node.
//
// It takes the arguments and returns the errors of NewWorkflowNode. Either
// schema may be nil, which means no validation on that side; both must
// already be resolved. The returned node's InputSchema and OutputSchema
// return them.
//
// The enclosing workflow validates the node like any other node. Its
// input is validated before the nested workflow starts, so a failure
// wraps [ErrInputValidation] and no inner node runs; as a successor of
// [Start] the node validates the user message as [WithInputSchema]
// describes. Its single terminal output is validated, with defaults
// filled, before the next node receives it, and a failure wraps
// [ErrNodeFailed].
//
// [NewWorkflowNodeWithOptions] with [WithInputSchema] and [WithOutputSchema] builds the same node and
// takes the nested workflow's other options too.
func NewWorkflowNodeWithSchemas(name string, edges []Edge, inputSchema, outputSchema *jsonschema.Resolved) (*WorkflowNode, error) {
	n, err := NewWorkflowNode(name, edges)
	if err != nil {
		return nil, err
	}
	n.BaseNode = NewBaseNodeWithSchemas(name, "", NodeConfig{RerunOnResume: new(true)}, inputSchema, outputSchema)
	return n, nil
}

// Run executes the sub-workflow with the given input.
//
// The sub-workflow's output is the output of its terminal node(s) —
// nodes with no outgoing edges — selected by graph topology, not by
// event arrival order. Exactly one terminal output is forwarded as
// this node's output; more than one is a graph-design error
// (ErrMultipleOutputs); zero means the node produces no output.
// Mirrors adk-python's _set_ctx_output_or_interrupts.
//
// As a successor of [Start], the node receives the outer workflow's
// START seed unchanged and passes it to the inner workflow, whose own
// [Start] gives each inner successor the form it declares (see
// [NodeConfig.ContentInput]).
func (n *WorkflowNode) Run(ctx agent.Context, input any) iter.Seq2[*session.Event, error] {
	terminals := n.subWorkflow.graph.terminalNodeNames()
	nodesByName := buildNodesByName(n.subWorkflow.graph)
	return func(yield func(*session.Event, error) bool) {
		// terminalOutputs is keyed by terminal node name (last write
		// wins), so re-running a terminal via loop-back does not inflate
		// the count.
		terminalOutputs := make(map[string]any)
		var pendingErr error
		// consumerGone becomes true once the parent yield returns false. After
		// that, calling yield again panics the iterator (Go 1.23 range-over-func
		// contract), so we only keep draining the sub-workflow to avoid leaks.
		consumerGone := false

		// Create a cancellable context to signal the sub-workflow to stop on error or break.
		subCtx, cancel := ctx.WithAgentCancel()
		defer cancel()
		ctx = ctx.WithAgentContext(subCtx)

		var subEvents iter.Seq2[*session.Event, error]
		if state, responses, err := n.resumeState(ctx, input); err != nil {
			yield(nil, err)
			return
		} else if len(responses) > 0 {
			for name := range terminals {
				if ns := state.Nodes[name]; ns != nil && ns.Status == NodeCompleted && ns.Output != nil {
					terminalOutputs[name] = ns.Output
				}
			}
			subEvents = n.subWorkflow.Resume(ctx, state, responses)
		} else {
			subEvents = n.subWorkflow.RunNode(ctx, input)
		}

		for ev, err := range subEvents {
			if err != nil {
				pendingErr = err
				// Signal the sub-workflow to stop execution. We use 'continue' instead of 'break'
				// to allow the for-range loop to run to completion. This ensures the sub-workflow's scheduler
				// can finish draining its event queue (avoiding goroutine leaks) without triggering
				// Go 1.23's panic ("runtime error: range function continued iteration after function for loop body returned false").
				cancel()
				continue
			}

			// Capture the output only from terminal nodes. childEventOutput
			// matches the scheduler: Event.Output, or model text when
			// MessageAsOutput is set.
			if out, ok := childEventOutput(ev); ok && ev.NodeInfo != nil {
				for name := range strings.SplitSeq(ev.NodeInfo.Path, "/") {
					if idx := strings.LastIndexByte(name, '@'); idx >= 0 {
						name = name[:idx]
					}
					if terminals[name] {
						terminalOutputs[name] = out
						break
					}
				}
			} else if out, delegated := completedDelegatedOutput(ev); delegated {
				owner := ownEventNodeName(graphHistoryEvent(ev, n.subWorkflow.Name()), nodesByName)
				if terminals[owner] && out != nil {
					terminalOutputs[owner] = out
				}
			}

			if consumerGone {
				// Drain remaining events without yielding.
				continue
			}

			if ev.Output == nil {
				if !yield(ev, nil) {
					consumerGone = true
					cancel()
				}
				continue
			}
			// Forward with Output stripped (see the doc comment), keeping the
			// inner node's output in sub-workflow metadata so ReconstructRunState
			// can restore completed predecessors when the sub-workflow resumes.
			evCopy := *ev
			evCopy.Output = nil
			if n.subWorkflow.Name() != "" {
				meta := maps.Clone(ev.CustomMetadata)
				if meta == nil {
					meta = map[string]any{}
				}
				meta[workflowSubNodeOutputPrefix+n.subWorkflow.Name()] = ev.Output
				evCopy.CustomMetadata = meta
			}
			// The inner runner checked ev; the copy carries that over, so the
			// enclosing runner does not check it against its own schema.
			if f := forwardedFrom(ctx); f != nil && f.take(ev) {
				f.mark(&evCopy)
			}
			if !yield(&evCopy, nil) {
				consumerGone = true
				cancel()
			}
		}

		// The consumer already broke the range loop; yielding again would panic.
		if consumerGone {
			return
		}

		// Yield the error at the very end if one occurred.
		if pendingErr != nil {
			yield(nil, pendingErr)
			return
		}

		if len(terminalOutputs) > 1 {
			yield(nil, fmt.Errorf("%w: sub-workflow %q has %d terminal nodes producing output; a workflow must have at most one terminal output",
				ErrMultipleOutputs, n.subWorkflow.Name(), len(terminalOutputs)))
			return
		}

		// Yield the terminal output at the end if one was produced.
		for _, out := range terminalOutputs {
			event := session.NewEvent(ctx, ctx.InvocationID())
			event.Output = out
			if !yield(event, nil) {
				return
			}
		}
	}
}

func (n *WorkflowNode) resumeState(ctx agent.Context, input any) (*RunState, map[string]any, error) {
	if ctx.Session() == nil {
		return nil, nil, nil
	}
	state, err := n.subWorkflow.reconstructRunState(ctx.Session(), ctx.InvocationID(), input, true)
	if err != nil {
		return nil, nil, err
	}
	if state == nil {
		return nil, nil, nil
	}
	var responses map[string]any
	for _, ns := range state.Nodes {
		if ns == nil {
			continue
		}
		for _, id := range ns.Interrupts {
			if v, ok := ctx.ResumedInput(id); ok {
				if responses == nil {
					responses = map[string]any{}
				}
				responses[id] = v
			}
		}
		for id, validated := range ns.ResumedInputs {
			if _, ok := ctx.ResumedInput(id); ok {
				if responses == nil {
					responses = map[string]any{}
				}
				responses[id] = validated
			}
		}
	}
	return state, responses, nil
}
