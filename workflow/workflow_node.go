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
	"fmt"
	"iter"
	"strings"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/session"
)

// WorkflowNode wraps a Workflow to be used as a Node.
type WorkflowNode struct {
	BaseNode
	subWorkflow *Workflow
}

// NewWorkflowNode creates a new node that runs a nested workflow.
// It uses the same arguments as New to construct the inner workflow.
func NewWorkflowNode(name string, edges []Edge) (*WorkflowNode, error) {
	wf, err := New(name, edges)
	if err != nil {
		return nil, err
	}
	return &WorkflowNode{
		BaseNode:    NewBaseNode(name, "", NodeConfig{}),
		subWorkflow: wf,
	}, nil
}

// Run executes the sub-workflow with the given input.
//
// The sub-workflow's output is the output of its terminal node(s) —
// nodes with no outgoing edges — selected by graph topology, not by
// event arrival order. Exactly one terminal output is forwarded as
// this node's output; more than one is a graph-design error
// (ErrMultipleOutputs); zero means the node produces no output.
// Mirrors adk-python's _set_ctx_output_or_interrupts.
func (n *WorkflowNode) Run(ctx context.Context, invCleanCtx agent.Context, input any) iter.Seq2[*session.Event, error] {
	terminals := n.subWorkflow.graph.terminalNodeNames()
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
		subCtx, cancel := context.WithCancel(ctx)
		defer cancel()
		ctx = subCtx

		for ev, err := range n.subWorkflow.RunNode(ctx, invCleanCtx, input) {
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
				for _, seg := range strings.Split(ev.NodeInfo.Path, "/") {
					name := seg
					if idx := strings.IndexByte(name, '@'); idx >= 0 {
						name = name[:idx]
					}
					if terminals[name] {
						terminalOutputs[name] = out
						break
					}
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
			// Forward with Output stripped (see the doc comment).
			evCopy := *ev
			evCopy.Output = nil
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
			event := session.NewEvent(invCleanCtx.InvocationID())
			event.Output = out
			if !yield(event, nil) {
				return
			}
		}
	}
}
