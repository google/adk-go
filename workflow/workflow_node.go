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
	"iter"

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
// It intercepts events from the sub-workflow to ensure that only the final
// output is yielded to the parent scheduler, preventing ErrMultipleOutputs
// if intermediate steps also produced outputs.
func (n *WorkflowNode) Run(ctx agent.Context, input any) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		var lastOutput any
		var pendingErr error

		// Create a cancellable context to signal the sub-workflow to stop on error or break.
		subCtx, cancel := context.WithCancel(ctx)
		defer cancel()

		for ev, err := range n.subWorkflow.RunNode(ctx.WithContext(subCtx), input) {
			if err != nil {
				pendingErr = err
				// Signal the sub-workflow to stop execution. We use 'continue' instead of 'break'
				// to allow the for-range loop to run to completion. This ensures the sub-workflow's scheduler
				// can finish draining its event queue (avoiding goroutine leaks) without triggering
				// Go 1.23's panic ("runtime error: range function continued iteration after function for loop body returned false").
				cancel()
				continue
			}

			if ev.Output != nil {
				lastOutput = ev.Output
				// Create a shallow copy and clear Output so the parent
				// scheduler doesn't fail on multiple outputs for this node.
				evCopy := *ev
				evCopy.Output = nil
				if !yield(&evCopy, nil) {
					// Same as above: cancel and drain to avoid leaks and panics.
					cancel()
					continue
				}
			} else {
				if !yield(ev, nil) {
					// Same as above: cancel and drain to avoid leaks and panics.
					cancel()
					continue
				}
			}
		}

		// Yield the error at the very end if one occurred.
		if pendingErr != nil {
			yield(nil, pendingErr)
			return
		}

		// Yield the final output at the end if we captured any.
		if lastOutput != nil {
			event := session.NewEvent(ctx.InvocationID())
			event.Output = lastOutput
			yield(event, nil)
		}
	}
}
