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
	"errors"
	"time"

	"google.golang.org/adk/v2/agent"
	agentinternal "google.golang.org/adk/v2/internal/agent"
)

// nodeTimeout is the timeout of one attempt of one node. The zero value means
// the node has no timeout.
type nodeTimeout struct {
	name  string
	limit time.Duration // NodeConfig.Timeout; zero: no timeout
	own   bool          // the attempt's own deadline is earlier than every deadline of the parent context
	leaf  bool          // the node runs its own code: not *WorkflowNode, not *ParallelWorker, not an *AgentNode over a workflow agent
}

// withNodeTimeout bounds ctx by n's NodeConfig.Timeout. With Timeout <= 0 it
// returns ctx, a no-op cancel function and the zero nodeTimeout. Otherwise the
// context is ctx.WithAgentTimeout(Timeout) (no custom cause), and the caller
// must call the cancel function when the attempt ends.
func withNodeTimeout(ctx agent.Context, n Node) (agent.Context, context.CancelFunc, nodeTimeout) {
	limit := n.Config().Timeout
	if limit <= 0 {
		return ctx, func() {}, nodeTimeout{}
	}
	parentDeadline, parentHasDeadline := ctx.Deadline()
	bounded, cancel := ctx.WithAgentTimeout(limit)
	ownDeadline, _ := bounded.Deadline()
	_, isWF := n.(*WorkflowNode)
	_, isPW := n.(*ParallelWorker)
	return bounded, cancel, nodeTimeout{
		name:  n.Name(),
		limit: limit,
		own:   !parentHasDeadline || ownDeadline.Before(parentDeadline),
		leaf:  !isWF && !isPW && !isWorkflowAgentNode(n),
	}
}

// attribute returns the error to report for an attempt that ran under
// attemptCtx and ended with err (nil for a clean return). It returns err
// unchanged unless the attempt's own deadline passed and the node returned
// nothing or a deadline error, or a leaf node returned a deadline error of its
// own before any deadline passed; then it returns a *NodeTimeoutError for t.
//
// An error that already holds a *NodeTimeoutError is returned unchanged, so
// the innermost node whose timeout elapsed stays named. A deadline error that
// wraps ErrNodeFailed is a dynamic child's failure, not the node's own
// timeout.
func (t nodeTimeout) attribute(attemptCtx context.Context, err error) error {
	var nte *NodeTimeoutError
	if t.limit == 0 || errors.As(err, &nte) {
		return err
	}
	if attemptCtx.Err() == context.DeadlineExceeded && t.own && (err == nil || errors.Is(err, context.DeadlineExceeded)) {
		return &NodeTimeoutError{NodeName: t.name, Timeout: t.limit}
	}
	if attemptCtx.Err() == nil && t.leaf && errors.Is(err, context.DeadlineExceeded) && !errors.Is(err, ErrNodeFailed) {
		return &NodeTimeoutError{NodeName: t.name, Timeout: t.limit, cause: err}
	}
	return err
}

// isWorkflowAgentNode reports whether n is an *AgentNode over a workflow agent
// (built by workflowagent.New), which runs a graph of nodes rather than code
// of its own.
func isWorkflowAgentNode(n Node) bool {
	an, ok := n.(*AgentNode)
	if !ok {
		return false
	}
	ia, ok := an.agent.(agentinternal.Agent)
	return ok && agentinternal.Reveal(ia).AgentType == agentinternal.TypeWorkflowAgent
}
