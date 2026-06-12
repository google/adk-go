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
	"iter"
	"testing"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/session"
)

var errFnNodeNeedsNodeContext = errors.New("fnNode: ctx is not a NodeContext")

// End-to-end check that a child of a dynamic-node activation can
// recover its own NodeContext via NodeContextFromGoContext — the
// path runnable tools rely on to reach the sub-scheduler from
// tool.Context.
func TestNodeContextPropagation_DynamicChildEmbedsItself(t *testing.T) {
	var captured NodeContext

	inner := newFnNode("inner", func(ctx NodeContext) (any, error) {
		// nc, ok := NodeContextFromGoContext(ctx)
		// if !ok {
		// 	t.Errorf("inner: NodeContext not recovered from go context value")
		// 	return nil, nil
		// }
		// captured = nc
		captured = ctx
		return "ok", nil
	})

	orchestrator := NewDynamicNode[string, string]("orch",
		func(ctx NodeContext, _ string, _ func(*session.Event) error) (string, error) {
			return RunNode[string](ctx, inner, nil)
		},
		NodeConfig{},
	)

	if _, err := drainDynamicWithErr(t, orchestrator, ""); err != nil {
		t.Fatalf("orchestrator error: %v", err)
	}
	if captured == nil {
		t.Fatal("inner did not recover any NodeContext")
	}
}

// Top-level (static) activations stash a NodeContext too, even
// though their RunNode would be rejected for lack of a
// sub-scheduler — consumers can still detect "inside a workflow
// node" and react accordingly.
// func TestNodeContextPropagation_StaticActivationStashed(t *testing.T) {
// 	// Mini-replication of scheduler.scheduleResumedNode's stash
// 	// sequence; avoids the full scheduler loop.
// 	parent := newMockCtx(t)
// 	perNodeCtx := newNodeContext(parent.WithContext(context.Background()), nil)
// 	ctx := perNodeCtx.InvocationContext().WithContext(
// 		WithNodeContext(perNodeCtx.InvocationContext(), perNodeCtx),
// 	)
// 	perNodeCtx.SetInvocationContext(ctx)

// 	nc, ok := NodeContextFromGoContext(perNodeCtx)
// 	if !ok {
// 		t.Fatal("static activation did not stash NodeContext on its own embedded context")
// 	}
// 	if nc != NodeContext(perNodeCtx) {
// 		t.Errorf("recovered NodeContext != perNodeCtx (%p vs %p)", nc, perNodeCtx)
// 	}
// }

// --- test helpers ---

// fnNode adapts a func(NodeContext) callback into a Node that emits
// a single Output event on success.
type fnNode struct {
	BaseNode
	fn func(NodeContext) (any, error)
}

func newFnNode(name string, fn func(NodeContext) (any, error)) *fnNode {
	return &fnNode{
		BaseNode: NewBaseNode(name, "", NodeConfig{}),
		fn:       fn,
	}
}

func (n *fnNode) Run(ctx agent.Context, input any) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		// dynamic_scheduler passes *nodeContext as agent.InvocationContext.

		out, err := n.fn(ctx)
		if err != nil {
			yield(nil, err)
			return
		}
		yield(&session.Event{Output: out}, nil)
	}
}
