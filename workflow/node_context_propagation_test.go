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
	"testing"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/session"
)

// End-to-end check that a child of a dynamic-node activation can
// recover its own NodeContext via NodeContextFromGoContext — the
// path runnable tools rely on to reach the sub-scheduler from
// tool.Context.
func TestNodeContextPropagation_DynamicChildEmbedsItself(t *testing.T) {
	var captured NodeContext

	inner := newFnNode("inner", func(ctx NodeContext) (any, error) {
		captured = ctx
		return "ok", nil
	})

	orchestrator := NewDynamicNode[string, string]("orch",
		func(ctx context.Context, invCleanCtx NodeContext, _ string, _ func(*session.Event) error) (string, error) {
			return RunNode[string](ctx, invCleanCtx, inner, nil)
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

func (n *fnNode) Run(_ context.Context, ctx agent.Context, input any) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		out, err := n.fn(ctx)
		if err != nil {
			yield(nil, err)
			return
		}
		yield(&session.Event{Output: out}, nil)
	}
}
