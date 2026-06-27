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
	"fmt"
	"iter"
	"sync"

	"github.com/google/jsonschema-go/jsonschema"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/internal/typeutil"
	"google.golang.org/adk/session"
)

// DynamicFn is the orchestrator body of a dynamic node. emit publishes
// mid-body events (state updates, HITL requests, progress); the return
// value becomes the node's terminal Event.Output.
type DynamicFn[IN, OUT any] = func(ctx agent.Context, in IN, emit func(*session.Event) error) (OUT, error)

type dynamicNode[IN, OUT any] struct {
	BaseNode
	fn           DynamicFn[IN, OUT]
	inputSchema  *jsonschema.Resolved
	outputSchema *jsonschema.Resolved
}

// NewDynamicNode wraps fn as a workflow Node whose execution order is
// expressed as Go code calling RunNode for each child. cfg.RerunOnResume
// defaults to &true when nil (needed so resume can re-enter the
// orchestrator and deliver cached child results); explicit &false is
// respected.
func NewDynamicNode[IN, OUT any](name string, fn DynamicFn[IN, OUT], cfg NodeConfig) Node {
	return newDynamicNodeWithResolvedSchemas[IN, OUT](name, fn, nil, nil, applyDynamicDefaults(cfg))
}

// NewDynamicNodeWithSchema is the explicit-schema variant.
func NewDynamicNodeWithSchema[IN, OUT any](
	name string,
	fn DynamicFn[IN, OUT],
	inputSchema, outputSchema *jsonschema.Schema,
	cfg NodeConfig,
) (Node, error) {
	var ischema *jsonschema.Resolved
	if inputSchema != nil {
		r, err := inputSchema.Resolve(nil)
		if err != nil {
			return nil, fmt.Errorf("resolving input schema: %w", err)
		}
		ischema = r
	}
	var oschema *jsonschema.Resolved
	if outputSchema != nil {
		r, err := outputSchema.Resolve(nil)
		if err != nil {
			return nil, fmt.Errorf("resolving output schema: %w", err)
		}
		oschema = r
	}
	return newDynamicNodeWithResolvedSchemas[IN, OUT](name, fn, ischema, oschema, applyDynamicDefaults(cfg)), nil
}

func newDynamicNodeWithResolvedSchemas[IN, OUT any](
	name string,
	fn DynamicFn[IN, OUT],
	inputSchema, outputSchema *jsonschema.Resolved,
	cfg NodeConfig,
) *dynamicNode[IN, OUT] {
	return &dynamicNode[IN, OUT]{
		BaseNode:     NewBaseNodeWithSchemas(name, "", cfg, inputSchema, outputSchema),
		fn:           fn,
		inputSchema:  inputSchema,
		outputSchema: outputSchema,
	}
}

func applyDynamicDefaults(cfg NodeConfig) NodeConfig {
	if cfg.RerunOnResume == nil {
		t := true
		cfg.RerunOnResume = &t
	}
	return cfg
}

func (n *dynamicNode[IN, OUT]) Run(ctx agent.Context, input any) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		typedInput, err := n.coerceInput(input)
		if err != nil {
			yield(nil, err)
			return
		}

		emit := makeEmit(yield, ctx)
		sub := newDynamicSubScheduler(ctx, n.composePath(ctx), emit)
		orchestratorCtx := agent.NewDynamicNodeContext(ctx, sub.ParentPath(), "", sub, sub.OutputForAncestors())

		out, err := n.fn(orchestratorCtx, typedInput, emit)
		if err != nil {
			// WaitForOutput park emitted no interrupt event, so surface
			// the sentinel for the scheduler to park the parent.
			if errors.Is(err, ErrNodeWaitingForOutput) {
				yield(nil, err)
				return
			}
			// HITL: the RequestedInput was already forwarded upstream;
			// swallow the sentinel so the engine sees only the pause event.
			if errors.Is(err, ErrNodeInterrupted) {
				return
			}
			yield(nil, err)
			return
		}

		// A WithUseAsOutput child already emitted this output on its own
		// event (stamped for this node), so emit no duplicate terminal
		// event. Mirrors adk-python's _output_delegated.
		if _, delegated := sub.DelegatedOutput(); delegated {
			return
		}

		// nil output: nothing to emit as a terminal event — the body
		// either produced no output or already carried it on a content
		// event.
		if any(out) == nil {
			return
		}
		ev := session.NewEventWithContext(ctx, ctx.InvocationID())
		ev.Output = out
		ev.NodeInfo = &session.NodeInfo{Path: sub.ParentPath()}
		// TODO(wolo): validate ev.Output against n.outputSchema,
		// mirroring function_node.go:87-92.
		yield(ev, nil)
	}
}

// coerceInput converts the raw input to IN, using a typeutil JSON
// roundtrip as a fallback (mirrors FunctionNode for tool-node →
// dynamic-node edges where the upstream emits map[string]any).
func (n *dynamicNode[IN, OUT]) coerceInput(input any) (IN, error) {
	var zero IN
	if input == nil {
		return zero, nil
	}
	if v, ok := input.(IN); ok {
		return v, nil
	}
	typed, err := typeutil.ConvertToWithJSONSchema[any, IN](input, n.inputSchema)
	if err != nil {
		return zero, fmt.Errorf("dynamic node %q: invalid input type, expected %T: %w", n.Name(), zero, err)
	}
	return typed, nil
}

// composePath returns this dynamic node's own composite path. When this
// node runs as a dynamic child, the scheduler already created its
// context with the full child path ("<parent>/<name>@<runID>"), so that
// path is used as-is. A top-level activation has no parent path and
// gets the bare Name().
func (n *dynamicNode[IN, OUT]) composePath(parent agent.Context) string {
	if p := parent.Path(); p != "" {
		return p
	}
	return n.Name()
}

// makeEmit wraps yield as an emit callback. Contract: nil return =
// delivered, non-nil = orchestrator must stop (calling yield after
// it returned false is a runtime error per the iter spec).
//
// When yield returns false without ctx cancellation (no current
// consumer triggers this, but the contract must not depend on it),
// return context.Canceled as a stand-in.
//
// A single mutex serializes yield: a DynamicFn may run concurrent
// children (see WithUseSubBranch) that all emit through this one
// callback, and calling the same yield from multiple goroutines panics
// the iterator and races the parent runNode's completion accumulator.
func makeEmit(yield func(*session.Event, error) bool, parentCtx agent.Context) func(*session.Event) error {
	var mu sync.Mutex
	return func(ev *session.Event) error {
		mu.Lock()
		defer mu.Unlock()
		if err := parentCtx.Err(); err != nil {
			return err
		}
		if !yield(ev, nil) {
			if err := parentCtx.Err(); err != nil {
				return err
			}
			return context.Canceled
		}
		return nil
	}
}
