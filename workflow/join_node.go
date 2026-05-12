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

	"google.golang.org/adk/agent"
	"google.golang.org/adk/session"
)

// joinAccumulator is the per-(workflow-run × node-name) scratchpad
// JoinNode (and any custom fan-in node defined in this package)
// uses to merge per-predecessor activations. The scheduler creates
// one lazily on first record() and clears it once the join's
// terminal output activation has been scheduled (see
// scheduler.handleCompletion).
//
// Concurrency: the scheduler serialises activations of the same
// node name (see scheduler.trigger), so a JoinNode never has two
// activations modifying its accumulator concurrently. No mutex.
type joinAccumulator struct {
	// inputsByPredecessor records {triggeringNodeName: nodeInput}
	// for every predecessor activation observed so far. Lookup
	// against ctx.InNodes() decides when the join is complete.
	inputsByPredecessor map[string]any
}

// record stores input under the triggering predecessor's name. The
// second call from the same predecessor overwrites the first
// (mirrors adk-python's `_join_node.py:join_state[triggering_node] =
// node_input` behaviour).
func (a *joinAccumulator) record(predecessor string, input any) {
	if a.inputsByPredecessor == nil {
		a.inputsByPredecessor = map[string]any{}
	}
	a.inputsByPredecessor[predecessor] = input
}

// snapshot returns a copy of the per-predecessor input map. Used by
// JoinNode to emit its terminal output without exposing the
// internal map for accidental mutation.
func (a *joinAccumulator) snapshot() map[string]any {
	out := make(map[string]any, len(a.inputsByPredecessor))
	for k, v := range a.inputsByPredecessor {
		out[k] = v
	}
	return out
}

// JoinNode is a fan-in primitive: it waits until every predecessor
// declared in the workflow graph has triggered it once, then emits
// a single output event whose value is a `map[string]any` keyed by
// the predecessor's node name with the corresponding per-predecessor
// input as the value.
//
// Mirrors adk-python's `JoinNode` (`_join_node.py`). Like its
// Python counterpart, JoinNode runs once per upstream completion,
// records the input on the engine-managed accumulator, and either
//
//   - emits an aggregated output event (when every predecessor in
//     ctx.InNodes() has been seen), at which point the engine
//     schedules its successors and clears the accumulator; or
//   - emits no output, leaving the engine to mark the node as
//     NodeWaiting (because Config().WaitForOutput=true) and resume
//     it on the next predecessor activation.
//
// Construction-time validation in workflow.New rejects a JoinNode
// with zero incoming edges; at runtime the node still defends
// against it for ad-hoc Run calls outside the engine.
//
// Example:
//
//	join := workflow.NewJoinNode("join", workflow.NodeConfig{})
//	edges := workflow.Concat(
//	    workflow.Chain(workflow.Start, taskA, join),
//	    workflow.Chain(workflow.Start, taskB, join),
//	    workflow.Chain(workflow.Start, taskC, join),
//	    []workflow.Edge{{From: join, To: taskD}},
//	)
type JoinNode struct {
	BaseNode
}

// NewJoinNode constructs a JoinNode with the given name and config.
// The Config().WaitForOutput field is forced to true regardless of
// what the caller passes — the fan-in semantics depend on it.
func NewJoinNode(name string, cfg NodeConfig) *JoinNode {
	// Force WaitForOutput=true; JoinNode's contract requires the
	// scheduler to keep it in NodeWaiting between per-predecessor
	// activations until the final output is emitted.
	t := true
	cfg.WaitForOutput = &t
	return &JoinNode{BaseNode: NewBaseNode(name, "fan-in: aggregates per-predecessor inputs into a single map[string]any output", cfg)}
}

// Run records the current activation's input under the triggering
// predecessor's name in the per-run accumulator, then either emits
// the merged output (when every predecessor has fired) or yields
// nothing — letting the scheduler put the node into NodeWaiting
// until the next predecessor triggers it.
//
// When the node is invoked outside the workflow engine (no
// joinAccumulator on the context) it degrades to "emit current
// input as a single-key map", which is the most useful behaviour
// for ad-hoc Run() calls in unit tests.
func (n *JoinNode) Run(ctx agent.InvocationContext, input any) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		nc, ok := ctx.(*nodeContext)
		if !ok || nc.joinAccumulator == nil {
			// Engine-detached path: emit a degenerate single-trigger
			// output so callers exercising Run() in isolation get a
			// useful event rather than silence.
			ev := session.NewEvent(ctx.InvocationID())
			ev.Actions.StateDelta["output"] = map[string]any{ctx.TriggeredBy(): input}
			yield(ev, nil)
			return
		}

		predecessors := nc.inNodes
		if len(predecessors) == 0 {
			// Defensive: workflow.New rejects this case at build
			// time, but a JoinNode could still be embedded in a
			// detached graph or constructed outside the validation
			// path.
			yield(nil, fmt.Errorf("workflow: JoinNode %q has no incoming edges", n.Name()))
			return
		}

		triggering := nc.triggeredBy
		if triggering == "" {
			yield(nil, fmt.Errorf("workflow: JoinNode %q activated without a triggering predecessor", n.Name()))
			return
		}
		if _, known := predecessors[triggering]; !known {
			yield(nil, fmt.Errorf("workflow: JoinNode %q triggered by %q which is not a declared predecessor", n.Name(), triggering))
			return
		}

		nc.joinAccumulator.record(triggering, input)

		// Have we now seen every predecessor? If not, return without
		// emitting an output event; the scheduler will keep us in
		// NodeWaiting until the next predecessor triggers us.
		if !haveAllPredecessors(nc.joinAccumulator.inputsByPredecessor, predecessors) {
			return
		}

		// All predecessors have triggered: emit the merged output.
		ev := session.NewEvent(ctx.InvocationID())
		ev.Actions.StateDelta["output"] = nc.joinAccumulator.snapshot()
		yield(ev, nil)
	}
}

// haveAllPredecessors reports whether every predecessor name in
// expected has produced an entry in seen. seen may contain extra
// keys (defensive: Run rejects them above, but the scheduler is
// not required to).
func haveAllPredecessors(seen map[string]any, expected map[string]struct{}) bool {
	for name := range expected {
		if _, ok := seen[name]; !ok {
			return false
		}
	}
	return true
}
