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

import "google.golang.org/adk/agent"

// nodeContext is the per-node InvocationContext seen inside Node.Run.
// It wraps the workflow's incoming agent.InvocationContext and adds
// engine-supplied metadata: the upstream node name (TriggeredBy),
// the static set of predecessor names (InNodes), and a reference to
// the per-run fan-in accumulator that JoinNode (and any custom
// fan-in node) uses to merge per-predecessor outputs.
//
// TODO(wolo): replace once context-unification work lands.
type nodeContext struct {
	agent.InvocationContext
	triggeredBy string
	inNodes     map[string]struct{}

	// joinAccumulator is the per-run fan-in scratchpad supplied by
	// the scheduler. JoinNode (and any custom fan-in node defined
	// inside this package) reaches it via a type assertion on the
	// context. Nil for invocations that did not originate from the
	// scheduler (e.g. ad-hoc Node.Run calls in tests), in which
	// case fan-in nodes degrade to single-activation behaviour.
	joinAccumulator *joinAccumulator
}

// newNodeContext returns a nodeContext wrapping parent with the given
// upstream-node name and predecessor-name set. triggeredBy is empty
// for the initial START activation. inNodes is the set returned by
// graph.inNodeNamesOf for this node and may be nil for a node with
// no incoming edges (e.g. START, or detached nodes). acc is the
// per-name fan-in accumulator reference handed in by the scheduler;
// nil disables JoinNode aggregation (single-activation fallback).
//
// The inNodes map is shared by reference with the graph and must
// not be mutated by callers; InNodes() defensively copies.
func newNodeContext(parent agent.InvocationContext, triggeredBy string, inNodes map[string]struct{}, acc *joinAccumulator) *nodeContext {
	return &nodeContext{
		InvocationContext: parent,
		triggeredBy:       triggeredBy,
		inNodes:           inNodes,
		joinAccumulator:   acc,
	}
}

// TriggeredBy returns the name of the upstream node whose output
// scheduled this node activation. Empty for the initial START
// trigger and for non-workflow invocations (where the wrapper is not
// used).
func (c *nodeContext) TriggeredBy() string { return c.triggeredBy }

// InNodes returns the set of upstream node names that point at the
// currently-running node via any edge. Mirrors adk-python's
// `Context.in_nodes`, used by JoinNode to know how many distinct
// predecessors must trigger before it emits its aggregated output.
//
// The returned slice is unsorted (set semantics) and freshly
// allocated; callers may mutate it without affecting future calls.
// Returns an empty (nil) slice for a node with no incoming edges.
func (c *nodeContext) InNodes() []string {
	if len(c.inNodes) == 0 {
		return nil
	}
	out := make([]string, 0, len(c.inNodes))
	for name := range c.inNodes {
		out = append(out, name)
	}
	return out
}
