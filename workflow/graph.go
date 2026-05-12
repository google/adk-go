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

// graph is the precomputed structural view of a workflow's edges.
// Built once at workflow construction; queried by the engine at
// dispatch time.
type graph struct {
	successors map[Node][]Edge

	// predecessorNames maps a node to the deduplicated set of
	// upstream node names that point at it via any edge. Used by
	// the engine to populate ctx.InNodes() (the static list of
	// predecessors a JoinNode aggregates over) and by build-time
	// validation that JoinNode has at least one incoming edge.
	predecessorNames map[Node]map[string]struct{}
}

// newGraph indexes edges by source node so successor lookups are
// O(1) at dispatch time. It also indexes per-target predecessor
// names so the engine can answer ctx.InNodes() in O(1).
//
// The returned graph references the input edges by value; mutating
// the input edge slice afterwards does not affect the graph.
func newGraph(edges []Edge) *graph {
	succ := make(map[Node][]Edge)
	preds := make(map[Node]map[string]struct{})
	for _, edge := range edges {
		succ[edge.From] = append(succ[edge.From], edge)
		set, ok := preds[edge.To]
		if !ok {
			set = make(map[string]struct{})
			preds[edge.To] = set
		}
		set[edge.From.Name()] = struct{}{}
	}
	return &graph{successors: succ, predecessorNames: preds}
}

// successorsOf returns the outgoing edges for a node.
// Returns nil if n has no outgoing edges
// (including the case where n is not in the graph at all). The
// returned slice is owned by the graph and must not be mutated by
// callers.
func (g *graph) successorsOf(n Node) []Edge {
	return g.successors[n]
}

// inNodeNamesOf returns the set of upstream node names that point
// at n via any edge. The returned set is owned by the graph and
// must not be mutated by callers. Returns nil if n has no incoming
// edges or is not in the graph.
func (g *graph) inNodeNamesOf(n Node) map[string]struct{} {
	return g.predecessorNames[n]
}
