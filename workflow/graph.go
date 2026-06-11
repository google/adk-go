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
	successors   map[Node][]Edge
	predecessors map[Node][]Edge
}

// newGraph indexes edges by source and target node so successor
// and predecessor lookups are both O(1) at dispatch time. The
// returned graph references the input edges by value; mutating the
// input edge slice afterwards does not affect the graph.
func newGraph(edges []Edge) *graph {
	succ := make(map[Node][]Edge)
	pred := make(map[Node][]Edge)
	for _, edge := range edges {
		succ[edge.From] = append(succ[edge.From], edge)
		pred[edge.To] = append(pred[edge.To], edge)
	}
	return &graph{successors: succ, predecessors: pred}
}

// successorsOf returns the outgoing edges for a node.
// Returns nil if n has no outgoing edges
// (including the case where n is not in the graph at all). The
// returned slice is owned by the graph and must not be mutated by
// callers.
func (g *graph) successorsOf(n Node) []Edge {
	return g.successors[n]
}

// predecessorsOf returns the incoming edges for a node. Returns
// nil if n has no incoming edges (including the case where n is
// not in the graph at all). The returned slice is owned by the
// graph and must not be mutated by callers.
func (g *graph) predecessorsOf(n Node) []Edge {
	return g.predecessors[n]
}

// allNodes returns all nodes in the graph.
func (g *graph) allNodes() []Node {
	nodes := make(map[Node]bool)
	for n := range g.successors {
		nodes[n] = true
	}
	for n := range g.predecessors {
		nodes[n] = true
	}
	var res []Node
	for n := range nodes {
		res = append(res, n)
	}
	return res
}
