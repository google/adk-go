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
	"slices"

	"github.com/google/jsonschema-go/jsonschema"
)

// graph is the precomputed structural view of a workflow's edges.
// Built once at workflow construction; queried by the engine at
// dispatch time.
type graph struct {
	// edges holds the edges in declaration order.
	edges []Edge
	// nodes holds each node once, in first-mention order: the edges in
	// declaration order, and within an edge From before To.
	nodes         []Node
	successors    map[Node][]Edge
	predecessors  map[Node][]Edge
	isRootWrapper bool
	// isNodeRootWrapper is set by WithNodeRootWrapper. New then
	// guarantees a single edge, so every node other than Start is the
	// node root.
	isNodeRootWrapper bool
	// outputSchema is set by WithOutputSchema. The scheduler's finalize
	// validates the single terminal output of a finished run or resume
	// against it. Nil means no validation.
	outputSchema *jsonschema.Resolved
}

// newGraph indexes edges by source and target node so successor
// and predecessor lookups are both O(1) at dispatch time. The
// returned graph references the input edges by value; mutating the
// input edge slice afterwards does not affect the graph.
func newGraph(edges []Edge) *graph {
	g := &graph{
		edges:        slices.Clone(edges),
		successors:   make(map[Node][]Edge),
		predecessors: make(map[Node][]Edge),
	}
	seen := make(map[Node]bool)
	for _, edge := range g.edges {
		g.successors[edge.From] = append(g.successors[edge.From], edge)
		g.predecessors[edge.To] = append(g.predecessors[edge.To], edge)
		for _, n := range [...]Node{edge.From, edge.To} {
			if !seen[n] {
				seen[n] = true
				g.nodes = append(g.nodes, n)
			}
		}
	}
	return g
}

// allEdges returns all edges in the graph.
func (g *graph) allEdges() []Edge {
	var edges []Edge
	for _, succs := range g.successors {
		edges = append(edges, succs...)
	}
	return edges
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

// allNodes returns all nodes in the graph, each once, in first-mention
// order. The caller owns the returned slice.
func (g *graph) allNodes() []Node {
	return slices.Clone(g.nodes)
}

// terminalNodeNames returns the names of terminal nodes: those with no
// outgoing edges, excluding the Start sentinel.
func (g *graph) terminalNodeNames() map[string]bool {
	sources := make(map[string]bool)
	for n := range g.successors {
		sources[n.Name()] = true
	}
	terminals := make(map[string]bool)
	for _, n := range g.allNodes() {
		if n.Name() == Start.Name() || sources[n.Name()] {
			continue
		}
		terminals[n.Name()] = true
	}
	return terminals
}
