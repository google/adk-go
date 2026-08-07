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
	"maps"
	"slices"
)

// EdgeBuilder provides a fluent API for building a list of Edges.
type EdgeBuilder struct {
	edges []Edge
}

// NewEdgeBuilder creates a new EdgeBuilder.
func NewEdgeBuilder() *EdgeBuilder {
	return &EdgeBuilder{}
}

// Add adds a new edge between two nodes.
func (b *EdgeBuilder) Add(from, to Node) *EdgeBuilder {
	b.edges = append(b.edges, Edge{From: from, To: to})
	return b
}

// AddRoute adds a new edge with a route condition between two nodes.
// The route condition is of type any and is converted to a Route interface.
// Supported routes are StringRoute, IntRoute, BoolRoute and MultiRoute.
func (b *EdgeBuilder) AddRoute(from, to Node, route Route) *EdgeBuilder {
	b.edges = append(b.edges, Edge{From: from, To: to, Route: route})
	return b
}

// AddFanOut adds multiple edges from a single source node to multiple target nodes.
func (b *EdgeBuilder) AddFanOut(from Node, to ...Node) *EdgeBuilder {
	for _, t := range to {
		b.Add(from, t)
	}
	return b
}

// AddFanIn adds multiple edges from multiple source nodes to a single target node.
func (b *EdgeBuilder) AddFanIn(to Node, from ...Node) *EdgeBuilder {
	for _, f := range from {
		b.Add(f, to)
	}
	return b
}

// AddRoutes adds multiple edges from a single source node to multiple target nodes with different route conditions.
//
// Edges are added sorted by route, in Go string order (so "10" sorts before
// "2"). A Go map has no order of its own, so without this the graph would be
// built differently on every run, and edge order is observable: it drives the
// order successors are started in, and the pending queue under
// [WithMaxConcurrency]. Call [EdgeBuilder.AddRoute] per route to choose the
// order yourself, or to add a [Default] edge, which AddRoutes cannot express.
func (b *EdgeBuilder) AddRoutes(from Node, routes map[string]Node) *EdgeBuilder {
	for _, route := range slices.Sorted(maps.Keys(routes)) {
		b.AddRoute(from, routes[route], StringRoute(route))
	}
	return b
}

// Build returns the list of edges.
func (b *EdgeBuilder) Build() []Edge {
	return b.edges
}

// Chain generates a slice of Edges to form a chain of nodes.
func Chain(nodes ...Node) []Edge {
	if len(nodes) < 2 {
		return nil
	}
	edges := make([]Edge, len(nodes)-1)
	for i := 0; i < len(nodes)-1; i++ {
		edges[i] = Edge{From: nodes[i], To: nodes[i+1]}
	}
	return edges
}

// Concat combines Edges and []Edge slices into a single slice of edges.
func Concat(items ...any) []Edge {
	var edges []Edge
	for _, item := range items {
		switch v := item.(type) {
		case Edge:
			edges = append(edges, v)
		case []Edge:
			edges = append(edges, v...)
		}
	}
	return edges
}
