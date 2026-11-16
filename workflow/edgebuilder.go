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
	"fmt"
	"maps"
	"slices"
)

// ErrEmptyRoutingMap is the error [EdgeBuilder.Err] reports after
// [EdgeBuilder.AddRoutes] was given a map with no entry. [EdgeBuilder.AddRoutes]
// wraps it with the name of the source node, so match it with [errors.Is].
var ErrEmptyRoutingMap = errors.New("routing map must not be empty: provide at least one route-to-node mapping")

// ErrNilRoutingTarget is the error [EdgeBuilder.Err] reports after
// [EdgeBuilder.AddRoutes] was given a map in which a route has a nil target
// node. [EdgeBuilder.AddRoutes] wraps it with the route and the name of the
// source node, so match it with [errors.Is].
var ErrNilRoutingTarget = errors.New("routing map target must not be nil")

// EdgeBuilder provides a fluent API for building a list of Edges.
type EdgeBuilder struct {
	edges []Edge
	err   error // the first error a method recorded; see Err
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
// route may be any [Route], such as [StringRoute], [IntRoute], [BoolRoute],
// [MultiRoute], or [Default] — the edge taken when no concrete route matched.
// A nil route makes the edge unconditional.
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
// Edges are added in byte-wise lexicographic order of the route key, so "10"
// precedes "2". That order is observable: it drives the order successors are
// started in, and the pending queue under [WithMaxConcurrency]. Call
// [EdgeBuilder.AddRoute] per route to choose the order yourself, or to add a
// [Default] edge, which AddRoutes cannot express.
//
// An empty or nil routes map adds no edge and records [ErrEmptyRoutingMap]
// for [EdgeBuilder.Err]. A route whose target is nil adds no edge for that
// route and records [ErrNilRoutingTarget]; the other routes of the map are
// still added. Only the first recorded error is kept.
func (b *EdgeBuilder) AddRoutes(from Node, routes map[string]Node) *EdgeBuilder {
	if len(routes) == 0 {
		b.recordErr(fmt.Errorf("%w: from node %q", ErrEmptyRoutingMap, nodeNameOrNil(from)))
		return b
	}
	for _, route := range slices.Sorted(maps.Keys(routes)) {
		to := routes[route]
		if to == nil {
			b.recordErr(fmt.Errorf("%w: route %q from node %q", ErrNilRoutingTarget, route, nodeNameOrNil(from)))
			continue
		}
		b.AddRoute(from, to, StringRoute(route))
	}
	return b
}

// recordErr keeps err unless an earlier error is already recorded.
func (b *EdgeBuilder) recordErr(err error) {
	if b.err == nil {
		b.err = err
	}
}

// nodeNameOrNil returns n.Name(), or "<nil>" for a nil n, so an error about
// a routing map can be built before any check of its source node.
func nodeNameOrNil(n Node) string {
	if n == nil {
		return "<nil>"
	}
	return n.Name()
}

// Err returns the first error a builder method recorded, or nil. Call it
// after the last method and before [New]: an error here means the edges
// Build returns are not the ones the caller described.
func (b *EdgeBuilder) Err() error {
	return b.err
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
