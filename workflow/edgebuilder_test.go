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
	"iter"
	"reflect"
	"slices"
	"testing"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/session"
)

func TestEdgeBuilder(t *testing.T) {
	nodeA := &dummyNode{BaseNode: NewBaseNode("A", "", NodeConfig{})}
	nodeB := &dummyNode{BaseNode: NewBaseNode("B", "", NodeConfig{})}
	nodeC := &dummyNode{BaseNode: NewBaseNode("C", "", NodeConfig{})}

	tests := []struct {
		name     string
		build    func(*EdgeBuilder) *EdgeBuilder
		expected []Edge
	}{
		{
			name: "Add",
			build: func(b *EdgeBuilder) *EdgeBuilder {
				return b.Add(nodeA, nodeB)
			},
			expected: []Edge{{From: nodeA, To: nodeB}},
		},
		{
			name: "AddRoute",
			build: func(b *EdgeBuilder) *EdgeBuilder {
				return b.AddRoute(nodeA, nodeB, StringRoute("test-route"))
			},
			expected: []Edge{{From: nodeA, To: nodeB, Route: StringRoute("test-route")}},
		},
		{
			name: "AddRoute MultiRoute",
			build: func(b *EdgeBuilder) *EdgeBuilder {
				return b.AddRoute(nodeA, nodeB, MultiRoute[int]{1, 2, 3})
			},
			expected: []Edge{{From: nodeA, To: nodeB, Route: MultiRoute[int]{1, 2, 3}}},
		},
		{
			name: "AddFanOut",
			build: func(b *EdgeBuilder) *EdgeBuilder {
				return b.AddFanOut(nodeA, nodeB, nodeC)
			},
			expected: []Edge{
				{From: nodeA, To: nodeB},
				{From: nodeA, To: nodeC},
			},
		},
		{
			name: "AddFanIn",
			build: func(b *EdgeBuilder) *EdgeBuilder {
				return b.AddFanIn(nodeA, nodeB, nodeC)
			},
			expected: []Edge{
				{From: nodeB, To: nodeA},
				{From: nodeC, To: nodeA},
			},
		},
		{
			name: "AddRoutes",
			build: func(b *EdgeBuilder) *EdgeBuilder {
				// Literal order is the reverse of sorted order, so this case can
				// tell the two apart.
				return b.AddRoutes(nodeA, map[string]Node{
					"workflow_C": nodeC,
					"42":         nodeB,
				})
			},
			expected: []Edge{
				{From: nodeA, To: nodeB, Route: StringRoute("42")},
				{From: nodeA, To: nodeC, Route: StringRoute("workflow_C")},
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			edges := tc.build(NewEdgeBuilder()).Build()

			if len(edges) != len(tc.expected) {
				t.Fatalf("got %d edges, want %d", len(edges), len(tc.expected))
			}
			for i, want := range tc.expected {
				if got := edges[i]; got.From != want.From || got.To != want.To || !reflect.DeepEqual(got.Route, want.Route) {
					t.Errorf("edge %d = %s→%s (route %v), want %s→%s (route %v)",
						i, got.From.Name(), got.To.Name(), got.Route,
						want.From.Name(), want.To.Name(), want.Route)
				}
			}
		})
	}
}

func TestEdgeBuilder_AddRoutesSortsByRoute(t *testing.T) {
	from := newDummyNode("router")
	// Insertion order deliberately differs from sorted order; ranging the map
	// would pick a fresh random order on every run.
	routes := map[string]Node{
		"tech":    newDummyNode("tech_node"),
		"billing": newDummyNode("billing_node"),
		"sales":   newDummyNode("sales_node"),
		"abuse":   newDummyNode("abuse_node"),
	}

	edges := NewEdgeBuilder().AddRoutes(from, routes).Build()

	if len(edges) != len(routes) {
		t.Fatalf("got %d edges, want %d", len(edges), len(routes))
	}
	got := make([]string, len(edges))
	for i, e := range edges {
		route, ok := e.Route.(StringRoute)
		if !ok {
			t.Fatalf("edge %d route = %T, want StringRoute", i, e.Route)
		}
		got[i] = string(route)
		if e.From != from {
			t.Errorf("edge %d from = %s, want %s", i, e.From.Name(), from.Name())
		}
		if want := routes[got[i]]; e.To != want {
			t.Errorf("edge %d route %q points at %s, want %s", i, got[i], e.To.Name(), want.Name())
		}
	}
	if want := []string{"abuse", "billing", "sales", "tech"}; !slices.Equal(got, want) {
		t.Errorf("route order = %v, want %v", got, want)
	}
}

func TestEdgeBuilder_AddRoutesSortsBytewise(t *testing.T) {
	// Keys and target names chosen so byte order differs from numeric,
	// case-folded and by-target-name order; each of those would otherwise
	// satisfy TestEdgeBuilder_AddRoutesSortsByRoute.
	routes := map[string]Node{
		"2":     newDummyNode("d"),
		"10":    newDummyNode("c"),
		"Beta":  newDummyNode("b"),
		"alpha": newDummyNode("a"),
	}

	edges := NewEdgeBuilder().AddRoutes(newDummyNode("router"), routes).Build()

	got := make([]string, len(edges))
	for i, e := range edges {
		route, ok := e.Route.(StringRoute)
		if !ok {
			t.Fatalf("edge %d route = %T, want StringRoute", i, e.Route)
		}
		got[i] = string(route)
	}
	if want := []string{"10", "2", "Beta", "alpha"}; !slices.Equal(got, want) {
		t.Errorf("route order = %v, want %v", got, want)
	}
}

// dummyNode is a minimal implementation of Node for testing purposes.
type dummyNode struct {
	BaseNode
}

func newDummyNode(name string) *dummyNode {
	return &dummyNode{BaseNode: NewBaseNode(name, "", NodeConfig{})}
}

func (n *dummyNode) ValidateOutput(output any) (any, error) {
	return output, nil
}

func (n *dummyNode) Run(ctx agent.Context, input any) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {}
}
