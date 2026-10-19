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
	"sync"
	"testing"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/session"
)

func TestEdgeBuilder(t *testing.T) {
	nodeA := &dummyNode{BaseNode: NewBaseNode("A", "", NodeConfig{})}
	nodeB := &dummyNode{BaseNode: NewBaseNode("B", "", NodeConfig{})}
	nodeC := &dummyNode{BaseNode: NewBaseNode("C", "", NodeConfig{})}
	nodeD := &dummyNode{BaseNode: NewBaseNode("D", "", NodeConfig{})}

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
				// Three keys in reverse-sorted order, for the reason given on
				// TestEdgeBuilder_AddRoutesSortsByRoute. The two keys this
				// case used to carry let the unfixed code through 46 times
				// in 300 runs.
				return b.AddRoutes(nodeA, map[string]Node{
					"workflow_C": nodeC,
					"beta":       nodeD,
					"42":         nodeB,
				})
			},
			expected: []Edge{
				{From: nodeA, To: nodeB, Route: StringRoute("42")},
				{From: nodeA, To: nodeD, Route: StringRoute("beta")},
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
	// Keep the literal in reverse-sorted order, and keep at least three keys.
	// A map small enough to fit one group iterates as a rotation of its
	// insertion order and nothing else, so a test only separates sorted from
	// unsorted where the sorted sequence is outside that set of rotations.
	// Reversing achieves it for three keys or more, and for two it does not,
	// because there the reverse is itself a rotation.
	//
	// Those rotations are not equally likely, which makes tidying this
	// literal worse than it looks. Iteration starts at one of the group's
	// eight slots, and with four entries the four empty slots all fall
	// through to the first entry, so insertion order comes up five times in
	// eight and each other rotation once in eight. Sorting the literal, the
	// obvious tidy-up, would leave the test passing against the unfixed code
	// about five runs in eight. Measured on go1.26.6 over a million draws:
	// 0.625 for sorted, 0.125 for any other rotation.
	routes := map[string]Node{
		"tech":    newDummyNode("tech_node"),
		"sales":   newDummyNode("sales_node"),
		"billing": newDummyNode("billing_node"),
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
		want, ok := routes[got[i]]
		if !ok {
			t.Errorf("edge %d has route %q, which is not one of the routes built", i, got[i])
			continue
		}
		if e.To != want {
			t.Errorf("edge %d route %q points at %s, want %s", i, got[i], e.To.Name(), want.Name())
		}
	}
	if want := []string{"abuse", "billing", "sales", "tech"}; !slices.Equal(got, want) {
		t.Errorf("route order = %v, want %v", got, want)
	}
}

func TestEdgeBuilder_AddRoutesSortsBytewise(t *testing.T) {
	// Keys and target names chosen so byte order differs from numeric,
	// case-folded and by-target-name order. Each of those alternatives would
	// otherwise satisfy TestEdgeBuilder_AddRoutesSortsByRoute. The literal is
	// reverse-sorted for the reason given on that test.
	routes := map[string]Node{
		"alpha": newDummyNode("a"),
		"Beta":  newDummyNode("b"),
		"2":     newDummyNode("d"),
		"10":    newDummyNode("c"),
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

// TestEdgeBuilder_AddRoutesDispatchOrder runs a graph built by AddRoutes
// through the scheduler, which is the only thing that makes the edge order an
// observable property rather than a detail of the builder. The router emits
// every route, so all four edges match one event and edge order alone decides
// who runs. WithMaxConcurrency(1) turns that into an assertable sequence: the
// first successor dispatches, the rest queue, and the queue drains FIFO.
func TestEdgeBuilder_AddRoutesDispatchOrder(t *testing.T) {
	// Reverse-sorted, for the reason given on
	// TestEdgeBuilder_AddRoutesSortsByRoute.
	declared := []string{"alpha", "Beta", "2", "10"}

	var mu sync.Mutex
	var started []string
	routes := make(map[string]Node, len(declared))
	for _, r := range declared {
		routes[r] = NewFunctionNode("target_"+r,
			func(ctx agent.Context, input any) (string, error) {
				mu.Lock()
				defer mu.Unlock()
				started = append(started, r)
				return "ok", nil
			}, defaultNodeConfig)
	}

	router := &CustomRouteNode{
		BaseNode: NewBaseNode("router", "", defaultNodeConfig),
		route:    declared,
	}
	// The four targets would otherwise all be terminal and all produce
	// output, which Run rejects with ErrMultipleTerminalOutputs when the
	// scheduler finalizes. New does not: it validates graph shape and cannot
	// know which nodes produce output. A JoinNode downstream leaves the graph
	// with a single terminal node.
	join := NewJoinNode("join")
	b := NewEdgeBuilder().Add(Start, router).AddRoutes(router, routes)
	for _, r := range declared {
		b.Add(routes[r], join)
	}

	w, err := New("", b.Build(), WithMaxConcurrency(1))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	drain(t, w.Run(newSeededMockCtx(t)))

	mu.Lock()
	defer mu.Unlock()
	if want := []string{"10", "2", "Beta", "alpha"}; !slices.Equal(started, want) {
		t.Errorf("dispatch order = %v, want %v", started, want)
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
