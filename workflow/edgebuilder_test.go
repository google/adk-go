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
	"testing"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/session"
)

func TestEdgeBuilder(t *testing.T) {
	nodeA := &dummyNode{name: "A"}
	nodeB := &dummyNode{name: "B"}
	nodeC := &dummyNode{name: "C"}

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
				return b.AddRoutes(nodeA, map[string]Node{
					"42":         nodeB,
					"workflow_C": nodeC,
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
				t.Fatalf("expected %d edges, got %d", len(tc.expected), len(edges))
			}

			for _, exp := range tc.expected {
				found := false
				for _, actual := range edges {
					if actual.From == exp.From && actual.To == exp.To && reflect.DeepEqual(actual.Route, exp.Route) {
						found = true
						break
					}
				}
				if !found {
					t.Errorf("expected edge not found: From %s, To %s, Route %v", exp.From.Name(), exp.To.Name(), exp.Route)
				}
			}
		})
	}
}

// dummyNode is a minimal implementation of Node for testing purposes.
type dummyNode struct {
	name string
}

func (n *dummyNode) Name() string        { return n.name }
func (n *dummyNode) Description() string { return "" }
func (n *dummyNode) Run(ctx agent.InvocationContext, input any) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {}
}
