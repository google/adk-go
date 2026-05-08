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
	"strings"
	"testing"
)

func TestUniqueNames(t *testing.T) {
	type nodeSetup struct{ aName, bName, cName string }
	tests := []struct {
		name           string
		setup          nodeSetup
		expectErrorMsg string
	}{
		{
			name:  "unique names",
			setup: nodeSetup{"A", "B", "C"},
		},
		{
			name:           "duplicate node names in From",
			setup:          nodeSetup{"A", "A", "C"},
			expectErrorMsg: "duplicate node name: A",
		},
		{
			name:           "duplicate node names in To",
			setup:          nodeSetup{"A", "B", "A"},
			expectErrorMsg: "duplicate node name: A",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			nodeA := &dummyNode{name: tc.setup.aName}
			nodeB := &dummyNode{name: tc.setup.bName}
			nodeC := &dummyNode{name: tc.setup.cName}
			edges := []Edge{
				{From: nodeA, To: nodeB},
				{From: nodeB, To: nodeC},
			}
			err := validateUniqueNames(edges)
			if tc.expectErrorMsg != "" {
				if err == nil {
					t.Errorf("expected error matching %q, got none", tc.expectErrorMsg)
				} else if !strings.Contains(err.Error(), tc.expectErrorMsg) {
					t.Errorf("expected error containing %q, got %v", tc.expectErrorMsg, err)
				}
			} else if err != nil {
				t.Errorf("expected no error, got %v", err)
			}
		})
	}
}

func TestValidateCycles(t *testing.T) {
	tests := []struct {
		name      string
		edges     func() []Edge
		expectErr bool
	}{
		{
			name: "no cycles",
			edges: func() []Edge {
				nodeA := &dummyNode{name: "A"}
				nodeB := &dummyNode{name: "B"}
				return []Edge{
					{From: Start, To: nodeA},
					{From: nodeA, To: nodeB},
				}
			},
			expectErr: false,
		},
		{
			name: "no cycles diamond graph",
			edges: func() []Edge {
				nodeA := &dummyNode{name: "A"}
				nodeB := &dummyNode{name: "B"}
				nodeC := &dummyNode{name: "C"}
				nodeD := &dummyNode{name: "D"}
				return []Edge{
					{From: Start, To: nodeA},
					{From: nodeA, To: nodeB},
					{From: nodeA, To: nodeC},
					{From: nodeB, To: nodeD},
					{From: nodeC, To: nodeD},
				}
			},
			expectErr: false,
		},
		{
			name: "only conditional cycle",
			edges: func() []Edge {
				nodeA := &dummyNode{name: "A"}
				nodeB := &dummyNode{name: "B"}
				return []Edge{
					{From: Start, To: nodeA},
					{From: nodeA, To: nodeB},
					{From: nodeB, To: nodeA, Route: StringRoute("back")},
				}
			},
			expectErr: false,
		},
		{
			name: "both conditional and unconditional cycles",
			edges: func() []Edge {
				nodeA := &dummyNode{name: "A"}
				nodeB := &dummyNode{name: "B"}
				nodeC := &dummyNode{name: "C"}
				nodeD := &dummyNode{name: "D"}
				return []Edge{
					{From: Start, To: nodeA},
					{From: nodeA, To: nodeB},
					{From: nodeB, To: nodeA, Route: StringRoute("back")},
					{From: Start, To: nodeC},
					{From: nodeC, To: nodeD},
					{From: nodeD, To: nodeC}, // Unconditional
				}
			},
			expectErr: true,
		},
		{
			name: "cycle with default route",
			edges: func() []Edge {
				nodeA := &dummyNode{name: "A"}
				nodeB := &dummyNode{name: "B"}
				return []Edge{
					{From: Start, To: nodeA},
					{From: nodeA, To: nodeB},
					{From: nodeB, To: nodeA, Route: Default},
				}
			},
			expectErr: false,
		},
		{
			name:      "empty graph",
			edges:     func() []Edge { return []Edge{} },
			expectErr: false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			edges := tc.edges()
			adj := make(map[Node][]Edge)
			for _, edge := range edges {
				adj[edge.From] = append(adj[edge.From], edge)
			}
			w := &Workflow{edges: adj}

			err := validateCycles(w)
			if tc.expectErr && err == nil {
				t.Errorf("expected error, got none")
			} else if !tc.expectErr && err != nil {
				t.Errorf("expected no error, got %v", err)
			}
		})
	}
}
