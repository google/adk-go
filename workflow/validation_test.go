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

func TestStartNodePresent(t *testing.T) {
	tests := []struct {
		name           string
		edges          []Edge
		expectErrorMsg string
	}{
		{
			name: "with start node",
			edges: func() []Edge {
				nodeA := &dummyNode{name: "A"}
				nodeB := &dummyNode{name: "B"}
				return []Edge{
					{From: Start, To: nodeA},
					{From: nodeA, To: nodeB},
				}
			}(),
		},
		{
			name: "no start node",
			edges: func() []Edge {
				nodeA := &dummyNode{name: "A"}
				nodeB := &dummyNode{name: "B"}
				return []Edge{
					{From: nodeA, To: nodeB},
				}
			}(),
			expectErrorMsg: "no start node found",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := validateStartNodePresent(tc.edges)
			if tc.expectErrorMsg != "" {
				if err == nil {
					t.Errorf("expected error matching %q, got none", tc.expectErrorMsg)
				} else if !strings.Contains(err.Error(), tc.expectErrorMsg) {
					t.Errorf("expected error containing %q, got %v", tc.expectErrorMsg, err)
				}
			}
		})
	}
}

func TestStartNodeNoIncomingEdges(t *testing.T) {
	tests := []struct {
		name           string
		edges          []Edge
		expectErrorMsg string
	}{
		{
			name: "start node with no incoming edges",
			edges: func() []Edge {
				nodeA := &dummyNode{name: "A"}
				nodeB := &dummyNode{name: "B"}
				return []Edge{
					{From: Start, To: nodeA},
					{From: nodeA, To: nodeB},
				}
			}(),
		},
		{
			name: "start node has incoming edges",
			edges: func() []Edge {
				nodeA := &dummyNode{name: "A"}
				nodeB := &dummyNode{name: "B"}
				return []Edge{
					{From: nodeA, To: Start},
					{From: Start, To: nodeB},
				}
			}(),
			expectErrorMsg: "node points to start node: A",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := validateStartNodeNoIncoming(tc.edges)
			if tc.expectErrorMsg != "" {
				if err == nil {
					t.Errorf("expected error matching %q, got none", tc.expectErrorMsg)
				} else if !strings.Contains(err.Error(), tc.expectErrorMsg) {
					t.Errorf("expected error containing %q, got %v", tc.expectErrorMsg, err)
				}
			}
		})
	}
}

func TestDuplicateEdges(t *testing.T) {
	nodeA := &dummyNode{name: "A"}
	nodeB := &dummyNode{name: "B"}
	tests := []struct {
		name      string
		edges     []Edge
		expectErr bool
	}{
		{
			name:  "no duplicate edges",
			edges: []Edge{{From: nodeA, To: nodeB}},
		},
		{
			name:      "duplicate edges",
			edges:     []Edge{{From: nodeA, To: nodeB}, {From: nodeA, To: nodeB}},
			expectErr: true,
		},
		{
			name:      "duplicate edges with different routes",
			edges:     []Edge{{From: nodeA, To: nodeB, Route: StringRoute("test1")}, {From: nodeA, To: nodeB, Route: StringRoute("test2")}},
			expectErr: true,
		},
		{
			name:      "duplicate edges one without route",
			edges:     []Edge{{From: nodeA, To: nodeB, Route: StringRoute("test1")}, {From: nodeA, To: nodeB}},
			expectErr: true,
		},
		{
			name:  "empty edges",
			edges: []Edge{},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if err := validateUniqueEdges(newGraph(tc.edges)); err != nil && !tc.expectErr {
				t.Errorf("got an error %v, expected none", err)
			} else if err == nil && tc.expectErr {
				t.Errorf("expected an error, got none")
			}
		})
	}
}

func TestDefaultRoute(t *testing.T) {
	nodeA := &dummyNode{name: "A"}
	nodeB := &dummyNode{name: "B"}
	nodeC := &dummyNode{name: "C"}
	tests := []struct {
		name      string
		edges     []Edge
		expectErr error
	}{
		{
			name:  "single default route",
			edges: []Edge{{From: nodeA, To: nodeB, Route: Default}},
		},
		{
			name:      "multiple default routes",
			edges:     []Edge{{From: nodeA, To: nodeB, Route: Default}, {From: nodeA, To: nodeC, Route: Default}},
			expectErr: ErrMultipleDefaultRoutes,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if err := validateDefaultRoute(newGraph(tc.edges)); !errors.Is(err, tc.expectErr) {
				t.Errorf("got %v, expected %v", err, tc.expectErr)
			}
		})
	}
}

func TestConnectivity(t *testing.T) {
	nodeA := &dummyNode{name: "A"}
	nodeB := &dummyNode{name: "B"}
	nodeC := &dummyNode{name: "C"}
	tests := []struct {
		name           string
		edges          []Edge
		expectErrorMsg string
	}{
		{
			name: "all nodes connected",
			edges: []Edge{
				{From: Start, To: nodeA},
				{From: nodeA, To: nodeB},
				{From: nodeB, To: nodeC},
			},
		},
		{
			name: "disconnected nodes",
			edges: []Edge{
				{From: Start, To: nodeA},
				{From: nodeB, To: nodeC},
			},
			expectErrorMsg: "nodes not reachable from start node: \"B, C\"",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := validateConnectivity(newGraph(tc.edges))
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
			err := validateCycles(newGraph(tc.edges()))
			if tc.expectErr && err == nil {
				t.Errorf("expected error, got none")
			} else if !tc.expectErr && err != nil {
				t.Errorf("expected no error, got %v", err)
			}
		})
	}
}
