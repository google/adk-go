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

// TestValidateJoinNodesHaveIncoming verifies the build-time check
// that every JoinNode in the edge set has at least one incoming
// edge. adk-python catches this at runtime via a ValueError; we
// surface it earlier so misconstructed graphs fail to build.
func TestValidateJoinNodesHaveIncoming(t *testing.T) {
	t.Run("JoinNode with one incoming edge is OK", func(t *testing.T) {
		jn := NewJoinNode("J", NodeConfig{})
		nodeA := &dummyNode{name: "A"}
		edges := []Edge{
			{From: Start, To: nodeA},
			{From: nodeA, To: jn},
		}
		if err := validateJoinNodesHaveIncoming(edges); err != nil {
			t.Errorf("validateJoinNodesHaveIncoming returned %v, want nil", err)
		}
	})

	t.Run("JoinNode with multiple incoming edges is OK", func(t *testing.T) {
		jn := NewJoinNode("J", NodeConfig{})
		nodeA := &dummyNode{name: "A"}
		nodeB := &dummyNode{name: "B"}
		edges := []Edge{
			{From: Start, To: nodeA},
			{From: Start, To: nodeB},
			{From: nodeA, To: jn},
			{From: nodeB, To: jn},
		}
		if err := validateJoinNodesHaveIncoming(edges); err != nil {
			t.Errorf("validateJoinNodesHaveIncoming returned %v, want nil", err)
		}
	})

	t.Run("JoinNode that only appears as edge source fails", func(t *testing.T) {
		// JoinNode with no incoming edges — only a successor edge
		// exists. This is the misconfiguration the validator catches.
		jn := NewJoinNode("J", NodeConfig{})
		nodeD := &dummyNode{name: "D"}
		edges := []Edge{
			{From: Start, To: jn}, // illegal anyway (Start has no JoinNode predecessors), but the check we exercise here is the JoinNode rule
			{From: jn, To: nodeD},
		}
		// Trim the Start→J edge so we isolate the JoinNode check.
		edges = []Edge{{From: jn, To: nodeD}}
		err := validateJoinNodesHaveIncoming(edges)
		if !errors.Is(err, ErrJoinNodeNoIncoming) {
			t.Errorf("validateJoinNodesHaveIncoming returned %v, want ErrJoinNodeNoIncoming", err)
		}
		if err != nil && !strings.Contains(err.Error(), "J") {
			t.Errorf("error message %q does not include the offending node name", err.Error())
		}
	})

	t.Run("graph with no JoinNode passes", func(t *testing.T) {
		nodeA := &dummyNode{name: "A"}
		nodeB := &dummyNode{name: "B"}
		edges := []Edge{
			{From: Start, To: nodeA},
			{From: nodeA, To: nodeB},
		}
		if err := validateJoinNodesHaveIncoming(edges); err != nil {
			t.Errorf("validateJoinNodesHaveIncoming returned %v, want nil for JoinNode-free graph", err)
		}
	})

	t.Run("New() rejects JoinNode without incoming via full validation pipeline", func(t *testing.T) {
		// Belt-and-braces: the same condition surfaced through
		// New() (the constructor users actually call).
		jn := NewJoinNode("J", NodeConfig{})
		nodeD := &dummyNode{name: "D"}
		_, err := New([]Edge{
			{From: Start, To: nodeD},
			{From: jn, To: nodeD},
		})
		if !errors.Is(err, ErrJoinNodeNoIncoming) {
			t.Errorf("New returned %v, want it to wrap ErrJoinNodeNoIncoming", err)
		}
	})
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
