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
			if err := validateDefaultRoute(&Workflow{edges: map[Node][]Edge{nodeA: tc.edges}}); !errors.Is(err, tc.expectErr) {
				t.Errorf("got %v, expected %v", err, tc.expectErr)
			}
		})
	}
}
