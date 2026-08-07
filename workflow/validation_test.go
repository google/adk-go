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
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"github.com/google/jsonschema-go/jsonschema"
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
			nodeA := newDummyNode(tc.setup.aName)
			nodeB := newDummyNode(tc.setup.bName)
			nodeC := newDummyNode(tc.setup.cName)
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
				nodeA := newDummyNode("A")
				nodeB := newDummyNode("B")
				return []Edge{
					{From: Start, To: nodeA},
					{From: nodeA, To: nodeB},
				}
			}(),
		},
		{
			name: "no start node",
			edges: func() []Edge {
				nodeA := newDummyNode("A")
				nodeB := newDummyNode("B")
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
				nodeA := newDummyNode("A")
				nodeB := newDummyNode("B")
				return []Edge{
					{From: Start, To: nodeA},
					{From: nodeA, To: nodeB},
				}
			}(),
		},
		{
			name: "start node has incoming edges",
			edges: func() []Edge {
				nodeA := newDummyNode("A")
				nodeB := newDummyNode("B")
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
	nodeA := newDummyNode("A")
	nodeB := newDummyNode("B")
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
	nodeA := newDummyNode("A")
	nodeB := newDummyNode("B")
	nodeC := newDummyNode("C")
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
	nodeA := newDummyNode("A")
	nodeB := newDummyNode("B")
	nodeC := newDummyNode("C")
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
				nodeA := newDummyNode("A")
				nodeB := newDummyNode("B")
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
				nodeA := newDummyNode("A")
				nodeB := newDummyNode("B")
				nodeC := newDummyNode("C")
				nodeD := newDummyNode("D")
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
				nodeA := newDummyNode("A")
				nodeB := newDummyNode("B")
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
				nodeA := newDummyNode("A")
				nodeB := newDummyNode("B")
				nodeC := newDummyNode("C")
				nodeD := newDummyNode("D")
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
				nodeA := newDummyNode("A")
				nodeB := newDummyNode("B")
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

func TestValidateFanIn(t *testing.T) {
	tests := []struct {
		name      string
		edges     func() []Edge
		expectErr bool
	}{
		{
			name: "non-Join diamond fan-in rejected",
			edges: func() []Edge {
				a, b, c, d := newDummyNode("A"), newDummyNode("B"), newDummyNode("C"), newDummyNode("D")
				return []Edge{
					{From: Start, To: a},
					{From: a, To: b},
					{From: a, To: c},
					{From: b, To: d},
					{From: c, To: d},
				}
			},
			expectErr: true,
		},
		{
			name: "JoinNode diamond fan-in allowed",
			edges: func() []Edge {
				a, b, c := newDummyNode("A"), newDummyNode("B"), newDummyNode("C")
				j := NewJoinNode("J")
				return []Edge{
					{From: Start, To: a},
					{From: a, To: b},
					{From: a, To: c},
					{From: b, To: j},
					{From: c, To: j},
				}
			},
			expectErr: false,
		},
		{
			name: "conditional loop-back not rejected",
			edges: func() []Edge {
				a, b := newDummyNode("A"), newDummyNode("B")
				// A has two incoming edges (Start + back-edge from B), but
				// the back-edge is conditional, so they don't fire together.
				return []Edge{
					{From: Start, To: a},
					{From: a, To: b},
					{From: b, To: a, Route: StringRoute("retry")},
				}
			},
			expectErr: false,
		},
		{
			name: "conditional fan-in not rejected",
			edges: func() []Edge {
				a, b, c, d := newDummyNode("A"), newDummyNode("B"), newDummyNode("C"), newDummyNode("D")
				// Only one of B/C routes into D at a time.
				return []Edge{
					{From: Start, To: a},
					{From: a, To: b},
					{From: a, To: c},
					{From: b, To: d, Route: StringRoute("x")},
					{From: c, To: d, Route: StringRoute("y")},
				}
			},
			expectErr: false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := validateFanIn(newGraph(tc.edges()))
			if tc.expectErr && !errors.Is(err, ErrUnsupportedFanIn) {
				t.Errorf("got %v, want ErrUnsupportedFanIn", err)
			} else if !tc.expectErr && err != nil {
				t.Errorf("expected no error, got %v", err)
			}
		})
	}
}

// violations flattens a validation error into the individual findings
// it carries. A phase joins the results of its checks, each of which is
// itself a join, so the tree has to be walked to the leaves — unwrapping
// one level would count failing checks, not violations.
func violations(err error) []error {
	if err == nil {
		return nil
	}
	multi, ok := err.(interface{ Unwrap() []error })
	if !ok {
		return []error{err}
	}
	var leaves []error
	for _, e := range multi.Unwrap() {
		leaves = append(leaves, violations(e)...)
	}
	return leaves
}

func violationMessages(err error) []string {
	var msgs []string
	for _, e := range violations(err) {
		msgs = append(msgs, e.Error())
	}
	return msgs
}

func TestValidateNodes_ReportsEveryViolation(t *testing.T) {
	// Two pairs of same-named nodes and two edges back into Start:
	// two checks, each with two findings, all from one call.
	a1, a2 := newDummyNode("A"), newDummyNode("A")
	b1, b2 := newDummyNode("B"), newDummyNode("B")
	c, d := newDummyNode("C"), newDummyNode("D")
	err := validateNodes([]Edge{
		{From: Start, To: a1},
		{From: a1, To: a2},
		{From: a2, To: b1},
		{From: b1, To: b2},
		{From: c, To: Start},
		{From: d, To: Start},
	})
	want := []string{
		"duplicate node name: A",
		"duplicate node name: B",
		"node points to start node: C",
		"node points to start node: D",
	}
	if diff := cmp.Diff(want, violationMessages(err)); diff != "" {
		t.Errorf("validateNodes() violations mismatch (-want +got):\n%s", diff)
	}
	for _, sentinel := range []error{ErrDuplicateNodeName, ErrNodePointsToStart} {
		if !errors.Is(err, sentinel) {
			t.Errorf("validateNodes() = %v, want it to match %v", err, sentinel)
		}
	}
}

// A single violation must come back exactly as it did before validation
// started aggregating: unjoined, one line, same message.
func TestNew_SingleViolationUnchanged(t *testing.T) {
	a, b := newDummyNode("A"), newDummyNode("B")
	_, err := New("wf", []Edge{{From: a, To: b}})
	if !errors.Is(err, ErrNoStartNode) {
		t.Fatalf("New() = %v, want it to match ErrNoStartNode", err)
	}
	if got, want := err.Error(), ErrNoStartNode.Error(); got != want {
		t.Errorf("New() error = %q, want %q (a lone violation must not be wrapped)", got, want)
	}
}

// Every back-edge into a node closing a cycle used to add a line, so a
// dense graph produced O(edges) identical findings.
func TestValidateCycles_OneFindingPerNode(t *testing.T) {
	// Complete graph: every node is on an unconditional cycle.
	const n = 20
	nodes := make([]Node, n)
	for i := range nodes {
		nodes[i] = newDummyNode(fmt.Sprintf("N%02d", i))
	}
	var edges []Edge
	for i, from := range nodes {
		for j, to := range nodes {
			if i != j {
				edges = append(edges, Edge{From: from, To: to})
			}
		}
	}
	got := len(violations(validateCycles(newGraph(edges))))
	if got == 0 || got > n {
		t.Errorf("validateCycles() reported %d findings for a %d-node complete graph, want 1..%d", got, n, n)
	}
}

func TestValidateWorkflow_ReportsEveryViolation(t *testing.T) {
	a, b, c := newDummyNode("A"), newDummyNode("B"), newDummyNode("C")
	d, e := newDummyNode("D"), newDummyNode("E")
	g := newGraph([]Edge{
		{From: Start, To: a},
		{From: a, To: b, Route: Default},
		{From: a, To: c, Route: Default}, // two default routes out of A
		{From: d, To: e},                 // unreachable pair, unconditionally cyclic
		{From: e, To: d},
	})
	err := validateWorkflow(g, nil)
	for _, want := range []error{ErrMultipleDefaultRoutes, ErrNodesNotReachable, ErrUnconditionalCycle} {
		if !errors.Is(err, want) {
			t.Errorf("validateWorkflow() = %v, want it to match %v", err, want)
		}
	}
	if got, want := len(violations(err)), 3; got != want {
		t.Errorf("validateWorkflow() reported %d violations, want %d: %v", got, want, err)
	}
}

// A failing phase stops the next one: later phases assume the earlier
// ones hold, so their findings would be noise.
func TestNew_StopsAfterFailingPhase(t *testing.T) {
	a1, a2, b := newDummyNode("A"), newDummyNode("A"), newDummyNode("B")
	_, err := New("wf", []Edge{
		{From: Start, To: a1},
		{From: a1, To: a2},
		{From: a1, To: b},
		{From: a1, To: b}, // duplicate edge: a later-phase violation
	})
	if !errors.Is(err, ErrDuplicateNodeName) {
		t.Fatalf("New() = %v, want it to match ErrDuplicateNodeName", err)
	}
	if errors.Is(err, ErrDuplicateEdge) {
		t.Errorf("New() = %v, want no graph-phase findings while the node phase fails", err)
	}
}

// Graph-level checks walk maps, so they sort by node name; without that
// the reported order (and for New, the error string) would vary per run.
func TestValidateWorkflow_StableViolationOrder(t *testing.T) {
	var edges []Edge
	for _, name := range []string{"C", "A", "D", "B"} {
		src := newDummyNode(name)
		edges = append(edges,
			Edge{From: src, To: newDummyNode(name + "1"), Route: Default},
			Edge{From: src, To: newDummyNode(name + "2"), Route: Default},
		)
	}
	want := []string{
		`node has more than one default route: "A"`,
		`node has more than one default route: "B"`,
		`node has more than one default route: "C"`,
		`node has more than one default route: "D"`,
	}
	for range 50 {
		got := violationMessages(validateDefaultRoute(newGraph(edges)))
		if diff := cmp.Diff(want, got); diff != "" {
			t.Fatalf("validateDefaultRoute() violations mismatch (-want +got):\n%s", diff)
		}
	}
}

// TestNew_NonJoinFanIn_Rejected confirms the fan-in check is wired into
// the public New constructor.
func TestNew_NonJoinFanIn_Rejected(t *testing.T) {
	a, b, c, d := newDummyNode("A"), newDummyNode("B"), newDummyNode("C"), newDummyNode("D")
	_, err := New("wf", []Edge{
		{From: Start, To: a},
		{From: a, To: b},
		{From: a, To: c},
		{From: b, To: d},
		{From: c, To: d},
	})
	if !errors.Is(err, ErrUnsupportedFanIn) {
		t.Errorf("New() error = %v, want ErrUnsupportedFanIn", err)
	}
}

func TestValidateSubWorkflowNames(t *testing.T) {
	// Create a valid sub-workflow
	subWf, err := New("inner_wf", []Edge{{From: Start, To: newDummyNode("A")}})
	if err != nil {
		t.Fatalf("failed to create sub-workflow: %v", err)
	}

	wfNode := &WorkflowNode{
		BaseNode:    NewBaseNode("nested_node", "", NodeConfig{}),
		subWorkflow: subWf,
	}

	tests := []struct {
		name           string
		parentName     string
		edges          []Edge
		expectErrorMsg string
	}{
		{
			name:       "no collision",
			parentName: "outer_wf",
			edges:      []Edge{{From: Start, To: wfNode}},
		},
		{
			name:           "collision with sub-workflow name",
			parentName:     "inner_wf",
			edges:          []Edge{{From: Start, To: wfNode}},
			expectErrorMsg: `sub-workflow name collision: "inner_wf"`,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := validateSubWorkflowNames(tc.parentName, tc.edges)
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

func TestDefaultValidateInput(t *testing.T) {
	// Set up schemas
	intSchema, err := (&jsonschema.Schema{Type: "integer"}).Resolve(nil)
	if err != nil {
		t.Fatalf("failed to resolve integer schema: %v", err)
	}

	enumSchema, err := (&jsonschema.Schema{
		Type: "string",
		Enum: []any{"yes", "no"},
	}).Resolve(nil)
	if err != nil {
		t.Fatalf("failed to resolve enum schema: %v", err)
	}

	structSchemaRaw, err := jsonschema.For[testValidationStruct](nil)
	if err != nil {
		t.Fatalf("failed to generate struct schema: %v", err)
	}
	structSchema, err := structSchemaRaw.Resolve(nil)
	if err != nil {
		t.Fatalf("failed to resolve struct schema: %v", err)
	}

	stringSchema, err := (&jsonschema.Schema{Type: "string"}).Resolve(nil)
	if err != nil {
		t.Fatalf("failed to resolve string schema: %v", err)
	}

	tests := []struct {
		name          string
		data          any
		schema        *jsonschema.Resolved
		want          any
		wantErrSubstr string
	}{
		{
			name:   "nil schema returns data as-is",
			data:   "hello",
			schema: nil,
			want:   "hello",
		},
		{
			name:   "nil data returns nil",
			data:   nil,
			schema: intSchema,
			want:   nil,
		},
		{
			name:   "string integer parsed as JSON integer",
			data:   "123",
			schema: intSchema,
			want:   float64(123), // JSON numbers unmarshal as float64 in any
		},
		{
			name:   "string matching enum element (raw fallback)",
			data:   "yes",
			schema: enumSchema,
			want:   "yes",
		},
		{
			name:          "plain text string failing integer schema",
			data:          "plain text",
			schema:        intSchema,
			wantErrSubstr: "loading",
		},
		{
			name: "map coerced to struct type via standard schema validation/conversion",
			data: map[string]any{
				"x": 42,
				"y": "hello",
			},
			schema: structSchema,
			want: map[string]any{
				"x": float64(42),
				"y": "hello",
			},
		},
		{
			name:   "string matching string schema does not attempt JSON parse",
			data:   "hello",
			schema: stringSchema,
			want:   "hello",
		},
		{
			name:          "valid JSON object missing required field",
			data:          `{"x":1}`,
			schema:        structSchema,
			wantErrSubstr: "missing properties",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := defaultValidateInput(tt.data, tt.schema)
			if tt.wantErrSubstr != "" {
				if err == nil {
					t.Fatalf("expected error containing %q, got nil", tt.wantErrSubstr)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			if diff := cmp.Diff(tt.want, got, cmpopts.EquateEmpty()); diff != "" {
				t.Errorf("defaultValidateInput() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

type testValidationStruct struct {
	X int    `json:"x"`
	Y string `json:"y"`
}

func TestSchemaIsString(t *testing.T) {
	tests := []struct {
		name   string
		schema *jsonschema.Schema
		want   bool
	}{
		{
			name:   "nil schema",
			schema: nil,
			want:   false,
		},
		{
			name: "type string",
			schema: &jsonschema.Schema{
				Type: "string",
			},
			want: true,
		},
		{
			name: "type integer",
			schema: &jsonschema.Schema{
				Type: "integer",
			},
			want: false,
		},
		{
			name: "types with string",
			schema: &jsonschema.Schema{
				Types: []string{"integer", "string"},
			},
			want: true,
		},
		{
			name: "types without string",
			schema: &jsonschema.Schema{
				Types: []string{"integer", "boolean"},
			},
			want: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var resolved *jsonschema.Resolved
			if tt.schema != nil {
				var err error
				resolved, err = tt.schema.Resolve(nil)
				if err != nil {
					t.Fatalf("failed to resolve schema: %v", err)
				}
			}
			got := schemaIsString(resolved)
			if got != tt.want {
				t.Errorf("schemaIsString() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestStaticSchemaValidation(t *testing.T) {
	type schemaTypeA struct {
		X int    `json:"x"`
		Y string `json:"y"`
	}
	type schemaTypeB struct {
		Val string `json:"val"`
	}

	schemaA, err := jsonschema.For[schemaTypeA](nil)
	if err != nil {
		t.Fatalf("failed to create schemaA: %v", err)
	}
	schemaAResolved, err := schemaA.Resolve(nil)
	if err != nil {
		t.Fatalf("failed to resolve schemaA: %v", err)
	}

	schemaB, err := jsonschema.For[schemaTypeB](nil)
	if err != nil {
		t.Fatalf("failed to create schemaB: %v", err)
	}
	schemaBResolved, err := schemaB.Resolve(nil)
	if err != nil {
		t.Fatalf("failed to resolve schemaB: %v", err)
	}

	// Schema A with custom PropertyOrder
	schemaA1 := &jsonschema.Schema{
		Type: "object",
		Properties: map[string]*jsonschema.Schema{
			"foo": {Type: "string"},
			"bar": {Type: "integer"},
		},
		PropertyOrder: []string{"foo", "bar"},
	}
	schemaA1Resolved, err := schemaA1.Resolve(nil)
	if err != nil {
		t.Fatalf("failed to resolve schemaA1: %v", err)
	}

	schemaA2 := &jsonschema.Schema{
		Type: "object",
		Properties: map[string]*jsonschema.Schema{
			"foo": {Type: "string"},
			"bar": {Type: "integer"},
		},
		PropertyOrder: []string{"bar", "foo"},
	}
	schemaA2Resolved, err := schemaA2.Resolve(nil)
	if err != nil {
		t.Fatalf("failed to resolve schemaA2: %v", err)
	}

	type compatTypeA struct {
		Foo string `json:"foo"`
		Bar int    `json:"bar"`
	}
	type compatTypeB struct {
		Foo string `json:"foo"`
		Bar int    `json:"bar"`
	}

	schemaCompatA, err := jsonschema.For[compatTypeA](nil)
	if err != nil {
		t.Fatalf("failed to create schemaCompatA: %v", err)
	}
	schemaCompatAResolved, err := schemaCompatA.Resolve(nil)
	if err != nil {
		t.Fatalf("failed to resolve schemaCompatA: %v", err)
	}

	schemaCompatB, err := jsonschema.For[compatTypeB](nil)
	if err != nil {
		t.Fatalf("failed to create schemaCompatB: %v", err)
	}
	schemaCompatBResolved, err := schemaCompatB.Resolve(nil)
	if err != nil {
		t.Fatalf("failed to resolve schemaCompatB: %v", err)
	}

	tests := []struct {
		name           string
		edges          func() []Edge
		expectErrorMsg string
	}{
		{
			name: "same Go type -> success",
			edges: func() []Edge {
				nodeA := &dummyNode{BaseNode: NewBaseNodeWithSchemas("A", "", NodeConfig{}, nil, schemaAResolved)}
				nodeB := &dummyNode{BaseNode: NewBaseNodeWithSchemas("B", "", NodeConfig{}, schemaAResolved, nil)}
				return []Edge{
					{From: Start, To: nodeA},
					{From: nodeA, To: nodeB},
				}
			},
		},
		{
			name: "different Go types -> error",
			edges: func() []Edge {
				nodeA := &dummyNode{BaseNode: NewBaseNodeWithSchemas("A", "", NodeConfig{}, nil, schemaAResolved)}
				nodeB := &dummyNode{BaseNode: NewBaseNodeWithSchemas("B", "", NodeConfig{}, schemaBResolved, nil)}
				return []Edge{
					{From: Start, To: nodeA},
					{From: nodeA, To: nodeB},
				}
			},
			expectErrorMsg: "schema mismatch on edge A -> B",
		},
		{
			name: "only one endpoint has schema -> success",
			edges: func() []Edge {
				nodeA := &dummyNode{BaseNode: NewBaseNodeWithSchemas("A", "", NodeConfig{}, nil, schemaAResolved)}
				nodeB := &dummyNode{BaseNode: NewBaseNodeWithSchemas("B", "", NodeConfig{}, nil, nil)}
				return []Edge{
					{From: Start, To: nodeA},
					{From: nodeA, To: nodeB},
				}
			},
		},
		{
			name: "differ only in PropertyOrder -> success",
			edges: func() []Edge {
				nodeA := &dummyNode{BaseNode: NewBaseNodeWithSchemas("A", "", NodeConfig{}, nil, schemaA1Resolved)}
				nodeB := &dummyNode{BaseNode: NewBaseNodeWithSchemas("B", "", NodeConfig{}, schemaA2Resolved, nil)}
				return []Edge{
					{From: Start, To: nodeA},
					{From: nodeA, To: nodeB},
				}
			},
		},
		{
			name: "different Go types with same fields -> success",
			edges: func() []Edge {
				nodeA := &dummyNode{BaseNode: NewBaseNodeWithSchemas("A", "", NodeConfig{}, nil, schemaCompatAResolved)}
				nodeB := &dummyNode{BaseNode: NewBaseNodeWithSchemas("B", "", NodeConfig{}, schemaCompatBResolved, nil)}
				return []Edge{
					{From: Start, To: nodeA},
					{From: nodeA, To: nodeB},
				}
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, err := New("test_wf", tc.edges())
			if tc.expectErrorMsg != "" {
				if err == nil {
					t.Fatalf("expected error containing %q, got nil", tc.expectErrorMsg)
				}
				if !strings.Contains(err.Error(), tc.expectErrorMsg) {
					t.Errorf("expected error to contain %q, got: %v", tc.expectErrorMsg, err)
				}
			} else {
				if err != nil {
					t.Fatalf("expected no error, got: %v", err)
				}
			}
		})
	}
}
