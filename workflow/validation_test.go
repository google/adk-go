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

	"google.golang.org/adk/v2/agent"
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
		{
			name: "duplicate edge is not fan-in",
			edges: func() []Edge {
				a, b := newDummyNode("A"), newDummyNode("B")
				// B has two incoming edges but one predecessor. That is a
				// duplicate edge, which validateUniqueEdges reports on
				// its own. A JoinNode would not resolve it.
				return []Edge{
					{From: Start, To: a},
					{From: a, To: b},
					{From: a, To: b},
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
// started aggregating. The message alone does not pin this: errors.Join
// of one error renders identically, so identity is what has to be
// asserted.
func TestNew_SingleViolationUnchanged(t *testing.T) {
	a, b := newDummyNode("A"), newDummyNode("B")
	_, err := New("wf", []Edge{{From: a, To: b}})
	//nolint:errorlint // pointer identity is the property under test
	if err != ErrNoStartNode {
		t.Errorf("New() = %#v, want the ErrNoStartNode value itself, unwrapped", err)
	}
	if _, joined := err.(interface{ Unwrap() []error }); joined { //nolint:errorlint // ditto
		t.Errorf("New() = %v, want a lone violation not to be joined", err)
	}
	if got, want := err.Error(), ErrNoStartNode.Error(); got != want {
		t.Errorf("New() error = %q, want %q", got, want)
	}

	// A wrapped lone violation keeps errors.Unwrap too.
	c, d := newDummyNode("C"), newDummyNode("C")
	_, err = New("wf", []Edge{{From: Start, To: c}, {From: c, To: d}})
	if got := errors.Unwrap(err); got != ErrDuplicateNodeName {
		t.Errorf("errors.Unwrap(New() error) = %v, want ErrDuplicateNodeName", got)
	}
}

// Independent cycles are each reported, and a node that several
// back-edges close on is reported once. Returning at the first cycle
// found — as this check used to — would report exactly one of A and C,
// so asserting both is what pins the change.
func TestValidateCycles_OneFindingPerClosingNode(t *testing.T) {
	a, b, c, d, e := newDummyNode("A"), newDummyNode("B"), newDummyNode("C"),
		newDummyNode("D"), newDummyNode("E")
	g := newGraph([]Edge{
		{From: Start, To: a},
		{From: a, To: b},
		{From: b, To: a}, // one back-edge closes on A
		{From: Start, To: c},
		{From: c, To: d},
		{From: d, To: c}, // two back-edges close on C
		{From: c, To: e},
		{From: e, To: c},
	})
	want := []string{
		`unconditional cycle detected: "A"`,
		`unconditional cycle detected: "C"`,
	}
	if diff := cmp.Diff(want, violationMessages(validateCycles(g))); diff != "" {
		t.Errorf("validateCycles() violations mismatch (-want +got):\n%s", diff)
	}
}

// A dense graph has O(edges) back-edges, but only n-1 nodes close a
// cycle in this DFS, and joinViolations collapses the rest. The base
// returned 1 finding for any cyclic graph, so the exact count is what
// distinguishes them.
func TestValidateCycles_FindingCountIsLinearInNodes(t *testing.T) {
	// Complete graph: every node is on an unconditional cycle. The
	// deepest node of the DFS closes nothing, so n-1 nodes are reported.
	const n = 20
	edges := completeGraphEdges(n)
	if got, want := len(violations(validateCycles(newGraph(edges)))), n-1; got != want {
		t.Errorf("validateCycles() reported %d findings for a %d-node complete graph (%d edges), want %d",
			got, n, len(edges), want)
	}
}

// joinViolations would collapse a per-back-edge report to the same
// output, so the in-traversal dedup is invisible in the error and only
// an allocation count can pin it. Measured on this 60-node graph: 229
// allocations with the dedup, 5359 without.
func TestValidateCycles_DoesNotAllocatePerBackEdge(t *testing.T) {
	const n = 60
	g := newGraph(completeGraphEdges(n))
	const limit = 1000
	if got := testing.AllocsPerRun(5, func() { _ = validateCycles(g) }); got > limit {
		t.Errorf("validateCycles() on a %d-node complete graph allocated %.0f times, want <= %d",
			n, got, limit)
	}
}

// A node referencing several undeclared fields gets one finding, not
// one per field: the declared-field list is identical for all of them,
// and repeating it made a wide schema's error unreadable.
func TestValidateStateSchemaConsistency_OneFindingPerNode(t *testing.T) {
	schema, err := (&jsonschema.Schema{
		Type:       "object",
		Properties: map[string]*jsonschema.Schema{"Foo": {Type: "string"}},
	}).Resolve(nil)
	if err != nil {
		t.Fatalf("resolving schema: %v", err)
	}
	node, err := NewFunctionNodeFromState("n", dummyFnTwoUndeclared, NodeConfig{})
	if err != nil {
		t.Fatalf("NewFunctionNodeFromState: %v", err)
	}
	g := newGraph([]Edge{{From: Start, To: node}})
	want := []string{
		`node "n" references state fields "bar", "baz" which are not declared in StateSchema (declared: [Foo])`,
	}
	got := violationMessages(validateStateSchemaConsistency(g, schema))
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("validateStateSchemaConsistency() violations mismatch (-want +got):\n%s", diff)
	}
}

type twoUndeclaredParams struct {
	Bar string `state:"bar"`
	Baz string `state:"baz"`
}

// dummyFnTwoUndeclared reads two state fields that no schema here declares.
func dummyFnTwoUndeclared(ctx agent.InvocationContext, p twoUndeclaredParams) (string, error) {
	return "ok", nil
}

// completeGraphEdges returns the edges of a complete graph on n nodes,
// in which every node lies on an unconditional cycle.
func completeGraphEdges(n int) []Edge {
	nodes := make([]Node, n)
	for i := range nodes {
		nodes[i] = newDummyNode(fmt.Sprintf("N%03d", i))
	}
	var edges []Edge
	for i, from := range nodes {
		for j, to := range nodes {
			if i != j {
				edges = append(edges, Edge{From: from, To: to})
			}
		}
	}
	return edges
}

// joinViolations deduplicates by message, so a finding whose message
// interpolates two names must quote them. Unquoted, these two edges
// render identically and one real violation is dropped.
func TestValidateStaticSchemas_NamesAreDelimited(t *testing.T) {
	intSchema, err := (&jsonschema.Schema{Type: "integer"}).Resolve(nil)
	if err != nil {
		t.Fatalf("resolving int schema: %v", err)
	}
	strSchema, err := (&jsonschema.Schema{Type: "string"}).Resolve(nil)
	if err != nil {
		t.Fatalf("resolving string schema: %v", err)
	}
	out := func(name string) Node {
		return &dummyNode{BaseNode: NewBaseNodeWithSchemas(name, "", NodeConfig{}, nil, intSchema)}
	}
	in := func(name string) Node {
		return &dummyNode{BaseNode: NewBaseNodeWithSchemas(name, "", NodeConfig{}, strSchema, nil)}
	}
	got := violationMessages(validateStaticSchemas(newGraph([]Edge{
		{From: out("A"), To: in("B -> C")},
		{From: out("A -> B"), To: in("C")},
	})))
	want := []string{
		`graph validation failed: schema mismatch on edge "A" -> "B -> C"`,
		`graph validation failed: schema mismatch on edge "A -> B" -> "C"`,
	}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("validateStaticSchemas() violations mismatch (-want +got):\n%s", diff)
	}
}

// A phase joins its checks without deduplicating: two checks never
// render the same message, and asking would render each check's whole
// joined text. Dropping the dedup here must not drop a finding.
func TestJoinChecks_KeepsEveryCheck(t *testing.T) {
	a, b := errors.New("same message"), errors.New("same message")
	if got, want := len(violations(joinChecks(a, nil, b))), 2; got != want {
		t.Errorf("joinChecks() kept %d errors, want %d", got, want)
	}
	if got := joinChecks(nil, nil); got != nil {
		t.Errorf("joinChecks(nil, nil) = %v, want nil", got)
	}
	if got := joinChecks(nil, a); got != a { //nolint:errorlint // identity is the property
		t.Errorf("joinChecks(nil, a) = %v, want a unwrapped", got)
	}
}

// Two edges breaking one rule the same way read as one problem.
func TestJoinViolations_DropsRepeatedMessages(t *testing.T) {
	a, b := newDummyNode("A"), newDummyNode("B")
	err := validateStartNodeNoIncoming([]Edge{
		{From: a, To: Start},
		{From: a, To: Start}, // same message as the edge above
		{From: b, To: Start},
	})
	want := []string{
		"node points to start node: A",
		"node points to start node: B",
	}
	if diff := cmp.Diff(want, violationMessages(err)); diff != "" {
		t.Errorf("validateStartNodeNoIncoming() violations mismatch (-want +got):\n%s", diff)
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

// A check must not report a violation that exists only because of
// another check's finding. Both cases are one mistake — a copy-pasted
// edge, and a loop-back — and the second finding would point the caller
// at a change that does not fix the graph.
func TestValidateWorkflow_NoDerivativeFindings(t *testing.T) {
	tests := []struct {
		name    string
		edges   func() []Edge
		want    []string
		wantNot error
	}{
		{
			name: "duplicated default edge is not multiple default routes",
			edges: func() []Edge {
				a, b := newDummyNode("A"), newDummyNode("B")
				return []Edge{
					{From: Start, To: a},
					{From: a, To: b, Route: Default},
					{From: a, To: b, Route: Default}, // the one mistake
				}
			},
			want:    []string{`duplicate edge: from "A" to "B"`},
			wantNot: ErrMultipleDefaultRoutes,
		},
		{
			name: "unconditional self-loop is not fan-in",
			edges: func() []Edge {
				b := newDummyNode("B")
				return []Edge{{From: Start, To: b}, {From: b, To: b}}
			},
			want:    []string{`unconditional cycle detected: "B"`},
			wantNot: ErrUnsupportedFanIn,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := validateWorkflow(newGraph(tc.edges()), nil)
			if diff := cmp.Diff(tc.want, violationMessages(err)); diff != "" {
				t.Errorf("validateWorkflow() violations mismatch (-want +got):\n%s", diff)
			}
			if errors.Is(err, tc.wantNot) {
				t.Errorf("validateWorkflow() = %v, want it not to match %v", err, tc.wantNot)
			}
		})
	}
}

// Two default routes to different targets are still rejected: that is
// the ambiguity the check exists for.
func TestValidateDefaultRoute_DistinctTargetsStillRejected(t *testing.T) {
	a, b, c := newDummyNode("A"), newDummyNode("B"), newDummyNode("C")
	err := validateDefaultRoute(newGraph([]Edge{
		{From: Start, To: a},
		{From: a, To: b, Route: Default},
		{From: a, To: c, Route: Default},
	}))
	if !errors.Is(err, ErrMultipleDefaultRoutes) {
		t.Errorf("validateDefaultRoute() = %v, want ErrMultipleDefaultRoutes", err)
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

// Graph-level checks walk maps, so they sort by node name. Without
// that the reported order — and for New, the error string — would vary
// from run to run.
// Every check that reaches the graph through sortedNodes or allEdges is
// covered, so a single one reverting to direct map iteration fails here.
func TestValidateWorkflow_StableViolationOrder(t *testing.T) {
	// Declaration order is deliberately not alphabetical: it is what a
	// findings list would inherit if a check walked the edge slice, and
	// what the map-iteration order is seeded from.
	const unsorted = "CAB"

	stateSchema, err := (&jsonschema.Schema{
		Type:       "object",
		Properties: map[string]*jsonschema.Schema{"Foo": {Type: "string"}},
	}).Resolve(nil)
	if err != nil {
		t.Fatalf("resolving state schema: %v", err)
	}
	intSchema, err := (&jsonschema.Schema{Type: "integer"}).Resolve(nil)
	if err != nil {
		t.Fatalf("resolving int schema: %v", err)
	}
	strSchema, err := (&jsonschema.Schema{Type: "string"}).Resolve(nil)
	if err != nil {
		t.Fatalf("resolving string schema: %v", err)
	}

	tests := []struct {
		name  string
		edges []Edge
		check func(*graph) error
		want  []string
	}{
		{
			name: "validateUniqueEdges",
			edges: edgesPerName(unsorted, func(name string) []Edge {
				src, dst := newDummyNode(name), newDummyNode(name+"_t")
				return []Edge{{From: src, To: dst}, {From: src, To: dst}}
			}),
			check: validateUniqueEdges,
			want: []string{
				`duplicate edge: from "A" to "A_t"`,
				`duplicate edge: from "B" to "B_t"`,
				`duplicate edge: from "C" to "C_t"`,
			},
		},
		{
			name: "validateDefaultRoute",
			edges: edgesPerName(unsorted, func(name string) []Edge {
				src := newDummyNode(name)
				return []Edge{
					{From: src, To: newDummyNode(name + "1"), Route: Default},
					{From: src, To: newDummyNode(name + "2"), Route: Default},
				}
			}),
			check: validateDefaultRoute,
			want: []string{
				`node has more than one default route: "A"`,
				`node has more than one default route: "B"`,
				`node has more than one default route: "C"`,
			},
		},
		{
			name: "validateCycles",
			edges: edgesPerName(unsorted, func(name string) []Edge {
				src, loop := newDummyNode(name), newDummyNode(name+"_loop")
				return []Edge{{From: src, To: loop}, {From: loop, To: src}}
			}),
			check: validateCycles,
			want: []string{
				`unconditional cycle detected: "A"`,
				`unconditional cycle detected: "B"`,
				`unconditional cycle detected: "C"`,
			},
		},
		{
			name: "validateFanIn",
			edges: edgesPerName(unsorted, func(name string) []Edge {
				dst := newDummyNode(name)
				return []Edge{
					{From: newDummyNode(name + "_p1"), To: dst},
					{From: newDummyNode(name + "_p2"), To: dst},
				}
			}),
			check: validateFanIn,
			want: []string{
				`non-JoinNode fan-in is not yet supported: node "A" has 2 unconditional predecessors; use a JoinNode to converge branches`,
				`non-JoinNode fan-in is not yet supported: node "B" has 2 unconditional predecessors; use a JoinNode to converge branches`,
				`non-JoinNode fan-in is not yet supported: node "C" has 2 unconditional predecessors; use a JoinNode to converge branches`,
			},
		},
		{
			name: "validateStaticSchemas",
			edges: edgesPerName(unsorted, func(name string) []Edge {
				src := &dummyNode{BaseNode: NewBaseNodeWithSchemas(name, "", NodeConfig{}, nil, intSchema)}
				dst := &dummyNode{BaseNode: NewBaseNodeWithSchemas(name+"_t", "", NodeConfig{}, strSchema, nil)}
				return []Edge{{From: src, To: dst}}
			}),
			check: validateStaticSchemas,
			want: []string{
				`graph validation failed: schema mismatch on edge "A" -> "A_t"`,
				`graph validation failed: schema mismatch on edge "B" -> "B_t"`,
				`graph validation failed: schema mismatch on edge "C" -> "C_t"`,
			},
		},
		{
			name: "validateStateSchemaConsistency",
			edges: edgesPerName(unsorted, func(name string) []Edge {
				// dummyFnInvalid reads state field "foo", which the
				// schema does not declare.
				src, err := NewFunctionNodeFromState(name, dummyFnInvalid, NodeConfig{})
				if err != nil {
					t.Fatalf("NewFunctionNodeFromState(%q): %v", name, err)
				}
				return []Edge{{From: src, To: newDummyNode(name + "_t")}}
			}),
			check: func(g *graph) error { return validateStateSchemaConsistency(g, stateSchema) },
			want: []string{
				`node "A" references state field "foo" which is not declared in StateSchema (declared: [Foo])`,
				`node "B" references state field "foo" which is not declared in StateSchema (declared: [Foo])`,
				`node "C" references state field "foo" which is not declared in StateSchema (declared: [Foo])`,
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			// Repeated: a check walking a map unsorted can still land on
			// the right order by chance on any single run.
			for range 50 {
				got := violationMessages(tc.check(newGraph(tc.edges)))
				if diff := cmp.Diff(tc.want, got); diff != "" {
					t.Fatalf("violations mismatch (-want +got):\n%s", diff)
				}
			}
		})
	}
}

// edgesPerName builds the edges for one violation per name in names.
func edgesPerName(names string, build func(name string) []Edge) []Edge {
	var edges []Edge
	for _, r := range names {
		edges = append(edges, build(string(r))...)
	}
	return edges
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
			if tc.expectErrorMsg == "" {
				if err != nil {
					t.Errorf("expected no error, got %v", err)
				}
				return
			}
			if err == nil {
				t.Fatalf("expected error %q, got none", tc.expectErrorMsg)
			}
			// Exact, not a prefix. NewWorkflowNode gives a node and its
			// sub-workflow the same name, so a collision through New
			// means the node is named after the parent too and a
			// per-node suffix would only repeat the name already here.
			// A second colliding node is unreachable through New for
			// the same reason: validateUniqueNames rejects the two
			// same-named nodes a phase earlier.
			if got := err.Error(); got != tc.expectErrorMsg {
				t.Errorf("error = %q, want %q", got, tc.expectErrorMsg)
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

	// Same fields declared in a different order. jsonschema.For builds the
	// "required" list in field order, so these produce ["foo","bar"] and
	// ["bar","foo"] respectively while being otherwise identical.
	type reorderTypeA struct {
		Foo string `json:"foo"`
		Bar int    `json:"bar"`
	}
	type reorderTypeB struct {
		Bar int    `json:"bar"`
		Foo string `json:"foo"`
	}

	schemaReorderA, err := jsonschema.For[reorderTypeA](nil)
	if err != nil {
		t.Fatalf("failed to create schemaReorderA: %v", err)
	}
	schemaReorderAResolved, err := schemaReorderA.Resolve(nil)
	if err != nil {
		t.Fatalf("failed to resolve schemaReorderA: %v", err)
	}

	schemaReorderB, err := jsonschema.For[reorderTypeB](nil)
	if err != nil {
		t.Fatalf("failed to create schemaReorderB: %v", err)
	}
	schemaReorderBResolved, err := schemaReorderB.Resolve(nil)
	if err != nil {
		t.Fatalf("failed to resolve schemaReorderB: %v", err)
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
			expectErrorMsg: `schema mismatch on edge "A" -> "B"`,
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
		{
			name: "differ only in required order -> success",
			edges: func() []Edge {
				nodeA := &dummyNode{BaseNode: NewBaseNodeWithSchemas("A", "", NodeConfig{}, nil, schemaReorderAResolved)}
				nodeB := &dummyNode{BaseNode: NewBaseNodeWithSchemas("B", "", NodeConfig{}, schemaReorderBResolved, nil)}
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
