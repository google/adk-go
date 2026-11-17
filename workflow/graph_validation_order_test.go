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
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/google/jsonschema-go/jsonschema"

	"google.golang.org/adk/v2/internal/workflowroute"
)

// gcvUncomparableSchema resolves a schema whose Const holds a func, which
// resolves but cannot be marshaled, so two such schemas cannot be compared.
func gcvUncomparableSchema(t *testing.T) *jsonschema.Resolved {
	t.Helper()
	c := any(func() {})
	r, err := (&jsonschema.Schema{Const: &c}).Resolve(nil)
	if err != nil {
		t.Fatalf("Resolve(schema with a func Const): %v", err)
	}
	return r
}

// gcvInnerWorkflowNode returns a WorkflowNode whose nested workflow is valid
// and not empty.
func gcvInnerWorkflowNode(t *testing.T, name string) *WorkflowNode {
	t.Helper()
	n, err := NewWorkflowNode(name, Chain(Start, newDummyNode("x")))
	if err != nil {
		t.Fatalf("NewWorkflowNode(%q): %v", name, err)
	}
	return n
}

// For a graph with two defects, New reports the one whose rule comes first:
// P1 to P9 in adk-python's order, then the Go-only rules.
func TestNew_RuleOrder(t *testing.T) {
	a, b, c := newDummyNode("A"), newDummyNode("B"), newDummyNode("C")
	schemaOutX := gcvSchemaNode("A", nil, gcvTitledSchema(t, "X"))
	schemaInY := gcvSchemaNode("B", gcvTitledSchema(t, "Y"), nil)
	workflowroute.Register(func(node any, out workflowroute.OutgoingSummary) error {
		if fn, ok := node.(*failingEdgeValidatorNode); ok && len(out.Routes) > 0 {
			return errors.New("edge validation failed for " + fn.Name())
		}
		return nil
	})
	refusesRoutes := &failingEdgeValidatorNode{BaseNode: NewBaseNode("v", "", NodeConfig{})}
	sink := newDummyNode("sink")
	tests := []struct {
		name       string
		wfName     string
		edges      []Edge
		wantMsg    string
		absentText string
	}{
		{
			name:       "R1 duplicate names before missing Start",
			edges:      []Edge{{From: newDummyNode("A"), To: newDummyNode("A")}},
			wantMsg:    "duplicate node name: A",
			absentText: "no start node found",
		},
		{
			name:       "R2 routed Start edge before reachability",
			edges:      []Edge{{From: Start, To: a, Route: StringRoute("r")}, {From: b, To: a}},
			wantMsg:    `edge from "START" to "A" has a route; edges from START take none`,
			absentText: "nodes not reachable",
		},
		{
			name:       "R3 reachability before an edge into Start",
			edges:      []Edge{{From: Start, To: a}, {From: a, To: Start}, {From: b, To: a}},
			wantMsg:    `nodes not reachable from start node: "B"`,
			absentText: "node points to start node",
		},
		{
			name:       "R4 edge into Start before a duplicate edge",
			edges:      []Edge{{From: Start, To: a}, {From: a, To: Start}, {From: Start, To: a}},
			wantMsg:    "node points to start node: A",
			absentText: "duplicate edge",
		},
		{
			name: "R5 duplicate edge before Default in a MultiRoute",
			edges: []Edge{
				{From: Start, To: a},
				{From: a, To: b},
				{From: a, To: b, Route: StringRoute("x")},
				{From: b, To: c, Route: MultiRoute[Route]{StringRoute("y"), Default}},
			},
			wantMsg:    `duplicate edge: from "A" to "B"`,
			absentText: "a MultiRoute must not contain Default",
		},
		{
			name: "R6 Default in a MultiRoute before a cycle",
			edges: []Edge{
				{From: Start, To: a},
				{From: a, To: b, Route: MultiRoute[Route]{StringRoute("x"), Default}},
				{From: b, To: c},
				{From: c, To: b},
			},
			wantMsg:    `edge from "A" to "B": a MultiRoute must not contain Default`,
			absentText: "unconditional cycle detected",
		},
		{
			name:       "R7 cycle before a schema mismatch",
			edges:      []Edge{{From: Start, To: schemaOutX}, {From: schemaOutX, To: schemaInY}, {From: schemaInY, To: c}, {From: c, To: schemaInY}},
			wantMsg:    "unconditional cycle detected",
			absentText: "schema mismatch",
		},
		{
			name:       "R13 reachability before a sub-workflow name collision",
			wfName:     "inner",
			edges:      []Edge{{From: Start, To: gcvInnerWorkflowNode(t, "inner")}, {From: Start, To: a}, {From: b, To: a}},
			wantMsg:    `nodes not reachable from start node: "B"`,
			absentText: "sub-workflow name collision",
		},
		{
			name: "R14 reachability before an outgoing-edge validator",
			edges: []Edge{
				{From: Start, To: refusesRoutes},
				{From: refusesRoutes, To: sink, Route: StringRoute("x")},
				{From: newDummyNode("orphan"), To: sink},
			},
			wantMsg:    `nodes not reachable from start node: "orphan"`,
			absentText: "edge validation failed",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, err := New(tc.wfName, tc.edges)

			if err == nil {
				t.Fatalf("New() error = nil, want an error containing %q", tc.wantMsg)
			}
			if !strings.Contains(err.Error(), tc.wantMsg) {
				t.Errorf("New() error = %q, want it to contain %q", err, tc.wantMsg)
			}
			if strings.Contains(err.Error(), tc.absentText) {
				t.Errorf("New() error = %q, want no %q (that rule comes later)", err, tc.absentText)
			}
		})
	}
}

// Every graph rule's error matches ErrGraphValidation, keeps its specific
// sentinel, and carries the umbrella's prefix exactly once.
func TestNew_ErrGraphValidationMatches(t *testing.T) {
	a, b, c := newDummyNode("A"), newDummyNode("B"), newDummyNode("C")
	schemaOutX := gcvSchemaNode("A", nil, gcvTitledSchema(t, "X"))
	schemaInY := gcvSchemaNode("B", gcvTitledSchema(t, "Y"), nil)
	uncomparableOut := gcvSchemaNode("A", nil, gcvUncomparableSchema(t))
	tests := []struct {
		name     string
		wfName   string
		edges    []Edge
		sentinel error
	}{
		{
			name:     "P1 duplicate node names",
			edges:    Chain(Start, newDummyNode("A"), newDummyNode("A")),
			sentinel: ErrDuplicateNodeName,
		},
		{
			name:     "P2 no start node",
			edges:    Chain(a, b),
			sentinel: ErrNoStartNode,
		},
		{
			name:  "P3 routed edge from Start",
			edges: []Edge{{From: Start, To: a, Route: StringRoute("r")}},
		},
		{
			name:     "P4 unreachable node",
			edges:    []Edge{{From: Start, To: a}, {From: b, To: c}},
			sentinel: ErrNodesNotReachable,
		},
		{
			name:     "P5 edge into Start",
			edges:    []Edge{{From: Start, To: a}, {From: a, To: Start}},
			sentinel: ErrNodePointsToStart,
		},
		{
			name:     "P6 duplicate edge",
			edges:    []Edge{{From: Start, To: a}, {From: a, To: b}, {From: a, To: b}},
			sentinel: ErrDuplicateEdge,
		},
		{
			name:  "P7 Default in a MultiRoute",
			edges: []Edge{{From: Start, To: a}, {From: a, To: b, Route: MultiRoute[Route]{Default}}},
		},
		{
			name:     "P8 unconditional cycle",
			edges:    Chain(Start, a, b, a),
			sentinel: ErrUnconditionalCycle,
		},
		{
			name:  "P9 schema mismatch",
			edges: []Edge{{From: Start, To: schemaOutX}, {From: schemaOutX, To: schemaInY}},
		},
		{
			name:  "P9 schemas that cannot be compared",
			edges: []Edge{{From: Start, To: uncomparableOut}, {From: uncomparableOut, To: schemaInY}},
		},
		{
			name:     "G2 several Default edges",
			edges:    []Edge{{From: Start, To: a}, {From: a, To: b, Route: Default}, {From: a, To: c, Route: Default}},
			sentinel: ErrMultipleDefaultRoutes,
		},
		{
			name:     "G3 fan-in into a node that is not a JoinNode",
			edges:    []Edge{{From: Start, To: a}, {From: Start, To: b}, {From: a, To: c}, {From: b, To: c}},
			sentinel: ErrUnsupportedFanIn,
		},
		{
			name:     "G5 sub-workflow named like its parent",
			wfName:   "inner",
			edges:    []Edge{{From: Start, To: gcvInnerWorkflowNode(t, "inner")}},
			sentinel: ErrSubWorkflowNameCollision,
		},
		{
			name:  "P0 nil node",
			edges: []Edge{{From: Start, To: nil}},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, err := New(tc.wfName, tc.edges)

			if !errors.Is(err, ErrGraphValidation) {
				t.Fatalf("New() error = %v, want an error matching ErrGraphValidation", err)
			}
			if tc.sentinel != nil && !errors.Is(err, tc.sentinel) {
				t.Errorf("New() error = %v, want it to match %v as well", err, tc.sentinel)
			}
			if got := strings.Count(err.Error(), "graph validation failed: "); got != 1 || !strings.HasPrefix(err.Error(), "graph validation failed: ") {
				t.Errorf("New() error = %q, want the prefix %q exactly once, at the start", err, "graph validation failed: ")
			}
		})
	}

	t.Run("P9 schemas that cannot be compared name the edge and wrap the cause", func(t *testing.T) {
		_, err := New("w", []Edge{{From: Start, To: uncomparableOut}, {From: uncomparableOut, To: schemaInY}})

		const wantMsg = "comparing schemas on edge A->B"
		if err == nil || !strings.Contains(err.Error(), wantMsg) {
			t.Errorf("New() error = %v, want it to contain %q", err, wantMsg)
		}
		if _, ok := errors.AsType[*json.MarshalerError](err); !ok {
			t.Errorf("New() error = %v, want a *json.MarshalerError in its chain", err)
		}
	})
}

// Errors that do not come from the shape of the edge list do not match
// ErrGraphValidation.
func TestNew_ErrGraphValidationDoesNotMatchOtherErrors(t *testing.T) {
	t.Run("WithNodeRootWrapper edge count", func(t *testing.T) {
		_, err := New("w", nil, WithNodeRootWrapper())

		if err == nil {
			t.Fatal("New() error = nil, want the WithNodeRootWrapper edge-count error")
		}
		if errors.Is(err, ErrGraphValidation) {
			t.Errorf("New() error = %v, want no match with ErrGraphValidation", err)
		}
		if !strings.Contains(err.Error(), "needs exactly one edge") {
			t.Errorf("New() error = %q, want it to contain %q", err, "needs exactly one edge")
		}
	})

	t.Run("state field not declared in the state schema", func(t *testing.T) {
		stateSchema := &jsonschema.Schema{
			Type:       "object",
			Properties: map[string]*jsonschema.Schema{"Foo": {Type: "string"}},
		}
		resolved, err := stateSchema.Resolve(nil)
		if err != nil {
			t.Fatalf("Resolve: %v", err)
		}
		x1, err := NewFunctionNodeFromState("x1", dummyFnInvalid, NodeConfig{})
		if err != nil {
			t.Fatalf("NewFunctionNodeFromState: %v", err)
		}

		_, err = New("w", []Edge{{From: Start, To: x1}}, WithStateSchema(resolved))

		if err == nil {
			t.Fatal("New() error = nil, want the state-schema error")
		}
		if errors.Is(err, ErrGraphValidation) {
			t.Errorf("New() error = %v, want no match with ErrGraphValidation", err)
		}
	})
}
