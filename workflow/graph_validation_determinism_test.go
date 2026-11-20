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

	"github.com/google/jsonschema-go/jsonschema"

	"google.golang.org/adk/v2/agent"
)

// gcvDeterminismCalls is how many times each test below builds the same
// graph. On these graphs a report that depends on map order names the other
// defect in about one call in eight or more often, so 50 calls that all
// name the same one leave well under a 1 in 100 chance of missing it.
const gcvDeterminismCalls = 50

// gcvNewErrors calls New gcvDeterminismCalls times on the same edges and
// returns the errors, failing the test if any call succeeds.
func gcvNewErrors(t *testing.T, edges []Edge, opts ...Option) []error {
	t.Helper()
	errs := make([]error, 0, gcvDeterminismCalls)
	for range gcvDeterminismCalls {
		_, err := New("w", edges, opts...)
		if err == nil {
			t.Fatal("New() error = nil, want an error")
		}
		errs = append(errs, err)
	}
	return errs
}

type gcvBarParams struct {
	Bar string `state:"bar"`
}

func gcvFnBar(agent.InvocationContext, gcvBarParams) (string, error) {
	return "ok", nil
}

// Of two cycles, the one reached first from Start in declaration order is
// reported, on every call.
func TestNew_CycleReportDeterministic(t *testing.T) {
	a, b, c, d := newDummyNode("A"), newDummyNode("B"), newDummyNode("C"), newDummyNode("D")
	edges := []Edge{
		{From: Start, To: c},
		{From: c, To: d},
		{From: d, To: c},
		{From: Start, To: a},
		{From: a, To: b},
		{From: b, To: a},
	}

	for i, err := range gcvNewErrors(t, edges) {
		if !errors.Is(err, ErrUnconditionalCycle) {
			t.Fatalf("call %d: New() error = %v, want an error matching ErrUnconditionalCycle", i, err)
		}
		if !strings.Contains(err.Error(), `unconditional cycle detected: "C" -> "D" -> "C"`) || strings.Contains(err.Error(), `"A" -> "B"`) {
			t.Fatalf("call %d: New() error = %q, want the cycle \"C\" -> \"D\" -> \"C\"", i, err)
		}
	}
}

// The reported duplicate is the first edge in declaration order that repeats
// an earlier edge's (From, To), on every call.
func TestNew_DuplicateEdgeReportDeterministic(t *testing.T) {
	a, b, c := newDummyNode("A"), newDummyNode("B"), newDummyNode("C")
	edges := []Edge{{From: Start, To: a}, {From: a, To: b}, {From: b, To: c}, {From: b, To: c}, {From: a, To: b}}

	for i, err := range gcvNewErrors(t, edges) {
		if !errors.Is(err, ErrDuplicateEdge) {
			t.Fatalf("call %d: New() error = %v, want an error matching ErrDuplicateEdge", i, err)
		}
		if !strings.Contains(err.Error(), `duplicate edge: from "B" to "C"`) {
			t.Fatalf("call %d: New() error = %q, want it to contain %q", i, err, `duplicate edge: from "B" to "C"`)
		}
	}
}

func TestNew_SchemaMismatchReportDeterministic(t *testing.T) {
	t.Run("first mismatched edge in declaration order", func(t *testing.T) {
		a := gcvSchemaNode("A", nil, gcvTitledSchema(t, "X"))
		b := gcvSchemaNode("B", gcvTitledSchema(t, "Y"), nil)
		c := gcvSchemaNode("C", nil, gcvTitledSchema(t, "P"))
		d := gcvSchemaNode("D", gcvTitledSchema(t, "Q"), nil)
		edges := []Edge{{From: Start, To: c}, {From: c, To: d}, {From: Start, To: a}, {From: a, To: b}}

		for i, err := range gcvNewErrors(t, edges) {
			const want = "graph validation failed: schema mismatch on edge C -> D"
			if err.Error() != want {
				t.Fatalf("call %d: New() error = %q, want %q", i, err, want)
			}
		}
	})

	t.Run("schema on one side only is not compared", func(t *testing.T) {
		a := gcvSchemaNode("A", nil, gcvTitledSchema(t, "X"))
		b := gcvSchemaNode("B", nil, nil)

		_, err := New("w", []Edge{{From: Start, To: a}, {From: a, To: b}})
		if err != nil {
			t.Errorf("New() error = %v, want nil", err)
		}
	})
}

// Of two nodes that use undeclared state fields, the first mentioned is
// reported, on every call, and the error is not a graph validation error.
func TestNew_StateSchemaReportDeterministic(t *testing.T) {
	stateSchema, err := (&jsonschema.Schema{
		Type:       "object",
		Properties: map[string]*jsonschema.Schema{"Foo": {Type: "string"}},
	}).Resolve(nil)
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}
	x1, err := NewFunctionNodeFromState("x1", dummyFnInvalid, NodeConfig{})
	if err != nil {
		t.Fatalf("NewFunctionNodeFromState(x1): %v", err)
	}
	x2, err := NewFunctionNodeFromState("x2", gcvFnBar, NodeConfig{})
	if err != nil {
		t.Fatalf("NewFunctionNodeFromState(x2): %v", err)
	}
	edges := []Edge{{From: Start, To: x2}, {From: x2, To: x1}}

	for i, err := range gcvNewErrors(t, edges, WithStateSchema(stateSchema)) {
		if !strings.Contains(err.Error(), `node "x2" references state field "bar"`) {
			t.Fatalf("call %d: New() error = %q, want it to contain %q", i, err, `node "x2" references state field "bar"`)
		}
		if errors.Is(err, ErrGraphValidation) {
			t.Fatalf("call %d: New() error = %v, want no match with ErrGraphValidation", i, err)
		}
	}
}

// Of two nodes with several Default edges, the first mentioned is reported,
// on every call.
func TestNew_DefaultRouteReportDeterministic(t *testing.T) {
	a, b, x, y := newDummyNode("A"), newDummyNode("B"), newDummyNode("X"), newDummyNode("Y")
	edges := []Edge{
		{From: Start, To: a},
		{From: Start, To: b},
		{From: a, To: x, Route: Default},
		{From: a, To: y, Route: Default},
		{From: b, To: x, Route: Default},
		{From: b, To: y, Route: Default},
	}

	for i, err := range gcvNewErrors(t, edges) {
		if !errors.Is(err, ErrMultipleDefaultRoutes) {
			t.Fatalf("call %d: New() error = %v, want an error matching ErrMultipleDefaultRoutes", i, err)
		}
		if !strings.Contains(err.Error(), `"A"`) || strings.Contains(err.Error(), `"B"`) {
			t.Fatalf("call %d: New() error = %q, want it to name \"A\" and not \"B\"", i, err)
		}
	}
}

// Of two fan-in targets, the first mentioned is reported, on every call.
func TestNew_FanInReportDeterministic(t *testing.T) {
	a, b, c, d := newDummyNode("A"), newDummyNode("B"), newDummyNode("C"), newDummyNode("D")
	edges := []Edge{
		{From: Start, To: a},
		{From: Start, To: b},
		{From: a, To: c},
		{From: b, To: c},
		{From: a, To: d},
		{From: b, To: d},
	}

	for i, err := range gcvNewErrors(t, edges) {
		if !errors.Is(err, ErrUnsupportedFanIn) {
			t.Fatalf("call %d: New() error = %v, want an error matching ErrUnsupportedFanIn", i, err)
		}
		if !strings.Contains(err.Error(), `node "C"`) || strings.Contains(err.Error(), `node "D"`) {
			t.Fatalf("call %d: New() error = %q, want it to name node \"C\" and not node \"D\"", i, err)
		}
	}
}
