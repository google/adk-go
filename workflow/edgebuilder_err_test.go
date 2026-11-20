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
	"reflect"
	"strings"
	"testing"

	"google.golang.org/adk/v2/session"
)

// edgeString names an edge's endpoints without calling Name on a nil node,
// so a failure message can show an edge whose To is nil.
func edgeString(e Edge) string {
	name := func(n Node) string {
		if n == nil {
			return "<nil>"
		}
		return n.Name()
	}
	return name(e.From) + "→" + name(e.To)
}

func assertEdges(t *testing.T, got, want []Edge) {
	t.Helper()
	if len(got) != len(want) {
		gotNames := make([]string, len(got))
		for i, e := range got {
			gotNames[i] = edgeString(e)
		}
		t.Fatalf("got %d edges %v, want %d", len(got), gotNames, len(want))
	}
	for i := range want {
		if got[i].From != want[i].From || got[i].To != want[i].To || !reflect.DeepEqual(got[i].Route, want[i].Route) {
			t.Errorf("edge %d = %s (route %v), want %s (route %v)",
				i, edgeString(got[i]), got[i].Route, edgeString(want[i]), want[i].Route)
		}
	}
}

func TestEdgeBuilder_EmptyRoutingMapSetsError(t *testing.T) {
	classifier := newDummyNode("classifier")
	a := newDummyNode("A")
	b := newDummyNode("B")
	c := newDummyNode("C")

	t.Run("zero value has no error", func(t *testing.T) {
		var eb EdgeBuilder
		if err := eb.Err(); err != nil {
			t.Errorf("Err() = %v, want nil", err)
		}
		assertEdges(t, eb.Build(), nil)
	})

	t.Run("NewEdgeBuilder has no error", func(t *testing.T) {
		eb := NewEdgeBuilder()
		if err := eb.Err(); err != nil {
			t.Errorf("Err() = %v, want nil", err)
		}
		assertEdges(t, eb.Build(), nil)
	})

	t.Run("non-empty map records no error", func(t *testing.T) {
		eb := NewEdgeBuilder().AddRoutes(a, map[string]Node{"r1": b})
		if err := eb.Err(); err != nil {
			t.Errorf("Err() = %v, want nil", err)
		}
		assertEdges(t, eb.Build(), []Edge{{From: a, To: b, Route: StringRoute("r1")}})
	})

	for _, tc := range []struct {
		name   string
		routes map[string]Node
	}{
		{name: "nil map", routes: nil},
		{name: "empty map", routes: map[string]Node{}},
	} {
		t.Run(tc.name+" records ErrEmptyRoutingMap", func(t *testing.T) {
			eb := NewEdgeBuilder().AddRoutes(classifier, tc.routes)
			err := eb.Err()
			if !errors.Is(err, ErrEmptyRoutingMap) {
				t.Fatalf("Err() = %v, want an error matching ErrEmptyRoutingMap", err)
			}
			if !strings.Contains(err.Error(), "classifier") {
				t.Errorf("Err() = %q, want it to name the source node %q", err, "classifier")
			}
			assertEdges(t, eb.Build(), []Edge{
				{Route: refusedEdge{err: fmt.Errorf("%w: from node %q", ErrEmptyRoutingMap, "classifier")}},
			})
		})
	}

	t.Run("later calls keep the error and add their edges", func(t *testing.T) {
		eb := NewEdgeBuilder().
			Add(Start, a).
			AddRoutes(a, map[string]Node{}).
			Add(a, b).
			AddRoutes(b, map[string]Node{"r": c})
		if err := eb.Err(); !errors.Is(err, ErrEmptyRoutingMap) {
			t.Errorf("Err() = %v, want an error matching ErrEmptyRoutingMap", err)
		}
		assertEdges(t, eb.Build(), []Edge{
			{From: Start, To: a},
			{Route: refusedEdge{err: fmt.Errorf("%w: from node %q", ErrEmptyRoutingMap, "A")}},
			{From: a, To: b},
			{From: b, To: c, Route: StringRoute("r")},
		})
	})

	t.Run("nil source node", func(t *testing.T) {
		eb := NewEdgeBuilder().AddRoutes(nil, nil)
		if err := eb.Err(); !errors.Is(err, ErrEmptyRoutingMap) {
			t.Errorf("Err() = %v, want an error matching ErrEmptyRoutingMap", err)
		}
	})
}

func TestEdgeBuilder_ErrPreservesFirstError(t *testing.T) {
	classifier := newDummyNode("classifier")
	escalator := newDummyNode("escalator")

	eb := NewEdgeBuilder().
		AddRoutes(classifier, nil).
		AddRoutes(escalator, map[string]Node{})

	err := eb.Err()
	if !errors.Is(err, ErrEmptyRoutingMap) {
		t.Fatalf("Err() = %v, want an error matching ErrEmptyRoutingMap", err)
	}
	if !strings.Contains(err.Error(), "classifier") {
		t.Errorf("Err() = %q, want it to name the first empty map's source %q", err, "classifier")
	}
	if strings.Contains(err.Error(), "escalator") {
		t.Errorf("Err() = %q, want the first error kept, not the later one for %q", err, "escalator")
	}
}

func TestEdgeBuilder_NilRoutingTargetSetsError(t *testing.T) {
	a := newDummyNode("A")
	b := newDummyNode("B")

	t.Run("nil target among others", func(t *testing.T) {
		eb := NewEdgeBuilder().AddRoutes(a, map[string]Node{"x": nil, "y": b})
		err := eb.Err()
		if !errors.Is(err, ErrNilRoutingTarget) {
			t.Fatalf("Err() = %v, want an error matching ErrNilRoutingTarget", err)
		}
		if msg := err.Error(); !strings.Contains(msg, `"x"`) || !strings.Contains(msg, `"A"`) {
			t.Errorf("Err() = %q, want it to name route %q and source node %q", msg, "x", "A")
		}
		assertEdges(t, eb.Build(), []Edge{
			{Route: refusedEdge{err: fmt.Errorf("%w: route %q from node %q", ErrNilRoutingTarget, "x", "A")}},
			{From: a, To: b, Route: StringRoute("y")},
		})
	})

	t.Run("only entry has nil target", func(t *testing.T) {
		eb := NewEdgeBuilder().AddRoutes(a, map[string]Node{"x": nil})
		err := eb.Err()
		if !errors.Is(err, ErrNilRoutingTarget) {
			t.Fatalf("Err() = %v, want an error matching ErrNilRoutingTarget", err)
		}
		if errors.Is(err, ErrEmptyRoutingMap) {
			t.Errorf("Err() = %v, want no match with ErrEmptyRoutingMap for a map with an entry", err)
		}
		assertEdges(t, eb.Build(), []Edge{
			{Route: refusedEdge{err: fmt.Errorf("%w: route %q from node %q", ErrNilRoutingTarget, "x", "A")}},
		})
	})
}

// TestNew_RefusesAddRoutesErrors checks that New refuses the edges of a
// builder whose AddRoutes recorded an error, so a caller that never calls
// Err still gets an error instead of a graph without the described routes.
func TestNew_RefusesAddRoutesErrors(t *testing.T) {
	a := newDummyNode("a")
	b := newDummyNode("b")

	for _, tc := range []struct {
		name     string
		routes   map[string]Node
		sentinel error
		wantErr  string
	}{
		{
			name:     "empty map",
			routes:   map[string]Node{},
			sentinel: ErrEmptyRoutingMap,
			wantErr:  `graph validation failed: edges[1]: routing map must not be empty: provide at least one route-to-node mapping: from node "a"`,
		},
		{
			name:     "nil map",
			routes:   nil,
			sentinel: ErrEmptyRoutingMap,
			wantErr:  `graph validation failed: edges[1]: routing map must not be empty: provide at least one route-to-node mapping: from node "a"`,
		},
		{
			name:     "nil target",
			routes:   map[string]Node{"x": nil, "y": b},
			sentinel: ErrNilRoutingTarget,
			wantErr:  `graph validation failed: edges[1]: routing map target must not be nil: route "x" from node "a"`,
		},
		{
			name:     "nil target after a valid route",
			routes:   map[string]Node{"w": b, "x": nil},
			sentinel: ErrNilRoutingTarget,
			wantErr:  `graph validation failed: edges[2]: routing map target must not be nil: route "x" from node "a"`,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			edges := NewEdgeBuilder().Add(Start, a).AddRoutes(a, tc.routes).Build()

			_, err := New("w", edges)
			if !errors.Is(err, ErrGraphValidation) || !errors.Is(err, tc.sentinel) {
				t.Fatalf("New() error = %v, want an error matching ErrGraphValidation and %v", err, tc.sentinel)
			}
			if err.Error() != tc.wantErr {
				t.Errorf("New() error = %q, want %q", err, tc.wantErr)
			}
		})
	}
}

// TestEdgeBuilder_AddRoutesPlaceholderPerError checks the edges Build
// returns after two AddRoutes calls that each record an error: one
// placeholder per error, at the position of the bad map or route, carrying
// that call's error, with a route that matches no event.
func TestEdgeBuilder_AddRoutesPlaceholderPerError(t *testing.T) {
	a := newDummyNode("A")
	b := newDummyNode("B")
	c := newDummyNode("C")

	edges := NewEdgeBuilder().
		AddRoutes(a, nil).
		AddRoutes(b, map[string]Node{"x": nil, "y": c}).
		Build()

	assertEdges(t, edges, []Edge{
		{Route: refusedEdge{err: fmt.Errorf("%w: from node %q", ErrEmptyRoutingMap, "A")}},
		{Route: refusedEdge{err: fmt.Errorf("%w: route %q from node %q", ErrNilRoutingTarget, "x", "B")}},
		{From: b, To: c, Route: StringRoute("y")},
	})
	if edges[0].Route.Matches(&session.Event{}) {
		t.Errorf("placeholder route Matches(event) = true, want false")
	}
}
