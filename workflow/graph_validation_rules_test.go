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
)

// gcvTitledSchema resolves an object schema whose identity is its title.
func gcvTitledSchema(t *testing.T, title string) *jsonschema.Resolved {
	t.Helper()
	s := &jsonschema.Schema{
		Type:       "object",
		Title:      title,
		Properties: map[string]*jsonschema.Schema{"v": {Type: "string"}},
	}
	r, err := s.Resolve(nil)
	if err != nil {
		t.Fatalf("Resolve(%q): %v", title, err)
	}
	return r
}

// gcvSchemaNode returns a node with the given input and output schemas;
// a nil schema means none on that side.
func gcvSchemaNode(name string, in, out *jsonschema.Resolved) *dummyNode {
	return &dummyNode{BaseNode: NewBaseNodeWithSchemas(name, "", NodeConfig{}, in, out)}
}

// An edge from Start takes no route of any kind, Default included.
func TestNew_RoutedStartEdgeRefused(t *testing.T) {
	tests := []struct {
		name  string
		route Route
	}{
		{name: "StringRoute", route: StringRoute("r")},
		{name: "Default", route: Default},
		{name: "IntRoute", route: IntRoute(1)},
		{name: "MultiRoute", route: MultiRoute[string]{"x"}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			a := newDummyNode("A")

			_, err := New("w", []Edge{{From: Start, To: a, Route: tc.route}})

			if !errors.Is(err, ErrGraphValidation) {
				t.Fatalf("New() error = %v, want an error matching ErrGraphValidation", err)
			}
			const wantMsg = `edge from "START" to "A" has a route; edges from START take none`
			if !strings.Contains(err.Error(), wantMsg) {
				t.Errorf("New() error = %q, want it to contain %q", err, wantMsg)
			}
		})
	}
}

// Start counts as present when an edge names it at either end; a graph
// where Start is only a target then fails on reachability.
func TestNew_StartPresence(t *testing.T) {
	t.Run("Start only as a target", func(t *testing.T) {
		a := newDummyNode("A")

		_, err := New("w", []Edge{{From: a, To: Start}})

		if !errors.Is(err, ErrNodesNotReachable) {
			t.Fatalf("New() error = %v, want an error matching ErrNodesNotReachable", err)
		}
		if errors.Is(err, ErrNoStartNode) {
			t.Errorf("New() error = %v, want no match with ErrNoStartNode", err)
		}
		if !strings.Contains(err.Error(), `"A"`) {
			t.Errorf("New() error = %q, want it to name \"A\"", err)
		}
	})

	t.Run("no edge names Start", func(t *testing.T) {
		a, b := newDummyNode("A"), newDummyNode("B")

		_, err := New("w", Chain(a, b))

		if !errors.Is(err, ErrNoStartNode) {
			t.Errorf("New() error = %v, want an error matching ErrNoStartNode", err)
		}
	})
}

// A node that several edges mention is listed once among the unreachable
// nodes.
func TestNew_UnreachableNodesListedOnce(t *testing.T) {
	a, b, c, d := newDummyNode("A"), newDummyNode("B"), newDummyNode("C"), newDummyNode("D")

	_, err := New("w", []Edge{{From: Start, To: a}, {From: b, To: c}, {From: b, To: d}})

	const wantMsg = `nodes not reachable from start node: "B, C, D"`
	if err == nil || !strings.Contains(err.Error(), wantMsg) {
		t.Errorf("New() error = %v, want it to contain %q", err, wantMsg)
	}
}

func TestNew_DefaultInsideMultiRouteRefused(t *testing.T) {
	tests := []struct {
		name  string
		route Route
	}{
		{name: "MultiRoute[Route] with a value and Default", route: MultiRoute[Route]{StringRoute("x"), Default}},
		{name: "MultiRoute[any] with a value and Default", route: MultiRoute[any]{"x", Default}},
		{name: "MultiRoute[Route] with Default only", route: MultiRoute[Route]{Default}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			a, b := newDummyNode("A"), newDummyNode("B")

			_, err := New("w", []Edge{{From: Start, To: a}, {From: a, To: b, Route: tc.route}})

			if !errors.Is(err, ErrGraphValidation) {
				t.Fatalf("New() error = %v, want an error matching ErrGraphValidation", err)
			}
			const wantMsg = `edge from "A" to "B": a MultiRoute must not contain Default`
			if !strings.Contains(err.Error(), wantMsg) {
				t.Errorf("New() error = %q, want it to contain %q", err, wantMsg)
			}
		})
	}

	t.Run("MultiRoute without Default is accepted", func(t *testing.T) {
		a, b := newDummyNode("A"), newDummyNode("B")

		_, err := New("w", []Edge{{From: Start, To: a}, {From: a, To: b, Route: MultiRoute[string]{"x", "y"}}})
		if err != nil {
			t.Errorf("New() error = %v, want nil", err)
		}
	})
}

// Every duplicated name is listed once, sorted.
func TestNew_DuplicateNamesListed(t *testing.T) {
	t.Run("two names, each on two nodes", func(t *testing.T) {
		b1, b2 := newDummyNode("B"), newDummyNode("B")
		a1, a2 := newDummyNode("A"), newDummyNode("A")

		_, err := New("w", Chain(Start, b1, b2, a1, a2))

		if !errors.Is(err, ErrDuplicateNodeName) {
			t.Fatalf("New() error = %v, want an error matching ErrDuplicateNodeName", err)
		}
		const wantMsg = "duplicate node name: A, B"
		if !strings.Contains(err.Error(), wantMsg) {
			t.Errorf("New() error = %q, want it to contain %q", err, wantMsg)
		}
	})

	t.Run("one name on three nodes", func(t *testing.T) {
		a1, a2, a3 := newDummyNode("A"), newDummyNode("A"), newDummyNode("A")

		_, err := New("w", Chain(Start, a1, a2, a3))

		if !errors.Is(err, ErrDuplicateNodeName) {
			t.Fatalf("New() error = %v, want an error matching ErrDuplicateNodeName", err)
		}
		if !strings.Contains(err.Error(), "duplicate node name: A") {
			t.Errorf("New() error = %q, want it to contain %q", err, "duplicate node name: A")
		}
		if strings.Contains(err.Error(), "A, A") {
			t.Errorf("New() error = %q, want the name listed once", err)
		}
	})
}

// The cycle error names the cycle itself, from the node where it closes,
// and not the path that led to it.
func TestNew_CycleErrorNamesPath(t *testing.T) {
	a, b, c := newDummyNode("A"), newDummyNode("B"), newDummyNode("C")
	tests := []struct {
		name    string
		edges   []Edge
		wantMsg string
	}{
		{
			name:    "self-loop",
			edges:   []Edge{{From: Start, To: a}, {From: a, To: a}},
			wantMsg: `unconditional cycle detected: "A" -> "A"`,
		},
		{
			name:    "two-node cycle",
			edges:   Chain(Start, a, b, a),
			wantMsg: `unconditional cycle detected: "A" -> "B" -> "A"`,
		},
		{
			name:    "cycle after a lead-in",
			edges:   Chain(Start, a, b, c, b),
			wantMsg: `unconditional cycle detected: "B" -> "C" -> "B"`,
		},
		{
			name:    "cycle after a dead-end branch",
			edges:   []Edge{{From: Start, To: a}, {From: a, To: b}, {From: a, To: c}, {From: c, To: a}},
			wantMsg: `unconditional cycle detected: "A" -> "C" -> "A"`,
		},
		{
			name:    "Start edge declared last",
			edges:   []Edge{{From: a, To: b}, {From: b, To: a}, {From: Start, To: a}},
			wantMsg: `unconditional cycle detected: "A" -> "B" -> "A"`,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, err := New("w", tc.edges)

			if !errors.Is(err, ErrUnconditionalCycle) {
				t.Fatalf("New() error = %v, want an error matching ErrUnconditionalCycle", err)
			}
			if !strings.Contains(err.Error(), tc.wantMsg) {
				t.Errorf("New() error = %q, want it to contain %q", err, tc.wantMsg)
			}
		})
	}

	t.Run("lead-in node is not on the path", func(t *testing.T) {
		_, err := New("w", Chain(Start, a, b, c, b))

		if err == nil || strings.Contains(err.Error(), `"A"`) {
			t.Errorf("New() error = %v, want an error that does not name \"A\"", err)
		}
	})
}
