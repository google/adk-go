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
	"iter"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/adk/agent"
	"google.golang.org/adk/session"
)

// edgeIDs renders edges as "From→To" strings using node names.
func edgeIDs(es []Edge) []string {
	if len(es) == 0 {
		return nil
	}
	out := make([]string, len(es))
	for i, e := range es {
		out[i] = e.From.Name() + "→" + e.To.Name()
	}
	return out
}

func TestNewGraph_SuccessorsOf(t *testing.T) {
	a := newTestNode("A")
	b := newTestNode("B")
	c := newTestNode("C")

	tests := []struct {
		name  string
		edges []Edge
		// want maps a node name to the expected outgoing-edge IDs
		// ("From→To" strings). Nodes absent from want are expected
		// to have no successors.
		want map[string][]string
	}{
		{
			name:  "empty graph",
			edges: nil,
			want:  map[string][]string{},
		},
		{
			name:  "single edge",
			edges: []Edge{{From: Start, To: a}},
			want: map[string][]string{
				"START": {"START→A"},
			},
		},
		{
			name: "linear chain",
			edges: []Edge{
				{From: Start, To: a},
				{From: a, To: b},
				{From: b, To: c},
			},
			want: map[string][]string{
				"START": {"START→A"},
				"A":     {"A→B"},
				"B":     {"B→C"},
			},
		},
		{
			name: "fan-out from Start to three nodes",
			edges: []Edge{
				{From: Start, To: a},
				{From: Start, To: b},
				{From: Start, To: c},
			},
			want: map[string][]string{
				"START": {"START→A", "START→B", "START→C"},
			},
		},
		{
			name: "fan-in: two upstreams converge on c",
			edges: []Edge{
				{From: Start, To: a},
				{From: Start, To: b},
				{From: a, To: c},
				{From: b, To: c},
			},
			want: map[string][]string{
				"START": {"START→A", "START→B"},
				"A":     {"A→C"},
				"B":     {"B→C"},
			},
		},
		{
			name: "cycle a → b → a",
			edges: []Edge{
				{From: a, To: b},
				{From: b, To: a},
			},
			want: map[string][]string{
				"A": {"A→B"},
				"B": {"B→A"},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := newGraph(tt.edges)
			for _, n := range []Node{Start, a, b, c} {
				want := tt.want[n.Name()]
				got := edgeIDs(g.successorsOf(n))
				if diff := cmp.Diff(want, got); diff != "" {
					t.Errorf("successorsOf(%q) mismatch (-want +got):\n%s", n.Name(), diff)
				}
			}
		})
	}
}

func TestNewGraph_SuccessorsOf_NodeNotInGraph(t *testing.T) {
	a := newTestNode("A")
	b := newTestNode("B")
	stranger := newTestNode("stranger")

	g := newGraph([]Edge{{From: a, To: b}})

	if got := g.successorsOf(stranger); got != nil {
		t.Errorf("successorsOf(stranger) = %v, want nil", got)
	}
	// Terminal-detection callsite pattern: a node with no outgoing
	// edges (here: b) should also report nil/empty so the engine can
	// stop dispatching successors.
	if got := g.successorsOf(b); len(got) != 0 {
		t.Errorf("successorsOf(b) = %v, want empty (b is terminal)", got)
	}
}

func TestNewGraph_PreservesEdgeOrder(t *testing.T) {
	// Within a single From node, successorsOf returns edges in input
	// order.
	a := newTestNode("A")
	b := newTestNode("B")
	c := newTestNode("C")
	d := newTestNode("D")

	edges := []Edge{
		{From: a, To: c},
		{From: a, To: d},
		{From: a, To: b},
	}
	g := newGraph(edges)

	got := edgeIDs(g.successorsOf(a))
	want := []string{"A→C", "A→D", "A→B"}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("successorsOf(a) mismatch (-want +got):\n%s", diff)
	}
}

// newTestNode returns a minimal Node implementation suitable for
// graph-structure tests. Run is a no-op; only Name matters for
// diagnostic output.
func newTestNode(name string) Node {
	return &testNode{BaseNode: NewBaseNode(name, "", NodeConfig{})}
}

type testNode struct {
	BaseNode
}

func (n *testNode) Run(_ agent.InvocationContext, _ any) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {}
}
