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
	"iter"
	"slices"
	"strings"
	"testing"

	"github.com/google/jsonschema-go/jsonschema"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/session"
)

// A nil node in an edge is a graph validation error naming the edge and the
// end that is nil, not a panic.
func TestNew_NilNodeIsError(t *testing.T) {
	tests := []struct {
		name    string
		edges   []Edge
		wantMsg string
	}{
		{
			name:    "nil To",
			edges:   []Edge{{From: Start, To: nil}},
			wantMsg: "edges[0] has a nil To node",
		},
		{
			name:    "nil From",
			edges:   []Edge{{From: nil, To: Start}},
			wantMsg: "edges[0] has a nil From node",
		},
		{
			name:    "Chain with a nil node",
			edges:   Chain(Start, nil),
			wantMsg: "edges[0] has a nil To node",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, err := New("w", tc.edges)

			if !errors.Is(err, ErrGraphValidation) {
				t.Fatalf("New() error = %v, want an error matching ErrGraphValidation", err)
			}
			if !strings.Contains(err.Error(), tc.wantMsg) {
				t.Errorf("New() error = %q, want it to contain %q", err, tc.wantMsg)
			}
		})
	}
}

// gcvFuncNode is a Node whose underlying kind is func, with value receivers.
type gcvFuncNode func()

func (gcvFuncNode) Name() string                        { return "fn" }
func (gcvFuncNode) Description() string                 { return "" }
func (gcvFuncNode) Config() NodeConfig                  { return NodeConfig{} }
func (gcvFuncNode) InputSchema() *jsonschema.Resolved   { return nil }
func (gcvFuncNode) OutputSchema() *jsonschema.Resolved  { return nil }
func (gcvFuncNode) ValidateInput(in any) (any, error)   { return in, nil }
func (gcvFuncNode) ValidateOutput(out any) (any, error) { return out, nil }
func (gcvFuncNode) Run(agent.Context, any) iter.Seq2[*session.Event, error] {
	return func(func(*session.Event, error) bool) {}
}

// gcvValueNode is a Node held as a struct value, with value receivers.
type gcvValueNode struct{ name string }

func (n gcvValueNode) Name() string                      { return n.name }
func (gcvValueNode) Description() string                 { return "" }
func (gcvValueNode) Config() NodeConfig                  { return NodeConfig{} }
func (gcvValueNode) InputSchema() *jsonschema.Resolved   { return nil }
func (gcvValueNode) OutputSchema() *jsonschema.Resolved  { return nil }
func (gcvValueNode) ValidateInput(in any) (any, error)   { return in, nil }
func (gcvValueNode) ValidateOutput(out any) (any, error) { return out, nil }
func (gcvValueNode) Run(agent.Context, any) iter.Seq2[*session.Event, error] {
	return func(func(*session.Event, error) bool) {}
}

// A nil pointer or nil func held in a non-nil Node is reported like an
// untyped nil.
func TestNew_TypedNilNodeIsError(t *testing.T) {
	var typedNil *FunctionNode
	tests := []struct {
		name    string
		edges   []Edge
		wantMsg string
	}{
		{
			name:    "typed nil To",
			edges:   []Edge{{From: Start, To: typedNil}},
			wantMsg: "edges[0] has a nil To node",
		},
		{
			name:    "typed nil From",
			edges:   []Edge{{From: typedNil, To: Start}},
			wantMsg: "edges[0] has a nil From node",
		},
		{
			name:    "nil func To",
			edges:   []Edge{{From: Start, To: gcvFuncNode(nil)}},
			wantMsg: "edges[0] has a nil To node",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, err := New("w", tc.edges)

			if !errors.Is(err, ErrGraphValidation) {
				t.Fatalf("New() error = %v, want an error matching ErrGraphValidation", err)
			}
			if !strings.Contains(err.Error(), tc.wantMsg) {
				t.Errorf("New() error = %q, want it to contain %q", err, tc.wantMsg)
			}
		})
	}
}

// A node held as a struct value is not a nil node.
func TestNew_ValueNodeAccepted(t *testing.T) {
	if _, err := New("w", Chain(Start, gcvValueNode{name: "v"})); err != nil {
		t.Errorf("New() error = %v, want nil", err)
	}
}

// Concat keeps an item that is neither an Edge nor a []Edge at its position,
// and New reports it by position and Go type.
func TestConcat_ForeignItemFailsNew(t *testing.T) {
	a, b, c := newDummyNode("a"), newDummyNode("b"), newDummyNode("c")

	edges := Concat(Chain(Start, a, b), 42, []Edge{{From: b, To: c}})
	_, err := New("w", edges)

	if len(edges) != 4 {
		t.Errorf("len(Concat(2 edges, 42, 1 edge)) = %d, want 4", len(edges))
	}
	if !errors.Is(err, ErrGraphValidation) {
		t.Fatalf("New() error = %v, want an error matching ErrGraphValidation", err)
	}
	const wantMsg = "edges[2]: Concat was given a value of type int, which is neither an Edge nor a []Edge"
	if !strings.Contains(err.Error(), wantMsg) {
		t.Errorf("New() error = %q, want it to contain %q", err, wantMsg)
	}
}

func TestConcat_KeepsItemOrder(t *testing.T) {
	a, b, c, d := newDummyNode("a"), newDummyNode("b"), newDummyNode("c"), newDummyNode("d")
	e1 := Edge{From: Start, To: a}
	e2 := Edge{From: a, To: b}
	e3 := Edge{From: b, To: c, Route: StringRoute("r")}
	e4 := Edge{From: c, To: d}

	got := Concat(e1, []Edge{e2, e3}, e4)

	want := []Edge{e1, e2, e3, e4}
	if !slices.Equal(got, want) {
		t.Errorf("Concat(e1, []Edge{e2, e3}, e4) = %v, want %v", got, want)
	}
}
