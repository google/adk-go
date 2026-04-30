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

package workflow_test

import (
	"strings"
	"testing"

	"google.golang.org/adk/workflow"
)

// stubNode is a minimal Node used to exercise graph construction.
type stubNode struct {
	workflow.Base
}

func (s *stubNode) RunImpl(_ *workflow.NodeContext, _ any, _ workflow.EventEmitter) error {
	return nil
}

func newStub(t *testing.T, name string) *stubNode {
	t.Helper()
	n := &stubNode{}
	if err := n.SetMetadata(name, "", workflow.NodeSpec{}); err != nil {
		t.Fatalf("SetMetadata: %v", err)
	}
	return n
}

func TestNew_RejectsEmptyEdges(t *testing.T) {
	_, err := workflow.New(workflow.Config{Name: "wf", Edges: nil})
	if err == nil {
		t.Fatal("expected error for empty edges")
	}
}

func TestNew_RejectsInvalidName(t *testing.T) {
	a := newStub(t, "a")
	_, err := workflow.New(workflow.Config{
		Name:  "1bad",
		Edges: []workflow.Edge{workflow.Connect(workflow.START, a)},
	})
	if err == nil {
		t.Fatal("expected error for invalid workflow name")
	}
}

func TestNew_RejectsUnreachableNode(t *testing.T) {
	a := newStub(t, "a")
	b := newStub(t, "b")
	c := newStub(t, "c")
	_, err := workflow.New(workflow.Config{
		Name: "wf",
		Edges: []workflow.Edge{
			workflow.Connect(workflow.START, a),
			workflow.Connect(a, b),
			workflow.Connect(c, b), // c has no incoming edge from anything reachable
		},
	})
	if err == nil {
		t.Fatal("expected unreachable error")
	}
	if !strings.Contains(err.Error(), "unreachable") {
		t.Errorf("err = %v, want unreachable", err)
	}
}

func TestNew_RejectsNoStartEdge(t *testing.T) {
	a := newStub(t, "a")
	b := newStub(t, "b")
	_, err := workflow.New(workflow.Config{
		Name:  "wf",
		Edges: []workflow.Edge{workflow.Connect(a, b)},
	})
	if err == nil {
		t.Fatal("expected error for missing START edge")
	}
}

func TestNew_DuplicateNameRejected(t *testing.T) {
	a1 := newStub(t, "a")
	a2 := newStub(t, "a") // distinct instance, same name
	_, err := workflow.New(workflow.Config{
		Name:  "wf",
		Edges: []workflow.Edge{workflow.Connect(workflow.START, a1), workflow.Connect(a1, a2)},
	})
	if err == nil {
		t.Fatal("expected duplicate-name error")
	}
}

func TestNew_BuildsValidGraph(t *testing.T) {
	a := newStub(t, "a")
	b := newStub(t, "b")
	c := newStub(t, "c")
	wf, err := workflow.New(workflow.Config{
		Name: "wf",
		Edges: []workflow.Edge{
			workflow.Connect(workflow.START, a),
			workflow.Connect(a, b),
			workflow.Connect(a, c, workflow.RouteString("alt")),
		},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	g := wf.Graph()
	if !g.HasNode("a") || !g.HasNode("b") || !g.HasNode("c") {
		t.Errorf("nodes missing: %v", g.Nodes())
	}
	succA := g.Successors("a")
	if len(succA) != 2 {
		t.Errorf("a successors = %v, want 2", succA)
	}
	pred := g.Predecessors("b")
	if len(pred) != 1 || pred[0] != "a" {
		t.Errorf("b predecessors = %v, want [a]", pred)
	}
}

func TestRouteMatch(t *testing.T) {
	cases := []struct {
		a, b workflow.Route
		want bool
	}{
		{workflow.RouteString("x"), workflow.RouteString("x"), true},
		{workflow.RouteString("x"), workflow.RouteString("y"), false},
		{workflow.RouteString("x"), workflow.RouteInt(1), false},
		{workflow.RouteInt(1), workflow.RouteInt(1), true},
		{workflow.RouteBool(true), workflow.RouteBool(true), true},
		{workflow.RouteBool(true), workflow.RouteBool(false), false},
		{workflow.DefaultRoute, workflow.DefaultRoute, true},
	}
	for _, tc := range cases {
		if got := tc.a.Match(tc.b); got != tc.want {
			t.Errorf("%v.Match(%v) = %v, want %v", tc.a, tc.b, got, tc.want)
		}
	}
}

func TestRetryConfig_DelayFor_Backoff(t *testing.T) {
	cfg := workflow.RetryConfig{
		MaxAttempts:   5,
		InitialDelay:  100,
		MaxDelay:      10000,
		BackoffFactor: 2.0,
		Jitter:        -1, // disable jitter for determinism
	}
	d1 := cfg.DelayFor(1)
	d2 := cfg.DelayFor(2)
	d3 := cfg.DelayFor(3)
	if d1 != 100 {
		t.Errorf("attempt 1 = %v, want 100", d1)
	}
	if d2 != 200 {
		t.Errorf("attempt 2 = %v, want 200", d2)
	}
	if d3 != 400 {
		t.Errorf("attempt 3 = %v, want 400", d3)
	}
}

func TestRetryConfig_DelayFor_Caps(t *testing.T) {
	cfg := workflow.RetryConfig{
		MaxAttempts:   10,
		InitialDelay:  1000,
		MaxDelay:      2000,
		BackoffFactor: 2.0,
		Jitter:        -1,
	}
	d := cfg.DelayFor(5)
	if d > 2000 {
		t.Errorf("delay = %v, want <= 2000", d)
	}
}

func TestStartSentinel(t *testing.T) {
	if !workflow.IsStart(workflow.START) {
		t.Error("START should be IsStart")
	}
	a := newStub(t, "a")
	if workflow.IsStart(a) {
		t.Error("normal node should not be IsStart")
	}
}

func TestSchema_JSONSchemaForStruct(t *testing.T) {
	type req struct {
		Name string `json:"name"`
		Age  int    `json:"age"`
	}
	s := workflow.JSONSchemaFor[req]()
	got, err := s.Validate(map[string]any{"name": "Alice", "age": 30})
	if err != nil {
		t.Fatalf("Validate: %v", err)
	}
	r, ok := got.(req)
	if !ok {
		t.Fatalf("type = %T, want req", got)
	}
	if r.Name != "Alice" || r.Age != 30 {
		t.Errorf("decoded = %+v", r)
	}
}

func TestSchema_PassthroughExactType(t *testing.T) {
	type req struct {
		N int `json:"n"`
	}
	s := workflow.JSONSchemaFor[req]()
	in := req{N: 7}
	got, err := s.Validate(in)
	if err != nil {
		t.Fatalf("Validate: %v", err)
	}
	if got.(req) != in {
		t.Errorf("got %+v, want %+v", got, in)
	}
}
