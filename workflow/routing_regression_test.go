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
	"sync"
	"testing"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/session"
)

// ranRecorder collects the names of nodes that ran. The scheduler runs
// sibling nodes concurrently, so appends are serialized.
type ranRecorder struct {
	mu  sync.Mutex
	ran []string
}

func (r *ranRecorder) record(name string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.ran = append(r.ran, name)
}

func (r *ranRecorder) names() []string {
	r.mu.Lock()
	defer r.mu.Unlock()
	return slices.Clone(r.ran)
}

func TestWorkflowRouting_DefaultRouteWhenNoRouteEmitted(t *testing.T) {
	rec := &ranRecorder{}
	nodeA := NewFunctionNode("A", func(agent.Context, any) (string, error) {
		return "fromA", nil
	}, defaultNodeConfig)
	nodeB := NewFunctionNode("B", func(_ agent.Context, in string) (string, error) {
		rec.record("B")
		return in + ":B", nil
	}, defaultNodeConfig)
	nodeC := NewFunctionNode("C", func(_ agent.Context, in string) (string, error) {
		rec.record("C")
		return in + ":C", nil
	}, defaultNodeConfig)

	w := mustNew(t, NewEdgeBuilder().
		Add(Start, nodeA).
		AddRoute(nodeA, nodeB, StringRoute("r1")).
		AddRoute(nodeA, nodeC, Default).
		Build())

	var outputs []any
	for _, ev := range drain(t, w.Run(newMockCtx(t))) {
		if ev.Output != nil {
			outputs = append(outputs, ev.Output)
		}
	}

	if want := []any{"fromA", "fromA:C"}; !slices.Equal(outputs, want) {
		t.Errorf("outputs = %v, want %v", outputs, want)
	}
	if got, want := rec.names(), []string{"C"}; !slices.Equal(got, want) {
		t.Errorf("nodes run after A = %v, want %v", got, want)
	}
}

// multiRoutingNode yields one event per entry of routes, each carrying that
// entry as its Routes, all in one activation.
type multiRoutingNode struct {
	BaseNode
	routes [][]string
}

func (n *multiRoutingNode) Run(ctx agent.Context, _ any) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		for _, r := range n.routes {
			ev := session.NewEvent(ctx, ctx.InvocationID())
			ev.Routes = r
			if !yield(ev, nil) {
				return
			}
		}
	}
}

func TestScheduler_MultipleRoutingEventsFailNode(t *testing.T) {
	rec := &ranRecorder{}
	multiRouter := &multiRoutingNode{
		BaseNode: NewBaseNode("multiRouter", "", defaultNodeConfig),
		routes:   [][]string{{"r1"}, {"r2"}},
	}
	nodeB := NewFunctionNode("B", func(agent.Context, any) (any, error) {
		rec.record("B")
		return nil, nil
	}, defaultNodeConfig)
	nodeC := NewFunctionNode("C", func(agent.Context, any) (any, error) {
		rec.record("C")
		return nil, nil
	}, defaultNodeConfig)

	w := mustNew(t, NewEdgeBuilder().
		Add(Start, multiRouter).
		AddRoute(multiRouter, nodeB, StringRoute("r1")).
		AddRoute(multiRouter, nodeC, StringRoute("r2")).
		Build())

	err := drainErr(t, w.Run(newMockCtx(t)))

	if !errors.Is(err, ErrMultipleRoutingEvents) {
		t.Errorf("Run error = %v, want an error matching ErrMultipleRoutingEvents", err)
	}
	if got := rec.names(); len(got) != 0 {
		t.Errorf("successors run = %v, want none", got)
	}
}
