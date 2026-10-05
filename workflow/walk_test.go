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
	"slices"
	"testing"

	"github.com/google/go-cmp/cmp"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/workflow"
)

func newWalkAgent(t *testing.T, name string) agent.Agent {
	t.Helper()
	a, err := agent.New(agent.Config{Name: name})
	if err != nil {
		t.Fatalf("agent.New(%q) failed: %v", name, err)
	}
	return a
}

func newWalkAgentNode(t *testing.T, a agent.Agent) *workflow.AgentNode {
	t.Helper()
	n, err := workflow.NewAgentNode(a, workflow.NodeConfig{})
	if err != nil {
		t.Fatalf("NewAgentNode(%q) failed: %v", a.Name(), err)
	}
	return n
}

func newWalkWorkflowNode(t *testing.T, name string, edges []workflow.Edge) *workflow.WorkflowNode {
	t.Helper()
	n, err := workflow.NewWorkflowNode(name, edges)
	if err != nil {
		t.Fatalf("NewWorkflowNode(%q) failed: %v", name, err)
	}
	return n
}

// walkedNames returns the names of the agents WalkAgents returns for edges,
// sorted, since the walk promises no order.
func walkedNames(edges []workflow.Edge) []string {
	var names []string
	for _, a := range workflow.WalkAgents(edges) {
		names = append(names, a.Name())
	}
	slices.Sort(names)
	return names
}

func TestWalkAgents(t *testing.T) {
	for _, tc := range []struct {
		name  string
		edges func(t *testing.T) []workflow.Edge
		want  []string
	}{
		{
			name: "agent nodes on the edges",
			edges: func(t *testing.T) []workflow.Edge {
				return workflow.Chain(workflow.Start,
					newWalkAgentNode(t, newWalkAgent(t, "first")),
					newWalkAgentNode(t, newWalkAgent(t, "second")))
			},
			want: []string{"first", "second"},
		},
		{
			name: "a node on two edges is visited once",
			edges: func(t *testing.T) []workflow.Edge {
				middle := newWalkAgentNode(t, newWalkAgent(t, "middle"))
				last := newWalkAgentNode(t, newWalkAgent(t, "last"))
				return []workflow.Edge{
					{From: workflow.Start, To: middle},
					{From: middle, To: last},
				}
			},
			want: []string{"last", "middle"},
		},
		{
			// Every node on an edge counts, including one no edge leads to.
			name: "a node that only starts an edge",
			edges: func(t *testing.T) []workflow.Edge {
				first := newWalkAgentNode(t, newWalkAgent(t, "first"))
				second := newWalkAgentNode(t, newWalkAgent(t, "second"))
				return []workflow.Edge{{From: first, To: second}}
			},
			want: []string{"first", "second"},
		},
		{
			name: "agents in a nested sub-workflow",
			edges: func(t *testing.T) []workflow.Edge {
				inner := newWalkWorkflowNode(t, "inner",
					workflow.Chain(workflow.Start, newWalkAgentNode(t, newWalkAgent(t, "buried"))))
				return workflow.Chain(workflow.Start, newWalkAgentNode(t, newWalkAgent(t, "outer")), inner)
			},
			want: []string{"buried", "outer"},
		},
		{
			// Each node is visited once, not each agent: the same agent behind
			// two nodes comes back twice, and deduplicating is the caller's job.
			name: "an agent two nodes run is returned twice",
			edges: func(t *testing.T) []workflow.Edge {
				shared := newWalkAgent(t, "shared")
				inner := newWalkWorkflowNode(t, "inner",
					workflow.Chain(workflow.Start, newWalkAgentNode(t, shared)))
				return workflow.Chain(workflow.Start, newWalkAgentNode(t, shared), inner)
			},
			want: []string{"shared", "shared"},
		},
		{
			name: "nodes that run no agent",
			edges: func(t *testing.T) []workflow.Edge {
				fn := workflow.NewFunctionNode("echo",
					func(_ agent.Context, in any) (any, error) { return in, nil }, workflow.NodeConfig{})
				return workflow.Chain(workflow.Start, fn)
			},
			want: nil,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if diff := cmp.Diff(tc.want, walkedNames(tc.edges(t))); diff != "" {
				t.Errorf("agents mismatch (-want +got):\n%s", diff)
			}
		})
	}
}
