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
	"testing"

	"github.com/google/go-cmp/cmp"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/workflow"
)

// newInspectionAgent builds a minimal agent to hang off a node.
func newInspectionAgent(t *testing.T, name string) agent.Agent {
	t.Helper()
	a, err := agent.New(agent.Config{Name: name, Description: name + " description"})
	if err != nil {
		t.Fatalf("agent.New(%q) failed: %v", name, err)
	}
	return a
}

// TestAgentNodeAgent covers reading back the agent a node runs. A caller
// describing an app cannot otherwise find it: the agents a workflow reaches
// through its edges are not the workflow agent's sub-agents.
func TestAgentNodeAgent(t *testing.T) {
	want := newInspectionAgent(t, "worker")
	node, err := workflow.NewAgentNode(want, workflow.NodeConfig{})
	if err != nil {
		t.Fatalf("NewAgentNode failed: %v", err)
	}
	if got := node.Agent(); got != want {
		t.Errorf("Agent() = %v, want %v", got, want)
	}
}

// TestWorkflowEdges pins that Edges returns every edge in construction order.
// The graph indexes edges in maps, so an order read back from them changes from
// call to call, and a caller that takes the first match of something would get
// a different answer each time. The chain is long enough that map order all but
// never reproduces it. Neither the slice passed to New nor the one Edges
// returns is shared with the workflow.
func TestWorkflowEdges(t *testing.T) {
	var edges []workflow.Edge
	var want []string
	from := workflow.Start
	for _, name := range []string{"a", "b", "c", "d", "e", "f", "g", "h"} {
		node, err := workflow.NewAgentNode(newInspectionAgent(t, name), workflow.NodeConfig{})
		if err != nil {
			t.Fatalf("NewAgentNode(%q) failed: %v", name, err)
		}
		edges = append(edges, workflow.Edge{From: from, To: node})
		want = append(want, from.Name()+"->"+name)
		from = node
	}
	w, err := workflow.New("chain", edges)
	if err != nil {
		t.Fatalf("workflow.New failed: %v", err)
	}

	for range 5 {
		var got []string
		for _, e := range w.Edges() {
			got = append(got, e.From.Name()+"->"+e.To.Name())
		}
		if diff := cmp.Diff(want, got); diff != "" {
			t.Fatalf("Edges() mismatch (-want +got):\n%s", diff)
		}
	}

	w.Edges()[0] = workflow.Edge{}
	if w.Edges()[0].From == nil {
		t.Error("writing to the slice Edges returned changed the workflow's edges")
	}
	edges[1] = workflow.Edge{}
	if w.Edges()[1].From == nil {
		t.Error("writing to the slice passed to New changed the workflow's edges")
	}
}

func TestWorkflowNodeWorkflow(t *testing.T) {
	inner, err := workflow.NewAgentNode(newInspectionAgent(t, "inner_worker"), workflow.NodeConfig{})
	if err != nil {
		t.Fatalf("NewAgentNode failed: %v", err)
	}
	node, err := workflow.NewWorkflowNode("inner", []workflow.Edge{{From: workflow.Start, To: inner}})
	if err != nil {
		t.Fatalf("NewWorkflowNode failed: %v", err)
	}

	sub := node.Workflow()
	if sub == nil {
		t.Fatal("Workflow() = nil, want the nested workflow")
	}
	edges := sub.Edges()
	if len(edges) != 1 {
		t.Fatalf("len(Workflow().Edges()) = %d, want 1", len(edges))
	}
	agentNode, ok := edges[0].To.(*workflow.AgentNode)
	if !ok {
		t.Fatalf("edge target is %T, want *workflow.AgentNode", edges[0].To)
	}
	if got := agentNode.Agent().Name(); got != "inner_worker" {
		t.Errorf("nested agent name = %q, want inner_worker", got)
	}
}

// TestWorkflowEdgesNilReceiver pins that inspecting a zero workflow reports
// nothing rather than panicking.
func TestWorkflowEdgesNilReceiver(t *testing.T) {
	var w *workflow.Workflow
	if got := w.Edges(); got != nil {
		t.Errorf("Edges() on a nil workflow = %v, want nil", got)
	}
}
