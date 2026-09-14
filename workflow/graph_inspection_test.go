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

// TestWorkflowEdges covers reading a workflow's edges back, and
// TestWorkflowNodeWorkflow the descent into a nested one, so a caller can walk
// a whole graph without running it.
func TestWorkflowEdges(t *testing.T) {
	first, err := workflow.NewAgentNode(newInspectionAgent(t, "first"), workflow.NodeConfig{})
	if err != nil {
		t.Fatalf("NewAgentNode failed: %v", err)
	}
	second, err := workflow.NewAgentNode(newInspectionAgent(t, "second"), workflow.NodeConfig{})
	if err != nil {
		t.Fatalf("NewAgentNode failed: %v", err)
	}
	w, err := workflow.New("chain", []workflow.Edge{
		{From: workflow.Start, To: first},
		{From: first, To: second},
	})
	if err != nil {
		t.Fatalf("workflow.New failed: %v", err)
	}

	var names []string
	for _, e := range w.Edges() {
		names = append(names, e.From.Name()+"->"+e.To.Name())
	}
	slices.Sort(names)
	want := []string{"first->second", workflow.Start.Name() + "->first"}
	slices.Sort(want)
	if diff := cmp.Diff(want, names); diff != "" {
		t.Errorf("Edges() mismatch (-want +got):\n%s", diff)
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
