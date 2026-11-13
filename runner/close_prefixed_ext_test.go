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

package runner_test

import (
	"testing"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/agent/workflowagent"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/agenttool"
	"google.golang.org/adk/v2/workflow"
)

// oneToolToolset lists t on every call.
type oneToolToolset struct{ t tool.Tool }

func (o *oneToolToolset) Name() string { return "agents" }
func (o *oneToolToolset) Tools(agent.ReadonlyContext) ([]tool.Tool, error) {
	return []tool.Tool{o.t}, nil
}

// TestRunner_Close_ReachesAgentToolChildThroughPrefixedStandIn checks that the
// prefixed stand-in of an agent tool, run by a workflow ToolNode, still leads
// Runner.Close to the toolsets of the agent it wraps.
func TestRunner_Close_ReachesAgentToolChildThroughPrefixedStandIn(t *testing.T) {
	ts := &closeCountingToolset{}
	child, err := llmagent.New(llmagent.Config{Name: "child", Toolsets: []tool.Toolset{ts}})
	if err != nil {
		t.Fatalf("llmagent.New(child) error = %v", err)
	}
	listed, err := tool.WithToolNamePrefix(&oneToolToolset{t: agenttool.New(child, nil)}, "p").Tools(nil)
	if err != nil {
		t.Fatalf("Tools() error = %v", err)
	}
	if got := listed[0].Name(); got != "p_child" {
		t.Fatalf("listed[0].Name() = %q, want %q", got, "p_child")
	}
	tn, err := workflow.NewToolNode(listed[0], workflow.NodeConfig{})
	if err != nil {
		t.Fatalf("workflow.NewToolNode() error = %v", err)
	}
	wf, err := workflowagent.New(workflowagent.Config{Name: "wf", Edges: []workflow.Edge{{From: workflow.Start, To: tn}}})
	if err != nil {
		t.Fatalf("workflowagent.New() error = %v", err)
	}
	r, err := runner.New(runner.Config{AppName: "close_test", Agent: wf, SessionService: session.InMemoryService()})
	if err != nil {
		t.Fatalf("runner.New() error = %v", err)
	}
	if got := ts.closes.Load(); got != 0 {
		t.Fatalf("closes of the child's toolset before Close = %d, want 0", got)
	}

	if err := r.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}

	if got := ts.closes.Load(); got != 1 {
		t.Errorf("closes of the child's toolset = %d, want 1", got)
	}
}
