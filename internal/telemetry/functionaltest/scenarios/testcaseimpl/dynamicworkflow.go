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

package testcaseimpl

import (
	"errors"
	"strings"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/agent/workflowagent"
	"google.golang.org/adk/v2/internal/telemetry/functionaltest/scenarios/testcasedata/dynamicworkflow"
	"google.golang.org/adk/v2/internal/testutil"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/workflow"
)

func newDynamicWorkflow(t *testing.T, s dynamicworkflow.Scenario) agent.Agent {
	t.Helper()

	// No RetryConfig: a failing node fails on its first attempt, so a failure
	// emits one span per node rather than one per retry.
	nodeCfg := workflow.NodeConfig{}

	staticNode := workflow.NewFunctionNode("static_node",
		func(_ agent.Context, in string) (string, error) {
			if s.Failure == dynamicworkflow.StaticNodeError {
				return "", errors.New("boom: static node failed")
			}
			return strings.ToUpper(in), nil
		}, nodeCfg)

	taskAgent := newCollaborativeAgent(t, "task_agent", llmagent.ModeTask, s.Failure != dynamicworkflow.FirstAgentError, "task complete")
	taskNode, err := workflow.NewAgentNode(taskAgent, nodeCfg)
	if err != nil {
		t.Fatalf("NewAgentNode(task_agent): %v", err)
	}

	singleTurnAgent := newCollaborativeAgent(t, "single_turn_agent", llmagent.ModeSingleTurn, s.Failure != dynamicworkflow.SecondAgentError, "single turn complete")
	singleTurnNode, err := workflow.NewAgentNode(singleTurnAgent, nodeCfg)
	if err != nil {
		t.Fatalf("NewAgentNode(single_turn_agent): %v", err)
	}

	// A plain function node, unlike the agent nodes, gets an invoke_node span
	// of its own, a cache hit included.
	echoNode := workflow.NewFunctionNode("echo_node",
		func(_ agent.Context, in string) (string, error) { return in, nil }, nodeCfg)

	routerNode := workflow.NewDynamicNode("router_node",
		func(ctx agent.Context, in string, _ func(*session.Event) error) (string, error) {
			if s.Failure == dynamicworkflow.DynamicNodeError {
				return "", errors.New("boom: dynamic node failed")
			}
			if _, err := workflow.RunNode[any](ctx, taskNode, in); err != nil {
				return "", err
			}
			if _, err := workflow.RunNode[any](ctx, singleTurnNode, in); err != nil {
				return "", err
			}
			// The same run id twice: the second call is served from the
			// RunNode cache.
			if _, err := workflow.RunNode[any](ctx, echoNode, in, workflow.WithRunID("echo")); err != nil {
				return "", err
			}
			if _, err := workflow.RunNode[any](ctx, echoNode, in, workflow.WithRunID("echo")); err != nil {
				return "", err
			}
			return "done", nil
		}, nodeCfg)

	a, err := workflowagent.New(workflowagent.Config{
		Name:        "my_workflow",
		Description: "static node then a dynamic node delegating to two collaborative agents and a cached function node",
		Edges:       workflow.Chain(workflow.Start, staticNode, routerNode),
	})
	if err != nil {
		t.Fatalf("workflowagent.New: %v", err)
	}
	return a
}

// newCollaborativeAgent builds an llmagent in the given collaborative mode.
// When ok is false its model has no response to give, so the agent fails.
func newCollaborativeAgent(t *testing.T, name string, mode llmagent.Mode, ok bool, reply string) agent.Agent {
	t.Helper()
	var responses []*genai.Content
	if ok {
		responses = []*genai.Content{{Role: genai.RoleModel, Parts: []*genai.Part{{Text: reply}}}}
	}
	a, err := llmagent.New(llmagent.Config{
		Name:        name,
		Description: "collaborative " + name,
		Model:       &testutil.MockModel{Responses: responses},
		Instruction: "you are helpful",
		Mode:        mode,
	})
	if err != nil {
		t.Fatalf("llmagent.New(%s): %v", name, err)
	}
	return a
}
