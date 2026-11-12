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

package llmagent_test

import (
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/agent/workflowagent"
	"google.golang.org/adk/v2/internal/workflowinternal"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/workflow"
)

// TestTaskDelegation_NestedTaskAgentDoesNotResolveIt pins the consumer side of
// #2059. A coordinator delegates to the task agent "outer", which transfers to
// a workflow whose dynamic node dispatches another task agent, "inner", with
// workflow.RunNode. inner's own runTask stamps Output on its finish_task
// response before it is forwarded through outer, so the guard in outer's
// runTask is not enough on its own: the scheduler running outer and the
// runner's node body must not take that Output as outer's. outer never calls
// finish_task, so its delegation must stay open.
func TestTaskDelegation_NestedTaskAgentDoesNotResolveIt(t *testing.T) {
	t.Parallel()
	for _, tc := range []struct {
		name string
		// routerOutput is what the dynamic node returns after inner finishes.
		routerOutput any
	}{
		{name: "router returns no output"},
		{name: "router returns its own output", routerOutput: "routed"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			innerLLM := &scriptedLLM{turns: []*model.LLMResponse{
				{Content: fcContent("ft-inner", workflowinternal.FinishTaskToolName, map[string]any{"result": "from inner"})},
			}}
			inner := makeLLMAgent(t, "inner", withMode(llmagent.ModeTask), func(c *llmagent.Config) { c.Model = innerLLM })
			innerNode, err := workflow.NewAgentNode(inner, workflow.NodeConfig{})
			if err != nil {
				t.Fatal(err)
			}
			router := workflow.NewDynamicNode("router",
				func(ctx agent.Context, in any, _ func(*session.Event) error) (any, error) {
					if _, err := workflow.RunNode[any](ctx, innerNode, in); err != nil {
						return nil, err
					}
					return tc.routerOutput, nil
				}, workflow.NodeConfig{})
			wf, err := workflowagent.New(workflowagent.Config{
				Name:        "wf",
				Description: "dispatches inner",
				Edges:       workflow.Chain(workflow.Start, router),
			})
			if err != nil {
				t.Fatal(err)
			}
			outerLLM := &scriptedLLM{turns: []*model.LLMResponse{
				{Content: fcContent("transfer-1", "transfer_to_agent", map[string]any{"agent_name": "wf"})},
			}}
			outer := makeLLMAgent(t, "outer", withMode(llmagent.ModeTask), func(c *llmagent.Config) {
				c.Model = outerLLM
				c.SubAgents = []agent.Agent{wf}
			})
			coordLLM := &scriptedLLM{turns: []*model.LLMResponse{
				{Content: fcContent("fc-task", "outer", map[string]any{"request": "do it"})},
			}}
			coord, err := llmagent.New(llmagent.Config{
				Name:      "coord",
				Model:     coordLLM,
				Mode:      llmagent.ModeChat,
				SubAgents: []agent.Agent{outer},
			})
			if err != nil {
				t.Fatal(err)
			}
			svc := session.InMemoryService()
			if _, err := svc.Create(t.Context(), &session.CreateRequest{AppName: "app", UserID: "u", SessionID: "s"}); err != nil {
				t.Fatal(err)
			}
			r, err := runner.New(runner.Config{Agent: coord, SessionService: svc, AppName: "app"})
			if err != nil {
				t.Fatal(err)
			}

			var sawInnerOutput bool
			for ev, err := range r.Run(t.Context(), "u", "s", genai.NewContentFromText("go", "user"), agent.RunConfig{}) {
				if err != nil {
					t.Fatalf("runner.Run: %v", err)
				}
				if ev.Author == "inner" && ev.Output != nil {
					sawInnerOutput = true
				}
				if ev.Content == nil {
					continue
				}
				for _, p := range ev.Content.Parts {
					if p != nil && p.FunctionResponse != nil && p.FunctionResponse.ID == "fc-task" {
						t.Errorf("coordinator's delegation to outer resolved with %v (event authored %q), although outer never called finish_task",
							p.FunctionResponse.Response, ev.Author)
					}
				}
			}
			if !sawInnerOutput {
				t.Fatal("inner never completed its own task, so the case under test did not run")
			}
			if outerLLM.callIdx != 1 {
				t.Errorf("outer model called %d times, want 1", outerLLM.callIdx)
			}
		})
	}
}
