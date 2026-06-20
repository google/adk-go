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

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/agent/workflowagent"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/session"
	"google.golang.org/adk/workflow"
)

// TestRunner_MessageAsOutput_ClearsOutput is the regression guard for
// the duplicate-output fix. An AgentNode wrapping an LlmAgent emits a
// final model message that synthesizeAgentOutput stamps with
// NodeInfo.MessageAsOutput and populates both Content (the model text)
// and Output (the same text). Driven through the runner, the yielded
// event must have Output cleared so downstream renderers don't surface
// the same text twice, while Content survives. Mirrors adk-python
// runners.py, which copies the event and sets output=None when
// node_info.message_as_output.
func TestRunner_MessageAsOutput_ClearsOutput(t *testing.T) {
	ctx := t.Context()
	svc := session.InMemoryService()
	newNodeTestSession(t, ctx, svc)

	m := &scriptedModel{responses: []*genai.Content{
		genai.NewContentFromText("the only answer", "model"),
	}}
	inner, err := llmagent.New(llmagent.Config{Name: "greeter", Model: m})
	if err != nil {
		t.Fatalf("llmagent.New() error = %v", err)
	}
	node, err := workflow.NewAgentNode(inner, workflow.NodeConfig{})
	if err != nil {
		t.Fatalf("workflow.NewAgentNode() error = %v", err)
	}
	wfAgent, err := workflowagent.New(workflowagent.Config{
		Name:  "wf",
		Edges: workflow.Chain(workflow.Start, node),
	})
	if err != nil {
		t.Fatalf("workflowagent.New() error = %v", err)
	}

	r, err := runner.New(runner.Config{
		AppName:        nodeTestApp,
		Agent:          wfAgent,
		SessionService: svc,
	})
	if err != nil {
		t.Fatalf("runner.New() error = %v", err)
	}

	var sawMessageAsOutput bool
	for ev, err := range r.Run(ctx, nodeTestUser, nodeTestSession, userText("hi"), agent.RunConfig{}) {
		if err != nil {
			t.Fatalf("Run() error = %v", err)
		}
		if ev == nil || ev.LLMResponse.Partial {
			continue
		}
		if ev.NodeInfo == nil || !ev.NodeInfo.MessageAsOutput || ev.LLMResponse.Content == nil {
			continue
		}
		sawMessageAsOutput = true

		if ev.Output != nil {
			t.Errorf("MessageAsOutput event Output = %v, want nil; must be cleared to avoid double-rendering the model text", ev.Output)
		}
		if len(ev.LLMResponse.Content.Parts) == 0 {
			t.Error("MessageAsOutput event lost its Content after Output was cleared")
		}
	}
	if !sawMessageAsOutput {
		t.Fatal("expected a non-partial event stamped with NodeInfo.MessageAsOutput")
	}
}
