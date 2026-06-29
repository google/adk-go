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

// Dynamic workflow + LLM example: a dynamic orchestrator calls a single
// LlmAgent-backed node via workflow.RunNode. Demonstrates the smallest
// useful composition of NewDynamicNode, NewAgentNode, and RunNode.
package main

import (
	"context"
	"log"
	"os"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/agent/workflowagent"
	"google.golang.org/adk/cmd/launcher"
	"google.golang.org/adk/cmd/launcher/full"
	"google.golang.org/adk/model/gemini"
	"google.golang.org/adk/session"
	"google.golang.org/adk/workflow"
)

func main() {
	ctx := context.Background()

	model, err := gemini.NewModel(ctx, "gemini-flash-latest", &genai.ClientConfig{})
	if err != nil {
		log.Fatalf("gemini.NewModel: %v", err)
	}

	// Greeter LLM agent: takes whatever the user typed and replies with a
	// one-sentence greeting. The exact prompt is intentionally tiny so the
	// example stays focused on the dynamic/AgentNode composition.
	greeterAgent, err := llmagent.New(llmagent.Config{
		Name:        "greeter",
		Model:       model,
		Description: "Greets the user in one short sentence.",
		Instruction: "You are a friendly assistant. Greet the user in exactly one short sentence.",
	})
	if err != nil {
		log.Fatalf("llmagent.New: %v", err)
	}

	// Wrap the agent as a workflow.Node so it can be invoked from inside a
	// dynamic orchestrator body via workflow.RunNode.
	greeterNode, err := workflow.NewAgentNode(greeterAgent, workflow.NodeConfig{})
	if err != nil {
		log.Fatalf("workflow.NewAgentNode: %v", err)
	}

	// Dynamic orchestrator: expresses execution order as Go code. Here it
	// just calls the greeter once and returns its output. The same shape
	// scales to multi-step pipelines, branching, loops, etc.
	myWorkflow := workflow.NewDynamicNode[string, string]("greeter_workflow",
		func(nc workflow.NodeContext, in string, _ func(*session.Event) error) (string, error) {
			return workflow.RunNode[string](nc, greeterNode, in)
		},
		workflow.NodeConfig{},
	)

	wa, err := workflowagent.New(workflowagent.Config{
		Name:        "dynamic_llm_sample",
		Description: "Minimal dynamic workflow that invokes one LlmAgent via RunNode.",
		Edges:       workflow.Chain(workflow.Start, myWorkflow),
	})
	if err != nil {
		log.Fatalf("workflowagent.New: %v", err)
	}

	l := full.NewLauncher()
	if err := l.Execute(ctx, &launcher.Config{
		AgentLoader: agent.NewSingleLoader(wa),
	}, os.Args[1:]); err != nil {
		log.Fatalf("Run failed: %v\n\n%s", err, l.CommandLineSyntax())
	}
}
