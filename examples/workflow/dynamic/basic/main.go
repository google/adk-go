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

// Dynamic workflow example: a parent dynamic node orchestrates a single
// child via workflow.RunNode. Mirrors the "Get started" snippet from
// https://adk.dev/graphs/dynamic/.
package main

import (
	"context"
	"log"
	"os"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/workflowagent"
	"google.golang.org/adk/cmd/launcher"
	"google.golang.org/adk/cmd/launcher/full"
	"google.golang.org/adk/session"
	"google.golang.org/adk/workflow"
)

func main() {
	ctx := context.Background()

	// Child node: returns a greeting. Equivalent to the Python
	// @node(name="hello_node") def my_node(node_input): return "Hello World".
	helloNode := workflow.NewFunctionNode("hello_node",
		func(_ agent.Context, _ string) (string, error) {
			return "Hello World", nil
		},
		workflow.NodeConfig{},
	)

	// Parent dynamic node: orchestrates children via RunNode.
	// RerunOnResume defaults to &true (required for dynamic nodes).
	myWorkflow := workflow.NewDynamicNode[string, string]("my_workflow",
		func(ctx agent.Context, _ string, _ func(*session.Event) error) (string, error) {
			return workflow.RunNode[string](ctx, helloNode, "hello")
		},
		workflow.NodeConfig{},
	)

	wa, err := workflowagent.New(workflowagent.Config{
		Name:        "dynamic_workflow_sample",
		Description: "Minimal dynamic workflow: parent orchestrator calls one child via RunNode.",
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
