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

package main

import (
	"context"
	"log"
	"os"
	"strings"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/workflowagent"
	"google.golang.org/adk/cmd/launcher"
	"google.golang.org/adk/cmd/launcher/full"
	"google.golang.org/adk/workflow"
)

func main() {
	ctx := context.Background()

	// 1. Define functions for nodes
	// The first node will receive the user message as input (string).
	upperFn := func(ctx agent.InvocationContext, input string) (string, error) {
		if input == "" {
			return "No input received", nil
		}
		return strings.ToUpper(input), nil
	}

	suffixFn := func(ctx agent.InvocationContext, input string) (string, error) {
		return input + " IS AWESOME!", nil
	}

	// 2. Create Nodes
	nodeConfig := workflow.NodeConfig{
		RetryConfig: workflow.NewRetryConfig(),
	}
	nodeA := workflow.NewFunctionNode("upper", upperFn, nodeConfig)
	nodeB := workflow.NewFunctionNode("suffix", suffixFn, nodeConfig)

	// 3. Define flow (Edges)
	edges := workflow.Chain(workflow.Start, nodeA, nodeB)

	// 4. Create Workflow Agent
	myWorkflow, err := workflowagent.New(workflowagent.Config{
		Name:        "simple_sequence_workflow",
		Description: "Converts string to uppercase and appends a suffix",
		Edges:       edges,
	})
	if err != nil {
		log.Fatalf("failed to create workflow: %v", err)
	}

	log.Printf("Successfully created root agent: %s", myWorkflow.Name())

	config := &launcher.Config{
		AgentLoader: agent.NewSingleLoader(myWorkflow),
	}
	l := full.NewLauncher()
	if err = l.Execute(ctx, config, os.Args[1:]); err != nil {
		log.Fatalf("Run failed: %v\n\n%s", err, l.CommandLineSyntax())
	}
}
