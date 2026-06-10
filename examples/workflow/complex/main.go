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
	"iter"
	"log"
	"os"
	"time"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/workflowagent"
	"google.golang.org/adk/cmd/launcher"
	"google.golang.org/adk/cmd/launcher/full"
	"google.golang.org/adk/session"
	"google.golang.org/adk/session/vertexai"
	"google.golang.org/adk/workflow"
	"google.golang.org/genai"
)

func main() {
	ctx := context.Background()

	poemAgent, err := agent.New(agent.Config{
		Name:        "poem agent",
		Description: "writes poems",
		Run: func(ic agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				ev1 := session.NewEvent(ic.InvocationID())
				ev1.Partial = true
				ev1.Content = &genai.Content{
					Parts: []*genai.Part{{Text: "the first line of the poem"}},
				}
				ic.Session().State().Set("firstLine", "done")
				if !yield(ev1, nil) {
					return
				}

				ev2 := session.NewEvent(ic.InvocationID())
				ev2.Partial = false
				ev2.Content = &genai.Content{
					Parts: []*genai.Part{{Text: "the second line of the poem"}},
				}
				ic.Session().State().Set("secondLine", "done")
				if !yield(ev2, nil) {
					return
				}
			}
		},
	})
	if err != nil {
		log.Fatalf("failed to create poem agent: %v", err)
	}

	nodeConfig := workflow.NodeConfig{
		RetryConfig: workflow.DefaultRetryConfig(),
	}

	initNode := workflow.NewFunctionNode("init", func(ctx agent.InvocationContext, _ string) (string, error) {
		ctx.Session().State().Set("a", "init")
		return "init done", nil
	}, nodeConfig)

	poemNode, err := workflow.NewAgentNode(poemAgent, nodeConfig)
	if err != nil {
		log.Fatalf("failed to create poem node: %v", err)
	}

	workerA := workflow.NewFunctionNode("workerA", func(ctx agent.InvocationContext, _ string) (string, error) {
		i := 10
		for i > 0 {
			v, err := ctx.Session().State().Get("a")
			log.Printf("WorkerA: v, err= %v, %v", v, err)
			time.Sleep(100 * time.Millisecond)
			i--
		}
		return "WorkerA done", nil
	}, nodeConfig)

	workerB := workflow.NewFunctionNode("workerB", func(ctx agent.InvocationContext, _ string) (string, error) {
		i := 10
		for i > 0 {
			if i == 5 {
				ctx.Session().State().Set("a", "workerB")
			}
			v, err := ctx.Session().State().Get("a")
			log.Printf("WorkerB: v, err= %v, %v", v, err)
			time.Sleep(57 * time.Millisecond)
			i--
		}
		return "WorkerB done", nil
	}, nodeConfig)

	finalNode := workflow.NewJoinNode("finalNode")

	// // 1. Define functions for nodes
	// // The first node will receive the user message as input (string).
	// upperFn := func(ctx agent.CallbackContext, input string) (string, error) {
	// 	if input == "" {
	// 		ctx.State().Set("input", "NONE!!")
	// 		return "No input received", nil
	// 	}
	// 	ctx.State().Set("input", input)

	// 	return strings.ToUpper(input), nil
	// }

	// suffixFn := func(ctx agent.CallbackContext, input string) (string, error) {
	// 	v, err := ctx.State().Get("input")
	// 	info := ""
	// 	if err == nil {
	// 		info = fmt.Sprintf("'input' found: %+v", v)
	// 	} else {
	// 		info = "no 'input' found"
	// 	}

	// 	return input + " IS AWESOME! " + info, nil
	// }

	// 2. Create Nodes

	// nodeA := workflow.NewFunctionNode("upper", upperFn, nodeConfig)
	// nodeB := workflow.NewFunctionNode("suffix", suffixFn, nodeConfig)

	// 3. Define flow (Edges)
	eb := workflow.NewEdgeBuilder()
	eb.Add(workflow.Start, poemNode)
	eb.Add(workflow.Start, initNode)
	eb.Add(initNode, workerA)
	eb.Add(initNode, workerB)
	eb.AddFanIn(finalNode, workerA, workerB, poemNode)

	// 4. Create Workflow Agent
	myWorkflow, err := workflowagent.New(workflowagent.Config{
		Name:        "simple_sequence_workflow",
		Description: "Converts string to uppercase and appends a suffix",
		Edges:       eb.Build(),
	})
	if err != nil {
		log.Fatalf("failed to create workflow: %v", err)
	}

	log.Printf("Successfully created root agent: %s", myWorkflow.Name())

	sess, err := vertexai.NewSessionService(ctx, vertexai.VertexAIServiceConfig{
		Location:        "us-central1",
		ProjectID:       "kdroste-adk-2026-05",
		ReasoningEngine: "6407573885411131392",
	})
	if err != nil {
		log.Fatalf("failed to create session service: %v", err)
	}

	config := &launcher.Config{
		AgentLoader:    agent.NewSingleLoader(myWorkflow),
		SessionService: sess,
	}
	l := full.NewLauncher()
	if err = l.Execute(ctx, config, os.Args[1:]); err != nil {
		log.Fatalf("Run failed: %v\n\n%s", err, l.CommandLineSyntax())
	}
}
