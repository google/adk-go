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

// hitl_simple is the minimal end-to-end HITL workflow for
// verifying the console launcher's pause/resume support.
// No LLM, no API key, no streaming — just two workflow nodes:
//
//	Start → ask_name → greet
//
// The ask_name node yields a RequestInput that pauses the
// workflow. The console launcher renders the prompt; the user's
// reply is delivered to greet as its input.
//
//	go run ./examples/workflow/hitl_simple/ console
//
//	User -> hello
//	Agent -> What's your name?
//	User -> Alice
//	Agent -> Hello, Alice!
//	User ->
package main

import (
	"context"
	"fmt"
	"iter"
	"log"
	"os"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/workflowagent"
	"google.golang.org/adk/cmd/launcher"
	"google.golang.org/adk/cmd/launcher/full"
	"google.golang.org/adk/session"
	"google.golang.org/adk/workflow"
)

// inlineNode wraps a closure as a workflow.Node so we can yield
// RequestInput (FunctionNode's "input → output" shape does not
// cover that).
type inlineNode struct {
	workflow.BaseNode
	run func(agent.Context, any) iter.Seq2[*session.Event, error]
}

func (n *inlineNode) Run(ctx agent.Context, input any) iter.Seq2[*session.Event, error] {
	return n.run(ctx, input)
}

func mkNode(name, desc string, run func(agent.Context, any) iter.Seq2[*session.Event, error]) *inlineNode {
	return &inlineNode{BaseNode: workflow.NewBaseNode(name, desc, workflow.NodeConfig{}), run: run}
}

// askName pauses the workflow with a RequestInput asking for the
// user's name. The handoff resume delivers the reply as the next
// node's input.
func askName(ctx agent.Context, _ any) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		yield(workflow.NewRequestInputEvent(ctx.InvocationContext(), session.RequestInput{
			InterruptID: "ask_name",
			Message:     "What's your name?",
		}), nil)
	}
}

// greet receives the user's reply (as plain text) and yields one
// event with the greeting as both the workflow output (StateDelta)
// and Content (so the console launcher prints it).
func greet(ctx agent.Context, input any) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		name, _ := input.(string)
		if name == "" {
			name = "stranger"
		}
		msg := fmt.Sprintf("Hello, %s!", name)
		ev := session.NewEvent(ctx.InvocationID())
		ev.Output = msg
		ev.Content = &genai.Content{
			Parts: []*genai.Part{{Text: msg}},
		}
		yield(ev, nil)
	}
}

func main() {
	ctx := context.Background()

	ask := mkNode("ask_name", "asks the user for their name", askName)
	hello := mkNode("greet", "greets the user by name", greet)

	edges := workflow.Chain(workflow.Start, ask, hello)

	rootAgent, err := workflowagent.New(workflowagent.Config{
		Name:        "hitl_simple",
		Description: "minimal HITL workflow for console launcher verification",
		Edges:       edges,
	})
	if err != nil {
		log.Fatalf("failed to create workflow agent: %v", err)
	}

	log.Printf("hitl_simple sample ready — type anything to start, then answer the prompt")

	launcherCfg := &launcher.Config{
		AgentLoader: agent.NewSingleLoader(rootAgent),
	}
	l := full.NewLauncher()
	if err := l.Execute(ctx, launcherCfg, os.Args[1:]); err != nil {
		log.Fatalf("Run failed: %v\n\n%s", err, l.CommandLineSyntax())
	}
}
