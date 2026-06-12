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

// Dynamic workflow + HITL example: a dynamic orchestrator pauses for
// human input via workflow.RunNode, then resumes and greets the user.
// Demonstrates the re-entry resume pattern: dynamic nodes default to
// RerunOnResume=&true, so the orchestrator body is re-invoked from the
// top after the human replies, and the reply is delivered via
// NodeContext.ResumedInput.
//
//	go run ./examples/workflow/dynamic/hitl/ console
//
//	User -> start
//	Agent -> [HITL input] What's your name?
//	User -> Alice
//	Agent -> Hello, Alice!
//
// Compare with examples/workflow/hitl_simple/: the static-chain
// variant of the same scenario. Both rely on the console launcher's
// HITL support to render the prompt and forward the reply.
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

// askName is an inline workflow.Node that pauses with a RequestInput
// asking for the user's name. We can't use FunctionNode because its
// "input → output" shape does not cover RequestInput emission.
type askName struct {
	workflow.BaseNode
}

func newAskName() *askName {
	return &askName{
		BaseNode: workflow.NewBaseNode("ask_name", "asks the user for their name", workflow.NodeConfig{}),
	}
}

func (n *askName) Run(ctx agent.Context, _ any) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		yield(workflow.NewRequestInputEvent(ctx, session.RequestInput{
			InterruptID: "ask_name",
			Message:     "What's your name?",
		}), nil)
	}
}

func main() {
	ctx := context.Background()

	ask := newAskName()

	greeter := workflow.NewDynamicNode[string, string]("hitl_demo",
		func(nc workflow.NodeContext, _ string, emit func(*session.Event) error) (string, error) {
			// Resume re-entry: the reply is in ResumedInput.
			if reply, ok := nc.ResumedInput("ask_name"); ok {
				name, _ := reply.(string)
				if name == "" {
					name = "stranger"
				}
				greeting := fmt.Sprintf("Hello, %s!", name)
				// Emit Content so the console renders the greeting; the
				// terminal Output below is for downstream nodes / state.
				ev := session.NewEvent(nc.InvocationID())
				ev.Content = &genai.Content{Parts: []*genai.Part{{Text: greeting}}}
				if err := emit(ev); err != nil {
					return "", err
				}
				return greeting, nil
			}
			_, err := workflow.RunNode[any](nc, ask, nil)
			return "", err
		},
		workflow.NodeConfig{},
	)

	wa, err := workflowagent.New(workflowagent.Config{
		Name:        "dynamic_hitl_sample",
		Description: "Minimal dynamic workflow that pauses for human input via RunNode + ResumedInput.",
		Edges:       workflow.Chain(workflow.Start, greeter),
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
