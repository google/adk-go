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

// hitl_rerun shows the re-entry HITL pattern: a single emitting
// FunctionNode both pauses for input and produces the final output.
// Unlike hitl_simple (two nodes, handoff resume), here one node is
// re-run from scratch on resume (NodeConfig.RerunOnResume = &true).
//
// workflow.ResumeOrRequestInput collapses the two phases into one
// call: on the first pass it emits a RequestInput and returns
// ErrNodeInterrupted (pause, no output); after resume it returns the
// human's reply, which the body turns into the terminal output.
//
//	go run ./examples/workflow/hitl_rerun/ console
//
//	User -> hello
//	Agent -> What's your name?
//	User -> Alice
//	Agent -> Hello, Alice!
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/workflowagent"
	"google.golang.org/adk/v2/cmd/launcher"
	"google.golang.org/adk/v2/cmd/launcher/full"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/workflow"
)

func main() {
	ctx := context.Background()

	rerun := true
	greet := workflow.NewEmittingFunctionNode[any, any]("greet",
		func(nc agent.Context, _ any, emit func(*session.Event) error) (any, error) {
			// ResumeOrRequestInput pauses (asks and returns
			// ErrNodeInterrupted) on the first pass, and returns the
			// human's reply once the node is re-run after the answer.
			// InterruptID embeds the invocation ID: stable across this
			// run's re-entry so the reply still correlates, yet unique
			// per run so the Dev UI re-prompts on a later run.
			reply, err := workflow.ResumeOrRequestInput(nc, emit, session.RequestInput{
				InterruptID: "ask_name-" + nc.InvocationID(),
				Message:     "What's your name?",
			})
			if err != nil {
				return nil, err
			}

			name, _ := reply.(string)
			if name == "" {
				name = "stranger"
			}
			return fmt.Sprintf("Hello, %s!", name), nil
		},
		workflow.NodeConfig{RerunOnResume: &rerun},
	)

	rootAgent, err := workflowagent.New(workflowagent.Config{
		Name:        "hitl_rerun",
		Description: "single-node re-entry HITL workflow",
		Edges:       workflow.Chain(workflow.Start, greet),
	})
	if err != nil {
		log.Fatalf("failed to create workflow agent: %v", err)
	}

	log.Printf("hitl_rerun sample ready — type anything to start, then answer the prompt")

	launcherCfg := &launcher.Config{
		AgentLoader: agent.NewSingleLoader(rootAgent),
	}
	l := full.NewLauncher()
	if err := l.Execute(ctx, launcherCfg, os.Args[1:]); err != nil {
		log.Fatalf("Run failed: %v\n\n%s", err, l.CommandLineSyntax())
	}
}
