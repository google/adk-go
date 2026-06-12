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
// re-run from scratch on resume (NodeConfig.RerunOnResume = &true),
// so the same body has two branches:
//
//   - first pass: emit a RequestInput and return ErrNodeInterrupted
//     (pause, no output);
//   - after resume: read the reply via NodeContext.ResumedInput and
//     return the greeting as the terminal output.
//
// This is where ErrNodeInterrupted matters: the pause branch must end
// without an output, while the resume branch returns one.
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

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/workflowagent"
	"google.golang.org/adk/cmd/launcher"
	"google.golang.org/adk/cmd/launcher/full"
	"google.golang.org/adk/session"
	"google.golang.org/adk/workflow"
)

func main() {
	ctx := context.Background()

	rerun := true
	greet := workflow.NewEmittingFunctionNode[any, any]("greet",
		func(nc workflow.NodeContext, _ any, emit func(*session.Event) error) (any, error) {
			// Resume branch: the node was re-run after the human
			// replied, so the answer is available here.
			if reply, ok := nc.ResumedInput("ask_name"); ok {
				name, _ := reply.(string)
				if name == "" {
					name = "stranger"
				}
				return fmt.Sprintf("Hello, %s!", name), nil
			}

			// First pass: ask and pause. ErrNodeInterrupted ends the
			// activation without a terminal output.
			if err := emit(workflow.NewRequestInputEvent(nc, session.RequestInput{
				InterruptID: "ask_name",
				Message:     "What's your name?",
			})); err != nil {
				return nil, err
			}
			return nil, workflow.ErrNodeInterrupted
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
