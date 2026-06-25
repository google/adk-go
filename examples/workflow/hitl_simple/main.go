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
// The ask_name node emits a RequestInput that pauses the
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

	ask := workflow.NewEmittingFunctionNode[any, any]("ask_name",
		func(ctx context.Context, invCleanCtx agent.Context, _ any, emit func(*session.Event) error) (any, error) {
			if err := emit(workflow.NewRequestInputEvent(invCleanCtx, session.RequestInput{
				InterruptID: "ask_name",
				Message:     "What's your name?",
			})); err != nil {
				return nil, err
			}
			return nil, workflow.ErrNodeInterrupted
		},
		workflow.NodeConfig{},
	)

	// greet receives the user's reply (a string) and returns the
	// greeting. The classic NewFunctionNode is enough — no events
	// to emit beyond the terminal output.
	greet := workflow.NewFunctionNode("greet",
		func(_ context.Context, _ agent.Context, name string) (string, error) {
			if name == "" {
				name = "stranger"
			}
			return fmt.Sprintf("Hello, %s!", name), nil
		},
		workflow.NodeConfig{},
	)

	rootAgent, err := workflowagent.New(workflowagent.Config{
		Name:        "hitl_simple",
		Description: "minimal HITL workflow for console launcher verification",
		Edges:       workflow.Chain(workflow.Start, ask, greet),
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
