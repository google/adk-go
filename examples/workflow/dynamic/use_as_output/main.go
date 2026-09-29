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

// Conditional output delegation in a dynamic workflow. The orchestrator
// decides in plain Go whether a request is in scope: an in-scope request is
// delegated to an LlmAgent child with workflow.WithUseAsOutput, so the agent's
// own event becomes the orchestrator's terminal output, while an out-of-scope
// request is answered by the orchestrator itself with an ordinary return.
package main

import (
	"context"
	"log"
	"os"
	"strings"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/agent/workflowagent"
	"google.golang.org/adk/v2/cmd/launcher"
	"google.golang.org/adk/v2/cmd/launcher/full"
	"google.golang.org/adk/v2/model/gemini"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/workflow"
)

func main() {
	ctx := context.Background()

	model, err := gemini.NewModel(ctx, "gemini-3.5-flash", &genai.ClientConfig{})
	if err != nil {
		log.Fatalf("gemini.NewModel: %v", err)
	}

	drafterAgent, err := llmagent.New(llmagent.Config{
		Name:        "drafter",
		Model:       model,
		Description: "Writes a short business email for a request.",
		Instruction: "Write a concise, polite business email for the request the user provides. " +
			"Reply with the email text only.",
	})
	if err != nil {
		log.Fatalf("llmagent.New: %v", err)
	}
	drafterNode, err := workflow.NewAgentNode(drafterAgent, workflow.NodeConfig{})
	if err != nil {
		log.Fatalf("workflow.NewAgentNode: %v", err)
	}

	assistant := workflow.NewDynamicNode[string, string]("assistant",
		func(nc agent.Context, request string, _ func(*session.Event) error) (string, error) {
			if !isEmailRequest(request) {
				// No delegation, so the string returned here becomes the
				// node's terminal output. The model is never called.
				return "Out of scope. Ask me to draft an email.", nil
			}
			// WithUseAsOutput hands the terminal output to the child: the
			// drafter's own event carries it and this node emits no event of
			// its own, so the reply is not repeated in a second event. The
			// value returned here is discarded.
			return workflow.RunNode[string](nc, drafterNode, request, workflow.WithUseAsOutput())
		},
		workflow.NodeConfig{},
	)

	wa, err := workflowagent.New(workflowagent.Config{
		Name:        "use_as_output_sample",
		Description: "Delegates the orchestrator's output to an LlmAgent child, or returns its own.",
		// Registering the wrapped LlmAgent lets the runner resolve its event
		// author. Without it the runner logs "Event from an unknown agent"
		// on every turn after a delegated one, when it walks the history.
		SubAgents: []agent.Agent{drafterAgent},
		Edges:     workflow.Chain(workflow.Start, assistant),
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

// isEmailRequest gates the delegating branch. The check is deliberately
// trivial and runs before any model call, so the reader can pick either branch
// from the console: the sample is about which node produces the output, not
// about how the decision is made.
func isEmailRequest(request string) bool {
	return strings.Contains(strings.ToLower(request), "email")
}
