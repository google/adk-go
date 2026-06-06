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

// Conditional dispatch with WithUseAsOutput: an email drafter
// generates a draft; the orchestrator either delegates to a sender
// (happy path, WithUseAsOutput) or runs a reviser and composes its
// own feedback (revise path, plain return).
package main

import (
	"context"
	"log"
	"os"
	"strings"

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

// Drafts longer than this are sent back to the reviser.
const maxDraftChars = 60

func main() {
	ctx := context.Background()

	model, err := gemini.NewModel(ctx, "gemini-3.1-flash-lite", &genai.ClientConfig{})
	if err != nil {
		log.Fatalf("gemini.NewModel: %v", err)
	}

	drafterAgent, err := llmagent.New(llmagent.Config{
		Name:        "drafter",
		Model:       model,
		Description: "Drafts a short business email from a topic.",
		Instruction: "Write a concise, polite business email about the topic the user provides.",
	})
	if err != nil {
		log.Fatalf("llmagent.New drafter: %v", err)
	}
	drafterNode, err := workflow.NewAgentNode(drafterAgent, workflow.NodeConfig{})
	if err != nil {
		log.Fatalf("workflow.NewAgentNode drafter: %v", err)
	}

	reviserAgent, err := llmagent.New(llmagent.Config{
		Name:        "reviser",
		Model:       model,
		Description: "Suggests how to shorten an over-long email draft.",
		Instruction: "The user will send you an email draft that is too long. " +
			"Reply with a one-paragraph suggestion on how to shorten it.",
	})
	if err != nil {
		log.Fatalf("llmagent.New reviser: %v", err)
	}
	reviserNode, err := workflow.NewAgentNode(reviserAgent, workflow.NodeConfig{})
	if err != nil {
		log.Fatalf("workflow.NewAgentNode reviser: %v", err)
	}

	// sender mocks delivery; a real system would call the outbound
	// mail API here.
	senderNode := workflow.NewFunctionNode(
		"sender",
		func(_ agent.InvocationContext, approvedDraft string) (string, error) {
			return "EMAIL SENT — " + approvedDraft, nil
		},
		workflow.NodeConfig{},
	)

	orchestrate := workflow.NewDynamicNode[string, string](
		"orchestrate",
		func(nc agent.Context, topic string, _ func(*session.Event) error) (string, error) {
			draft, err := workflow.RunNode[string](nc, drafterNode, topic)
			if err != nil {
				return "", err
			}
			log.Println("Draft: ", len(draft))
			if len(draft) <= maxDraftChars {
				// Happy path: delegate so the sender's confirmation
				// becomes orchestrate's output and flows downstream.
				return workflow.RunNode[string](nc, senderNode, draft, workflow.WithUseAsOutput())
			}

			// Revise path: no WithUseAsOutput; the composed string
			// below is what downstream sees.
			feedback, err := workflow.RunNode[string](nc, reviserNode, draft)
			if err != nil {
				return "", err
			}
			return "NEEDS REVISION — original draft: " + draft +
				" | reviser feedback: " + feedback, nil
		},
		workflow.NodeConfig{},
	)

	report := workflow.NewFunctionNode(
		"report",
		func(_ agent.InvocationContext, orchestrateOutput string) (string, error) {
			return "REPORT: " + strings.TrimSpace(orchestrateOutput), nil
		},
		workflow.NodeConfig{},
	)

	wa, err := workflowagent.New(workflowagent.Config{
		Name:        "use_as_output_sample",
		Description: "Conditional dispatch: send-or-revise email workflow.",
		Edges:       workflow.Chain(workflow.Start, orchestrate, report),
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
