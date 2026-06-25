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

// Command llm demonstrates LLM-driven routing: an LLM agent
// classifies the user's message into one word, and a downstream
// node turns that into an Event.Routes value dispatched via
// workflow.StringRoute. This is the canonical "LLM as the brain,
// engine does the routing" pattern.
//
// Requires GOOGLE_API_KEY in the environment.
//
//	export GOOGLE_API_KEY=...
//	go run ./examples/workflow/routing/llm/ console
package main

import (
	"context"
	"fmt"
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

// classifierInstruction is the only prompt the LLM sees. The
// "answer with EXACTLY one word" constraint is the contract the
// downstream routing node depends on; deviations from it (e.g.
// "this is a question") fall through to the default branch.
const classifierInstruction = `Classify the user's message into one of three categories:
- "question": ends with '?' or asks for information
- "exclamation": expresses strong emotion, often ends with '!'
- "statement": a neutral declarative sentence

Answer with EXACTLY one word, lowercase, no punctuation: question, exclamation, or statement.`

// routeByClassification turns the classifier's one-word output into a
// routing event; returning nil suppresses the default terminal event.
func routeByClassification(ctx agent.Context, input any, emit func(*session.Event) error) (any, error) {
	// Normalise defensively in case the LLM ignored the one-word
	// instruction; off-script replies fall through to "statement".
	category := strings.TrimRight(strings.ToLower(strings.TrimSpace(fmt.Sprint(input))), ".")
	if category != "question" && category != "exclamation" && category != "statement" {
		category = "statement"
	}
	ev := session.NewEventWithContext(ctx, ctx.InvocationID())
	ev.Routes = []string{category}
	if err := emit(ev); err != nil {
		return nil, err
	}
	return nil, nil
}

func answerQuestion(ctx agent.Context, _ any) (string, error) {
	return "answering question: " + userMessage(ctx), nil
}

func commentOnStatement(ctx agent.Context, _ any) (string, error) {
	return "commenting on statement: " + userMessage(ctx), nil
}

func reactToExclamation(ctx agent.Context, _ any) (string, error) {
	return "reacting to exclamation: " + userMessage(ctx), nil
}

// userMessage reads the original user text from ctx.UserContent.
// Handlers read it here rather than as graph input, since the
// route node forwards only the one-word classification.
func userMessage(ctx agent.Context) string {
	uc := ctx.UserContent()
	if uc == nil {
		return ""
	}
	var sb strings.Builder
	for _, p := range uc.Parts {
		sb.WriteString(p.Text)
	}
	return strings.TrimSpace(sb.String())
}

func main() {
	ctx := context.Background()

	apiKey := os.Getenv("GOOGLE_API_KEY")
	if apiKey == "" {
		log.Fatalf("GOOGLE_API_KEY is required to run this sample")
	}

	model, err := gemini.NewModel(ctx, "gemini-flash-latest", &genai.ClientConfig{APIKey: apiKey})
	if err != nil {
		log.Fatalf("failed to create model: %v", err)
	}

	classifier, err := llmagent.New(llmagent.Config{
		Name:        "classify",
		Model:       model,
		Description: "classifies the user's message as question / exclamation / statement",
		Instruction: classifierInstruction,
		// OutputKey persists the classifier's reply to session
		// state under "output". It is optional for the routing data
		// flow: AgentNode synthesizes the reply into Event.Output
		// regardless, and the scheduler feeds that to the next node.
		OutputKey: "output",
	})
	if err != nil {
		log.Fatalf("failed to create classifier agent: %v", err)
	}

	classifyNode, err := workflow.NewAgentNode(classifier, workflow.NodeConfig{})
	if err != nil {
		log.Fatalf("failed to create classify node: %v", err)
	}
	routeNode := workflow.NewEmittingFunctionNode("route_by_classification", routeByClassification, workflow.NodeConfig{})
	question := workflow.NewFunctionNode("answer_question", answerQuestion, workflow.NodeConfig{})
	statement := workflow.NewFunctionNode("comment_statement", commentOnStatement, workflow.NodeConfig{})
	exclamation := workflow.NewFunctionNode("react_exclamation", reactToExclamation, workflow.NodeConfig{})

	// Graph:
	//
	//   START → classify (LLM) → route_by_classification ─┬─ "question"    → answer_question
	//                                                     ├─ "statement"   → comment_statement
	//                                                     └─ "exclamation" → react_exclamation
	edges := workflow.Concat(
		workflow.Chain(workflow.Start, classifyNode, routeNode),
		[]workflow.Edge{
			{From: routeNode, To: question, Route: workflow.StringRoute("question")},
			{From: routeNode, To: statement, Route: workflow.StringRoute("statement")},
			{From: routeNode, To: exclamation, Route: workflow.StringRoute("exclamation")},
		},
	)

	rootAgent, err := workflowagent.New(workflowagent.Config{
		Name:        "llm_router",
		Description: "asks an LLM to classify the user's message and routes to one of three handlers",
		Edges:       edges,
		// Register the wrapped LLM agent so the runner can resolve
		// its event author; otherwise it logs a harmless "Event
		// from an unknown agent: classify" on every turn.
		SubAgents: []agent.Agent{classifier},
	})
	if err != nil {
		log.Fatalf("failed to create workflow agent: %v", err)
	}

	log.Printf("LLM router ready — try messages of different shapes")

	cfg := &launcher.Config{
		AgentLoader: agent.NewSingleLoader(rootAgent),
	}
	l := full.NewLauncher()
	if err := l.Execute(ctx, cfg, os.Args[1:]); err != nil {
		log.Fatalf("Run failed: %v\n\n%s", err, l.CommandLineSyntax())
	}
}
