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

// Binary route_llm is a workflow sample that uses an LLM to pick
// the route. The LLM agent is wrapped via workflow.NewAgentNode
// and writes its one-word classification into the workflow's
// "output" magic state key (LLMAgent.OutputKey="output"). A
// trivial routing node downstream reads that classification and
// emits the corresponding Event.Routes value, dispatching to one
// of three handlers via workflow.StringRoute.
//
// Mirrors the structure of adk-python's
// contributing/workflow_samples/route/ sample (LLM classifier +
// plain function emitting the routing event), the canonical
// pattern for "use an LLM as the brain inside a graph that the
// engine routes for you."
//
// Requires GOOGLE_API_KEY in the environment.
//
//	export GOOGLE_API_KEY=...
//	go run ./examples/workflow/route_llm/ console
//
//	User -> What time is it?
//	Agent -> answering question: What time is it?
//
//	User -> Hello world!
//	Agent -> reacting to exclamation: Hello world!
//
//	User -> The sky is blue.
//	Agent -> commenting on statement: The sky is blue.
package main

import (
	"context"
	"fmt"
	"iter"
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

// routeFromClassificationNode reads the classifier's one-word
// output, normalises it, and emits the corresponding routing
// event. Cannot be a FunctionNode because FunctionNode does not
// let its body set Event.Routes — same gap the route_string and
// route samples already note.
type routeFromClassificationNode struct {
	workflow.BaseNode
}

func newRouteNode() *routeFromClassificationNode {
	return &routeFromClassificationNode{
		BaseNode: workflow.NewBaseNode(
			"route_by_classification",
			"emits a routing event keyed on the LLM's one-word classification",
			workflow.NodeConfig{},
		),
	}
}

func (n *routeFromClassificationNode) Run(ctx agent.InvocationContext, input any) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		// Input is whatever the upstream LLMAgent wrote to
		// StateDelta["output"] via OutputKey="output". For a
		// well-behaved classifier that's a one-word string;
		// other shapes get coerced via fmt.Sprint and then
		// normalised to lowercase / trimmed.
		category := strings.ToLower(strings.TrimSpace(fmt.Sprint(input)))
		// Strip a trailing period in case the LLM ignored
		// "no punctuation" — defensive.
		category = strings.TrimRight(category, ".")
		if category != "question" && category != "exclamation" && category != "statement" {
			// Anything off-script falls through to "statement"
			// so the workflow always lands on a handler. A
			// stricter implementation could route to an error
			// branch instead.
			category = "statement"
		}
		ev := session.NewEvent(ctx.InvocationID())
		ev.Routes = []string{category}
		yield(ev, nil)
	}
}

func answerQuestion(ctx agent.InvocationContext, _ any) (string, error) {
	return "answering question: " + userMessage(ctx), nil
}

func commentOnStatement(ctx agent.InvocationContext, _ any) (string, error) {
	return "commenting on statement: " + userMessage(ctx), nil
}

func reactToExclamation(ctx agent.InvocationContext, _ any) (string, error) {
	return "reacting to exclamation: " + userMessage(ctx), nil
}

// userMessage extracts the original user text from the
// invocation context. Each handler reads it directly from
// ctx.UserContent rather than threading it as graph input,
// because the route node doesn't have it (it sees only the
// classifier's one-word output) and forwarding it would mean
// every routing handler taking a (message, classification) pair.
func userMessage(ctx agent.InvocationContext) string {
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
		// OutputKey="output" is the magic that lets the
		// classifier's reply flow into the workflow's
		// per-activation StateDelta["output"], which the
		// scheduler hands to the next node as its input.
		OutputKey: "output",
	})
	if err != nil {
		log.Fatalf("failed to create classifier agent: %v", err)
	}

	classifyNode := workflow.NewAgentNode(classifier, workflow.NodeConfig{})
	routeNode := newRouteNode()
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
		// Register the wrapped LLM agent as a sub-agent so the
		// runner's findAgentToRun (runner/runner.go:329-356)
		// can resolve event.Author="classify" against the
		// agent tree. Without it the runner logs a harmless
		// "Event from an unknown agent: classify" warning on
		// every turn — workflow nodes are not visible in the
		// agent tree by default. The runner still routes the
		// turn to llm_router as a whole because
		// isTransferableAcrossAgentTree returns false when the
		// chain to root contains a non-LLMAgent (the workflow
		// wrapper itself), so this registration is purely for
		// FindAgent's lookup.
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
