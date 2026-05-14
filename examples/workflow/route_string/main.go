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

// Binary route_string is a minimal workflow sample that shows
// string routing: a node classifies the user's message and emits
// a routing event whose value is a category name; the engine
// dispatches to one of three downstream branches via
// workflow.StringRoute.
//
// No LLM, no HITL, no random — the smallest end-to-end
// demonstration of workflow.StringRoute and the Event.Routes
// contract.
//
//	go run ./examples/workflow/route_string/ console
//
//	User -> What time is it?
//	Agent -> answering question: What time is it?
//
//	User -> The sky is blue.
//	Agent -> commenting on statement: The sky is blue.
//
//	User -> Hello world!
//	Agent -> reacting to exclamation: Hello world!
package main

import (
	"context"
	"fmt"
	"iter"
	"log"
	"os"
	"strings"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/workflowagent"
	"google.golang.org/adk/cmd/launcher"
	"google.golang.org/adk/cmd/launcher/full"
	"google.golang.org/adk/session"
	"google.golang.org/adk/workflow"
)

// classifyNode inspects the user's message and emits an event
// whose Routes carries one of "question", "statement", or
// "exclamation". Cannot be a FunctionNode because FunctionNode
// does not let its body set Event.Routes; see the corresponding
// note in the route/ sample.
type classifyNode struct {
	workflow.BaseNode
}

func newClassifyNode() *classifyNode {
	return &classifyNode{
		BaseNode: workflow.NewBaseNode(
			"classify",
			"emits a routing event keyed on the message's terminal punctuation",
			workflow.NodeConfig{},
		),
	}
}

func (n *classifyNode) Run(ctx agent.InvocationContext, input any) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		msg, ok := input.(string)
		if !ok {
			yield(nil, fmt.Errorf("classify: expected string input, got %T", input))
			return
		}
		category := classify(msg)
		ev := session.NewEvent(ctx.InvocationID())
		ev.Routes = []string{category}
		ev.Actions.StateDelta["output"] = msg
		yield(ev, nil)
	}
}

// classify maps a message to a category by its last
// non-whitespace character. Trivial logic — the point of the
// sample is the routing primitive, not the classifier.
func classify(msg string) string {
	trimmed := strings.TrimSpace(msg)
	switch {
	case strings.HasSuffix(trimmed, "?"):
		return "question"
	case strings.HasSuffix(trimmed, "!"):
		return "exclamation"
	default:
		return "statement"
	}
}

func answerQuestion(_ agent.InvocationContext, msg string) (string, error) {
	return "answering question: " + msg, nil
}

func commentOnStatement(_ agent.InvocationContext, msg string) (string, error) {
	return "commenting on statement: " + msg, nil
}

func reactToExclamation(_ agent.InvocationContext, msg string) (string, error) {
	return "reacting to exclamation: " + msg, nil
}

func main() {
	ctx := context.Background()

	classify := newClassifyNode()
	question := workflow.NewFunctionNode("answer_question", answerQuestion, workflow.NodeConfig{})
	statement := workflow.NewFunctionNode("comment_statement", commentOnStatement, workflow.NodeConfig{})
	exclamation := workflow.NewFunctionNode("react_exclamation", reactToExclamation, workflow.NodeConfig{})

	// Graph:
	//
	//   START → classify ─┬─ "question"    → answer_question
	//                     ├─ "statement"   → comment_statement
	//                     └─ "exclamation" → react_exclamation
	edges := workflow.Concat(
		workflow.Chain(workflow.Start, classify),
		[]workflow.Edge{
			{From: classify, To: question, Route: workflow.StringRoute("question")},
			{From: classify, To: statement, Route: workflow.StringRoute("statement")},
			{From: classify, To: exclamation, Route: workflow.StringRoute("exclamation")},
		},
	)

	rootAgent, err := workflowagent.New(workflowagent.Config{
		Name:        "string_router",
		Description: "classifies a message and routes to one of three handlers",
		Edges:       edges,
	})
	if err != nil {
		log.Fatalf("failed to create workflow agent: %v", err)
	}

	log.Printf("string router ready — try messages ending in '?', '!', or '.'")

	cfg := &launcher.Config{
		AgentLoader: agent.NewSingleLoader(rootAgent),
	}
	l := full.NewLauncher()
	if err := l.Execute(ctx, cfg, os.Args[1:]); err != nil {
		log.Fatalf("Run failed: %v\n\n%s", err, l.CommandLineSyntax())
	}
}
