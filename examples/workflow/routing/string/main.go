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

// Command string demonstrates string routing with
// workflow.StringRoute: a node classifies the user's message into a
// category and the engine dispatches to one of three branches.
//
//	go run ./examples/workflow/routing/string/ console
package main

import (
	"context"
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

// classifyAndRoute emits a routing event keyed on the message
// category; returning nil suppresses the default terminal event.
func classifyAndRoute(ctx context.Context, invCleanCtx agent.Context, msg string, emit func(*session.Event) error) (any, error) {
	ev := session.NewEvent(invCleanCtx.InvocationID())
	ev.Routes = []string{classify(msg)}
	ev.Output = msg // feeds the successor's typed input
	if err := emit(ev); err != nil {
		return nil, err
	}
	return nil, nil
}

// classify maps a message to a category by its terminal punctuation.
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

func answerQuestion(_ context.Context, _ agent.Context, msg string) (string, error) {
	return "answering question: " + msg, nil
}

func commentOnStatement(_ context.Context, _ agent.Context, msg string) (string, error) {
	return "commenting on statement: " + msg, nil
}

func reactToExclamation(_ context.Context, _ agent.Context, msg string) (string, error) {
	return "reacting to exclamation: " + msg, nil
}

func main() {
	ctx := context.Background()

	classify := workflow.NewEmittingFunctionNode("classify", classifyAndRoute, workflow.NodeConfig{})
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
