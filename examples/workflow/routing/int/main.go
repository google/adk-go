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

// Command int demonstrates numeric routing with workflow.IntRoute /
// workflow.MultiRoute: a node rolls a random integer 1..10 and the
// engine dispatches to one of three branches based on the value.
//
//	go run ./examples/workflow/routing/int/ console
package main

import (
	"context"
	"fmt"
	"log"
	"math/rand/v2"
	"os"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/workflowagent"
	"google.golang.org/adk/cmd/launcher"
	"google.golang.org/adk/cmd/launcher/full"
	"google.golang.org/adk/session"
	"google.golang.org/adk/workflow"
)

// rollDie ignores the user message; the random number is what
// drives the routing.
func rollDie(_ agent.Context, _ string) (int, error) {
	return rand.IntN(10) + 1, nil
}

// routeByValue emits a routing event keyed on the upstream integer.
// Setting Event.Routes requires emitting a custom event, which an
// emitting FunctionNode can do; returning nil suppresses the default
// terminal event so this single emit carries both the route and the
// output.
func routeByValue(ctx agent.Context, value int, emit func(*session.Event) error) (any, error) {
	ev := session.NewEvent(ctx, ctx.InvocationID())
	// IntRoute and MultiRoute[int] both compare against the
	// stringified value, so emit it that way.
	ev.Routes = []string{fmt.Sprint(value)}
	// Event.Output feeds the successor node's typed input.
	ev.Output = value
	if err := emit(ev); err != nil {
		return nil, err
	}
	return nil, nil
}

func handleLow(_ agent.Context, value int) (string, error) {
	return fmt.Sprintf("rolled %d — handling LOW range (1..3)", value), nil
}

func handleMid(_ agent.Context, value int) (string, error) {
	return fmt.Sprintf("rolled %d — handling MID range (4..7)", value), nil
}

func handleHigh(_ agent.Context, value int) (string, error) {
	return fmt.Sprintf("rolled %d — handling HIGH range (8..10)", value), nil
}

func main() {
	ctx := context.Background()

	rollNode := workflow.NewFunctionNode("roll_die", rollDie, workflow.NodeConfig{})
	routeNode := workflow.NewEmittingFunctionNode("route_by_value", routeByValue, workflow.NodeConfig{})
	lowNode := workflow.NewFunctionNode("handle_low", handleLow, workflow.NodeConfig{})
	midNode := workflow.NewFunctionNode("handle_mid", handleMid, workflow.NodeConfig{})
	highNode := workflow.NewFunctionNode("handle_high", handleHigh, workflow.NodeConfig{})

	// Graph:
	//
	//   START → roll_die → route_by_value
	//                         ├─ {1, 2, 3}      → handle_low
	//                         ├─ {4, 5, 6, 7}   → handle_mid
	//                         └─ {8, 9, 10}     → handle_high
	//
	// MultiRoute[int] matches a set of values per edge.
	edges := workflow.Concat(
		workflow.Chain(workflow.Start, rollNode, routeNode),
		[]workflow.Edge{
			{From: routeNode, To: lowNode, Route: workflow.MultiRoute[int]{1, 2, 3}},
			{From: routeNode, To: midNode, Route: workflow.MultiRoute[int]{4, 5, 6, 7}},
			{From: routeNode, To: highNode, Route: workflow.MultiRoute[int]{8, 9, 10}},
		},
	)

	rootAgent, err := workflowagent.New(workflowagent.Config{
		Name:        "random_router",
		Description: "rolls a 1..10 die and routes to one of three handlers",
		Edges:       edges,
	})
	if err != nil {
		log.Fatalf("failed to create workflow agent: %v", err)
	}

	log.Printf("random router ready — type any message and watch the route change between runs")

	cfg := &launcher.Config{
		AgentLoader: agent.NewSingleLoader(rootAgent),
	}
	l := full.NewLauncher()
	if err := l.Execute(ctx, cfg, os.Args[1:]); err != nil {
		log.Fatalf("Run failed: %v\n\n%s", err, l.CommandLineSyntax())
	}
}
