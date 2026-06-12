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
	"iter"
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

// routeByValueNode emits the routing event. A FunctionNode can't
// set Event.Routes from its body, so routing nodes drop down to
// BaseNode.
type routeByValueNode struct {
	workflow.BaseNode
}

func newRouteByValueNode() *routeByValueNode {
	return &routeByValueNode{
		BaseNode: workflow.NewBaseNode(
			"route_by_value",
			"emits a routing event keyed on the upstream integer",
			workflow.NodeConfig{},
		),
	}
}

func (n *routeByValueNode) Run(ctx agent.Context, input any) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		value, ok := input.(int)
		if !ok {
			yield(nil, fmt.Errorf("route_by_value: expected int input, got %T", input))
			return
		}
		ev := session.NewEvent(ctx.InvocationID())
		// IntRoute and MultiRoute[int] both compare against the
		// stringified value, so emit it that way.
		ev.Routes = []string{fmt.Sprint(value)}
		// Event.Output is the channel the engine reads to feed the
		// successor node's typed input.
		ev.Output = value
		yield(ev, nil)
	}
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
	routeNode := newRouteByValueNode()
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
