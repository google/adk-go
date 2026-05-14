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

// Binary route is a minimal workflow sample that shows numeric
// routing: a node generates a random integer 1..10, the next node
// emits a routing event with that integer, and the engine
// dispatches to one of three downstream branches based on the
// value.
//
// No LLM, no HITL, no persistence — the smallest end-to-end
// demonstration of workflow.IntRoute / workflow.MultiRoute and
// the Event.Routes contract. Run it a few times to see different
// branches fire:
//
//	go run ./examples/workflow/route/ console
//
//	User -> hi
//	Agent -> rolled 7 — handling MID range (4..7)
//
//	User -> hi
//	Agent -> rolled 2 — handling LOW range (1..3)
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

// rollDie returns a random integer in 1..10. Ignores the user
// message — the input is irrelevant for this sample, the random
// number is what drives the routing.
func rollDie(_ agent.InvocationContext, _ string) (int, error) {
	return rand.IntN(10) + 1, nil
}

// routeByValueNode emits an event whose Routes carries the
// stringified upstream value. Cannot be a FunctionNode because
// FunctionNode does not let its body set Event.Routes — see the
// "Open question: should FunctionNode learn to emit Routes?"
// section in the HITL design doc — so we drop down to BaseNode.
//
// Output is the same value as the input so successor nodes can
// read it as their typed input.
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

func (n *routeByValueNode) Run(ctx agent.InvocationContext, input any) iter.Seq2[*session.Event, error] {
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
		ev.Actions.StateDelta["output"] = value
		yield(ev, nil)
	}
}

// handleLow / handleMid / handleHigh are tiny FunctionNodes that
// turn the integer into the agent's user-visible reply. The
// engine wires them up via routes declared on the edges below.
func handleLow(_ agent.InvocationContext, value int) (string, error) {
	return fmt.Sprintf("rolled %d — handling LOW range (1..3)", value), nil
}

func handleMid(_ agent.InvocationContext, value int) (string, error) {
	return fmt.Sprintf("rolled %d — handling MID range (4..7)", value), nil
}

func handleHigh(_ agent.InvocationContext, value int) (string, error) {
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
	// The first edge uses MultiRoute[int] to match a small set of
	// integers; the others use the same primitive for the rest of
	// the range. Could equivalently have used three IntRoute
	// edges per branch, but MultiRoute keeps it compact.
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
