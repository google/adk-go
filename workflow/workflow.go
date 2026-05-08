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

package workflow

import (
	"fmt"
	"iter"
	"strings"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/session"
)

// Node is the interface for all nodes in a workflow.
type Node interface {
	Name() string
	Description() string
	Run(ctx agent.InvocationContext, input any) iter.Seq2[*session.Event, error]
	Config() NodeConfig
}

// Route defines the interface for matching execution results to edges.
type Route interface {
	Matches(event *session.Event) bool
}

func matchRoute(routeValue string, event *session.Event) bool {
	for _, v := range event.Routes {
		if v == routeValue {
			return true
		}
	}
	return false
}

// StringRoute is a route defined by a string value.
type StringRoute string

func (r StringRoute) Matches(event *session.Event) bool {
	return matchRoute(string(r), event)
}

// IntRoute is a route defined by an integer value.
type IntRoute int

func (r IntRoute) Matches(event *session.Event) bool {
	return matchRoute(fmt.Sprint(r), event)
}

// BoolRoute is a route defined by a boolean value.
type BoolRoute bool

func (r BoolRoute) Matches(event *session.Event) bool {
	return matchRoute(fmt.Sprint(r), event)
}

// MultiRoute matches any value within a specified list of allowed routes.
type MultiRoute[T comparable] []T

func (r MultiRoute[T]) Matches(event *session.Event) bool {
	for _, route := range r {
		if matchRoute(fmt.Sprint(route), event) {
			return true
		}
	}
	return false
}

// DefaultRoute is a special route that matches when no other concrete routes match.
var Default = &defaultRoute{}

type defaultRoute struct{}

func (r *defaultRoute) Matches(event *session.Event) bool {
	return false
}

// baseNode provides common fields for all nodes.
type baseNode struct {
	name        string
	description string
	config      NodeConfig
}

func (b *baseNode) Name() string        { return b.name }
func (b *baseNode) Description() string { return b.description }
func (b *baseNode) Config() NodeConfig  { return b.config }


// Edge defines a directed connection between nodes in the workflow graph.
type Edge struct {
	From  Node  // The source node
	To    Node  // The target node
	Route Route // Routing condition
}

// Start is a sentinel node used to indicate the entry point of the workflow.
var Start Node = &startNode{}

type startNode struct{}

func (s *startNode) Name() string        { return "START" }
func (s *startNode) Description() string { return "Start node" }
func (s *startNode) Run(ctx agent.InvocationContext, input any) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {}
}
func (s *startNode) Config() NodeConfig { return NodeConfig{} }

// Workflow manages the workflow graph execution.
type Workflow struct {
	edges map[Node][]Edge
}

// New creates a new Workflow engine with the given edges.
func New(edges []Edge) *Workflow {
	adj := make(map[Node][]Edge)
	for _, edge := range edges {
		adj[edge.From] = append(adj[edge.From], edge)
	}
	return &Workflow{edges: adj}
}

type nodeInput struct {
	node  Node
	input any
}

// findNextNodes determines the set of nodes to execute next based on the outgoing edges of the currentNode.
// It evaluates routes attached to edges against the provided session.Event.
//
// Behavior:
//   - Edges with no route condition always match.
//   - Edges with a route condition match only if the route matches the event.
//   - Duplicate target nodes are excluded to avoid queuing the same node multiple times.
//   - If there are outgoing edges but none of them match (neither by route nor by being unrouted),
//     it falls back to the default route (TODO: hanorik - add default route support).
func (w *Workflow) findNextNodes(currentNode Node, input any, event *session.Event) []nodeInput {
	if len(w.edges[currentNode]) == 0 {
		return nil
	}
	matched := false
	queue := []nodeInput{}
	added := make(map[Node]struct{})
	var defaultRouteNode Node
	for _, edge := range w.edges[currentNode] {
		if _, ok := added[edge.To]; ok {
			continue
		}
		if edge.Route == nil {
			queue = append(queue, nodeInput{node: edge.To, input: input})
			added[edge.To] = struct{}{}
			continue
		}
		if edge.Route == Default {
			defaultRouteNode = edge.To
			continue
		}
		if edge.Route.Matches(event) {
			queue = append(queue, nodeInput{node: edge.To, input: input})
			added[edge.To] = struct{}{}
			matched = true
		}
	}
	if !matched && defaultRouteNode != nil {
		queue = append(queue, nodeInput{node: defaultRouteNode, input: input})
	}

	return queue
}

// Run executes the workflow.
func (w *Workflow) Run(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		var input any
		userContent := ctx.UserContent()
		if userContent != nil {
			var sb strings.Builder
			for _, part := range userContent.Parts {
				if part.Text != "" {
					sb.WriteString(part.Text)
				}
			}
			input = sb.String()
		}

		queue := []nodeInput{{Start, input}}

		for len(queue) > 0 {
			currentNode := queue[0].node
			input = queue[0].input
			queue = queue[1:]

			var eventsWithRoutes []*session.Event
			var outputData any
			for ev, err := range currentNode.Run(ctx, input) {
				if err != nil {
					yield(nil, err)
					return
				}
				if !yield(ev, nil) {
					return
				}

				if ev.Routes != nil {
					eventsWithRoutes = append(eventsWithRoutes, ev)
				}

				// Extract output for next node
				if ev.Actions.StateDelta != nil {
					if out, ok := ev.Actions.StateDelta["output"]; ok {
						outputData = out
					}
				}
			}

			if len(eventsWithRoutes) > 1 {
				yield(nil, fmt.Errorf("node %s produced multiple events with route tags. Only one event per execution can specify routes", currentNode.Name()))
				return
			}
			var event *session.Event
			if len(eventsWithRoutes) == 1 {
				event = eventsWithRoutes[0]
			}
			if currentNode != Start {
				input = outputData
			}
			nextNodes := w.findNextNodes(currentNode, input, event)
			queue = append(queue, nextNodes...)
		}
	}
}
