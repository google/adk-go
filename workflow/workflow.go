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

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/session"
)

// Node is the interface for all nodes in a workflow.
type Node interface {
	Name() string
	Description() string
	Run(ctx agent.InvocationContext, input any) iter.Seq2[*session.Event, error]
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

type StringRoute string

func (r StringRoute) Matches(event *session.Event) bool {
	return matchRoute(string(r), event)
}

type IntRoute int

func (r IntRoute) Matches(event *session.Event) bool {
	return matchRoute(fmt.Sprint(r), event)
}

type BoolRoute bool

func (r BoolRoute) Matches(event *session.Event) bool {
	return matchRoute(fmt.Sprint(r), event)
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
}

func (b *baseNode) Name() string        { return b.name }
func (b *baseNode) Description() string { return b.description }

// FunctionNode wraps a custom function.
type FunctionNode struct {
	baseNode
	fn func(ctx agent.InvocationContext, input any) (any, error)
}

// NewFunctionNode creates a new node wrapping a custom function using generics to automatically infer input and output types.
func NewFunctionNode[IN any, OUT any](name string, fn func(ctx agent.InvocationContext, input IN) (OUT, error)) *FunctionNode {
	wrappedFn := func(ctx agent.InvocationContext, input any) (any, error) {
		if input == nil {
			var zero IN
			return fn(ctx, zero)
		}
		typedInput, ok := input.(IN)
		if !ok {
			return nil, fmt.Errorf("invalid input type, expected %T", new(IN))
		}
		return fn(ctx, typedInput)
	}
	return &FunctionNode{
		baseNode: baseNode{name: name},
		fn:       wrappedFn,
	}
}

func (n *FunctionNode) Run(ctx agent.InvocationContext, input any) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		output, err := n.fn(ctx, input)
		if err != nil {
			yield(nil, err)
			return
		}

		event := session.NewEvent(ctx.InvocationID())
		event.Actions.StateDelta["output"] = output
		if s, ok := output.(string); ok {
			event.Content = &genai.Content{
				Parts: []*genai.Part{{Text: s}},
			}
		}
		yield(event, nil)
	}
}

// Edge defines a directed connection between nodes in the workflow graph.
type Edge struct {
	From  Node  // The source node
	To    Node  // The target node
	Route Route // Routing condition
}

// Chain generates a slice of Edges to form a chain of nodes.
func Chain(nodes ...Node) []Edge {
	if len(nodes) < 2 {
		return nil
	}
	edges := make([]Edge, len(nodes)-1)
	for i := 0; i < len(nodes)-1; i++ {
		edges[i] = Edge{From: nodes[i], To: nodes[i+1]}
	}
	return edges
}

// Concat combines Edges and []Edge slices into a single slice of edges.
func Concat(items ...any) []Edge {
	var edges []Edge
	for _, item := range items {
		switch v := item.(type) {
		case Edge:
			edges = append(edges, v)
		case []Edge:
			edges = append(edges, v...)
		}
	}
	return edges
}

// Start is a sentinel node used to indicate the entry point of the workflow.
var Start Node = &startNode{}

type startNode struct{}

func (s *startNode) Name() string        { return "START" }
func (s *startNode) Description() string { return "Start node" }
func (s *startNode) Run(ctx agent.InvocationContext, input any) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {}
}

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
				yield(nil, fmt.Errorf("node %s produced multiple events with route tags. Only one event per execution can specify routes.", currentNode.Name()))
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
