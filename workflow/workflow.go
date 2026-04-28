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

type StringRoute string

func (r StringRoute) Matches(event *session.Event) bool {
	for _, v := range event.Route {
		if v == string(r) {
			return true
		}
	}
	return false
}

type IntRoute int

func (r IntRoute) Matches(event *session.Event) bool {
	str := fmt.Sprint(r)
	for _, v := range event.Route {
		if v == str {
			return true
		}
	}
	return false
}

type BoolRoute bool

func (r BoolRoute) Matches(event *session.Event) bool {
	str := fmt.Sprint(r)
	for _, v := range event.Route {
		if v == str {
			return true
		}
	}
	return false
}

// BaseNode provides common fields for all nodes.
type BaseNode struct {
	name        string
	description string
}

func (b *BaseNode) Name() string        { return b.name }
func (b *BaseNode) Description() string { return b.description }

// FunctionNode wraps a custom function.
type FunctionNode struct {
	BaseNode
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
		BaseNode: BaseNode{name: name},
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

// START is a sentinel node used to indicate the entry point of the workflow.
var START Node = &startNode{}

type startNode struct{}

func (s *startNode) Name() string        { return "START" }
func (s *startNode) Description() string { return "Start node" }
func (s *startNode) Run(ctx agent.InvocationContext, input any) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {}
}

const DEFAULT_ROUTE = "__DEFAULT__"

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

type queueItem struct {
	node  Node
	input any
}

func (w *Workflow) findNextNodes(currentNode Node, input any, eventList []*session.Event) ([]queueItem, error) {
	if len(w.edges[currentNode]) == 0 {
		return nil, nil
	}
	matched := false
	queue := []queueItem{}
	for _, edge := range w.edges[currentNode] {
		if edge.Route == nil {
			queue = append(queue, queueItem{node: edge.To, input: input})
			matched = true
			continue
		}

		for _, event := range eventList {
			if edge.Route.Matches(event) {
				queue = append(queue, queueItem{node: edge.To, input: input})
				matched = true
				break
			}
		}
	}
	if !matched {
		return nil, fmt.Errorf("node %s produces route tags that do not match any valid outgoing edge", currentNode.Name())
	}
	return queue, nil
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

		var queue []queueItem
		if startEdges, ok := w.edges[START]; ok {
			for _, edge := range startEdges {
				queue = append(queue, queueItem{node: edge.To, input: input})
			}
		}

		if len(queue) == 0 {
			yield(nil, fmt.Errorf("no start node found"))
			return
		}

		for len(queue) > 0 {
			currentNode := queue[0].node
			input = queue[0].input
			queue = queue[1:]

			var eventList []*session.Event
			events := currentNode.Run(ctx, input)
			for ev, err := range events {
				if err != nil {
					yield(nil, err)
					return
				}
				if !yield(ev, nil) {
					return
				}

				eventList = append(eventList, ev)

				// Extract output for next node
				if ev.Actions.StateDelta != nil {
					if out, ok := ev.Actions.StateDelta["output"]; ok {
						input = out
					}
				}
			}
			nextNodes, err := w.findNextNodes(currentNode, input, eventList)
			if err != nil {
				yield(nil, err)
				return
			}
			queue = append(queue, nextNodes...)
		}
	}
}
