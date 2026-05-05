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
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/internal/typeutil"
	"google.golang.org/adk/session"
)

// defaultRetryConfig is the default retry configuration for a node.
var defaultRetryConfig = RetryConfig{
	MaxAttempts:   5,
	InitialDelay:  time.Second,
	MaxDelay:      60 * time.Second,
	BackoffFactor: 2.0,
	Jitter:        1.0,
	ShouldRetry: func(err error) bool {
		return true
	},
}

// Node is the interface for all nodes in a workflow.
type Node interface {
	Name() string
	Description() string
	Run(ctx agent.InvocationContext, input any) iter.Seq2[*session.Event, error]
	Config() NodeConfig
}

// NodeConfig defines the configuration for a node.
type NodeConfig struct {
	// Enables data parallelism (runs node concurrently for each item in input collection)
	ParallelWorker bool
	// Re-runs node on resume. Defaults to true for AgentNode
	RerunOnResume *bool
	// Wait for output before triggering edges. Defaults to true for Task agents
	WaitForOutput *bool
	// Retry configuration on failure
	RetryConfig *RetryConfig
	// Max duration for node to complete. Optional for global defaults
	Timeout *time.Duration
}

// RetryConfig defines the parameters for retrying a failed node.
type RetryConfig struct {
	// Maximum number of attempts, including the original request. If 0 or 1, it means no retries. If not specified, default to 5.
	MaxAttempts int
	// Initial delay before the first retry, in fractions of a second. If not specified, default to 1 second.
	InitialDelay time.Duration
	// Maximum delay between retries, in fractions of a second. If not specified, default to 60 seconds.
	MaxDelay time.Duration
	// Multiplier by which the delay increases after each attempt. If not specified, default to 2.0.
	BackoffFactor float64
	// Randomness factor for the delay. Use 0.0 to remove randomness. If not specified, default to 1.0.
	Jitter float64
	// Predicate that defines when to retry (true means retry). If not specified, default to true.
	ShouldRetry func(error) bool
}

// RetryOption defines a function that configures a RetryConfig.
type RetryOption func(*RetryConfig)

// NewRetryConfig creates a RetryConfig with sensible defaults and applies the provided options.
func NewRetryConfig(opts ...RetryOption) *RetryConfig {
	cfg := defaultRetryConfig
	for _, opt := range opts {
		opt(&cfg)
	}
	return &cfg
}

// WithMaxAttempts sets the maximum number of attempts.
func WithMaxAttempts(attempts int) RetryOption {
	return func(c *RetryConfig) {
		c.MaxAttempts = attempts
	}
}

// WithInitialDelay sets the initial delay before the first retry.
func WithInitialDelay(delay time.Duration) RetryOption {
	return func(c *RetryConfig) {
		c.InitialDelay = delay
	}
}

// WithMaxDelay sets the maximum delay between retries.
func WithMaxDelay(delay time.Duration) RetryOption {
	return func(c *RetryConfig) {
		c.MaxDelay = delay
	}
}

// WithBackoffFactor sets the multiplier by which the delay increases.
func WithBackoffFactor(factor float64) RetryOption {
	return func(c *RetryConfig) {
		c.BackoffFactor = factor
	}
}

// WithJitter sets the randomness factor for the delay.
func WithJitter(jitter float64) RetryOption {
	return func(c *RetryConfig) {
		c.Jitter = jitter
	}
}

// WithShouldRetry sets the predicate that defines when to retry.
func WithShouldRetry(fn func(error) bool) RetryOption {
	return func(c *RetryConfig) {
		c.ShouldRetry = fn
	}
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

// baseNode provides common fields for all nodes.
type baseNode struct {
	name        string
	description string
	config      NodeConfig
}

func (b *baseNode) Name() string        { return b.name }
func (b *baseNode) Description() string { return b.description }
func (b *baseNode) Config() NodeConfig  { return b.config }

// FunctionNode wraps a custom function.
type FunctionNode struct {
	baseNode
	fn     func(ctx agent.InvocationContext, input any) (any, error)
	config NodeConfig
}

// NewFunctionNode creates a new node wrapping a custom function using generics to automatically infer input and output types.
func NewFunctionNode[IN, OUT any](name string, fn func(ctx agent.InvocationContext, input IN) (OUT, error), cfg NodeConfig) *FunctionNode {
	wrappedFn := func(ctx agent.InvocationContext, input any) (any, error) {
		if input == nil {
			var zero IN
			return fn(ctx, zero)
		}
		typedInput, ok := input.(IN)
		if !ok {
			// Fallback to the json-like input types that cannot be converted by the standard type assertion.
			// E.g. tool nodes return map[string]any as input and user may define a struct as the target type.
			var err error
			typedInput, err = typeutil.ConvertToWithJSONSchema[any, IN](input, nil)
			if err != nil {
				return nil, fmt.Errorf("new function node: invalid input type, expected %T: %v", new(IN), err)
			}
		}
		return fn(ctx, typedInput)
	}
	return &FunctionNode{
		baseNode: baseNode{name: name},
		fn:       wrappedFn,
		config:   cfg,
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
func (s *startNode) Config() NodeConfig { return NodeConfig{} }

const defaultRoute = "__DEFAULT__"

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
func (w *Workflow) findNextNodes(currentNode Node, input any, event *session.Event) ([]nodeInput, error) {
	if len(w.edges[currentNode]) == 0 {
		return nil, nil
	}
	matched := false
	queue := []nodeInput{}
	added := make(map[Node]struct{})
	for _, edge := range w.edges[currentNode] {
		if _, ok := added[edge.To]; ok {
			continue
		}
		if edge.Route == nil {
			queue = append(queue, nodeInput{node: edge.To, input: input})
			added[edge.To] = struct{}{}
			matched = true
			continue
		}

		if edge.Route.Matches(event) {
			queue = append(queue, nodeInput{node: edge.To, input: input})
			added[edge.To] = struct{}{}
			matched = true
		}
	}
	if !matched {
		return nil, fmt.Errorf("no outgoing edge matches the event with routes %v emitted by node %s", event.Routes, currentNode.Name())
	}
	return queue, nil
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
			nextNodes, err := w.findNextNodes(currentNode, input, event)
			if err != nil {
				yield(nil, err)
				return
			}
			queue = append(queue, nextNodes...)
		}
	}
}
