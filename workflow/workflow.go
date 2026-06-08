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

	"github.com/google/jsonschema-go/jsonschema"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/session"
)

// Node is the interface for all nodes in a workflow.
//
// Custom nodes typically embed BaseNode (constructed via NewBaseNode)
// to inherit Name, Description, and Config implementations, and
// supply only Run.
type Node interface {
	Name() string
	Description() string
	Config() NodeConfig
	// InputSchema returns the JSON schema for the node's input.
	// If it returns nil, it indicates there is no schema and no input validation is performed.
	InputSchema() *jsonschema.Resolved
	// OutputSchema returns the JSON schema for the node's output.
	// If it returns nil, it indicates there is no schema and no output validation is performed.
	OutputSchema() *jsonschema.Resolved
	// ValidateInput validates and optionally coerces/transforms the input before the node runs.
	// It returns the validated input (which might be coerced/parsed/transformed) or an error.
	// It will be called by the scheduler before Run on every activation.
	ValidateInput(input any) (any, error)
	// ValidateOutput validates and optionally coerces/transforms an output emitted by the node.
	// It returns the validated output (which might be coerced/parsed/transformed) or an error.
	// The scheduler invokes it on every yielded event with a non-nil output, before forwarding it to the consumer.
	ValidateOutput(output any) (any, error)
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

// Edge defines a directed connection between nodes in the workflow graph.
type Edge struct {
	From  Node  // The source node
	To    Node  // The target node
	Route Route // Routing condition
}

// Start is a sentinel node used to indicate the entry point of the workflow.
var Start Node = &startNode{}

type startNode struct{}

func (s *startNode) Name() string                       { return "START" }
func (s *startNode) Description() string                { return "Start node" }
func (s *startNode) Config() NodeConfig                 { return NodeConfig{} }
func (s *startNode) InputSchema() *jsonschema.Resolved  { return nil }
func (s *startNode) OutputSchema() *jsonschema.Resolved { return nil }
func (s *startNode) ValidateInput(input any) (any, error) {
	return input, nil
}

func (s *startNode) ValidateOutput(output any) (any, error) {
	return output, nil
}

func (s *startNode) Run(ctx agent.InvocationContext, input any) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {}
}

// Workflow manages the workflow graph execution.
type Workflow struct {
	graph *graph

	// name is the per-session-unique identifier under which this
	// workflow's RunState is persisted in session.State. Empty
	// disables persistence. Set at construction by New.
	name string

	// maxConcurrency caps the number of graph-scheduled nodes that
	// may run concurrently within a single Run invocation. 0
	// (the default) means unlimited. Set via WithMaxConcurrency.
	maxConcurrency int
}

// Option configures a Workflow at construction time. Pass options
// as trailing variadic arguments to New.
type Option func(*workflowOptions)

// workflowOptions holds the resolved settings derived from the
// caller's Option list. Zero values mean "no override / use
// engine defaults".
type workflowOptions struct {
	maxConcurrency int
}

// WithMaxConcurrency caps how many graph-scheduled nodes may run
// concurrently in a single Workflow invocation; nodes beyond the
// cap queue as NodePending. n <= 0 disables the cap (unlimited).
//
// Does NOT apply to dynamic sub-nodes invoked via workflow.RunNode
// from inside a DynamicNode body — they are awaited inline by the
// parent and gating them would deadlock.
func WithMaxConcurrency(n int) Option {
	return func(o *workflowOptions) {
		if n < 0 {
			n = 0
		}
		o.maxConcurrency = n
	}
}

// New creates a new Workflow engine with the given name and edges.
//
// The name forms part of the session.State key under which this
// workflow's RunState is persisted (see RunStateSessionKey for
// the exact key shape). It must be unique within any session that
// runs more than one workflow: two workflows sharing a name and a
// session will silently overwrite each other's RunState, leading
// to corrupted resume behaviour. The same workflow may safely
// share a name across different sessions.
//
// An empty name disables persistence: the workflow runs normally
// but its RunState is neither saved nor loaded, so Resume on a
// follow-up turn will find nothing to resume from.
//
// Optional Option values configure engine behaviour
// (concurrency cap, etc.); see WithMaxConcurrency.
func New(name string, edges []Edge, opts ...Option) (*Workflow, error) {
	if err := validateNodes(edges); err != nil {
		return nil, err
	}
	if err := validateSubWorkflowNames(name, edges); err != nil {
		return nil, err
	}
	// TODO(wolo): sanity-check name (reject whitespace-only,
	// reject characters that break the session.State key shape).
	// TODO(wolo): record a graph fingerprint (e.g. sorted node
	// names hash) on the Workflow and verify it against any
	// loaded RunState in Resume; today a name collision or a
	// graph evolution between deploys silently corrupts the
	// resume path.
	graph := newGraph(edges)
	if err := validateWorkflow(graph); err != nil {
		return nil, err
	}
	var o workflowOptions
	for _, opt := range opts {
		opt(&o)
	}
	return &Workflow{
		graph:          graph,
		name:           name,
		maxConcurrency: o.maxConcurrency,
	}, nil
}

// Name returns the workflow's persistence-namespacing name as set
// by New. Empty when the workflow is anonymous (does not persist
// its RunState).
func (w *Workflow) Name() string {
	return w.name
}

// Run drives the workflow to completion or to a graceful pause
// when any node enters NodeWaiting. It returns an iter.Seq2 that
// yields events from per-node goroutines in arrival order; the
// caller may break from the range loop at any point and the
// engine will cancel all in-flight nodes before returning.
//
// The engine model: each scheduled node runs in its own goroutine
// pushing events into a buffered channel. A single consumer
// goroutine (this one) drains the channel, applies state-side
// effects, yields events to the caller, and schedules successors
// when nodes complete. The consumer is the only mutator of the
// per-node lifecycle map and of session state.
func (w *Workflow) Run(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
	return w.RunNode(ctx, userInput(ctx))
}

// RunNode drives the workflow with the given input.
// This is used by WorkflowNode to run nested workflows.
func (w *Workflow) RunNode(ctx agent.InvocationContext, input any) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		s := newScheduler(ctx, w.graph, w.maxConcurrency)
		// Seed: schedule START with the supplied input.
		startState := s.state.EnsureNode(Start.Name())
		startState.Input = input
		s.scheduleNode(Start, input, "", ctx.Branch())

		s.run(yield)

		// All goroutines have returned; ensure no leak.
		s.wg.Wait()
	}
}

// userInput extracts the workflow's seed input from the
// InvocationContext's UserContent. Concatenates all text parts;
// returns nil for an empty UserContent.
func userInput(ctx agent.InvocationContext) any {
	uc := ctx.UserContent()
	if uc == nil {
		return nil
	}
	var sb strings.Builder
	for _, part := range uc.Parts {
		if part.Text != "" {
			sb.WriteString(part.Text)
		}
	}
	return sb.String()
}
