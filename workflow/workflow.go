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
//
// Custom nodes typically embed BaseNode (constructed via NewBaseNode)
// to inherit Name, Description, and Config implementations, and
// supply only Run.
type Node interface {
	Name() string
	Description() string
	Config() NodeConfig
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

func (s *startNode) Name() string        { return "START" }
func (s *startNode) Description() string { return "Start node" }
func (s *startNode) Config() NodeConfig  { return NodeConfig{} }
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
func New(name string, edges []Edge) (*Workflow, error) {
	if err := validateNodes(edges); err != nil {
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
	return &Workflow{graph: graph, name: name}, nil
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
	return func(yield func(*session.Event, error) bool) {
		input := userInput(ctx)

		s := newScheduler(ctx, w.graph)
		// Seed: schedule START with the user-supplied input.
		startState := s.state.EnsureNode(Start.Name())
		startState.Input = input
		s.scheduleNode(Start, input, "")

		s.run(yield)

		// All goroutines have returned; ensure no leak.
		s.wg.Wait()

		// Persist the run state so a follow-up turn can call
		// Workflow.Resume with the recovered NodeWaiting set.
		// Emitted as a session.Event with StateDelta so the
		// surrounding event-append pipeline can propagate it to
		// storage; see NewRunStateEvent for why a direct
		// State.Set is not sufficient.
		yieldRunStateEvent(ctx, w.name, s.state, yield)
	}
}

// yieldRunStateEvent emits a session.Event carrying the workflow's
// serialised RunState in Actions.StateDelta. No-op when the
// workflow is anonymous (no name → no persistence) or when the
// caller has stopped consuming the iterator. See NewRunStateEvent
// for why the state must be delivered as an event rather than via
// a direct State.Set call.
func yieldRunStateEvent(
	ctx agent.InvocationContext,
	workflowName string,
	state *RunState,
	yield func(*session.Event, error) bool,
) {
	ev, err := NewRunStateEvent(ctx.InvocationID(), workflowName, state)
	if err != nil {
		yield(nil, err)
		return
	}
	if ev == nil {
		return
	}
	yield(ev, nil)
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
