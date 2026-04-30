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

	"google.golang.org/adk/agent"
	"google.golang.org/adk/session"
)

// Config is the construction parameter for New.
type Config struct {
	Name        string
	Description string
	Edges       []Edge

	// MaxConcurrency caps the number of nodes the engine schedules in
	// parallel. 0 (the default) means unbounded; dynamic-node executions
	// never count toward this cap.
	MaxConcurrency int

	InputSchema   Schema
	OutputSchema  Schema
	StateSchema   Schema
	RetryConfig   *RetryConfig
	WaitForOutput bool
	RerunOnResume bool
}

// Workflow is a graph orchestrator. It is itself a Node (so workflows can
// be nested) and an agent.Agent (so a workflow drops into runner.Config).
//
// Phase 2 ships the static-graph variant. Per-node parallelism, dynamic
// nodes, resume, and HITL land in subsequent phases.
type Workflow struct {
	Base
	edges          []Edge
	maxConcurrency int

	// graph is built lazily on first Run. The build also validates the
	// edge set (every node uniquely-named, no edges to / from unknown
	// nodes, every non-START node reachable, etc.).
	graph *workflowGraph
}

// New constructs a Workflow. Validation is eager: New returns an error if
// any node has an invalid name, if there are duplicate node names, or if
// any node referenced by an edge isn't reachable from START.
func New(cfg Config) (*Workflow, error) {
	w := &Workflow{
		edges:          cfg.Edges,
		maxConcurrency: cfg.MaxConcurrency,
	}
	spec := NodeSpec{
		InputSchema:   cfg.InputSchema,
		OutputSchema:  cfg.OutputSchema,
		StateSchema:   cfg.StateSchema,
		RetryConfig:   cfg.RetryConfig,
		WaitForOutput: cfg.WaitForOutput,
		RerunOnResume: cfg.RerunOnResume,
	}
	if err := w.SetMetadata(cfg.Name, cfg.Description, spec); err != nil {
		return nil, err
	}
	g, err := buildGraph(cfg.Edges)
	if err != nil {
		return nil, fmt.Errorf("workflow %q: %w", cfg.Name, err)
	}
	w.graph = g
	return w, nil
}

// Edges returns the workflow's edge set. The returned slice aliases the
// internal storage; do not mutate.
func (w *Workflow) Edges() []Edge { return w.edges }

// Graph returns the compiled graph. Useful for visualization (Phase 9).
func (w *Workflow) Graph() *WorkflowGraph { return (*WorkflowGraph)(w.graph) }

// AsAgent wraps the workflow as an agent.Agent so it drops directly into
// runner.Config.Agent. The agent.Agent interface is sealed (only
// agent.New can produce one), so we go through agent.New here rather than
// implementing the interface on Workflow directly.
//
// The returned Agent's Name and Description come from the Workflow.
func (w *Workflow) AsAgent() (agent.Agent, error) {
	return agent.New(agent.Config{
		Name:        w.Name(),
		Description: w.Description(),
		Run: func(ic agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return w.run(ic, nil, "")
		},
	})
}

// run is the entry point used by both AsAgent (top-level) and RunImpl
// (nested). It forwards to runSequential with the user content as the
// initial input when one is available.
func (w *Workflow) run(ic agent.InvocationContext, input any, parentPath string) iter.Seq2[*session.Event, error] {
	if input == nil {
		if uc := ic.UserContent(); uc != nil {
			input = uc
		}
	}
	return w.runSequential(ic, input, parentPath)
}

// RunImpl is the Node-side hook used when a Workflow is nested inside
// another Workflow. It threads the parent NodeContext into the orchestrator
// so the nested run gets a hierarchical path.
func (w *Workflow) RunImpl(ctx *NodeContext, input any, em EventEmitter) error {
	for ev, err := range w.run(ctx.InvocationContext, input, ctx.NodePath()) {
		if err != nil {
			return err
		}
		if err := em.Event(ev); err != nil {
			return err
		}
	}
	return nil
}

// silence unused-import warning while RunImpl evolves.
var _ = fmt.Sprintf

// WorkflowGraph is the public alias for the compiled graph type. Accessors
// are added alongside the engine implementation.
type WorkflowGraph workflowGraph

// Predecessors returns the nodes that have an edge into target.
func (g *WorkflowGraph) Predecessors(target string) []string {
	return (*workflowGraph)(g).predecessors(target)
}

// Successors returns the nodes that target has outgoing edges to.
func (g *WorkflowGraph) Successors(source string) []string {
	return (*workflowGraph)(g).successors(source)
}

// Nodes returns the names of every node in the graph.
func (g *WorkflowGraph) Nodes() []string {
	return (*workflowGraph)(g).nodeNames()
}

// HasNode reports whether name is in the graph.
func (g *WorkflowGraph) HasNode(name string) bool {
	return (*workflowGraph)(g).has(name)
}
