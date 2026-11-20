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

// Package agenttree finds the agents an agent or a workflow node leads to,
// for code in this module that walks an app's agent tree: the experimental
// app-info endpoint (exp/appinfo) and the runner's Close (package runner).
//
// It sits above workflow and agent/workflowagent, because neither of them can
// import the other's side of the link: workflowagent imports workflow, and
// workflow imports internal/workflowwalk.
package agenttree

import (
	"slices"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/workflowagent"
	agentinternal "google.golang.org/adk/v2/internal/agent"
	"google.golang.org/adk/v2/internal/workflowwalk"
	"google.golang.org/adk/v2/workflow"
)

// Children returns the agents a walk descends into from a: its sub-agents,
// followed by the agents its workflow graph runs. It returns only the
// sub-agents when a is not a workflow agent. The result is a new slice.
//
// A workflow agent keeps the agents of its graph in its edges, not in
// SubAgents, so following SubAgents alone reports nothing for a graph-rooted
// app.
func Children(a agent.Agent) []agent.Agent {
	// TODO: make agent name clashes deterministic. A nested workflow's
	// agents come in map order (workflow's graph.allEdges ranges over its
	// successors map), so which of two agents sharing a name is described can
	// change between requests.
	return slices.Concat(a.SubAgents(), workflowwalk.WalkAgents(workflowEdges(a)))
}

// NodeAgents returns the agents run by root, and by the nodes inside root
// (nested workflows, parallel workers), as a graph with the single edge
// Start -> root would reach them. A root that runs no agent, such as a
// function node, gives nil.
func NodeAgents(root workflow.Node) []agent.Agent {
	return workflowwalk.WalkAgents([]workflow.Edge{{From: workflow.Start, To: root}})
}

// ToolAgents returns the agents behind the agent tools that the ToolNodes of
// a's workflow graph run, nested workflows and ParallelWorkers included. It
// returns nil when a is not a workflow agent.
//
// Children leaves these agents out, because the app-info endpoint does not
// report an agent tool that a ToolNode runs.
func ToolAgents(a agent.Agent) []agent.Agent {
	return workflowwalk.WalkToolAgents(workflowEdges(a))
}

// NodeToolAgents returns the agents behind the agent tools that root, and the
// ToolNodes inside root, run, as a graph with the single edge Start -> root
// would reach them.
func NodeToolAgents(root workflow.Node) []agent.Agent {
	return workflowwalk.WalkToolAgents([]workflow.Edge{{From: workflow.Start, To: root}})
}

// workflowEdges returns the edges of a's workflow graph, or nil when a is not
// a workflow agent.
func workflowEdges(a agent.Agent) []workflow.Edge {
	internalAgent, ok := a.(agentinternal.Agent)
	if !ok {
		return nil
	}
	cfg, ok := agentinternal.Reveal(internalAgent).Config.(workflowagent.Config)
	if !ok {
		return nil
	}
	return cfg.Edges
}
