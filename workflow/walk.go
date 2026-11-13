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
	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/internal/toolinternal"
	"google.golang.org/adk/v2/internal/workflowwalk"
)

// init registers walkAgents as internal [workflowwalk.WalkAgents] for the experimental
// app-info endpoint (exp/appinfo) and the runner's Close (package runner),
// through internal/agenttree, and walkToolAgents as
// [workflowwalk.WalkToolAgents] for the runner's Close. The walks read
// unexported node fields, so they have to live in this package, and a
// function variable in an internal package keeps them reachable only from
// this module.
//
// Exporting the walk would make it public API for a feature that is still
// experimental. It would also give callers the agents a running graph uses.
// Those agents are live: SubAgents returns the agent's own slice, not a copy,
// so a caller could change which agents the app hands off to. An example of
// hypothetical exported walk:
//
//	walked := workflow.WalkAgents(edges)
//	walked[0].SubAgents()[0] = other // walked[0] now transfers to other
func init() {
	workflowwalk.WalkAgents = func(edges any) []agent.Agent {
		return walkAgents(edges.([]Edge))
	}
	workflowwalk.WalkToolAgents = func(edges any) []agent.Agent {
		return walkToolAgents(edges.([]Edge))
	}
}

// walkAgents is the walk behind [workflowwalk.WalkAgents].
func walkAgents(edges []Edge) []agent.Agent {
	var agents []agent.Agent
	walkNodes(edges, func(n Node) {
		if n, ok := n.(*AgentNode); ok {
			agents = append(agents, n.agent)
		}
	})
	return agents
}

// walkToolAgents is the walk behind [workflowwalk.WalkToolAgents].
func walkToolAgents(edges []Edge) []agent.Agent {
	var agents []agent.Agent
	walkNodes(edges, func(n Node) {
		if n, ok := n.(*ToolNode); ok {
			if wrapper, ok := n.tool.(toolinternal.AgentWrapper); ok {
				agents = append(agents, wrapper.WrappedAgent())
			}
		}
	})
	return agents
}

// walkNodes calls visit once for every node of edges, and for the nodes of
// nested workflows and those ParallelWorkers wrap.
func walkNodes(edges []Edge, visit func(Node)) {
	seen := make(map[Node]bool)
	var walk func(n Node)
	walkAll := func(edges []Edge) {
		for _, e := range edges {
			walk(e.From)
			walk(e.To)
		}
	}
	walk = func(n Node) {
		if seen[n] {
			return
		}
		seen[n] = true
		visit(n)
		switch n := n.(type) {
		case *WorkflowNode:
			walkAll(n.subWorkflow.graph.allEdges())
		case *ParallelWorker:
			walk(n.wrapped)
		}
	}
	walkAll(edges)
}
