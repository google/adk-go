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
	"google.golang.org/adk/v2/internal/workflowwalk"
)

// init registers walkAgents as internal [workflowwalk.WalkAgents] for the experimental
// app-info endpoint (exp/appinfo). The walk reads unexported node fields, so it
// has to live in this package, and a function variable in an internal package
// keeps it reachable only from this module.
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
}

// walkAgents is the walk behind [workflowwalk.WalkAgents].
func walkAgents(edges []Edge) []agent.Agent {
	var agents []agent.Agent
	seen := make(map[Node]bool)
	var visit func(n Node)
	visitAll := func(edges []Edge) {
		for _, e := range edges {
			visit(e.From)
			visit(e.To)
		}
	}
	visit = func(n Node) {
		if seen[n] {
			return
		}
		seen[n] = true
		switch n := n.(type) {
		case *AgentNode:
			agents = append(agents, n.agent)
		case *WorkflowNode:
			visitAll(n.subWorkflow.graph.allEdges())
		case *ParallelWorker:
			visit(n.wrapped)
		}
	}
	visitAll(edges)
	return agents
}
