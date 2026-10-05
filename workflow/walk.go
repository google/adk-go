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

import "google.golang.org/adk/v2/agent"

// WalkAgents returns the agents the nodes of edges run: the agent of each
// [AgentNode], including those inside nested sub-workflows. It shows what a
// graph is built from without running it. The agents a workflow agent reaches through its edges are not among its
// [agent.Agent.SubAgents], so walking sub-agents alone never finds them.
//
// Each node is visited once, so an agent that two nodes run is returned twice.
// The order is not stable: the edges of a nested sub-workflow come back in map
// order. An agent that a node runs from its own code, such as a dynamic node,
// is not returned. The result is nil when no node runs an agent.
func WalkAgents(edges []Edge) []agent.Agent {
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
		}
	}
	visitAll(edges)
	return agents
}
