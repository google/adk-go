// Copyright 2025 Google LLC
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

package services

import (
	"context"
	"fmt"
	"slices"

	"github.com/awalterschulze/gographviz"

	"google.golang.org/adk/agent"
	agentinternal "google.golang.org/adk/internal/agent"
	llmagentinternal "google.golang.org/adk/internal/llminternal"
	"google.golang.org/adk/tool"
)

const (
	DarkGreen  = "\"#0F5223\""
	LightGreen = "\"#69CB87\""
	LightGray  = "\"#cccccc\""
	White      = "\"#ffffff\""
	Background = "\"#333537\""
)

var supportedClusterAgents = []agentinternal.Type{
	agentinternal.TypeLoopAgent,
	agentinternal.TypeSequentialAgent,
	agentinternal.TypeParallelAgent,
}

type namedInstance interface {
	Name() string
}

func nodeName(instance any) string {
	switch i := instance.(type) {
	case agent.Agent:
		return i.Name()
	case tool.Tool:
		return i.Name()
	default:
		return "Unknown instance type"
	}
}

func nodeCaption(instance any) string {
	caption := ""
	switch i := instance.(type) {
	case agent.Agent:
		caption = "🤖 " + i.Name()
		typedAgent, ok := i.(agentinternal.Agent)
		if ok {
			if slices.Contains(supportedClusterAgents, agentinternal.Reveal(typedAgent).AgentType) {
				caption = i.Name() + " (" + string(agentinternal.Reveal(typedAgent).AgentType) + ")"
			}
		}
	case tool.Tool:
		caption = "🔧 " + i.Name()
	default:
		caption = "Unsupported agent or tool type"
	}
	return "\"" + caption + "\""
}

func nodeShape(instance any) string {
	switch instance.(type) {
	case agent.Agent:
		return "ellipse"
	case tool.Tool:
		return "box"
	default:
		return "cylinder"
	}
}

func shouldBuildAgentCluster(instance any) bool {
	switch i := instance.(type) {
	case agent.Agent:
		agent, ok := i.(agentinternal.Agent)
		if !ok {
			return false
		}
		return slices.Contains(supportedClusterAgents, agentinternal.Reveal(agent).AgentType)
	default:
		return false
	}
}

func highlighted(nodeName string, higlightedPairs [][]string) bool {
	if len(higlightedPairs) == 0 {
		return false
	}
	for _, pair := range higlightedPairs {
		if slices.Contains(pair, nodeName) {
			return true
		}
	}
	return false
}

func boolPtr(b bool) *bool {
	return &b
}

// Function returns whether the edge should be highlighted.
// The graph could have the pairs highlighted in different directions.
// If nil is returned, means the nodes aren't highlithed.
// Otherwise, pointer to bool type is returned, where true
// means the directed connection between nodes, while false means
// there is a reversed order between nodes.
func edgeHighlighted(from, to string, higlightedPairs [][]string) *bool {
	if len(higlightedPairs) == 0 {
		return nil
	}
	for _, pair := range higlightedPairs {
		if len(pair) == 2 {
			if pair[0] == from && pair[1] == to {
				return boolPtr(true)
			}
			if pair[0] == to && pair[1] == from {
				return boolPtr(false)
			}
		}
	}
	return nil
}

func drawCluster(parentGraph, cluster *gographviz.Graph, agent agent.Agent, highlightedPairs [][]string, visitedNodes map[string]bool) error {
	agentInternal, ok := agent.(agentinternal.Agent)
	if !ok {
		return nil
	}
	for i, subAgent := range agent.SubAgents() {
		err := buildGraph(cluster, parentGraph, subAgent, highlightedPairs, visitedNodes)
		if err != nil {
			return fmt.Errorf("draw cluster: build graph: %w", err)
		}
		switch agentinternal.Reveal(agentInternal).AgentType {
		// Sequential sub-agents should be connected one after another with edges.
		case agentinternal.TypeSequentialAgent:
			if i < len(agent.SubAgents())-1 {
				err = drawEdge(parentGraph, nodeName(subAgent), nodeName(agent.SubAgents()[i+1]), highlightedPairs)
				if err != nil {
					return fmt.Errorf("draw cluster: draw edge: %w", err)
				}
			}
		// Sequential sub-agents should be connected one after another with edges, but the last one should point to the first agent.
		case agentinternal.TypeLoopAgent:
			nextAgentIdx := i + 1
			if nextAgentIdx >= len(agent.SubAgents()) {
				nextAgentIdx = 0
			}
			err = drawEdge(parentGraph, nodeName(subAgent), nodeName(agent.SubAgents()[nextAgentIdx]), highlightedPairs)
			if err != nil {
				return fmt.Errorf("draw cluster: draw edge: %w", err)
			}
		}
		// Parallel sub-agents shouldn't be connected, they will be a part of the sub graph.
	}
	return nil
}

func drawNode(graph, parentGraph *gographviz.Graph, instance any, highlightedPairs [][]string, visitedNodes map[string]bool) error {
	name := nodeName(instance)
	shape := nodeShape(instance)
	caption := nodeCaption(instance)
	highlighted := highlighted(name, highlightedPairs)
	isCluster := shouldBuildAgentCluster(instance)

	visitedNodes[name] = true
	if isCluster {
		agent, ok := instance.(agent.Agent)
		if !ok {
			return nil
		}
		cluster := gographviz.NewGraph()
		err := cluster.SetName("cluster_" + name)
		if err != nil {
			return fmt.Errorf("set cluster name: %w", err)
		}
		err = graph.AddSubGraph(graph.Name, cluster.Name, map[string]string{
			"style":     "rounded",
			"color":     White,
			"label":     caption,
			"fontcolor": LightGray,
		})
		if err != nil {
			return fmt.Errorf("add cluster: %w", err)
		}
		return drawCluster(graph, cluster, agent, highlightedPairs, visitedNodes)
	} else {
		nodeAttributes := map[string]string{
			"label":     caption,
			"shape":     shape,
			"fontcolor": LightGray,
		}

		if highlighted {
			nodeAttributes["color"] = DarkGreen
			nodeAttributes["style"] = "filled"
		} else {
			nodeAttributes["color"] = LightGray
			nodeAttributes["style"] = "rounded"
		}
		return parentGraph.AddNode(graph.Name, name, nodeAttributes)
	}
}

func drawEdge(graph *gographviz.Graph, from, to string, highlightedPairs [][]string) error {
	edgeHighlighted := edgeHighlighted(from, to, highlightedPairs)
	edgeAttributes := map[string]string{}
	if edgeHighlighted != nil {
		edgeAttributes["color"] = LightGreen
		if !*edgeHighlighted {
			edgeAttributes["arrowhead"] = "normal"
			edgeAttributes["dir"] = "back"
		} else {
			edgeAttributes["arrowhead"] = "normal"
		}
	} else {
		edgeAttributes["color"] = LightGray
		edgeAttributes["arrowhead"] = "none"
	}
	return graph.AddEdge(from, to, true, edgeAttributes)
}

func buildGraph(graph, parentGraph *gographviz.Graph, instance any, highlightedPairs [][]string, visitedNodes map[string]bool) error {
	namedInstance, ok := instance.(namedInstance)
	if !ok {
		return nil
	}
	if visitedNodes[namedInstance.Name()] {
		return nil
	}

	err := drawNode(graph, parentGraph, instance, highlightedPairs, visitedNodes)
	if err != nil {
		return fmt.Errorf("draw node: %w", err)
	}
	agent, ok := instance.(agent.Agent)
	if !ok {
		return nil
	}
	llmAgent, ok := instance.(llmagentinternal.Agent)
	if ok {
		tools := llmagentinternal.Reveal(llmAgent).Tools
		for _, tool := range tools {
			err = drawNode(graph, parentGraph, tool, highlightedPairs, visitedNodes)
			if err != nil {
				return fmt.Errorf("draw tool node: %w", err)
			}
			err = drawEdge(graph, nodeName(agent), nodeName(tool), highlightedPairs)
			if err != nil {
				return fmt.Errorf("draw tool edge: %w", err)
			}
		}
	}
	for _, subAgent := range agent.SubAgents() {
		err = buildGraph(graph, parentGraph, subAgent, highlightedPairs, visitedNodes)
		if err != nil {
			return fmt.Errorf("build sub agent graph: %w", err)
		}
	}
	return nil
}

func GetAgentGraph(ctx context.Context, agent agent.Agent, highlightedPairs [][]string) (string, error) {
	graph := gographviz.NewGraph()
	if err := graph.SetName("AgentGraph"); err != nil {
		return "", fmt.Errorf("set graph name: %w", err)
	}
	if err := graph.SetDir(true); err != nil {
		return "", fmt.Errorf("set graph direction: %w", err)
	}
	if err := graph.AddAttr(graph.Name, "rankdir", "LR"); err != nil {
		return "", fmt.Errorf("set graph rank direction: %w", err)
	}
	if err := graph.AddAttr(graph.Name, "bgcolor", Background); err != nil {
		return "", fmt.Errorf("set graph background color: %w", err)
	}
	visitedNodes := map[string]bool{}
	err := buildGraph(graph, graph, agent, highlightedPairs, visitedNodes)
	if err != nil {
		return "", fmt.Errorf("build root graph: %w", err)
	}
	return graph.String(), nil
}
