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
	"slices"

	"github.com/awalterschulze/gographviz"
	"google.golang.org/adk/agent"
	"google.golang.org/adk/tool"
)

// TODO:
// Missing handling of different agent and tool types.

const (
	DarkGreen  = "\"#0F5223\""
	LightGreen = "\"#69CB87\""
	LightGray  = "\"#cccccc\""
	White      = "\"#ffffff\""
	Background = "\"#333537\""
)

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
	case tool.Tool:
		caption = "🔧 " + i.Name()
	default:
		caption = "Unsupported tool type"
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

func edgeHighlighted(from string, to string, higlightedPairs [][]string) *bool {
	highlighted := false
	if len(higlightedPairs) == 0 {
		return nil
	}
	for _, pair := range higlightedPairs {
		if len(pair) == 2 {
			if pair[0] == from && pair[1] == to {
				highlighted = true
				return &highlighted
			}
			if pair[0] == to && pair[1] == from {
				return &highlighted
			}
		}
	}
	return nil
}

func drawNode(graph *gographviz.Graph, instance any, highlightedPairs [][]string, visitedNodes map[string]bool) error {
	name := nodeName(instance)
	shape := nodeShape(instance)
	caption := nodeCaption(instance)
	highlighted := highlighted(name, highlightedPairs)

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
	visitedNodes[name] = true

	return graph.AddNode(graph.Name, name, nodeAttributes)
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

func buildGraph(graph *gographviz.Graph, instance any, highlightedPairs [][]string, visitedNodes map[string]bool) error {
	namedInstance, ok := instance.(namedInstance)
	if !ok {
		return nil
	}
	if visitedNodes[namedInstance.Name()] {
		return nil
	}

	err := drawNode(graph, instance, highlightedPairs, visitedNodes)
	if err != nil {
		return err
	}
	agent, ok := instance.(agent.Agent)
	if !ok {
		return nil
	}
	for _, subAgent := range agent.SubAgents() {
		err = drawEdge(graph, nodeName(agent), nodeName(subAgent), highlightedPairs)
		if err != nil {
			return err
		}
		err = buildGraph(graph, subAgent, highlightedPairs, visitedNodes)
		if err != nil {
			return err
		}
	}
	return nil
}

func GetAgentGraph(ctx context.Context, agent agent.Agent, highlightedPairs [][]string) (string, error) {
	graph := gographviz.NewGraph()
	if err := graph.SetName("AgentGraph"); err != nil {
		return "", err
	}
	if err := graph.SetDir(true); err != nil {
		return "", err
	}
	if err := graph.AddAttr(graph.Name, "rankdir", "LR"); err != nil {
		return "", err
	}
	if err := graph.AddAttr(graph.Name, "bgcolor", Background); err != nil {
		return "", err
	}
	visitedNodes := map[string]bool{}
	err := buildGraph(graph, agent, highlightedPairs, visitedNodes)
	if err != nil {
		return "", err
	}
	return graph.String(), nil
}
