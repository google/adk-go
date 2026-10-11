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

	"google.golang.org/adk/v2/agent"
	agentinternal "google.golang.org/adk/v2/internal/agent"
	llmagentinternal "google.golang.org/adk/v2/internal/llminternal"
	"google.golang.org/adk/v2/tool"
)

const (
	DarkGreen  = "\"#0F5223\""
	LightGreen = "\"#69CB87\""
	LightGray  = "\"#cccccc\""
	White      = "\"#ffffff\""
	Background = "\"#333537\""
)

// Colors for the light theme. The web UI preloads a light and a dark rendering
// of the agent graph and picks one by the page theme, so a graph drawn only in
// dark colors is unreadable for half the users.
const (
	DarkGray        = "\"#3c4043\""
	MidGray         = "\"#5f6368\""
	LightBackground = "\"#ffffff\""
)

// Theme is the palette an agent graph is drawn with.
type Theme struct {
	// Background is the graph canvas color.
	Background string
	// Foreground draws node labels, node borders and unhighlighted edges.
	Foreground string
	// ClusterBorder outlines a workflow-agent cluster.
	ClusterBorder string
}

// DarkTheme is the palette ADK has always drawn with.
var DarkTheme = Theme{Background: Background, Foreground: LightGray, ClusterBorder: White}

// LightTheme is the palette for a light-mode page.
var LightTheme = Theme{Background: LightBackground, Foreground: DarkGray, ClusterBorder: MidGray}

// ThemeFor maps the web UI's dark_mode query parameter to a palette.
func ThemeFor(darkMode bool) Theme {
	if darkMode {
		return DarkTheme
	}
	return LightTheme
}

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

// edgeAnchor is the node an edge should touch. When the member is a cluster,
// Graphviz can only draw to that cluster's border from a node inside it, so
// the returned cluster id is applied as ltail or lhead. A cluster that was
// never added, or that has no node inside it, falls back to the plain name
// with no cluster id.
func edgeAnchor(graph *gographviz.Graph, instance any, fromClusterEnd bool) (nodeID, clusterID string) {
	if !shouldBuildAgentCluster(instance) {
		return nodeName(instance), ""
	}
	clusterID = "cluster_" + nodeName(instance)
	if !graph.IsSubGraph(clusterID) {
		return nodeName(instance), ""
	}
	subs := instance.(agent.Agent).SubAgents()
	if len(subs) == 0 {
		return nodeName(instance), ""
	}
	child := subs[0]
	if fromClusterEnd {
		child = subs[len(subs)-1]
	}
	nodeID, _ = edgeAnchor(graph, child, fromClusterEnd)
	if !graph.IsNode(nodeID) {
		return nodeName(instance), ""
	}
	return nodeID, clusterID
}

func drawClusterEdge(graph *gographviz.Graph, from, to any, highlightedPairs [][]string, theme Theme) error {
	src, ltail := edgeAnchor(graph, from, true)
	dst, lhead := edgeAnchor(graph, to, false)
	// cluster_<name> is a subgraph, not a node. Using it as an endpoint makes
	// Graphviz draw a stray ellipse. Skip the edge when either side was never
	// drawn as a node, which is what an empty cluster leaves us with.
	if !graph.IsNode(src) || !graph.IsNode(dst) {
		return nil
	}
	if err := drawEdge(graph, src, dst, highlightedPairs, theme); err != nil {
		return err
	}
	if ltail == "" && lhead == "" {
		return nil
	}
	if err := graph.AddAttr(graph.Name, "compound", "true"); err != nil {
		return err
	}
	edges := graph.Edges.SrcToDsts[src][dst]
	edge := edges[len(edges)-1]
	if ltail != "" {
		if err := edge.Attrs.Add("ltail", ltail); err != nil {
			return err
		}
	}
	if lhead != "" {
		if err := edge.Attrs.Add("lhead", lhead); err != nil {
			return err
		}
	}
	return nil
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

func drawCluster(parentGraph, cluster *gographviz.Graph, agent agent.Agent, highlightedPairs [][]string, visitedNodes map[string]bool, theme Theme) error {
	agentInternal, ok := agent.(agentinternal.Agent)
	if !ok {
		return nil
	}
	subs := agent.SubAgents()
	// Draw every member before any edge. An edge anchor has to see whether the
	// destination cluster was actually added; a name already visited as a tool
	// is skipped and must not be linked through a node that was never drawn.
	for _, subAgent := range subs {
		err := buildGraph(cluster, parentGraph, subAgent, highlightedPairs, visitedNodes, theme)
		if err != nil {
			return fmt.Errorf("draw cluster: build graph: %w", err)
		}
	}
	switch agentinternal.Reveal(agentInternal).AgentType {
	// Sequential sub-agents should be connected one after another with edges.
	case agentinternal.TypeSequentialAgent:
		for i := range len(subs) - 1 {
			err := drawClusterEdge(parentGraph, subs[i], subs[i+1], highlightedPairs, theme)
			if err != nil {
				return fmt.Errorf("draw cluster: draw edge: %w", err)
			}
		}
	// Loop sub-agents should be connected one after another, and the last one should point to the first.
	case agentinternal.TypeLoopAgent:
		for i := range subs {
			next := subs[0]
			if i+1 < len(subs) {
				next = subs[i+1]
			}
			err := drawClusterEdge(parentGraph, subs[i], next, highlightedPairs, theme)
			if err != nil {
				return fmt.Errorf("draw cluster: draw edge: %w", err)
			}
		}
	}
	// Parallel sub-agents shouldn't be connected, they will be a part of the sub graph.
	return nil
}

func drawNode(graph, parentGraph *gographviz.Graph, instance any, highlightedPairs [][]string, visitedNodes map[string]bool, theme Theme) error {
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
		// A nested cluster is drawn while graph is only a name holder for the
		// parent subgraph. Attach it to parentGraph, which is what gets serialized.
		err = parentGraph.AddSubGraph(graph.Name, cluster.Name, map[string]string{
			"style":     "rounded",
			"color":     theme.ClusterBorder,
			"label":     caption,
			"fontcolor": theme.Foreground,
		})
		if err != nil {
			return fmt.Errorf("add cluster: %w", err)
		}
		return drawCluster(parentGraph, cluster, agent, highlightedPairs, visitedNodes, theme)
	} else {
		nodeAttributes := map[string]string{
			"label":     caption,
			"shape":     shape,
			"fontcolor": theme.Foreground,
		}

		if highlighted {
			nodeAttributes["color"] = DarkGreen
			nodeAttributes["style"] = "filled"
		} else {
			nodeAttributes["color"] = theme.Foreground
			nodeAttributes["style"] = "rounded"
		}
		return parentGraph.AddNode(graph.Name, name, nodeAttributes)
	}
}

func drawEdge(graph *gographviz.Graph, from, to string, highlightedPairs [][]string, theme Theme) error {
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
		edgeAttributes["color"] = theme.Foreground
		edgeAttributes["arrowhead"] = "none"
	}
	return graph.AddEdge(from, to, true, edgeAttributes)
}

func buildGraph(graph, parentGraph *gographviz.Graph, instance any, highlightedPairs [][]string, visitedNodes map[string]bool, theme Theme) error {
	namedInstance, ok := instance.(namedInstance)
	if !ok {
		return nil
	}
	if visitedNodes[namedInstance.Name()] {
		return nil
	}

	err := drawNode(graph, parentGraph, instance, highlightedPairs, visitedNodes, theme)
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
			err = drawNode(graph, parentGraph, tool, highlightedPairs, visitedNodes, theme)
			if err != nil {
				return fmt.Errorf("draw tool node: %w", err)
			}
			err = drawEdge(graph, nodeName(agent), nodeName(tool), highlightedPairs, theme)
			if err != nil {
				return fmt.Errorf("draw tool edge: %w", err)
			}
		}
	}
	for _, subAgent := range agent.SubAgents() {
		err = buildGraph(graph, parentGraph, subAgent, highlightedPairs, visitedNodes, theme)
		if err != nil {
			return fmt.Errorf("build sub agent graph: %w", err)
		}
	}
	return nil
}

// GetAgentGraph renders the agent tree as Graphviz DOT source in the dark
// theme. It is kept for callers that do not care about the palette.
func GetAgentGraph(ctx context.Context, agent agent.Agent, highlightedPairs [][]string) (string, error) {
	return GetAgentGraphWithTheme(ctx, agent, highlightedPairs, DarkTheme)
}

// GetAgentGraphWithTheme renders the agent tree as Graphviz DOT source using
// the given palette.
func GetAgentGraphWithTheme(ctx context.Context, agent agent.Agent, highlightedPairs [][]string, theme Theme) (string, error) {
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
	if err := graph.AddAttr(graph.Name, "bgcolor", theme.Background); err != nil {
		return "", fmt.Errorf("set graph background color: %w", err)
	}
	visitedNodes := map[string]bool{}
	err := buildGraph(graph, graph, agent, highlightedPairs, visitedNodes, theme)
	if err != nil {
		return "", fmt.Errorf("build root graph: %w", err)
	}
	return graph.String(), nil
}
