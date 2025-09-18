package services

import (
	"bytes"
	"context"
	"fmt"
	"slices"

	"github.com/goccy/go-graphviz"
	"google.golang.org/adk/agent"
	"google.golang.org/adk/tool"
)

// TODO:
// Missing handling of different agent and tool types.

const (
	DarkGreen  = "#0F5223"
	LightGreen = "#69CB87"
	LightGray  = "#cccccc"
	White      = "#ffffff"
	Background = "#333537"
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
	switch i := instance.(type) {
	case agent.Agent:
		return "🤖 " + i.Name()
	case tool.Tool:
		return "🔧 " + i.Name()
	default:
		return "Unsupported tool type"
	}
}

func nodeShape(instance any) graphviz.Shape {
	switch instance.(type) {
	case agent.Agent:
		return graphviz.EllipseShape
	case tool.Tool:
		return graphviz.BoxShape
	default:
		return graphviz.CylinderShape
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

func getOrCreateNode(graph *graphviz.Graph, name string) (*graphviz.Node, error) {
	node, err := graph.NodeByName(name)
	if err != nil || node == nil {
		node, err = graph.CreateNodeByName(name)
		if err != nil {
			return nil, err
		}
	}
	return node, nil
}

func drawNode(graph *graphviz.Graph, instance any, highlightedPairs [][]string, visitedNodes map[string]bool) error {
	name := nodeName(instance)
	shape := nodeShape(instance)
	caption := nodeCaption(instance)
	highlighted := highlighted(name, highlightedPairs)

	node, err := getOrCreateNode(graph, name)
	if err != nil {
		return err
	}
	node.SetLabel(caption)
	node.SetShape(shape)
	node.SetFontColor(LightGray)

	if highlighted {
		node.SetColor(DarkGreen)
		node.SetStyle(graphviz.FilledNodeStyle)
	} else {
		node.SetColor(LightGray)
		node.SetStyle(graphviz.RoundedNodeStyle)
	}
	visitedNodes[name] = true
	return nil
}

func drawEdge(graph *graphviz.Graph, from, to string, highlightedPairs [][]string) error {
	fromNode, err := getOrCreateNode(graph, from)
	if err != nil {
		return err
	}
	toNode, err := getOrCreateNode(graph, to)
	if err != nil {
		return err
	}
	edgeHighlighted := edgeHighlighted(from, to, highlightedPairs)
	edge, err := graph.CreateEdgeByName(fmt.Sprintf("%s->%s", from, to), fromNode, toNode)
	if err != nil {
		return err
	}
	if edgeHighlighted != nil {
		edge.SetColor(LightGreen)
		if !*edgeHighlighted {
			edge.SetArrowHead(graphviz.NormalArrow).SetDir(graphviz.BackDir)
		} else {
			edge.SetArrowHead(graphviz.NormalArrow)
		}
	} else {
		edge.SetColor(LightGray).SetArrowHead(graphviz.NoneArrow)
	}
	return nil
}

func buildGraph(graph *graphviz.Graph, instance any, highlightedPairs [][]string, visitedNodes map[string]bool) error {
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
	viz, err := graphviz.New(ctx)
	if err != nil {
		return "", err
	}
	defer viz.Close()

	graph, err := viz.Graph(
		graphviz.WithName("AgentGraph"),
		graphviz.WithDirectedType(graphviz.StrictDirected),
	)
	graph.SetRankDir(graphviz.LRRank).SetBackgroundColor(Background)
	if err != nil {
		return "", err
	}
	visitedNodes := map[string]bool{}
	err = buildGraph(graph, agent, highlightedPairs, visitedNodes)
	if err != nil {
		return "", err
	}
	var buf bytes.Buffer
	if err := viz.Render(ctx, graph, "dot", &buf); err != nil {
		return "", err
	}
	return buf.String(), nil
}
