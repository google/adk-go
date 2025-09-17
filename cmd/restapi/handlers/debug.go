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

package handlers

import (
	"fmt"
	"net/http"

	"github.com/goccy/go-graphviz"
	"google.golang.org/adk/agent"
	"google.golang.org/adk/tool"
)

const (
	DarkGreen  = "#0F5223"
	LightGreen = "#69CB87"
	LightGray  = "#cccccc"
	White      = "#ffffff"
)

type Graph struct {
	// Agent agent.Agent
	// HighlightPairs bool
}

func (g Graph) nodeName(instance any) (string, error) {
	switch i := instance.(type) {
	case agent.Agent:
		return i.Name() + "(" + i.Type() + " Agent)", nil
	case tool.Tool:
		return i.Name() + "Tool", nil
	default:
		return "", fmt.Errorf("unknown instance type: %T", instance)
	}
}
func (g Graph) nodeCaption(instance any) string {
	switch i := instance.(type) {
	case agent.Agent:
		return i.Name() + "Agent"
	case tool.Tool:
		return i.Name() + "Tool"
	default:
		return "Unsupported tool type"
	}
}

func (g Graph) nodeShape(instance any) graphviz.Shape {
	switch instance.(type) {
	case agent.Agent:
		return graphviz.EllipseShape
	case tool.Tool:
		return graphviz.BoxShape
	default:
		return graphviz.CylinderShape
	}
}

func (g Graph) agentCluster(instance any) bool {
	switch instance.(type) {
	case agent.Agent:
		return true
	case tool.Tool:
		return false
	default:
		return false
	}
}

func (g Graph) drawCluster(graph graphviz.Graph, instance any, parent agent.Agent) error {
	agent, ok := instance.(agent.Agent)
	if !ok {
		return fmt.Errorf("instance is not an agent")
	}
	switch agent.Type() {
	case "Loop":
		if parent != nil {
			err := g.drawEdge(graph, parent.Name(), agent.Name(), false)
			if err != nil {
				return err
			}
		}
		for _, subAgent := range agent.SubAgents() {
		}
	case "Sequential":
		length := len(agent.SubAgents())

	case "Parallel":

	}

}

func (g Graph) drawNode(graph graphviz.Graph, instance any) error {
	name, err := g.nodeName(instance)
	if err != nil {
		return nil
	}
	shape := g.nodeShape(instance)
	caption := g.nodeCaption(instance)
	asAgentCluster := g.agentCluster(instance)
	if asAgentCluster {
		subg, err := graph.CreateSubGraphByName("cluster_" + name)
		if err != nil {
			return err
		}
	} else {
		node, err := graph.CreateNodeByName(name)
		if err != nil {
			return err
		}
		node.SetColor(DarkGreen)
		node.SetLabel(caption)
		node.SetShape(shape)
		node.SetFontColor(LightGray)
		node.SetFillColor(DarkGreen)
	}
}

func (g Graph) drawEdge(graph graphviz.Graph, from, to string, cluster bool) error {
	fromNode, err := graph.NodeByName(from)
	if err != nil {
		return err
	}
	toNode, err := graph.NodeByName(to)
	if err != nil {
		return err
	}

	edge, err := graph.CreateEdgeByName("", fromNode, toNode)
	if err != nil {
		return err
	}
	edge.SetColor(LightGray)
	if !cluster {
		edge.SetArrowHead(graphviz.NoneArrow)
	}
	return nil
}

func (g Graph) buildGraph(graph graphviz.Graph, agent agent.Agent, parent agent.Agent) error {
	err := g.drawNode(graph, agent)
	if err != nil {
		return err
	}
	for _, subAgent := range agent.SubAgents() {
		err := g.buildGraph(graph, subAgent, agent)
		if err != nil {
			return err
		}
		cluster := g.agentCluster(subAgent)
		if cluster {
			err := g.drawEdge(graph, agent.Name(), subAgent.Name(), cluster)
			if err != nil {
				return err
			}
		}
		// llmagent
	}
	return nil
}

// DebugAPIController is the controller for the Debug API.
type DebugAPIController struct{}

// TraceDict returns the debug information for the session in form of dictionary.
func (*DebugAPIController) TraceDict(rw http.ResponseWriter, req *http.Request) {
	unimplemented(rw, req)
}

// EventGraph returns the debug information for the session and session events in form of graph.
func (*DebugAPIController) EventGraph(rw http.ResponseWriter, req *http.Request) {
	unimplemented(rw, req)
}
