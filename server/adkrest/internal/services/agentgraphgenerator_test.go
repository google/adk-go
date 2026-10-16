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
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"iter"
	"log"
	"log/slog"
	"strings"
	"testing"
	"time"

	"github.com/awalterschulze/gographviz"
	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/agent/workflowagents/loopagent"
	"google.golang.org/adk/v2/agent/workflowagents/parallelagent"
	"google.golang.org/adk/v2/agent/workflowagents/sequentialagent"
	agentinternal "google.golang.org/adk/v2/internal/agent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/agenttool"
	"google.golang.org/adk/v2/tool/functiontool"
)

type dummyLLM struct {
	name string
}

func (d *dummyLLM) Name() string {
	return d.name
}

func (d *dummyLLM) GenerateContent(ctx context.Context, req *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		yield(&model.LLMResponse{
			Content: &genai.Content{
				Parts: []*genai.Part{{Text: "Response from agentgrapgenerator test."}},
			},
		}, nil)
	}
}

// Helper to create a generic agent.Agent
func newTestAgent(t *testing.T, name, description string, agentType agentinternal.Type, subAgents []agent.Agent, tools []tool.Tool) agent.Agent {
	var a agent.Agent
	var err error

	switch agentType {
	case agentinternal.TypeSequentialAgent:
		a, err = sequentialagent.New(sequentialagent.Config{
			AgentConfig: agent.Config{
				Name:        name,
				Description: description,
				SubAgents:   subAgents,
			},
		})
	case agentinternal.TypeLoopAgent:
		a, err = loopagent.New(loopagent.Config{
			AgentConfig: agent.Config{
				Name:        name,
				Description: description,
				SubAgents:   subAgents,
			},
			MaxIterations: 1,
		})
	case agentinternal.TypeParallelAgent:
		a, err = parallelagent.New(parallelagent.Config{
			AgentConfig: agent.Config{
				Name:        name,
				Description: description,
				SubAgents:   subAgents,
			},
		})
	case agentinternal.TypeCustomAgent, agentinternal.TypeLLMAgent:
		a, err = llmagent.New(llmagent.Config{
			Name:        name,
			Description: description,
			Model:       &dummyLLM{},
			Tools:       tools,
			SubAgents:   subAgents,
		})
	default:
		t.Fatalf("Unsupported agent type: %v", agentType)
	}

	if err != nil {
		t.Fatalf("failed to create agent: %v", err)
	}
	return a
}

// Mock tool for testing
type mockTool struct {
	name string
}

func (m *mockTool) Name() string { return m.name }

func (m *mockTool) Description() string { return "" }

func (m *mockTool) IsLongRunning() bool { return false }

func TestNodeName(t *testing.T) {
	tests := []struct {
		name     string
		instance any
		expected string
	}{
		{
			name:     "agent",
			instance: newTestAgent(t, "TestAgent", "", agentinternal.TypeCustomAgent, nil, nil),
			expected: "TestAgent",
		},
		{
			name:     "tool",
			instance: &mockTool{name: "TestTool"},
			expected: "TestTool",
		},
		{
			name:     "unknown",
			instance: "some string",
			expected: "Unknown instance type",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := nodeName(tt.instance); got != tt.expected {
				t.Errorf("nodeName(%v) = %s; want %s", tt.instance, got, tt.expected)
			}
		})
	}
}

func TestNodeCaption(t *testing.T) {
	tests := []struct {
		name     string
		instance any
		expected string
	}{
		{
			name:     "llm agent",
			instance: newTestAgent(t, "LLMAgent", "", agentinternal.TypeLLMAgent, nil, nil),
			expected: "\"🤖 LLMAgent\"",
		},
		{
			name:     "sequential agent",
			instance: newTestAgent(t, "SeqAgent", "", agentinternal.TypeSequentialAgent, nil, nil),
			expected: "\"SeqAgent (SequentialAgent)\"",
		},
		{
			name:     "loop agent",
			instance: newTestAgent(t, "LoopAgent", "", agentinternal.TypeLoopAgent, nil, nil),
			expected: "\"LoopAgent (LoopAgent)\"",
		},
		{
			name:     "parallel agent",
			instance: newTestAgent(t, "ParAgent", "", agentinternal.TypeParallelAgent, nil, nil),
			expected: "\"ParAgent (ParallelAgent)\"",
		},
		{
			name:     "tool",
			instance: &mockTool{name: "TestTool"},
			expected: "\"🔧 TestTool\"",
		},
		{
			name:     "unknown",
			instance: "some string",
			expected: "\"Unsupported agent or tool type\"",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := nodeCaption(tt.instance); got != tt.expected {
				t.Errorf("nodeCaption(%v) = %s; want %s", tt.instance, got, tt.expected)
			}
		})
	}
}

func TestNodeShape(t *testing.T) {
	tests := []struct {
		name     string
		instance any
		expected string
	}{
		{
			name:     "agent",
			instance: newTestAgent(t, "TestAgent", "", agentinternal.TypeCustomAgent, nil, nil),
			expected: "ellipse",
		},
		{
			name:     "tool",
			instance: &mockTool{name: "TestTool"},
			expected: "box",
		},
		{
			name:     "unknown",
			instance: "some string",
			expected: "cylinder",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := nodeShape(tt.instance); got != tt.expected {
				t.Errorf("nodeShape(%v) = %s; want %s", tt.instance, got, tt.expected)
			}
		})
	}
}

func TestShouldBuildAgentCluster(t *testing.T) {
	tests := []struct {
		name     string
		instance any
		expected bool
	}{
		{
			name:     "llm agent",
			instance: newTestAgent(t, "LLMAgent", "", agentinternal.TypeLLMAgent, nil, nil),
			expected: false,
		},
		{
			name:     "sequential agent",
			instance: newTestAgent(t, "SeqAgent", "", agentinternal.TypeSequentialAgent, nil, nil),
			expected: true,
		},
		{
			name:     "loop agent",
			instance: newTestAgent(t, "LoopAgent", "", agentinternal.TypeLoopAgent, nil, nil),
			expected: true,
		},
		{
			name:     "parallel agent",
			instance: newTestAgent(t, "ParAgent", "", agentinternal.TypeParallelAgent, nil, nil),
			expected: true,
		},
		{
			name:     "tool",
			instance: &mockTool{name: "TestTool"},
			expected: false,
		},
		{
			name:     "unknown",
			instance: "some string",
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := shouldBuildAgentCluster(tt.instance); got != tt.expected {
				t.Errorf("shouldBuildAgentCluster(%v) = %t; want %t", tt.instance, got, tt.expected)
			}
		})
	}
}

func TestHighlighted(t *testing.T) {
	tests := []struct {
		name             string
		nodeName         string
		highlightedPairs [][]string
		expected         bool
	}{
		{
			name:     "no highlight",
			nodeName: "NodeA", highlightedPairs: [][]string{},
			expected: false,
		},
		{
			name:     "node in pair",
			nodeName: "NodeA", highlightedPairs: [][]string{{"NodeA", "NodeB"}},
			expected: true,
		},
		{
			name:             "node not in pair",
			nodeName:         "NodeC",
			highlightedPairs: [][]string{{"NodeA", "NodeB"}},
			expected:         false,
		},
		{
			name:             "multiple pairs",
			nodeName:         "NodeB",
			highlightedPairs: [][]string{{"NodeA", "NodeB"}, {"NodeC", "NodeD"}},
			expected:         true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := highlighted(tt.nodeName, tt.highlightedPairs); got != tt.expected {
				t.Errorf("highlighted(%s, %v) = %t; want %t", tt.nodeName, tt.highlightedPairs, got, tt.expected)
			}
		})
	}
}

func TestEdgeHighlighted(t *testing.T) {
	tests := []struct {
		name             string
		from             string
		to               string
		highlightedPairs [][]string
		expected         *bool // Use pointer to distinguish nil from false
	}{
		{
			name:             "no highlight pairs",
			from:             "A",
			to:               "B",
			highlightedPairs: [][]string{},
			expected:         nil,
		},
		{
			name:             "matching forward",
			from:             "A",
			to:               "B",
			highlightedPairs: [][]string{{"A", "B"}},
			expected:         boolPtr(true),
		},
		{
			name:             "matching backward",
			from:             "B",
			to:               "A",
			highlightedPairs: [][]string{{"A", "B"}},
			expected:         boolPtr(false),
		},
		{
			name: "no match",
			from: "C",
			to:   "D", highlightedPairs: [][]string{{"A", "B"}},
			expected: nil,
		},
		{
			name:             "partial match",
			from:             "A",
			to:               "C",
			highlightedPairs: [][]string{{"A", "B"}},
			expected:         nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := edgeHighlighted(tt.from, tt.to, tt.highlightedPairs)
			if (got == nil && tt.expected != nil) || (got != nil && tt.expected == nil) {
				t.Errorf("edgeHighlighted(%s, %s, %v) = %v; want %v", tt.from, tt.to, tt.highlightedPairs, got, tt.expected)
			} else if got != nil && tt.expected != nil && *got != *tt.expected {
				t.Errorf("edgeHighlighted(%s, %s, %v) = %t; want %t", tt.from, tt.to, tt.highlightedPairs, *got, *tt.expected)
			}
		})
	}
}

func TestDrawNode(t *testing.T) {
	tests := []struct {
		name             string
		agent            agent.Agent
		tool             tool.Tool
		highlightedPairs [][]string
		expected         gographviz.Attrs
	}{
		{
			name:             "draw agent node",
			agent:            newTestAgent(t, "MyAgent", "", agentinternal.TypeCustomAgent, nil, nil),
			highlightedPairs: [][]string{},
			expected: gographviz.Attrs{
				"color":     LightGray,
				"label":     "\"🤖 MyAgent\"",
				"shape":     "ellipse",
				"fontcolor": LightGray,
				"style":     "rounded",
			},
		},
		{
			name:             "draw agent node highlighted",
			agent:            newTestAgent(t, "HighlightedAgent", "", agentinternal.TypeCustomAgent, nil, nil),
			highlightedPairs: [][]string{{"HighlightedAgent", "Tool1"}},
			expected: gographviz.Attrs{
				"color":     DarkGreen,
				"label":     "\"🤖 HighlightedAgent\"",
				"shape":     "ellipse",
				"fontcolor": LightGray,
				"style":     "filled",
			},
		},
		{
			name:             "draw tool node",
			tool:             &mockTool{name: "MyTool"},
			highlightedPairs: [][]string{},
			expected: gographviz.Attrs{
				"color":     LightGray,
				"label":     "\"🔧 MyTool\"",
				"shape":     "box",
				"fontcolor": LightGray,
				"style":     "rounded",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			graph := gographviz.NewGraph()
			err := graph.SetName("G")
			if err != nil {
				t.Fatalf("failed to set graph name: %v", err)
			}
			parentGraph := graph
			visitedNodes := make(map[string]bool)
			nodeName := ""
			if tt.agent != nil {
				err = drawNode(context.Background(), graph, parentGraph, tt.agent, tt.highlightedPairs, visitedNodes, DarkTheme)
				if err != nil {
					t.Fatalf("drawNode failed: %v", err)
				}
				nodeName = tt.agent.Name()
			}
			if tt.tool != nil {
				err = drawNode(context.Background(), graph, parentGraph, tt.tool, tt.highlightedPairs, visitedNodes, DarkTheme)
				if err != nil {
					t.Fatalf("drawNode failed: %v", err)
				}
				nodeName = tt.tool.Name()
			}
			if nodeName == "" {
				t.Fatalf("No node name found: %v", nodeName)
			}
			node := graph.Nodes.Lookup[nodeName]
			if node == nil {
				t.Fatal("Agent node not found in graph")
				// to prevent SA5011: possible nil pointer dereference (staticcheck)
				return
			}
			if diff := cmp.Diff(tt.expected, node.Attrs); diff != "" {
				t.Fatalf("drawNode mismatch (-want +got):\n%s", diff)
			}
			if !visitedNodes[nodeName] {
				t.Error("Agent node not marked as visited")
			}
		})
	}
}

func TestDrawClusterNode(t *testing.T) {
	graph := gographviz.NewGraph()
	err := graph.SetName("G")
	if err != nil {
		t.Fatalf("failed to set graph name: %v", err)
	}
	parentGraph := graph
	visitedNodes := make(map[string]bool)
	agent := newTestAgent(t, "MyClusterAgent", "", agentinternal.TypeSequentialAgent, nil, nil)
	err = drawNode(context.Background(), graph, parentGraph, agent, [][]string{}, visitedNodes, DarkTheme)
	if err != nil {
		t.Fatalf("drawNode failed: %v", err)
	}
	clusterName := "cluster_MyClusterAgent"
	cluster := graph.SubGraphs.SubGraphs[clusterName]
	if cluster == nil {
		t.Fatal("Cluster not found in graph")
		// to prevent SA5011: possible nil pointer dereference (staticcheck)
		return
	}
	if cluster.Attrs["label"] != "\"MyClusterAgent (SequentialAgent)\"" {
		t.Errorf("Cluster label mismatch: got %s", cluster.Attrs["label"])
	}
	if cluster.Attrs["style"] != "rounded" {
		t.Errorf("Cluster style mismatch: got %s", cluster.Attrs["style"])
	}
	if !visitedNodes["MyClusterAgent"] {
		t.Error("Cluster agent not marked as visited")
	}
}

func lookupEdge(t *testing.T, graph *gographviz.Graph, src, dst string) *gographviz.Edge {
	node := graph.Edges.SrcToDsts[src]
	if node == nil {
		return nil
	}
	edges := node[dst]
	if edges == nil {
		return nil
	}
	if len(edges) != 1 {
		t.Fatalf("Expected 1 edge, got %d", len(edges))
	}
	return edges[0]
}

func TestDrawEdge(t *testing.T) {
	tests := []struct {
		name             string
		from             string
		to               string
		highlightedPairs [][]string
		expected         gographviz.Attrs
	}{
		{
			name:             "draw unhighlighted edge",
			from:             "NodeA",
			to:               "NodeB",
			highlightedPairs: [][]string{},
			expected: gographviz.Attrs{
				"color":     LightGray,
				"arrowhead": "none",
			},
		},
		{
			name:             "draw highlighted edge",
			from:             "NodeC",
			to:               "NodeD",
			highlightedPairs: [][]string{{"NodeC", "NodeD"}},
			expected: gographviz.Attrs{
				"color":     LightGreen,
				"arrowhead": "normal",
			},
		},
		{
			name:             "draw highlighted backward edge",
			from:             "NodeE",
			to:               "NodeF",
			highlightedPairs: [][]string{{"NodeF", "NodeE"}},
			expected: gographviz.Attrs{
				"color":     LightGreen,
				"arrowhead": "normal",
				"dir":       "back",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			graph := gographviz.NewGraph()
			err := graph.SetName("G")
			if err != nil {
				t.Fatalf("failed to set graph name: %v", err)
			}

			for _, node := range []string{tt.from, tt.to} {
				err := graph.AddNode("G", node, nil)
				if err != nil {
					t.Fatalf("failed to add node %s: %v", node, err)
				}
			}

			err = drawEdge(graph, tt.from, tt.to, tt.highlightedPairs, DarkTheme)
			if err != nil {
				t.Fatalf("drawEdge failed: %v", err)
			}
			edge := lookupEdge(t, graph, tt.from, tt.to)
			if edge == nil {
				t.Fatalf("Edge between %v and %v not found", tt.from, tt.to)
				// to prevent SA5011: possible nil pointer dereference (staticcheck)
				return
			}

			if diff := cmp.Diff(tt.expected, edge.Attrs); diff != "" {
				t.Fatalf("drawEdge mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestDrawCluster(t *testing.T) {
	tests := []struct {
		name      string
		agentType agentinternal.Type
	}{
		{
			name:      "sequential agent cluster",
			agentType: agentinternal.TypeSequentialAgent,
		},
		{
			name:      "parallel agent cluster",
			agentType: agentinternal.TypeParallelAgent,
		},
		{
			name:      "loop agent cluster",
			agentType: agentinternal.TypeLoopAgent,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			parentGraph := gographviz.NewGraph()
			err := parentGraph.SetName("ParentG")
			if err != nil {
				t.Fatalf("failed to set parent graph name: %v", err)
			}

			visitedNodes := make(map[string]bool)
			subAgent1 := newTestAgent(t, "SubAgent1", "", agentinternal.TypeLLMAgent, nil, nil)
			subAgent2 := newTestAgent(t, "SubAgent2", "", agentinternal.TypeLLMAgent, nil, nil)
			parentAgent := newTestAgent(t, "ParentAgent", "", tt.agentType, []agent.Agent{subAgent1, subAgent2}, nil)

			clusterGraph := gographviz.NewGraph()
			err = drawCluster(context.Background(), parentGraph, clusterGraph, parentAgent, [][]string{}, visitedNodes, DarkTheme)
			if err != nil {
				t.Fatalf("drawCluster failed: %v", err)
			}

			if parentGraph.Nodes.Lookup["SubAgent1"] == nil || parentGraph.Nodes.Lookup["SubAgent2"] == nil {
				t.Error("Sub-agents not drawn as nodes in parent graph")
			}

			switch tt.agentType {
			case agentinternal.TypeSequentialAgent:
				// Check if sub-agents are drawn as nodes in the parent graph (since drawNode adds to parentGraph)
				edge := lookupEdge(t, parentGraph, "SubAgent1", "SubAgent2")
				// Check if edge exists between sub-agents
				if edge == nil {
					t.Fatalf("Edge between SubAgent1 and SubAgent2 not found")
					// to prevent SA5011: possible nil pointer dereference (staticcheck)
					return
				}
				if edge.Attrs["arrowhead"] != "none" {
					t.Errorf("Sequential agent edge arrowhead mismatch: got %s", edge.Attrs["arrowhead"])
				}
			case agentinternal.TypeParallelAgent:
				// Check that no edges exist between parallel sub-agents
				if lookupEdge(t, parentGraph, "SubAgent1", "SubAgent2") != nil || lookupEdge(t, parentGraph, "ParSubAgent2", "ParSubAgent1") != nil {
					t.Error("Unexpected edge found between parallel sub-agents")
				}
			case agentinternal.TypeLoopAgent:
				// Check if edges exist between sub-agents and back to the first
				if lookupEdge(t, parentGraph, "SubAgent1", "SubAgent2") == nil {
					t.Error("Edge between SubAgent1 and SubAgent2 not found")
				}
				if lookupEdge(t, parentGraph, "SubAgent1", "SubAgent2") == nil {
					t.Error("Edge between SubAgent1 and LoopSubAgent1 not found")
				}
			default:
				t.Fatalf("Wrong agent type provided: %v", tt.agentType)
			}
		})
	}
}

func TestBuildGraph(t *testing.T) {
	graph := gographviz.NewGraph()
	err := graph.SetName("G")
	if err != nil {
		t.Fatalf("failed to set parent graph name: %v", err)
	}
	parentGraph := graph
	visitedNodes := make(map[string]bool)

	tool1 := &mockTool{name: "Tool1"}
	tool2 := &mockTool{name: "Tool2"}

	subAgent1 := newTestAgent(t, "SubAgent1", "", agentinternal.TypeLLMAgent, nil, []tool.Tool{tool1})
	subAgent2 := newTestAgent(t, "SubAgent2", "", agentinternal.TypeLLMAgent, nil, nil)
	mainAgent := newTestAgent(t, "MainAgent", "", agentinternal.TypeLLMAgent, []agent.Agent{subAgent1, subAgent2}, []tool.Tool{tool2})

	err = buildGraph(context.Background(), graph, parentGraph, mainAgent, [][]string{}, visitedNodes, DarkTheme)
	if err != nil {
		t.Fatalf("buildGraph failed: %v", err)
	}

	// Check if all nodes are present
	expectedNodes := []string{"MainAgent", "SubAgent1", "SubAgent2", "Tool1", "Tool2"}
	for _, nodeName := range expectedNodes {
		if graph.Nodes.Lookup[nodeName] == nil {
			t.Errorf("Node %s not found in graph", nodeName)
		}
		if !visitedNodes[nodeName] {
			t.Errorf("Node %s not marked as visited", nodeName)
		}
	}

	// Check edges from MainAgent to its tools
	if lookupEdge(t, graph, "MainAgent", "Tool2") == nil {
		t.Error("Edge from MainAgent to Tool2 not found")
	}

	// // Check edges from SubAgent1 to its tools
	if lookupEdge(t, graph, "SubAgent1", "Tool1") == nil {
		t.Error("Edge from SubAgent1 to Tool1 not found")
	}
}

func parseGraph(t *testing.T, dot string) *gographviz.Graph {
	t.Helper()
	astGraph, err := gographviz.ParseString(dot)
	if err != nil {
		t.Fatalf("parse DOT: %v\n%s", err, dot)
	}
	g := gographviz.NewGraph()
	if err := gographviz.Analyse(astGraph, g); err != nil {
		t.Fatalf("analyse DOT: %v\n%s", err, dot)
	}
	return g
}

func TestNestedWorkflowGraph(t *testing.T) {
	leaf := func(name string) agent.Agent {
		return newTestAgent(t, name, "", agentinternal.TypeLLMAgent, nil, nil)
	}

	t.Run("sequential with nested parallel", func(t *testing.T) {
		fanOut := newTestAgent(t, "fan_out", "", agentinternal.TypeParallelAgent, []agent.Agent{leaf("a"), leaf("b")}, nil)
		pipeline := newTestAgent(t, "pipeline", "", agentinternal.TypeSequentialAgent, []agent.Agent{fanOut, leaf("merge")}, nil)

		dot, err := GetAgentGraph(context.Background(), pipeline, nil)
		if err != nil {
			t.Fatalf("GetAgentGraph: %v", err)
		}
		if strings.HasPrefix(dot, "error:") {
			t.Fatalf("GetAgentGraph: %s", dot)
		}
		g := parseGraph(t, dot)
		if !g.IsSubGraph("cluster_pipeline") || !g.IsSubGraph("cluster_fan_out") {
			t.Fatalf("missing nested clusters\n%s", dot)
		}
		if parents := g.Relations.ChildToParents["cluster_fan_out"]; !parents["cluster_pipeline"] {
			t.Fatalf("cluster_fan_out parents = %v, want cluster_pipeline\n%s", parents, dot)
		}
		for _, name := range []string{"a", "b"} {
			if parents := g.Relations.ChildToParents[name]; !g.IsNode(name) || !parents["cluster_fan_out"] {
				t.Fatalf("node %s parents = %v, want cluster_fan_out\n%s", name, parents, dot)
			}
		}
		if parents := g.Relations.ChildToParents["merge"]; !g.IsNode("merge") || !parents["cluster_pipeline"] {
			t.Fatalf("merge parents = %v, want cluster_pipeline\n%s", parents, dot)
		}
		edge := lookupEdge(t, g, "b", "merge")
		if edge == nil {
			t.Fatalf("missing edge from the nested cluster to merge\n%s", dot)
		}
		if edge.Attrs["ltail"] != "cluster_fan_out" {
			t.Fatalf("ltail = %q, want cluster_fan_out\n%s", edge.Attrs["ltail"], dot)
		}
		if g.IsNode("cluster_fan_out") {
			t.Fatalf("cluster id was emitted as a node\n%s", dot)
		}
		if !strings.Contains(dot, "compound=true") {
			t.Fatalf("missing compound=true\n%s", dot)
		}
	})

	t.Run("loop with nested sequential", func(t *testing.T) {
		inner := newTestAgent(t, "inner", "", agentinternal.TypeSequentialAgent, []agent.Agent{leaf("x"), leaf("y")}, nil)
		cycle := newTestAgent(t, "cycle", "", agentinternal.TypeLoopAgent, []agent.Agent{inner, leaf("tail")}, nil)

		dot, err := GetAgentGraph(context.Background(), cycle, nil)
		if err != nil {
			t.Fatalf("GetAgentGraph: %v", err)
		}
		if strings.HasPrefix(dot, "error:") {
			t.Fatalf("GetAgentGraph: %s", dot)
		}
		g := parseGraph(t, dot)
		forward := lookupEdge(t, g, "y", "tail")
		if forward == nil || forward.Attrs["ltail"] != "cluster_inner" {
			t.Fatalf("forward edge = %v, want y -> tail ltail=cluster_inner\n%s", forward, dot)
		}
		back := lookupEdge(t, g, "tail", "x")
		if back == nil || back.Attrs["lhead"] != "cluster_inner" {
			t.Fatalf("back edge = %v, want tail -> x lhead=cluster_inner\n%s", back, dot)
		}
		if lookupEdge(t, g, "x", "y") == nil {
			t.Fatalf("missing edge x -> y inside nested sequential\n%s", dot)
		}
		if g.IsNode("cluster_inner") {
			t.Fatalf("cluster id was emitted as a node\n%s", dot)
		}
	})

	t.Run("three level nesting", func(t *testing.T) {
		fan := newTestAgent(t, "fan", "", agentinternal.TypeParallelAgent, []agent.Agent{leaf("a"), leaf("b")}, nil)
		mid := newTestAgent(t, "mid", "", agentinternal.TypeSequentialAgent, []agent.Agent{fan}, nil)
		outer := newTestAgent(t, "outer", "", agentinternal.TypeSequentialAgent, []agent.Agent{mid, leaf("merge")}, nil)

		dot, err := GetAgentGraph(context.Background(), outer, nil)
		if err != nil {
			t.Fatalf("GetAgentGraph: %v", err)
		}
		if strings.HasPrefix(dot, "error:") {
			t.Fatalf("GetAgentGraph: %s", dot)
		}
		g := parseGraph(t, dot)
		if parents := g.Relations.ChildToParents["cluster_mid"]; !g.IsSubGraph("cluster_mid") || !parents["cluster_outer"] {
			t.Fatalf("cluster_mid parents = %v, want cluster_outer\n%s", parents, dot)
		}
		if parents := g.Relations.ChildToParents["cluster_fan"]; !g.IsSubGraph("cluster_fan") || !parents["cluster_mid"] {
			t.Fatalf("cluster_fan parents = %v, want cluster_mid\n%s", parents, dot)
		}
		for _, name := range []string{"a", "b"} {
			if parents := g.Relations.ChildToParents[name]; !g.IsNode(name) || !parents["cluster_fan"] {
				t.Fatalf("node %s parents = %v, want cluster_fan\n%s", name, parents, dot)
			}
		}
		edge := lookupEdge(t, g, "b", "merge")
		if edge == nil || edge.Attrs["ltail"] != "cluster_mid" {
			t.Fatalf("edge = %v, want b -> merge ltail=cluster_mid\n%s", edge, dot)
		}
		if g.IsNode("fan") || g.IsNode("cluster_fan") || g.IsNode("cluster_mid") {
			t.Fatalf("cluster id was emitted as a node\n%s", dot)
		}
	})

	t.Run("empty cluster", func(t *testing.T) {
		fanOut := newTestAgent(t, "fan_out", "", agentinternal.TypeParallelAgent, nil, nil)
		s := newTestAgent(t, "s", "", agentinternal.TypeSequentialAgent, []agent.Agent{fanOut, leaf("merge")}, nil)

		dot, err := GetAgentGraph(context.Background(), s, nil)
		if err != nil {
			t.Fatalf("GetAgentGraph: %v", err)
		}
		if strings.HasPrefix(dot, "error:") {
			t.Fatalf("GetAgentGraph: %s", dot)
		}
		g := parseGraph(t, dot)
		if parents := g.Relations.ChildToParents["cluster_fan_out"]; !g.IsSubGraph("cluster_fan_out") || !parents["cluster_s"] {
			t.Fatalf("cluster_fan_out parents = %v, want cluster_s\n%s", parents, dot)
		}
		if g.IsNode("cluster_fan_out") {
			t.Fatalf("cluster id was emitted as a node\n%s", dot)
		}
		if lookupEdge(t, g, "cluster_fan_out", "merge") != nil {
			t.Fatalf("empty cluster was used as an edge endpoint\n%s", dot)
		}
		if parents := g.Relations.ChildToParents["merge"]; !g.IsNode("merge") || !parents["cluster_s"] {
			t.Fatalf("merge parents = %v, want cluster_s\n%s", parents, dot)
		}
	})

	t.Run("workflow also used as a tool", func(t *testing.T) {
		research := newTestAgent(t, "research", "", agentinternal.TypeParallelAgent, []agent.Agent{leaf("web"), leaf("docs")}, nil)
		triage := newTestAgent(t, "triage", "", agentinternal.TypeLLMAgent, nil, []tool.Tool{agenttool.New(research, nil)})
		root := newTestAgent(t, "root", "", agentinternal.TypeSequentialAgent, []agent.Agent{triage, research, leaf("writer")}, nil)

		dot, err := GetAgentGraph(context.Background(), root, nil)
		if err != nil {
			t.Fatalf("GetAgentGraph: %v", err)
		}
		if strings.HasPrefix(dot, "error:") {
			t.Fatalf("GetAgentGraph: %s", dot)
		}
		g := parseGraph(t, dot)
		for _, pair := range [][2]string{{"triage", "research"}, {"research", "writer"}} {
			edge := lookupEdge(t, g, pair[0], pair[1])
			if edge == nil {
				t.Fatalf("missing edge %s -> %s\n%s", pair[0], pair[1], dot)
			}
			if edge.Attrs["ltail"] != "" || edge.Attrs["lhead"] != "" {
				t.Fatalf("edge %s -> %s attrs = %v, want no ltail or lhead\n%s", pair[0], pair[1], edge.Attrs, dot)
			}
		}
		researchNode := g.Nodes.Lookup["research"]
		if researchNode == nil || researchNode.Attrs["shape"] != "box" {
			t.Fatalf("research node = %v, want the tool box\n%s", researchNode, dot)
		}
		if g.IsSubGraph("cluster_research") || g.IsNode("web") || g.IsNode("docs") {
			t.Fatalf("undrawn workflow was anchored inside its cluster\n%s", dot)
		}
	})
}

type args struct{}

func newTestTool(name string) tool.Tool {
	t, err := functiontool.New(functiontool.Config{Name: name, Description: name},
		func(agent.Context, args) (string, error) { return ``, nil })
	if err != nil {
		panic(err)
	}
	return t
}

type staticToolset struct{}

func (staticToolset) Name() string { return "my_toolset" }
func (staticToolset) Tools(agent.ReadonlyContext) ([]tool.Tool, error) {
	return []tool.Tool{newTestTool("toolset_tool")}, nil
}

func TestAgentGraphWithToolsetAndSubAgent(t *testing.T) {
	helper, err := llmagent.New(llmagent.Config{Name: "helper", Description: "helper", Model: &dummyLLM{}})
	if err != nil {
		t.Fatal(err)
	}

	root, err := llmagent.New(llmagent.Config{
		Name:      "root",
		Model:     &dummyLLM{},
		Tools:     []tool.Tool{newTestTool("direct_tool")},
		Toolsets:  []tool.Toolset{staticToolset{}},
		SubAgents: []agent.Agent{helper},
	})
	if err != nil {
		t.Fatal(err)
	}

	dot, err := GetAgentGraph(context.Background(), root, nil)
	if err != nil {
		t.Fatal(err)
	}
	g := parseGraph(t, dot)
	for _, to := range []string{"direct_tool", "toolset_tool", "helper"} {
		if lookupEdge(t, g, "root", to) == nil {
			t.Errorf("missing edge root -> %s\n%s", to, dot)
		}
	}
}

// failingToolset fails to list its tools the way a toolset does outside an
// invocation, for example an MCP server that is down or per-user auth with no
// user. It returns a tool alongside the error, which must not be drawn: a
// result that comes with an error is not used.
type failingToolset struct{}

func (failingToolset) Name() string { return "failing" }

func (failingToolset) Tools(agent.ReadonlyContext) ([]tool.Tool, error) {
	return []tool.Tool{&mockTool{name: "partial_tool"}}, errors.New("cannot list tools")
}

// panickingToolset panics the way a toolset does when it reads a part of the
// context that only a run has, such as the session.
type panickingToolset struct{}

func (panickingToolset) Name() string { return "panicking" }

func (panickingToolset) Tools(ctx agent.ReadonlyContext) ([]tool.Tool, error) {
	panic("no session outside a run")
}

// TestAgentGraphSkipsToolsetsThatCannotListTools pins that a toolset that
// fails or panics is left out and the rest of the graph, including toolsets
// after it, still renders. adk-python logs and skips such a toolset too.
func TestAgentGraphSkipsToolsetsThatCannotListTools(t *testing.T) {
	for _, tt := range []struct {
		name    string
		toolset tool.Toolset
	}{
		{"error", failingToolset{}},
		{"panic", panickingToolset{}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			helper := newTestAgent(t, "helper", "", agentinternal.TypeLLMAgent, nil, nil)
			root, err := llmagent.New(llmagent.Config{
				Name:      "root",
				Model:     &dummyLLM{},
				Tools:     []tool.Tool{newTestTool("direct_tool")},
				Toolsets:  []tool.Toolset{tt.toolset, staticToolset{}},
				SubAgents: []agent.Agent{helper},
			})
			if err != nil {
				t.Fatal(err)
			}

			dot, err := GetAgentGraph(context.Background(), root, nil)
			if err != nil {
				t.Fatalf("GetAgentGraph: %v", err)
			}
			g := parseGraph(t, dot)
			for _, to := range []string{"direct_tool", "toolset_tool", "helper"} {
				if lookupEdge(t, g, "root", to) == nil {
					t.Errorf("missing edge root -> %s\n%s", to, dot)
				}
			}
			if g.IsNode("partial_tool") {
				t.Errorf("drew a tool from a toolset that returned an error\n%s", dot)
			}
		})
	}
}

// TestAgentGraphToolsetSeesItsAgent pins that toolset resolution is given the
// agent, so a filter that picks tools per agent works in the graph as it does
// in a run, instead of panicking and dropping the toolset.
func TestAgentGraphToolsetSeesItsAgent(t *testing.T) {
	onlyRoot := tool.FilterToolset(staticToolset{}, func(ctx agent.ReadonlyContext, _ tool.Tool) bool {
		return ctx.AgentName() == "root"
	})
	helper, err := llmagent.New(llmagent.Config{Name: "helper", Model: &dummyLLM{}, Toolsets: []tool.Toolset{onlyRoot}})
	if err != nil {
		t.Fatal(err)
	}
	root, err := llmagent.New(llmagent.Config{
		Name:      "root",
		Model:     &dummyLLM{},
		Toolsets:  []tool.Toolset{onlyRoot},
		SubAgents: []agent.Agent{helper},
	})
	if err != nil {
		t.Fatal(err)
	}

	dot, err := GetAgentGraph(context.Background(), root, nil)
	if err != nil {
		t.Fatalf("GetAgentGraph: %v", err)
	}
	g := parseGraph(t, dot)
	if lookupEdge(t, g, "root", "toolset_tool") == nil {
		t.Errorf("missing edge root -> toolset_tool\n%s", dot)
	}
	if lookupEdge(t, g, "helper", "toolset_tool") != nil {
		t.Errorf("edge helper -> toolset_tool drawn, but the filter keeps the tool for root only\n%s", dot)
	}
}

// captureDefaultLog sends slog's default logger to a JSON buffer for the rest
// of the test.
func captureDefaultLog(t *testing.T) *bytes.Buffer {
	t.Helper()
	var buf bytes.Buffer
	prev, prevOut, prevFlags := slog.Default(), log.Writer(), log.Flags()
	slog.SetDefault(slog.New(slog.NewJSONHandler(&buf, nil)))
	t.Cleanup(func() {
		// SetDefault also redirects the log package; put both back.
		slog.SetDefault(prev)
		log.SetOutput(prevOut)
		log.SetFlags(prevFlags)
	})
	return &buf
}

// TestAgentGraphLogsSkippedToolsets pins that a skipped toolset leaves a trace,
// so a developer can see why its tools are missing, and that the trace names
// types only: an error or panic value can carry a server URL or credentials.
func TestAgentGraphLogsSkippedToolsets(t *testing.T) {
	logs := captureDefaultLog(t)
	root, err := llmagent.New(llmagent.Config{
		Name:     "root",
		Model:    &dummyLLM{},
		Toolsets: []tool.Toolset{failingToolset{}, panickingToolset{}, nil, staticToolset{}},
	})
	if err != nil {
		t.Fatal(err)
	}

	dot, err := GetAgentGraph(context.Background(), root, nil)
	if err != nil {
		t.Fatalf("GetAgentGraph: %v", err)
	}
	if lookupEdge(t, parseGraph(t, dot), "root", "toolset_tool") == nil {
		t.Errorf("missing edge root -> toolset_tool\n%s", dot)
	}

	var got []map[string]any
	dec := json.NewDecoder(logs)
	for dec.More() {
		var rec map[string]any
		if err := dec.Decode(&rec); err != nil {
			t.Fatalf("decode log record: %v", err)
		}
		delete(rec, "time")
		got = append(got, rec)
	}
	const msg = "agent graph: skipping a toolset that could not list its tools"
	want := []map[string]any{
		{"level": "WARN", "msg": msg, "agent": "root", "toolset": "failing", "toolset_type": "services.failingToolset", "kind": "error", "error_type": "*errors.errorString"},
		{"level": "WARN", "msg": msg, "agent": "root", "toolset": "panicking", "toolset_type": "services.panickingToolset", "kind": "panic", "error_type": "string"},
	}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("log records mismatch (-want +got):\n%s", diff)
	}
}

// blockingToolset never lists its tools on its own, like an MCP server that
// accepts the connection and never answers. It gives up when its context does.
type blockingToolset struct{}

func (blockingToolset) Name() string { return "blocking" }

func (blockingToolset) Tools(ctx agent.ReadonlyContext) ([]tool.Tool, error) {
	<-ctx.Done()
	return nil, ctx.Err()
}

// TestAgentGraphBoundsToolsetResolution pins that an unresponsive toolset
// costs the graph at most the resolution budget. Without one the graph waits
// on it forever, where it used to render without the toolset's tools.
func TestAgentGraphBoundsToolsetResolution(t *testing.T) {
	prev := toolsetResolveBudget
	toolsetResolveBudget = 50 * time.Millisecond
	t.Cleanup(func() { toolsetResolveBudget = prev })

	helper := newTestAgent(t, "helper", "", agentinternal.TypeLLMAgent, nil, nil)
	root, err := llmagent.New(llmagent.Config{
		Name:      "root",
		Model:     &dummyLLM{},
		Tools:     []tool.Tool{newTestTool("direct_tool")},
		Toolsets:  []tool.Toolset{blockingToolset{}, staticToolset{}},
		SubAgents: []agent.Agent{helper},
	})
	if err != nil {
		t.Fatal(err)
	}

	type result struct {
		dot string
		err error
	}
	done := make(chan result, 1)
	go func() {
		dot, err := GetAgentGraph(context.Background(), root, nil)
		done <- result{dot, err}
	}()
	var res result
	select {
	case res = <-done:
	case <-time.After(10 * time.Second):
		t.Fatal("GetAgentGraph still waiting on an unresponsive toolset after 10s")
	}
	if res.err != nil {
		t.Fatalf("GetAgentGraph: %v", res.err)
	}
	g := parseGraph(t, res.dot)
	for _, to := range []string{"direct_tool", "toolset_tool", "helper"} {
		if lookupEdge(t, g, "root", to) == nil {
			t.Errorf("missing edge root -> %s\n%s", to, res.dot)
		}
	}
}

// TestAgentGraphLLMAgentWithWorkflowSubAgent pins the edge from an llmagent
// into a workflow sub-agent. The workflow is drawn as a cluster, not a node,
// so the edge goes to its first member with lhead set, as adk-python does.
func TestAgentGraphLLMAgentWithWorkflowSubAgent(t *testing.T) {
	for _, agentType := range []agentinternal.Type{
		agentinternal.TypeSequentialAgent,
		agentinternal.TypeLoopAgent,
		agentinternal.TypeParallelAgent,
	} {
		t.Run(string(agentType), func(t *testing.T) {
			pipeline := newTestAgent(t, "pipeline", "", agentType, []agent.Agent{
				newTestAgent(t, "a", "", agentinternal.TypeLLMAgent, nil, nil),
				newTestAgent(t, "b", "", agentinternal.TypeLLMAgent, nil, nil),
			}, nil)
			root := newTestAgent(t, "root", "", agentinternal.TypeLLMAgent, []agent.Agent{pipeline}, nil)

			dot, err := GetAgentGraph(context.Background(), root, nil)
			if err != nil {
				t.Fatalf("GetAgentGraph: %v", err)
			}
			g := parseGraph(t, dot)
			if g.IsNode("pipeline") || g.IsNode("cluster_pipeline") {
				t.Fatalf("workflow agent was drawn as a node\n%s", dot)
			}
			for _, name := range []string{"a", "b"} {
				if parents := g.Relations.ChildToParents[name]; !parents["cluster_pipeline"] {
					t.Errorf("node %s parents = %v, want cluster_pipeline\n%s", name, parents, dot)
				}
			}
			edge := lookupEdge(t, g, "root", "a")
			if edge == nil || edge.Attrs["lhead"] != "cluster_pipeline" {
				t.Fatalf("edge = %v, want root -> a with lhead cluster_pipeline\n%s", edge, dot)
			}
			if g.Attrs["compound"] != "true" {
				t.Errorf("compound = %q, want true so Graphviz clips the edge at the cluster\n%s", g.Attrs["compound"], dot)
			}
		})
	}
}

// TestAgentGraphLLMAgentInsideWorkflow pins the edges of an llmagent nested in
// a workflow agent. Inside a cluster, graph is only the cluster's name holder,
// so an edge drawn there is never serialized.
func TestAgentGraphLLMAgentInsideWorkflow(t *testing.T) {
	helper := newTestAgent(t, "helper", "", agentinternal.TypeLLMAgent, nil, nil)
	worker, err := llmagent.New(llmagent.Config{
		Name:      "worker",
		Model:     &dummyLLM{},
		Tools:     []tool.Tool{newTestTool("direct_tool")},
		Toolsets:  []tool.Toolset{staticToolset{}},
		SubAgents: []agent.Agent{helper},
	})
	if err != nil {
		t.Fatal(err)
	}
	last := newTestAgent(t, "last", "", agentinternal.TypeLLMAgent, nil, nil)
	pipeline := newTestAgent(t, "pipeline", "", agentinternal.TypeSequentialAgent, []agent.Agent{worker, last}, nil)

	dot, err := GetAgentGraph(context.Background(), pipeline, nil)
	if err != nil {
		t.Fatalf("GetAgentGraph: %v", err)
	}
	g := parseGraph(t, dot)
	for _, to := range []string{"direct_tool", "toolset_tool", "helper", "last"} {
		if lookupEdge(t, g, "worker", to) == nil {
			t.Errorf("missing edge worker -> %s\n%s", to, dot)
		}
	}
	for _, name := range []string{"worker", "direct_tool", "toolset_tool", "helper", "last"} {
		if parents := g.Relations.ChildToParents[name]; !g.IsNode(name) || !parents["cluster_pipeline"] {
			t.Errorf("node %s parents = %v, want cluster_pipeline\n%s", name, parents, dot)
		}
	}
}

func TestDotID(t *testing.T) {
	for _, tt := range []struct {
		name string
		want string
	}{
		// Plain identifiers stay bare, so graphs that rendered before are unchanged.
		{"root", "root"},
		{"_tool_2", "_tool_2"},
		{"SubAgent1", "SubAgent1"},
		{"nodes", "nodes"},
		// Everything else is quoted.
		{"", `""`},
		{"resolve-library-id", `"resolve-library-id"`},
		{"1st", `"1st"`},
		{"a.b", `"a.b"`},
		{"ns:tool", `"ns:tool"`},
		{"two words", `"two words"`},
		{"naïve", `"naïve"`},
		{"node", `"node"`},
		{"EDGE", `"EDGE"`},
		{"Subgraph", `"Subgraph"`},
		{`say"hi`, `"say\"hi"`},
		{`trailing\`, `"trailing\\"`},
		{`x\"y`, `"x\\\"y"`},
	} {
		if got := dotID(tt.name); got != tt.want {
			t.Errorf("dotID(%q) = %s, want %s", tt.name, got, tt.want)
		}
	}
}

// namedToolset returns tools with the given names, the way an MCP server
// does: the server picks the names, and nothing limits them to DOT identifiers.
type namedToolset struct {
	name  string
	tools []string
}

func (s namedToolset) Name() string { return s.name }

func (s namedToolset) Tools(agent.ReadonlyContext) ([]tool.Tool, error) {
	var tools []tool.Tool
	for _, name := range s.tools {
		tools = append(tools, &mockTool{name: name})
	}
	return tools, nil
}

// TestAgentGraphQuotesNamesThatAreNotDOTIdentifiers pins that every ID the
// generator emits parses, whatever an agent or tool is called. Graphviz rejects
// a bare resolve-library-id, and reads a bare node as the node-defaults
// statement and ns:tool as a node with a port, so these must all be quoted.
func TestAgentGraphQuotesNamesThatAreNotDOTIdentifiers(t *testing.T) {
	toolsetTools := []string{
		"resolve-library-id", // Context7's MCP server
		`say"hi`,
		`trailing\`,
		"1st",
		"a.b",
		"ns:tool",
		"node",
		"Graph",
		"naïve",
	}
	steps := newTestAgent(t, "my-pipeline", "", agentinternal.TypeSequentialAgent, []agent.Agent{
		newTestAgent(t, "step-one", "", agentinternal.TypeLLMAgent, nil, nil),
		newTestAgent(t, "step_two", "", agentinternal.TypeLLMAgent, nil, nil),
	}, nil)
	root, err := llmagent.New(llmagent.Config{
		Name:      "root-agent",
		Model:     &dummyLLM{},
		Tools:     []tool.Tool{&mockTool{name: "direct-tool"}},
		Toolsets:  []tool.Toolset{namedToolset{name: "mcp", tools: toolsetTools}},
		SubAgents: []agent.Agent{steps},
	})
	if err != nil {
		t.Fatal(err)
	}

	dot, err := GetAgentGraph(context.Background(), root, [][]string{{"resolve-library-id", "root-agent"}})
	if err != nil {
		t.Fatalf("GetAgentGraph: %v", err)
	}
	g := parseGraph(t, dot)

	// Quoted IDs keep their quotes and escapes after parsing.
	wantTools := []string{
		`"direct-tool"`,
		`"resolve-library-id"`,
		`"say\"hi"`,
		`"trailing\\"`,
		`"1st"`,
		`"a.b"`,
		`"ns:tool"`,
		`"node"`,
		`"Graph"`,
		`"naïve"`,
	}
	for _, id := range wantTools {
		if !g.IsNode(id) {
			t.Errorf("missing tool node %s\n%s", id, dot)
		}
		if lookupEdge(t, g, `"root-agent"`, id) == nil {
			t.Errorf("missing edge root-agent -> %s\n%s", id, dot)
		}
	}
	if label := g.Nodes.Lookup[`"say\"hi"`]; label == nil || label.Attrs["label"] != `"🔧 say\"hi"` {
		t.Errorf("say\"hi node = %v, want its name in the label\n%s", label, dot)
	}
	if !g.IsSubGraph(`"cluster_my-pipeline"`) {
		t.Fatalf("missing cluster for my-pipeline\n%s", dot)
	}
	for _, id := range []string{`"step-one"`, "step_two"} {
		if parents := g.Relations.ChildToParents[id]; !parents[`"cluster_my-pipeline"`] {
			t.Errorf("node %s parents = %v, want cluster_my-pipeline\n%s", id, parents, dot)
		}
	}
	into := lookupEdge(t, g, `"root-agent"`, `"step-one"`)
	if into == nil || into.Attrs["lhead"] != `"cluster_my-pipeline"` {
		t.Errorf("edge into the pipeline = %v, want root-agent -> step-one with lhead cluster_my-pipeline\n%s", into, dot)
	}
	if lookupEdge(t, g, `"step-one"`, "step_two") == nil {
		t.Errorf("missing edge step-one -> step_two\n%s", dot)
	}
	// Highlighting matches the raw names an event carries, not the quoted IDs.
	hl := lookupEdge(t, g, `"root-agent"`, `"resolve-library-id"`)
	if hl == nil || hl.Attrs["color"] != LightGreen || hl.Attrs["dir"] != "back" {
		t.Errorf("highlighted edge = %v, want color %s and dir back\n%s", hl, LightGreen, dot)
	}
}
