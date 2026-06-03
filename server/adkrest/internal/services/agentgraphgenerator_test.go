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
	"iter"
	"testing"

	"github.com/awalterschulze/gographviz"
	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/agent/workflowagents/loopagent"
	"google.golang.org/adk/agent/workflowagents/parallelagent"
	"google.golang.org/adk/agent/workflowagents/sequentialagent"
	agentinternal "google.golang.org/adk/internal/agent"
	"google.golang.org/adk/model"
	"google.golang.org/adk/tool"
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
				err = drawNode(graph, parentGraph, tt.agent, tt.highlightedPairs, visitedNodes)
				if err != nil {
					t.Fatalf("drawNode failed: %v", err)
				}
				nodeName = tt.agent.Name()
			}
			if tt.tool != nil {
				err = drawNode(graph, parentGraph, tt.tool, tt.highlightedPairs, visitedNodes)
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
	err = drawNode(graph, parentGraph, agent, [][]string{}, visitedNodes)
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

			err = drawEdge(graph, tt.from, tt.to, tt.highlightedPairs)
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
			err = drawCluster(parentGraph, clusterGraph, parentAgent, [][]string{}, visitedNodes)
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

	err = buildGraph(graph, parentGraph, mainAgent, [][]string{}, visitedNodes)
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
