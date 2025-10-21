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
	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/agent/workflowagents/loopagent"
	"google.golang.org/adk/agent/workflowagents/parallelagent"
	"google.golang.org/adk/agent/workflowagents/sequentialagent"
	agentinternal "google.golang.org/adk/internal/agent"
	"google.golang.org/adk/model"
	"google.golang.org/adk/tool"
	"google.golang.org/genai"
)

type dummyLLM struct {
	name string
}

func (d *dummyLLM) Name() string {
	return d.name
}

func (d *dummyLLM) GenerateContent(ctx context.Context, req *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		yield(model.CreateResponse(&genai.GenerateContentResponse{
			Candidates: []*genai.Candidate{
				{
					Content: &genai.Content{
						Parts: []*genai.Part{{Text: "Response from agentgrapgenerator test."}},
					},
				},
			},
		}), nil)
	}
}

// Helper to create a generic agent.Agent
func newTestAgent(t *testing.T, name, description string, agentType agentinternal.Type, subAgents []agent.Agent) agent.Agent {
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
	default:
		// Fallback to a basic LLM agent for other types, as it's a concrete, non-cluster agent.
		return newTestLLMAgent(name, description, nil, subAgents)
	}

	if err != nil {
		t.Fatalf("failed to create agent: %v", err)
	}
	return a
}

// Helper to create an LLM agent
func newTestLLMAgent(name, description string, tools []tool.Tool, subAgents []agent.Agent) agent.Agent {
	llm := &dummyLLM{}
	a, _ := llmagent.New(llmagent.Config{
		Name:        name,
		Description: description,
		Model:       llm,
		Tools:       tools,
		SubAgents:   subAgents,
	})
	return a
}

// Mock tool for testing
type mockTool struct {
	name string
}

func (m *mockTool) Name() string        { return m.name }
func (m *mockTool) Description() string { return "" }
func (m *mockTool) IsLongRunning() bool { return false }

func TestNodeName(t *testing.T) {
	tests := []struct {
		name     string
		instance any
		expected string
	}{
		{"agent", newTestAgent(t, "TestAgent", "", agentinternal.TypeCustomAgent, nil), "TestAgent"},
		{"tool", &mockTool{name: "TestTool"}, "TestTool"},
		{"unknown", "some string", "Unknown instance type"},
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
		{"llm agent", newTestLLMAgent("LLMAgent", "", nil, nil), "\"🤖 LLMAgent\""},
		{"sequential agent", newTestAgent(t, "SeqAgent", "", agentinternal.TypeSequentialAgent, nil), "\"SeqAgent (SequentialAgent)\""},
		{"loop agent", newTestAgent(t, "LoopAgent", "", agentinternal.TypeLoopAgent, nil), "\"LoopAgent (LoopAgent)\""},
		{"parallel agent", newTestAgent(t, "ParAgent", "", agentinternal.TypeParallelAgent, nil), "\"ParAgent (ParallelAgent)\""},
		{"tool", &mockTool{name: "TestTool"}, "\"🔧 TestTool\""},
		{"unknown", "some string", "\"Unsupported agent or tool type\""},
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
		{"agent", newTestAgent(t, "TestAgent", "", agentinternal.TypeCustomAgent, nil), "ellipse"},
		{"tool", &mockTool{name: "TestTool"}, "box"},
		{"unknown", "some string", "cylinder"},
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
		{"llm agent", newTestLLMAgent("LLMAgent", "", nil, nil), false},
		{"sequential agent", newTestAgent(t, "SeqAgent", "", agentinternal.TypeSequentialAgent, nil), true},
		{"loop agent", newTestAgent(t, "LoopAgent", "", agentinternal.TypeLoopAgent, nil), true},
		{"parallel agent", newTestAgent(t, "ParAgent", "", agentinternal.TypeParallelAgent, nil), true},
		{"tool", &mockTool{name: "TestTool"}, false},
		{"unknown", "some string", false},
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
		{name: "no highlight", nodeName: "NodeA", highlightedPairs: [][]string{}, expected: false},
		{name: "node in pair", nodeName: "NodeA", highlightedPairs: [][]string{{"NodeA", "NodeB"}}, expected: true},
		{name: "node not in pair", nodeName: "NodeC", highlightedPairs: [][]string{{"NodeA", "NodeB"}}, expected: false},
		{name: "multiple pairs", nodeName: "NodeB", highlightedPairs: [][]string{{"NodeA", "NodeB"}, {"NodeC", "NodeD"}}, expected: true},
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
		{name: "no highlight pairs", from: "A", to: "B", highlightedPairs: [][]string{}, expected: nil},
		{name: "matching forward", from: "A", to: "B", highlightedPairs: [][]string{{"A", "B"}}, expected: boolPtr(true)},
		{name: "matching backward", from: "B", to: "A", highlightedPairs: [][]string{{"A", "B"}}, expected: boolPtr(false)},
		{name: "no match", from: "C", to: "D", highlightedPairs: [][]string{{"A", "B"}}, expected: nil},
		{name: "partial match", from: "A", to: "C", highlightedPairs: [][]string{{"A", "B"}}, expected: nil},
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
	graph := gographviz.NewGraph()
	graph.SetName("G")
	parentGraph := graph
	visitedNodes := make(map[string]bool)

	t.Run("draw agent node", func(t *testing.T) {
		agent := newTestAgent(t, "MyAgent", "", agentinternal.TypeCustomAgent, nil)
		err := drawNode(graph, parentGraph, agent, [][]string{}, visitedNodes)
		if err != nil {
			t.Fatalf("drawNode failed: %v", err)
		}
		node := graph.Nodes.Lookup["MyAgent"]
		if node == nil {
			t.Fatal("Agent node not found in graph")
		}
		if node.Attrs["label"] != "\"🤖 MyAgent\"" {
			t.Errorf("Agent node label mismatch: got %s", node.Attrs["label"])
		}
		if node.Attrs["shape"] != "ellipse" {
			t.Errorf("Agent node shape mismatch: got %s", node.Attrs["shape"])
		}
		if node.Attrs["color"] != LightGray {
			t.Errorf("Agent node color mismatch: got %s", node.Attrs["color"])
		}
		if !visitedNodes["MyAgent"] {
			t.Error("Agent node not marked as visited")
		}
	})

	// Reset visitedNodes for the next test case
	visitedNodes = make(map[string]bool)

	t.Run("draw highlighted agent node", func(t *testing.T) {
		agent := newTestAgent(t, "HighlightedAgent", "", agentinternal.TypeCustomAgent, nil)
		highlightedPairs := [][]string{{"HighlightedAgent", "Tool1"}}
		err := drawNode(graph, parentGraph, agent, highlightedPairs, visitedNodes)
		if err != nil {
			t.Fatalf("drawNode failed: %v", err)
		}
		node := graph.Nodes.Lookup["HighlightedAgent"]
		if node == nil {
			t.Fatal("Highlighted agent node not found in graph")
		}
		if node.Attrs["color"] != DarkGreen {
			t.Errorf("Highlighted agent node color mismatch: got %s", node.Attrs["color"])
		}
		if node.Attrs["style"] != "filled" {
			t.Errorf("Highlighted agent node style mismatch: got %s", node.Attrs["style"])
		}
	})

	// Reset visitedNodes for the next test case
	visitedNodes = make(map[string]bool)

	t.Run("draw tool node", func(t *testing.T) {
		tool := &mockTool{name: "MyTool"}
		err := drawNode(graph, parentGraph, tool, [][]string{}, visitedNodes)
		if err != nil {
			t.Fatalf("drawNode failed: %v", err)
		}
		node := graph.Nodes.Lookup["MyTool"]
		if node == nil {
			t.Fatal("Tool node not found in graph")
		}
		if node.Attrs["label"] != "\"🔧 MyTool\"" {
			t.Errorf("Tool node label mismatch: got %s", node.Attrs["label"])
		}
		if node.Attrs["shape"] != "box" {
			t.Errorf("Tool node shape mismatch: got %s", node.Attrs["shape"])
		}
		if node.Attrs["color"] != LightGray {
			t.Errorf("Tool node color mismatch: got %s", node.Attrs["color"])
		}
		if !visitedNodes["MyTool"] {
			t.Error("Tool node not marked as visited")
		}
	})

	// Reset visitedNodes for the next test case
	visitedNodes = make(map[string]bool)

	t.Run("draw cluster agent", func(t *testing.T) {
		agent := newTestAgent(t, "MyClusterAgent", "", agentinternal.TypeSequentialAgent, nil)
		err := drawNode(graph, parentGraph, agent, [][]string{}, visitedNodes)
		if err != nil {
			t.Fatalf("drawNode failed: %v", err)
		}
		clusterName := "cluster_MyClusterAgent"
		cluster := graph.SubGraphs.SubGraphs[clusterName]
		if cluster == nil {
			t.Fatal("Cluster not found in graph")
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
	})
}

func lookupEdge(t *testing.T, graph *gographviz.Graph, src string, dst string) *gographviz.Edge {
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
	graph := gographviz.NewGraph()
	graph.SetName("G")

	// Add nodes for edges to connect
	graph.AddNode("G", "NodeA", nil)
	graph.AddNode("G", "NodeB", nil)
	graph.AddNode("G", "NodeC", nil)
	graph.AddNode("G", "NodeD", nil)
	graph.AddNode("G", "NodeE", nil)
	graph.AddNode("G", "NodeF", nil)

	t.Run("draw unhighlighted edge", func(t *testing.T) {
		err := drawEdge(graph, "NodeA", "NodeB", [][]string{})
		if err != nil {
			t.Fatalf("drawEdge failed: %v", err)
		}
		edge := lookupEdge(t, graph, "NodeA", "NodeB")
		if edge == nil {
			t.Fatalf("Edge between NodeA and NodeB not found")
		}
		if edge.Attrs["color"] != LightGray {
			t.Errorf("Edge color mismatch: got %s", edge.Attrs["color"])
		}
		if edge.Attrs["arrowhead"] != "none" {
			t.Errorf("Edge arrowhead mismatch: got %s", edge.Attrs["arrowhead"])
		}
	})

	t.Run("draw highlighted forward edge", func(t *testing.T) {
		err := drawEdge(graph, "NodeC", "NodeD", [][]string{{"NodeC", "NodeD"}})
		if err != nil {
			t.Fatalf("drawEdge failed: %v", err)
		}
		edge := lookupEdge(t, graph, "NodeC", "NodeD")
		if edge == nil {
			t.Fatalf("Edge between NodeC and NodeD not found")
		}
		if edge.Attrs["color"] != LightGreen {
			t.Errorf("Highlighted edge color mismatch: got %s", edge.Attrs["color"])
		}
		if edge.Attrs["arrowhead"] != "normal" {
			t.Errorf("Highlighted edge arrowhead mismatch: got %s", edge.Attrs["arrowhead"])
		}
		if edge.Attrs["dir"] != "" {
			t.Errorf("Highlighted edge dir mismatch: got %s", edge.Attrs["dir"])
		}
	})

	t.Run("draw highlighted backward edge", func(t *testing.T) {
		err := drawEdge(graph, "NodeE", "NodeF", [][]string{{"NodeF", "NodeE"}})
		if err != nil {
			t.Fatalf("drawEdge failed: %v", err)
		}
		edge := lookupEdge(t, graph, "NodeE", "NodeF")
		if edge == nil {
			t.Fatal("Highlighted backward edge not found in graph")
		}
		if edge.Attrs["color"] != LightGreen {
			t.Errorf("Highlighted backward edge color mismatch: got %s", edge.Attrs["color"])
		}
		if edge.Attrs["arrowhead"] != "normal" {
			t.Errorf("Highlighted backward edge arrowhead mismatch: got %s", edge.Attrs["arrowhead"])
		}
		if edge.Attrs["dir"] != "back" {
			t.Errorf("Highlighted backward edge dir mismatch: got %s", edge.Attrs["dir"])
		}
	})
}

func TestDrawCluster(t *testing.T) {
	parentGraph := gographviz.NewGraph()
	parentGraph.SetName("ParentG")
	visitedNodes := make(map[string]bool)

	t.Run("sequential agent cluster", func(t *testing.T) {
		subAgent1 := newTestLLMAgent("SubAgent1", "", nil, nil)
		subAgent2 := newTestLLMAgent("SubAgent2", "", nil, nil)
		seqAgent := newTestAgent(t, "SeqAgent", "", agentinternal.TypeSequentialAgent, []agent.Agent{subAgent1, subAgent2})

		clusterGraph := gographviz.NewGraph()
		clusterGraph.SetName("cluster_SeqAgent")

		err := drawCluster(parentGraph, clusterGraph, seqAgent, [][]string{}, visitedNodes)
		if err != nil {
			t.Fatalf("drawCluster failed: %v", err)
		}

		// Check if sub-agents are drawn as nodes in the parent graph (since drawNode adds to parentGraph)
		if parentGraph.Nodes.Lookup["SubAgent1"] == nil || parentGraph.Nodes.Lookup["SubAgent2"] == nil {
			t.Error("Sub-agents not drawn as nodes in parent graph")
		}
		edge := lookupEdge(t, parentGraph, "SubAgent1", "SubAgent2")

		// Check if edge exists between sub-agents
		if edge == nil {
			t.Fatalf("Edge between SubAgent1 and SubAgent2 not found")
		}
		if edge.Attrs["arrowhead"] != "none" {
			t.Errorf("Sequential agent edge arrowhead mismatch: got %s", edge.Attrs["arrowhead"])
		}
	})

	visitedNodes = make(map[string]bool)

	t.Run("loop agent cluster", func(t *testing.T) {
		subAgent1 := newTestLLMAgent("LoopSubAgent1", "", nil, nil)
		subAgent2 := newTestLLMAgent("LoopSubAgent2", "", nil, nil)
		loopAgent := newTestAgent(t, "LoopAgent", "", agentinternal.TypeLoopAgent, []agent.Agent{subAgent1, subAgent2})

		clusterGraph := gographviz.NewGraph()
		clusterGraph.SetName("cluster_LoopAgent")

		err := drawCluster(parentGraph, clusterGraph, loopAgent, [][]string{}, visitedNodes)
		if err != nil {
			t.Fatalf("drawCluster failed: %v", err)
		}

		// Check if edges exist between sub-agents and back to the first
		if lookupEdge(t, parentGraph, "LoopSubAgent1", "LoopSubAgent2") == nil {
			t.Error("Edge between LoopSubAgent1 and LoopSubAgent2 not found")
		}
		if lookupEdge(t, parentGraph, "LoopSubAgent2", "LoopSubAgent1") == nil {
			t.Error("Edge between LoopSubAgent2 and LoopSubAgent1 not found")
		}
	})

	visitedNodes = make(map[string]bool)

	t.Run("parallel agent cluster", func(t *testing.T) {
		subAgent1 := newTestLLMAgent("ParSubAgent1", "", nil, nil)
		subAgent2 := newTestLLMAgent("ParSubAgent2", "", nil, nil)
		parAgent := newTestAgent(t, "ParAgent", "", agentinternal.TypeParallelAgent, []agent.Agent{subAgent1, subAgent2})

		clusterGraph := gographviz.NewGraph()
		clusterGraph.SetName("cluster_ParAgent")

		err := drawCluster(parentGraph, clusterGraph, parAgent, [][]string{}, visitedNodes)
		if err != nil {
			t.Fatalf("drawCluster failed: %v", err)
		}

		// Check that no edges exist between parallel sub-agents
		if lookupEdge(t, parentGraph, "ParSubAgent1", "ParSubAgent2") != nil || lookupEdge(t, parentGraph, "ParSubAgent2", "ParSubAgent1") != nil {
			t.Error("Unexpected edge found between parallel sub-agents")
		}
	})
}

func TestBuildGraph(t *testing.T) {
	graph := gographviz.NewGraph()
	graph.SetName("G")
	parentGraph := graph
	visitedNodes := make(map[string]bool)

	tool1 := &mockTool{name: "Tool1"}
	tool2 := &mockTool{name: "Tool2"}

	subAgent1 := newTestLLMAgent("SubAgent1", "", []tool.Tool{tool1}, nil)
	subAgent2 := newTestLLMAgent("SubAgent2", "", nil, nil)
	mainAgent := newTestLLMAgent("MainAgent", "", []tool.Tool{tool2}, []agent.Agent{subAgent1, subAgent2})

	err := buildGraph(graph, parentGraph, mainAgent, [][]string{}, visitedNodes)
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
