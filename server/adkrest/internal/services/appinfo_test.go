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

package services_test

import (
	"context"
	"slices"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/modelcontextprotocol/go-sdk/mcp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/agent/workflowagents/sequentialagent"
	"google.golang.org/adk/v2/server/adkrest/internal/models"
	"google.golang.org/adk/v2/server/adkrest/internal/services"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
	"google.golang.org/adk/v2/tool/geminitool"
	"google.golang.org/adk/v2/tool/mcptoolset"
)

type weatherArgs struct {
	City string `json:"city"`
}

type weatherResult struct {
	Temp int `json:"temp"`
}

func newWeatherTool(t *testing.T, name string) tool.Tool {
	t.Helper()
	ft, err := functiontool.New(functiontool.Config{
		Name:        name,
		Description: "Returns the current weather for a city.",
	}, func(ctx agent.Context, args weatherArgs) (weatherResult, error) {
		return weatherResult{Temp: 21}, nil
	})
	if err != nil {
		t.Fatalf("functiontool.New(%q) failed: %v", name, err)
	}
	return ft
}

func newLLMAgent(t *testing.T, cfg llmagent.Config) agent.Agent {
	t.Helper()
	a, err := llmagent.New(cfg)
	if err != nil {
		t.Fatalf("llmagent.New(%q) failed: %v", cfg.Name, err)
	}
	return a
}

// toolNames returns the function declaration names of tools, so a test can
// assert on tools without depending on the full schema.
func toolNames(tools []*genai.Tool) []string {
	names := make([]string, 0, len(tools))
	for _, t := range tools {
		for _, decl := range t.FunctionDeclarations {
			names = append(names, decl.Name)
		}
	}
	slices.Sort(names)
	return names
}

// fakeToolset returns a fixed set of tools without touching the network.
type fakeToolset struct {
	name  string
	tools []tool.Tool
}

func (f *fakeToolset) Name() string { return f.name }

func (f *fakeToolset) Tools(ctx agent.ReadonlyContext) ([]tool.Tool, error) {
	return f.tools, nil
}

func TestGetAppInfo(t *testing.T) {
	tests := []struct {
		name string
		// root builds the agent tree under test.
		root func(t *testing.T) agent.Agent
		// wantAgents is the expected set of keys in the returned map.
		wantAgents []string
		// check makes assertions beyond the set of agent names.
		check func(t *testing.T, agents map[string]*models.AgentInfo)
	}{
		{
			name: "single LLM agent without tools",
			root: func(t *testing.T) agent.Agent {
				return newLLMAgent(t, llmagent.Config{
					Name:        "assistant",
					Description: "A plain assistant.",
					Instruction: "Answer briefly.",
				})
			},
			wantAgents: []string{"assistant"},
			check: func(t *testing.T, agents map[string]*models.AgentInfo) {
				got := agents["assistant"]
				if got.Instruction != "Answer briefly." {
					t.Errorf("Instruction = %q, want %q", got.Instruction, "Answer briefly.")
				}
				if got.Tools == nil {
					t.Error("Tools = nil, want an empty slice (a nil slice marshals to null)")
				}
				if len(got.Tools) != 0 {
					t.Errorf("len(Tools) = %d, want 0", len(got.Tools))
				}
				if got.SubAgents != nil {
					t.Errorf("SubAgents = %v, want nil", got.SubAgents)
				}
			},
		},
		{
			name: "function tools are reported as declarations",
			root: func(t *testing.T) agent.Agent {
				return newLLMAgent(t, llmagent.Config{
					Name:        "assistant",
					Description: "Checks the weather.",
					Instruction: "Use the tool.",
					Tools:       []tool.Tool{newWeatherTool(t, "get_weather")},
				})
			},
			wantAgents: []string{"assistant"},
			check: func(t *testing.T, agents map[string]*models.AgentInfo) {
				got := toolNames(agents["assistant"].Tools)
				if diff := cmp.Diff([]string{"get_weather"}, got); diff != "" {
					t.Errorf("tool names mismatch (-want +got):\n%s", diff)
				}
				decl := agents["assistant"].Tools[0].FunctionDeclarations[0]
				if decl.Description != "Returns the current weather for a city." {
					t.Errorf("declaration description = %q", decl.Description)
				}
			},
		},
		{
			name: "built-in tools without a declaration are omitted",
			root: func(t *testing.T) agent.Agent {
				return newLLMAgent(t, llmagent.Config{
					Name:        "searcher",
					Description: "Searches the web.",
					Instruction: "Search.",
					Tools:       []tool.Tool{newWeatherTool(t, "get_weather"), geminitool.GoogleSearch{}},
				})
			},
			wantAgents: []string{"searcher"},
			check: func(t *testing.T, agents map[string]*models.AgentInfo) {
				if diff := cmp.Diff([]string{"get_weather"}, toolNames(agents["searcher"].Tools)); diff != "" {
					t.Errorf("tool names mismatch (-want +got):\n%s", diff)
				}
				if got := len(agents["searcher"].Tools); got != 1 {
					t.Errorf("len(Tools) = %d, want 1", got)
				}
			},
		},
		{
			name: "toolsets are expanded",
			root: func(t *testing.T) agent.Agent {
				return newLLMAgent(t, llmagent.Config{
					Name:        "db",
					Description: "Talks to a database.",
					Instruction: "Query.",
					Tools:       []tool.Tool{newWeatherTool(t, "ping")},
					Toolsets: []tool.Toolset{&fakeToolset{
						name:  "db_toolset",
						tools: []tool.Tool{newWeatherTool(t, "list_tables"), newWeatherTool(t, "run_query")},
					}},
				})
			},
			wantAgents: []string{"db"},
			check: func(t *testing.T, agents map[string]*models.AgentInfo) {
				want := []string{"list_tables", "ping", "run_query"}
				if diff := cmp.Diff(want, toolNames(agents["db"].Tools)); diff != "" {
					t.Errorf("tool names mismatch (-want +got):\n%s", diff)
				}
			},
		},
		{
			name: "nested agents are flattened and linked by name",
			root: func(t *testing.T) agent.Agent {
				deepest := newLLMAgent(t, llmagent.Config{
					Name:        "currency",
					Description: "Converts currencies.",
					Instruction: "Convert.",
					Tools:       []tool.Tool{newWeatherTool(t, "convert")},
				})
				middle := newLLMAgent(t, llmagent.Config{
					Name:        "hotel",
					Description: "Books hotels.",
					Instruction: "Book.",
					Tools:       []tool.Tool{newWeatherTool(t, "book")},
					SubAgents:   []agent.Agent{deepest},
				})
				return newLLMAgent(t, llmagent.Config{
					Name:        "concierge",
					Description: "Plans trips.",
					Instruction: "Coordinate.",
					SubAgents:   []agent.Agent{middle},
				})
			},
			wantAgents: []string{"concierge", "currency", "hotel"},
			check: func(t *testing.T, agents map[string]*models.AgentInfo) {
				if diff := cmp.Diff([]string{"hotel"}, agents["concierge"].SubAgents); diff != "" {
					t.Errorf("concierge sub-agents mismatch (-want +got):\n%s", diff)
				}
				if diff := cmp.Diff([]string{"currency"}, agents["hotel"].SubAgents); diff != "" {
					t.Errorf("hotel sub-agents mismatch (-want +got):\n%s", diff)
				}
			},
		},
		{
			name: "non-LLM sub-agents are included, along with their children",
			root: func(t *testing.T) agent.Agent {
				child := newLLMAgent(t, llmagent.Config{
					Name:        "writer",
					Description: "Writes a draft.",
					Instruction: "Write.",
					Tools:       []tool.Tool{newWeatherTool(t, "draft")},
				})
				pipeline, err := sequentialagent.New(sequentialagent.Config{
					AgentConfig: agent.Config{
						Name:        "pipeline",
						Description: "Runs steps in order.",
						SubAgents:   []agent.Agent{child},
					},
				})
				if err != nil {
					t.Fatalf("sequentialagent.New failed: %v", err)
				}
				return newLLMAgent(t, llmagent.Config{
					Name:        "root",
					Description: "Root agent.",
					Instruction: "Delegate.",
					SubAgents:   []agent.Agent{pipeline},
				})
			},
			wantAgents: []string{"pipeline", "root", "writer"},
			check: func(t *testing.T, agents map[string]*models.AgentInfo) {
				pipeline := agents["pipeline"]
				if pipeline.Instruction != "" {
					t.Errorf("pipeline Instruction = %q, want empty", pipeline.Instruction)
				}
				if pipeline.Tools == nil || len(pipeline.Tools) != 0 {
					t.Errorf("pipeline Tools = %v, want an empty non-nil slice", pipeline.Tools)
				}
				if diff := cmp.Diff([]string{"writer"}, pipeline.SubAgents); diff != "" {
					t.Errorf("pipeline sub-agents mismatch (-want +got):\n%s", diff)
				}
			},
		},
		{
			name: "an agent reachable twice is reported once",
			root: func(t *testing.T) agent.Agent {
				shared := newLLMAgent(t, llmagent.Config{
					Name:        "shared",
					Description: "Reachable from two parents.",
					Instruction: "Help.",
				})
				left := newLLMAgent(t, llmagent.Config{
					Name:        "left",
					Description: "Left branch.",
					Instruction: "Left.",
					SubAgents:   []agent.Agent{shared},
				})
				return newLLMAgent(t, llmagent.Config{
					Name:        "root",
					Description: "Root agent.",
					Instruction: "Delegate.",
					SubAgents:   []agent.Agent{left},
				})
			},
			wantAgents: []string{"left", "root", "shared"},
		},
		{
			name: "an instruction provider cannot be resolved statically",
			root: func(t *testing.T) agent.Agent {
				return newLLMAgent(t, llmagent.Config{
					Name:        "dynamic",
					Description: "Builds its instruction at run time.",
					InstructionProvider: func(ctx agent.ReadonlyContext) (string, error) {
						return "resolved at run time", nil
					},
				})
			},
			wantAgents: []string{"dynamic"},
			check: func(t *testing.T, agents map[string]*models.AgentInfo) {
				if got := agents["dynamic"].Instruction; got != "" {
					t.Errorf("Instruction = %q, want empty", got)
				}
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			agents := services.GetAppInfo(context.Background(), "test_app", tc.root(t)).Agents

			gotNames := make([]string, 0, len(agents))
			for name := range agents {
				gotNames = append(gotNames, name)
			}
			slices.Sort(gotNames)
			if diff := cmp.Diff(tc.wantAgents, gotNames); diff != "" {
				t.Errorf("agent names mismatch (-want +got):\n%s", diff)
			}

			for name, info := range agents {
				if info.Name != name {
					t.Errorf("agents[%q].Name = %q, want %q", name, info.Name, name)
				}
			}

			if tc.check != nil {
				tc.check(t, agents)
			}
		})
	}
}

func TestGetAppInfoNilRoot(t *testing.T) {
	if got := services.GetAppInfo(context.Background(), "test_app", nil); got != nil {
		t.Errorf("GetAppInfo(nil root) = %v, want nil", got)
	}
}

// mcpEchoInput is the argument struct of the tool served by the in-memory MCP
// server below.
type mcpEchoInput struct {
	Text string `json:"text" jsonschema:"the text to echo"`
}

type mcpEchoOutput struct {
	Echoed string `json:"echoed"`
}

func mcpEcho(ctx context.Context, req *mcp.CallToolRequest, in mcpEchoInput) (*mcp.CallToolResult, mcpEchoOutput, error) {
	return nil, mcpEchoOutput{Echoed: in.Text}, nil
}

// TestGetAppInfoWithMCPToolset covers a real remote-style toolset: tools are
// discovered over the MCP protocol at request time rather than being known
// statically. The server runs in memory, so the test stays offline.
func TestGetAppInfoWithMCPToolset(t *testing.T) {
	clientTransport, serverTransport := mcp.NewInMemoryTransports()

	server := mcp.NewServer(&mcp.Implementation{Name: "echo_server", Version: "v1.0.0"}, nil)
	mcp.AddTool(server, &mcp.Tool{Name: "echo", Description: "Echoes the given text."}, mcpEcho)
	if _, err := server.Connect(t.Context(), serverTransport, nil); err != nil {
		t.Fatalf("failed to connect MCP server: %v", err)
	}

	ts, err := mcptoolset.New(mcptoolset.Config{Transport: clientTransport})
	if err != nil {
		t.Fatalf("mcptoolset.New failed: %v", err)
	}

	root := newLLMAgent(t, llmagent.Config{
		Name:        "echo_agent",
		Description: "Echoes text.",
		Instruction: "Use the echo tool.",
		Tools:       []tool.Tool{newWeatherTool(t, "local_tool")},
		Toolsets:    []tool.Toolset{ts},
	})

	info := services.GetAppInfo(t.Context(), "mcp_app", root)
	if info == nil {
		t.Fatal("GetAppInfo returned nil")
	}

	got := toolNames(info.Agents["echo_agent"].Tools)
	if diff := cmp.Diff([]string{"echo", "local_tool"}, got); diff != "" {
		t.Errorf("tool names mismatch (-want +got):\n%s", diff)
	}

	// The declaration must survive the MCP round trip, schema included.
	for _, tl := range info.Agents["echo_agent"].Tools {
		decl := tl.FunctionDeclarations[0]
		if decl.Name != "echo" {
			continue
		}
		if decl.Description != "Echoes the given text." {
			t.Errorf("echo description = %q, want %q", decl.Description, "Echoes the given text.")
		}
		if decl.Parameters == nil && decl.ParametersJsonSchema == nil {
			t.Error("echo declaration has no parameter schema")
		}
	}
}

// TestGetAppInfoMCPToolsetContextCancelled covers a toolset that cannot be
// reached in time: the agent must still be described, minus its toolset's
// tools. It also pins the context propagation -- the request context reaches
// ts.Tools, so a client disconnect stops toolset resolution.
func TestGetAppInfoMCPToolsetContextCancelled(t *testing.T) {
	clientTransport, serverTransport := mcp.NewInMemoryTransports()

	server := mcp.NewServer(&mcp.Implementation{Name: "echo_server", Version: "v1.0.0"}, nil)
	mcp.AddTool(server, &mcp.Tool{Name: "echo", Description: "Echoes the given text."}, mcpEcho)
	if _, err := server.Connect(t.Context(), serverTransport, nil); err != nil {
		t.Fatalf("failed to connect MCP server: %v", err)
	}

	ts, err := mcptoolset.New(mcptoolset.Config{Transport: clientTransport})
	if err != nil {
		t.Fatalf("mcptoolset.New failed: %v", err)
	}

	root := newLLMAgent(t, llmagent.Config{
		Name:        "echo_agent",
		Description: "Echoes text.",
		Instruction: "Use the echo tool.",
		Tools:       []tool.Tool{newWeatherTool(t, "local_tool")},
		Toolsets:    []tool.Toolset{ts},
	})

	ctx, cancel := context.WithCancel(t.Context())
	cancel()

	info := services.GetAppInfo(ctx, "mcp_app", root)
	if info == nil {
		t.Fatal("GetAppInfo returned nil")
	}
	if diff := cmp.Diff([]string{"local_tool"}, toolNames(info.Agents["echo_agent"].Tools)); diff != "" {
		t.Errorf("tool names mismatch (-want +got):\n%s", diff)
	}
}
