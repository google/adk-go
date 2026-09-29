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

package appinfo

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"log"
	"maps"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/modelcontextprotocol/go-sdk/mcp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/agent/workflowagent"
	"google.golang.org/adk/v2/agent/workflowagents/sequentialagent"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/agenttool"
	"google.golang.org/adk/v2/tool/functiontool"
	"google.golang.org/adk/v2/tool/geminitool"
	"google.golang.org/adk/v2/tool/mcptoolset"
	"google.golang.org/adk/v2/workflow"
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

// computeInstruction is a named InstructionProvider, so a test can assert on
// the name app-info reports for it.
func computeInstruction(ctx agent.ReadonlyContext) (string, error) {
	return "resolved at run time", nil
}

// newWorkflowAgent builds a workflow agent whose graph runs the given agents in
// a chain. Their agents live in the workflow's edges, not in its SubAgents.
func newWorkflowAgent(t *testing.T, name, description string, agents ...agent.Agent) agent.Agent {
	t.Helper()
	var edges []workflow.Edge
	from := workflow.Start
	for _, a := range agents {
		node, err := workflow.NewAgentNode(a, workflow.NodeConfig{})
		if err != nil {
			t.Fatalf("workflow.NewAgentNode(%q) failed: %v", a.Name(), err)
		}
		edges = append(edges, workflow.Edge{From: from, To: node})
		from = node
	}
	wa, err := workflowagent.New(workflowagent.Config{
		Name:        name,
		Description: description,
		Edges:       edges,
	})
	if err != nil {
		t.Fatalf("workflowagent.New(%q) failed: %v", name, err)
	}
	return wa
}

// newSequentialAgent builds an agent that is not an LLM agent, to sit between
// LLM agents in the trees below.
func newSequentialAgent(t *testing.T, name, description string, subAgents ...agent.Agent) agent.Agent {
	t.Helper()
	a, err := sequentialagent.New(sequentialagent.Config{
		AgentConfig: agent.Config{
			Name:        name,
			Description: description,
			SubAgents:   subAgents,
		},
	})
	if err != nil {
		t.Fatalf("sequentialagent.New(%q) failed: %v", name, err)
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

// captureLog sends the standard logger to a buffer for the rest of the test.
func captureLog(t *testing.T) *bytes.Buffer {
	t.Helper()
	var buf bytes.Buffer
	prevOut, prevFlags := log.Writer(), log.Flags()
	log.SetOutput(&buf)
	log.SetFlags(0)
	t.Cleanup(func() {
		log.SetOutput(prevOut)
		log.SetFlags(prevFlags)
	})
	return &buf
}

// deadlineToolset records the context deadline each agent's toolset is
// resolved under, so a test can tell a budget shared by the whole walk from one
// handed out afresh per agent.
type deadlineToolset struct {
	name string

	mu        sync.Mutex
	deadlines []time.Time
}

func (d *deadlineToolset) Name() string { return d.name }

func (d *deadlineToolset) Tools(ctx agent.ReadonlyContext) ([]tool.Tool, error) {
	deadline, ok := ctx.Deadline()
	if !ok {
		return nil, errors.New("toolset resolved with no deadline")
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	d.deadlines = append(d.deadlines, deadline)
	return nil, nil
}

// fakeToolset returns a fixed set of tools, or a fixed error, without touching
// the network.
type fakeToolset struct {
	name  string
	tools []tool.Tool
	err   error
}

func (f *fakeToolset) Name() string { return f.name }

func (f *fakeToolset) Tools(ctx agent.ReadonlyContext) ([]tool.Tool, error) {
	return f.tools, f.err
}

func TestBuild(t *testing.T) {
	tests := []struct {
		name string
		// root builds the agent tree under test.
		root func(t *testing.T) agent.Agent
		// wantAgents is the expected set of keys in the returned map.
		wantAgents []string
		// check makes assertions beyond the set of agent names.
		check func(t *testing.T, agents map[string]*AgentInfo)
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
			check: func(t *testing.T, agents map[string]*AgentInfo) {
				got := agents["assistant"]
				if got.Description != "A plain assistant." {
					t.Errorf("Description = %q, want %q", got.Description, "A plain assistant.")
				}
				if got.Instruction != "Answer briefly." {
					t.Errorf("Instruction = %q, want %q", got.Instruction, "Answer briefly.")
				}
				if got.Tools == nil || len(got.Tools) != 0 {
					t.Errorf("Tools = %#v, want an empty, non-nil slice (a nil slice marshals to null)", got.Tools)
				}
				if got.SubAgents == nil || len(got.SubAgents) != 0 {
					t.Errorf("SubAgents = %#v, want an empty, non-nil slice (a nil slice marshals to null)", got.SubAgents)
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
			check: func(t *testing.T, agents map[string]*AgentInfo) {
				if diff := cmp.Diff([]string{"get_weather"}, toolNames(agents["assistant"].Tools)); diff != "" {
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
			check: func(t *testing.T, agents map[string]*AgentInfo) {
				if got := len(agents["searcher"].Tools); got != 1 {
					t.Errorf("len(Tools) = %d, want 1", got)
				}
				if diff := cmp.Diff([]string{"get_weather"}, toolNames(agents["searcher"].Tools)); diff != "" {
					t.Errorf("tool names mismatch (-want +got):\n%s", diff)
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
			check: func(t *testing.T, agents map[string]*AgentInfo) {
				want := []string{"list_tables", "ping", "run_query"}
				if diff := cmp.Diff(want, toolNames(agents["db"].Tools)); diff != "" {
					t.Errorf("tool names mismatch (-want +got):\n%s", diff)
				}
			},
		},
		{
			name: "a toolset that fails, and a nil one, are skipped and the agent still described",
			root: func(t *testing.T) agent.Agent {
				return newLLMAgent(t, llmagent.Config{
					Name:        "db",
					Description: "Talks to a database.",
					Instruction: "Query.",
					Tools:       []tool.Tool{newWeatherTool(t, "ping")},
					Toolsets: []tool.Toolset{
						&fakeToolset{name: "broken", err: errors.New("server unreachable")},
						nil,
						&fakeToolset{name: "working", tools: []tool.Tool{newWeatherTool(t, "run_query")}},
					},
				})
			},
			wantAgents: []string{"db"},
			check: func(t *testing.T, agents map[string]*AgentInfo) {
				if diff := cmp.Diff([]string{"ping", "run_query"}, toolNames(agents["db"].Tools)); diff != "" {
					t.Errorf("tool names mismatch (-want +got):\n%s", diff)
				}
			},
		},
		{
			name: "LLM sub-agents are described and listed by name",
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
			check: func(t *testing.T, agents map[string]*AgentInfo) {
				if diff := cmp.Diff([]string{"hotel"}, agents["concierge"].SubAgents); diff != "" {
					t.Errorf("concierge sub-agents mismatch (-want +got):\n%s", diff)
				}
				if diff := cmp.Diff([]string{"currency"}, agents["hotel"].SubAgents); diff != "" {
					t.Errorf("hotel sub-agents mismatch (-want +got):\n%s", diff)
				}
			},
		},
		{
			// adk-python stops at the SequentialAgent, so writer goes missing.
			// It is described here, but root does not list it: root hands off
			// to pipeline, and pipeline runs writer.
			name: "an LLM agent below a non-LLM agent is described but not listed by the LLM agent above",
			root: func(t *testing.T) agent.Agent {
				child := newLLMAgent(t, llmagent.Config{
					Name:        "writer",
					Description: "Writes a draft.",
					Instruction: "Write.",
					Tools:       []tool.Tool{newWeatherTool(t, "draft")},
				})
				pipeline := newSequentialAgent(t, "pipeline", "Runs steps in order.", child)
				return newLLMAgent(t, llmagent.Config{
					Name:        "root",
					Description: "Root agent.",
					Instruction: "Delegate.",
					SubAgents:   []agent.Agent{pipeline},
				})
			},
			wantAgents: []string{"root", "writer"},
			check: func(t *testing.T, agents map[string]*AgentInfo) {
				if got := agents["root"].SubAgents; len(got) != 0 {
					t.Errorf("root SubAgents = %v, want empty; pipeline is not an LLM agent", got)
				}
				if diff := cmp.Diff([]string{"draft"}, toolNames(agents["writer"].Tools)); diff != "" {
					t.Errorf("writer tool names mismatch (-want +got):\n%s", diff)
				}
			},
		},
		{
			name: "a chain of non-LLM agents is walked through",
			root: func(t *testing.T) agent.Agent {
				leaf := newLLMAgent(t, llmagent.Config{
					Name:        "leaf",
					Description: "Does the work.",
					Instruction: "Work.",
				})
				inner := newSequentialAgent(t, "inner", "Inner pipeline.", leaf)
				outer := newSequentialAgent(t, "outer", "Outer pipeline.", inner)
				return newLLMAgent(t, llmagent.Config{
					Name:        "root",
					Description: "Root agent.",
					Instruction: "Delegate.",
					SubAgents:   []agent.Agent{outer},
				})
			},
			wantAgents: []string{"leaf", "root"},
		},
		{
			// adk-python answers 400 "Root agent is not an LlmAgent" here
			// and describes nothing.
			name: "a non-LLM root still yields the LLM agents below it",
			root: func(t *testing.T) agent.Agent {
				first := newLLMAgent(t, llmagent.Config{
					Name:        "drafter",
					Description: "Drafts.",
					Instruction: "Draft.",
				})
				second := newLLMAgent(t, llmagent.Config{
					Name:        "editor",
					Description: "Edits.",
					Instruction: "Edit.",
				})
				return newSequentialAgent(t, "pipeline", "Drafts, then edits.", first, second)
			},
			wantAgents: []string{"drafter", "editor"},
		},
		{
			name: "an agent reachable through two branches is described once",
			root: func(t *testing.T) agent.Agent {
				shared := newLLMAgent(t, llmagent.Config{
					Name:        "shared",
					Description: "Reachable from two branches.",
					Instruction: "Help.",
				})
				left := newSequentialAgent(t, "left", "Left branch.", shared)
				right := newSequentialAgent(t, "right", "Right branch.", shared)
				return newLLMAgent(t, llmagent.Config{
					Name:        "root",
					Description: "Root agent.",
					Instruction: "Delegate.",
					SubAgents:   []agent.Agent{left, right},
				})
			},
			wantAgents: []string{"root", "shared"},
		},
		{
			// A nil sub-agent is a caller's mistake, but the constructors
			// accept one, and dereferencing it here would 500 the whole app.
			name: "a nil sub-agent is skipped",
			root: func(t *testing.T) agent.Agent {
				child := newLLMAgent(t, llmagent.Config{
					Name:        "child",
					Description: "The real sub-agent.",
					Instruction: "Help.",
				})
				return newLLMAgent(t, llmagent.Config{
					Name:        "root",
					Description: "Root agent.",
					Instruction: "Delegate.",
					SubAgents:   []agent.Agent{nil, child},
				})
			},
			wantAgents: []string{"child", "root"},
			check: func(t *testing.T, agents map[string]*AgentInfo) {
				if diff := cmp.Diff([]string{"child"}, agents["root"].SubAgents); diff != "" {
					t.Errorf("root sub-agents mismatch (-want +got):\n%s", diff)
				}
			},
		},
		{
			name: "an instruction template is reported as written",
			root: func(t *testing.T) agent.Agent {
				return newLLMAgent(t, llmagent.Config{
					Name:        "templated",
					Description: "Greets the user by name.",
					Instruction: "You are helping {user_name}. Reply in {language?}.",
				})
			},
			wantAgents: []string{"templated"},
			check: func(t *testing.T, agents map[string]*AgentInfo) {
				want := "You are helping {user_name}. Reply in {language?}."
				if got := agents["templated"].Instruction; got != want {
					t.Errorf("Instruction = %q, want %q", got, want)
				}
			},
		},
		{
			name: "an instruction provider is named, not resolved",
			root: func(t *testing.T) agent.Agent {
				return newLLMAgent(t, llmagent.Config{
					Name:                "dynamic",
					Description:         "Builds its instruction at run time.",
					InstructionProvider: computeInstruction,
				})
			},
			wantAgents: []string{"dynamic"},
			check: func(t *testing.T, agents map[string]*AgentInfo) {
				want := "<InstructionProvider: appinfo.computeInstruction>"
				if got := agents["dynamic"].Instruction; got != want {
					t.Errorf("Instruction = %q, want %q", got, want)
				}
			},
		},
		{
			// The provider takes over from Instruction when the agent runs, so
			// the static text is never sent to the model.
			name: "an instruction provider wins over a static instruction",
			root: func(t *testing.T) agent.Agent {
				return newLLMAgent(t, llmagent.Config{
					Name:                "both",
					Description:         "Sets both.",
					Instruction:         "Never sent to the model.",
					InstructionProvider: computeInstruction,
				})
			},
			wantAgents: []string{"both"},
			check: func(t *testing.T, agents map[string]*AgentInfo) {
				want := "<InstructionProvider: appinfo.computeInstruction>"
				if got := agents["both"].Instruction; got != want {
					t.Errorf("Instruction = %q, want %q", got, want)
				}
			},
		},
		{
			// A workflow agent keeps the agents of its graph in its edges, not
			// in SubAgents, so following SubAgents alone finds none of them.
			name: "agents in a workflow graph are described",
			root: func(t *testing.T) agent.Agent {
				return newWorkflowAgent(t, "graph_root", "Runs a graph.",
					newLLMAgent(t, llmagent.Config{
						Name:        "researcher",
						Description: "Researches.",
						Instruction: "Research.",
						Tools:       []tool.Tool{newWeatherTool(t, "lookup")},
					}),
					newLLMAgent(t, llmagent.Config{
						Name:        "summarizer",
						Description: "Summarizes.",
						Instruction: "Summarize.",
					}),
				)
			},
			wantAgents: []string{"researcher", "summarizer"},
			check: func(t *testing.T, agents map[string]*AgentInfo) {
				if diff := cmp.Diff([]string{"lookup"}, toolNames(agents["researcher"].Tools)); diff != "" {
					t.Errorf("researcher tool names mismatch (-want +got):\n%s", diff)
				}
			},
		},
		{
			// A graph node can itself be a graph, so the walk has to descend
			// into a sub-workflow rather than stopping at the node holding it.
			name: "agents in a nested sub-workflow are described",
			root: func(t *testing.T) agent.Agent {
				buried := newLLMAgent(t, llmagent.Config{
					Name:        "buried",
					Description: "Runs inside a nested graph.",
					Instruction: "Work.",
				})
				buriedNode, err := workflow.NewAgentNode(buried, workflow.NodeConfig{})
				if err != nil {
					t.Fatalf("workflow.NewAgentNode failed: %v", err)
				}
				inner, err := workflow.NewWorkflowNode("inner_graph",
					[]workflow.Edge{{From: workflow.Start, To: buriedNode}})
				if err != nil {
					t.Fatalf("workflow.NewWorkflowNode failed: %v", err)
				}
				root, err := workflowagent.New(workflowagent.Config{
					Name:        "outer_graph",
					Description: "Runs a graph that contains a graph.",
					Edges:       []workflow.Edge{{From: workflow.Start, To: inner}},
				})
				if err != nil {
					t.Fatalf("workflowagent.New failed: %v", err)
				}
				return root
			},
			wantAgents: []string{"buried"},
		},
		{
			name: "an LLM agent above a workflow agent does not list the graph's agents",
			root: func(t *testing.T) agent.Agent {
				graph := newWorkflowAgent(t, "graph", "Runs a graph.",
					newLLMAgent(t, llmagent.Config{
						Name:        "worker",
						Description: "Works.",
						Instruction: "Work.",
					}),
				)
				return newLLMAgent(t, llmagent.Config{
					Name:        "root",
					Description: "Root agent.",
					Instruction: "Delegate.",
					SubAgents:   []agent.Agent{graph},
				})
			},
			wantAgents: []string{"root", "worker"},
			check: func(t *testing.T, agents map[string]*AgentInfo) {
				if got := agents["root"].SubAgents; len(got) != 0 {
					t.Errorf("root SubAgents = %v, want empty; graph is not an LLM agent", got)
				}
			},
		},
		{
			// An agent tool runs its agent under its own runner and session, so
			// none of that agent's events reach the stream and none is ever
			// attributed to it. It is still reported as a tool on its caller.
			name: "an agent used as a tool is reported as a tool, not as an agent",
			root: func(t *testing.T) agent.Agent {
				wrapped := newLLMAgent(t, llmagent.Config{
					Name:        "translator",
					Description: "Translates text.",
					Instruction: "Translate.",
					Tools:       []tool.Tool{newWeatherTool(t, "lookup_phrase")},
				})
				return newLLMAgent(t, llmagent.Config{
					Name:        "root",
					Description: "Root agent.",
					Instruction: "Call the translator.",
					Tools:       []tool.Tool{agenttool.New(wrapped, nil)},
				})
			},
			wantAgents: []string{"root"},
			check: func(t *testing.T, agents map[string]*AgentInfo) {
				if got := agents["root"].SubAgents; len(got) != 0 {
					t.Errorf("root SubAgents = %v, want empty; an agent tool is not a sub-agent", got)
				}
				if diff := cmp.Diff([]string{"translator"}, toolNames(agents["root"].Tools)); diff != "" {
					t.Errorf("root tool names mismatch (-want +got):\n%s", diff)
				}
			},
		},
		{
			// Which agents a dynamic node runs is decided by Go code at run
			// time, so there is nothing to walk before it runs.
			name: "an agent run by a dynamic node is not found",
			root: func(t *testing.T) agent.Agent {
				hidden, err := workflow.NewAgentNode(newLLMAgent(t, llmagent.Config{
					Name:        "hidden",
					Description: "Run from Go code.",
					Instruction: "Work.",
				}), workflow.NodeConfig{})
				if err != nil {
					t.Fatalf("workflow.NewAgentNode failed: %v", err)
				}
				dynamic := workflow.NewDynamicNode("orchestrator",
					func(ctx agent.Context, in string, _ func(*session.Event) error) (string, error) {
						return workflow.RunNode[string](ctx, hidden, in)
					}, workflow.NodeConfig{})
				root, err := workflowagent.New(workflowagent.Config{
					Name:        "dynamic_graph",
					Description: "Runs a dynamic node.",
					Edges:       workflow.Chain(workflow.Start, dynamic),
				})
				if err != nil {
					t.Fatalf("workflowagent.New failed: %v", err)
				}
				return root
			},
			wantAgents: nil,
		},
		{
			// Documented as not found: the walk does not unwrap a
			// ParallelWorker, which keeps the node it runs private.
			name: "an agent wrapped in a ParallelWorker is not found",
			root: func(t *testing.T) agent.Agent {
				wrapped, err := workflow.NewAgentNode(newLLMAgent(t, llmagent.Config{
					Name:        "per_item",
					Description: "Runs once per input item.",
					Instruction: "Work.",
				}), workflow.NodeConfig{})
				if err != nil {
					t.Fatalf("workflow.NewAgentNode failed: %v", err)
				}
				worker, err := workflow.NewParallelWorker("fan_out", wrapped, 0, workflow.NodeConfig{})
				if err != nil {
					t.Fatalf("workflow.NewParallelWorker failed: %v", err)
				}
				root, err := workflowagent.New(workflowagent.Config{
					Name:        "parallel_graph",
					Description: "Fans out over its input.",
					Edges:       workflow.Chain(workflow.Start, worker),
				})
				if err != nil {
					t.Fatalf("workflowagent.New failed: %v", err)
				}
				return root
			},
			wantAgents: nil,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			agents := build(t.Context(), "test_app", tc.root(t)).Agents

			gotNames := slices.Sorted(maps.Keys(agents))
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

// TestBuildNameClash covers two different agents sharing a name. The response
// is keyed by name, so only one can be described: the first the walk reaches.
// The walk still descends into the other, so an agent below it is not lost.
func TestBuildNameClash(t *testing.T) {
	logs := captureLog(t)

	first := newLLMAgent(t, llmagent.Config{
		Name:        "helper",
		Description: "The first helper.",
		Instruction: "Help first.",
	})
	below := newLLMAgent(t, llmagent.Config{
		Name:        "below",
		Description: "Below the second helper.",
		Instruction: "Work.",
	})
	second := newLLMAgent(t, llmagent.Config{
		Name:        "helper",
		Description: "The second helper.",
		Instruction: "Help second.",
		SubAgents:   []agent.Agent{below},
	})
	root := newLLMAgent(t, llmagent.Config{
		Name:        "root",
		Description: "Root agent.",
		Instruction: "Delegate.",
		SubAgents: []agent.Agent{
			newSequentialAgent(t, "left", "Left branch.", first),
			newSequentialAgent(t, "right", "Right branch.", second),
		},
	})

	agents := build(t.Context(), "test_app", root).Agents

	if diff := cmp.Diff([]string{"below", "helper", "root"}, slices.Sorted(maps.Keys(agents))); diff != "" {
		t.Errorf("agent names mismatch (-want +got):\n%s", diff)
	}
	if got := agents["helper"].Description; got != "The first helper." {
		t.Errorf("helper Description = %q, want the first helper's", got)
	}
	if !strings.Contains(logs.String(), `two different agents are named "helper"`) {
		t.Errorf("log does not report the name clash; got:\n%s", logs.String())
	}
}

// TestBuildSharedAgentIsNotAClash covers the other side of the name clash: one
// agent reached along two paths is the same agent, so it is described once and
// no clash is logged.
func TestBuildSharedAgentIsNotAClash(t *testing.T) {
	logs := captureLog(t)

	shared := newLLMAgent(t, llmagent.Config{
		Name:        "shared",
		Description: "Reachable from two branches.",
		Instruction: "Help.",
	})
	root := newLLMAgent(t, llmagent.Config{
		Name:        "root",
		Description: "Root agent.",
		Instruction: "Delegate.",
		SubAgents: []agent.Agent{
			newSequentialAgent(t, "left", "Left branch.", shared),
			newSequentialAgent(t, "right", "Right branch.", shared),
		},
	})

	build(t.Context(), "test_app", root)

	if strings.Contains(logs.String(), "two different agents") {
		t.Errorf("a shared agent was logged as a name clash:\n%s", logs.String())
	}
}

// TestBuildFailingToolsetLogsNoErrorText pins that a toolset's error text stays
// out of the log. It can carry a server URL with a token in its query, and any
// caller who reaches the endpoint can make it be logged.
func TestBuildFailingToolsetLogsNoErrorText(t *testing.T) {
	logs := captureLog(t)

	root := newLLMAgent(t, llmagent.Config{
		Name:        "db",
		Description: "Talks to a database.",
		Instruction: "Query.",
		Toolsets: []tool.Toolset{&fakeToolset{
			name: "broken",
			err:  errors.New(`Get "https://mcp.example/?key=SECRET-TOKEN": connection refused`),
		}},
	})

	build(t.Context(), "test_app", root)

	got := logs.String()
	if !strings.Contains(got, `skipping toolset "broken"`) {
		t.Errorf("log does not report the skipped toolset:\n%s", got)
	}
	if strings.Contains(got, "SECRET-TOKEN") {
		t.Errorf("log contains the toolset's error text:\n%s", got)
	}
}

// TestBuildToolsetBudgetIsPerRequest pins that the toolset timeout is one
// budget for the whole walk. Handing each agent its own would multiply it by
// the number of agents holding a toolset, so a request could outlive the
// server's write timeout many times over while the client already saw a broken
// response.
func TestBuildToolsetBudgetIsPerRequest(t *testing.T) {
	first := &deadlineToolset{name: "first"}
	second := &deadlineToolset{name: "second"}

	child := newLLMAgent(t, llmagent.Config{
		Name:        "child",
		Description: "Holds the second toolset.",
		Instruction: "Help.",
		Toolsets:    []tool.Toolset{second},
	})
	root := newLLMAgent(t, llmagent.Config{
		Name:        "root",
		Description: "Holds the first toolset.",
		Instruction: "Delegate.",
		Toolsets:    []tool.Toolset{first},
		SubAgents:   []agent.Agent{child},
	})

	build(t.Context(), "test_app", root)

	if len(first.deadlines) != 1 || len(second.deadlines) != 1 {
		t.Fatalf("toolsets resolved %d and %d times, want 1 each",
			len(first.deadlines), len(second.deadlines))
	}
	// One context reaches both agents, so the deadline is the same instant and
	// not merely a close one: a second WithTimeout lands nanoseconds later.
	if !second.deadlines[0].Equal(first.deadlines[0]) {
		t.Errorf("deadlines differ by %v; each agent got its own budget, want one shared across the walk",
			second.deadlines[0].Sub(first.deadlines[0]))
	}
}

// TestBuildLoop covers an agent graph that is not a tree. The constructors
// cannot build one, because a sub-agent exists before its parent, but
// SubAgents returns the live slice, so a caller can close a loop afterwards.
// The walk must terminate and still reach every agent the loop leaves
// reachable.
//
//	root  -> [outer(Seq), sibling(LLM)]
//	outer -> [inner(Seq), leaf(LLM)]
//	inner -> outer                       (the loop)
//	sibling -> inner
func TestBuildLoop(t *testing.T) {
	leaf := newLLMAgent(t, llmagent.Config{
		Name:        "leaf",
		Description: "Does the work.",
		Instruction: "Work.",
	})
	placeholder := newLLMAgent(t, llmagent.Config{
		Name:        "placeholder",
		Description: "Replaced below to close the loop.",
		Instruction: "Unused.",
	})
	inner := newSequentialAgent(t, "inner", "Inner pipeline.", placeholder)
	outer := newSequentialAgent(t, "outer", "Outer pipeline.", inner, leaf)
	sibling := newLLMAgent(t, llmagent.Config{
		Name:        "sibling",
		Description: "Reaches inner from outside the loop.",
		Instruction: "Delegate.",
		SubAgents:   []agent.Agent{inner},
	})
	root := newLLMAgent(t, llmagent.Config{
		Name:        "root",
		Description: "Root agent.",
		Instruction: "Delegate.",
		SubAgents:   []agent.Agent{outer, sibling},
	})

	subAgents := inner.SubAgents()
	if len(subAgents) != 1 {
		t.Fatalf("len(inner.SubAgents()) = %d, want 1", len(subAgents))
	}
	subAgents[0] = outer

	agents := build(t.Context(), "test_app", root).Agents

	if diff := cmp.Diff([]string{"leaf", "root", "sibling"}, slices.Sorted(maps.Keys(agents))); diff != "" {
		t.Errorf("agent names mismatch (-want +got):\n%s", diff)
	}
}

// TestBuildAgentsAlwaysPresent covers an app with no LLM agent anywhere. The
// agents key is read on every response, so an empty map has to marshal to {}
// and not vanish, and isComputerUse is always emitted, as adk-python does.
func TestBuildAgentsAlwaysPresent(t *testing.T) {
	root, err := agent.New(agent.Config{
		Name:        "custom_root",
		Description: "A custom agent with no LLM agent below it.",
	})
	if err != nil {
		t.Fatalf("agent.New failed: %v", err)
	}

	body, err := json.Marshal(build(t.Context(), "test_app", root))
	if err != nil {
		t.Fatalf("json.Marshal failed: %v", err)
	}
	for _, want := range []string{`"agents":{}`, `"isComputerUse":false`, `"language":"go"`} {
		if !strings.Contains(string(body), want) {
			t.Errorf("response has no %s; body: %s", want, body)
		}
	}
}

// TestBuildNonLLMRootIsNamedButNotDescribed pins the one place the agents map
// and RootAgentName disagree: the root is always named, and it is only
// described when it is an LLM agent.
func TestBuildNonLLMRootIsNamedButNotDescribed(t *testing.T) {
	writer := newLLMAgent(t, llmagent.Config{
		Name:        "writer",
		Description: "Writes a draft.",
		Instruction: "Write.",
	})
	root := newSequentialAgent(t, "pipeline", "Runs steps in order.", writer)

	info := build(t.Context(), "test_app", root)
	if info.Name != "test_app" {
		t.Errorf("Name = %q, want %q", info.Name, "test_app")
	}
	if info.RootAgentName != "pipeline" {
		t.Errorf("RootAgentName = %q, want %q", info.RootAgentName, "pipeline")
	}
	if info.Description != "Runs steps in order." {
		t.Errorf("Description = %q, want %q", info.Description, "Runs steps in order.")
	}
	if _, ok := info.Agents["pipeline"]; ok {
		t.Error("agents contains the non-LLM root; only LLM agents are described")
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

// newMCPToolset serves one echo tool from an in-memory MCP server, so a test
// covers tools discovered over the protocol at request time while staying
// offline.
func newMCPToolset(t *testing.T) tool.Toolset {
	t.Helper()
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
	return ts
}

func TestBuildWithMCPToolset(t *testing.T) {
	root := newLLMAgent(t, llmagent.Config{
		Name:        "echo_agent",
		Description: "Echoes text.",
		Instruction: "Use the echo tool.",
		Tools:       []tool.Tool{newWeatherTool(t, "local_tool")},
		Toolsets:    []tool.Toolset{newMCPToolset(t)},
	})

	info := build(t.Context(), "mcp_app", root)

	if diff := cmp.Diff([]string{"echo", "local_tool"}, toolNames(info.Agents["echo_agent"].Tools)); diff != "" {
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

// TestBuildMCPToolsetContextCancelled covers a toolset that cannot be reached
// in time: the agent must still be described, minus its toolset's tools. It
// also pins that the request context reaches ts.Tools, so a client disconnect
// stops toolset resolution.
func TestBuildMCPToolsetContextCancelled(t *testing.T) {
	root := newLLMAgent(t, llmagent.Config{
		Name:        "echo_agent",
		Description: "Echoes text.",
		Instruction: "Use the echo tool.",
		Tools:       []tool.Tool{newWeatherTool(t, "local_tool")},
		Toolsets:    []tool.Toolset{newMCPToolset(t)},
	})

	ctx, cancel := context.WithCancel(t.Context())
	cancel()

	info := build(ctx, "mcp_app", root)
	if diff := cmp.Diff([]string{"local_tool"}, toolNames(info.Agents["echo_agent"].Tools)); diff != "" {
		t.Errorf("tool names mismatch (-want +got):\n%s", diff)
	}
}
