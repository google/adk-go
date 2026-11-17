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

package agent_test

import (
	"context"
	"iter"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"sync"
	"testing"

	"github.com/gorilla/websocket"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/agent/workflowagent"
	"google.golang.org/adk/v2/internal/testutil"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/agenttool"
	"google.golang.org/adk/v2/tool/functiontool"
	"google.golang.org/adk/v2/workflow"
)

// agentNames records the name of the agent each tool context reports.
type agentNames struct {
	mu    sync.Mutex
	names []string
}

func (r *agentNames) record(ctx agent.Context) {
	name := "<nil>"
	if a := ctx.Agent(); a != nil {
		name = a.Name()
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	r.names = append(r.names, name)
}

func (r *agentNames) get() []string {
	r.mu.Lock()
	defer r.mu.Unlock()
	return slices.Clone(r.names)
}

// newRecordAgentTool returns a function tool named record_agent that records
// ctx.Agent() of each call.
func newRecordAgentTool(t *testing.T, names *agentNames) tool.Tool {
	t.Helper()
	tl, err := functiontool.New(functiontool.Config{Name: "record_agent", Description: "Records the calling agent."},
		func(ctx agent.Context, _ struct{}) (map[string]any, error) {
			names.record(ctx)
			return map[string]any{"ok": true}, nil
		})
	if err != nil {
		t.Fatalf("functiontool.New() error = %v", err)
	}
	return tl
}

func newTestLLMAgent(t *testing.T, cfg llmagent.Config) agent.Agent {
	t.Helper()
	a, err := llmagent.New(cfg)
	if err != nil {
		t.Fatalf("llmagent.New(%q) error = %v", cfg.Name, err)
	}
	return a
}

func callContent(name string, args map[string]any) *genai.Content {
	return genai.NewContentFromFunctionCall(name, args, genai.RoleModel)
}

func runToEnd(t *testing.T, a agent.Agent) {
	t.Helper()
	r := testutil.NewTestAgentRunner(t, a)
	for _, err := range r.Run(t, "s1", "go") {
		if err != nil {
			t.Fatalf("Run() error = %v", err)
		}
	}
}

// A tool called by a sub-agent after a transfer sees the sub-agent, not the
// runner's root. Fails if the tool context's Agent returns nil or the agent
// of an enclosing invocation.
func TestToolContextAgent_ToolCallReportsTheAgentWhoseTurnCallsIt(t *testing.T) {
	var names agentNames
	helper := newTestLLMAgent(t, llmagent.Config{
		Name: "helper",
		Model: &testutil.MockModel{Responses: []*genai.Content{
			callContent("record_agent", nil),
			genai.NewContentFromText("done", genai.RoleModel),
		}},
		Tools: []tool.Tool{newRecordAgentTool(t, &names)},
	})
	root := newTestLLMAgent(t, llmagent.Config{
		Name: "coordinator",
		Model: &testutil.MockModel{Responses: []*genai.Content{
			callContent("transfer_to_agent", map[string]any{"agent_name": "helper"}),
		}},
		SubAgents: []agent.Agent{helper},
	})

	runToEnd(t, root)

	if got, want := names.get(), []string{"helper"}; !slices.Equal(got, want) {
		t.Errorf("ctx.Agent().Name() in the tool = %q, want %q", got, want)
	}
}

// recordingToolset lists one recordingTool and records ctx.Agent() in its
// own request hook.
type recordingToolset struct {
	names *agentNames
}

func (s *recordingToolset) Name() string { return "recording_toolset" }

func (s *recordingToolset) Tools(agent.ReadonlyContext) ([]tool.Tool, error) {
	return []tool.Tool{&recordingTool{names: s.names}}, nil
}

func (s *recordingToolset) ProcessRequest(ctx agent.Context, _ *model.LLMRequest) error {
	s.names.record(ctx)
	return nil
}

// recordingTool records ctx.Agent() in its request hook and declares nothing.
type recordingTool struct {
	names *agentNames
}

func (t *recordingTool) Name() string        { return "recording_tool" }
func (t *recordingTool) Description() string { return "" }
func (t *recordingTool) IsLongRunning() bool { return false }

func (t *recordingTool) ProcessRequest(ctx agent.Context, _ *model.LLMRequest) error {
	t.names.record(ctx)
	return nil
}

// The request hooks of a toolset and of the tool it lists see the agent whose
// request they prepare. Fails if the tool context the flow builds for a hook
// returns nil from Agent.
func TestToolContextAgent_RequestHooksReportTheAgentBuildingTheRequest(t *testing.T) {
	var names agentNames
	a := newTestLLMAgent(t, llmagent.Config{
		Name:     "planner_agent",
		Model:    &testutil.MockModel{Responses: []*genai.Content{genai.NewContentFromText("done", genai.RoleModel)}},
		Toolsets: []tool.Toolset{&recordingToolset{names: &names}},
	})

	runToEnd(t, a)

	// The toolset's hook runs first, then the hook of the tool it listed.
	if got, want := names.get(), []string{"planner_agent", "planner_agent"}; !slices.Equal(got, want) {
		t.Errorf("ctx.Agent().Name() in the request hooks = %q, want %q", got, want)
	}
}

// A tool run as a workflow node sees the workflow agent. Fails if the tool
// context ToolNode builds returns nil from Agent.
func TestToolContextAgent_ToolNodeReportsTheWorkflowAgent(t *testing.T) {
	var names agentNames
	node, err := workflow.NewToolNode(newRecordAgentTool(t, &names), workflow.NodeConfig{})
	if err != nil {
		t.Fatalf("workflow.NewToolNode() error = %v", err)
	}
	// The tool node's input becomes the call's arguments, so a function node
	// turns the user's text into an empty argument map first.
	noArgs := workflow.NewFunctionNode("no_args", func(agent.Context, any) (map[string]any, error) {
		return map[string]any{}, nil
	}, workflow.NodeConfig{})
	wf, err := workflowagent.New(workflowagent.Config{
		Name:  "pipeline",
		Edges: workflow.Chain(workflow.Start, noArgs, node),
	})
	if err != nil {
		t.Fatalf("workflowagent.New() error = %v", err)
	}

	runToEnd(t, wf)

	if got, want := names.get(), []string{"pipeline"}; !slices.Equal(got, want) {
		t.Errorf("ctx.Agent().Name() in the tool node = %q, want %q", got, want)
	}
}

// A tool called inside an agent tool's run sees the wrapped agent, not the
// agent that called the agent tool. Fails if Agent returns nil or the outer
// agent.
func TestToolContextAgent_AgentToolInnerRunReportsTheInnerAgent(t *testing.T) {
	var names agentNames
	inner := newTestLLMAgent(t, llmagent.Config{
		Name:        "researcher",
		Description: "Researches.",
		Model: &testutil.MockModel{Responses: []*genai.Content{
			callContent("record_agent", nil),
			genai.NewContentFromText("found it", genai.RoleModel),
		}},
		Tools: []tool.Tool{newRecordAgentTool(t, &names)},
	})
	outer := newTestLLMAgent(t, llmagent.Config{
		Name: "assistant",
		Model: &testutil.MockModel{Responses: []*genai.Content{
			callContent("researcher", map[string]any{"request": "look it up"}),
			genai.NewContentFromText("done", genai.RoleModel),
		}},
		Tools: []tool.Tool{agenttool.New(inner, nil)},
	})

	runToEnd(t, outer)

	if got, want := names.get(), []string{"researcher"}; !slices.Equal(got, want) {
		t.Errorf("ctx.Agent().Name() in the inner agent's tool = %q, want %q", got, want)
	}
}

// liveModel is a model.LLM whose live connection goes to a fake Gemini Live
// server; RunLive type-asserts on Client.
type liveModel struct{ client *genai.Client }

func (m *liveModel) Name() string { return "fake-live-model" }

func (m *liveModel) GenerateContent(context.Context, *model.LLMRequest, bool) iter.Seq2[*model.LLMResponse, error] {
	return func(func(*model.LLMResponse, error) bool) {}
}

func (m *liveModel) Client() *genai.Client { return m.client }

// newFakeLiveClient returns a genai client whose live connection completes
// the setup, sends toolCall, and then waits for the client to close.
func newFakeLiveClient(t *testing.T, toolCall string) *genai.Client {
	t.Helper()
	var upgrader websocket.Upgrader
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			t.Errorf("websocket upgrade: %v", err)
			return
		}
		defer func() { _ = conn.Close() }()
		if _, _, err := conn.ReadMessage(); err != nil {
			return
		}
		if err := conn.WriteMessage(websocket.TextMessage, []byte(`{"setupComplete":{}}`)); err != nil {
			return
		}
		if err := conn.WriteMessage(websocket.TextMessage, []byte(toolCall)); err != nil {
			return
		}
		for {
			if _, _, err := conn.ReadMessage(); err != nil {
				return
			}
		}
	}))
	t.Cleanup(ts.Close)
	client, err := genai.NewClient(t.Context(), &genai.ClientConfig{
		Backend:     genai.BackendGeminiAPI,
		APIKey:      "test-api-key",
		HTTPOptions: genai.HTTPOptions{BaseURL: strings.Replace(ts.URL, "http", "ws", 1)},
	})
	if err != nil {
		t.Fatalf("genai.NewClient() error = %v", err)
	}
	return client
}

// A tool called in a live run sees the agent running live. Fails if the tool
// context of a live tool call returns nil from Agent.
func TestToolContextAgent_LiveToolCallReportsTheLiveAgent(t *testing.T) {
	var names agentNames
	client := newFakeLiveClient(t, `{"toolCall":{"functionCalls":[{"id":"call-1","name":"record_agent","args":{}}]}}`)
	a := newTestLLMAgent(t, llmagent.Config{
		Name:  "voice_agent",
		Model: &liveModel{client: client},
		Tools: []tool.Tool{newRecordAgentTool(t, &names)},
	})
	r, err := runner.New(runner.Config{AppName: "app", Agent: a, SessionService: session.InMemoryService(), AutoCreateSession: true})
	if err != nil {
		t.Fatalf("runner.New() error = %v", err)
	}

	liveSess, events, err := r.RunLive(t.Context(), "u1", "s1", agent.LiveRunConfig{})
	if err != nil {
		t.Fatalf("RunLive() error = %v", err)
	}
	for ev, err := range events {
		if err != nil {
			t.Fatalf("live event error = %v", err)
		}
		if ev.Content != nil && slices.ContainsFunc(ev.Content.Parts, func(p *genai.Part) bool { return p.FunctionResponse != nil }) {
			break
		}
	}
	if err := liveSess.Close(); err != nil {
		t.Errorf("Close() error = %v", err)
	}

	if got, want := names.get(), []string{"voice_agent"}; !slices.Equal(got, want) {
		t.Errorf("ctx.Agent().Name() in the live tool = %q, want %q", got, want)
	}
}
