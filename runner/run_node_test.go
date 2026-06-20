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

package runner_test

import (
	"context"
	"iter"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/agent/workflowagent"
	"google.golang.org/adk/model"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/functiontool"
	"google.golang.org/adk/workflow"
)

const (
	nodeTestApp     = "node_test_app"
	nodeTestUser    = "u"
	nodeTestSession = "s"
)

// TestRunner_LlmAgent_FreshTurn verifies that a plain LlmAgent root is
// automatically driven through the node path and yields its model text.
// The user configures nothing special — only Config.Agent.
func TestRunner_LlmAgent_FreshTurn(t *testing.T) {
	ctx := t.Context()
	svc := session.InMemoryService()
	newNodeTestSession(t, ctx, svc)

	m := &scriptedModel{responses: []*genai.Content{
		genai.NewContentFromText("hello there", "model"),
	}}
	a, err := llmagent.New(llmagent.Config{Name: "greeter", Model: m})
	if err != nil {
		t.Fatalf("llmagent.New() error = %v", err)
	}

	r := newNodeTestRunner(t, a, svc)

	var gotText string
	var sawNodeInfo bool
	for ev, err := range r.Run(ctx, nodeTestUser, nodeTestSession, userText("hi"), agent.RunConfig{}) {
		if err != nil {
			t.Fatalf("Run() error = %v", err)
		}
		if ev == nil {
			continue
		}
		// The event must be stamped by the node runtime with the
		// agent's name as its path — this is what distinguishes the
		// node path from the legacy agent path.
		if ev.NodeInfo != nil && (ev.NodeInfo.Path == "greeter" || ev.NodeInfo.Path == "") {
			sawNodeInfo = true
		}
		if ev.LLMResponse.Content != nil {
			for _, p := range ev.LLMResponse.Content.Parts {
				gotText += p.Text
			}
		}
	}
	if gotText != "hello there" {
		t.Errorf("model text = %q, want %q", gotText, "hello there")
	}
	if !sawNodeInfo {
		t.Error("expected an event stamped with NodeInfo")
	}
}

// TestRunner_LlmAgent_YieldUserMessage verifies WithYieldUserMessage makes
// the node path emit the user message event before any agent events.
func TestRunner_LlmAgent_YieldUserMessage(t *testing.T) {
	ctx := t.Context()
	svc := session.InMemoryService()
	newNodeTestSession(t, ctx, svc)

	m := &scriptedModel{responses: []*genai.Content{
		genai.NewContentFromText("hi back", "model"),
	}}
	a, err := llmagent.New(llmagent.Config{Name: "greeter", Model: m})
	if err != nil {
		t.Fatalf("llmagent.New() error = %v", err)
	}
	r := newNodeTestRunner(t, a, svc)

	var firstAuthor string
	var sawUser bool
	for ev, err := range r.Run(ctx, nodeTestUser, nodeTestSession, userText("hi"), agent.RunConfig{}, runner.WithYieldUserMessage()) {
		if err != nil {
			t.Fatalf("Run() error = %v", err)
		}
		if ev == nil {
			continue
		}
		if firstAuthor == "" {
			firstAuthor = ev.Author
		}
		if ev.Author == "user" {
			sawUser = true
		}
	}
	if !sawUser {
		t.Error("expected a user message event to be yielded")
	}
	if firstAuthor != "user" {
		t.Errorf("first yielded event author = %q, want the user message first", firstAuthor)
	}
}

// TestRunner_LlmAgent_LongRunningTool_PausesAndResumes covers the HITL
// bridge: an LlmAgent that calls a long-running tool emits
// LongRunningToolIDs; the node wrapper translates that into a workflow
// pause (RequestedInput) and persists RunState. A follow-up turn carrying
// the matching FunctionResponse resumes the LlmAgent.
func TestRunner_LlmAgent_LongRunningTool_PausesAndResumes(t *testing.T) {
	ctx := t.Context()
	svc := session.InMemoryService()
	newNodeTestSession(t, ctx, svc)

	// Turn 1: model calls the long-running tool, then (after the tool's
	// pending result is fed back) emits a follow-up text and the turn
	// pauses on the unresolved long-running call.
	// Turn 2 (resume): model produces a final text answer.
	m := &scriptedModel{responses: []*genai.Content{
		genai.NewContentFromFunctionCall("ask_human", map[string]any{}, "model"),
		genai.NewContentFromText("waiting for approval", "model"),
		genai.NewContentFromText("all done", "model"),
	}}

	type askArgs struct{}
	longTool, err := functiontool.New(functiontool.Config{
		Name:          "ask_human",
		Description:   "asks a human and waits",
		IsLongRunning: true,
	}, func(_ agent.Context, _ askArgs) (map[string]string, error) {
		return map[string]string{"status": "pending"}, nil
	})
	if err != nil {
		t.Fatalf("functiontool.New() error = %v", err)
	}

	a, err := llmagent.New(llmagent.Config{
		Name:  "approver",
		Model: m,
		Tools: []tool.Tool{longTool},
	})
	if err != nil {
		t.Fatalf("llmagent.New() error = %v", err)
	}

	r := newNodeTestRunner(t, a, svc)

	// --- Turn 1: should pause on the long-running tool -----------------
	// The pause signal is the long-running tool-call event itself
	// (Event.LongRunningToolIDs); the scheduler parks the node on it
	// with no separate RequestInput event, matching adk-python.
	var longRunningID string
	for ev, err := range r.Run(ctx, nodeTestUser, nodeTestSession, userText("please approve"), agent.RunConfig{}) {
		if err != nil {
			t.Fatalf("turn 1 Run() error = %v", err)
		}
		if ev == nil {
			continue
		}
		if len(ev.LongRunningToolIDs) > 0 {
			longRunningID = ev.LongRunningToolIDs[0]
		}
	}
	if longRunningID == "" {
		t.Fatal("did not observe a long-running tool call from the LlmAgent")
	}
	// A long-running tool that returns a pending (non-empty) result feeds
	// it back to the model once more in the same turn (matching adk-python),
	// so the model can emit a follow-up before the turn pauses on the
	// unresolved long-running call. Two model calls are expected in turn 1.
	if m.call != 2 {
		t.Errorf("turn 1 made %d model calls, want 2 (pending long-running result is fed back to the model once)", m.call)
	}

	// RunState must be persisted with the pause so turn 2 can resume.
	state := runStateForAgent(t, ctx, svc, a)
	if !hasWaitingInterrupt(state, longRunningID) {
		t.Errorf("expected a waiting node with InterruptID %q in persisted state", longRunningID)
	}

	// --- Turn 2: resume with the function response --------------------
	reply := &genai.Content{
		Role: genai.RoleUser,
		Parts: []*genai.Part{{
			FunctionResponse: &genai.FunctionResponse{
				ID:       longRunningID,
				Name:     "ask_human",
				Response: map[string]any{"output": "approved"},
			},
		}},
	}
	var sawDone bool
	for ev, err := range r.Run(ctx, nodeTestUser, nodeTestSession, reply, agent.RunConfig{}) {
		if err != nil {
			t.Fatalf("turn 2 Run() error = %v", err)
		}
		if ev != nil && ev.LLMResponse.Content != nil {
			for _, p := range ev.LLMResponse.Content.Parts {
				if p.Text == "all done" {
					sawDone = true
				}
			}
		}
	}
	if !sawDone {
		t.Error("did not observe the LlmAgent's final answer after resume")
	}
}

// scriptedModel is a fake model.LLM that yields a fixed sequence of
// contents, one per GenerateContent call. It avoids importing
// internal/testutil (which imports runner, creating a cycle).
type scriptedModel struct {
	responses []*genai.Content
	call      int
}

func (m *scriptedModel) Name() string { return "scripted" }

func (m *scriptedModel) GenerateContent(_ context.Context, _ *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		i := m.call
		if i >= len(m.responses) {
			i = len(m.responses) - 1
		}
		m.call++
		yield(&model.LLMResponse{Content: m.responses[i]}, nil)
	}
}

func newNodeTestRunner(t *testing.T, a agent.Agent, svc session.Service) *runner.Runner {
	t.Helper()
	r, err := runner.New(runner.Config{
		AppName:        nodeTestApp,
		Agent:          a,
		SessionService: svc,
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	return r
}

func newNodeTestSession(t *testing.T, ctx context.Context, svc session.Service) {
	t.Helper()
	if _, err := svc.Create(ctx, &session.CreateRequest{
		AppName:   nodeTestApp,
		UserID:    nodeTestUser,
		SessionID: nodeTestSession,
	}); err != nil {
		t.Fatalf("sessionService.Create() error = %v", err)
	}
}

func userText(text string) *genai.Content {
	return &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{{Text: text}}}
}

// runStateForAgent reconstructs the paused RunState from session
// history the same way the runner does.
func runStateForAgent(t *testing.T, ctx context.Context, svc session.Service, a agent.Agent) *workflow.RunState {
	t.Helper()
	got, err := svc.Get(ctx, &session.GetRequest{AppName: nodeTestApp, UserID: nodeTestUser, SessionID: nodeTestSession})
	if err != nil {
		t.Fatalf("Get() error = %v", err)
	}
	// Rebuild a single-node workflow whose node name matches the
	// agent (ReconstructRunState attributes events by author == node
	// name), so reconstruction sees the same waiting node the runner
	// would.
	node, err := workflow.NewAgentNode(a, workflow.NodeConfig{})
	if err != nil {
		t.Fatalf("workflow.NewAgentNode() error = %v", err)
	}
	wf, err := workflow.New(nodeTestApp+"/"+a.Name(), []workflow.Edge{
		{From: workflow.Start, To: node},
	})
	if err != nil {
		t.Fatalf("workflow.New() error = %v", err)
	}
	state, err := wf.ReconstructRunState(got.Session)
	if err != nil {
		t.Fatalf("ReconstructRunState() error = %v", err)
	}
	return state
}

// hasWaitingInterrupt reports whether the RunState has a node parked
// (NodeWaiting) on the given long-running interrupt ID. The node path
// pauses on Event.LongRunningToolIDs recorded as NodeState.Interrupts
// (no synthetic RequestInput), matching adk-python.
func hasWaitingInterrupt(state *workflow.RunState, id string) bool {
	if state == nil {
		return false
	}
	for _, ns := range state.Nodes {
		if ns == nil || ns.Status != workflow.NodeWaiting {
			continue
		}
		for _, got := range ns.Interrupts {
			if got == id {
				return true
			}
		}
	}
	return false
}

// TestRunner_MessageAsOutput_ClearsOutput is the regression guard for
// the duplicate-output fix. An AgentNode wrapping an LlmAgent emits a
// final model message that synthesizeAgentOutput stamps with
// NodeInfo.MessageAsOutput and populates both Content (the model text)
// and Output (the same text). Driven through the runner, the yielded
// event must have Output cleared so downstream renderers don't surface
// the same text twice, while Content survives. Mirrors adk-python
// runners.py, which copies the event and sets output=None when
// node_info.message_as_output.
func TestRunner_MessageAsOutput_ClearsOutput(t *testing.T) {
	ctx := t.Context()
	svc := session.InMemoryService()
	newNodeTestSession(t, ctx, svc)

	m := &scriptedModel{responses: []*genai.Content{
		genai.NewContentFromText("the only answer", "model"),
	}}
	inner, err := llmagent.New(llmagent.Config{Name: "greeter", Model: m})
	if err != nil {
		t.Fatalf("llmagent.New() error = %v", err)
	}
	node, err := workflow.NewAgentNode(inner, workflow.NodeConfig{})
	if err != nil {
		t.Fatalf("workflow.NewAgentNode() error = %v", err)
	}
	wfAgent, err := workflowagent.New(workflowagent.Config{
		Name:  "wf",
		Edges: workflow.Chain(workflow.Start, node),
	})
	if err != nil {
		t.Fatalf("workflowagent.New() error = %v", err)
	}

	r, err := runner.New(runner.Config{
		AppName:        nodeTestApp,
		Agent:          wfAgent,
		SessionService: svc,
	})
	if err != nil {
		t.Fatalf("runner.New() error = %v", err)
	}

	var sawMessageAsOutput bool
	for ev, err := range r.Run(ctx, nodeTestUser, nodeTestSession, userText("hi"), agent.RunConfig{}) {
		if err != nil {
			t.Fatalf("Run() error = %v", err)
		}
		if ev == nil || ev.LLMResponse.Partial {
			continue
		}
		if ev.NodeInfo == nil || !ev.NodeInfo.MessageAsOutput || ev.LLMResponse.Content == nil {
			continue
		}
		sawMessageAsOutput = true

		if ev.Output != nil {
			t.Errorf("MessageAsOutput event Output = %v, want nil; must be cleared to avoid double-rendering the model text", ev.Output)
		}
		if len(ev.LLMResponse.Content.Parts) == 0 {
			t.Error("MessageAsOutput event lost its Content after Output was cleared")
		}
	}
	if !sawMessageAsOutput {
		t.Fatal("expected a non-partial event stamped with NodeInfo.MessageAsOutput")
	}
}
