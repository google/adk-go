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

// runStateForAgent loads the RunState the node path persists, using the
// same naming scheme the runner uses internally ("<app>/<agent name>").
func runStateForAgent(t *testing.T, ctx context.Context, svc session.Service, a agent.Agent) *workflow.RunState {
	t.Helper()
	got, err := svc.Get(ctx, &session.GetRequest{AppName: nodeTestApp, UserID: nodeTestUser, SessionID: nodeTestSession})
	if err != nil {
		t.Fatalf("Get() error = %v", err)
	}
	state, err := workflow.LoadRunState(got.Session, nodeTestApp+"/"+a.Name())
	if err != nil {
		t.Fatalf("LoadRunState() error = %v", err)
	}
	return state
}

// hasWaitingInterrupt reports whether the RunState has a node parked on a
// human-input request with the given InterruptID.
func hasWaitingInterrupt(state *workflow.RunState, id string) bool {
	if state == nil {
		return false
	}
	for _, ns := range state.Nodes {
		if ns != nil && ns.Status == workflow.NodeWaiting && ns.PendingRequest != nil && ns.PendingRequest.InterruptID == id {
			return true
		}
	}
	return false
}

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

	var texts []string
	for ev, err := range r.Run(ctx, nodeTestUser, nodeTestSession, userText("hi"), agent.RunConfig{}) {
		if err != nil {
			t.Fatalf("Run() error = %v", err)
		}
		if ev != nil && ev.LLMResponse.Content != nil {
			for _, p := range ev.LLMResponse.Content.Parts {
				if p.Text != "" {
					texts = append(texts, p.Text)
				}
			}
		}
	}
	if len(texts) == 0 {
		t.Fatal("expected at least one model text event from the LlmAgent")
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

	// Turn 1: model calls the long-running tool.
	// Turn 2 (resume): model produces a final text answer.
	m := &scriptedModel{responses: []*genai.Content{
		genai.NewContentFromFunctionCall("ask_human", map[string]any{}, "model"),
		genai.NewContentFromText("all done", "model"),
	}}

	type askArgs struct{}
	longTool, err := functiontool.New(functiontool.Config{
		Name:          "ask_human",
		Description:   "asks a human and waits",
		IsLongRunning: true,
	}, func(_ agent.ToolContext, _ askArgs) (map[string]string, error) {
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
	var (
		longRunningID string
		sawPause      bool
	)
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
		if ev.RequestedInput != nil {
			sawPause = true
		}
	}
	if longRunningID == "" {
		t.Fatal("did not observe a long-running tool call from the LlmAgent")
	}
	if !sawPause {
		t.Fatal("node wrapper did not bridge the long-running tool into a workflow RequestedInput pause")
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
