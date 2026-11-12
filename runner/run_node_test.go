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
	"errors"
	"fmt"
	"iter"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/google/jsonschema-go/jsonschema"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/agent/workflowagent"
	"google.golang.org/adk/v2/agent/workflowagents/parallelagent"
	"google.golang.org/adk/v2/agent/workflowagents/sequentialagent"
	"google.golang.org/adk/v2/internal/plugininternal"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/platform"
	"google.golang.org/adk/v2/plugin"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
	"google.golang.org/adk/v2/workflow"
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
	state, err := wf.ReconstructRunState(got.Session, "")
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
		if slices.Contains(ns.Interrupts, id) {
			return true
		}
	}
	return false
}

// TestRunner_MessageAsOutput_ClearsOutput asserts that for a
// NodeInfo.MessageAsOutput event (Content holds the model text), the
// runner yields it with Output cleared and Content kept, so renderers
// don't surface the text twice. Mirrors adk-python runners.py, which
// clears output when node_info.message_as_output.
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

// OutputKey + OutputSchema agent run through the Runner must yield one
// output-carrying event, else the scheduler raises ErrMultipleOutputs.
func TestRunner_LlmAgent_OutputKeySchema_SingleOutput(t *testing.T) {
	ctx := t.Context()
	svc := session.InMemoryService()
	newNodeTestSession(t, ctx, svc)

	const jsonResp = `{"text":"ok"}`
	m := &scriptedModel{responses: []*genai.Content{
		genai.NewContentFromText(jsonResp, "model"),
	}}
	a, err := llmagent.New(llmagent.Config{
		Name:      "structured",
		Model:     m,
		OutputKey: "resp",
		OutputSchema: &genai.Schema{
			Type:       genai.TypeObject,
			Properties: map[string]*genai.Schema{"text": {Type: genai.TypeString}},
		},
	})
	if err != nil {
		t.Fatalf("llmagent.New() error = %v", err)
	}
	r := newNodeTestRunner(t, a, svc)

	events := 0
	for ev, err := range r.Run(ctx, nodeTestUser, nodeTestSession, userText("hi"), agent.RunConfig{}) {
		if err != nil {
			t.Fatalf("Run() error = %v", err)
		}
		if ev != nil {
			events++
		}
	}
	if events != 1 {
		t.Fatalf("yielded %d events, want 1", events)
	}

	got, err := svc.Get(ctx, &session.GetRequest{AppName: nodeTestApp, UserID: nodeTestUser, SessionID: nodeTestSession})
	if err != nil {
		t.Fatalf("Get() error = %v", err)
	}
	saved, err := got.Session.State().Get("resp")
	if err != nil {
		t.Fatalf("State().Get(resp) error = %v", err)
	}
	if saved != jsonResp {
		t.Errorf("state[resp] = %v, want %q", saved, jsonResp)
	}
}

// newUpperWorkflowAgent wraps a single function node that upper-cases the
// user's text. adk-go has no Runner(node=...) constructor, so node graphs
// are driven through the node runtime by wrapping them in a workflowagent
// (the LlmAgent cases use an LlmAgent root directly instead).
func newUpperWorkflowAgent(t *testing.T, name string) agent.Agent {
	t.Helper()
	upper := workflow.NewFunctionNode(name+"_upper", func(_ agent.Context, in string) (string, error) {
		return strings.ToUpper(in), nil
	}, workflow.NodeConfig{})
	wf, err := workflowagent.New(workflowagent.Config{
		Name:  name,
		Edges: workflow.Chain(workflow.Start, upper),
	})
	if err != nil {
		t.Fatalf("workflowagent.New() error = %v", err)
	}
	return wf
}

func TestRunner_WorkflowNode_OutputAndPersistence(t *testing.T) {
	ctx := t.Context()
	svc := session.InMemoryService()
	newNodeTestSession(t, ctx, svc)

	r := newNodeTestRunner(t, newUpperWorkflowAgent(t, "wf"), svc)

	var gotOutputs []any
	for ev, err := range r.Run(ctx, nodeTestUser, nodeTestSession, userText("hi"), agent.RunConfig{}) {
		if err != nil {
			t.Fatalf("Run() error = %v", err)
		}
		if ev != nil && ev.Output != nil {
			gotOutputs = append(gotOutputs, ev.Output)
		}
	}

	if len(gotOutputs) != 1 || gotOutputs[0] != "HI" {
		t.Fatalf("yielded outputs = %v, want [HI]", gotOutputs)
	}

	// The output event must survive into session history.
	got, err := svc.Get(ctx, &session.GetRequest{AppName: nodeTestApp, UserID: nodeTestUser, SessionID: nodeTestSession})
	if err != nil {
		t.Fatalf("Get() error = %v", err)
	}
	if !slices.Contains(sessionOutputs(got.Session), "HI") {
		t.Errorf("session does not contain the persisted output %q", "HI")
	}
}

func TestRunner_WorkflowNode_ErrorPropagates(t *testing.T) {
	ctx := t.Context()
	svc := session.InMemoryService()
	newNodeTestSession(t, ctx, svc)

	boom := workflow.NewFunctionNode("boom", func(_ agent.Context, _ string) (string, error) {
		return "", errNodeBoom
	}, workflow.NodeConfig{})
	wf, err := workflowagent.New(workflowagent.Config{
		Name:  "wf_err",
		Edges: workflow.Chain(workflow.Start, boom),
	})
	if err != nil {
		t.Fatalf("workflowagent.New() error = %v", err)
	}
	r := newNodeTestRunner(t, wf, svc)

	var gotErr error
	for _, err := range r.Run(ctx, nodeTestUser, nodeTestSession, userText("hi"), agent.RunConfig{}) {
		if err != nil {
			gotErr = err
		}
	}
	if gotErr == nil {
		t.Fatal("Run() yielded no error, want the node failure to propagate")
	}
	if !strings.Contains(gotErr.Error(), "node failure") {
		t.Errorf("error = %v, want it to contain %q", gotErr, "node failure")
	}
}

// TestRunner_WorkflowNode_MultipleInvocationsAccumulate checks that each
// turn appends to the same session rather than replacing its history.
func TestRunner_WorkflowNode_MultipleInvocationsAccumulate(t *testing.T) {
	ctx := t.Context()
	svc := session.InMemoryService()
	newNodeTestSession(t, ctx, svc)

	r := newNodeTestRunner(t, newUpperWorkflowAgent(t, "wf"), svc)

	for _, in := range []string{"first", "second", "third"} {
		for _, err := range r.Run(ctx, nodeTestUser, nodeTestSession, userText(in), agent.RunConfig{}) {
			if err != nil {
				t.Fatalf("Run(%q) error = %v", in, err)
			}
		}
	}

	got, err := svc.Get(ctx, &session.GetRequest{AppName: nodeTestApp, UserID: nodeTestUser, SessionID: nodeTestSession})
	if err != nil {
		t.Fatalf("Get() error = %v", err)
	}
	want := []string{"FIRST", "SECOND", "THIRD"}
	gotOutputs := sessionOutputs(got.Session)
	if len(gotOutputs) != len(want) {
		t.Fatalf("session outputs = %v, want %v", gotOutputs, want)
	}
	for i, w := range want {
		if gotOutputs[i] != w {
			t.Errorf("session output[%d] = %v, want %q", i, gotOutputs[i], w)
		}
	}
}

// TestRunner_Node_YieldUserMessageFalseByDefault is the negative case for
// TestRunner_LlmAgent_YieldUserMessage: without the opt-in the user event
// is persisted but not yielded.
func TestRunner_Node_YieldUserMessageFalseByDefault(t *testing.T) {
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

	for ev, err := range r.Run(ctx, nodeTestUser, nodeTestSession, userText("hi"), agent.RunConfig{}) {
		if err != nil {
			t.Fatalf("Run() error = %v", err)
		}
		if ev != nil && ev.Author == "user" {
			t.Errorf("yielded a user message event %v, want none by default", ev)
		}
	}

	// The user message is still recorded in history; only the yielding
	// is gated by WithYieldUserMessage.
	got, err := svc.Get(ctx, &session.GetRequest{AppName: nodeTestApp, UserID: nodeTestUser, SessionID: nodeTestSession})
	if err != nil {
		t.Fatalf("Get() error = %v", err)
	}
	if !sessionHasUserText(got.Session, "hi") {
		t.Errorf("session does not contain the persisted user message %q", "hi")
	}
}

// TestRunner_Node_StateDeltaAppliedBeforeNodeRuns checks that a state
// delta passed to Run lands in session state before the node body runs.
func TestRunner_Node_StateDeltaAppliedBeforeNodeRuns(t *testing.T) {
	ctx := t.Context()
	svc := session.InMemoryService()
	newNodeTestSession(t, ctx, svc)

	reader := workflow.NewFunctionNode("reader", func(c agent.Context, _ string) (string, error) {
		v, err := c.State().Get("test_state")
		if err != nil {
			return "", err
		}
		s, _ := v.(string)
		return "state:" + s, nil
	}, workflow.NodeConfig{})
	wf, err := workflowagent.New(workflowagent.Config{
		Name:  "wf_state",
		Edges: workflow.Chain(workflow.Start, reader),
	})
	if err != nil {
		t.Fatalf("workflowagent.New() error = %v", err)
	}
	r := newNodeTestRunner(t, wf, svc)

	var gotOutputs []any
	for ev, err := range r.Run(ctx, nodeTestUser, nodeTestSession, userText("go"), agent.RunConfig{},
		runner.WithStateDelta(map[string]any{"test_state": "must_change"})) {
		if err != nil {
			t.Fatalf("Run() error = %v", err)
		}
		if ev != nil && ev.Output != nil {
			gotOutputs = append(gotOutputs, ev.Output)
		}
	}

	if len(gotOutputs) != 1 || gotOutputs[0] != "state:must_change" {
		t.Fatalf("outputs = %v, want [state:must_change]", gotOutputs)
	}

	got, err := svc.Get(ctx, &session.GetRequest{AppName: nodeTestApp, UserID: nodeTestUser, SessionID: nodeTestSession})
	if err != nil {
		t.Fatalf("Get() error = %v", err)
	}
	v, err := got.Session.State().Get("test_state")
	if err != nil {
		t.Fatalf("State().Get(test_state) error = %v", err)
	}
	if v != "must_change" {
		t.Errorf("session state[test_state] = %v, want %q", v, "must_change")
	}
}

// TestRunner_Node_YieldsUserEventWithStateDelta checks that the yielded
// user event carries the run's state delta.
func TestRunner_Node_YieldsUserEventWithStateDelta(t *testing.T) {
	ctx := t.Context()
	svc := session.InMemoryService()
	newNodeTestSession(t, ctx, svc)

	m := &scriptedModel{responses: []*genai.Content{
		genai.NewContentFromText("done", "model"),
	}}
	a, err := llmagent.New(llmagent.Config{Name: "noop", Model: m})
	if err != nil {
		t.Fatalf("llmagent.New() error = %v", err)
	}
	r := newNodeTestRunner(t, a, svc)

	var first *session.Event
	for ev, err := range r.Run(ctx, nodeTestUser, nodeTestSession, userText("go"), agent.RunConfig{},
		runner.WithStateDelta(map[string]any{"test_state": "must_change"}),
		runner.WithYieldUserMessage()) {
		if err != nil {
			t.Fatalf("Run() error = %v", err)
		}
		if ev != nil && first == nil {
			first = ev
		}
	}

	if first == nil {
		t.Fatal("Run() yielded no events")
	}
	if first.Author != "user" {
		t.Fatalf("first yielded event author = %q, want %q", first.Author, "user")
	}
	if got := first.Actions.StateDelta["test_state"]; got != "must_change" {
		t.Errorf("user event state delta[test_state] = %v, want %q", got, "must_change")
	}
}

// TestRunner_StampsEventsYieldedWithoutTimestamp checks that an event built as
// a struct literal, with no timestamp, is stored with the run's current time,
// on the agent path (sequential root) and on the node path (an LlmAgent root
// that transfers to the agent). That holds when an agent yielded the event,
// when a plugin returned a copy of it, which inherits the timestamp the runner
// stamped on the original before the plugin ran, and when a plugin overwrote
// it in place. An event that already carries a timestamp keeps it.
func TestRunner_StampsEventsYieldedWithoutTimestamp(t *testing.T) {
	runTime := time.Date(2026, time.March, 2, 10, 0, 0, 0, time.UTC)
	ownTime := time.Date(2026, time.March, 1, 9, 0, 0, 0, time.UTC)

	newCustom := func(t *testing.T, own time.Time) agent.Agent {
		a, err := agent.New(agent.Config{
			Name: "custom",
			Run: func(agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					yield(&session.Event{
						Author:      "custom",
						Timestamp:   own,
						LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("reply", genai.RoleModel)},
					}, nil)
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New() error = %v", err)
		}
		return a
	}

	newSequentialRoot := func(t *testing.T, own time.Time) agent.Agent {
		a, err := sequentialagent.New(sequentialagent.Config{
			AgentConfig: agent.Config{Name: "seq_root", SubAgents: []agent.Agent{newCustom(t, own)}},
		})
		if err != nil {
			t.Fatalf("sequentialagent.New() error = %v", err)
		}
		return a
	}
	newLlmRoot := func(t *testing.T, own time.Time) agent.Agent {
		m := &scriptedModel{responses: []*genai.Content{{
			Role: genai.RoleModel,
			Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{
				ID: "transfer-1", Name: "transfer_to_agent", Args: map[string]any{"agent_name": "custom"},
			}}},
		}}}
		a, err := llmagent.New(llmagent.Config{Name: "llm_root", Model: m, SubAgents: []agent.Agent{newCustom(t, own)}})
		if err != nil {
			t.Fatalf("llmagent.New() error = %v", err)
		}
		return a
	}
	tests := []struct {
		name string
		root func(t *testing.T, own time.Time) agent.Agent
		// own is the timestamp the custom agent gives its event; zero means
		// none.
		own time.Time
		// replace makes a plugin swap the custom agent's event for a struct
		// literal copy with no timestamp of its own. The copy takes the
		// original's, which the runner stamped before the plugin ran.
		replace bool
		// rewrite makes a plugin overwrite the custom agent's event in place
		// with such a copy and return nil.
		rewrite bool
		want    time.Time
	}{
		{name: "sequential agent root", root: newSequentialRoot, want: runTime},
		{name: "llm agent root", root: newLlmRoot, want: runTime},
		{name: "sequential agent root, plugin replacement", root: newSequentialRoot, replace: true, want: runTime},
		{name: "llm agent root, plugin replacement", root: newLlmRoot, replace: true, want: runTime},
		{name: "sequential agent root, plugin rewrites in place", root: newSequentialRoot, rewrite: true, want: runTime},
		{name: "llm agent root, plugin rewrites in place", root: newLlmRoot, rewrite: true, want: runTime},
		{name: "sequential agent root, own timestamp", root: newSequentialRoot, own: ownTime, want: ownTime},
		{name: "llm agent root, own timestamp", root: newLlmRoot, own: ownTime, want: ownTime},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := platform.WithTimeProvider(t.Context(), func() time.Time { return runTime })
			svc := session.InMemoryService()
			newNodeTestSession(t, ctx, svc)
			// Plugins see the event as it will be stored, timestamp included.
			sawUnstamped := false
			p, err := plugin.New(plugin.Config{
				Name: "observe_and_replace",
				OnEventCallback: func(_ agent.InvocationContext, ev *session.Event) (*session.Event, error) {
					if ev.Author != "custom" {
						return nil, nil
					}
					sawUnstamped = sawUnstamped || ev.Timestamp.IsZero()
					if tt.rewrite {
						*ev = session.Event{Author: ev.Author, LLMResponse: ev.LLMResponse}
						return nil, nil
					}
					if !tt.replace {
						return nil, nil
					}
					return &session.Event{Author: ev.Author, LLMResponse: ev.LLMResponse}, nil
				},
			})
			if err != nil {
				t.Fatalf("plugin.New() error = %v", err)
			}
			r, err := runner.New(runner.Config{
				AppName:        nodeTestApp,
				Agent:          tt.root(t, tt.own),
				SessionService: svc,
				PluginConfig:   runner.PluginConfig{Plugins: []*plugin.Plugin{p}},
			})
			if err != nil {
				t.Fatalf("runner.New() error = %v", err)
			}

			for _, err := range r.Run(ctx, nodeTestUser, nodeTestSession, userText("go"), agent.RunConfig{}) {
				if err != nil {
					t.Fatalf("Run() error = %v", err)
				}
			}

			got, err := svc.Get(ctx, &session.GetRequest{AppName: nodeTestApp, UserID: nodeTestUser, SessionID: nodeTestSession})
			if err != nil {
				t.Fatalf("Get() error = %v", err)
			}
			var sawReply bool
			for ev := range got.Session.Events().All() {
				if ev.Author != "custom" {
					continue
				}
				sawReply = true
				if !ev.Timestamp.Equal(tt.want) {
					t.Errorf("stored event from %q has timestamp %v, want %v", ev.Author, ev.Timestamp, tt.want)
				}
			}
			if !sawReply {
				t.Fatal("no event from the custom agent was stored")
			}
			if got.Session.LastUpdateTime().IsZero() {
				t.Error("session LastUpdateTime() is zero after the run")
			}
			if sawUnstamped {
				t.Error("OnEventCallback saw the custom agent's event with a zero timestamp")
			}
		})
	}
}

// TestRunner_PluginReplacementKeepsEventIdentity checks that when an OnEvent
// plugin returns a replacement event, the runner streams and stores it with the
// original event's ID, invocation ID and timestamp, and with the original's
// author when the replacement names none, on the agent path (sequential root)
// and on the node path (an LlmAgent root that transfers to the agent). The
// agent gives its event a timestamp different from the run's time and the
// plugin's.
func TestRunner_PluginReplacementKeepsEventIdentity(t *testing.T) {
	runTime := time.Date(2026, time.March, 2, 10, 0, 0, 0, time.UTC)
	pluginTime := time.Date(2026, time.March, 3, 11, 0, 0, 0, time.UTC)
	agentTime := time.Date(2026, time.March, 1, 9, 0, 0, 0, time.UTC)

	newCustom := func(t *testing.T) agent.Agent {
		a, err := agent.New(agent.Config{
			Name: "custom",
			Run: func(ic agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					yield(&session.Event{
						ID:           "agent-event",
						InvocationID: ic.InvocationID(),
						Timestamp:    agentTime,
						Author:       "custom",
						LLMResponse:  model.LLMResponse{Content: genai.NewContentFromText("reply", genai.RoleModel)},
					}, nil)
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New() error = %v", err)
		}
		return a
	}
	newSequentialRoot := func(t *testing.T) agent.Agent {
		a, err := sequentialagent.New(sequentialagent.Config{
			AgentConfig: agent.Config{Name: "seq_root", SubAgents: []agent.Agent{newCustom(t)}},
		})
		if err != nil {
			t.Fatalf("sequentialagent.New() error = %v", err)
		}
		return a
	}
	newLlmRoot := func(t *testing.T) agent.Agent {
		m := &scriptedModel{responses: []*genai.Content{{
			Role: genai.RoleModel,
			Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{
				ID: "transfer-1", Name: "transfer_to_agent", Args: map[string]any{"agent_name": "custom"},
			}}},
		}}}
		a, err := llmagent.New(llmagent.Config{Name: "llm_root", Model: m, SubAgents: []agent.Agent{newCustom(t)}})
		if err != nil {
			t.Fatalf("llmagent.New() error = %v", err)
		}
		return a
	}
	isReplacement := func(ev *session.Event) bool {
		return ev.Content != nil && len(ev.Content.Parts) == 1 && ev.Content.Parts[0].Text == "replaced"
	}
	tests := []struct {
		name string
		root func(t *testing.T) agent.Agent
		// author is the author the plugin's replacement names.
		author     string
		wantAuthor string
	}{
		{name: "sequential agent root, no author", root: newSequentialRoot, wantAuthor: "custom"},
		{name: "llm agent root, no author", root: newLlmRoot, wantAuthor: "custom"},
		{name: "sequential agent root, own author", root: newSequentialRoot, author: "rewriter", wantAuthor: "rewriter"},
		{name: "llm agent root, own author", root: newLlmRoot, author: "rewriter", wantAuthor: "rewriter"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := platform.WithTimeProvider(t.Context(), func() time.Time { return runTime })
			svc := session.InMemoryService()
			newNodeTestSession(t, ctx, svc)
			var original session.Event
			p, err := plugin.New(plugin.Config{
				Name: "replace",
				OnEventCallback: func(_ agent.InvocationContext, ev *session.Event) (*session.Event, error) {
					if ev.Author != "custom" {
						return nil, nil
					}
					original = *ev
					return &session.Event{
						ID:           "plugin-id",
						InvocationID: "plugin-invocation",
						Timestamp:    pluginTime,
						Author:       tt.author,
						LLMResponse:  model.LLMResponse{Content: genai.NewContentFromText("replaced", genai.RoleModel)},
					}, nil
				},
			})
			if err != nil {
				t.Fatalf("plugin.New() error = %v", err)
			}
			r, err := runner.New(runner.Config{
				AppName:        nodeTestApp,
				Agent:          tt.root(t),
				SessionService: svc,
				PluginConfig:   runner.PluginConfig{Plugins: []*plugin.Plugin{p}},
			})
			if err != nil {
				t.Fatalf("runner.New() error = %v", err)
			}

			var streamed *session.Event
			for ev, err := range r.Run(ctx, nodeTestUser, nodeTestSession, userText("go"), agent.RunConfig{}) {
				if err != nil {
					t.Fatalf("Run() error = %v", err)
				}
				if isReplacement(ev) {
					streamed = ev
				}
			}
			if original.Author == "" {
				t.Fatal("the plugin never saw the custom agent's event")
			}
			if streamed == nil {
				t.Fatal("the replacement was not streamed")
			}
			got, err := svc.Get(ctx, &session.GetRequest{AppName: nodeTestApp, UserID: nodeTestUser, SessionID: nodeTestSession})
			if err != nil {
				t.Fatalf("Get() error = %v", err)
			}
			var stored *session.Event
			for ev := range got.Session.Events().All() {
				if isReplacement(ev) {
					stored = ev
				}
			}
			if stored == nil {
				t.Fatal("the replacement was not stored")
			}

			checkIdentity := func(where string, ev *session.Event) {
				t.Helper()
				if ev.ID != original.ID {
					t.Errorf("%s replacement ID = %q, want the original's %q", where, ev.ID, original.ID)
				}
				if ev.InvocationID != original.InvocationID {
					t.Errorf("%s replacement InvocationID = %q, want the original's %q", where, ev.InvocationID, original.InvocationID)
				}
				if !ev.Timestamp.Equal(agentTime) {
					t.Errorf("%s replacement Timestamp = %v, want the original's %v", where, ev.Timestamp, agentTime)
				}
				if ev.Author != tt.wantAuthor {
					t.Errorf("%s replacement Author = %q, want %q", where, ev.Author, tt.wantAuthor)
				}
			}
			checkIdentity("streamed", streamed)
			checkIdentity("stored", stored)
		})
	}
}

// TestRunner_PluginSharedReplacementIsNotModified checks that a plugin that
// returns one replacement object for events by two different agents gets each
// event back under its own author and ID, and that the runner leaves the
// replacement as the plugin built it, including its compaction record.
func TestRunner_PluginSharedReplacementIsNotModified(t *testing.T) {
	runTime := time.Date(2026, time.March, 2, 10, 0, 0, 0, time.UTC)
	ctx := platform.WithTimeProvider(t.Context(), func() time.Time { return runTime })
	svc := session.InMemoryService()
	newNodeTestSession(t, ctx, svc)

	newSpeaker := func(name string) agent.Agent {
		a, err := agent.New(agent.Config{
			Name: name,
			Run: func(ic agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					yield(&session.Event{
						ID:           name + "-event",
						InvocationID: ic.InvocationID(),
						Author:       name,
						LLMResponse:  model.LLMResponse{Content: genai.NewContentFromText("reply", genai.RoleModel)},
					}, nil)
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(%q) error = %v", name, err)
		}
		return a
	}
	root, err := sequentialagent.New(sequentialagent.Config{
		AgentConfig: agent.Config{Name: "seq_root", SubAgents: []agent.Agent{newSpeaker("custom"), newSpeaker("other")}},
	})
	if err != nil {
		t.Fatalf("sequentialagent.New() error = %v", err)
	}

	planted := &session.EventCompaction{}
	shared := &session.Event{
		LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("redacted", genai.RoleModel)},
		Actions:     session.EventActions{Compaction: planted},
	}
	p, err := plugin.New(plugin.Config{
		Name: "redact",
		OnEventCallback: func(_ agent.InvocationContext, ev *session.Event) (*session.Event, error) {
			if ev.Author != "custom" && ev.Author != "other" {
				return nil, nil
			}
			return shared, nil
		},
	})
	if err != nil {
		t.Fatalf("plugin.New() error = %v", err)
	}
	r, err := runner.New(runner.Config{
		AppName:        nodeTestApp,
		Agent:          root,
		SessionService: svc,
		PluginConfig:   runner.PluginConfig{Plugins: []*plugin.Plugin{p}},
	})
	if err != nil {
		t.Fatalf("runner.New() error = %v", err)
	}

	type identity struct{ ID, Author string }
	isRedacted := func(ev *session.Event) bool {
		return ev.Content != nil && len(ev.Content.Parts) == 1 && ev.Content.Parts[0].Text == "redacted"
	}
	var streamed []identity
	for ev, err := range r.Run(ctx, nodeTestUser, nodeTestSession, userText("go"), agent.RunConfig{}) {
		if err != nil {
			t.Fatalf("Run() error = %v", err)
		}
		if isRedacted(ev) {
			streamed = append(streamed, identity{ev.ID, ev.Author})
		}
	}
	got, err := svc.Get(ctx, &session.GetRequest{AppName: nodeTestApp, UserID: nodeTestUser, SessionID: nodeTestSession})
	if err != nil {
		t.Fatalf("Get() error = %v", err)
	}
	var stored []identity
	for ev := range got.Session.Events().All() {
		if isRedacted(ev) {
			stored = append(stored, identity{ev.ID, ev.Author})
		}
	}

	want := []identity{{"custom-event", "custom"}, {"other-event", "other"}}
	if !slices.Equal(want, streamed) {
		t.Errorf("streamed replacements = %v, want %v", streamed, want)
	}
	if !slices.Equal(want, stored) {
		t.Errorf("stored replacements = %v, want %v", stored, want)
	}
	if shared.ID != "" || shared.InvocationID != "" || shared.Author != "" || !shared.Timestamp.IsZero() {
		t.Errorf("shared replacement was modified: ID = %q, InvocationID = %q, Author = %q, Timestamp = %v; want all empty",
			shared.ID, shared.InvocationID, shared.Author, shared.Timestamp)
	}
	if shared.Actions.Compaction != planted {
		t.Errorf("shared replacement Actions.Compaction = %p, want the planted record %p", shared.Actions.Compaction, planted)
	}
}

// TestRunner_StampsBeforeRunReplacement checks that a BeforeRun reply that an
// OnEvent plugin replaces with a struct literal is stored with a timestamp, and
// that a replacement with an ID, invocation ID and timestamp of its own and no
// author is stored with the reply's ID, invocation ID and timestamp and the
// agent's name as author.
func TestRunner_StampsBeforeRunReplacement(t *testing.T) {
	runTime := time.Date(2026, time.March, 2, 10, 0, 0, 0, time.UTC)
	pluginTime := time.Date(2026, time.March, 3, 11, 0, 0, 0, time.UTC)
	for _, tt := range []struct {
		name string
		// rewrite makes the plugin overwrite the reply in place with a struct
		// literal copy and return nil, instead of returning that copy.
		rewrite bool
	}{
		{name: "replacement"},
		{name: "rewrite in place", rewrite: true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			ctx := platform.WithTimeProvider(t.Context(), func() time.Time { return runTime })
			svc := session.InMemoryService()
			newNodeTestSession(t, ctx, svc)
			a, err := agent.New(agent.Config{Name: "custom"})
			if err != nil {
				t.Fatalf("agent.New() error = %v", err)
			}
			var original session.Event
			p, err := plugin.New(plugin.Config{
				Name: "cache",
				BeforeRunCallback: func(agent.InvocationContext) (*genai.Content, error) {
					return genai.NewContentFromText("cached", genai.RoleModel), nil
				},
				OnEventCallback: func(_ agent.InvocationContext, ev *session.Event) (*session.Event, error) {
					if ev.Author != "custom" {
						return nil, nil
					}
					original = *ev
					if tt.rewrite {
						*ev = session.Event{Author: ev.Author, LLMResponse: ev.LLMResponse}
						return nil, nil
					}
					return &session.Event{
						ID:           "plugin-id",
						InvocationID: "plugin-invocation",
						Timestamp:    pluginTime,
						LLMResponse:  ev.LLMResponse,
					}, nil
				},
			})
			if err != nil {
				t.Fatalf("plugin.New() error = %v", err)
			}
			r, err := runner.New(runner.Config{
				AppName:        nodeTestApp,
				Agent:          a,
				SessionService: svc,
				PluginConfig:   runner.PluginConfig{Plugins: []*plugin.Plugin{p}},
			})
			if err != nil {
				t.Fatalf("runner.New() error = %v", err)
			}

			for _, err := range r.Run(ctx, nodeTestUser, nodeTestSession, userText("go"), agent.RunConfig{}) {
				if err != nil {
					t.Fatalf("Run() error = %v", err)
				}
			}

			got, err := svc.Get(ctx, &session.GetRequest{AppName: nodeTestApp, UserID: nodeTestUser, SessionID: nodeTestSession})
			if err != nil {
				t.Fatalf("Get() error = %v", err)
			}
			isReply := func(ev *session.Event) bool {
				return ev.Content != nil && len(ev.Content.Parts) == 1 && ev.Content.Parts[0].Text == "cached"
			}
			var sawReply bool
			for ev := range got.Session.Events().All() {
				if !isReply(ev) {
					continue
				}
				sawReply = true
				if ev.Author != "custom" {
					t.Errorf("stored BeforeRun reply Author = %q, want %q", ev.Author, "custom")
				}
				if !ev.Timestamp.Equal(runTime) {
					t.Errorf("stored BeforeRun reply has timestamp %v, want %v", ev.Timestamp, runTime)
				}
				if tt.rewrite {
					continue
				}
				if ev.ID != original.ID {
					t.Errorf("stored BeforeRun reply ID = %q, want the original's %q", ev.ID, original.ID)
				}
				if ev.InvocationID != original.InvocationID {
					t.Errorf("stored BeforeRun reply InvocationID = %q, want the original's %q", ev.InvocationID, original.InvocationID)
				}
			}
			if !sawReply {
				t.Fatal("the BeforeRun reply was not stored")
			}
			if original.ID == "" || original.InvocationID == "" {
				t.Errorf("the plugin saw a reply with ID %q and InvocationID %q, want both set", original.ID, original.InvocationID)
			}
			if got.Session.LastUpdateTime().IsZero() {
				t.Error("session LastUpdateTime() is zero after the run")
			}
		})
	}
}

// TestRunner_Node_StateDeltaAppliedBeforeLlmAgentRuns checks that an
// LlmAgent's before-agent callback observes the run's state delta. The
// callback returns content, which short-circuits the agent, so the model
// is never consulted.
func TestRunner_Node_StateDeltaAppliedBeforeLlmAgentRuns(t *testing.T) {
	ctx := t.Context()
	svc := session.InMemoryService()
	newNodeTestSession(t, ctx, svc)

	var captured any
	m := &scriptedModel{responses: []*genai.Content{
		genai.NewContentFromText("unused", "model"),
	}}
	a, err := llmagent.New(llmagent.Config{
		Name:  "state_agent",
		Model: m,
		BeforeAgentCallbacks: []agent.BeforeAgentCallback{
			func(c agent.Context) (*genai.Content, error) {
				v, err := c.State().Get("test_state")
				if err != nil {
					return nil, err
				}
				captured = v
				s, _ := v.(string)
				return genai.NewContentFromText("state:"+s, genai.RoleModel), nil
			},
		},
	})
	if err != nil {
		t.Fatalf("llmagent.New() error = %v", err)
	}
	r := newNodeTestRunner(t, a, svc)

	var sawResponse bool
	for ev, err := range r.Run(ctx, nodeTestUser, nodeTestSession, userText("go"), agent.RunConfig{},
		runner.WithStateDelta(map[string]any{"test_state": "must_change"})) {
		if err != nil {
			t.Fatalf("Run() error = %v", err)
		}
		if ev != nil && ev.LLMResponse.Content != nil {
			for _, p := range ev.LLMResponse.Content.Parts {
				if p.Text == "state:must_change" {
					sawResponse = true
				}
			}
		}
	}

	if captured != "must_change" {
		t.Errorf("before-agent callback saw state = %v, want %q", captured, "must_change")
	}
	if !sawResponse {
		t.Error("did not observe the before-agent callback response carrying the state value")
	}
}

// TestRunner_Node_PlainTextDoesNotTriggerResume checks that a follow-up
// plain-text turn starts a fresh run instead of resuming.
func TestRunner_Node_PlainTextDoesNotTriggerResume(t *testing.T) {
	ctx := t.Context()
	svc := session.InMemoryService()
	newNodeTestSession(t, ctx, svc)

	r := newNodeTestRunner(t, newUpperWorkflowAgent(t, "wf"), svc)

	for _, err := range r.Run(ctx, nodeTestUser, nodeTestSession, userText("first"), agent.RunConfig{}) {
		if err != nil {
			t.Fatalf("turn 1 Run() error = %v", err)
		}
	}

	var gotOutputs []any
	for ev, err := range r.Run(ctx, nodeTestUser, nodeTestSession, userText("second"), agent.RunConfig{}) {
		if err != nil {
			t.Fatalf("turn 2 Run() error = %v", err)
		}
		if ev != nil && ev.Output != nil {
			gotOutputs = append(gotOutputs, ev.Output)
		}
	}
	if len(gotOutputs) != 1 || gotOutputs[0] != "SECOND" {
		t.Fatalf("turn 2 outputs = %v, want [SECOND]", gotOutputs)
	}
}

// adk-python rejects malformed resume messages (test_mixed_fr_and_text_raises,
// test_resume_raises_on_unmatched_fr): a message must not mix function
// responses with text, and every response must match a call in history.
// adk-go is deliberately more permissive — the A2A flow resumes a paused
// task by sending the human's reply text alongside the approval response
// in one message (see agent/remoteagent/v2/a2a_e2e_test.go, msg3), and
// buildResumeResponses ignores responses that answer no pending
// interrupt. These tests guard that permissive behavior against a future
// "parity" change that would break A2A.

// newPermissiveAgent's model is consulted because adk-go treats these
// messages as a fresh run rather than rejecting them.
func newPermissiveAgent(t *testing.T) agent.Agent {
	t.Helper()
	m := &scriptedModel{responses: []*genai.Content{
		genai.NewContentFromText("handled", "model"),
	}}
	a, err := llmagent.New(llmagent.Config{Name: "approver", Model: m})
	if err != nil {
		t.Fatalf("llmagent.New() error = %v", err)
	}
	return a
}

func TestRunner_Node_AllowsMixedFunctionResponseAndText(t *testing.T) {
	ctx := t.Context()
	svc := session.InMemoryService()
	newNodeTestSession(t, ctx, svc)
	r := newNodeTestRunner(t, newPermissiveAgent(t), svc)

	msg := &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{
		{Text: "some text"},
		{FunctionResponse: &genai.FunctionResponse{ID: "fc-1", Name: "tool", Response: map[string]any{"v": 1}}},
	}}

	var sawText bool
	for ev, err := range r.Run(ctx, nodeTestUser, nodeTestSession, msg, agent.RunConfig{}) {
		if err != nil {
			t.Fatalf("Run() error = %v, want nil (adk-go allows mixed function-response + text)", err)
		}
		if ev != nil && ev.LLMResponse.Content != nil {
			for _, p := range ev.LLMResponse.Content.Parts {
				if p.Text == "handled" {
					sawText = true
				}
			}
		}
	}
	if !sawText {
		t.Error("expected a fresh run to consult the model for a mixed function-response + text message")
	}
}

func TestRunner_Node_ToleratesUnmatchedFunctionResponse(t *testing.T) {
	ctx := t.Context()
	svc := session.InMemoryService()
	newNodeTestSession(t, ctx, svc)
	r := newNodeTestRunner(t, newPermissiveAgent(t), svc)

	msg := &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{
		{FunctionResponse: &genai.FunctionResponse{ID: "no-such-fc", Name: "unknown", Response: map[string]any{}}},
	}}

	var sawText bool
	for ev, err := range r.Run(ctx, nodeTestUser, nodeTestSession, msg, agent.RunConfig{}) {
		if err != nil {
			t.Fatalf("Run() error = %v, want nil (adk-go tolerates an unmatched function response)", err)
		}
		if ev != nil && ev.LLMResponse.Content != nil {
			for _, p := range ev.LLMResponse.Content.Parts {
				if p.Text == "handled" {
					sawText = true
				}
			}
		}
	}
	if !sawText {
		t.Error("expected a fresh run to consult the model after an unmatched function response")
	}
}

// Answers resume the invocation of the calls they answer. On the node path
// that also holds for calls of two invocations: the first call's invocation
// is resumed, where adk-python raises for an LlmAgent root.
func TestRunner_FunctionResponsesResumeTheirInvocation(t *testing.T) {
	tests := []struct {
		name string
		root func(*testing.T) agent.Agent
		ids  []string
	}{
		{name: "agent path, calls of one invocation", root: newCallerRoot, ids: []string{"fc-1a", "fc-1b"}},
		{name: "node path, calls of one invocation", root: newPermissiveAgent, ids: []string{"fc-1a", "fc-1b"}},
		{name: "node path, calls of two invocations", root: newPermissiveAgent, ids: []string{"fc-1a", "fc-2"}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := runFunctionResponses(t, tt.root(t), tt.ids...)
			for _, id := range got {
				if id != "inv-1" {
					t.Errorf("event InvocationID = %q, want %q", id, "inv-1")
				}
			}
		})
	}
}

// On the agent path, answers to calls of two invocations go back to the
// calling agent in a new invocation, as adk-python does for a BaseAgent root
// of an app that is not resumable.
func TestRunner_FunctionResponsesStartNewInvocation(t *testing.T) {
	tests := []struct {
		name string
		root func(*testing.T) agent.Agent
		ids  []string
	}{
		{name: "agent path, calls of two invocations", root: newCallerRoot, ids: []string{"fc-1a", "fc-2"}},
		{name: "agent path, unknown calls", root: newCallerRoot, ids: []string{"no-such-fc"}},
		{name: "node path, unknown calls", root: newPermissiveAgent, ids: []string{"no-such-fc"}},
		// fc-0's event has no invocation ID, and that counts as an invocation
		// of its own, so the run gets a new ID as when fc-0 is answered alone.
		{name: "agent path, call without an invocation ID first", root: newCallerRoot, ids: []string{"fc-0", "fc-2"}},
		{name: "node path, call without an invocation ID first", root: newPermissiveAgent, ids: []string{"fc-0", "fc-2"}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := runFunctionResponses(t, tt.root(t), tt.ids...)
			for _, id := range got {
				if id == "" || id == "inv-1" || id == "inv-2" {
					t.Errorf("event InvocationID = %q, want a new invocation ID", id)
				}
			}
		})
	}
}

// newCallerRoot returns a custom root agent whose sub-agent "approver" made
// the seeded calls, so the runner's agent path sends answers to those calls
// back to it. The root itself yields one text event when it runs.
func newCallerRoot(t *testing.T) agent.Agent {
	t.Helper()
	a, err := agent.New(agent.Config{
		Name:      "root",
		SubAgents: []agent.Agent{newPermissiveAgent(t)},
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				ev := session.NewEvent(ctx, ctx.InvocationID())
				ev.Author = "root"
				ev.LLMResponse.Content = genai.NewContentFromText("root", genai.RoleModel)
				yield(ev, nil)
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New() error = %v", err)
	}
	return a
}

// runFunctionResponses seeds call fc-0 in an event without an invocation ID,
// calls fc-1a and fc-1b from invocation inv-1 and fc-2 from inv-2, all made
// by "approver", runs root on a message answering callIDs, and returns the
// invocation ID of every event the run yields.
func runFunctionResponses(t *testing.T, root agent.Agent, callIDs ...string) []string {
	t.Helper()
	ctx := t.Context()
	svc := session.InMemoryService()
	newNodeTestSession(t, ctx, svc)
	seedFunctionCall(t, ctx, svc, "", "fc-0")
	seedFunctionCall(t, ctx, svc, "inv-1", "fc-1a", "fc-1b")
	seedFunctionCall(t, ctx, svc, "inv-2", "fc-2")
	r := newNodeTestRunner(t, root, svc)

	msg := &genai.Content{Role: genai.RoleUser}
	for _, id := range callIDs {
		msg.Parts = append(msg.Parts, &genai.Part{FunctionResponse: &genai.FunctionResponse{ID: id, Name: "tool", Response: map[string]any{}}})
	}
	var got []string
	for ev, err := range r.Run(ctx, nodeTestUser, nodeTestSession, msg, agent.RunConfig{}) {
		if err != nil {
			t.Fatalf("Run() error = %v", err)
		}
		got = append(got, ev.InvocationID)
	}
	if len(got) == 0 {
		t.Fatal("Run() yielded no events")
	}
	return got
}

// seedFunctionCall appends a model event carrying function calls with the
// given IDs, as if invocationID had issued them.
func seedFunctionCall(t *testing.T, ctx context.Context, svc session.Service, invocationID string, callIDs ...string) {
	t.Helper()
	got, err := svc.Get(ctx, &session.GetRequest{AppName: nodeTestApp, UserID: nodeTestUser, SessionID: nodeTestSession})
	if err != nil {
		t.Fatalf("Get() error = %v", err)
	}
	ev := session.NewEvent(ctx, invocationID)
	ev.Author = "approver"
	ev.LLMResponse.Content = &genai.Content{Role: "model"}
	for _, id := range callIDs {
		ev.LLMResponse.Content.Parts = append(ev.LLMResponse.Content.Parts, &genai.Part{FunctionCall: &genai.FunctionCall{ID: id, Name: "tool"}})
	}
	if err := svc.AppendEvent(ctx, got.Session, ev); err != nil {
		t.Fatalf("AppendEvent() error = %v", err)
	}
}

// errNodeBoom is the failure the boom node returns in
// TestRunner_WorkflowNode_ErrorPropagates.
var errNodeBoom = errors.New("node failure")

// sessionHasUserText reports whether session history holds a user event
// whose content carries the given text.
func sessionHasUserText(sess session.Session, want string) bool {
	events := sess.Events()
	for i := 0; i < events.Len(); i++ {
		ev := events.At(i)
		if ev == nil || ev.Author != "user" || ev.LLMResponse.Content == nil {
			continue
		}
		for _, p := range ev.LLMResponse.Content.Parts {
			if p.Text == want {
				return true
			}
		}
	}
	return false
}

// sessionOutputs returns the terminal outputs of all events, in order.
func sessionOutputs(sess session.Session) []any {
	var outs []any
	events := sess.Events()
	for i := 0; i < events.Len(); i++ {
		ev := events.At(i)
		if ev != nil && ev.Output != nil {
			outs = append(outs, ev.Output)
		}
	}
	return outs
}

// requestRecordingModel replies with a fixed text and records the texts of
// every user content of each request it receives.
type requestRecordingModel struct {
	reply     string
	userTexts [][]string
}

func (*requestRecordingModel) Name() string { return "request_recording" }

func (m *requestRecordingModel) GenerateContent(_ context.Context, req *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	var texts []string
	for _, c := range req.Contents {
		if c == nil || c.Role != genai.RoleUser {
			continue
		}
		for _, p := range c.Parts {
			if p != nil && p.Text != "" {
				texts = append(texts, p.Text)
			}
		}
	}
	m.userTexts = append(m.userTexts, texts)
	return func(yield func(*model.LLMResponse, error) bool) {
		yield(&model.LLMResponse{Content: genai.NewContentFromText(m.reply, genai.RoleModel)}, nil)
	}
}

// newNodeTestRunnerWithPlugins is newNodeTestRunner with plugins.
func newNodeTestRunnerWithPlugins(t *testing.T, a agent.Agent, svc session.Service, plugins ...*plugin.Plugin) *runner.Runner {
	t.Helper()
	r, err := runner.New(runner.Config{
		AppName:        nodeTestApp,
		Agent:          a,
		SessionService: svc,
		PluginConfig:   runner.PluginConfig{Plugins: plugins},
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	return r
}

// replaceMessagePlugin returns a plugin whose OnUserMessageCallback replaces
// every user message with replacement.
func replaceMessagePlugin(t *testing.T, replacement *genai.Content) *plugin.Plugin {
	t.Helper()
	p, err := plugin.New(plugin.Config{
		Name: "replace_message",
		OnUserMessageCallback: func(agent.InvocationContext, *genai.Content) (*genai.Content, error) {
			return replacement, nil
		},
	})
	if err != nil {
		t.Fatalf("plugin.New() error = %v", err)
	}
	return p
}

// collectRun ranges seq and returns its non-nil events and its errors.
func collectRun(seq iter.Seq2[*session.Event, error]) ([]*session.Event, []error) {
	var events []*session.Event
	var errs []error
	for ev, err := range seq {
		if err != nil {
			errs = append(errs, err)
			continue
		}
		if ev != nil {
			events = append(events, ev)
		}
	}
	return events, errs
}

// storedNodeTestSession reads the node test session back from svc.
func storedNodeTestSession(t *testing.T, ctx context.Context, svc session.Service) session.Session {
	t.Helper()
	got, err := svc.Get(ctx, &session.GetRequest{AppName: nodeTestApp, UserID: nodeTestUser, SessionID: nodeTestSession})
	if err != nil {
		t.Fatalf("Get() error = %v", err)
	}
	return got.Session
}

// appendSeedEvent appends ev to the stored node test session.
func appendSeedEvent(t *testing.T, ctx context.Context, svc session.Service, ev *session.Event) {
	t.Helper()
	if err := svc.AppendEvent(ctx, storedNodeTestSession(t, ctx, svc), ev); err != nil {
		t.Fatalf("AppendEvent() error = %v", err)
	}
}

// eventText concatenates the text parts of ev's content.
func eventText(ev *session.Event) string {
	if ev == nil || ev.Content == nil {
		return ""
	}
	var b strings.Builder
	for _, p := range ev.Content.Parts {
		if p != nil {
			b.WriteString(p.Text)
		}
	}
	return b.String()
}

// textReplyAgent returns a custom (non-LLM) agent that yields one event with
// the given text.
func textReplyAgent(t *testing.T, name, text string) agent.Agent {
	t.Helper()
	a, err := agent.New(agent.Config{
		Name: name,
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				ev := session.NewEvent(ctx, ctx.InvocationID())
				ev.Author = name
				ev.Branch = ctx.Branch()
				ev.Content = genai.NewContentFromText(text, genai.RoleModel)
				yield(ev, nil)
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New() error = %v", err)
	}
	return a
}

// textReplyLlmAgent returns an LlmAgent whose model always replies text.
func textReplyLlmAgent(t *testing.T, name, text string) agent.Agent {
	t.Helper()
	a, err := llmagent.New(llmagent.Config{
		Name:  name,
		Model: &scriptedModel{responses: []*genai.Content{genai.NewContentFromText(text, genai.RoleModel)}},
	})
	if err != nil {
		t.Fatalf("llmagent.New() error = %v", err)
	}
	return a
}

func TestRunner_GetSessionConfig_ForwardsLimits(t *testing.T) {
	// seed stores five user events "m0" to "m4", one second apart.
	seed := func(t *testing.T, ctx context.Context, svc session.Service) {
		t.Helper()
		newNodeTestSession(t, ctx, svc)
		for i := range 5 {
			ev := session.NewEvent(ctx, "inv_seed")
			ev.Author = "user"
			ev.Timestamp = time.Date(2026, 1, 1, 0, 0, 1+i, 0, time.UTC)
			ev.Content = genai.NewContentFromText(fmt.Sprintf("m%d", i), genai.RoleUser)
			appendSeedEvent(t, ctx, svc, ev)
		}
	}
	// newReader returns a custom agent that records the texts of the session
	// events it sees.
	newReader := func(t *testing.T, got *[]string) agent.Agent {
		t.Helper()
		a, err := agent.New(agent.Config{
			Name: "reader",
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(func(*session.Event, error) bool) {
					events := ctx.Session().Events()
					for i := 0; i < events.Len(); i++ {
						*got = append(*got, eventText(events.At(i)))
					}
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New() error = %v", err)
		}
		return a
	}

	t.Run("node_path_num_recent_events", func(t *testing.T) {
		ctx := t.Context()
		svc := session.InMemoryService()
		seed(t, ctx, svc)
		m := &requestRecordingModel{reply: "ok"}
		a, err := llmagent.New(llmagent.Config{Name: "greeter", Model: m})
		if err != nil {
			t.Fatalf("llmagent.New() error = %v", err)
		}
		r := newNodeTestRunner(t, a, svc)

		cfg := agent.RunConfig{GetSessionConfig: &agent.GetSessionConfig{NumRecentEvents: 2}}
		if _, errs := collectRun(r.Run(ctx, nodeTestUser, nodeTestSession, userText("hi"), cfg)); len(errs) > 0 {
			t.Fatalf("Run() errors = %v", errs)
		}

		want := [][]string{{"m3", "m4", "hi"}}
		if diff := cmp.Diff(want, m.userTexts); diff != "" {
			t.Errorf("user texts of the model requests mismatch (-want +got):\n%s", diff)
		}
	})

	for _, tc := range []struct {
		name string
		cfg  *agent.GetSessionConfig
		want []string
	}{
		{
			name: "classic_path_after",
			cfg:  &agent.GetSessionConfig{After: time.Date(2026, 1, 1, 0, 0, 4, 0, time.UTC)},
			want: []string{"m3", "m4", "hi"},
		},
		{
			name: "nil_config",
			cfg:  nil,
			want: []string{"m0", "m1", "m2", "m3", "m4", "hi"},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx := t.Context()
			svc := session.InMemoryService()
			seed(t, ctx, svc)
			var got []string
			r := newNodeTestRunner(t, newReader(t, &got), svc)

			if _, errs := collectRun(r.Run(ctx, nodeTestUser, nodeTestSession, userText("hi"), agent.RunConfig{GetSessionConfig: tc.cfg})); len(errs) > 0 {
				t.Fatalf("Run() errors = %v", errs)
			}

			if diff := cmp.Diff(tc.want, got); diff != "" {
				t.Errorf("session event texts the agent saw mismatch (-want +got):\n%s", diff)
			}
		})
	}

	t.Run("snapshot_on_run_call", func(t *testing.T) {
		ctx := t.Context()
		svc := session.InMemoryService()
		seed(t, ctx, svc)
		var got []string
		r := newNodeTestRunner(t, newReader(t, &got), svc)

		cfg := agent.RunConfig{GetSessionConfig: &agent.GetSessionConfig{NumRecentEvents: 2}}
		seq := r.Run(ctx, nodeTestUser, nodeTestSession, userText("hi"), cfg)
		cfg.GetSessionConfig.NumRecentEvents = 1
		if _, errs := collectRun(seq); len(errs) > 0 {
			t.Fatalf("Run() errors = %v", errs)
		}

		want := []string{"m3", "m4", "hi"}
		if diff := cmp.Diff(want, got); diff != "" {
			t.Errorf("session event texts the agent saw mismatch (-want +got):\n%s", diff)
		}
	})
}

func TestRunner_RejectsUserFunctionCall_BeforeAppendAndAfterPlugin(t *testing.T) {
	const wantErr = "user message cannot contain function calls"
	functionCallMsg := func() *genai.Content {
		return &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{
			{FunctionCall: &genai.FunctionCall{Name: "exec", Args: map[string]any{"k": "v"}}},
		}}
	}
	injected := &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{
		{FunctionCall: &genai.FunctionCall{Name: "injected"}},
	}}
	roots := []struct {
		kind string
		new  func(t *testing.T) agent.Agent
	}{
		{kind: "classic", new: func(t *testing.T) agent.Agent { return textReplyAgent(t, "custom", "ok") }},
		{kind: "node", new: func(t *testing.T) agent.Agent { return textReplyLlmAgent(t, "greeter", "ok") }},
	}

	t.Run("node_path_direct_function_call", func(t *testing.T) {
		ctx := t.Context()
		svc := session.InMemoryService()
		newNodeTestSession(t, ctx, svc)
		r := newNodeTestRunner(t, textReplyLlmAgent(t, "greeter", "ok"), svc)

		events, errs := collectRun(r.Run(ctx, nodeTestUser, nodeTestSession, functionCallMsg(), agent.RunConfig{}))

		if len(events) != 0 {
			t.Errorf("Run() yielded %d events, want 0", len(events))
		}
		if len(errs) != 1 || errs[0].Error() != wantErr {
			t.Errorf("Run() errors = %v, want exactly [%q]", errs, wantErr)
		}
		if n := storedNodeTestSession(t, ctx, svc).Events().Len(); n != 0 {
			t.Errorf("stored session has %d events, want 0", n)
		}
	})

	for _, root := range roots {
		t.Run("plugin_injects_function_call/"+root.kind, func(t *testing.T) {
			ctx := t.Context()
			svc := session.InMemoryService()
			newNodeTestSession(t, ctx, svc)
			r := newNodeTestRunnerWithPlugins(t, root.new(t), svc, replaceMessagePlugin(t, injected))

			events, errs := collectRun(r.Run(ctx, nodeTestUser, nodeTestSession, userText("hello"), agent.RunConfig{}))

			if len(events) != 0 {
				t.Errorf("Run() yielded %d events, want 0", len(events))
			}
			if len(errs) != 1 || errs[0].Error() != wantErr {
				t.Errorf("Run() errors = %v, want exactly [%q]", errs, wantErr)
			}
			if n := storedNodeTestSession(t, ctx, svc).Events().Len(); n != 0 {
				t.Errorf("stored session has %d events, want 0", n)
			}
		})
	}

	for _, root := range roots {
		t.Run("plugin_replaces_function_call_with_text/"+root.kind, func(t *testing.T) {
			ctx := t.Context()
			svc := session.InMemoryService()
			newNodeTestSession(t, ctx, svc)
			safe := genai.NewContentFromText("safe text", genai.RoleUser)
			r := newNodeTestRunnerWithPlugins(t, root.new(t), svc, replaceMessagePlugin(t, safe))

			events, errs := collectRun(r.Run(ctx, nodeTestUser, nodeTestSession, functionCallMsg(), agent.RunConfig{}))

			if len(errs) != 0 {
				t.Fatalf("Run() errors = %v, want none", errs)
			}
			if len(events) != 1 || eventText(events[0]) != "ok" {
				t.Errorf("Run() yielded %d events (first text %q), want exactly one with text %q", len(events), eventText(firstOrNil(events)), "ok")
			}
			stored := storedNodeTestSession(t, ctx, svc).Events()
			if stored.Len() == 0 {
				t.Fatal("stored session has no events, want the user event first")
			}
			if first := stored.At(0); first.Author != "user" || eventText(first) != "safe text" {
				t.Errorf("stored event 0 = author %q text %q, want author %q text %q", first.Author, eventText(first), "user", "safe text")
			}
		})
	}
}

// firstOrNil returns events[0], or nil for an empty slice.
func firstOrNil(events []*session.Event) *session.Event {
	if len(events) == 0 {
		return nil
	}
	return events[0]
}

func TestRunner_EmptyMessageParts_RefusedOnClassicAllowedOnNode(t *testing.T) {
	const wantErr = "no parts in the new_message"

	for _, tc := range []struct {
		name    string
		msg     *genai.Content
		plugins func(t *testing.T) []*plugin.Plugin
	}{
		{
			name:    "classic_direct_empty_parts",
			msg:     &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{}},
			plugins: func(*testing.T) []*plugin.Plugin { return nil },
		},
		{
			name: "classic_plugin_returns_empty_parts",
			msg:  userText("hello"),
			plugins: func(t *testing.T) []*plugin.Plugin {
				return []*plugin.Plugin{replaceMessagePlugin(t, &genai.Content{Role: genai.RoleUser, Parts: nil})}
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx := t.Context()
			svc := session.InMemoryService()
			newNodeTestSession(t, ctx, svc)
			r := newNodeTestRunnerWithPlugins(t, textReplyAgent(t, "custom", "ok"), svc, tc.plugins(t)...)

			events, errs := collectRun(r.Run(ctx, nodeTestUser, nodeTestSession, tc.msg, agent.RunConfig{}))

			if len(events) != 0 {
				t.Errorf("Run() yielded %d events, want 0", len(events))
			}
			if len(errs) != 1 || errs[0].Error() != wantErr {
				t.Errorf("Run() errors = %v, want exactly [%q]", errs, wantErr)
			}
			if n := storedNodeTestSession(t, ctx, svc).Events().Len(); n != 0 {
				t.Errorf("stored session has %d events, want 0", n)
			}
		})
	}

	t.Run("node_path_allows_empty_parts", func(t *testing.T) {
		ctx := t.Context()
		svc := session.InMemoryService()
		newNodeTestSession(t, ctx, svc)
		r := newNodeTestRunner(t, textReplyLlmAgent(t, "greeter", "ack"), svc)

		events, errs := collectRun(r.Run(ctx, nodeTestUser, nodeTestSession, &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{}}, agent.RunConfig{}))

		if len(errs) != 0 {
			t.Fatalf("Run() errors = %v, want none", errs)
		}
		if len(events) == 0 || events[0].Author != "greeter" || eventText(events[0]) != "ack" {
			t.Errorf("first yielded event = %+v, want author %q text %q", firstOrNil(events), "greeter", "ack")
		}
		stored := storedNodeTestSession(t, ctx, svc).Events()
		if stored.Len() != 2 {
			t.Fatalf("stored session has %d events, want 2", stored.Len())
		}
		if first := stored.At(0); first.Author != "user" || first.Content == nil || len(first.Content.Parts) != 0 {
			t.Errorf("stored event 0 = author %q content %+v, want an empty user event", first.Author, first.Content)
		}
	})
}

// seedPendingSubAgentCall creates the node test session holding a user turn
// "Start" and sub_agent's long-running call fc_1 to its tool "ask", both of
// invocation inv_prev, the call on branch root_agent.sub_agent. It returns the
// chat root root_agent, whose model replies "root_reply"; sub_agent cannot
// transfer back to it and its model replies "sub_done".
func seedPendingSubAgentCall(t *testing.T, ctx context.Context, svc session.Service) agent.Agent {
	t.Helper()
	newNodeTestSession(t, ctx, svc)

	askTool, err := functiontool.New(functiontool.Config{
		Name:          "ask",
		Description:   "asks a human and waits",
		IsLongRunning: true,
	}, func(agent.Context, struct{}) (map[string]string, error) {
		return map[string]string{"status": "pending"}, nil
	})
	if err != nil {
		t.Fatalf("functiontool.New() error = %v", err)
	}
	sub, err := llmagent.New(llmagent.Config{
		Name:                     "sub_agent",
		Model:                    &scriptedModel{responses: []*genai.Content{genai.NewContentFromText("sub_done", genai.RoleModel)}},
		Tools:                    []tool.Tool{askTool},
		DisallowTransferToParent: true,
	})
	if err != nil {
		t.Fatalf("llmagent.New(sub_agent) error = %v", err)
	}
	root, err := llmagent.New(llmagent.Config{
		Name:      "root_agent",
		Model:     &scriptedModel{responses: []*genai.Content{genai.NewContentFromText("root_reply", genai.RoleModel)}},
		SubAgents: []agent.Agent{sub},
	})
	if err != nil {
		t.Fatalf("llmagent.New(root_agent) error = %v", err)
	}

	start := session.NewEvent(ctx, "inv_prev")
	start.Author = "user"
	start.Content = genai.NewContentFromText("Start", genai.RoleUser)
	appendSeedEvent(t, ctx, svc, start)
	call := session.NewEvent(ctx, "inv_prev")
	call.Author = "sub_agent"
	call.Branch = "root_agent.sub_agent"
	call.LongRunningToolIDs = []string{"fc_1"}
	call.Content = &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{
		{FunctionCall: &genai.FunctionCall{ID: "fc_1", Name: "ask"}},
	}}
	appendSeedEvent(t, ctx, svc, call)
	return root
}

func TestRunner_OnUserMessageCallback_ReroutesAndUpdatesInvocationIDAndBranch(t *testing.T) {
	t.Run("node_path", func(t *testing.T) {
		ctx := t.Context()
		svc := session.InMemoryService()
		root := seedPendingSubAgentCall(t, ctx, svc)

		answer := &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{
			{FunctionResponse: &genai.FunctionResponse{ID: "fc_1", Name: "ask", Response: map[string]any{"status": "done"}}},
		}}
		r := newNodeTestRunnerWithPlugins(t, root, svc, replaceMessagePlugin(t, answer))

		events, errs := collectRun(r.Run(ctx, nodeTestUser, nodeTestSession, userText("approve"), agent.RunConfig{}))

		if len(errs) != 0 {
			t.Fatalf("Run() errors = %v, want none", errs)
		}
		if len(events) != 1 {
			t.Fatalf("Run() yielded %d events, want 1", len(events))
		}
		got := events[0]
		if got.Author != "sub_agent" || got.InvocationID != "inv_prev" || got.Branch != "root_agent.sub_agent" || eventText(got) != "sub_done" {
			t.Errorf("yielded event = author %q invocation %q branch %q text %q, want author %q invocation %q branch %q text %q",
				got.Author, got.InvocationID, got.Branch, eventText(got), "sub_agent", "inv_prev", "root_agent.sub_agent", "sub_done")
		}
		stored := storedNodeTestSession(t, ctx, svc).Events()
		if stored.Len() < 3 {
			t.Fatalf("stored session has %d events, want the user event at index 2", stored.Len())
		}
		userEv := stored.At(2)
		frs := userEv.Content.Parts
		if userEv.Author != "user" || userEv.InvocationID != "inv_prev" || len(frs) != 1 || frs[0].FunctionResponse == nil || frs[0].FunctionResponse.ID != "fc_1" {
			t.Errorf("stored event 2 = author %q invocation %q content %+v, want the user's function response fc_1 under invocation %q",
				userEv.Author, userEv.InvocationID, userEv.Content, "inv_prev")
		}
	})

	t.Run("classic_path", func(t *testing.T) {
		ctx := t.Context()
		svc := session.InMemoryService()
		newNodeTestSession(t, ctx, svc)

		type seen struct{ invocationID, branch, agent string }
		var callerRuns, otherRuns []seen
		recorder := func(name string, runs *[]seen) agent.Agent {
			a, err := agent.New(agent.Config{
				Name: name,
				Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
					return func(func(*session.Event, error) bool) {
						*runs = append(*runs, seen{ctx.InvocationID(), ctx.Branch(), ctx.Agent().Name()})
					}
				},
			})
			if err != nil {
				t.Fatalf("agent.New(%s) error = %v", name, err)
			}
			return a
		}
		root, err := parallelagent.New(parallelagent.Config{AgentConfig: agent.Config{
			Name:      "root",
			SubAgents: []agent.Agent{recorder("caller", &callerRuns), recorder("other", &otherRuns)},
		}})
		if err != nil {
			t.Fatalf("parallelagent.New() error = %v", err)
		}

		call := session.NewEvent(ctx, "inv_par")
		call.Author = "caller"
		call.Branch = "root.caller"
		call.Content = &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{
			{FunctionCall: &genai.FunctionCall{ID: "fc_p", Name: "ask"}},
		}}
		appendSeedEvent(t, ctx, svc, call)

		answer := &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{
			{FunctionResponse: &genai.FunctionResponse{ID: "fc_p", Name: "ask", Response: map[string]any{"ok": true}}},
		}}
		r := newNodeTestRunnerWithPlugins(t, root, svc, replaceMessagePlugin(t, answer))

		if _, errs := collectRun(r.Run(ctx, nodeTestUser, nodeTestSession, userText("approve"), agent.RunConfig{})); len(errs) != 0 {
			t.Fatalf("Run() errors = %v, want none", errs)
		}

		wantCaller := []seen{{invocationID: "inv_par", branch: "root.caller", agent: "caller"}}
		if diff := cmp.Diff(wantCaller, callerRuns, cmp.AllowUnexported(seen{})); diff != "" {
			t.Errorf("caller runs mismatch (-want +got):\n%s", diff)
		}
		if len(otherRuns) != 0 {
			t.Errorf("other ran %d times, want 0", len(otherRuns))
		}
		stored := storedNodeTestSession(t, ctx, svc).Events()
		if stored.Len() < 2 {
			t.Fatalf("stored session has %d events, want the user event at index 1", stored.Len())
		}
		if userEv := stored.At(1); userEv.Author != "user" || userEv.InvocationID != "inv_par" {
			t.Errorf("stored event 1 = author %q invocation %q, want author %q invocation %q", userEv.Author, userEv.InvocationID, "user", "inv_par")
		}
	})
}

// TestRunner_OnUserMessageCallback_TextReplacementDoesNotTakeEarlierBranch pins the
// branch restore after a replaced message on the node path: it considers only
// events of the invocation's own id (adk-python
// workflow/_node_runner_utils.py:200-203), so an agent routed for a
// replacement that answers no call does not continue on the branch it had in
// an earlier invocation.
func TestRunner_OnUserMessageCallback_TextReplacementDoesNotTakeEarlierBranch(t *testing.T) {
	ctx := t.Context()
	svc := session.InMemoryService()
	newNodeTestSession(t, ctx, svc)

	sub := textReplyLlmAgent(t, "sub_agent", "sub_reply")
	root, err := llmagent.New(llmagent.Config{
		Name:      "root_agent",
		Model:     &scriptedModel{responses: []*genai.Content{genai.NewContentFromText("root_reply", genai.RoleModel)}},
		SubAgents: []agent.Agent{sub},
	})
	if err != nil {
		t.Fatalf("llmagent.New(root_agent) error = %v", err)
	}

	start := session.NewEvent(ctx, "inv_old")
	start.Author = "user"
	start.Content = genai.NewContentFromText("Start", genai.RoleUser)
	appendSeedEvent(t, ctx, svc, start)
	earlier := session.NewEvent(ctx, "inv_old")
	earlier.Author = "sub_agent"
	earlier.Branch = "root_agent.sub_agent"
	earlier.Content = genai.NewContentFromText("earlier", genai.RoleModel)
	appendSeedEvent(t, ctx, svc, earlier)

	r := newNodeTestRunnerWithPlugins(t, root, svc, replaceMessagePlugin(t, genai.NewContentFromText("hello again", genai.RoleUser)))

	events, errs := collectRun(r.Run(ctx, nodeTestUser, nodeTestSession, userText("hello"), agent.RunConfig{}))

	if len(errs) != 0 {
		t.Fatalf("Run() errors = %v, want none", errs)
	}
	if len(events) != 1 {
		t.Fatalf("Run() yielded %d events, want 1", len(events))
	}
	if got := events[0]; got.Author != "sub_agent" || got.Branch != "" || eventText(got) != "sub_reply" {
		t.Errorf("yielded event = author %q branch %q text %q, want author %q branch %q text %q",
			got.Author, got.Branch, eventText(got), "sub_agent", "", "sub_reply")
	}
}

func TestRunner_RootTaskMode_EventSequenceAndGoIntSchemaValidation(t *testing.T) {
	ctx := t.Context()
	svc := session.InMemoryService()
	newNodeTestSession(t, ctx, svc)
	m := &scriptedModel{responses: []*genai.Content{
		genai.NewContentFromFunctionCall("finish_task", map[string]any{"result": int(42)}, genai.RoleModel),
	}}
	a, err := llmagent.New(llmagent.Config{
		Name:         "task_int_agent",
		Mode:         llmagent.ModeTask,
		OutputKey:    "answer_key",
		OutputSchema: &genai.Schema{Type: genai.TypeInteger},
		Model:        m,
	})
	if err != nil {
		t.Fatalf("llmagent.New() error = %v", err)
	}
	r := newNodeTestRunner(t, a, svc)

	events, errs := collectRun(r.Run(ctx, nodeTestUser, nodeTestSession, userText("compute"), agent.RunConfig{}))

	if len(errs) != 0 {
		t.Fatalf("Run() errors = %v, want none", errs)
	}
	if len(events) != 2 {
		t.Fatalf("Run() yielded %d events, want 2", len(events))
	}
	call, resp := events[0], events[1]
	if call.Author != "task_int_agent" || call.Content == nil || len(call.Content.Parts) == 0 ||
		call.Content.Parts[0].FunctionCall == nil || call.Content.Parts[0].FunctionCall.Name != "finish_task" || call.Output != nil {
		t.Errorf("event 0 = author %q content %+v output %v, want the finish_task call by %q with no output", call.Author, call.Content, call.Output, "task_int_agent")
	}
	if resp.Author != "task_int_agent" || resp.Content == nil || len(resp.Content.Parts) == 0 ||
		resp.Content.Parts[0].FunctionResponse == nil || resp.Content.Parts[0].FunctionResponse.Name != "finish_task" {
		t.Fatalf("event 1 = author %q content %+v, want the finish_task response by %q", resp.Author, resp.Content, "task_int_agent")
	}
	if diff := cmp.Diff(map[string]any{"result": "Task completed."}, resp.Content.Parts[0].FunctionResponse.Response); diff != "" {
		t.Errorf("finish_task response mismatch (-want +got):\n%s", diff)
	}
	if resp.Output != 42 {
		t.Errorf("event 1 Output = %#v, want int 42", resp.Output)
	}
	if diff := cmp.Diff(map[string]any{"answer_key": 42}, resp.Actions.StateDelta); diff != "" {
		t.Errorf("event 1 StateDelta mismatch (-want +got):\n%s", diff)
	}
	got, err := storedNodeTestSession(t, ctx, svc).State().Get("answer_key")
	if err != nil {
		t.Fatalf("State().Get(answer_key) error = %v", err)
	}
	if got != 42 {
		t.Errorf("state[answer_key] = %#v, want int 42", got)
	}
}

func TestRunner_InvalidRootMode_FailsBeforeSessionReadOrCreate(t *testing.T) {
	ctx := t.Context()
	svc := session.InMemoryService()
	a, err := llmagent.New(llmagent.Config{
		Name:  "single_turn_root",
		Mode:  llmagent.ModeSingleTurn,
		Model: &scriptedModel{responses: []*genai.Content{genai.NewContentFromText("ok", genai.RoleModel)}},
	})
	if err != nil {
		t.Fatalf("llmagent.New() error = %v", err)
	}
	r, err := runner.New(runner.Config{
		AppName:           nodeTestApp,
		Agent:             a,
		SessionService:    svc,
		AutoCreateSession: true,
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	events, errs := collectRun(r.Run(ctx, nodeTestUser, "missing_session", userText("hi"), agent.RunConfig{}))

	const wantErr = "LlmAgent as root agent must have mode='chat' or 'task', but got mode='single_turn'"
	if len(events) != 0 {
		t.Errorf("Run() yielded %d events, want 0", len(events))
	}
	if len(errs) != 1 || errs[0].Error() != wantErr {
		t.Errorf("Run() errors = %v, want exactly [%q]", errs, wantErr)
	}
	if _, err := svc.Get(ctx, &session.GetRequest{AppName: nodeTestApp, UserID: nodeTestUser, SessionID: "missing_session"}); !errors.Is(err, session.ErrNotFound) {
		t.Errorf("Get(missing_session) error = %v, want session.ErrNotFound (no session created)", err)
	}
}

// TestRunner_Node_FunctionResponseRestoresSubAgentBranch pins the branch
// restore on the node path's first route: a function response answering a
// sub-agent's call runs that sub-agent on the branch of the call (adk-python
// workflow/_node_runner_utils.py:117-121).
func TestRunner_Node_FunctionResponseRestoresSubAgentBranch(t *testing.T) {
	ctx := t.Context()
	svc := session.InMemoryService()
	root := seedPendingSubAgentCall(t, ctx, svc)
	r := newNodeTestRunner(t, root, svc)
	answer := &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{
		{FunctionResponse: &genai.FunctionResponse{ID: "fc_1", Name: "ask", Response: map[string]any{"status": "done"}}},
	}}

	events, errs := collectRun(r.Run(ctx, nodeTestUser, nodeTestSession, answer, agent.RunConfig{}))

	if len(errs) != 0 {
		t.Fatalf("Run() errors = %v, want none", errs)
	}
	if len(events) != 1 {
		t.Fatalf("Run() yielded %d events, want 1", len(events))
	}
	if got := events[0]; got.Author != "sub_agent" || got.InvocationID != "inv_prev" || got.Branch != "root_agent.sub_agent" {
		t.Errorf("yielded event = author %q invocation %q branch %q, want author %q invocation %q branch %q",
			got.Author, got.InvocationID, got.Branch, "sub_agent", "inv_prev", "root_agent.sub_agent")
	}
}

// TestRunner_TaskRoot_RunsItselfWhenChatSubAgentSpokeLast pins that a
// task-mode root never hands a new message to the sub-agent that spoke last:
// only a chat root picks the agent from history (adk-python
// runners.py:1803-1818, Runner._uses_legacy_sub_agent_picker).
func TestRunner_TaskRoot_RunsItselfWhenChatSubAgentSpokeLast(t *testing.T) {
	ctx := t.Context()
	svc := session.InMemoryService()
	newNodeTestSession(t, ctx, svc)
	appendSeedEvent(t, ctx, svc, &session.Event{
		ID: "h1", InvocationID: "inv_prev", Author: "helper",
		LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("helper said", genai.RoleModel)},
	})
	root, err := llmagent.New(llmagent.Config{
		Name:      "task_root",
		Mode:      llmagent.ModeTask,
		SubAgents: []agent.Agent{textReplyLlmAgent(t, "helper", "helper_reply")},
		Model: &scriptedModel{responses: []*genai.Content{
			genai.NewContentFromFunctionCall("finish_task", map[string]any{"result": "done"}, genai.RoleModel),
		}},
	})
	if err != nil {
		t.Fatalf("llmagent.New() error = %v", err)
	}
	r := newNodeTestRunner(t, root, svc)

	events, errs := collectRun(r.Run(ctx, nodeTestUser, nodeTestSession, userText("go"), agent.RunConfig{}))

	if len(errs) != 0 {
		t.Fatalf("Run() errors = %v, want none", errs)
	}
	var authors []string
	for _, ev := range events {
		authors = append(authors, ev.Author)
	}
	if diff := cmp.Diff([]string{"task_root", "task_root"}, authors); diff != "" {
		t.Errorf("yielded event authors mismatch (-want +got):\n%s", diff)
	}
}

// newNodeRootRunner returns a runner whose root is the node n.
func newNodeRootRunner(t *testing.T, n workflow.Node, svc session.Service, plugins ...*plugin.Plugin) *runner.Runner {
	t.Helper()
	r, err := runner.New(runner.Config{
		AppName:        nodeTestApp,
		Node:           n,
		SessionService: svc,
		PluginConfig:   runner.PluginConfig{Plugins: plugins},
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	return r
}

// funcNode is a node whose body the test supplies.
type funcNode struct {
	workflow.BaseNode
	run func(ctx agent.Context, input any, yield func(*session.Event, error) bool)
}

func newFuncNode(name string, run func(ctx agent.Context, input any, yield func(*session.Event, error) bool)) *funcNode {
	return &funcNode{BaseNode: workflow.NewBaseNode(name, "", workflow.NodeConfig{}), run: run}
}

func (n *funcNode) Run(ctx agent.Context, input any) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		n.run(ctx, input, yield)
	}
}

// inputRecorder records node inputs by node name. Nodes run on scheduler
// goroutines, so it is guarded by a mutex.
type inputRecorder struct {
	mu     sync.Mutex
	inputs map[string][]any
}

func (r *inputRecorder) record(name string, in any) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.inputs == nil {
		r.inputs = map[string][]any{}
	}
	r.inputs[name] = append(r.inputs[name], in)
}

func (r *inputRecorder) recorded() map[string][]any {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.inputs
}

func TestRunner_Node_YieldUserMessageWithStateDeltaAndNilMessage(t *testing.T) {
	ctx := t.Context()
	svc := session.InMemoryService()
	newNodeTestSession(t, ctx, svc)
	noop := newFuncNode("noop", func(agent.Context, any, func(*session.Event, error) bool) {})
	r := newNodeRootRunner(t, noop, svc)

	events, errs := collectRun(r.Run(ctx, nodeTestUser, nodeTestSession, nil, agent.RunConfig{},
		runner.WithYieldUserMessage(), runner.WithStateDelta(map[string]any{"k": "v"})))

	if len(errs) > 0 {
		t.Fatalf("Run() errors = %v", errs)
	}
	if len(events) == 0 {
		t.Fatal("Run() yielded no events, want the state-delta user event first")
	}
	first := events[0]
	if first.Author != "user" || first.Content != nil {
		t.Errorf("first event author/content = %q/%v, want %q/nil", first.Author, first.Content, "user")
	}
	if diff := cmp.Diff(map[string]any{"k": "v"}, first.Actions.StateDelta); diff != "" {
		t.Errorf("first event state delta (-want +got):\n%s", diff)
	}
	got, err := storedNodeTestSession(t, ctx, svc).State().Get("k")
	if err != nil || got != "v" {
		t.Errorf("session state[k] = %v, %v; want %q", got, err, "v")
	}

	// An empty delta carries nothing, so no event is stored or yielded
	// (adk-python's `elif state_delta:`).
	events, errs = collectRun(r.Run(ctx, nodeTestUser, nodeTestSession, nil, agent.RunConfig{},
		runner.WithYieldUserMessage(), runner.WithStateDelta(map[string]any{})))
	if len(errs) > 0 || len(events) != 0 {
		t.Errorf("Run() with an empty delta = %d events, errors %v; want none", len(events), errs)
	}
	if got := storedNodeTestSession(t, ctx, svc).Events().Len(); got != 1 {
		t.Errorf("session holds %d events, want only the first run's state-delta event", got)
	}
}

func TestRunner_Node_BeforeRunCallback_ReturnsContent(t *testing.T) {
	ctx := t.Context()
	svc := session.InMemoryService()
	newNodeTestSession(t, ctx, svc)
	runs := 0
	n := newFuncNode("myNode", func(agent.Context, any, func(*session.Event, error) bool) { runs++ })
	p, err := plugin.New(plugin.Config{
		Name: "short_circuit",
		BeforeRunCallback: func(agent.InvocationContext) (*genai.Content, error) {
			return genai.NewContentFromText("short-circuit", genai.RoleModel), nil
		},
	})
	if err != nil {
		t.Fatalf("plugin.New() error = %v", err)
	}
	r := newNodeRootRunner(t, n, svc, p)

	events, errs := collectRun(r.Run(ctx, nodeTestUser, nodeTestSession, genai.NewContentFromText("hi", genai.RoleUser), agent.RunConfig{}))

	if len(errs) > 0 {
		t.Fatalf("Run() errors = %v", errs)
	}
	if len(events) != 1 {
		t.Fatalf("Run() yielded %d events, want 1", len(events))
	}
	if events[0].Author != "model" || eventText(events[0]) != "short-circuit" {
		t.Errorf("reply author/text = %q/%q, want %q/%q", events[0].Author, eventText(events[0]), "model", "short-circuit")
	}
	if events[0].CustomMetadata[plugininternal.BeforeRunReplyKey] != true {
		t.Errorf("reply custom metadata = %v, want key %q set", events[0].CustomMetadata, plugininternal.BeforeRunReplyKey)
	}
	if runs != 0 {
		t.Errorf("node ran %d times, want 0", runs)
	}
}

func TestRunner_BareNodeRoot_ReceivesContentAndOriginalContentOnResume(t *testing.T) {
	ctx := t.Context()
	svc := session.InMemoryService()
	newNodeTestSession(t, ctx, svc)
	rec := &inputRecorder{}
	n := newFuncNode("approver", func(ctx agent.Context, input any, yield func(*session.Event, error) bool) {
		rec.record("approver", input)
		if _, ok := ctx.ResumedInput("fc-1"); ok {
			in, _ := input.(*genai.Content)
			ev := session.NewEvent(ctx, ctx.InvocationID())
			ev.Output = "resumed with prompt: " + in.Parts[0].Text
			yield(ev, nil)
			return
		}
		ev := session.NewEvent(ctx, ctx.InvocationID())
		ev.Content = &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{
			{FunctionCall: &genai.FunctionCall{ID: "fc-1", Name: "approve"}},
		}}
		ev.LongRunningToolIDs = []string{"fc-1"}
		yield(ev, nil)
	})
	r := newNodeRootRunner(t, n, svc)

	turn1, errs := collectRun(r.Run(ctx, nodeTestUser, nodeTestSession,
		&genai.Content{Parts: []*genai.Part{{Text: "initial prompt"}}}, agent.RunConfig{}))
	if len(errs) > 0 {
		t.Fatalf("turn 1 Run() errors = %v", errs)
	}
	var paused bool
	for _, ev := range turn1 {
		if cmp.Equal(ev.LongRunningToolIDs, []string{"fc-1"}) {
			paused = true
		}
	}
	if !paused {
		t.Error("turn 1 yielded no event with LongRunningToolIDs [fc-1]")
	}

	turn2, errs := collectRun(r.Run(ctx, nodeTestUser, nodeTestSession, &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{
		{FunctionResponse: &genai.FunctionResponse{ID: "fc-1", Name: "approve", Response: map[string]any{"ok": true}}},
	}}, agent.RunConfig{}))
	if len(errs) > 0 {
		t.Fatalf("turn 2 Run() errors = %v", errs)
	}

	original := &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{{Text: "initial prompt"}}}
	if diff := cmp.Diff(map[string][]any{"approver": {original, original}}, rec.recorded()); diff != "" {
		t.Errorf("node root inputs over both turns (-want +got):\n%s", diff)
	}
	var outputs []any
	for _, ev := range turn2 {
		if ev.Output != nil {
			outputs = append(outputs, ev.Output)
		}
	}
	if diff := cmp.Diff([]any{"resumed with prompt: initial prompt"}, outputs); diff != "" {
		t.Errorf("turn 2 outputs (-want +got):\n%s", diff)
	}
}

func TestRunner_NodePath_ResultWrappedAnswerMatchesRequestSchema(t *testing.T) {
	// newAsker returns the node "asker", which requests a string answer and,
	// on re-entry, records the answer it receives and outputs "done".
	newAsker := func(rec *inputRecorder) workflow.Node {
		rerun := true
		return &funcNode{
			BaseNode: workflow.NewBaseNode("asker", "", workflow.NodeConfig{RerunOnResume: &rerun}),
			run: func(ctx agent.Context, _ any, yield func(*session.Event, error) bool) {
				if v, ok := ctx.ResumedInput("approval"); ok {
					rec.record("asker", v)
					ev := session.NewEvent(ctx, ctx.InvocationID())
					ev.Output = "done"
					yield(ev, nil)
					return
				}
				yield(workflow.NewRequestInputEvent(ctx, session.RequestInput{
					InterruptID:    "approval",
					ResponseSchema: &jsonschema.Schema{Type: "string"},
				}), nil)
			},
		}
	}
	tests := []struct {
		name      string
		newRunner func(t *testing.T, asker workflow.Node, svc session.Service) *runner.Runner
		// wantAnswers is what the asker receives on re-entry; wantTexts is
		// the text of the second turn's events.
		wantAnswers map[string][]any
		wantTexts   []string
	}{
		{
			name: "node_root",
			newRunner: func(t *testing.T, asker workflow.Node, svc session.Service) *runner.Runner {
				return newNodeRootRunner(t, asker, svc)
			},
			wantAnswers: map[string][]any{"asker": {"approved"}},
		},
		{
			name: "workflow_node_root",
			newRunner: func(t *testing.T, asker workflow.Node, svc session.Service) *runner.Runner {
				wf, err := workflow.NewWorkflowNode("wf", []workflow.Edge{{From: workflow.Start, To: asker}})
				if err != nil {
					t.Fatalf("NewWorkflowNode() error = %v", err)
				}
				return newNodeRootRunner(t, wf, svc)
			},
			wantAnswers: map[string][]any{"asker": {"approved"}},
		},
		{
			name: "dynamic_node_root",
			newRunner: func(t *testing.T, asker workflow.Node, svc session.Service) *runner.Runner {
				dyn := workflow.NewDynamicNode("dyn", func(ctx agent.Context, in any, _ func(*session.Event) error) (any, error) {
					return workflow.RunNode[any](ctx, asker, in)
				}, workflow.NodeConfig{})
				return newNodeRootRunner(t, dyn, svc)
			},
			wantAnswers: map[string][]any{"asker": {"approved"}},
		},
		{
			// The answer goes to the LlmAgent, which calls the model again
			// without re-running the tool.
			name: "llm_agent_root_tool_runs_node",
			newRunner: func(t *testing.T, asker workflow.Node, svc session.Service) *runner.Runner {
				type noArgs struct{}
				runAsker, err := functiontool.New(functiontool.Config{
					Name:        "run_asker",
					Description: "runs the asker node",
				}, func(tc agent.Context, _ noArgs) (string, error) {
					return workflow.RunNode[string](tc, asker, nil)
				})
				if err != nil {
					t.Fatalf("functiontool.New() error = %v", err)
				}
				a, err := llmagent.New(llmagent.Config{
					Name: "approver",
					Model: &scriptedModel{responses: []*genai.Content{
						genai.NewContentFromFunctionCall("run_asker", map[string]any{}, genai.RoleModel),
						genai.NewContentFromText("finished", genai.RoleModel),
					}},
					Tools: []tool.Tool{runAsker},
				})
				if err != nil {
					t.Fatalf("llmagent.New() error = %v", err)
				}
				return newNodeTestRunner(t, a, svc)
			},
			wantTexts: []string{"finished"},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ctx := t.Context()
			svc := session.InMemoryService()
			newNodeTestSession(t, ctx, svc)
			rec := &inputRecorder{}
			r := tc.newRunner(t, newAsker(rec), svc)
			if _, errs := collectRun(r.Run(ctx, nodeTestUser, nodeTestSession, userText("go"), agent.RunConfig{})); len(errs) > 0 {
				t.Fatalf("turn 1 Run() errors = %v", errs)
			}

			// adk-python's FunctionResponse shape for a RequestInput answer.
			answer := genai.NewContentFromFunctionResponse(workflow.WorkflowInputFunctionCallName, map[string]any{"result": "approved"}, genai.RoleUser)
			answer.Parts[0].FunctionResponse.ID = "approval"
			events, errs := collectRun(r.Run(ctx, nodeTestUser, nodeTestSession, answer, agent.RunConfig{}))
			if len(errs) > 0 {
				t.Fatalf("turn 2 Run() errors = %v", errs)
			}

			if diff := cmp.Diff(tc.wantAnswers, rec.recorded()); diff != "" {
				t.Errorf("answers the asker received (-want +got):\n%s", diff)
			}
			var texts []string
			for _, ev := range events {
				if text := eventText(ev); text != "" {
					texts = append(texts, text)
				}
			}
			if diff := cmp.Diff(tc.wantTexts, texts); diff != "" {
				t.Errorf("turn 2 texts (-want +got):\n%s", diff)
			}
		})
	}
}

func TestRunner_RootPathScoping_BareNodeVsLlmAgent(t *testing.T) {
	// nodeInfoOf returns the NodeInfo of the first yielded event whose
	// Output is want.
	nodeInfoOf := func(t *testing.T, r *runner.Runner, ctx context.Context, want any) *session.NodeInfo {
		t.Helper()
		events, errs := collectRun(r.Run(ctx, nodeTestUser, nodeTestSession, userText("hi"), agent.RunConfig{}))
		if len(errs) > 0 {
			t.Fatalf("Run() errors = %v", errs)
		}
		for _, ev := range events {
			if ev.Output == want {
				if ev.NodeInfo == nil {
					t.Fatalf("output event %q has no NodeInfo", want)
				}
				return ev.NodeInfo
			}
		}
		t.Fatalf("Run() yielded no event with output %q", want)
		return nil
	}

	t.Run("node_root", func(t *testing.T) {
		ctx := t.Context()
		svc := session.InMemoryService()
		newNodeTestSession(t, ctx, svc)
		echo := workflow.NewFunctionNode("echo", func(_ agent.Context, in *genai.Content) (string, error) {
			return "Echo: " + in.Parts[0].Text, nil
		}, workflow.NodeConfig{})

		info := nodeInfoOf(t, newNodeRootRunner(t, echo, svc), ctx, "Echo: hi")

		if info.Path != "echo@1" {
			t.Errorf("NodeInfo.Path = %q, want %q", info.Path, "echo@1")
		}
	})

	t.Run("dynamic_node_root_delegates_output", func(t *testing.T) {
		ctx := t.Context()
		svc := session.InMemoryService()
		newNodeTestSession(t, ctx, svc)
		child := workflow.NewFunctionNode("child", func(agent.Context, any) (string, error) {
			return "from child", nil
		}, workflow.NodeConfig{})
		parent := workflow.NewDynamicNode("parent", func(ctx agent.Context, in any, _ func(*session.Event) error) (any, error) {
			return workflow.RunNode[string](ctx, child, in, workflow.WithUseAsOutput())
		}, workflow.NodeConfig{})

		info := nodeInfoOf(t, newNodeRootRunner(t, parent, svc), ctx, "from child")

		want := session.NodeInfo{Path: "parent@1/child@1", OutputFor: []string{"parent@1/child@1", "parent@1"}}
		if diff := cmp.Diff(want, *info); diff != "" {
			t.Errorf("child output NodeInfo (-want +got):\n%s", diff)
		}
	})

	t.Run("llm_agent_root", func(t *testing.T) {
		ctx := t.Context()
		svc := session.InMemoryService()
		newNodeTestSession(t, ctx, svc)
		r := newNodeTestRunner(t, textReplyLlmAgent(t, "greeter", "hello"), svc)

		events, errs := collectRun(r.Run(ctx, nodeTestUser, nodeTestSession, userText("hi"), agent.RunConfig{}))
		if len(errs) > 0 {
			t.Fatalf("Run() errors = %v", errs)
		}
		var paths []string
		for _, ev := range events {
			if eventText(ev) == "hello" && ev.NodeInfo != nil {
				paths = append(paths, ev.NodeInfo.Path)
			}
		}
		if diff := cmp.Diff([]string{"greeter"}, paths); diff != "" {
			t.Errorf("agent reply NodeInfo.Path (-want +got):\n%s", diff)
		}
	})
}

func TestRunner_BareNodeRoot_MixedMessageRefusalAndScoping(t *testing.T) {
	const wantErr = "message cannot contain both function responses and text: function responses resume an existing invocation while text starts a new one"
	mixed := func(id string) *genai.Content {
		return &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{
			{Text: "hi"},
			{FunctionResponse: &genai.FunctionResponse{ID: id, Name: "t"}},
		}}
	}
	// newEcho returns the node root "echo" and a pointer to its run count.
	newEcho := func() (workflow.Node, *int) {
		runs := 0
		return workflow.NewFunctionNode("echo", func(_ agent.Context, in *genai.Content) (string, error) {
			runs++
			return "Echo: " + in.Parts[0].Text, nil
		}, workflow.NodeConfig{}), &runs
	}

	tests := []struct {
		name    string
		msg     *genai.Content
		plugins func(t *testing.T) []*plugin.Plugin
	}{
		{name: "caller_message", msg: mixed("fc-1")},
		{
			// The caller's message is checked before OnUserMessageCallback,
			// so a plugin that would clean it is never called (adk-python
			// workflow/_node_runner_utils.py).
			name: "caller_message_before_plugin",
			msg:  mixed("fc-1"),
			plugins: func(t *testing.T) []*plugin.Plugin {
				p, err := plugin.New(plugin.Config{
					Name: "clean_message",
					OnUserMessageCallback: func(agent.InvocationContext, *genai.Content) (*genai.Content, error) {
						t.Error("OnUserMessageCallback called for a refused caller message")
						return userText("clean"), nil
					},
				})
				if err != nil {
					t.Fatalf("plugin.New() error = %v", err)
				}
				return []*plugin.Plugin{p}
			},
		},
		{
			name: "plugin_replacement",
			msg:  userText("hi"),
			plugins: func(t *testing.T) []*plugin.Plugin {
				return []*plugin.Plugin{replaceMessagePlugin(t, mixed("fc-1"))}
			},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name+"_refused", func(t *testing.T) {
			ctx := t.Context()
			svc := session.InMemoryService()
			newNodeTestSession(t, ctx, svc)
			echo, runs := newEcho()
			var plugins []*plugin.Plugin
			if tc.plugins != nil {
				plugins = tc.plugins(t)
			}
			r := newNodeRootRunner(t, echo, svc, plugins...)

			events, errs := collectRun(r.Run(ctx, nodeTestUser, nodeTestSession, tc.msg, agent.RunConfig{}))

			if len(errs) != 1 || errs[0].Error() != wantErr {
				t.Errorf("Run() errors = %v, want [%s]", errs, wantErr)
			}
			if len(events) != 0 || *runs != 0 {
				t.Errorf("Run() yielded %d events and ran the node %d times, want 0 and 0", len(events), *runs)
			}
			if got := storedNodeTestSession(t, ctx, svc).Events().Len(); got != 0 {
				t.Errorf("session holds %d events, want 0", got)
			}
		})
	}

	t.Run("function_response_without_id_accepted", func(t *testing.T) {
		ctx := t.Context()
		svc := session.InMemoryService()
		newNodeTestSession(t, ctx, svc)
		echo, runs := newEcho()
		r := newNodeRootRunner(t, echo, svc)

		events, errs := collectRun(r.Run(ctx, nodeTestUser, nodeTestSession, mixed(""), agent.RunConfig{}))

		if len(errs) > 0 {
			t.Fatalf("Run() errors = %v", errs)
		}
		var outputs []any
		for _, ev := range events {
			if ev.Output != nil {
				outputs = append(outputs, ev.Output)
			}
		}
		if diff := cmp.Diff([]any{"Echo: hi"}, outputs); diff != "" || *runs != 1 {
			t.Errorf("node ran %d times, outputs (-want +got):\n%s", *runs, diff)
		}
	})

	t.Run("classic_agent_root_accepts_mixed_message", func(t *testing.T) {
		ctx := t.Context()
		svc := session.InMemoryService()
		newNodeTestSession(t, ctx, svc)
		subRuns := 0
		sub, err := agent.New(agent.Config{
			Name: "sub",
			Run: func(agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(func(*session.Event, error) bool) { subRuns++ }
			},
		})
		if err != nil {
			t.Fatalf("agent.New() error = %v", err)
		}
		seq, err := sequentialagent.New(sequentialagent.Config{AgentConfig: agent.Config{Name: "seq", SubAgents: []agent.Agent{sub}}})
		if err != nil {
			t.Fatalf("sequentialagent.New() error = %v", err)
		}
		r := newNodeTestRunner(t, seq, svc)

		_, errs := collectRun(r.Run(ctx, nodeTestUser, nodeTestSession, mixed("fc-1"), agent.RunConfig{}))

		if len(errs) > 0 {
			t.Errorf("Run() errors = %v, want none", errs)
		}
		if subRuns != 1 {
			t.Errorf("sub-agent ran %d times, want 1", subRuns)
		}
	})
}

func TestRunner_WorkflowNodeRoot_SeedResolvedPerInnerSuccessor(t *testing.T) {
	ctx := t.Context()
	svc := session.InMemoryService()
	newNodeTestSession(t, ctx, svc)
	rec := &inputRecorder{}
	anyNode := workflow.NewFunctionNode("any_node", func(_ agent.Context, in any) (string, error) {
		rec.record("any_node", in)
		return "a", nil
	}, workflow.NodeConfig{})
	contentNode := workflow.NewFunctionNode("content_node", func(_ agent.Context, in *genai.Content) (string, error) {
		rec.record("content_node", in)
		return "c", nil
	}, workflow.NodeConfig{})
	join := workflow.NewJoinNode("join")
	wf, err := workflow.NewWorkflowNode("wf", []workflow.Edge{
		{From: workflow.Start, To: anyNode},
		{From: workflow.Start, To: contentNode},
		{From: anyNode, To: join},
		{From: contentNode, To: join},
	})
	if err != nil {
		t.Fatalf("workflow.NewWorkflowNode() error = %v", err)
	}
	r := newNodeRootRunner(t, wf, svc)

	_, errs := collectRun(r.Run(ctx, nodeTestUser, nodeTestSession, genai.NewContentFromText("hi", genai.RoleUser), agent.RunConfig{}))

	if len(errs) > 0 {
		t.Errorf("Run() errors = %v, want none", errs)
	}
	want := map[string][]any{
		"any_node":     {"hi"},
		"content_node": {&genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{{Text: "hi"}}}},
	}
	if diff := cmp.Diff(want, rec.recorded()); diff != "" {
		t.Errorf("inner successor inputs (-want +got):\n%s", diff)
	}
}

// A node root runs on the root branch on every turn, as in adk-python,
// even after a child that ran on a sub-branch stored events under the
// node's name.
func TestRunner_BareNodeRoot_RunsOnRootBranch(t *testing.T) {
	ctx := t.Context()
	svc := session.InMemoryService()
	newNodeTestSession(t, ctx, svc)
	child := newFuncNode("child", func(ctx agent.Context, _ any, yield func(*session.Event, error) bool) {
		// As an agent does, the child stamps its own branch.
		ev := session.NewEvent(ctx, ctx.InvocationID())
		ev.Branch = ctx.Branch()
		ev.Content = genai.NewContentFromText("child progress", genai.RoleModel)
		yield(ev, nil)
	})
	var branches []string
	parent := workflow.NewDynamicNode("parent", func(ctx agent.Context, in any, _ func(*session.Event) error) (any, error) {
		branches = append(branches, ctx.Branch())
		return workflow.RunNode[any](ctx, child, in, workflow.WithUseSubBranch())
	}, workflow.NodeConfig{})
	r := newNodeRootRunner(t, parent, svc)

	for _, text := range []string{"turn 1", "turn 2"} {
		if _, errs := collectRun(r.Run(ctx, nodeTestUser, nodeTestSession, userText(text), agent.RunConfig{})); len(errs) > 0 {
			t.Fatalf("Run(%q) errors = %v", text, errs)
		}
	}

	if diff := cmp.Diff([]string{"", ""}, branches); diff != "" {
		t.Errorf("node root branch per turn (-want +got):\n%s", diff)
	}
}
