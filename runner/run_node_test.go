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
	"iter"
	"slices"
	"strings"
	"testing"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/agent/workflowagent"
	"google.golang.org/adk/v2/agent/workflowagents/sequentialagent"
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
