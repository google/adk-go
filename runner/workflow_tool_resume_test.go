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
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/agent/workflowagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
	"google.golang.org/adk/v2/tool/toolconfirmation"
	"google.golang.org/adk/v2/workflow"
)

// TestRunner_WorkflowAgentNodeConfirmationRepliesResumePausedGraph verifies
// that a confirmation reply routes to the paused LlmAgent node instead of
// starting a fresh graph run. Both approval and denial must execute the resume
// path; only approval should run the gated tool.
func TestRunner_WorkflowAgentNodeConfirmationRepliesResumePausedGraph(t *testing.T) {
	for _, tc := range []struct {
		name        string
		confirmed   bool
		wantToolRun int32
	}{
		{name: "confirmed", confirmed: true, wantToolRun: 1},
		{name: "denied", confirmed: false, wantToolRun: 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			f := newConfirmationGraphFixture(t, &scriptedModel{responses: []*genai.Content{
				genai.NewContentFromFunctionCall("confirm_action", map[string]any{}, "model"),
				genai.NewContentFromText("resume complete", "model"),
			}})

			turn1 := drainRunner(t, f.runner.Run(
				t.Context(), nodeTestUser, nodeTestSession,
				userText("start"), agent.RunConfig{},
			))
			callID, callName := findLongRunningInterrupt(turn1)
			if callID == "" {
				t.Fatalf("turn 1 produced no interrupt; events:\n%s", debugEvents(turn1))
			}
			if callName != toolconfirmation.FunctionCallName {
				t.Fatalf("interrupt name = %q, want %q", callName, toolconfirmation.FunctionCallName)
			}
			if got := f.upstreamRuns.Load(); got != 1 {
				t.Fatalf("upstream runs after turn 1 = %d, want 1", got)
			}

			state := f.reconstructedState(t, invocationID(turn1))
			ns := state.Nodes["gated"]
			if ns == nil {
				t.Fatalf("paused state has no gated node: %+v", state.Nodes)
			}
			if ns.Status != workflow.NodeWaiting || !slices.Contains(ns.Interrupts, callID) {
				t.Fatalf("gated state = %+v, want NodeWaiting on %q", ns, callID)
			}

			turn2 := drainRunner(t, f.runner.Run(
				t.Context(), nodeTestUser, nodeTestSession,
				confirmationReply(callID, tc.confirmed), agent.RunConfig{},
			))
			if id, _ := findLongRunningInterrupt(turn2); id != "" {
				t.Errorf("turn 2 unexpectedly re-paused on %q", id)
			}
			if got := f.upstreamRuns.Load(); got != 1 {
				t.Errorf("upstream runs after resume = %d, want 1 (resume restarted the graph)", got)
			}
			if got := f.toolRuns.Load(); got != tc.wantToolRun {
				t.Errorf("confirmed tool runs = %d, want %d; turn events:\n%s\nsession events:\n%s",
					got, tc.wantToolRun, debugEvents(turn2), debugSession(f.svc, t))
			}
			if !eventsContainText(turn2, "resume complete") {
				t.Errorf("resume did not reach the model's final response; events:\n%s", debugEvents(turn2))
			}
		})
	}
}

// TestRunner_WorkflowAgentNodeInputReplyResumesPausedGraph verifies that the
// original adk_request_input resume path still advances the graph without
// replaying a completed upstream node.
func TestRunner_WorkflowAgentNodeInputReplyResumesPausedGraph(t *testing.T) {
	var upstreamRuns, handlerRuns atomic.Int32
	upstream := workflow.NewFunctionNode("upstream", func(agent.Context, any) (string, error) {
		upstreamRuns.Add(1)
		return "draft", nil
	}, workflow.NodeConfig{})
	asker := newHitlAsker("asker", "input-1", false)
	handler := workflow.NewFunctionNode("handler", func(_ agent.Context, input string) (string, error) {
		handlerRuns.Add(1)
		return "handled:" + input, nil
	}, workflow.NodeConfig{})

	r, _ := newWorkflowRunnerWithService(t, workflow.Chain(workflow.Start, upstream, asker, handler))
	turn1 := drainRunner(t, r.Run(
		t.Context(), nodeTestUser, nodeTestSession,
		userText("start"), agent.RunConfig{},
	))
	callID, callName := findLongRunningInterrupt(turn1)
	if callID != "input-1" || callName != workflow.WorkflowInputFunctionCallName {
		t.Fatalf("turn 1 interrupt = (%q, %q), want (%q, %q)", callID, callName, "input-1", workflow.WorkflowInputFunctionCallName)
	}

	drainRunner(t, r.Run(
		t.Context(), nodeTestUser, nodeTestSession,
		resumeContent(callID, callName, "approved"), agent.RunConfig{},
	))
	if got := upstreamRuns.Load(); got != 1 {
		t.Errorf("upstream runs = %d, want 1", got)
	}
	if got := handlerRuns.Load(); got != 1 {
		t.Errorf("handler runs = %d, want 1", got)
	}
}

// TestRunner_WorkflowAgentNodeDuplicateConfirmationReplyIsNoop verifies that
// replaying an already-consumed confirmation does not re-enter the paused
// LlmAgent or replay the completed upstream node.
func TestRunner_WorkflowAgentNodeDuplicateConfirmationReplyIsNoop(t *testing.T) {
	f := newConfirmationGraphFixture(t, &scriptedModel{responses: []*genai.Content{
		genai.NewContentFromFunctionCall("confirm_action", map[string]any{}, "model"),
		genai.NewContentFromText("resume complete", "model"),
	}})

	turn1 := drainRunner(t, f.runner.Run(
		t.Context(), nodeTestUser, nodeTestSession,
		userText("start"), agent.RunConfig{},
	))
	callID, _ := findLongRunningInterrupt(turn1)
	if callID == "" {
		t.Fatalf("turn 1 produced no interrupt; events:\n%s", debugEvents(turn1))
	}

	reply := confirmationReply(callID, true)
	drainRunner(t, f.runner.Run(t.Context(), nodeTestUser, nodeTestSession, reply, agent.RunConfig{}))
	afterResumeUpstream := f.upstreamRuns.Load()
	afterResumeTool := f.toolRuns.Load()

	err := drainRunnerErr(t, f.runner.Run(t.Context(), nodeTestUser, nodeTestSession, reply, agent.RunConfig{}))
	if !errors.Is(err, workflow.ErrNothingToResume) {
		t.Errorf("duplicate reply error = %v, want %v", err, workflow.ErrNothingToResume)
	}
	if got := f.upstreamRuns.Load(); got != afterResumeUpstream {
		t.Errorf("upstream runs after duplicate = %d, want %d", got, afterResumeUpstream)
	}
	if got := f.toolRuns.Load(); got != afterResumeTool {
		t.Errorf("confirmed tool runs after duplicate = %d, want %d", got, afterResumeTool)
	}
}

// TestRunner_WorkflowAgentNodeResumeCancellationDoesNotRestartGraph verifies
// that cancelling an in-flight resume propagates context.Canceled without
// falling back to a fresh graph run.
func TestRunner_WorkflowAgentNodeResumeCancellationDoesNotRestartGraph(t *testing.T) {
	m := newCancelingResumeModel()
	f := newConfirmationGraphFixture(t, m)

	turn1 := drainRunner(t, f.runner.Run(
		t.Context(), nodeTestUser, nodeTestSession,
		userText("start"), agent.RunConfig{},
	))
	callID, _ := findLongRunningInterrupt(turn1)
	if callID == "" {
		t.Fatalf("turn 1 produced no interrupt; events:\n%s", debugEvents(turn1))
	}

	ctx, cancel := context.WithCancel(t.Context())
	errCh := make(chan error, 1)
	go func() {
		errCh <- drainRunnerErr(t, f.runner.Run(
			ctx, nodeTestUser, nodeTestSession,
			confirmationReply(callID, true), agent.RunConfig{},
		))
	}()

	select {
	case <-m.resumeStarted:
	case <-time.After(5 * time.Second):
		cancel()
		t.Fatal("resume never reached the second model call")
	}
	cancel()

	select {
	case err := <-errCh:
		if !errors.Is(err, context.Canceled) {
			t.Errorf("resume error = %v, want context.Canceled", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("cancelled resume did not return")
	}
	if got := f.upstreamRuns.Load(); got != 1 {
		t.Errorf("upstream runs after cancellation = %d, want 1", got)
	}
	if got := f.toolRuns.Load(); got != 1 {
		t.Errorf("confirmed tool runs after cancellation = %d, want 1", got)
	}
}

type confirmationGraphFixture struct {
	runner       *runner.Runner
	svc          session.Service
	wf           *workflow.Workflow
	upstreamRuns atomic.Int32
	toolRuns     atomic.Int32
}

func newConfirmationGraphFixture(t *testing.T, m model.LLM) *confirmationGraphFixture {
	t.Helper()
	f := &confirmationGraphFixture{}

	upstream := newCountingTextAgent(t, "upstream", &f.upstreamRuns, "prepared")
	confirmTool, err := functiontool.New(functiontool.Config{
		Name:                "confirm_action",
		Description:         "performs an action after confirmation",
		RequireConfirmation: true,
	}, func(agent.Context, struct{}) (map[string]string, error) {
		f.toolRuns.Add(1)
		return map[string]string{"result": "executed"}, nil
	})
	if err != nil {
		t.Fatalf("functiontool.New() error = %v", err)
	}
	gated, err := llmagent.New(llmagent.Config{
		Name:  "gated",
		Model: m,
		Tools: []tool.Tool{confirmTool},
	})
	if err != nil {
		t.Fatalf("llmagent.New() error = %v", err)
	}

	upstreamNode, err := workflow.NewAgentNode(upstream, workflow.NodeConfig{})
	if err != nil {
		t.Fatalf("workflow.NewAgentNode(upstream) error = %v", err)
	}
	gatedNode, err := workflow.NewAgentNode(gated, workflow.NodeConfig{})
	if err != nil {
		t.Fatalf("workflow.NewAgentNode(gated) error = %v", err)
	}
	edges := workflow.Chain(workflow.Start, upstreamNode, gatedNode)

	wfAgent, err := workflowagent.New(workflowagent.Config{
		Name:  workflowAgentName,
		Edges: edges,
	})
	if err != nil {
		t.Fatalf("workflowagent.New() error = %v", err)
	}
	f.wf, err = workflow.New(workflowAgentName, edges)
	if err != nil {
		t.Fatalf("workflow.New() error = %v", err)
	}

	f.svc = session.InMemoryService()
	newNodeTestSession(t, t.Context(), f.svc)
	f.runner = newNodeTestRunner(t, wfAgent, f.svc)
	return f
}

func (f *confirmationGraphFixture) reconstructedState(t *testing.T, invocationID string) *workflow.RunState {
	t.Helper()
	got, err := f.svc.Get(t.Context(), &session.GetRequest{
		AppName:   nodeTestApp,
		UserID:    nodeTestUser,
		SessionID: nodeTestSession,
	})
	if err != nil {
		t.Fatalf("session.Get() error = %v", err)
	}
	state, err := f.wf.ReconstructRunState(got.Session, invocationID)
	if err != nil {
		t.Fatalf("ReconstructRunState() error = %v", err)
	}
	if state == nil {
		t.Fatal("ReconstructRunState() returned nil state")
	}
	return state
}

func newCountingTextAgent(t *testing.T, name string, runs *atomic.Int32, text string) agent.Agent {
	t.Helper()
	a, err := agent.New(agent.Config{
		Name: name,
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				runs.Add(1)
				ev := session.NewEvent(ctx, ctx.InvocationID())
				ev.Author = name
				ev.LLMResponse.Content = &genai.Content{
					Role:  genai.RoleModel,
					Parts: []*genai.Part{{Text: text}},
				}
				ev.Output = text
				yield(ev, nil)
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New(%q) error = %v", name, err)
	}
	return a
}

func newWorkflowRunnerWithService(t *testing.T, edges []workflow.Edge) (*runner.Runner, session.Service) {
	t.Helper()
	wfAgent, err := workflowagent.New(workflowagent.Config{
		Name:  workflowAgentName,
		Edges: edges,
	})
	if err != nil {
		t.Fatalf("workflowagent.New() error = %v", err)
	}
	svc := session.InMemoryService()
	newNodeTestSession(t, t.Context(), svc)
	return newNodeTestRunner(t, wfAgent, svc), svc
}

func confirmationReply(id string, confirmed bool) *genai.Content {
	return &genai.Content{
		Role: genai.RoleUser,
		Parts: []*genai.Part{{
			FunctionResponse: &genai.FunctionResponse{
				ID:       id,
				Name:     toolconfirmation.FunctionCallName,
				Response: map[string]any{"confirmed": confirmed},
			},
		}},
	}
}

func invocationID(events []*session.Event) string {
	for _, ev := range events {
		if ev != nil && ev.InvocationID != "" {
			return ev.InvocationID
		}
	}
	return ""
}

func eventsContainText(events []*session.Event, text string) bool {
	for _, ev := range events {
		if ev == nil || ev.LLMResponse.Content == nil {
			continue
		}
		for _, p := range ev.LLMResponse.Content.Parts {
			if p != nil && p.Text == text {
				return true
			}
		}
	}
	return false
}

func debugSession(svc session.Service, t *testing.T) string {
	t.Helper()
	got, err := svc.Get(t.Context(), &session.GetRequest{
		AppName:   nodeTestApp,
		UserID:    nodeTestUser,
		SessionID: nodeTestSession,
	})
	if err != nil {
		return "get session: " + err.Error()
	}
	events := got.Session.Events()
	out := make([]*session.Event, 0, events.Len())
	for i := 0; i < events.Len(); i++ {
		out = append(out, events.At(i))
	}
	return debugEvents(out)
}

func drainRunnerErr(t *testing.T, seq iter.Seq2[*session.Event, error]) error {
	t.Helper()
	for _, err := range seq {
		if err != nil {
			return err
		}
	}
	return nil
}

type cancelingResumeModel struct {
	calls         atomic.Int32
	resumeStarted chan struct{}
	startOnce     sync.Once
}

func newCancelingResumeModel() *cancelingResumeModel {
	return &cancelingResumeModel{resumeStarted: make(chan struct{})}
}

func (m *cancelingResumeModel) Name() string { return "canceling-resume" }

func (m *cancelingResumeModel) GenerateContent(ctx context.Context, _ *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		if m.calls.Add(1) == 1 {
			yield(&model.LLMResponse{Content: genai.NewContentFromFunctionCall("confirm_action", map[string]any{}, "model")}, nil)
			return
		}
		m.startOnce.Do(func() { close(m.resumeStarted) })
		<-ctx.Done()
		yield(nil, ctx.Err())
	}
}
