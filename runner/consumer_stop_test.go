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
	"sync"
	"sync/atomic"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/agent/workflowagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
	"google.golang.org/adk/v2/workflow"
)

// newCountingBookTool returns a "book" tool that adds one to bookings each
// time it runs.
func newCountingBookTool(t *testing.T, bookings *atomic.Int32) tool.Tool {
	t.Helper()
	book, err := functiontool.New(functiontool.Config{Name: "book", Description: "Books a table."},
		func(agent.Context, struct{}) (map[string]any, error) {
			bookings.Add(1)
			return map[string]any{"table": 7}, nil
		})
	if err != nil {
		t.Fatalf("functiontool.New() error = %v", err)
	}
	return book
}

func newBookingAssistant(t *testing.T, bookings *atomic.Int32) agent.Agent {
	t.Helper()
	a, err := llmagent.New(llmagent.Config{
		Name: "assistant",
		Model: &cannedModel{responses: []*genai.Content{
			exampleCall("call-1", "book"),
			genai.NewContentFromText("You have table 7.", genai.RoleModel),
		}},
		Tools: []tool.Tool{newCountingBookTool(t, bookings)},
	})
	if err != nil {
		t.Fatalf("llmagent.New() error = %v", err)
	}
	return a
}

func newBookingGraphRoot(t *testing.T, bookings *atomic.Int32) agent.Agent {
	t.Helper()
	assistant := newBookingAssistant(t, bookings)
	node, err := workflow.NewAgentNode(assistant, workflow.NodeConfig{})
	if err != nil {
		t.Fatalf("workflow.NewAgentNode() error = %v", err)
	}
	wf, err := workflowagent.New(workflowagent.Config{
		Name:      "wf",
		Edges:     workflow.Chain(workflow.Start, node),
		SubAgents: []agent.Agent{assistant},
	})
	if err != nil {
		t.Fatalf("workflowagent.New() error = %v", err)
	}
	return wf
}

func hasFunctionCall(ev *session.Event) bool {
	if ev.Partial || ev.Content == nil {
		return false
	}
	for _, p := range ev.Content.Parts {
		if p.FunctionCall != nil {
			return true
		}
	}
	return false
}

// A caller that breaks from the range loop at the event that carries a
// function call leaves the call stored and unanswered, and the tool does not
// run. Resuming the invocation by its ID runs the call for an LlmAgent root,
// and runs nothing for a graph workflow root, as adk-python's Workflow root
// does.
//
// The test guards the wiring through the public API and the resume (B2), not
// the scheduler fix itself: without the fix it fails in about two of three
// batches of 200 iterations, not every time. The forced tests in package
// workflow (TestScheduler_StoppedConsumerDoesNotLetNodeGoOn and its siblings)
// and TestRun_ParallelWorkerItem_CallerStop_ToolDoesNotRun are the proof of
// the fix.
func TestRun_BreakAtFunctionCall_ToolDoesNotRun(t *testing.T) {
	const iterations = 200
	tests := []struct {
		name            string
		newRoot         func(*testing.T, *atomic.Int32) agent.Agent
		wantStored      []string
		wantAfterResume int32
	}{
		{
			name:            "LlmAgent root",
			newRoot:         newBookingAssistant,
			wantStored:      []string{"user text Book a table", "assistant call book"},
			wantAfterResume: 1,
		},
		{
			name:            "graph workflow root",
			newRoot:         newBookingGraphRoot,
			wantStored:      []string{"user text Book a table", "assistant call book"},
			wantAfterResume: 0,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			for i := range iterations {
				var bookings atomic.Int32
				svc := session.InMemoryService()
				newNodeTestSession(t, t.Context(), svc)
				r, err := runner.New(runner.Config{
					AppName:        nodeTestApp,
					Agent:          tc.newRoot(t, &bookings),
					SessionService: svc,
					Resumable:      true,
				})
				if err != nil {
					t.Fatalf("runner.New() error = %v", err)
				}

				invocationID := ""
				for ev, err := range r.Run(t.Context(), nodeTestUser, nodeTestSession, userText("Book a table"), agent.RunConfig{}) {
					if err != nil {
						t.Fatalf("iteration %d: Run() error = %v", i, err)
					}
					if hasFunctionCall(ev) {
						invocationID = ev.InvocationID
						break
					}
				}
				if invocationID == "" {
					t.Fatalf("iteration %d: Run() yielded no function call event", i)
				}

				if got := bookings.Load(); got != 0 {
					t.Fatalf("iteration %d: book ran %d times after the caller stopped at its call, want 0", i, got)
				}
				if diff := cmp.Diff(tc.wantStored, describeEvents(sessionEvents(t, svc))); diff != "" {
					t.Fatalf("iteration %d: stored events after the break mismatch (-want +got):\n%s", i, diff)
				}

				runTurn(t, r, nil, runner.WithInvocationID(invocationID))
				if got := bookings.Load(); got != tc.wantAfterResume {
					t.Fatalf("iteration %d: book ran %d times after the resume, want %d", i, got, tc.wantAfterResume)
				}
			}
		})
	}
}

// waitingModel's first call reports that it started, waits until its
// context is done and then returns a call to "book" anyway, as a model that
// ignores cancellation would. Later calls return text.
type waitingModel struct {
	started   chan struct{}
	startOnce sync.Once
	calls     atomic.Int32
}

func (*waitingModel) Name() string { return "waiting" }

func (m *waitingModel) GenerateContent(ctx context.Context, _ *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		if m.calls.Add(1) > 1 {
			yield(&model.LLMResponse{Content: genai.NewContentFromText("Booked.", genai.RoleModel)}, nil)
			return
		}
		m.startOnce.Do(func() { close(m.started) })
		<-ctx.Done()
		yield(&model.LLMResponse{Content: exampleCall("call-1", "book")}, nil)
	}
}

// An LlmAgent run by a ParallelWorker item does not start a tool after the
// caller stopped reading, even when its model answers with a function call
// after the stop. The caller breaks at a sibling node's event, which the
// sibling emits only once the item's model call started; the model returns
// its call only once the stop canceled the item's context.
func TestRun_ParallelWorkerItem_CallerStop_ToolDoesNotRun(t *testing.T) {
	var bookings atomic.Int32
	m := &waitingModel{started: make(chan struct{})}
	booker, err := llmagent.New(llmagent.Config{
		Name:  "booker",
		Model: m,
		Tools: []tool.Tool{newCountingBookTool(t, &bookings)},
	})
	if err != nil {
		t.Fatalf("llmagent.New() error = %v", err)
	}
	item, err := workflow.NewAgentNode(booker, workflow.NodeConfig{})
	if err != nil {
		t.Fatalf("workflow.NewAgentNode() error = %v", err)
	}
	pw, err := workflow.NewParallelWorker("pw", item, 0, workflow.NodeConfig{})
	if err != nil {
		t.Fatalf("workflow.NewParallelWorker() error = %v", err)
	}
	mk := workflow.NewFunctionNode("mk", func(agent.Context, any) ([]any, error) {
		return []any{"Book a table"}, nil
	}, workflow.NodeConfig{})
	const sibOutput = "sibling done"
	sib := workflow.NewFunctionNode("sib", func(ctx agent.Context, _ any) (string, error) {
		select {
		case <-m.started:
			return sibOutput, nil
		case <-ctx.Done():
			return "", ctx.Err()
		}
	}, workflow.NodeConfig{})
	wf, err := workflowagent.New(workflowagent.Config{
		Name: "wf",
		Edges: []workflow.Edge{
			{From: workflow.Start, To: sib},
			{From: workflow.Start, To: mk},
			{From: mk, To: pw},
		},
		SubAgents: []agent.Agent{booker},
	})
	if err != nil {
		t.Fatalf("workflowagent.New() error = %v", err)
	}
	svc := session.InMemoryService()
	newNodeTestSession(t, t.Context(), svc)
	r, err := runner.New(runner.Config{AppName: nodeTestApp, Agent: wf, SessionService: svc})
	if err != nil {
		t.Fatalf("runner.New() error = %v", err)
	}

	stopped := false
	for ev, err := range r.Run(t.Context(), nodeTestUser, nodeTestSession, userText("Book tables"), agent.RunConfig{}) {
		if err != nil {
			t.Fatalf("Run() error = %v", err)
		}
		if ev.Output == sibOutput {
			stopped = true
			break
		}
	}

	if !stopped {
		t.Fatal("Run() ended without yielding the sibling's event")
	}
	if got := bookings.Load(); got != 0 {
		t.Errorf("book ran %d times after the caller stopped reading, want 0", got)
	}
}
