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

package remoteagent

import (
	"context"
	"iter"
	"net/http/httptest"
	"testing"

	legacyA2A "github.com/a2aproject/a2a-go/a2a"
	legacyASrv "github.com/a2aproject/a2a-go/a2asrv"
	v2a2a "github.com/a2aproject/a2a-go/v2/a2a"
	"github.com/a2aproject/a2a-go/v2/a2acompat/a2av0"
	v2asrv "github.com/a2aproject/a2a-go/v2/a2asrv"
	"google.golang.org/adk/agent"
	icontext "google.golang.org/adk/internal/context"
	"google.golang.org/adk/model"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/server/adka2a"
	"google.golang.org/adk/session"
	"google.golang.org/genai"
)

func TestCompat_OldExecutor_Direct(t *testing.T) {
	// 1. Create Old Executor with an AfterEventCallback (uses legacy a2a types)
	callbackCalled := false
	agentName := "test-agent"
	agentObj, err := agent.New(agent.Config{
		Name: agentName,
		Run: func(ic agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				ev := &session.Event{
					LLMResponse: model.LLMResponse{
						Content: genai.NewContentFromText("hello", genai.RoleModel),
					},
					Author: agentName,
				}
				yield(ev, nil)
			}
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	executor := adka2a.NewExecutor(adka2a.ExecutorConfig{
		OutputMode: adka2a.OutputArtifactPerEvent, // Use per-event to ensure immediate processing in this test
		RunnerConfig: runner.Config{
			AppName:        "TestApp",
			Agent:          agentObj,
			SessionService: session.InMemoryService(),
		},
		RunConfig: agent.RunConfig{
			StreamingMode: agent.StreamingModeSSE,
		},
		GenAIPartConverter: func(ctx context.Context, adkEvent *session.Event, part *genai.Part) (legacyA2A.Part, error) {
			return a2av0.FromV1Part(v2a2a.NewTextPart(part.Text)), nil
		},
		AfterEventCallback: func(ctx adka2a.ExecutorContext, event *session.Event, processed *legacyA2A.TaskArtifactUpdateEvent) error {
			callbackCalled = true
			if processed.Artifact != nil && len(processed.Artifact.Parts) > 0 {
				processed.Artifact.Parts[0] = a2av0.FromV1Part(v2a2a.NewTextPart("modified-by-executor"))
			}
			return nil
		},
	})

	// 2. Directly call Execute with a mock queue
	reqCtx := &legacyASrv.RequestContext{
		ContextID: "test-context",
		TaskID:    legacyA2A.NewTaskID(),
		Message:   legacyA2A.NewMessage(legacyA2A.MessageRoleUser, a2av0.FromV1Part(v2a2a.NewTextPart("hi"))),
	}
	queue := &mockQueue{}
	err = executor.Execute(t.Context(), reqCtx, queue)
	if err != nil {
		t.Fatalf("executor.Execute() error = %v", err)
	}

	if !callbackCalled {
		t.Errorf("Executor AfterEventCallback was not called. Queue has %d events", len(queue.events))
	}

	found := false
	for _, ev := range queue.events {
		if ae, ok := ev.(*legacyA2A.TaskArtifactUpdateEvent); ok {
			for _, p := range ae.Artifact.Parts {
				gp, _ := adka2a.ToGenAIPart(p)
				if gp != nil && gp.Text == "modified-by-executor" {
					found = true
				}
			}
		}
	}
	if !found {
		t.Error("Did not find modified part in executor output events")
	}
}

func TestCompat_OldRemoteAgent_Harness(t *testing.T) {
	// 1. Create Mock V2 Executor to serve requests from root remoteagent (which uses v1 under the hood)
	mockExec := &mockV2Executor{
		events: []v2a2a.Event{
			v2a2a.NewMessage(v2a2a.MessageRoleAgent, v2a2a.NewTextPart("hello")),
		},
	}

	// 2. Start v2 A2A server
	handler := v2asrv.NewHandler(mockExec)
	server := httptest.NewServer(v2asrv.NewJSONRPCHandler(handler))
	defer server.Close()

	// 3. Create Old Remote Agent (root package) pointing to this server
	legacyCard := a2av0.FromV1AgentCard(&v2a2a.AgentCard{
		SupportedInterfaces: []*v2a2a.AgentInterface{
			v2a2a.NewAgentInterface(server.URL, v2a2a.TransportProtocolJSONRPC),
		},
		Capabilities: v2a2a.AgentCapabilities{Streaming: true},
	})

	callbackCalled := false
	oldAgent, err := NewA2A(A2AConfig{
		Name:      "remote-agent",
		AgentCard: legacyCard,
		AfterRequestCallbacks: []AfterA2ARequestCallback{
			func(ctx agent.CallbackContext, req *legacyA2A.MessageSendParams, resp *session.Event, err error) (*session.Event, error) {
				callbackCalled = true
				if resp != nil && resp.Content != nil && len(resp.Content.Parts) > 0 {
					resp.Content.Parts[0].Text = "modified-by-agent-callback"
				}
				return nil, nil
			},
		},
	})
	if err != nil {
		t.Fatalf("remoteagent.NewA2A() error = %v", err)
	}

	// 4. Run the agent and check that root-level callbacks were executed
	ic := newInvocationContext(t, []*session.Event{newUserHello()})
	events, err := runAndCollect(ic, oldAgent)
	if err != nil {
		t.Fatalf("agent.Run() error = %v", err)
	}

	if !callbackCalled {
		t.Error("Remote Agent AfterRequestCallback was not called")
	}

	found := false
	for _, ev := range events {
		if ev.Content != nil && len(ev.Content.Parts) > 0 {
			for _, p := range ev.Content.Parts {
				if p.Text == "modified-by-agent-callback" {
					found = true
				}
			}
		}
	}
	if !found {
		t.Error("Did not find modified part in remote agent events")
	}
}

// Mocks

type mockQueue struct {
	events []legacyA2A.Event
}

func (q *mockQueue) Write(ctx context.Context, event legacyA2A.Event) error {
	q.events = append(q.events, event)
	return nil
}

func (q *mockQueue) WriteVersioned(ctx context.Context, event legacyA2A.Event, version legacyA2A.TaskVersion) error {
	return q.Write(ctx, event)
}

func (q *mockQueue) Read(ctx context.Context) (legacyA2A.Event, legacyA2A.TaskVersion, error) {
	var v legacyA2A.TaskVersion
	return nil, v, nil
}

func (q *mockQueue) Close() error { return nil }

type mockV2Executor struct {
	events []v2a2a.Event
}

func (e *mockV2Executor) Execute(ctx context.Context, execCtx *v2asrv.ExecutorContext) iter.Seq2[v2a2a.Event, error] {
	return func(yield func(v2a2a.Event, error) bool) {
		if !yield(v2a2a.NewSubmittedTask(execCtx, execCtx.Message), nil) {
			return
		}
		for _, ev := range e.events {
			if !yield(ev, nil) {
				return
			}
		}
	}
}

func (e *mockV2Executor) Cancel(ctx context.Context, execCtx *v2asrv.ExecutorContext) iter.Seq2[v2a2a.Event, error] {
	return func(yield func(v2a2a.Event, error) bool) {
		yield(v2a2a.NewStatusUpdateEvent(execCtx, v2a2a.TaskStateCanceled, nil), nil)
	}
}

// Helpers

func newInvocationContext(t *testing.T, events []*session.Event) agent.InvocationContext {
	t.Helper()
	ctx := t.Context()
	service := session.InMemoryService()
	resp, err := service.Create(ctx, &session.CreateRequest{AppName: t.Name(), UserID: "test"})
	if err != nil {
		t.Fatalf("sessionService.Create() error = %v", err)
	}
	for _, event := range events {
		if err := service.AppendEvent(ctx, resp.Session, event); err != nil {
			t.Fatalf("sessionService.AppendEvent() error = %v", err)
		}
	}

	ic := icontext.NewInvocationContext(ctx, icontext.InvocationContextParams{
		Session: resp.Session,
		RunConfig: &agent.RunConfig{
			StreamingMode: agent.StreamingModeSSE,
		},
	})
	return ic
}

func newUserHello() *session.Event {
	event := session.NewEvent("invocation")
	event.Author = "user"
	event.LLMResponse = model.LLMResponse{
		Content: genai.NewContentFromText("hello", genai.RoleUser),
	}
	return event
}

func runAndCollect(ic agent.InvocationContext, agnt agent.Agent) ([]*session.Event, error) {
	var collected []*session.Event
	for ev, err := range agnt.Run(ic) {
		if err != nil {
			return collected, err
		}
		collected = append(collected, ev)
	}
	return collected, nil
}
