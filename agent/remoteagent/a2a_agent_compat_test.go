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
	"time"

	legacyA2A "github.com/a2aproject/a2a-go/a2a"
	"github.com/a2aproject/a2a-go/a2aclient"
	legacyASrv "github.com/a2aproject/a2a-go/a2asrv"
	v2a2a "github.com/a2aproject/a2a-go/v2/a2a"
	"github.com/a2aproject/a2a-go/v2/a2acompat/a2av0"
	v2asrv "github.com/a2aproject/a2a-go/v2/a2asrv"
	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	icontext "google.golang.org/adk/internal/context"
	"google.golang.org/adk/model"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/server/adka2a"
	"google.golang.org/adk/session"
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

func TestCompat_BeforeRequestCallback_ModifiesRequest(t *testing.T) {
	// Verify legacy BeforeRequestCallback can inspect and modify the request
	var capturedMsg string
	mockExec := &mockV2Executor{
		executeFn: func(ctx context.Context, execCtx *v2asrv.ExecutorContext) iter.Seq2[v2a2a.Event, error] {
			return func(yield func(v2a2a.Event, error) bool) {
				// Echo back the first part text from the received message
				text := ""
				if execCtx.Message != nil && len(execCtx.Message.Parts) > 0 {
					text = execCtx.Message.Parts[0].Text()
				}
				capturedMsg = text
				yield(v2a2a.NewMessage(v2a2a.MessageRoleAgent, v2a2a.NewTextPart("echo:"+text)), nil)
			}
		},
	}

	handler := v2asrv.NewHandler(mockExec)
	server := httptest.NewServer(v2asrv.NewJSONRPCHandler(handler))
	defer server.Close()

	legacyCard := a2av0.FromV1AgentCard(&v2a2a.AgentCard{
		SupportedInterfaces: []*v2a2a.AgentInterface{
			v2a2a.NewAgentInterface(server.URL, v2a2a.TransportProtocolJSONRPC),
		},
		Capabilities: v2a2a.AgentCapabilities{Streaming: true},
	})

	beforeCalled := false
	oldAgent, err := NewA2A(A2AConfig{
		Name:      "remote-agent",
		AgentCard: legacyCard,
		BeforeRequestCallbacks: []BeforeA2ARequestCallback{
			func(ctx agent.CallbackContext, req *legacyA2A.MessageSendParams) (*session.Event, error) {
				beforeCalled = true
				// Modify the request metadata
				req.Metadata = map[string]any{"injected": true}
				return nil, nil
			},
		},
	})
	if err != nil {
		t.Fatalf("NewA2A() error = %v", err)
	}

	ic := newInvocationContext(t, []*session.Event{newUserHello()})
	_, err = runAndCollect(ic, oldAgent)
	if err != nil {
		t.Fatalf("agent.Run() error = %v", err)
	}

	if !beforeCalled {
		t.Error("BeforeRequestCallback was not called")
	}
	if capturedMsg == "" {
		t.Error("Server did not receive a message")
	}
}

func TestCompat_BeforeRequestCallback_ShortCircuit(t *testing.T) {
	// Verify legacy BeforeRequestCallback can short-circuit and return a cached response
	mockExec := &mockV2Executor{
		executeFn: func(ctx context.Context, execCtx *v2asrv.ExecutorContext) iter.Seq2[v2a2a.Event, error] {
			return func(yield func(v2a2a.Event, error) bool) {
				t.Fatal("server should not be called when before callback short-circuits")
			}
		},
	}

	handler := v2asrv.NewHandler(mockExec)
	server := httptest.NewServer(v2asrv.NewJSONRPCHandler(handler))
	defer server.Close()

	legacyCard := a2av0.FromV1AgentCard(&v2a2a.AgentCard{
		SupportedInterfaces: []*v2a2a.AgentInterface{
			v2a2a.NewAgentInterface(server.URL, v2a2a.TransportProtocolJSONRPC),
		},
		Capabilities: v2a2a.AgentCapabilities{Streaming: true},
	})

	cachedEvent := &session.Event{
		LLMResponse: model.LLMResponse{
			Content: genai.NewContentFromText("cached-response", genai.RoleModel),
		},
	}

	oldAgent, err := NewA2A(A2AConfig{
		Name:      "remote-agent",
		AgentCard: legacyCard,
		BeforeRequestCallbacks: []BeforeA2ARequestCallback{
			func(ctx agent.CallbackContext, req *legacyA2A.MessageSendParams) (*session.Event, error) {
				return cachedEvent, nil
			},
		},
	})
	if err != nil {
		t.Fatalf("NewA2A() error = %v", err)
	}

	ic := newInvocationContext(t, []*session.Event{newUserHello()})
	events, err := runAndCollect(ic, oldAgent)
	if err != nil {
		t.Fatalf("agent.Run() error = %v", err)
	}

	found := false
	for _, ev := range events {
		if ev.Content != nil && len(ev.Content.Parts) > 0 {
			for _, p := range ev.Content.Parts {
				if p.Text == "cached-response" {
					found = true
				}
			}
		}
	}
	if !found {
		t.Error("Short-circuit cached response was not returned")
	}
}

func TestCompat_Converter(t *testing.T) {
	// Verify legacy A2AEventConverter adapter works through the compat layer
	mockExec := &mockV2Executor{
		events: []v2a2a.Event{
			v2a2a.NewMessage(v2a2a.MessageRoleAgent, v2a2a.NewTextPart("original")),
		},
	}

	handler := v2asrv.NewHandler(mockExec)
	server := httptest.NewServer(v2asrv.NewJSONRPCHandler(handler))
	defer server.Close()

	legacyCard := a2av0.FromV1AgentCard(&v2a2a.AgentCard{
		SupportedInterfaces: []*v2a2a.AgentInterface{
			v2a2a.NewAgentInterface(server.URL, v2a2a.TransportProtocolJSONRPC),
		},
		Capabilities: v2a2a.AgentCapabilities{Streaming: true},
	})

	converterCalled := false
	oldAgent, err := NewA2A(A2AConfig{
		Name:      "remote-agent",
		AgentCard: legacyCard,
		Converter: func(ctx agent.InvocationContext, req *legacyA2A.MessageSendParams, event legacyA2A.Event, err error) (*session.Event, error) {
			converterCalled = true
			ev := session.NewEvent("custom")
			ev.Author = ctx.Agent().Name()
			ev.LLMResponse = model.LLMResponse{
				Content:      genai.NewContentFromText("converted", genai.RoleModel),
				TurnComplete: true,
			}
			return ev, nil
		},
	})
	if err != nil {
		t.Fatalf("NewA2A() error = %v", err)
	}

	ic := newInvocationContext(t, []*session.Event{newUserHello()})
	events, err := runAndCollect(ic, oldAgent)
	if err != nil {
		t.Fatalf("agent.Run() error = %v", err)
	}

	if !converterCalled {
		t.Error("Converter was not called")
	}

	found := false
	for _, ev := range events {
		if ev.Content != nil && len(ev.Content.Parts) > 0 {
			for _, p := range ev.Content.Parts {
				if p.Text == "converted" {
					found = true
				}
			}
		}
	}
	if !found {
		t.Error("Custom converter output was not found in events")
	}
}

func TestCompat_GenAIPartConverter(t *testing.T) {
	// Verify legacy GenAIPartConverter is used when building messages
	var sentParts int
	mockExec := &mockV2Executor{
		executeFn: func(ctx context.Context, execCtx *v2asrv.ExecutorContext) iter.Seq2[v2a2a.Event, error] {
			return func(yield func(v2a2a.Event, error) bool) {
				if execCtx.Message != nil {
					sentParts = len(execCtx.Message.Parts)
					// Check that the custom converter was applied
					for _, p := range execCtx.Message.Parts {
						if p.Text() == "custom:hello" {
							yield(v2a2a.NewMessage(v2a2a.MessageRoleAgent, v2a2a.NewTextPart("converter-verified")), nil)
							return
						}
					}
				}
				yield(v2a2a.NewMessage(v2a2a.MessageRoleAgent, v2a2a.NewTextPart("converter-not-applied")), nil)
			}
		},
	}

	handler := v2asrv.NewHandler(mockExec)
	server := httptest.NewServer(v2asrv.NewJSONRPCHandler(handler))
	defer server.Close()

	legacyCard := a2av0.FromV1AgentCard(&v2a2a.AgentCard{
		SupportedInterfaces: []*v2a2a.AgentInterface{
			v2a2a.NewAgentInterface(server.URL, v2a2a.TransportProtocolJSONRPC),
		},
		Capabilities: v2a2a.AgentCapabilities{Streaming: true},
	})

	oldAgent, err := NewA2A(A2AConfig{
		Name:      "remote-agent",
		AgentCard: legacyCard,
		GenAIPartConverter: func(ctx context.Context, adkEvent *session.Event, part *genai.Part) (legacyA2A.Part, error) {
			if part.Text != "" {
				return a2av0.FromV1Part(v2a2a.NewTextPart("custom:" + part.Text)), nil
			}
			return adka2a.ToA2APart(part, nil)
		},
	})
	if err != nil {
		t.Fatalf("NewA2A() error = %v", err)
	}

	ic := newInvocationContext(t, []*session.Event{newUserHello()})
	events, err := runAndCollect(ic, oldAgent)
	if err != nil {
		t.Fatalf("agent.Run() error = %v", err)
	}

	if sentParts == 0 {
		t.Fatal("Server received no parts")
	}

	found := false
	for _, ev := range events {
		if ev.Content != nil {
			for _, p := range ev.Content.Parts {
				if p.Text == "converter-verified" {
					found = true
				}
			}
		}
	}
	if !found {
		t.Error("GenAIPartConverter was not applied to outgoing messages")
	}
}

func TestCompat_A2APartConverter(t *testing.T) {
	// Verify legacy A2APartConverter is used when converting incoming events
	mockExec := &mockV2Executor{
		events: []v2a2a.Event{
			v2a2a.NewMessage(v2a2a.MessageRoleAgent, v2a2a.NewTextPart("raw-response")),
		},
	}

	handler := v2asrv.NewHandler(mockExec)
	server := httptest.NewServer(v2asrv.NewJSONRPCHandler(handler))
	defer server.Close()

	legacyCard := a2av0.FromV1AgentCard(&v2a2a.AgentCard{
		SupportedInterfaces: []*v2a2a.AgentInterface{
			v2a2a.NewAgentInterface(server.URL, v2a2a.TransportProtocolJSONRPC),
		},
		Capabilities: v2a2a.AgentCapabilities{Streaming: true},
	})

	partConverterCalled := false
	oldAgent, err := NewA2A(A2AConfig{
		Name:      "remote-agent",
		AgentCard: legacyCard,
		A2APartConverter: func(ctx context.Context, a2aEvent legacyA2A.Event, part legacyA2A.Part) (*genai.Part, error) {
			partConverterCalled = true
			tp, ok := part.(legacyA2A.TextPart)
			if ok {
				return genai.NewPartFromText("custom:" + tp.Text), nil
			}
			return adka2a.ToGenAIPart(part)
		},
	})
	if err != nil {
		t.Fatalf("NewA2A() error = %v", err)
	}

	ic := newInvocationContext(t, []*session.Event{newUserHello()})
	events, err := runAndCollect(ic, oldAgent)
	if err != nil {
		t.Fatalf("agent.Run() error = %v", err)
	}

	if !partConverterCalled {
		t.Error("A2APartConverter was not called")
	}

	found := false
	for _, ev := range events {
		if ev.Content != nil {
			for _, p := range ev.Content.Parts {
				if p.Text == "custom:raw-response" {
					found = true
				}
			}
		}
	}
	if !found {
		t.Error("A2APartConverter was not applied to incoming events")
	}
}

func TestCompat_RemoteTaskCleanupCallback(t *testing.T) {
	// Verify legacy RemoteTaskCleanupCallback receives correct legacy types.
	// The executor keeps yielding events until context is canceled, ensuring
	// the last event is non-terminal and cleanup is triggered.
	mockExec := &mockV2Executor{
		executeFn: func(ctx context.Context, execCtx *v2asrv.ExecutorContext) iter.Seq2[v2a2a.Event, error] {
			return func(yield func(v2a2a.Event, error) bool) {
				if !yield(v2a2a.NewSubmittedTask(execCtx, execCtx.Message), nil) {
					return
				}
				for ctx.Err() == nil {
					data := v2a2a.NewDataPart(map[string]any{"tick": true})
					if !yield(v2a2a.NewArtifactEvent(execCtx, data), nil) {
						return
					}
					time.Sleep(1 * time.Millisecond)
				}
				yield(v2a2a.NewStatusUpdateEvent(execCtx, v2a2a.TaskStateCompleted, nil), nil)
			}
		},
	}

	handler := v2asrv.NewHandler(mockExec)
	server := httptest.NewServer(v2asrv.NewJSONRPCHandler(handler))
	defer server.Close()

	legacyCard := a2av0.FromV1AgentCard(&v2a2a.AgentCard{
		SupportedInterfaces: []*v2a2a.AgentInterface{
			v2a2a.NewAgentInterface(server.URL, v2a2a.TransportProtocolJSONRPC),
		},
		Capabilities: v2a2a.AgentCapabilities{Streaming: true},
	})

	cleanupCalled := false
	var cleanupTaskInfo legacyA2A.TaskInfo
	oldAgent, err := NewA2A(A2AConfig{
		Name:      "remote-agent",
		AgentCard: legacyCard,
		RemoteTaskCleanupCallback: func(ctx context.Context, card *legacyA2A.AgentCard, client *a2aclient.Client, taskInfo legacyA2A.TaskInfo, cause error) {
			cleanupCalled = true
			cleanupTaskInfo = taskInfo
		},
	})
	if err != nil {
		t.Fatalf("NewA2A() error = %v", err)
	}

	// Use a cancelable context so we can trigger cleanup by canceling mid-stream
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	svc := session.InMemoryService()
	resp, err := svc.Create(ctx, &session.CreateRequest{AppName: t.Name(), UserID: "test"})
	if err != nil {
		t.Fatalf("session.Create() error = %v", err)
	}
	hello := newUserHello()
	if err := svc.AppendEvent(ctx, resp.Session, hello); err != nil {
		t.Fatalf("AppendEvent() error = %v", err)
	}
	ic := icontext.NewInvocationContext(ctx, icontext.InvocationContextParams{
		Session:   resp.Session,
		RunConfig: &agent.RunConfig{StreamingMode: agent.StreamingModeSSE},
	})

	// Break out of the run after receiving a couple events to trigger cleanup
	count := 0
	for _, err := range oldAgent.Run(ic) {
		if err != nil {
			break
		}
		count++
		if count >= 2 {
			cancel()
		}
	}

	if !cleanupCalled {
		t.Error("RemoteTaskCleanupCallback was not called")
	}
	if cleanupTaskInfo.TaskID == "" {
		t.Error("RemoteTaskCleanupCallback received empty TaskID")
	}
}

func TestCompat_MultipleAfterRequestCallbacks(t *testing.T) {
	// Verify that multiple legacy AfterRequestCallbacks are properly adapted,
	// with short-circuit behavior preserved
	mockExec := &mockV2Executor{
		events: []v2a2a.Event{
			v2a2a.NewMessage(v2a2a.MessageRoleAgent, v2a2a.NewTextPart("hello")),
		},
	}

	handler := v2asrv.NewHandler(mockExec)
	server := httptest.NewServer(v2asrv.NewJSONRPCHandler(handler))
	defer server.Close()

	legacyCard := a2av0.FromV1AgentCard(&v2a2a.AgentCard{
		SupportedInterfaces: []*v2a2a.AgentInterface{
			v2a2a.NewAgentInterface(server.URL, v2a2a.TransportProtocolJSONRPC),
		},
		Capabilities: v2a2a.AgentCapabilities{Streaming: true},
	})

	callOrder := []string{}
	oldAgent, err := NewA2A(A2AConfig{
		Name:      "remote-agent",
		AgentCard: legacyCard,
		AfterRequestCallbacks: []AfterA2ARequestCallback{
			func(ctx agent.CallbackContext, req *legacyA2A.MessageSendParams, resp *session.Event, err error) (*session.Event, error) {
				callOrder = append(callOrder, "first")
				// Modify and pass through
				if resp != nil && resp.Content != nil && len(resp.Content.Parts) > 0 {
					resp.Content.Parts[0].Text += "-first"
				}
				return nil, nil
			},
			func(ctx agent.CallbackContext, req *legacyA2A.MessageSendParams, resp *session.Event, err error) (*session.Event, error) {
				callOrder = append(callOrder, "second")
				// Modify and pass through
				if resp != nil && resp.Content != nil && len(resp.Content.Parts) > 0 {
					resp.Content.Parts[0].Text += "-second"
				}
				return nil, nil
			},
		},
	})
	if err != nil {
		t.Fatalf("NewA2A() error = %v", err)
	}

	ic := newInvocationContext(t, []*session.Event{newUserHello()})
	events, err := runAndCollect(ic, oldAgent)
	if err != nil {
		t.Fatalf("agent.Run() error = %v", err)
	}

	if len(callOrder) < 2 {
		t.Fatalf("Expected both callbacks to be called, got order: %v", callOrder)
	}
	if callOrder[0] != "first" || callOrder[1] != "second" {
		t.Fatalf("Callbacks called in wrong order: %v", callOrder)
	}

	found := false
	for _, ev := range events {
		if ev.Content != nil {
			for _, p := range ev.Content.Parts {
				if p.Text == "hello-first-second" {
					found = true
				}
			}
		}
	}
	if !found {
		var texts []string
		for _, ev := range events {
			if ev.Content != nil {
				for _, p := range ev.Content.Parts {
					texts = append(texts, p.Text)
				}
			}
		}
		t.Errorf("Expected 'hello-first-second' in events, got texts: %v", texts)
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
	events    []v2a2a.Event
	executeFn func(ctx context.Context, execCtx *v2asrv.ExecutorContext) iter.Seq2[v2a2a.Event, error]
}

func (e *mockV2Executor) Execute(ctx context.Context, execCtx *v2asrv.ExecutorContext) iter.Seq2[v2a2a.Event, error] {
	if e.executeFn != nil {
		return e.executeFn(ctx, execCtx)
	}
	return func(yield func(v2a2a.Event, error) bool) {
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
