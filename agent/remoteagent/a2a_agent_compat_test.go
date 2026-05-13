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
	"google.golang.org/adk/internal/utils"
	"google.golang.org/adk/model"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/server/adka2a"
	"google.golang.org/adk/session"
)

func TestCompat_OldExecutor_Direct(t *testing.T) {
	agentName := "test-agent"
	agentObj := utils.Must(agent.New(agent.Config{
		Name: agentName,
		Run: func(ic agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				yield(&session.Event{
					LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("hello", genai.RoleModel)},
					Author:      agentName,
				}, nil)
			}
		},
	}))

	executor := adka2a.NewExecutor(adka2a.ExecutorConfig{
		OutputMode: adka2a.OutputArtifactPerEvent,
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
			if processed.Artifact != nil && len(processed.Artifact.Parts) > 0 {
				processed.Artifact.Parts[0] = a2av0.FromV1Part(v2a2a.NewTextPart("modified-by-executor"))
			}
			return nil
		},
	})

	reqCtx := &legacyASrv.RequestContext{
		ContextID: "test-context",
		TaskID:    legacyA2A.NewTaskID(),
		Message:   legacyA2A.NewMessage(legacyA2A.MessageRoleUser, a2av0.FromV1Part(v2a2a.NewTextPart("hi"))),
	}
	queue := &mockQueue{}
	if err := executor.Execute(t.Context(), reqCtx, queue); err != nil {
		t.Fatalf("executor.Execute() error = %v", err)
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

func TestCompat_RemoteAgent(t *testing.T) {
	tests := []struct {
		name              string
		executor          *mockV2Executor
		updateConfig      func(config *A2AConfig)
		wantEventWithText string
	}{
		{
			name: "after request callback modifies response",
			executor: &mockV2Executor{
				events: []v2a2a.Event{
					v2a2a.NewMessage(v2a2a.MessageRoleAgent, v2a2a.NewTextPart("hello")),
				},
			},
			updateConfig: func(config *A2AConfig) {
				config.AfterRequestCallbacks = []AfterA2ARequestCallback{
					func(ctx agent.CallbackContext, req *legacyA2A.MessageSendParams, resp *session.Event, err error) (*session.Event, error) {
						if resp != nil && resp.Content != nil && len(resp.Content.Parts) > 0 {
							resp.Content.Parts[0].Text = "modified-by-agent-callback"
						}
						return nil, nil
					},
				}
			},
			wantEventWithText: "modified-by-agent-callback",
		},
		{
			name: "before request callback modifies request",
			executor: &mockV2Executor{
				executeFn: func(ctx context.Context, execCtx *v2asrv.ExecutorContext) iter.Seq2[v2a2a.Event, error] {
					return func(yield func(v2a2a.Event, error) bool) {
						text := ""
						if execCtx.Message != nil && len(execCtx.Message.Parts) > 0 {
							text = execCtx.Message.Parts[0].Text()
						}
						yield(v2a2a.NewMessage(v2a2a.MessageRoleAgent, v2a2a.NewTextPart("echo:"+text)), nil)
					}
				},
			},
			updateConfig: func(config *A2AConfig) {
				config.BeforeRequestCallbacks = []BeforeA2ARequestCallback{
					func(ctx agent.CallbackContext, req *legacyA2A.MessageSendParams) (*session.Event, error) {
						req.Message = legacyA2A.NewMessage(legacyA2A.MessageRoleUser, legacyA2A.TextPart{Text: "42"})
						return nil, nil
					},
				}
			},
			wantEventWithText: "echo:42",
		},
		{
			name: "before request callback short-circuits",
			executor: &mockV2Executor{
				executeFn: func(ctx context.Context, execCtx *v2asrv.ExecutorContext) iter.Seq2[v2a2a.Event, error] {
					return func(yield func(v2a2a.Event, error) bool) {
						t.Fatal("server should not be called when before callback short-circuits")
					}
				},
			},
			updateConfig: func(config *A2AConfig) {
				config.BeforeRequestCallbacks = []BeforeA2ARequestCallback{
					func(ctx agent.CallbackContext, req *legacyA2A.MessageSendParams) (*session.Event, error) {
						return &session.Event{
							LLMResponse: model.LLMResponse{
								Content: genai.NewContentFromText("cached-response", genai.RoleModel),
							},
						}, nil
					},
				}
			},
			wantEventWithText: "cached-response",
		},
		{
			name: "custom converter",
			executor: &mockV2Executor{
				events: []v2a2a.Event{
					v2a2a.NewMessage(v2a2a.MessageRoleAgent, v2a2a.NewTextPart("original")),
				},
			},
			updateConfig: func(config *A2AConfig) {
				config.Converter = func(ctx agent.InvocationContext, req *legacyA2A.MessageSendParams, event legacyA2A.Event, err error) (*session.Event, error) {
					ev := session.NewEvent("custom")
					ev.Author = ctx.Agent().Name()
					ev.LLMResponse = model.LLMResponse{
						Content:      genai.NewContentFromText("converted", genai.RoleModel),
						TurnComplete: true,
					}
					return ev, nil
				}
			},
			wantEventWithText: "converted",
		},
		{
			name: "GenAI part converter",
			executor: &mockV2Executor{
				executeFn: func(ctx context.Context, execCtx *v2asrv.ExecutorContext) iter.Seq2[v2a2a.Event, error] {
					return func(yield func(v2a2a.Event, error) bool) {
						if execCtx.Message != nil {
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
			},
			updateConfig: func(config *A2AConfig) {
				config.GenAIPartConverter = func(ctx context.Context, adkEvent *session.Event, part *genai.Part) (legacyA2A.Part, error) {
					if part.Text != "" {
						return a2av0.FromV1Part(v2a2a.NewTextPart("custom:" + part.Text)), nil
					}
					return adka2a.ToA2APart(part, nil)
				}
			},
			wantEventWithText: "converter-verified",
		},
		{
			name: "A2A part converter",
			executor: &mockV2Executor{
				events: []v2a2a.Event{
					v2a2a.NewMessage(v2a2a.MessageRoleAgent, v2a2a.NewTextPart("raw-response")),
				},
			},
			updateConfig: func(config *A2AConfig) {
				config.A2APartConverter = func(ctx context.Context, a2aEvent legacyA2A.Event, part legacyA2A.Part) (*genai.Part, error) {
					tp, ok := part.(legacyA2A.TextPart)
					if ok {
						return genai.NewPartFromText("custom:" + tp.Text), nil
					}
					return adka2a.ToGenAIPart(part)
				}
			},
			wantEventWithText: "custom:raw-response",
		},
		{
			name: "multiple after request callbacks execute in order",
			executor: &mockV2Executor{
				events: []v2a2a.Event{
					v2a2a.NewMessage(v2a2a.MessageRoleAgent, v2a2a.NewTextPart("hello")),
				},
			},
			updateConfig: func(config *A2AConfig) {
				config.AfterRequestCallbacks = []AfterA2ARequestCallback{
					func(ctx agent.CallbackContext, req *legacyA2A.MessageSendParams, resp *session.Event, err error) (*session.Event, error) {
						if resp != nil && resp.Content != nil && len(resp.Content.Parts) > 0 {
							resp.Content.Parts[0].Text += "-first"
						}
						return nil, nil
					},
					func(ctx agent.CallbackContext, req *legacyA2A.MessageSendParams, resp *session.Event, err error) (*session.Event, error) {
						if resp != nil && resp.Content != nil && len(resp.Content.Parts) > 0 {
							resp.Content.Parts[0].Text += "-second"
						}
						return nil, nil
					},
				}
			},
			wantEventWithText: "hello-first-second",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			server := startA2AServer(t, tc.executor)
			card := newLegacyCard(server.URL)
			config := A2AConfig{Name: "remote-agent", AgentCard: card}
			tc.updateConfig(&config)
			agnt, err := NewA2A(config)
			if err != nil {
				t.Fatalf("NewA2A() error = %v", err)
			}
			ic := newInvocationContext(t, []*session.Event{newUserHello()})
			events, err := runAndCollect(ic, agnt)
			if err != nil {
				t.Fatalf("agent.Run() error = %v", err)
			}
			foundText := false
			var texts []string
			for _, ev := range events {
				if foundText {
					break
				}
				if ev.Content == nil {
					continue
				}
				for _, p := range ev.Content.Parts {
					if p.Text == tc.wantEventWithText {
						foundText = true
						break
					}
					if p.Text != "" {
						texts = append(texts, p.Text)
					}
				}
			}
			if !foundText {
				t.Errorf("expected text %q in events, got texts: %v", tc.wantEventWithText, texts)
			}
		})
	}
}

func TestCompat_RemoteTaskCleanupCallback(t *testing.T) {
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

	server := startA2AServer(t, mockExec)
	card := newLegacyCard(server.URL)

	cleanupCalled := false
	var cleanupTaskInfo legacyA2A.TaskInfo
	oldAgent := utils.Must(NewA2A(A2AConfig{
		Name:      "remote-agent",
		AgentCard: card,
		RemoteTaskCleanupCallback: func(ctx context.Context, card *legacyA2A.AgentCard, client *a2aclient.Client, taskInfo legacyA2A.TaskInfo, cause error) {
			cleanupCalled = true
			cleanupTaskInfo = taskInfo
		},
	}))

	// Use a cancelable context so we can trigger cleanup by canceling mid-stream.
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

	// Break out of the run after receiving a couple events to trigger cleanup.
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

func startA2AServer(t *testing.T, executor *mockV2Executor) *httptest.Server {
	t.Helper()
	handler := v2asrv.NewHandler(executor)
	server := httptest.NewServer(v2asrv.NewJSONRPCHandler(handler))
	t.Cleanup(server.Close)
	return server
}

func newLegacyCard(serverURL string) *legacyA2A.AgentCard {
	return a2av0.FromV1AgentCard(&v2a2a.AgentCard{
		SupportedInterfaces: []*v2a2a.AgentInterface{
			v2a2a.NewAgentInterface(serverURL, v2a2a.TransportProtocolJSONRPC),
		},
		Capabilities: v2a2a.AgentCapabilities{Streaming: true},
	})
}

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
