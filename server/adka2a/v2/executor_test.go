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

package adka2a

import (
	"context"
	"fmt"
	"iter"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/a2aproject/a2a-go/v2/a2a"
	"github.com/a2aproject/a2a-go/v2/a2aclient"
	"github.com/a2aproject/a2a-go/v2/a2asrv"
	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/internal/utils"
	"google.golang.org/adk/model"
	"google.golang.org/adk/plugin"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/session"
)

// No testQueue needed anymore.

// testQueue removed.

type testSessionService struct {
	session.Service
	createErr bool
}

func (s *testSessionService) Create(ctx context.Context, req *session.CreateRequest) (*session.CreateResponse, error) {
	if s.createErr {
		return nil, fmt.Errorf("session creation failed")
	}
	return s.Service.Create(ctx, req)
}

func newEventReplayAgent(events []*session.Event, failWith error) (agent.Agent, error) {
	return agent.New(agent.Config{
		Name: "test",
		Run: func(agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				for _, event := range events {
					if !yield(event, nil) {
						return
					}
				}
				if failWith != nil {
					yield(nil, failWith)
				}
			}
		},
	})
}

type eventIndex struct{ i int }

func TestExecutor_Execute(t *testing.T) {
	task := &a2a.Task{ID: a2a.NewTaskID(), ContextID: a2a.NewContextID()}
	hiMsg := a2a.NewMessage(a2a.MessageRoleUser, a2a.NewTextPart("hi"))
	hiMsgForTask := a2a.NewMessageForTask(a2a.MessageRoleUser, task, a2a.NewTextPart("hi"))

	testCases := []struct {
		name               string
		request            *a2a.SendMessageRequest
		events             []*session.Event
		wantEvents         []a2a.Event
		createSessionFails bool
		agentRunFails      error
		queueWriteFails    *eventIndex
		wantErr            bool
	}{
		{
			name:    "no message",
			request: &a2a.SendMessageRequest{},
			wantErr: true,
		},
		{
			name: "malformed data",
			request: &a2a.SendMessageRequest{
				Message: a2a.NewMessageForTask(a2a.MessageRoleUser, task, func() *a2a.Part {
					p := a2a.NewDataPart(make(chan int))
					p.SetMeta(a2aDataPartMetaTypeKey, a2aDataPartTypeFunctionCall)
					return p
				}()),
			},
			wantErr: true,
		},
		{
			name:               "session setup fails",
			request:            &a2a.SendMessageRequest{Message: hiMsgForTask},
			createSessionFails: true,
			wantEvents: []a2a.Event{
				newFinalStatusUpdate(
					task, a2a.TaskStateFailed,
					a2a.NewMessageForTask(a2a.MessageRoleAgent, task, a2a.NewTextPart("failed to create a session: session creation failed")),
				),
			},
		},
		{
			name:    "success for a new task",
			request: &a2a.SendMessageRequest{Message: hiMsg},
			events: []*session.Event{
				{LLMResponse: modelResponseFromParts(genai.NewPartFromText("Hello"))},
				{LLMResponse: modelResponseFromParts(genai.NewPartFromText(", world!"))},
			},
			wantEvents: []a2a.Event{
				a2a.NewSubmittedTask(task, hiMsg),
				a2a.NewStatusUpdateEvent(task, a2a.TaskStateWorking, nil),
				a2a.NewArtifactEvent(task, a2a.NewTextPart("Hello")),
				a2a.NewArtifactUpdateEvent(task, a2a.NewArtifactID(), a2a.NewTextPart(", world!")),
				newFinalStatusUpdate(task, a2a.TaskStateCompleted, nil),
			},
		},
		{
			name:    "success for existing task",
			request: &a2a.SendMessageRequest{Message: hiMsgForTask},
			events: []*session.Event{
				{LLMResponse: modelResponseFromParts(genai.NewPartFromText("Hello"))},
				{LLMResponse: modelResponseFromParts(genai.NewPartFromText(", world!"))},
			},
			wantEvents: []a2a.Event{
				a2a.NewStatusUpdateEvent(task, a2a.TaskStateWorking, nil),
				a2a.NewArtifactEvent(task, a2a.NewTextPart("Hello")),
				a2a.NewArtifactUpdateEvent(task, a2a.NewArtifactID(), a2a.NewTextPart(", world!")),
				newFinalStatusUpdate(task, a2a.TaskStateCompleted, nil),
			},
		},
		{
			name:            "queue write fails",
			request:         &a2a.SendMessageRequest{Message: hiMsgForTask},
			queueWriteFails: &eventIndex{0},
			wantErr:         true,
		},
		{
			name:    "llm fails",
			request: &a2a.SendMessageRequest{Message: hiMsgForTask},
			events: []*session.Event{
				{LLMResponse: modelResponseFromParts(genai.NewPartFromText("Hello"))},
				{LLMResponse: model.LLMResponse{ErrorCode: "418", ErrorMessage: "I'm a teapot"}},
				{LLMResponse: modelResponseFromParts(genai.NewPartFromText(", world!"))},
			},
			wantEvents: []a2a.Event{
				a2a.NewStatusUpdateEvent(task, a2a.TaskStateWorking, nil),
				a2a.NewArtifactEvent(task, a2a.NewTextPart("Hello")),
				a2a.NewArtifactUpdateEvent(task, a2a.NewArtifactID(), a2a.NewTextPart(", world!")),
				toTaskFailedUpdateEvent(
					task, errorFromResponse(&model.LLMResponse{ErrorCode: "418", ErrorMessage: "I'm a teapot"}),
					map[string]any{ToA2AMetaKey("error_code"): "418"},
				),
			},
		},
		{
			name:    "agent run fails",
			request: &a2a.SendMessageRequest{Message: hiMsgForTask},
			events: []*session.Event{
				{LLMResponse: modelResponseFromParts(genai.NewPartFromText("Hello"))},
			},
			agentRunFails: fmt.Errorf("OOF"),
			wantEvents: []a2a.Event{
				a2a.NewStatusUpdateEvent(task, a2a.TaskStateWorking, nil),
				a2a.NewArtifactEvent(task, a2a.NewTextPart("Hello")),
				newFinalStatusUpdate(
					task, a2a.TaskStateFailed,
					a2a.NewMessageForTask(a2a.MessageRoleAgent, task, a2a.NewTextPart("agent run failed: OOF")),
				),
			},
		},
		{
			name:    "agent run and queue write fail",
			request: &a2a.SendMessageRequest{Message: hiMsgForTask},
			events: []*session.Event{
				{LLMResponse: modelResponseFromParts(genai.NewPartFromText("Hello"))},
			},
			queueWriteFails: &eventIndex{2},
			agentRunFails:   fmt.Errorf("OOF"),
			wantErr:         true,
			wantEvents: []a2a.Event{
				a2a.NewStatusUpdateEvent(task, a2a.TaskStateWorking, nil),
				a2a.NewArtifactEvent(task, a2a.NewTextPart("Hello")),
			},
		},
	}

	for _, tc := range testCases {
		ignoreOptions := []cmp.Option{
			cmpopts.IgnoreFields(a2a.Message{}, "ID"),
			cmpopts.IgnoreFields(a2a.Artifact{}, "ID"),
			cmpopts.IgnoreFields(a2a.TaskStatus{}, "Timestamp"),
			cmpopts.IgnoreFields(a2a.TaskStatusUpdateEvent{}, "Metadata"),
			cmpopts.IgnoreFields(a2a.TaskArtifactUpdateEvent{}, "Metadata"),
		}

		t.Run(tc.name, func(t *testing.T) {
			agent, err := newEventReplayAgent(tc.events, tc.agentRunFails)
			if err != nil {
				t.Fatalf("newEventReplayAgent() error = %v, want nil", err)
			}
			sessionService := &testSessionService{Service: session.InMemoryService(), createErr: tc.createSessionFails}
			runnerConfig := runner.Config{AppName: agent.Name(), Agent: agent, SessionService: sessionService}
			executor := NewExecutor(ExecutorConfig{RunnerConfig: runnerConfig})
			var gotEvents []a2a.Event
			reqCtx := &a2asrv.ExecutorContext{TaskID: task.ID, ContextID: task.ContextID, Message: tc.request.Message}
			if tc.request.Message != nil && tc.request.Message.TaskID == task.ID {
				reqCtx.StoredTask = task
			}

			var executeErr error
			for event, err := range executor.Execute(t.Context(), reqCtx) {
				if err != nil {
					executeErr = err
					break
				}
				if tc.queueWriteFails != nil && tc.queueWriteFails.i == len(gotEvents) {
					executeErr = fmt.Errorf("queue write failed")
					break
				}
				gotEvents = append(gotEvents, event)
			}
			if executeErr != nil {
				if !tc.wantErr {
					t.Fatalf("executor.Execute() error = %v, want nil", executeErr)
				}
				return
			}
			if tc.wantErr {
				t.Fatalf("executor.Execute() error = nil, want error")
			}
			if diff := cmp.Diff(tc.wantEvents, gotEvents, ignoreOptions...); diff != "" {
				t.Errorf("executor.Execute() produced wrong events (-want +got):\n%s", diff)
			}
		})
	}
}

func TestExecutor_Cancel(t *testing.T) {
	task := &a2a.Task{ID: a2a.NewTaskID(), ContextID: a2a.NewContextID()}
	executor := NewExecutor(ExecutorConfig{})
	reqCtx := &a2asrv.ExecutorContext{TaskID: task.ID, ContextID: task.ContextID}

	var gotEvents []a2a.Event

	reqCtx.StoredTask = task
	for event, err := range executor.Cancel(t.Context(), reqCtx) {
		if err != nil {
			t.Fatalf("executor.Cancel() error = %v, want nil", err)
		}
		gotEvents = append(gotEvents, event)
	}
	if len(gotEvents) != 1 {
		t.Fatalf("executor.Cancel() produced %d events, want 1", len(gotEvents))
	}
	event := gotEvents[0].(*a2a.TaskStatusUpdateEvent)
	if event.Status.State != a2a.TaskStateCanceled {
		t.Fatalf("executor.Cancel() = %v, want a single TaskStateCanceled update", event)
	}
}

func TestExecutor_SessionReuse(t *testing.T) {
	ctx := t.Context()
	agent, err := newEventReplayAgent([]*session.Event{}, nil)
	if err != nil {
		t.Fatalf("newEventReplayAgent() error = %v, want nil", err)
	}

	sessionService := session.InMemoryService()
	task := &a2a.Task{ID: a2a.NewTaskID(), ContextID: a2a.NewContextID()}
	req := &a2a.SendMessageRequest{Message: a2a.NewMessageForTask(a2a.MessageRoleUser, task)}
	reqCtx := &a2asrv.ExecutorContext{TaskID: task.ID, ContextID: task.ContextID, Message: req.Message}
	runnerConfig := runner.Config{AppName: agent.Name(), Agent: agent, SessionService: sessionService}
	config := ExecutorConfig{RunnerConfig: runnerConfig}
	executor := NewExecutor(config)
	var gotEvents []a2a.Event

	for event, err := range executor.Execute(ctx, reqCtx) {
		if err != nil {
			t.Fatalf("executor.Execute() error = %v, want nil", err)
		}
		gotEvents = append(gotEvents, event)
	}
	for event, err := range executor.Execute(ctx, reqCtx) {
		if err != nil {
			t.Fatalf("executor.Execute() error = %v, want nil", err)
		}
		gotEvents = append(gotEvents, event)
	}

	meta := toInvocationMeta(ctx, toInternalRunnerConfig(config.RunnerConfig), reqCtx)
	sessions, err := sessionService.List(ctx, &session.ListRequest{AppName: runnerConfig.AppName, UserID: meta.userID})
	if err != nil {
		t.Fatalf("sessionService.List() error = %v, want nil", err)
	}
	if len(sessions.Sessions) != 1 {
		t.Fatalf("sessionService.List() got %d sessions, want 1", sessions.Sessions)
	}

	reqCtx.ContextID = a2a.NewContextID()
	otherContextMeta := toInvocationMeta(ctx, toInternalRunnerConfig(config.RunnerConfig), reqCtx)
	if meta.sessionID == otherContextMeta.sessionID {
		t.Fatal("want sessionID to be different for different contextIDs")
	}
}

func TestExecutor_Callbacks(t *testing.T) {
	type contextKeyType struct{}
	task := &a2a.Task{ID: a2a.NewTaskID(), ContextID: a2a.NewContextID()}
	hiMsg := a2a.NewMessageForTask(a2a.MessageRoleUser, task, a2a.NewTextPart("hi"))

	testCases := []struct {
		name               string
		createSessionFails bool
		events             []*session.Event
		beforeExecution    BeforeExecuteCallback
		afterEvent         AfterEventCallback
		afterExecution     AfterExecuteCallback
		wantEvents         []a2a.Event
		wantErr            error
	}{
		{
			name: "abort execution",
			beforeExecution: func(ctx context.Context, reqCtx *a2asrv.ExecutorContext) (context.Context, error) {
				return nil, fmt.Errorf("aborted")
			},
			wantErr: fmt.Errorf("aborted"),
		},
		{
			name: "instrument context",
			beforeExecution: func(ctx context.Context, reqCtx *a2asrv.ExecutorContext) (context.Context, error) {
				return context.WithValue(ctx, contextKeyType{}, "bar"), nil
			},
			afterExecution: func(ctx ExecutorContext, finalUpdate *a2a.TaskStatusUpdateEvent, err error) error {
				text, _ := ctx.Value(contextKeyType{}).(string)
				finalUpdate.Status.Message = a2a.NewMessage(a2a.MessageRoleAgent, a2a.NewTextPart(text))
				return nil
			},
			wantEvents: []a2a.Event{
				a2a.NewStatusUpdateEvent(task, a2a.TaskStateWorking, nil),
				newFinalStatusUpdate(task, a2a.TaskStateCompleted, a2a.NewMessage(a2a.MessageRoleAgent, a2a.NewTextPart("bar"))),
			},
		},
		{
			name: "intercept processing failure",
			events: []*session.Event{
				{LLMResponse: modelResponseFromParts(genai.NewPartFromText("Hello, world!"))},
			},
			afterEvent: func(ctx ExecutorContext, event *session.Event, processed *a2a.TaskArtifactUpdateEvent) error {
				return fmt.Errorf("fail!")
			},
			afterExecution: func(ctx ExecutorContext, finalUpdate *a2a.TaskStatusUpdateEvent, err error) error {
				finalUpdate.Status.Message = a2a.NewMessage(a2a.MessageRoleAgent, a2a.NewTextPart("bar"))
				return nil
			},
			wantEvents: []a2a.Event{
				a2a.NewStatusUpdateEvent(task, a2a.TaskStateWorking, nil),
				newFinalStatusUpdate(task, a2a.TaskStateFailed, a2a.NewMessage(a2a.MessageRoleAgent, a2a.NewTextPart("bar"))),
			},
		},
		{
			name:               "intercept session setup failure",
			createSessionFails: true,
			afterExecution: func(ctx ExecutorContext, finalUpdate *a2a.TaskStatusUpdateEvent, err error) error {
				eventCount := 0
				for range ctx.ReadonlyState().All() {
					eventCount++
				}
				finalUpdate.Status.Message = a2a.NewMessage(a2a.MessageRoleAgent, a2a.NewTextPart(fmt.Sprintf("%d events", eventCount)))
				return nil
			},
			wantEvents: []a2a.Event{newFinalStatusUpdate(task, a2a.TaskStateFailed, a2a.NewMessage(a2a.MessageRoleAgent, a2a.NewTextPart("0 events")))},
		},
		{
			name: "enrich event",
			events: []*session.Event{
				{LLMResponse: modelResponseFromParts(genai.NewPartFromText("Hello"))},
				{LLMResponse: modelResponseFromParts(genai.NewPartFromText(", world!"))},
			},
			afterEvent: func(ctx ExecutorContext, event *session.Event, processed *a2a.TaskArtifactUpdateEvent) error {
				processed.Artifact.Parts = append(processed.Artifact.Parts, a2a.NewTextPart(" (enriched)"))
				return nil
			},
			wantEvents: []a2a.Event{
				a2a.NewStatusUpdateEvent(task, a2a.TaskStateWorking, nil),
				a2a.NewArtifactEvent(task, a2a.NewTextPart("Hello"), a2a.NewTextPart(" (enriched)")),
				a2a.NewArtifactUpdateEvent(task, a2a.NewArtifactID(), a2a.NewTextPart(", world!"), a2a.NewTextPart(" (enriched)")),
				newFinalStatusUpdate(task, a2a.TaskStateCompleted, nil),
			},
		},
		{
			name: "can access session events",
			events: []*session.Event{
				{LLMResponse: modelResponseFromParts(genai.NewPartFromText("Hello"))},
				{LLMResponse: modelResponseFromParts(genai.NewPartFromText(", world!"))},
			},
			afterExecution: func(ctx ExecutorContext, finalUpdate *a2a.TaskStatusUpdateEvent, err error) error {
				eventCount := ctx.Events().Len()
				finalUpdate.Status.Message = a2a.NewMessage(a2a.MessageRoleAgent, a2a.NewTextPart(fmt.Sprintf("event count = %d", eventCount)))
				return nil
			},
			wantEvents: []a2a.Event{
				a2a.NewStatusUpdateEvent(task, a2a.TaskStateWorking, nil),
				a2a.NewArtifactEvent(task, a2a.NewTextPart("Hello")),
				a2a.NewArtifactUpdateEvent(task, a2a.NewArtifactID(), a2a.NewTextPart(", world!")),
				newFinalStatusUpdate(task, a2a.TaskStateCompleted,
					a2a.NewMessage(a2a.MessageRoleAgent, a2a.NewTextPart("event count = 3")),
				),
			},
		},
		{
			name: "abort execution",
			events: []*session.Event{
				{LLMResponse: modelResponseFromParts(genai.NewPartFromText("Hello"))},
				{LLMResponse: modelResponseFromParts(genai.NewPartFromText(", world!"))},
			},
			afterEvent: func(ctx ExecutorContext, event *session.Event, processed *a2a.TaskArtifactUpdateEvent) error {
				return fmt.Errorf("abort execution")
			},
			wantEvents: []a2a.Event{
				a2a.NewStatusUpdateEvent(task, a2a.TaskStateWorking, nil),
				toTaskFailedUpdateEvent(task, fmt.Errorf("processor failed: abort execution"), nil),
			},
		},
	}

	for _, tc := range testCases {
		ignoreOptions := []cmp.Option{
			cmpopts.IgnoreFields(a2a.Message{}, "ID"),
			cmpopts.IgnoreFields(a2a.Artifact{}, "ID"),
			cmpopts.IgnoreFields(a2a.TaskStatus{}, "Timestamp"),
			cmpopts.IgnoreFields(a2a.TaskStatusUpdateEvent{}, "Metadata"),
			cmpopts.IgnoreFields(a2a.TaskArtifactUpdateEvent{}, "Metadata"),
		}

		t.Run(tc.name, func(t *testing.T) {
			agent, err := newEventReplayAgent(tc.events, nil)
			if err != nil {
				t.Fatalf("newEventReplayAgent() error = %v, want nil", err)
			}
			sessionService := &testSessionService{Service: session.InMemoryService(), createErr: tc.createSessionFails}
			runnerConfig := runner.Config{AppName: agent.Name(), Agent: agent, SessionService: sessionService}
			executor := NewExecutor(ExecutorConfig{
				RunnerConfig:          runnerConfig,
				BeforeExecuteCallback: tc.beforeExecution,
				AfterEventCallback:    tc.afterEvent,
				AfterExecuteCallback:  tc.afterExecution,
			})
			var gotEvents []a2a.Event
			reqCtx := &a2asrv.ExecutorContext{TaskID: task.ID, ContextID: task.ContextID, Message: hiMsg, StoredTask: task}

			var executeErr error
			for event, err := range executor.Execute(t.Context(), reqCtx) {
				if err != nil {
					executeErr = err
					break
				}
				gotEvents = append(gotEvents, event)
			}
			if executeErr != nil {
				if tc.wantErr == nil {
					t.Fatalf("executor.Execute() error = %v, want nil", executeErr)
				}
				return
			}
			if tc.wantErr != nil {
				t.Fatalf("executor.Execute() error = nil, want %v", tc.wantErr)
			}
			if tc.wantEvents != nil {
				if diff := cmp.Diff(tc.wantEvents, gotEvents, ignoreOptions...); diff != "" {
					t.Errorf("executor.Execute() produced wrong events (-want +got):\n%s", diff)
				}
			}
		})
	}
}

func startA2AServer(agentExecutor a2asrv.AgentExecutor) *httptest.Server {
	requestHandler := a2asrv.NewHandler(agentExecutor)
	return httptest.NewServer(a2asrv.NewJSONRPCHandler(requestHandler))
}

func TestExecutor_Cancel_AfterEvent(t *testing.T) {
	sessionService := session.InMemoryService()
	channel := make(chan struct{})

	agent, err := agent.New(agent.Config{
		Name: "test",
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				defer close(channel)
				<-ctx.Done()
				yield(nil, ctx.Err())
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New() error = %v, want nil", err)
	}

	executor := NewExecutor(ExecutorConfig{
		RunnerConfig: runner.Config{
			AppName:        agent.Name(),
			Agent:          agent,
			SessionService: sessionService,
		},
	})

	server := startA2AServer(executor)
	defer server.Close()

	card := &a2a.AgentCard{
		SupportedInterfaces: []*a2a.AgentInterface{
			a2a.NewAgentInterface(server.URL, a2a.TransportProtocolJSONRPC),
		},
	}

	client, err := a2aclient.NewFromCard(t.Context(), card)
	if err != nil {
		t.Fatalf("a2aclient.NewFromCard() error = %v, want nil", err)
	}

	msgId := a2a.NewMessageID()
	result, sendErr := client.SendMessage(t.Context(), &a2a.SendMessageRequest{
		Message: &a2a.Message{ID: string(msgId), Parts: a2a.ContentParts{a2a.NewTextPart("TEST")}, Role: a2a.MessageRoleUser},
		Config:  &a2a.SendMessageConfig{ReturnImmediately: true},
	})

	if sendErr != nil {
		t.Fatalf("client.SendMessage() error = %v, want nil", sendErr)
	}

	taskID := result.TaskInfo().TaskID

	task, err := client.CancelTask(t.Context(), &a2a.CancelTaskRequest{ID: taskID})
	if err != nil {
		t.Fatalf("client.CancelTask() error = %v, want nil", err)
	}

	if task.Status.State != a2a.TaskStateCanceled {
		t.Fatalf("executor.Cancel() state = %v, want %v", task.Status.State, a2a.TaskStateCanceled)
	}

	// Verify that execution context is closed
	select {
	case <-channel:
		t.Log("Agent successfully unblocked")
	case <-time.After(1 * time.Second):
		t.Fatal("Agent did not unblock")
	}
}

func TestExecutor_Converters(t *testing.T) {
	task := &a2a.Task{ID: a2a.NewTaskID(), ContextID: a2a.NewContextID()}
	hiMsg := a2a.NewMessageForTask(a2a.MessageRoleUser, task, a2a.NewTextPart("hi"))

	t.Run("A2APartConverter", func(t *testing.T) {
		t.Run("modify input", func(t *testing.T) {
			var receivedText string
			agent, err := agent.New(agent.Config{
				Name: "test",
				Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
					if parts := ctx.UserContent().Parts; len(parts) > 0 {
						receivedText = parts[0].Text
					}
					return func(yield func(*session.Event, error) bool) {}
				},
			})
			if err != nil {
				t.Fatalf("agent.New() error = %v", err)
			}

			executor := NewExecutor(ExecutorConfig{
				RunnerConfig: runner.Config{AppName: agent.Name(), Agent: agent, SessionService: session.InMemoryService()},
				A2APartConverter: func(ctx context.Context, evt a2a.Event, part *a2a.Part) (*genai.Part, error) {
					if part.Text() == "hi" {
						return genai.NewPartFromText("HELLO"), nil
					}
					return nil, nil
				},
			})

			reqCtx := &a2asrv.ExecutorContext{TaskID: task.ID, ContextID: task.ContextID, Message: hiMsg, StoredTask: task}
			for event, err := range executor.Execute(t.Context(), reqCtx) {
				if err != nil {
					t.Fatalf("executor.Execute() error = %v", err)
				}
				_ = event // ignored
			}

			if receivedText != "HELLO" {
				t.Errorf("received text = %q, want %q", receivedText, "HELLO")
			}
		})

		t.Run("filter input", func(t *testing.T) {
			var receivedParts int
			agent, err := agent.New(agent.Config{
				Name: "test",
				Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
					receivedParts = len(ctx.UserContent().Parts)
					return func(yield func(*session.Event, error) bool) {}
				},
			})
			if err != nil {
				t.Fatalf("agent.New() error = %v", err)
			}

			executor := NewExecutor(ExecutorConfig{
				RunnerConfig: runner.Config{AppName: agent.Name(), Agent: agent, SessionService: session.InMemoryService()},
				A2APartConverter: func(ctx context.Context, evt a2a.Event, part *a2a.Part) (*genai.Part, error) {
					return nil, nil
				},
			})

			reqCtx := &a2asrv.ExecutorContext{TaskID: task.ID, ContextID: task.ContextID, Message: hiMsg, StoredTask: task}
			for event, err := range executor.Execute(t.Context(), reqCtx) {
				if err != nil {
					t.Fatalf("executor.Execute() error = %v", err)
				}
				_ = event // ignored
			}

			if receivedParts != 0 {
				t.Errorf("received parts count = %d, want 0", receivedParts)
			}
		})
	})

	t.Run("GenAIPartConverter", func(t *testing.T) {
		agentEvents := []*session.Event{
			{LLMResponse: model.LLMResponse{
				Content: &genai.Content{Parts: []*genai.Part{genai.NewPartFromText("world")}},
			}},
		}

		t.Run("modify output", func(t *testing.T) {
			agent, err := newEventReplayAgent(agentEvents, nil)
			if err != nil {
				t.Fatalf("newEventReplayAgent() error = %v", err)
			}

			executor := NewExecutor(ExecutorConfig{
				RunnerConfig: runner.Config{AppName: agent.Name(), Agent: agent, SessionService: session.InMemoryService()},
				GenAIPartConverter: func(ctx context.Context, evt *session.Event, part *genai.Part) (*a2a.Part, error) {
					if part.Text == "world" {
						return a2a.NewTextPart("WORLD"), nil
					}
					return a2a.NewTextPart(part.Text), nil
				},
			})

			var gotEvents []a2a.Event
			reqCtx := &a2asrv.ExecutorContext{TaskID: task.ID, ContextID: task.ContextID, Message: hiMsg, StoredTask: task}
			for event, err := range executor.Execute(t.Context(), reqCtx) {
				if err != nil {
					t.Fatalf("executor.Execute() error = %v", err)
				}
				gotEvents = append(gotEvents, event)
			}

			found := false
			for _, e := range gotEvents {
				if ae, ok := e.(*a2a.TaskArtifactUpdateEvent); ok {
					for _, p := range ae.Artifact.Parts {
						if p.Text() == "WORLD" {
							found = true
						}
					}
				}
			}
			if !found {
				t.Errorf("did not find 'WORLD' in events: %v", gotEvents)
			}
		})

		t.Run("filter output", func(t *testing.T) {
			agent, err := newEventReplayAgent(agentEvents, nil)
			if err != nil {
				t.Fatalf("newEventReplayAgent() error = %v", err)
			}

			executor := NewExecutor(ExecutorConfig{
				RunnerConfig: runner.Config{AppName: agent.Name(), Agent: agent, SessionService: session.InMemoryService()},
				GenAIPartConverter: func(ctx context.Context, evt *session.Event, part *genai.Part) (*a2a.Part, error) {
					return nil, nil
				},
			})

			var gotEvents []a2a.Event
			reqCtx := &a2asrv.ExecutorContext{TaskID: task.ID, ContextID: task.ContextID, Message: hiMsg, StoredTask: task}
			for event, err := range executor.Execute(t.Context(), reqCtx) {
				if err != nil {
					t.Fatalf("executor.Execute() error = %v", err)
				}
				gotEvents = append(gotEvents, event)
			}

			for _, e := range gotEvents {
				if ae, ok := e.(*a2a.TaskArtifactUpdateEvent); ok {
					if len(ae.Artifact.Parts) != 0 {
						t.Errorf("found %d parts but expected 0", len(ae.Artifact.Parts))
					}
				}
			}
		})
	})
}

func TestExecutor_OutputArtifactPerEvent(t *testing.T) {
	task := &a2a.Task{ID: a2a.NewTaskID(), ContextID: a2a.NewContextID()}
	hiMsg := a2a.NewMessageForTask(a2a.MessageRoleUser, task, a2a.NewTextPart("hi"))

	testCases := []struct {
		name             string
		events           []*session.Event
		wantEvents       []a2a.Event
		wantArtifactMeta []map[string]any
	}{
		{
			name: "single artifact per event chain",
			events: []*session.Event{
				{LLMResponse: modelPartialResponseFromParts(genai.NewPartFromText("Hello, ")), Author: "agent"},
				{LLMResponse: modelPartialResponseFromParts(genai.NewPartFromText("world!")), Author: "agent"},
				{LLMResponse: modelResponseFromParts(genai.NewPartFromText("Hello, world!")), Author: "agent"},
			},
			wantEvents: []a2a.Event{
				a2a.NewStatusUpdateEvent(task, a2a.TaskStateWorking, nil),
				&a2a.TaskArtifactUpdateEvent{
					Artifact: &a2a.Artifact{Parts: a2a.ContentParts{a2a.NewTextPart("Hello, ")}},
					Append:   false, LastChunk: false,
				},
				&a2a.TaskArtifactUpdateEvent{
					Artifact: &a2a.Artifact{Parts: a2a.ContentParts{a2a.NewTextPart("world!")}},
					Append:   true, LastChunk: false,
				},
				&a2a.TaskArtifactUpdateEvent{
					Artifact: &a2a.Artifact{Parts: a2a.ContentParts{a2a.NewTextPart("Hello, world!")}},
					Append:   false, LastChunk: true,
				},
				newFinalStatusUpdate(task, a2a.TaskStateCompleted, nil),
			},
		},
		{
			name: "multiple authors",
			events: []*session.Event{
				{LLMResponse: modelPartialResponseFromParts(genai.NewPartFromText("Agent1: H")), Author: "agent1"},
				{LLMResponse: modelPartialResponseFromParts(genai.NewPartFromText("Agent2: W")), Author: "agent2"},
				{LLMResponse: modelResponseFromParts(genai.NewPartFromText("Agent1: Hello")), Author: "agent1"},
				{LLMResponse: modelResponseFromParts(genai.NewPartFromText("Agent2: World")), Author: "agent2"},
			},
			wantEvents: []a2a.Event{
				a2a.NewStatusUpdateEvent(task, a2a.TaskStateWorking, nil),
				&a2a.TaskArtifactUpdateEvent{
					Artifact: &a2a.Artifact{Parts: a2a.ContentParts{a2a.NewTextPart("Agent1: H")}},
					Append:   false, LastChunk: false,
				},
				&a2a.TaskArtifactUpdateEvent{
					Artifact: &a2a.Artifact{Parts: a2a.ContentParts{a2a.NewTextPart("Agent2: W")}},
					Append:   false, LastChunk: false,
				},
				&a2a.TaskArtifactUpdateEvent{
					Artifact: &a2a.Artifact{Parts: a2a.ContentParts{a2a.NewTextPart("Agent1: Hello")}},
					Append:   false, LastChunk: true,
				},
				&a2a.TaskArtifactUpdateEvent{
					Artifact: &a2a.Artifact{Parts: a2a.ContentParts{a2a.NewTextPart("Agent2: World")}},
					Append:   false, LastChunk: true,
				},
				newFinalStatusUpdate(task, a2a.TaskStateCompleted, nil),
			},
		},
		{
			name: "metadata and thoughts",
			events: []*session.Event{
				{
					LLMResponse:  modelPartialResponseFromParts(&genai.Part{Text: "Thinking...", Thought: true}),
					Author:       "agent",
					InvocationID: "inv1",
				},
				{
					LLMResponse:  modelResponseFromParts(genai.NewPartFromText("Done")),
					Author:       "agent",
					InvocationID: "inv1",
				},
			},
			wantEvents: []a2a.Event{
				a2a.NewStatusUpdateEvent(task, a2a.TaskStateWorking, nil),
				&a2a.TaskArtifactUpdateEvent{
					Artifact: &a2a.Artifact{
						Parts: a2a.ContentParts{
							func() *a2a.Part {
								p := a2a.NewTextPart("Thinking...")
								p.SetMeta(ToA2AMetaKey("thought"), true)
								return p
							}(),
						},
					},
					Append: false, LastChunk: false,
				},
				&a2a.TaskArtifactUpdateEvent{
					Artifact: &a2a.Artifact{
						Parts: a2a.ContentParts{a2a.NewTextPart("Done")},
					},
					Append: false, LastChunk: true,
				},
				newFinalStatusUpdate(task, a2a.TaskStateCompleted, nil),
			},
			wantArtifactMeta: []map[string]any{
				{ToA2AMetaKey("invocation_id"): "inv1"},
				{ToA2AMetaKey("invocation_id"): "inv1"},
			},
		},
		{
			name: "mixed segments",
			events: []*session.Event{
				{
					LLMResponse: modelResponseFromParts(
						&genai.Part{Text: "Thought part", Thought: true},
						genai.NewPartFromText("Text part"),
					),
					Author: "agent",
				},
			},
			wantEvents: []a2a.Event{
				a2a.NewStatusUpdateEvent(task, a2a.TaskStateWorking, nil),
				&a2a.TaskArtifactUpdateEvent{
					Artifact: &a2a.Artifact{
						Parts: a2a.ContentParts{
							func() *a2a.Part {
								p := a2a.NewTextPart("Thought part")
								p.SetMeta(ToA2AMetaKey("thought"), true)
								return p
							}(),
							a2a.NewTextPart("Text part"),
						},
					},
					Append: false, LastChunk: true,
				},
				newFinalStatusUpdate(task, a2a.TaskStateCompleted, nil),
			},
		},
		{
			name: "sequential distinct artifact chains",
			events: []*session.Event{
				{LLMResponse: modelPartialResponseFromParts(genai.NewPartFromText("1.a")), Author: "agent"},
				{LLMResponse: modelResponseFromParts(genai.NewPartFromText("1.final")), Author: "agent"},
				{LLMResponse: modelPartialResponseFromParts(genai.NewPartFromText("2.a")), Author: "agent"},
				{LLMResponse: modelResponseFromParts(genai.NewPartFromText("2.final")), Author: "agent"},
			},
			wantEvents: []a2a.Event{
				a2a.NewStatusUpdateEvent(task, a2a.TaskStateWorking, nil),
				&a2a.TaskArtifactUpdateEvent{
					Artifact: &a2a.Artifact{Parts: a2a.ContentParts{a2a.NewTextPart("1.a")}},
					Append:   false, LastChunk: false,
				},
				&a2a.TaskArtifactUpdateEvent{
					Artifact: &a2a.Artifact{Parts: a2a.ContentParts{a2a.NewTextPart("1.final")}},
					Append:   false, LastChunk: true,
				},
				&a2a.TaskArtifactUpdateEvent{
					Artifact: &a2a.Artifact{Parts: a2a.ContentParts{a2a.NewTextPart("2.a")}},
					Append:   false, LastChunk: false,
				},
				&a2a.TaskArtifactUpdateEvent{
					Artifact: &a2a.Artifact{Parts: a2a.ContentParts{a2a.NewTextPart("2.final")}},
					Append:   false, LastChunk: true,
				},
				newFinalStatusUpdate(task, a2a.TaskStateCompleted, nil),
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			agent, _ := newEventReplayAgent(tc.events, nil)
			executor := NewExecutor(ExecutorConfig{
				RunnerConfig: runner.Config{AppName: agent.Name(), Agent: agent, SessionService: session.InMemoryService()},
				OutputMode:   OutputArtifactPerEvent,
			})

			var gotEvents []a2a.Event
			reqCtx := &a2asrv.ExecutorContext{TaskID: task.ID, ContextID: task.ContextID, Message: hiMsg, StoredTask: task}

			for event, err := range executor.Execute(t.Context(), reqCtx) {
				if err != nil {
					t.Fatalf("executor.Execute() error = %v", err)
				}
				gotEvents = append(gotEvents, event)
			}

			ignoreOptions := []cmp.Option{
				cmpopts.IgnoreFields(a2a.Message{}, "ID"),
				cmpopts.IgnoreFields(a2a.Artifact{}, "ID", "Metadata"), // checked manually
				cmpopts.IgnoreFields(a2a.TaskStatus{}, "Timestamp"),
				cmpopts.IgnoreFields(a2a.TaskStatusUpdateEvent{}, "Metadata", "TaskID", "ContextID"),
				cmpopts.IgnoreFields(a2a.TaskArtifactUpdateEvent{}, "Metadata", "TaskID", "ContextID"),
			}
			if len(gotEvents) != len(tc.wantEvents) {
				t.Fatalf("got %d events, want %d", len(gotEvents), len(tc.wantEvents))
			}

			authorToID, lastFinishedID := make(map[string]a2a.ArtifactID), make(map[string]a2a.ArtifactID)
			artifactCount := 0

			for i := range gotEvents {
				got := gotEvents[i]
				want := tc.wantEvents[i]

				if diff := cmp.Diff(want, got, ignoreOptions...); diff != "" {
					t.Errorf("event[%d] mismatch (-want +got):\n%s", i, diff)
				}

				// Metadata check for cases that care about it
				if ge, ok := got.(*a2a.TaskArtifactUpdateEvent); ok {
					if artifactCount < len(tc.wantArtifactMeta) {
						wantMeta := tc.wantArtifactMeta[artifactCount]
						for k, v := range wantMeta {
							if gotV := ge.Artifact.Metadata[k]; gotV != v {
								t.Errorf("event[%d] Metadata[%s] = %v, want %v", i, k, gotV, v)
							}
						}
					}
					artifactCount++
				}

				// Custom check for ArtifactID consistency
				if ge, ok := got.(*a2a.TaskArtifactUpdateEvent); ok {
					author := tc.events[i-1].Author // i-1 because first event is TaskStatusUpdateEvent
					if id, ok := authorToID[author]; ok {
						if ge.Artifact.ID != id {
							t.Errorf("event[%d] expected ArtifactID %v, got %v", i, id, ge.Artifact.ID)
						}
					} else {
						// New artifact stream started for this author
						if prevID, ok := lastFinishedID[author]; ok {
							if ge.Artifact.ID == prevID {
								t.Errorf("event[%d] expected NEW ArtifactID, but got same as previous chain: %v", i, ge.Artifact.ID)
							}
						}
						authorToID[author] = ge.Artifact.ID
					}
					if !tc.events[i-1].Partial {
						// Stream ended for this author
						lastFinishedID[author] = ge.Artifact.ID
						delete(authorToID, author)
					}
				}
			}
		})
	}
}

func TestExecutor_RunnerProvider(t *testing.T) {
	wantText := "Hello"
	ctx := t.Context()
	task := &a2a.Task{ID: a2a.NewTaskID(), ContextID: a2a.NewContextID()}
	hiMsg := a2a.NewMessageForTask(a2a.MessageRoleUser, task, a2a.NewTextPart("hi"))
	reqCtx := &a2asrv.ExecutorContext{TaskID: task.ID, ContextID: task.ContextID, Message: hiMsg, StoredTask: task}

	runnerConfig := runner.Config{
		AppName:        "test",
		SessionService: session.InMemoryService(),
		Agent:          utils.Must(agent.New(agent.Config{Name: "agent"})),
	}
	executor := NewExecutor(ExecutorConfig{
		RunnerConfig: runnerConfig,
		RunnerProvider: func(pCtx context.Context, pReqCtx *a2asrv.ExecutorContext, plugin *plugin.Plugin) (RunnerConfig, Runner, error) {
			return toInternalRunnerConfig(runnerConfig), &testRunner{
				runFunc: func(ctx context.Context, userID, sessionID string, msg *genai.Content, cfg agent.RunConfig) iter.Seq2[*session.Event, error] {
					return func(yield func(*session.Event, error) bool) {
						yield(&session.Event{LLMResponse: modelResponseFromParts(genai.NewPartFromText(wantText))}, nil)
					}
				},
			}, nil
		},
	})

	var events []a2a.Event
	for event, err := range executor.Execute(ctx, reqCtx) {
		if err != nil {
			t.Fatalf("executor.Execute() error = %v", err)
		}
		events = append(events, event)
	}
	if len(events) < 2 {
		t.Fatalf("executor.Execute() produced %d events, want at least 2", len(events))
	}
	ta, ok := events[1].(*a2a.TaskArtifactUpdateEvent)
	if !ok {
		t.Fatalf("queue.events[1] = %T, want a2a.TaskArtifactUpdateEvent", events[1])
	}
	if ta.Artifact.Parts[0].Text() != wantText {
		t.Fatalf("ta.Artifact.Parts[0] = %v, want text part with text = %q", ta.Artifact.Parts[0], wantText)
	}
}

type testRunner struct {
	runFunc func(ctx context.Context, userID, sessionID string, msg *genai.Content, cfg agent.RunConfig) iter.Seq2[*session.Event, error]
}

func (r *testRunner) Run(ctx context.Context, userID, sessionID string, msg *genai.Content, cfg agent.RunConfig) iter.Seq2[*session.Event, error] {
	return r.runFunc(ctx, userID, sessionID, msg, cfg)
}
