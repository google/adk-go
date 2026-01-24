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

	"github.com/a2aproject/a2a-go/a2a"
	"github.com/a2aproject/a2a-go/a2aclient"
	"github.com/a2aproject/a2a-go/a2asrv"
	"github.com/a2aproject/a2a-go/a2asrv/eventqueue"
	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/model"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/session"
)

type testQueue struct {
	eventqueue.Queue
	events   []a2a.Event
	writeErr *eventIndex
}

func (q *testQueue) Write(_ context.Context, e a2a.Event) error {
	if q.writeErr != nil && q.writeErr.i == len(q.events) {
		return fmt.Errorf("queue write failed")
	}
	q.events = append(q.events, e)
	return nil
}

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

func newInMemoryQueue(t *testing.T) eventqueue.Queue {
	t.Helper()
	qm := eventqueue.NewInMemoryManager()
	q, err := qm.GetOrCreate(t.Context(), "test")
	if err != nil {
		t.Fatalf("qm.GetOrCreate() error = %v", err)
	}
	return q
}

type eventIndex struct{ i int }

func TestExecutor_Execute(t *testing.T) {
	task := &a2a.Task{ID: a2a.NewTaskID(), ContextID: a2a.NewContextID()}
	hiMsg := a2a.NewMessage(a2a.MessageRoleUser, a2a.TextPart{Text: "hi"})
	hiMsgForTask := a2a.NewMessageForTask(a2a.MessageRoleUser, task, a2a.TextPart{Text: "hi"})

	testCases := []struct {
		name               string
		request            *a2a.MessageSendParams
		events             []*session.Event
		wantEvents         []a2a.Event
		createSessionFails bool
		agentRunFails      error
		queueWriteFails    *eventIndex
		wantErr            bool
	}{
		{
			name:    "no message",
			request: &a2a.MessageSendParams{},
			wantErr: true,
		},
		{
			name: "malformed data",
			request: &a2a.MessageSendParams{Message: a2a.NewMessageForTask(a2a.MessageRoleUser, task, a2a.FilePart{
				File: a2a.FileBytes{Bytes: "(*_*)"}, // malformed base64
			})},
			wantErr: true,
		},
		{
			name:               "session setup fails",
			request:            &a2a.MessageSendParams{Message: hiMsgForTask},
			createSessionFails: true,
			wantEvents: []a2a.Event{
				newFinalStatusUpdate(
					task, a2a.TaskStateFailed,
					a2a.NewMessageForTask(a2a.MessageRoleAgent, task, a2a.TextPart{Text: "failed to create a session: session creation failed"}),
				),
			},
		},
		{
			name:    "success for a new task",
			request: &a2a.MessageSendParams{Message: hiMsg},
			events: []*session.Event{
				{LLMResponse: modelResponseFromParts(genai.NewPartFromText("Hello"))},
				{LLMResponse: modelResponseFromParts(genai.NewPartFromText(", world!"))},
			},
			wantEvents: []a2a.Event{
				a2a.NewSubmittedTask(task, hiMsg),
				a2a.NewStatusUpdateEvent(task, a2a.TaskStateWorking, nil),
				a2a.NewArtifactEvent(task, a2a.TextPart{Text: "Hello"}),
				a2a.NewArtifactUpdateEvent(task, a2a.NewArtifactID(), a2a.TextPart{Text: ", world!"}),
				newArtifactLastChunkEvent(task),
				newFinalStatusUpdate(task, a2a.TaskStateCompleted, nil),
			},
		},
		{
			name:    "success for existing task",
			request: &a2a.MessageSendParams{Message: hiMsgForTask},
			events: []*session.Event{
				{LLMResponse: modelResponseFromParts(genai.NewPartFromText("Hello"))},
				{LLMResponse: modelResponseFromParts(genai.NewPartFromText(", world!"))},
			},
			wantEvents: []a2a.Event{
				a2a.NewStatusUpdateEvent(task, a2a.TaskStateWorking, nil),
				a2a.NewArtifactEvent(task, a2a.TextPart{Text: "Hello"}),
				a2a.NewArtifactUpdateEvent(task, a2a.NewArtifactID(), a2a.TextPart{Text: ", world!"}),
				newArtifactLastChunkEvent(task),
				newFinalStatusUpdate(task, a2a.TaskStateCompleted, nil),
			},
		},
		{
			name:            "queue write fails",
			request:         &a2a.MessageSendParams{Message: hiMsgForTask},
			queueWriteFails: &eventIndex{0},
			wantErr:         true,
		},
		{
			name:    "llm fails",
			request: &a2a.MessageSendParams{Message: hiMsgForTask},
			events: []*session.Event{
				{LLMResponse: modelResponseFromParts(genai.NewPartFromText("Hello"))},
				{LLMResponse: model.LLMResponse{ErrorCode: "418", ErrorMessage: "I'm a teapot"}},
				{LLMResponse: modelResponseFromParts(genai.NewPartFromText(", world!"))},
			},
			wantEvents: []a2a.Event{
				a2a.NewStatusUpdateEvent(task, a2a.TaskStateWorking, nil),
				a2a.NewArtifactEvent(task, a2a.TextPart{Text: "Hello"}),
				a2a.NewArtifactUpdateEvent(task, a2a.NewArtifactID(), a2a.TextPart{Text: ", world!"}),
				newArtifactLastChunkEvent(task),
				toTaskFailedUpdateEvent(
					task, errorFromResponse(&model.LLMResponse{ErrorCode: "418", ErrorMessage: "I'm a teapot"}),
					map[string]any{ToA2AMetaKey("error_code"): "418"},
				),
			},
		},
		{
			name:    "agent run fails",
			request: &a2a.MessageSendParams{Message: hiMsgForTask},
			events: []*session.Event{
				{LLMResponse: modelResponseFromParts(genai.NewPartFromText("Hello"))},
			},
			agentRunFails: fmt.Errorf("OOF"),
			wantEvents: []a2a.Event{
				a2a.NewStatusUpdateEvent(task, a2a.TaskStateWorking, nil),
				a2a.NewArtifactEvent(task, a2a.TextPart{Text: "Hello"}),
				newFinalStatusUpdate(
					task, a2a.TaskStateFailed,
					a2a.NewMessageForTask(a2a.MessageRoleAgent, task, a2a.TextPart{Text: "agent run failed: OOF"}),
				),
			},
		},
		{
			name:    "agent run and queue write fail",
			request: &a2a.MessageSendParams{Message: hiMsgForTask},
			events: []*session.Event{
				{LLMResponse: modelResponseFromParts(genai.NewPartFromText("Hello"))},
			},
			queueWriteFails: &eventIndex{2},
			agentRunFails:   fmt.Errorf("OOF"),
			wantErr:         true,
			wantEvents: []a2a.Event{
				a2a.NewStatusUpdateEvent(task, a2a.TaskStateWorking, nil),
				a2a.NewArtifactEvent(task, a2a.TextPart{Text: "Hello"}),
			},
		},
	}

	for _, tc := range testCases {
		ignoreFields := []cmp.Option{
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
			queue := &testQueue{Queue: newInMemoryQueue(t), writeErr: tc.queueWriteFails}
			reqCtx := &a2asrv.RequestContext{TaskID: task.ID, ContextID: task.ContextID, Message: tc.request.Message}
			if tc.request.Message != nil && tc.request.Message.TaskID == task.ID {
				reqCtx.StoredTask = task
			}

			err = executor.Execute(t.Context(), reqCtx, queue)
			if err != nil && !tc.wantErr {
				t.Fatalf("executor.Execute() error = %v, want nil", err)
			}
			if err == nil && tc.wantErr {
				t.Fatalf("executor.Execute() produced %d events, want error", len(queue.events))
			}
			if tc.wantEvents != nil {
				if diff := cmp.Diff(tc.wantEvents, queue.events, ignoreFields...); diff != "" {
					t.Fatalf("executor.Execute() wrong events (+got,-want):\ngot = %v\nwant = %v\ndiff = %s", queue.events, tc.wantEvents, diff)
				}
			}
		})
	}
}

func TestExecutor_Cancel(t *testing.T) {
	task := &a2a.Task{ID: a2a.NewTaskID(), ContextID: a2a.NewContextID()}
	executor := NewExecutor(ExecutorConfig{})
	reqCtx := &a2asrv.RequestContext{TaskID: task.ID, ContextID: task.ContextID}

	queue := &testQueue{Queue: newInMemoryQueue(t)}

	reqCtx.StoredTask = task
	err := executor.Cancel(t.Context(), reqCtx, queue)
	if err != nil {
		t.Fatalf("executor.Cancel() error = %v, want nil", err)
	}
	if len(queue.events) != 1 {
		t.Fatalf("executor.Cancel() produced %d events, want 1", queue.events)
	}
	event := queue.events[0].(*a2a.TaskStatusUpdateEvent)
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
	req := &a2a.MessageSendParams{Message: a2a.NewMessageForTask(a2a.MessageRoleUser, task)}
	reqCtx := &a2asrv.RequestContext{TaskID: task.ID, ContextID: task.ContextID, Message: req.Message}
	runnerConfig := runner.Config{AppName: agent.Name(), Agent: agent, SessionService: sessionService}
	config := ExecutorConfig{RunnerConfig: runnerConfig}
	executor := NewExecutor(config)
	queue := newInMemoryQueue(t)

	err = executor.Execute(ctx, reqCtx, queue)
	if err != nil {
		t.Fatalf("executor.Execute() error = %v, want nil", err)
	}
	err = executor.Execute(ctx, reqCtx, queue)
	if err != nil {
		t.Fatalf("executor.Execute() error = %v, want nil", err)
	}

	meta := toInvocationMeta(ctx, config, reqCtx)
	sessions, err := sessionService.List(ctx, &session.ListRequest{AppName: runnerConfig.AppName, UserID: meta.userID})
	if err != nil {
		t.Fatalf("sessionService.List() error = %v, want nil", err)
	}
	if len(sessions.Sessions) != 1 {
		t.Fatalf("sessionService.List() got %d sessions, want 1", sessions.Sessions)
	}

	reqCtx.ContextID = a2a.NewContextID()
	otherContextMeta := toInvocationMeta(ctx, config, reqCtx)
	if meta.sessionID == otherContextMeta.sessionID {
		t.Fatal("want sessionID to be different for different contextIDs")
	}
}

func TestExecutor_Callbacks(t *testing.T) {
	type contextKeyType struct{}
	task := &a2a.Task{ID: a2a.NewTaskID(), ContextID: a2a.NewContextID()}
	hiMsg := a2a.NewMessageForTask(a2a.MessageRoleUser, task, a2a.TextPart{Text: "hi"})

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
			beforeExecution: func(ctx context.Context, reqCtx *a2asrv.RequestContext) (context.Context, error) {
				return nil, fmt.Errorf("aborted")
			},
			wantErr: fmt.Errorf("aborted"),
		},
		{
			name: "instrument context",
			beforeExecution: func(ctx context.Context, reqCtx *a2asrv.RequestContext) (context.Context, error) {
				return context.WithValue(ctx, contextKeyType{}, "bar"), nil
			},
			afterExecution: func(ctx ExecutorContext, finalUpdate *a2a.TaskStatusUpdateEvent, err error) error {
				text, _ := ctx.Value(contextKeyType{}).(string)
				finalUpdate.Status.Message = a2a.NewMessage(a2a.MessageRoleAgent, a2a.TextPart{Text: text})
				return nil
			},
			wantEvents: []a2a.Event{
				a2a.NewStatusUpdateEvent(task, a2a.TaskStateWorking, nil),
				newFinalStatusUpdate(task, a2a.TaskStateCompleted, a2a.NewMessage(a2a.MessageRoleAgent, a2a.TextPart{Text: "bar"})),
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
				finalUpdate.Status.Message = a2a.NewMessage(a2a.MessageRoleAgent, a2a.TextPart{Text: "bar"})
				return nil
			},
			wantEvents: []a2a.Event{
				a2a.NewStatusUpdateEvent(task, a2a.TaskStateWorking, nil),
				newFinalStatusUpdate(task, a2a.TaskStateFailed, a2a.NewMessage(a2a.MessageRoleAgent, a2a.TextPart{Text: "bar"})),
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
				finalUpdate.Status.Message = a2a.NewMessage(a2a.MessageRoleAgent, a2a.TextPart{Text: fmt.Sprintf("%d events", eventCount)})
				return nil
			},
			wantEvents: []a2a.Event{newFinalStatusUpdate(task, a2a.TaskStateFailed, a2a.NewMessage(a2a.MessageRoleAgent, a2a.TextPart{Text: "0 events"}))},
		},
		{
			name: "enrich event",
			events: []*session.Event{
				{LLMResponse: modelResponseFromParts(genai.NewPartFromText("Hello"))},
				{LLMResponse: modelResponseFromParts(genai.NewPartFromText(", world!"))},
			},
			afterEvent: func(ctx ExecutorContext, event *session.Event, processed *a2a.TaskArtifactUpdateEvent) error {
				processed.Artifact.Parts = append(processed.Artifact.Parts, a2a.TextPart{Text: " (enriched)"})
				return nil
			},
			wantEvents: []a2a.Event{
				a2a.NewStatusUpdateEvent(task, a2a.TaskStateWorking, nil),
				a2a.NewArtifactEvent(task, a2a.TextPart{Text: "Hello"}, a2a.TextPart{Text: " (enriched)"}),
				a2a.NewArtifactUpdateEvent(task, a2a.NewArtifactID(), a2a.TextPart{Text: ", world!"}, a2a.TextPart{Text: " (enriched)"}),
				newArtifactLastChunkEvent(task),
				newFinalStatusUpdate(task, a2a.TaskStateCompleted, nil),
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
		ignoreFields := []cmp.Option{
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
			queue := &testQueue{Queue: newInMemoryQueue(t)}
			reqCtx := &a2asrv.RequestContext{TaskID: task.ID, ContextID: task.ContextID, Message: hiMsg, StoredTask: task}

			err = executor.Execute(t.Context(), reqCtx, queue)
			if err != nil && tc.wantErr == nil {
				t.Fatalf("executor.Execute() error = %v, want nil", err)
			}
			if err == nil && tc.wantErr != nil {
				t.Fatalf("executor.Execute() error = nil, want %v", tc.wantErr)
			}
			if tc.wantEvents != nil {
				if diff := cmp.Diff(tc.wantEvents, queue.events, ignoreFields...); diff != "" {
					t.Fatalf("executor.Execute() wrong events (+got,-want):\ngot = %v\nwant = %v\ndiff = %s", queue.events, tc.wantEvents, diff)
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
		Name:               "test",
		URL:                server.URL,
		PreferredTransport: a2a.TransportProtocolJSONRPC,
	}

	client, err := a2aclient.NewFromCard(t.Context(), card)
	if err != nil {
		t.Fatalf("a2aclient.NewFromCard() error = %v, want nil", err)
	}

	msgId := a2a.NewMessageID()
	blocking := false

	result, sendErr := client.SendMessage(t.Context(), &a2a.MessageSendParams{
		Message: &a2a.Message{ID: string(msgId), Parts: []a2a.Part{a2a.TextPart{Text: "TEST"}}, Role: a2a.MessageRoleUser},
		Config:  &a2a.MessageSendConfig{Blocking: &blocking},
	})

	if sendErr != nil {
		t.Fatalf("client.SendMessage() error = %v, want nil", sendErr)
	}

	taskID := result.TaskInfo().TaskID

	task, err := client.CancelTask(t.Context(), &a2a.TaskIDParams{ID: taskID})
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
