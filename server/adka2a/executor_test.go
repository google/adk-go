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
	"testing"

	"github.com/a2aproject/a2a-go/a2a"
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

type eventIndex struct{ i int }

func TestExecutor_Execute(t *testing.T) {
	task := &a2a.Task{ID: a2a.NewTaskID(), ContextID: a2a.NewContextID()}
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
			request: &a2a.MessageSendParams{Message: a2a.NewMessage(a2a.MessageRoleUser, a2a.TextPart{Text: "hi"})},
			events: []*session.Event{
				{LLMResponse: modelResponseFromParts(genai.NewPartFromText("Hello"))},
				{LLMResponse: modelResponseFromParts(genai.NewPartFromText(", world!"))},
			},
			wantEvents: []a2a.Event{
				a2a.NewStatusUpdateEvent(task, a2a.TaskStateSubmitted, nil),
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
			queue := &testQueue{Queue: eventqueue.NewInMemoryQueue(10), writeErr: tc.queueWriteFails}
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

	queue := &testQueue{Queue: eventqueue.NewInMemoryQueue(10)}

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
	queue := eventqueue.NewInMemoryQueue(100)

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
