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
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/a2aproject/a2a-go/a2a"
	"github.com/a2aproject/a2a-go/a2aclient"
	"github.com/a2aproject/a2a-go/a2agrpc"
	"github.com/a2aproject/a2a-go/a2asrv"
	"github.com/a2aproject/a2a-go/a2asrv/eventqueue"
	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"google.golang.org/genai"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/test/bufconn"

	"google.golang.org/adk/agent"
	icontext "google.golang.org/adk/internal/context"
	"google.golang.org/adk/model"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/server/adka2a"
	"google.golang.org/adk/session"
)

const connBufSize int = 1024 * 1024

type mockA2AExecutor struct {
	executeFn func(ctx context.Context, reqCtx *a2asrv.RequestContext, queue eventqueue.Queue) error
}

var _ a2asrv.AgentExecutor = (*mockA2AExecutor)(nil)

func (e *mockA2AExecutor) Execute(ctx context.Context, reqCtx *a2asrv.RequestContext, queue eventqueue.Queue) error {
	if e.executeFn != nil {
		return e.executeFn(ctx, reqCtx, queue)
	}
	return fmt.Errorf("not implemented")
}

func (e *mockA2AExecutor) Cancel(ctx context.Context, reqCtx *a2asrv.RequestContext, queue eventqueue.Queue) error {
	return fmt.Errorf("not implemented")
}

func startA2AServer(t *testing.T, agentExecutor a2asrv.AgentExecutor, listener *bufconn.Listener) {
	requestHandler := a2asrv.NewHandler(agentExecutor)
	grpcHandler := a2agrpc.NewHandler(requestHandler)

	s := grpc.NewServer()
	t.Cleanup(s.Stop)

	grpcHandler.RegisterWith(s)
	if err := s.Serve(listener); err != nil {
		if !errors.Is(err, grpc.ErrServerStopped) {
			t.Errorf("Server exited with error: %v", err)
		}
	}
}

func newTestClientFactory(listener *bufconn.Listener) *a2aclient.Factory {
	withInsecureGRPC := a2aclient.WithGRPCTransport(
		grpc.WithContextDialer(func(context.Context, string) (net.Conn, error) { return listener.Dial() }),
		grpc.WithTransportCredentials(insecure.NewCredentials()),
	)
	return a2aclient.NewFactory(withInsecureGRPC)
}

func newA2ARemoteAgent(t *testing.T, name string, listener *bufconn.Listener) agent.Agent {
	t.Helper()
	card := &a2a.AgentCard{PreferredTransport: a2a.TransportProtocolGRPC, URL: "passthrough:///bufnet", Capabilities: a2a.AgentCapabilities{Streaming: true}}
	clientFactory := newTestClientFactory(listener)
	agent, err := NewA2A(A2AConfig{Name: name, AgentCard: card, ClientFactory: clientFactory})
	if err != nil {
		t.Fatalf("remoteagent.NewA2A() error = %v", err)
	}
	return agent
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
	ic := icontext.NewInvocationContext(ctx, icontext.InvocationContextParams{Session: resp.Session})
	return ic
}

func runAndCollect(ic agent.InvocationContext, agnt agent.Agent) ([]*session.Event, error) {
	collected := []*session.Event{}
	for ev, err := range agnt.Run(ic) {
		if err != nil {
			return collected, err
		}
		collected = append(collected, ev)
	}
	return collected, nil
}

func toLLMResponses(events []*session.Event) []model.LLMResponse {
	result := make([]model.LLMResponse, len(events))
	for i, v := range events {
		result[i] = v.LLMResponse
	}
	return result
}

func newADKEventReplay(t *testing.T, events []*session.Event) a2asrv.AgentExecutor {
	t.Helper()
	agnt, err := agent.New(agent.Config{
		Run: func(ic agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				for _, ev := range events {
					if !yield(ev, nil) {
						return
					}
				}
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New() error = %v", err)
	}
	return adka2a.NewExecutor(adka2a.ExecutorConfig{
		RunnerConfig: runner.Config{
			AppName:        "RemoteAgentTest",
			SessionService: session.InMemoryService(),
			Agent:          agnt,
		},
	})
}

func newA2AEventReplay(t *testing.T, events []a2a.Event) a2asrv.AgentExecutor {
	return &mockA2AExecutor{
		executeFn: func(ctx context.Context, reqCtx *a2asrv.RequestContext, queue eventqueue.Queue) error {
			for _, ev := range events {
				// A2A stack is going to fail the request if events don't have correct taskID and contextID
				switch v := ev.(type) {
				case *a2a.Message:
					v.TaskID = reqCtx.TaskID
					v.ContextID = reqCtx.ContextID
				case *a2a.Task:
					v.ID = reqCtx.TaskID
					v.ContextID = reqCtx.ContextID
				case *a2a.TaskStatusUpdateEvent:
					v.TaskID = reqCtx.TaskID
					v.ContextID = reqCtx.ContextID
				case *a2a.TaskArtifactUpdateEvent:
					v.TaskID = reqCtx.TaskID
					v.ContextID = reqCtx.ContextID
				}
				if err := queue.Write(ctx, ev); err != nil {
					t.Errorf("queue.Write() error = %v", err)
				}
			}
			return nil
		},
	}
}

func newUserHello() *session.Event {
	event := session.NewEvent("invocation")
	event.Content = genai.NewContentFromText("hello", genai.RoleUser)
	return event
}

func newFinalStatusUpdate(task *a2a.Task, state a2a.TaskState, msgParts ...a2a.Part) *a2a.TaskStatusUpdateEvent {
	event := a2a.NewStatusUpdateEvent(task, state, nil)
	if len(msgParts) > 0 {
		event.Status.Message = a2a.NewMessageForTask(a2a.MessageRoleAgent, task, msgParts...)
	}
	event.Final = true
	return event
}

func TestRemoteAgent_ADK2ADK(t *testing.T) {
	testCases := []struct {
		name          string
		remoteEvents  []*session.Event
		wantResponses []model.LLMResponse
	}{
		{
			name: "text streaming",
			remoteEvents: []*session.Event{
				{LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("hello", genai.RoleModel)}},
				{LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("world", genai.RoleModel)}},
			},
			wantResponses: []model.LLMResponse{
				{Content: genai.NewContentFromText("hello", genai.RoleModel), Partial: true},
				{Content: genai.NewContentFromText("world", genai.RoleModel), Partial: true},
				{TurnComplete: true},
			},
		},
		{
			name: "code execution",
			remoteEvents: []*session.Event{
				{LLMResponse: model.LLMResponse{Content: genai.NewContentFromExecutableCode("print('hello')", genai.LanguagePython, genai.RoleModel)}},
				{LLMResponse: model.LLMResponse{Content: genai.NewContentFromCodeExecutionResult(genai.OutcomeOK, "hello", genai.RoleModel)}},
			},
			wantResponses: []model.LLMResponse{
				{Content: genai.NewContentFromExecutableCode("print('hello')", genai.LanguagePython, genai.RoleModel), Partial: true},
				{Content: genai.NewContentFromCodeExecutionResult(genai.OutcomeOK, "hello", genai.RoleModel), Partial: true},
				{TurnComplete: true},
			},
		},
		{
			name: "function calls",
			remoteEvents: []*session.Event{
				{LLMResponse: model.LLMResponse{Content: genai.NewContentFromFunctionCall("get_weather", map[string]any{"city": "Warsaw"}, genai.RoleModel)}},
				{LLMResponse: model.LLMResponse{Content: genai.NewContentFromFunctionResponse("get_weather", map[string]any{"temo": "1C"}, genai.RoleModel)}},
			},
			wantResponses: []model.LLMResponse{
				{Content: genai.NewContentFromFunctionCall("get_weather", map[string]any{"city": "Warsaw"}, genai.RoleModel), Partial: true},
				{Content: genai.NewContentFromFunctionResponse("get_weather", map[string]any{"temo": "1C"}, genai.RoleModel), Partial: true},
				{TurnComplete: true},
			},
		},
		{
			name: "files",
			remoteEvents: []*session.Event{
				{LLMResponse: model.LLMResponse{Content: genai.NewContentFromBytes([]byte("hello"), "text", genai.RoleModel)}},
				{LLMResponse: model.LLMResponse{Content: genai.NewContentFromURI("http://text.com/text.txt", "text", genai.RoleModel)}},
			},
			wantResponses: []model.LLMResponse{
				{Content: genai.NewContentFromBytes([]byte("hello"), "text", genai.RoleModel), Partial: true},
				{Content: genai.NewContentFromURI("http://text.com/text.txt", "text", genai.RoleModel), Partial: true},
				{TurnComplete: true},
			},
		},
	}

	ignoreFields := []cmp.Option{
		cmpopts.IgnoreFields(model.LLMResponse{}, "CustomMetadata"),
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			listener := bufconn.Listen(connBufSize)
			executor := newADKEventReplay(t, tc.remoteEvents)
			go startA2AServer(t, executor, listener)
			remoteAgent := newA2ARemoteAgent(t, "a2a", listener)

			ictx := newInvocationContext(t, []*session.Event{newUserHello()})
			gotEvents, err := runAndCollect(ictx, remoteAgent)
			if err != nil {
				t.Errorf("agent.Run() error = %v", err)
			}
			gotResponses := toLLMResponses(gotEvents)
			if diff := cmp.Diff(tc.wantResponses, gotResponses, ignoreFields...); diff != "" {
				t.Errorf("agent.Run() wrong result (+got,-want):\ngot = %+v\nwant = %+v\ndiff = %s", gotResponses, tc.wantResponses, diff)
			}
			for _, event := range gotEvents {
				if _, ok := event.CustomMetadata[adka2a.ToADKMetaKey("response")]; !ok {
					t.Errorf("event.CustomMetadata = %v, want meta[%q] = original a2a event", event.CustomMetadata, adka2a.ToADKMetaKey("response"))
				}
				if _, ok := event.CustomMetadata[adka2a.ToADKMetaKey("request")]; !ok {
					t.Errorf("event.CustomMetadata = %v, want meta[%q] = original a2a request", event.CustomMetadata, adka2a.ToADKMetaKey("request"))
				}
			}
		})
	}
}

func TestRemoteAgent_ADK2A2A(t *testing.T) {
	task := &a2a.Task{ID: a2a.NewTaskID(), ContextID: a2a.NewContextID()}
	artifactEvent := a2a.NewArtifactEvent(task)

	testCases := []struct {
		name          string
		remoteEvents  []a2a.Event
		wantResponses []model.LLMResponse
	}{
		{
			name:          "empty message",
			remoteEvents:  []a2a.Event{a2a.NewMessage(a2a.MessageRoleAgent)},
			wantResponses: []model.LLMResponse{{}},
		},
		{
			name: "message",
			remoteEvents: []a2a.Event{
				a2a.NewMessage(a2a.MessageRoleAgent, a2a.TextPart{Text: "hello"}, a2a.TextPart{Text: "world"}),
			},
			wantResponses: []model.LLMResponse{
				{
					Content: &genai.Content{
						Parts: []*genai.Part{genai.NewPartFromText("hello"), genai.NewPartFromText("world")},
						Role:  genai.RoleModel,
					},
				},
			},
		},
		{
			name: "empty task",
			remoteEvents: []a2a.Event{
				&a2a.Task{Status: a2a.TaskStatus{State: a2a.TaskStateCompleted}},
			},
			wantResponses: []model.LLMResponse{{}},
		},
		{
			name: "task with status message",
			remoteEvents: []a2a.Event{
				&a2a.Task{Status: a2a.TaskStatus{
					State:   a2a.TaskStateCompleted,
					Message: a2a.NewMessage(a2a.MessageRoleAgent, a2a.TextPart{Text: "hello"}),
				}},
			},
			wantResponses: []model.LLMResponse{{Content: genai.NewContentFromText("hello", genai.RoleModel)}},
		},
		{
			name: "task with multipart artifact",
			remoteEvents: []a2a.Event{
				&a2a.Task{
					Status: a2a.TaskStatus{State: a2a.TaskStateCompleted},
					Artifacts: []*a2a.Artifact{
						{Parts: a2a.ContentParts{a2a.TextPart{Text: "hello"}, a2a.TextPart{Text: "world"}}},
					},
				},
			},
			wantResponses: []model.LLMResponse{
				{
					Content: &genai.Content{
						Parts: []*genai.Part{genai.NewPartFromText("hello"), genai.NewPartFromText("world")},
						Role:  genai.RoleModel,
					},
				},
			},
		},
		{
			name: "multiple tasks",
			remoteEvents: []a2a.Event{
				&a2a.Task{Status: a2a.TaskStatus{
					State:   a2a.TaskStateWorking,
					Message: a2a.NewMessage(a2a.MessageRoleAgent, a2a.TextPart{Text: "hello"}),
				}},
				&a2a.Task{
					Status: a2a.TaskStatus{State: a2a.TaskStateCompleted},
					Artifacts: []*a2a.Artifact{
						{Parts: a2a.ContentParts{a2a.TextPart{Text: "world"}}},
					},
				},
			},
			wantResponses: []model.LLMResponse{
				{Content: genai.NewContentFromText("hello", genai.RoleModel), Partial: true},
				{Content: genai.NewContentFromText("world", genai.RoleModel)},
			},
		},
		{
			name: "task with multiple artifacts",
			remoteEvents: []a2a.Event{
				&a2a.Task{
					Status: a2a.TaskStatus{State: a2a.TaskStateCompleted},
					Artifacts: []*a2a.Artifact{
						{Parts: a2a.ContentParts{a2a.TextPart{Text: "hello"}}},
						{Parts: a2a.ContentParts{a2a.TextPart{Text: "world"}}},
					},
				},
			},
			wantResponses: []model.LLMResponse{
				{
					Content: &genai.Content{
						Parts: []*genai.Part{genai.NewPartFromText("hello"), genai.NewPartFromText("world")},
						Role:  genai.RoleModel,
					},
				},
			},
		},
		{
			name: "artifact parts translation",
			remoteEvents: []a2a.Event{
				artifactEvent,
				a2a.NewArtifactUpdateEvent(task, artifactEvent.Artifact.ID, a2a.TextPart{Text: "hello"}),
				a2a.NewArtifactUpdateEvent(task, artifactEvent.Artifact.ID, a2a.TextPart{Text: "world"}),
				newFinalStatusUpdate(task, a2a.TaskStateCompleted),
			},
			wantResponses: []model.LLMResponse{
				{Content: genai.NewContentFromText("hello", genai.RoleModel), Partial: true},
				{Content: genai.NewContentFromText("world", genai.RoleModel), Partial: true},
				{TurnComplete: true},
			},
		},
		{
			name: "non-final status update messages as thoughts",
			remoteEvents: []a2a.Event{
				a2a.NewStatusUpdateEvent(task, a2a.TaskStateSubmitted, a2a.NewMessage(a2a.MessageRoleAgent, a2a.TextPart{Text: "submitted..."})),
				a2a.NewStatusUpdateEvent(task, a2a.TaskStateWorking, a2a.NewMessage(a2a.MessageRoleAgent, a2a.TextPart{Text: "working..."})),
				newFinalStatusUpdate(task, a2a.TaskStateCompleted, a2a.TextPart{Text: "completed!"}),
			},
			wantResponses: []model.LLMResponse{
				{Content: &genai.Content{Parts: []*genai.Part{{Text: "submitted...", Thought: true}}, Role: genai.RoleModel}, Partial: true},
				{Content: &genai.Content{Parts: []*genai.Part{{Text: "working...", Thought: true}}, Role: genai.RoleModel}, Partial: true},
				{Content: genai.NewContentFromText("completed!", genai.RoleModel), TurnComplete: true},
			},
		},
		{
			name: "empty non-final status updates ignored",
			remoteEvents: []a2a.Event{
				a2a.NewStatusUpdateEvent(task, a2a.TaskStateSubmitted, nil),
				a2a.NewStatusUpdateEvent(task, a2a.TaskStateWorking, nil),
				newFinalStatusUpdate(task, a2a.TaskStateCompleted),
			},
			wantResponses: []model.LLMResponse{
				{TurnComplete: true},
			},
		},
	}

	ignoreFields := []cmp.Option{
		cmpopts.IgnoreFields(model.LLMResponse{}, "CustomMetadata"),
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			listener := bufconn.Listen(connBufSize)
			executor := newA2AEventReplay(t, tc.remoteEvents)
			go startA2AServer(t, executor, listener)
			remoteAgent := newA2ARemoteAgent(t, "a2a", listener)

			ictx := newInvocationContext(t, []*session.Event{newUserHello()})
			gotEvents, err := runAndCollect(ictx, remoteAgent)
			if err != nil {
				t.Errorf("agent.Run() error = %v", err)
			}
			gotResponses := toLLMResponses(gotEvents)
			if diff := cmp.Diff(tc.wantResponses, gotResponses, ignoreFields...); diff != "" {
				t.Errorf("agent.Run() wrong result (+got,-want):\ngot = %+v\nwant = %+v\ndiff = %s", gotResponses, tc.wantResponses, diff)
			}
			for _, event := range gotEvents {
				if _, ok := event.CustomMetadata[adka2a.ToADKMetaKey("response")]; !ok {
					t.Errorf("event.CustomMetadata = %v, want meta[%q] = original a2a event", event.CustomMetadata, adka2a.ToADKMetaKey("response"))
				}
				if _, ok := event.CustomMetadata[adka2a.ToADKMetaKey("request")]; !ok {
					t.Errorf("event.CustomMetadata = %v, want meta[%q] = original a2a request", event.CustomMetadata, adka2a.ToADKMetaKey("request"))
				}
			}
		})
	}
}

func TestRemoteAgent_EmptyResultForEmptySession(t *testing.T) {
	listener := bufconn.Listen(connBufSize)
	ictx := newInvocationContext(t, []*session.Event{})

	executor := newA2AEventReplay(t, []a2a.Event{
		a2a.NewMessage(a2a.MessageRoleAgent, a2a.TextPart{Text: "will not be invoked, because input is empty"}),
	})
	go startA2AServer(t, executor, listener)

	agentName := "a2a agent"
	remoteAgent := newA2ARemoteAgent(t, agentName, listener)

	gotEvents, err := runAndCollect(ictx, remoteAgent)
	if err != nil {
		t.Fatalf("runAndCollect() error = %v", err)
	}

	wantEvents := []*session.Event{
		{InvocationID: ictx.InvocationID(), Author: agentName, Branch: ictx.Branch()},
	}
	ignoreFields := []cmp.Option{
		cmpopts.IgnoreFields(session.Event{}, "ID"),
		cmpopts.IgnoreFields(session.Event{}, "Timestamp"),
		cmpopts.IgnoreFields(session.EventActions{}, "StateDelta"),
	}
	if diff := cmp.Diff(wantEvents, gotEvents, ignoreFields...); diff != "" {
		t.Fatalf("agent.Run() wrong result (+got,-want):\ngot = %+v\nwant = %+v\ndiff = %s", gotEvents, wantEvents, diff)
	}
}

func TestRemoteAgent_ResolvesAgentCard(t *testing.T) {
	remoteEvents := []a2a.Event{a2a.NewMessage(a2a.MessageRoleAgent, a2a.TextPart{Text: "Hello!"})}
	wantResponses := []model.LLMResponse{{Content: genai.NewContentFromText("Hello!", genai.RoleModel)}}

	listener := bufconn.Listen(connBufSize)
	executor := newA2AEventReplay(t, remoteEvents)
	go startA2AServer(t, executor, listener)

	mux := http.NewServeMux()
	mux.HandleFunc("/.well-known/agent-card.json", func(w http.ResponseWriter, r *http.Request) {
		card := &a2a.AgentCard{PreferredTransport: a2a.TransportProtocolGRPC, URL: "passthrough:///bufnet", Capabilities: a2a.AgentCapabilities{Streaming: true}}
		if err := json.NewEncoder(w).Encode(card); err != nil {
			t.Errorf("json.Encode(agentCard) error = %v", err)
		}
	})
	cardServer := httptest.NewServer(mux)

	clientFactory := newTestClientFactory(listener)
	remoteAgent, err := NewA2A(A2AConfig{Name: "a2a", AgentCardSource: cardServer.URL, ClientFactory: clientFactory})
	if err != nil {
		t.Fatalf("remoteagent.NewA2A() error = %v", err)
	}

	ictx := newInvocationContext(t, []*session.Event{newUserHello()})
	gotEvents, err := runAndCollect(ictx, remoteAgent)
	if err != nil {
		t.Fatalf("agent.Run() error = %v", err)
	}

	ignoreFields := []cmp.Option{
		cmpopts.IgnoreFields(model.LLMResponse{}, "CustomMetadata"),
	}
	gotResponses := toLLMResponses(gotEvents)
	if diff := cmp.Diff(wantResponses, gotResponses, ignoreFields...); diff != "" {
		t.Fatalf("agent.Run() wrong result (+got,-want):\ngot = %+v\nwant = %+v\ndiff = %s", gotResponses, wantResponses, diff)
	}
}

func TestRemoteAgent_ErrorEventIfNoCompatibleTransport(t *testing.T) {
	listener := bufconn.Listen(connBufSize)
	remoteEvents := []a2a.Event{a2a.NewMessage(a2a.MessageRoleAgent, a2a.TextPart{Text: "will not be invoked!"})}
	executor := newA2AEventReplay(t, remoteEvents)
	go startA2AServer(t, executor, listener)

	clientFactory := a2aclient.NewFactory(a2aclient.WithDefaultsDisabled())
	remoteAgent, err := NewA2A(A2AConfig{
		Name:          "a2a",
		AgentCard:     &a2a.AgentCard{PreferredTransport: a2a.TransportProtocolGRPC, URL: "passthrough:///bufnet"},
		ClientFactory: clientFactory,
	})
	if err != nil {
		t.Fatalf("remoteagent.NewA2A() error = %v", err)
	}

	ictx := newInvocationContext(t, []*session.Event{newUserHello()})
	gotEvents, err := runAndCollect(ictx, remoteAgent)
	if err != nil {
		t.Fatalf("agent.Run() error = %v", err)
	}

	if len(gotEvents) != 1 {
		t.Fatalf("len(events) = %d, want 1", len(gotEvents))
	}
	if !strings.Contains(gotEvents[0].ErrorMessage, "no compatible transports found") {
		t.Fatalf("event.ErrorMessage = %s, want to contain %q", gotEvents[0].ErrorMessage, "no compatible transports found")
	}
}

func TestRemoteAgent_ErrorEventOnServerError(t *testing.T) {
	listener := bufconn.Listen(connBufSize)

	executorErr := fmt.Errorf("mockExecutor failed")
	executor := &mockA2AExecutor{
		executeFn: func(ctx context.Context, reqCtx *a2asrv.RequestContext, q eventqueue.Queue) error {
			return executorErr
		},
	}
	go startA2AServer(t, executor, listener)

	remoteAgent := newA2ARemoteAgent(t, "a2a agent", listener)

	ictx := newInvocationContext(t, []*session.Event{newUserHello()})
	gotEvents, err := runAndCollect(ictx, remoteAgent)
	if err != nil {
		t.Fatalf("agent.Run() error = %v", err)
	}

	if len(gotEvents) != 1 {
		t.Fatalf("len(events) = %d, want 1", len(gotEvents))
	}
	if !strings.Contains(gotEvents[0].ErrorMessage, executorErr.Error()) {
		t.Fatalf("event.ErrorMessage = %s, want to contain %q", gotEvents[0].ErrorMessage, executorErr.Error())
	}
}
