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
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/a2aproject/a2a-go/a2a"
	"github.com/a2aproject/a2a-go/a2aclient"
	"github.com/a2aproject/a2a-go/a2asrv"
	"github.com/a2aproject/a2a-go/a2asrv/eventqueue"
	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	icontext "google.golang.org/adk/internal/context"
	"google.golang.org/adk/model"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/server/adka2a"
	"google.golang.org/adk/session"
)

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

type testA2AServer struct {
	*httptest.Server
	handler a2asrv.RequestHandler
}

func startA2AServer(agentExecutor a2asrv.AgentExecutor) *testA2AServer {
	requestHandler := a2asrv.NewHandler(agentExecutor)
	return &testA2AServer{
		Server:  httptest.NewServer(a2asrv.NewJSONRPCHandler(requestHandler)),
		handler: requestHandler,
	}
}

func newA2ARemoteAgent(t *testing.T, name string, server *testA2AServer) agent.Agent {
	t.Helper()
	card := &a2a.AgentCard{PreferredTransport: a2a.TransportProtocolJSONRPC, URL: server.URL, Capabilities: a2a.AgentCapabilities{Streaming: true}}
	agent, err := NewA2A(A2AConfig{Name: name, AgentCard: card})
	if err != nil {
		t.Fatalf("remoteagent.NewA2A() error = %v", err)
	}
	return agent
}

func newInvocationContext(t *testing.T, events []*session.Event) agent.InvocationContext {
	return newInvocationContextWithStreamingMode(t, events, agent.StreamingModeSSE)
}

func newInvocationContextWithStreamingMode(t *testing.T, events []*session.Event, streamingMode agent.StreamingMode) agent.InvocationContext {
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
			StreamingMode: streamingMode,
		},
	})
	return ic
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

func toLLMResponses(events []*session.Event) []model.LLMResponse {
	var result []model.LLMResponse
	for _, v := range events {
		result = append(result, v.LLMResponse)
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
	event.Author = "user"
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
		wantEscalate  bool
		wantTransfer  string
		noStreaming   bool
	}{
		{
			name: "text streaming",
			remoteEvents: []*session.Event{
				{LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("hello ", genai.RoleModel)}},
				{LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("world", genai.RoleModel)}},
			},
			wantResponses: []model.LLMResponse{
				{Content: genai.NewContentFromText("hello ", genai.RoleModel), Partial: true},
				{Content: genai.NewContentFromText("world", genai.RoleModel), Partial: true},
				{
					Content:      genai.NewContentFromText("hello world", genai.RoleModel),
					TurnComplete: true,
				},
			},
		},
		{
			name: "text streaming - no streaming mode",
			remoteEvents: []*session.Event{
				{LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("hello world", genai.RoleModel)}},
			},
			wantResponses: []model.LLMResponse{
				{
					Content:      genai.NewContentFromText("hello world", genai.RoleModel),
					TurnComplete: false,
				},
			},
			noStreaming: true,
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
		{
			name: "escalation",
			remoteEvents: []*session.Event{
				{
					LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("stop", genai.RoleModel)},
					Actions:     session.EventActions{Escalate: true},
				},
			},
			wantResponses: []model.LLMResponse{
				{Content: genai.NewContentFromText("stop", genai.RoleModel), Partial: true},
				{Content: genai.NewContentFromText("stop", genai.RoleModel), TurnComplete: true},
			},
			wantEscalate: true,
		},
		{
			name: "transfer",
			remoteEvents: []*session.Event{
				{
					LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("stop", genai.RoleModel)},
					Actions:     session.EventActions{TransferToAgent: "a-2"},
				},
			},
			wantResponses: []model.LLMResponse{
				{Content: genai.NewContentFromText("stop", genai.RoleModel), Partial: true},
				{Content: genai.NewContentFromText("stop", genai.RoleModel), TurnComplete: true},
			},
			wantTransfer: "a-2",
		},
		{
			name: "long-running function call",
			remoteEvents: []*session.Event{
				{LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("Hello!", genai.RoleModel)}},
				{LLMResponse: model.LLMResponse{Content: genai.NewContentFromText(" I'll need your approval first:", genai.RoleModel)}},
				{
					LLMResponse: model.LLMResponse{Content: genai.NewContentFromParts(
						[]*genai.Part{{FunctionCall: &genai.FunctionCall{Name: "create_ticket", ID: "abc-123"}}}, genai.RoleModel,
					)},
					LongRunningToolIDs: []string{"abc-123"},
				},
				{
					LLMResponse: model.LLMResponse{Content: genai.NewContentFromParts(
						[]*genai.Part{{FunctionResponse: &genai.FunctionResponse{
							Name: "create_ticket", ID: "abc-123", Response: map[string]any{"ticket_id": "123"},
						}}}, genai.RoleModel,
					)},
					LongRunningToolIDs: []string{"abc-123"},
				},
				{LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("Waiting for the approval to continue.", genai.RoleModel)}},
			},
			wantResponses: []model.LLMResponse{
				{Content: genai.NewContentFromText("Hello!", genai.RoleModel), Partial: true},
				{Content: genai.NewContentFromText(" I'll need your approval first:", genai.RoleModel), Partial: true},
				// Aggregated partial responses are emitted before a long-running function call
				{Content: genai.NewContentFromText("Hello! I'll need your approval first:", genai.RoleModel)},
				{Content: genai.NewContentFromParts(
					[]*genai.Part{{FunctionCall: &genai.FunctionCall{Name: "create_ticket", ID: "abc-123"}}}, genai.RoleModel,
				), Partial: true},
				{Content: genai.NewContentFromParts(
					[]*genai.Part{{FunctionResponse: &genai.FunctionResponse{
						Name: "create_ticket", ID: "abc-123", Response: map[string]any{"ticket_id": "123"},
					}}}, genai.RoleModel,
				), Partial: true},
				{Content: genai.NewContentFromText("Waiting for the approval to continue.", genai.RoleModel), Partial: true},
				// Only partial responses aggregated after the previous flush are included in the terminal event.
				{Content: genai.NewContentFromText("Waiting for the approval to continue.", genai.RoleModel)},
				{
					Content: genai.NewContentFromParts(
						[]*genai.Part{
							{FunctionCall: &genai.FunctionCall{Name: "create_ticket", ID: "abc-123"}},
							{FunctionResponse: &genai.FunctionResponse{Name: "create_ticket", ID: "abc-123", Response: map[string]any{"ticket_id": "123"}}},
						},
						genai.RoleModel,
					),
					TurnComplete: true,
				},
			},
		},
		{
			name: "metadata",
			remoteEvents: []*session.Event{
				{
					LLMResponse: model.LLMResponse{
						Content:           genai.NewContentFromText("hello", genai.RoleModel),
						CitationMetadata:  &genai.CitationMetadata{Citations: []*genai.Citation{{Title: "Title1"}, {Title: "Title2"}}},
						UsageMetadata:     &genai.GenerateContentResponseUsageMetadata{CandidatesTokenCount: 12, ThoughtsTokenCount: 42},
						GroundingMetadata: &genai.GroundingMetadata{SourceFlaggingUris: []*genai.GroundingMetadataSourceFlaggingURI{{SourceID: "id1"}}},
						CustomMetadata:    map[string]any{"nested": map[string]any{"key": "value"}},
					},
				},
			},
			wantResponses: []model.LLMResponse{
				{
					Content:           genai.NewContentFromText("hello", genai.RoleModel),
					CitationMetadata:  &genai.CitationMetadata{Citations: []*genai.Citation{{Title: "Title1"}, {Title: "Title2"}}},
					UsageMetadata:     &genai.GenerateContentResponseUsageMetadata{CandidatesTokenCount: 12, ThoughtsTokenCount: 42},
					GroundingMetadata: &genai.GroundingMetadata{SourceFlaggingUris: []*genai.GroundingMetadataSourceFlaggingURI{{SourceID: "id1"}}},
					CustomMetadata:    map[string]any{"nested": map[string]any{"key": "value"}},
					Partial:           true,
				},
				{
					Content:      genai.NewContentFromText("hello", genai.RoleModel),
					TurnComplete: true,
				},
			},
		},
	}

	ignoreFields := []cmp.Option{
		cmpopts.IgnoreFields(model.LLMResponse{}, "CustomMetadata"),
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			executor := newADKEventReplay(t, tc.remoteEvents)
			remoteAgent := newA2ARemoteAgent(t, "a2a", startA2AServer(executor))

			mode := agent.StreamingModeSSE
			if tc.noStreaming {
				mode = agent.StreamingModeNone
			}
			ictx := newInvocationContextWithStreamingMode(t, []*session.Event{newUserHello()}, mode)
			gotEvents, err := runAndCollect(ictx, remoteAgent)
			if err != nil {
				t.Fatalf("agent.Run() error = %v", err)
			}
			gotResponses := toLLMResponses(gotEvents)
			if diff := cmp.Diff(tc.wantResponses, gotResponses, ignoreFields...); diff != "" {
				t.Fatalf("agent.Run() wrong result (+got,-want):\ngot = %+v\nwant = %+v\ndiff = %s", gotResponses, tc.wantResponses, diff)
			}
			var lastActions *session.EventActions
			for _, event := range gotEvents {
				if _, ok := event.CustomMetadata[adka2a.ToADKMetaKey("response")]; !ok {
					if aggregated, _ := event.CustomMetadata[adka2a.ToADKMetaKey("aggregated")].(bool); !aggregated {
						t.Fatalf("event.CustomMetadata = %v, want meta[%q] = original event or meta[%q] = true", event.CustomMetadata, adka2a.ToADKMetaKey("response"), adka2a.ToADKMetaKey("aggregated"))
					}
				}
				if _, ok := event.CustomMetadata[adka2a.ToADKMetaKey("request")]; !ok {
					t.Fatalf("event.CustomMetadata = %v, want meta[%q] = original a2a request", event.CustomMetadata, adka2a.ToADKMetaKey("request"))
				}
				lastActions = &event.Actions
			}
			if tc.wantEscalate != lastActions.Escalate {
				t.Fatalf("lastActions.Escalate = %v, want %v", lastActions.Escalate, tc.wantEscalate)
			}
			if tc.wantTransfer != lastActions.TransferToAgent {
				t.Fatalf("lastActions.TransferToAgent = %v, want %v", lastActions.TransferToAgent, tc.wantTransfer)
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
				{Content: genai.NewContentFromText("hello", genai.RoleModel)},
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
				{
					Content:      genai.NewContentFromText("helloworld", genai.RoleModel),
					TurnComplete: true,
				},
			},
		},
		{
			name: "non-final status update messages as thoughts",
			remoteEvents: []a2a.Event{
				a2a.NewStatusUpdateEvent(task, a2a.TaskStateSubmitted, a2a.NewMessage(a2a.MessageRoleAgent, a2a.TextPart{Text: "submitted...\n"})),
				a2a.NewStatusUpdateEvent(task, a2a.TaskStateWorking, a2a.NewMessage(a2a.MessageRoleAgent, a2a.TextPart{Text: "working...\n"})),
				newFinalStatusUpdate(task, a2a.TaskStateCompleted, a2a.TextPart{Text: "completed!"}),
			},
			wantResponses: []model.LLMResponse{
				{Content: &genai.Content{Parts: []*genai.Part{{Text: "submitted...\n", Thought: true}}, Role: genai.RoleModel}, Partial: true},
				{Content: &genai.Content{Parts: []*genai.Part{{Text: "working...\n", Thought: true}}, Role: genai.RoleModel}, Partial: true},
				{Content: genai.NewContentFromParts([]*genai.Part{{Thought: true, Text: "submitted...\nworking...\n"}}, genai.RoleModel)},
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
			executor := newA2AEventReplay(t, tc.remoteEvents)
			remoteAgent := newA2ARemoteAgent(t, "a2a", startA2AServer(executor))

			ictx := newInvocationContext(t, []*session.Event{newUserHello()})
			gotEvents, err := runAndCollect(ictx, remoteAgent)
			if err != nil {
				t.Fatalf("agent.Run() error = %v", err)
			}
			gotResponses := toLLMResponses(gotEvents)
			if diff := cmp.Diff(tc.wantResponses, gotResponses, ignoreFields...); diff != "" {
				t.Fatalf("agent.Run() wrong result (+got,-want):\ngot = %+v\nwant = %+v\ndiff = %s", gotResponses, tc.wantResponses, diff)
			}
			for _, event := range gotEvents {
				if _, ok := event.CustomMetadata[adka2a.ToADKMetaKey("response")]; !ok {
					if aggregated, _ := event.CustomMetadata[adka2a.ToADKMetaKey("aggregated")].(bool); !aggregated {
						t.Fatalf("event.CustomMetadata = %v, want meta[%q] = original event or meta[%q] = true", event.CustomMetadata, adka2a.ToADKMetaKey("response"), adka2a.ToADKMetaKey("aggregated"))
					}
				}
				if _, ok := event.CustomMetadata[adka2a.ToADKMetaKey("request")]; !ok {
					t.Fatalf("event.CustomMetadata = %v, want meta[%q] = original a2a request", event.CustomMetadata, adka2a.ToADKMetaKey("request"))
				}
			}
		})
	}
}

func TestRemoteAgent_RequestCallbacks(t *testing.T) {
	testCases := []struct {
		name          string
		sessionEvents []*session.Event
		events        func(*a2asrv.RequestContext) []a2a.Event
		before        []BeforeA2ARequestCallback
		after         []AfterA2ARequestCallback
		converter     A2AEventConverter
		wantResponses []model.LLMResponse
		wantErr       error
	}{
		{
			name: "request and response modification",
			events: func(rc *a2asrv.RequestContext) []a2a.Event {
				return []a2a.Event{a2a.NewMessage(a2a.MessageRoleAgent, a2a.TextPart{Text: "foo"})}
			},
			before: []BeforeA2ARequestCallback{
				func(ctx agent.CallbackContext, req *a2a.MessageSendParams) (*session.Event, error) {
					req.Metadata = map[string]any{"counter": 1}
					return nil, nil
				},
			},
			after: []AfterA2ARequestCallback{
				func(ctx agent.CallbackContext, req *a2a.MessageSendParams, result *session.Event, err error) (*session.Event, error) {
					result.Content = genai.NewContentFromText(result.Content.Parts[0].Text+"bar", genai.RoleModel)
					result.CustomMetadata = req.Metadata
					return nil, nil
				},
			},
			wantResponses: []model.LLMResponse{
				{
					Content:        genai.NewContentFromText("foobar", genai.RoleModel),
					CustomMetadata: map[string]any{"counter": 1},
				},
			},
		},
		{
			name: "after invoked for every event",
			events: func(rc *a2asrv.RequestContext) []a2a.Event {
				artifactEvent := a2a.NewArtifactEvent(rc, a2a.TextPart{Text: "Hello"})
				finalEvent := a2a.NewStatusUpdateEvent(rc, a2a.TaskStateCompleted, nil)
				finalEvent.Final = true
				return []a2a.Event{
					artifactEvent,
					a2a.NewArtifactUpdateEvent(rc, artifactEvent.Artifact.ID, a2a.TextPart{Text: ", world!"}),
					finalEvent,
				}
			},
			after: []AfterA2ARequestCallback{
				func(ctx agent.CallbackContext, req *a2a.MessageSendParams, result *session.Event, err error) (*session.Event, error) {
					result.CustomMetadata = map[string]any{"foo": "bar"}
					return nil, nil
				},
			},
			wantResponses: []model.LLMResponse{
				{
					Partial:        true,
					Content:        genai.NewContentFromText("Hello", genai.RoleModel),
					CustomMetadata: map[string]any{"foo": "bar"},
				},
				{
					Partial:        true,
					Content:        genai.NewContentFromText(", world!", genai.RoleModel),
					CustomMetadata: map[string]any{"foo": "bar"},
				},
				{
					TurnComplete:   true,
					Content:        genai.NewContentFromText("Hello, world!", genai.RoleModel),
					CustomMetadata: map[string]any{"foo": "bar"},
				},
			},
		},
		{
			name: "after error stops the run",
			events: func(rc *a2asrv.RequestContext) []a2a.Event {
				finalEvent := a2a.NewStatusUpdateEvent(rc, a2a.TaskStateCompleted, nil)
				finalEvent.Final = true
				return []a2a.Event{
					a2a.NewArtifactEvent(rc, a2a.TextPart{Text: "Hello"}),
					finalEvent,
				}
			},
			after: []AfterA2ARequestCallback{
				func(ctx agent.CallbackContext, req *a2a.MessageSendParams, result *session.Event, err error) (*session.Event, error) {
					return nil, fmt.Errorf("rejected")
				},
			},
			wantErr: fmt.Errorf("rejected"),
		},
		{
			name: "request overwrite with response",
			before: []BeforeA2ARequestCallback{
				func(ctx agent.CallbackContext, req *a2a.MessageSendParams) (*session.Event, error) {
					return &session.Event{LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("hello", genai.RoleModel)}}, nil
				},
			},
			wantResponses: []model.LLMResponse{{Content: genai.NewContentFromText("hello", genai.RoleModel)}},
		},
		{
			name: "request overwrite with error",
			before: []BeforeA2ARequestCallback{
				func(ctx agent.CallbackContext, req *a2a.MessageSendParams) (*session.Event, error) {
					return nil, fmt.Errorf("failed")
				},
			},
			wantErr: fmt.Errorf("failed"),
		},
		{
			name: "response overwrite",
			after: []AfterA2ARequestCallback{
				func(ctx agent.CallbackContext, req *a2a.MessageSendParams, result *session.Event, err error) (*session.Event, error) {
					return &session.Event{LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("hello", genai.RoleModel)}}, nil
				},
			},
			wantResponses: []model.LLMResponse{{Content: genai.NewContentFromText("hello", genai.RoleModel)}},
		},
		{
			name: "response overwrite with error",
			after: []AfterA2ARequestCallback{
				func(ctx agent.CallbackContext, req *a2a.MessageSendParams, result *session.Event, err error) (*session.Event, error) {
					return nil, fmt.Errorf("failed")
				},
			},
			wantErr: fmt.Errorf("failed"),
		},
		{
			name: "before interceptor short-circuit",
			before: []BeforeA2ARequestCallback{
				func(ctx agent.CallbackContext, req *a2a.MessageSendParams) (*session.Event, error) {
					return nil, fmt.Errorf("failed")
				},
				func(ctx agent.CallbackContext, req *a2a.MessageSendParams) (*session.Event, error) {
					t.Fatalf("not called")
					return nil, nil
				},
			},
			wantErr: fmt.Errorf("failed"),
		},
		{
			name: "after interceptor short-circuit",
			after: []AfterA2ARequestCallback{
				func(ctx agent.CallbackContext, req *a2a.MessageSendParams, result *session.Event, err error) (*session.Event, error) {
					return nil, fmt.Errorf("failed")
				},
				func(ctx agent.CallbackContext, req *a2a.MessageSendParams, result *session.Event, err error) (*session.Event, error) {
					t.Fatalf("not called")
					return nil, nil
				},
			},
			wantErr: fmt.Errorf("failed"),
		},
		{
			name:          "after interceptor for empty session",
			sessionEvents: []*session.Event{},
			after: []AfterA2ARequestCallback{
				func(ctx agent.CallbackContext, req *a2a.MessageSendParams, result *session.Event, err error) (*session.Event, error) {
					if len(req.Message.Parts) != 0 {
						t.Fatalf("got %d parts, expected empty message", len(req.Message.Parts))
					}
					return nil, fmt.Errorf("empty session")
				},
			},
			wantErr: fmt.Errorf("empty session"),
		},
		{
			name: "converter error",
			converter: func(ctx agent.ReadonlyContext, req *a2a.MessageSendParams, event a2a.Event, err error) (*session.Event, error) {
				return nil, fmt.Errorf("failed")
			},
			wantErr: fmt.Errorf("failed"),
		},
		{
			name: "converter custom response",
			converter: func(ctx agent.ReadonlyContext, req *a2a.MessageSendParams, event a2a.Event, err error) (*session.Event, error) {
				return &session.Event{LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("hello", genai.RoleModel)}}, nil
			},
			wantResponses: []model.LLMResponse{{Content: genai.NewContentFromText("hello", genai.RoleModel)}},
		},
		{
			name: "after interceptor invoked with before result",
			before: []BeforeA2ARequestCallback{
				func(ctx agent.CallbackContext, req *a2a.MessageSendParams) (*session.Event, error) {
					return nil, fmt.Errorf("before error")
				},
			},
			after: []AfterA2ARequestCallback{
				func(ctx agent.CallbackContext, req *a2a.MessageSendParams, result *session.Event, err error) (*session.Event, error) {
					return nil, fmt.Errorf("after error")
				},
			},
			wantErr: fmt.Errorf("after error"),
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			executor := &mockA2AExecutor{
				executeFn: func(ctx context.Context, reqCtx *a2asrv.RequestContext, queue eventqueue.Queue) error {
					if tc.events != nil {
						for _, event := range tc.events(reqCtx) {
							if err := queue.Write(ctx, event); err != nil {
								return err
							}
						}
						return nil
					}
					return queue.Write(ctx, a2a.NewMessage(a2a.MessageRoleAgent, a2a.TextPart{Text: "Hi!"}))
				},
			}
			server := startA2AServer(executor)
			card := &a2a.AgentCard{PreferredTransport: a2a.TransportProtocolJSONRPC, URL: server.URL, Capabilities: a2a.AgentCapabilities{Streaming: true}}
			remoteAgent, err := NewA2A(A2AConfig{
				Name:                   "a2a",
				AgentCard:              card,
				BeforeRequestCallbacks: tc.before,
				AfterRequestCallbacks:  tc.after,
				Converter:              tc.converter,
			})
			if err != nil {
				t.Fatalf("remoteagent.NewA2A() error = %v", err)
			}

			sessionEvents := []*session.Event{newUserHello()}
			if tc.sessionEvents != nil {
				sessionEvents = tc.sessionEvents
			}
			ictx := newInvocationContext(t, sessionEvents)
			gotEvents, err := runAndCollect(ictx, remoteAgent)
			if err != nil && tc.wantErr == nil {
				t.Fatalf("agent.Run() error = %v, want nil", err)
			}
			if err == nil && tc.wantErr != nil {
				t.Fatalf("agent.Run() error = nil, want %v", tc.wantErr)
			}
			gotResponses := toLLMResponses(gotEvents)
			if diff := cmp.Diff(tc.wantResponses, gotResponses); diff != "" {
				t.Fatalf("agent.Run() wrong result (+got,-want):\ngot = %+v\nwant = %+v\ndiff = %s", gotResponses, tc.wantResponses, diff)
			}
		})
	}
}

func TestRemoteAgent_RequestPayload(t *testing.T) {
	remoteAgentName, notRemoteAgentName := "a2a", "not-a2a"
	testCases := []struct {
		name          string
		sessionEvents []*session.Event
		wantRequest   *a2a.MessageSendParams
	}{
		{
			name:          "only user message",
			sessionEvents: []*session.Event{newUserHello()},
			wantRequest: &a2a.MessageSendParams{
				Message: &a2a.Message{
					Role:  a2a.MessageRoleUser,
					Parts: []a2a.Part{a2a.TextPart{Text: "hello"}},
				},
			},
		},
		{
			name: "history included",
			sessionEvents: []*session.Event{
				newUserHello(),
				{
					Author: notRemoteAgentName,
					LLMResponse: model.LLMResponse{
						Content: genai.NewContentFromText("hi", genai.RoleModel),
					},
				},
				{
					Author: "user",
					LLMResponse: model.LLMResponse{
						Content: genai.NewContentFromText("how are you?", genai.RoleUser),
					},
				},
			},
			wantRequest: &a2a.MessageSendParams{
				Message: &a2a.Message{
					Role: a2a.MessageRoleUser,
					Parts: []a2a.Part{
						a2a.TextPart{Text: "hello"},
						a2a.TextPart{Text: "For context:"},
						a2a.TextPart{Text: fmt.Sprintf("[%s] said: hi", notRemoteAgentName)},
						a2a.TextPart{Text: "how are you?"},
					},
				},
			},
		},
		{
			name: "history split by remote agent response",
			sessionEvents: []*session.Event{
				{Author: "user", LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("msg1", genai.RoleUser)}},
				{Author: notRemoteAgentName, LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("resp1", genai.RoleModel)}},
				{
					Author: remoteAgentName,
					LLMResponse: model.LLMResponse{
						Content:        genai.NewContentFromText("resp2", genai.RoleModel),
						CustomMetadata: adka2a.ToCustomMetadata("", "ctx-123"),
					},
				},
				// only data from this point should be included, because other parts should already be present
				// in the remote agent's session
				{Author: "user", LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("msg3", genai.RoleUser)}},
				{Author: notRemoteAgentName, LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("resp3", genai.RoleModel)}},
			},
			wantRequest: &a2a.MessageSendParams{
				Message: &a2a.Message{
					Role:      a2a.MessageRoleUser,
					ContextID: "ctx-123",
					Parts: []a2a.Part{
						a2a.TextPart{Text: "msg3"},
						a2a.TextPart{Text: "For context:"},
						a2a.TextPart{Text: fmt.Sprintf("[%s] said: resp3", notRemoteAgentName)},
					},
				},
			},
		},
		{
			name: "function call response",
			sessionEvents: []*session.Event{
				{Author: "user", LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("start", genai.RoleUser)}},
				{
					Author: remoteAgentName,
					LLMResponse: model.LLMResponse{
						Content: genai.NewContentFromParts([]*genai.Part{
							{FunctionCall: &genai.FunctionCall{Name: "fn", ID: "call-1"}},
						}, genai.RoleModel),
						CustomMetadata: adka2a.ToCustomMetadata("task-1", "ctx-1"),
					},
					LongRunningToolIDs: []string{"call-1"},
				},
				{
					Author: remoteAgentName,
					LLMResponse: model.LLMResponse{
						Content: genai.NewContentFromParts([]*genai.Part{
							{FunctionResponse: &genai.FunctionResponse{Name: "fn", ID: "call-1", Response: map[string]any{"status": "pending"}}},
							genai.NewPartFromText("I'll need to wait for an approval first"),
						}, genai.RoleModel),
					},
				},
				{
					Author: "user",
					LLMResponse: model.LLMResponse{
						Content: genai.NewContentFromParts([]*genai.Part{
							genai.NewPartFromText("lgtm:"),
							{FunctionResponse: &genai.FunctionResponse{Name: "fn", ID: "call-1", Response: map[string]any{"status": "approved"}}},
						}, genai.RoleUser),
					},
				},
			},
			wantRequest: &a2a.MessageSendParams{
				Message: &a2a.Message{
					Role:      a2a.MessageRoleUser,
					TaskID:    "task-1",
					ContextID: "ctx-1",
					Parts: []a2a.Part{
						a2a.TextPart{Text: "lgtm:"},
						a2a.DataPart{
							Data: map[string]any{
								"id":       "call-1",
								"name":     "fn",
								"response": map[string]any{"status": "approved"},
							},
							Metadata: map[string]any{"adk_type": "function_response"},
						},
					},
				},
			},
		},
	}

	server := startA2AServer(newA2AEventReplay(t, []a2a.Event{}))
	card := &a2a.AgentCard{PreferredTransport: a2a.TransportProtocolJSONRPC, URL: server.URL, Capabilities: a2a.AgentCapabilities{Streaming: true}}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			errRejected := errors.New("rejected")
			var gotRequest *a2a.MessageSendParams
			remoteAgent, err := NewA2A(A2AConfig{
				Name:      remoteAgentName,
				AgentCard: card,
				BeforeRequestCallbacks: []BeforeA2ARequestCallback{
					func(ctx agent.CallbackContext, req *a2a.MessageSendParams) (*session.Event, error) {
						gotRequest = req
						return nil, errRejected
					},
				},
			})
			if err != nil {
				t.Fatalf("remoteagent.NewA2A() error = %v", err)
			}

			ictx := newInvocationContext(t, tc.sessionEvents)
			if _, err := runAndCollect(ictx, remoteAgent); !errors.Is(err, errRejected) {
				t.Fatalf("agent.Run() error = %v, want %v", err, errRejected)
			}

			ignoreFields := []cmp.Option{
				cmpopts.IgnoreFields(a2a.Message{}, "ID"),
			}
			if diff := cmp.Diff(tc.wantRequest, gotRequest, ignoreFields...); diff != "" {
				t.Fatalf("agent.Run() sent unexpected request (+got,-want):\ngot = %+v\nwant = %+v\ndiff = %s", gotRequest, tc.wantRequest, diff)
			}
		})
	}
}

func TestRemoteAgent_EmptyResultForEmptySession(t *testing.T) {
	ictx := newInvocationContext(t, []*session.Event{})

	executor := newA2AEventReplay(t, []a2a.Event{
		a2a.NewMessage(a2a.MessageRoleAgent, a2a.TextPart{Text: "will not be invoked, because input is empty"}),
	})

	agentName := "a2a agent"
	remoteAgent := newA2ARemoteAgent(t, agentName, startA2AServer(executor))

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

	executor := newA2AEventReplay(t, remoteEvents)
	handler := a2asrv.NewHandler(executor)

	var cardServer *httptest.Server
	mux := http.NewServeMux()
	mux.Handle("/invoke", a2asrv.NewJSONRPCHandler(handler))
	mux.HandleFunc("/.well-known/agent-card.json", func(w http.ResponseWriter, r *http.Request) {
		url := fmt.Sprintf("%s/invoke", cardServer.URL)
		card := &a2a.AgentCard{PreferredTransport: a2a.TransportProtocolJSONRPC, URL: url, Capabilities: a2a.AgentCapabilities{Streaming: true}}
		if err := json.NewEncoder(w).Encode(card); err != nil {
			t.Errorf("json.Encode(agentCard) error = %v", err)
		}
	})
	cardServer = httptest.NewServer(mux)

	remoteAgent, err := NewA2A(A2AConfig{Name: "a2a", AgentCardSource: cardServer.URL})
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
	remoteEvents := []a2a.Event{a2a.NewMessage(a2a.MessageRoleAgent, a2a.TextPart{Text: "will not be invoked!"})}
	executor := newA2AEventReplay(t, remoteEvents)
	server := startA2AServer(executor)

	remoteAgent, err := NewA2A(A2AConfig{
		Name:          "a2a",
		ClientFactory: a2aclient.NewFactory(a2aclient.WithDefaultsDisabled()),
		AgentCard: &a2a.AgentCard{
			PreferredTransport: a2a.TransportProtocolJSONRPC,
			URL:                server.URL,
		},
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
	executorErr := fmt.Errorf("mockExecutor failed")
	executor := &mockA2AExecutor{
		executeFn: func(ctx context.Context, reqCtx *a2asrv.RequestContext, q eventqueue.Queue) error {
			return executorErr
		},
	}

	remoteAgent := newA2ARemoteAgent(t, "a2a agent", startA2AServer(executor))

	ictx := newInvocationContext(t, []*session.Event{newUserHello()})
	gotEvents, err := runAndCollect(ictx, remoteAgent)
	if err != nil {
		t.Fatalf("agent.Run() error = %v", err)
	}

	if len(gotEvents) != 1 {
		t.Fatalf("len(events) = %d, want 1", len(gotEvents))
	}
	if gotEvents[0].ErrorMessage == "" {
		t.Fatal("event.ErrorMessage empty, want non-empty")
	}
}
