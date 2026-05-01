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
	"testing"

	"github.com/a2aproject/a2a-go/v2/a2a"
	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	icontext "google.golang.org/adk/internal/context"
	"google.golang.org/adk/model"
	"google.golang.org/adk/session"
)

func TestToSessionEvent(t *testing.T) {
	t.Parallel()
	taskID, contextID, branch, agentName := a2a.NewTaskID(), a2a.NewContextID(), "main", "a2a agent"
	a2aAgent, err := agent.New(agent.Config{Name: agentName})
	if err != nil {
		t.Fatalf("failed to create an agent: %v", err)
	}

	testCases := []struct {
		name                   string
		input                  a2a.Event
		want                   *session.Event
		longRunningFunctionIDs []string
	}{
		{
			name: "message",
			input: &a2a.Message{
				Parts:     a2a.ContentParts{a2a.NewTextPart("foo")},
				TaskID:    taskID,
				ContextID: contextID,
				Metadata: map[string]any{
					metadataGroundingKey:       map[string]any{"sourceFlaggingUris": []any{map[string]any{"sourceId": "id1"}}},
					metadataUsageKey:           map[string]any{"candidatesTokenCount": float64(12), "thoughtsTokenCount": float64(42)},
					metadataCustomMetaKey:      map[string]any{"nested": map[string]any{"key": "value"}},
					metadataTransferToAgentKey: "a-2",
					metadataEscalateKey:        true,
				},
			},
			want: &session.Event{
				LLMResponse: model.LLMResponse{
					Content:           genai.NewContentFromParts([]*genai.Part{{Text: "foo"}}, genai.RoleModel),
					UsageMetadata:     &genai.GenerateContentResponseUsageMetadata{CandidatesTokenCount: 12, ThoughtsTokenCount: 42},
					GroundingMetadata: &genai.GroundingMetadata{SourceFlaggingUris: []*genai.GroundingMetadataSourceFlaggingURI{{SourceID: "id1"}}},
					CustomMetadata: map[string]any{
						"nested":               map[string]any{"key": "value"},
						customMetaTaskIDKey:    string(taskID),
						customMetaContextIDKey: contextID,
					},
					TurnComplete: true,
				},
				Author:  agentName,
				Branch:  branch,
				Actions: session.EventActions{Escalate: true, TransferToAgent: "a-2"},
			},
		},
		{
			name: "nil values",
			input: &a2a.Message{
				Parts:     a2a.ContentParts{a2a.NewTextPart("foo")},
				TaskID:    taskID,
				ContextID: contextID,
				Metadata: map[string]any{
					metadataGroundingKey:  nil,
					metadataUsageKey:      nil,
					metadataCustomMetaKey: nil,
				},
			},
			want: &session.Event{
				LLMResponse: model.LLMResponse{
					Content:        genai.NewContentFromParts([]*genai.Part{{Text: "foo"}}, genai.RoleModel),
					CustomMetadata: map[string]any{customMetaTaskIDKey: string(taskID), customMetaContextIDKey: contextID},
					TurnComplete:   true,
				},
				Author: agentName,
				Branch: branch,
			},
		},
		{
			name: "message with no parts",
			input: &a2a.Message{
				TaskID:    taskID,
				ContextID: contextID,
				Parts:     a2a.ContentParts{},
			},
			want: &session.Event{
				LLMResponse: model.LLMResponse{
					CustomMetadata: map[string]any{
						customMetaTaskIDKey:    string(taskID),
						customMetaContextIDKey: contextID,
					},
					TurnComplete: true,
				},
				Author: agentName,
				Branch: branch,
			},
		},
		{
			name: "task",
			input: &a2a.Task{
				ID:        taskID,
				ContextID: contextID,
				Artifacts: []*a2a.Artifact{
					{ // long running key is ignored for non-input-required states
						ID: a2a.NewArtifactID(),
						Parts: a2a.ContentParts{
							func() *a2a.Part {
								p := a2a.NewDataPart(map[string]any{"id": "get_weather", "args": map[string]any{"city": "Warsaw"}, "name": "GetWeather"})
								p.Metadata = map[string]any{a2aDataPartMetaTypeKey: a2aDataPartTypeFunctionCall, a2aDataPartMetaLongRunningKey: true}
								return p
							}(),
						},
					},
					{ID: a2a.NewArtifactID(), Parts: a2a.ContentParts{a2a.NewTextPart("foo")}},
					{ID: a2a.NewArtifactID(), Parts: a2a.ContentParts{a2a.NewTextPart("bar")}},
				},
				Status: a2a.TaskStatus{
					State:   a2a.TaskStateCompleted,
					Message: a2a.NewMessage(a2a.MessageRoleAgent, a2a.NewTextPart("done")),
				},
				Metadata: map[string]any{
					metadataGroundingKey:  map[string]any{"sourceFlaggingUris": []any{map[string]any{"sourceId": "id1"}}},
					metadataUsageKey:      map[string]any{"candidatesTokenCount": float64(12), "thoughtsTokenCount": float64(42)},
					metadataCustomMetaKey: map[string]any{"nested": map[string]any{"key": "value"}},
					metadataEscalateKey:   true,
				},
			},
			want: &session.Event{
				LLMResponse: model.LLMResponse{
					Content: genai.NewContentFromParts([]*genai.Part{
						{
							FunctionCall: &genai.FunctionCall{
								ID:   "get_weather",
								Args: map[string]any{"city": "Warsaw"},
								Name: "GetWeather",
							},
						},
						{Text: "foo"},
						{Text: "bar"},
						{Text: "done"},
					}, genai.RoleModel),
					UsageMetadata:     &genai.GenerateContentResponseUsageMetadata{CandidatesTokenCount: 12, ThoughtsTokenCount: 42},
					GroundingMetadata: &genai.GroundingMetadata{SourceFlaggingUris: []*genai.GroundingMetadataSourceFlaggingURI{{SourceID: "id1"}}},
					CustomMetadata: map[string]any{
						"nested":               map[string]any{"key": "value"},
						customMetaTaskIDKey:    string(taskID),
						customMetaContextIDKey: contextID,
					},
					TurnComplete: true,
				},
				Author:  agentName,
				Branch:  branch,
				Actions: session.EventActions{Escalate: true},
			},
		},
		{
			name: "terminal task with no parts",
			input: &a2a.Task{
				ID:        taskID,
				ContextID: contextID,
				Status:    a2a.TaskStatus{State: a2a.TaskStateCompleted},
			},
			want: &session.Event{
				LLMResponse: model.LLMResponse{
					CustomMetadata: map[string]any{
						customMetaTaskIDKey:    string(taskID),
						customMetaContextIDKey: contextID,
					},
					TurnComplete: true,
				},
				Author: agentName,
				Branch: branch,
			},
		},
		{
			name: "non-terminal task with no parts",
			input: &a2a.Task{
				ID:        taskID,
				ContextID: contextID,
				Status:    a2a.TaskStatus{State: a2a.TaskStateSubmitted},
			},
			want: nil,
		},
		{
			name: "task in input required",
			input: &a2a.Task{
				ID:        taskID,
				ContextID: contextID,
				Artifacts: []*a2a.Artifact{
					{
						ID: a2a.NewArtifactID(),
						Parts: a2a.ContentParts{
							func() *a2a.Part {
								p := a2a.NewDataPart(map[string]any{"id": "get_weather", "args": map[string]any{"city": "Warsaw"}, "name": "GetWeather"})
								p.Metadata = map[string]any{a2aDataPartMetaTypeKey: a2aDataPartTypeFunctionCall, a2aDataPartMetaLongRunningKey: true}
								return p
							}(),
						},
					},
				},
				Status:   a2a.TaskStatus{State: a2a.TaskStateInputRequired},
				Metadata: map[string]any{metadataEscalateKey: true},
			},
			want: &session.Event{
				LLMResponse: model.LLMResponse{
					Content: genai.NewContentFromParts([]*genai.Part{
						{
							FunctionCall: &genai.FunctionCall{
								ID:   "get_weather",
								Args: map[string]any{"city": "Warsaw"},
								Name: "GetWeather",
							},
						},
					}, genai.RoleModel),
					CustomMetadata: map[string]any{
						customMetaTaskIDKey:    string(taskID),
						customMetaContextIDKey: contextID,
					},
					TurnComplete: true,
				},
				LongRunningToolIDs: []string{"get_weather"},
				Author:             agentName,
				Branch:             branch,
				Actions:            session.EventActions{Escalate: true},
			},
		},
		{
			name: "artifact update",
			input: &a2a.TaskArtifactUpdateEvent{
				TaskID:    taskID,
				ContextID: contextID,
				Artifact: &a2a.Artifact{
					ID: a2a.NewArtifactID(), Parts: a2a.ContentParts{a2a.NewTextPart("foo"), a2a.NewTextPart("bar")},
				},
				Metadata: map[string]any{
					metadataGroundingKey:  map[string]any{"sourceFlaggingUris": []any{map[string]any{"sourceId": "id1"}}},
					metadataUsageKey:      map[string]any{"candidatesTokenCount": float64(12), "thoughtsTokenCount": float64(42)},
					metadataCustomMetaKey: map[string]any{"nested": map[string]any{"key": "value"}},
				},
			},
			want: &session.Event{
				LLMResponse: model.LLMResponse{
					Content: genai.NewContentFromParts([]*genai.Part{
						{Text: "foo"},
						{Text: "bar"},
					}, genai.RoleModel),
					GroundingMetadata: &genai.GroundingMetadata{SourceFlaggingUris: []*genai.GroundingMetadataSourceFlaggingURI{{SourceID: "id1"}}},
					UsageMetadata:     &genai.GenerateContentResponseUsageMetadata{CandidatesTokenCount: 12, ThoughtsTokenCount: 42},
					CustomMetadata: map[string]any{
						"nested":               map[string]any{"key": "value"},
						customMetaTaskIDKey:    string(taskID),
						customMetaContextIDKey: contextID,
					},
					Partial: true,
				},
				Author: agentName,
				Branch: branch,
			},
		},
		{
			name: "artifact update with no parts is skipped",
			input: &a2a.TaskArtifactUpdateEvent{
				TaskID:    taskID,
				ContextID: contextID,
				Artifact: &a2a.Artifact{
					ID:    a2a.NewArtifactID(),
					Parts: a2a.ContentParts{},
				},
			},
			want: nil,
		},
		{
			name: "artifact update with long running tool call",
			input: &a2a.TaskArtifactUpdateEvent{
				TaskID:    taskID,
				ContextID: contextID,
				Artifact: &a2a.Artifact{
					ID: a2a.NewArtifactID(),
					Parts: a2a.ContentParts{
						func() *a2a.Part {
							p := a2a.NewDataPart(map[string]any{"id": "get_weather", "args": map[string]any{"city": "Warsaw"}, "name": "GetWeather"})
							p.Metadata = map[string]any{a2aDataPartMetaTypeKey: a2aDataPartTypeFunctionCall, a2aDataPartMetaLongRunningKey: true}
							return p
						}(),
					},
				},
			},
			want: &session.Event{
				LLMResponse: model.LLMResponse{
					Content: genai.NewContentFromParts([]*genai.Part{
						{
							FunctionCall: &genai.FunctionCall{
								ID:   "get_weather",
								Args: map[string]any{"city": "Warsaw"},
								Name: "GetWeather",
							},
						},
					}, genai.RoleModel),
					CustomMetadata: map[string]any{
						customMetaTaskIDKey:    string(taskID),
						customMetaContextIDKey: contextID,
					},
					Partial: true,
				},
				LongRunningToolIDs: []string{"get_weather"},
				Author:             agentName,
				Branch:             branch,
			},
		},
		{
			name: "final task status update with message",
			input: &a2a.TaskStatusUpdateEvent{
				TaskID:    taskID,
				ContextID: contextID,
				Status: a2a.TaskStatus{
					State: a2a.TaskStateCompleted,
					Message: &a2a.Message{
						Parts: a2a.ContentParts{a2a.NewTextPart("foo")},
					},
				},
				Metadata: map[string]any{
					metadataGroundingKey:  map[string]any{"sourceFlaggingUris": []any{map[string]any{"sourceId": "id1"}}},
					metadataUsageKey:      map[string]any{"candidatesTokenCount": float64(12), "thoughtsTokenCount": float64(42)},
					metadataCustomMetaKey: map[string]any{"nested": map[string]any{"key": "value"}},
					metadataEscalateKey:   true,
				},
			},
			want: &session.Event{
				LLMResponse: model.LLMResponse{
					Content: genai.NewContentFromParts([]*genai.Part{{Text: "foo"}}, genai.RoleModel),
					CustomMetadata: map[string]any{
						"nested":               map[string]any{"key": "value"},
						customMetaTaskIDKey:    string(taskID),
						customMetaContextIDKey: contextID,
					},
					TurnComplete:      true,
					GroundingMetadata: &genai.GroundingMetadata{SourceFlaggingUris: []*genai.GroundingMetadataSourceFlaggingURI{{SourceID: "id1"}}},
					UsageMetadata:     &genai.GenerateContentResponseUsageMetadata{CandidatesTokenCount: 12, ThoughtsTokenCount: 42},
				},
				Actions: session.EventActions{Escalate: true},
				Author:  agentName,
				Branch:  branch,
			},
		},
		{
			name:  "final task status update without message",
			input: &a2a.TaskStatusUpdateEvent{TaskID: taskID, ContextID: contextID, Status: a2a.TaskStatus{State: a2a.TaskStateCompleted}},
			want: &session.Event{
				LLMResponse: model.LLMResponse{
					CustomMetadata: map[string]any{
						customMetaTaskIDKey:    string(taskID),
						customMetaContextIDKey: contextID,
					},
					TurnComplete: true,
				},
				Author: agentName,
				Branch: branch,
			},
		},
		{
			name: "non final task status update message is a thought",
			input: &a2a.TaskStatusUpdateEvent{
				TaskID:    taskID,
				ContextID: contextID,
				Status: a2a.TaskStatus{
					State: a2a.TaskStateWorking,
					Message: &a2a.Message{
						Parts: a2a.ContentParts{a2a.NewTextPart("foo")},
					},
				},
			},
			want: &session.Event{
				LLMResponse: model.LLMResponse{
					Content: genai.NewContentFromParts([]*genai.Part{{Text: "foo", Thought: true}}, genai.RoleModel),
					CustomMetadata: map[string]any{
						customMetaTaskIDKey:    string(taskID),
						customMetaContextIDKey: contextID,
					},
					Partial: true,
				},
				Author: agentName,
				Branch: branch,
			},
		},
		{
			name:  "non-final task status update without message is skipped",
			input: &a2a.TaskStatusUpdateEvent{TaskID: taskID, ContextID: contextID},
			want:  nil,
		},
		{
			name: "task status failed with single-part message",
			input: &a2a.TaskStatusUpdateEvent{
				TaskID:    taskID,
				ContextID: contextID,
				Status: a2a.TaskStatus{
					State:   a2a.TaskStateFailed,
					Message: &a2a.Message{Parts: a2a.ContentParts{a2a.NewTextPart("failed with an error")}},
				},
			},
			want: &session.Event{
				LLMResponse: model.LLMResponse{
					ErrorMessage: "failed with an error",
					CustomMetadata: map[string]any{
						customMetaTaskIDKey:    string(taskID),
						customMetaContextIDKey: contextID,
					},
					TurnComplete: true,
				},
				Author: agentName,
				Branch: branch,
			},
		},
		{
			name: "task with multiple artifacts and mixed long-running tools",
			input: &a2a.Task{
				ID:        taskID,
				ContextID: contextID,
				Artifacts: []*a2a.Artifact{
					{
						ID: "artifact-1",
						Parts: []*a2a.Part{
							a2a.NewTextPart("Checking weather..."),
							{
								Content: a2a.Data{Value: map[string]any{"id": "tool_1", "name": "GetWeather", "args": map[string]any{"city": "London"}}},
								Metadata: map[string]any{
									a2aDataPartMetaTypeKey:        a2aDataPartTypeFunctionCall,
									a2aDataPartMetaLongRunningKey: true,
								},
							},
						},
					},
					{
						ID: "artifact-2",
						Parts: []*a2a.Part{
							{
								Content: a2a.Data{Value: map[string]any{"id": "tool_2", "name": "GetNews", "args": map[string]any{"topic": "tech"}}},
								Metadata: map[string]any{
									a2aDataPartMetaTypeKey:        a2aDataPartTypeFunctionCall,
									a2aDataPartMetaLongRunningKey: true,
								},
							},
						},
					},
				},
				Status: a2a.TaskStatus{State: a2a.TaskStateInputRequired},
			},
			want: &session.Event{
				LLMResponse: model.LLMResponse{
					Content: genai.NewContentFromParts([]*genai.Part{
						{Text: "Checking weather..."},
						{FunctionCall: &genai.FunctionCall{ID: "tool_1", Name: "GetWeather", Args: map[string]any{"city": "London"}}},
						{FunctionCall: &genai.FunctionCall{ID: "tool_2", Name: "GetNews", Args: map[string]any{"topic": "tech"}}},
					}, genai.RoleModel),
					CustomMetadata: map[string]any{
						customMetaTaskIDKey:    string(taskID),
						customMetaContextIDKey: contextID,
					},
					TurnComplete: true,
				},
				LongRunningToolIDs: []string{"tool_1", "tool_2"},
				Author:             agentName,
				Branch:             branch,
			},
		},
		{
			name: "task with single-part text status",
			input: &a2a.Task{
				ID:        taskID,
				ContextID: contextID,
				Status: a2a.TaskStatus{
					State:   a2a.TaskStateFailed,
					Message: &a2a.Message{Parts: a2a.ContentParts{a2a.NewTextPart("failed with an error")}},
				},
			},
			want: &session.Event{
				LLMResponse: model.LLMResponse{
					ErrorMessage:   "failed with an error",
					CustomMetadata: map[string]any{customMetaTaskIDKey: string(taskID), customMetaContextIDKey: contextID},
					TurnComplete:   true,
				},
				Author: agentName,
				Branch: branch,
			},
		},
	}

	ignoreFields := []cmp.Option{
		cmpopts.IgnoreFields(session.Event{}, "ID"),
		cmpopts.IgnoreFields(session.Event{}, "Timestamp"),
		cmpopts.IgnoreFields(session.Event{}, "InvocationID"),
		cmpopts.IgnoreFields(session.EventActions{}, "StateDelta"),
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			ictx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{Branch: branch, Agent: a2aAgent})
			got, err := ToSessionEvent(ictx, tc.input)
			if err != nil {
				t.Errorf("ToSessionEvent() error = %v, want nil", err)
			}
			if diff := cmp.Diff(tc.want, got, ignoreFields...); diff != "" {
				t.Errorf("ToSessionEvent() wrong result (+got,-want)\ngot = %v\nwant = %v\ndiff = %s", got, tc.want, diff)
			}
		})
	}
}

func TestToSessionEventWithParts_NilResultFiltered(t *testing.T) {
	taskID, contextID, branch, agentName := a2a.NewTaskID(), a2a.NewContextID(), "main", "a2a agent"
	a2aAgent, err := agent.New(agent.Config{Name: agentName})
	if err != nil {
		t.Fatalf("failed to create an agent: %v", err)
	}

	keepPart := a2a.NewTextPart("KEEP")
	dropPart := a2a.NewTextPart("DROP")

	filterConverter := func(ctx context.Context, ev a2a.Event, p *a2a.Part) (*genai.Part, error) {
		if p.Text() == "DROP" {
			return nil, nil
		}
		return ToGenAIPart(p)
	}

	testCases := []struct {
		name  string
		input a2a.Event
	}{
		{
			name: "task event",
			input: &a2a.Task{
				ID:        taskID,
				ContextID: contextID,
				Artifacts: []*a2a.Artifact{{Parts: []*a2a.Part{keepPart, dropPart}}},
				Status: a2a.TaskStatus{
					State:   a2a.TaskStateCompleted,
					Message: &a2a.Message{Parts: []*a2a.Part{keepPart, dropPart}},
				},
			},
		},
		{
			name: "message event",
			input: &a2a.Message{
				Parts: []*a2a.Part{keepPart, dropPart},
			},
		},
		{
			name: "artifact update event",
			input: &a2a.TaskArtifactUpdateEvent{
				Artifact: &a2a.Artifact{Parts: []*a2a.Part{keepPart, dropPart}},
			},
		},
		{
			name: "status update event",
			input: &a2a.TaskStatusUpdateEvent{
				TaskID:    taskID,
				ContextID: contextID,
				Status: a2a.TaskStatus{
					State:   a2a.TaskStateCompleted,
					Message: &a2a.Message{Parts: []*a2a.Part{keepPart, dropPart}},
				},
			},
		},
	}

	ictx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{Branch: branch, Agent: a2aAgent})

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := ToSessionEventWithParts(ictx, tc.input, filterConverter)
			if err != nil {
				t.Fatalf("ToSessionEventWithParts() error = %v", err)
			}
			if got == nil {
				t.Fatal("got event is nil, expected valid event with filtered parts")
			}

			parts := got.LLMResponse.Content.Parts
			for _, p := range parts {
				if p == nil {
					t.Fatalf("got nil part, want it filtered out.")
				}
				if p.Text != "KEEP" {
					t.Errorf("got %s, want 'KEEP'", p.Text)
				}
			}
		})
	}
}
