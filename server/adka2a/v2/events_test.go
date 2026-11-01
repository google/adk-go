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
	"testing"
	"time"

	"github.com/a2aproject/a2a-go/v2/a2a"
	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/platform"
	"google.golang.org/adk/v2/session"
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
				Author: agentName,
				Branch: branch,
				// TransferToAgent is peer-supplied metadata; this conversion
				// layer never restores it -- see
				// TestToSessionEventWithParts_TransferToAgentAlwaysRedacted and
				// TransferToAgentFromMeta, which a caller uses to restore it
				// explicitly after making its own trust decision about the peer.
				Actions: session.EventActions{Escalate: true},
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
					Content:      genai.NewContentFromParts([]*genai.Part{genai.NewPartFromText("failed with an error")}, genai.RoleModel),
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
			name: "failed task with single-part text status",
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
					Content:        genai.NewContentFromParts([]*genai.Part{genai.NewPartFromText("failed with an error")}, genai.RoleModel),
					ErrorMessage:   "failed with an error",
					CustomMetadata: map[string]any{customMetaTaskIDKey: string(taskID), customMetaContextIDKey: contextID},
					TurnComplete:   true,
				},
				Author: agentName,
				Branch: branch,
			},
		},
		{
			name: "completed task with single-part text status",
			input: &a2a.Task{
				ID:        taskID,
				ContextID: contextID,
				Status: a2a.TaskStatus{
					State:   a2a.TaskStateCompleted,
					Message: &a2a.Message{Parts: a2a.ContentParts{a2a.NewTextPart("failed with an error")}},
				},
			},
			want: &session.Event{
				LLMResponse: model.LLMResponse{
					Content:        genai.NewContentFromParts([]*genai.Part{genai.NewPartFromText("failed with an error")}, genai.RoleModel),
					CustomMetadata: map[string]any{customMetaTaskIDKey: string(taskID), customMetaContextIDKey: contextID},
					TurnComplete:   true,
				},
				Author: agentName,
				Branch: branch,
			},
		},
		{
			name: "failed task with a multi-part status",
			input: &a2a.Task{
				ID:        taskID,
				ContextID: contextID,
				Status: a2a.TaskStatus{
					State: a2a.TaskStateFailed,
					Message: a2a.NewMessage(a2a.MessageRoleAgent,
						a2a.NewTextPart("failed with an error"),
						a2a.NewTextPart("extra details"),
					),
				},
			},
			want: &session.Event{
				LLMResponse: model.LLMResponse{
					ErrorMessage: "failed with an error",
					Content: genai.NewContentFromParts([]*genai.Part{
						genai.NewPartFromText("failed with an error"),
						genai.NewPartFromText("extra details"),
					}, genai.RoleModel),
					CustomMetadata: map[string]any{customMetaTaskIDKey: string(taskID), customMetaContextIDKey: contextID},
					TurnComplete:   true,
				},
				Author: agentName,
				Branch: branch,
			},
		},
		{
			name: "failed task status with a multi-part status",
			input: &a2a.TaskStatusUpdateEvent{
				TaskID:    taskID,
				ContextID: contextID,
				Status: a2a.TaskStatus{
					State: a2a.TaskStateFailed,
					Message: a2a.NewMessage(a2a.MessageRoleAgent,
						a2a.NewTextPart("failed with an error"),
						a2a.NewDataPart(map[string]any{"structured": true}),
					),
				},
			},
			want: &session.Event{
				LLMResponse: model.LLMResponse{
					ErrorMessage: "failed with an error",
					Content: genai.NewContentFromParts([]*genai.Part{
						genai.NewPartFromText("failed with an error"),
						{InlineData: &genai.Blob{
							Data:     []byte(`<a2a_datapart_json>{"structured":true}</a2a_datapart_json>`),
							MIMEType: "text/plain",
						}},
					}, genai.RoleModel),
					CustomMetadata: map[string]any{customMetaTaskIDKey: string(taskID), customMetaContextIDKey: contextID},
					TurnComplete:   true,
				},
				Author: agentName,
				Branch: branch,
			},
		},
		{
			name: "failed task status metadataIsErrMessageKey not in content",
			input: &a2a.TaskStatusUpdateEvent{
				TaskID:    taskID,
				ContextID: contextID,
				Status: a2a.TaskStatus{
					State: a2a.TaskStateFailed,
					Message: a2a.NewMessage(a2a.MessageRoleAgent,
						&a2a.Part{Content: a2a.Text("failed with an error"), Metadata: map[string]any{metadataIsErrMessageKey: true}},
						a2a.NewTextPart("extra details"),
					),
				},
			},
			want: &session.Event{
				LLMResponse: model.LLMResponse{
					ErrorMessage:   "failed with an error",
					Content:        genai.NewContentFromParts([]*genai.Part{genai.NewPartFromText("extra details")}, genai.RoleModel),
					CustomMetadata: map[string]any{customMetaTaskIDKey: string(taskID), customMetaContextIDKey: contextID},
					TurnComplete:   true,
				},
				Author: agentName,
				Branch: branch,
			},
		},
		{
			name: "input-required task status with err message and long-running tool",
			input: &a2a.TaskStatusUpdateEvent{
				TaskID:    taskID,
				ContextID: contextID,
				Status: a2a.TaskStatus{
					State: a2a.TaskStateFailed,
					Message: a2a.NewMessage(a2a.MessageRoleAgent,
						&a2a.Part{Content: a2a.Text("requesting input"), Metadata: map[string]any{metadataIsErrMessageKey: true}},
						func() *a2a.Part {
							p := a2a.NewDataPart(map[string]any{"id": "tool_lr", "name": "LongRunning", "args": map[string]any{}})
							p.Metadata = map[string]any{a2aDataPartMetaTypeKey: a2aDataPartTypeFunctionCall, a2aDataPartMetaLongRunningKey: true}
							return p
						}(),
					),
				},
			},
			want: &session.Event{
				LLMResponse: model.LLMResponse{
					Content: genai.NewContentFromParts([]*genai.Part{
						{FunctionCall: &genai.FunctionCall{ID: "tool_lr", Name: "LongRunning", Args: map[string]any{}}},
					}, genai.RoleModel),
					ErrorMessage:   "requesting input",
					CustomMetadata: map[string]any{customMetaTaskIDKey: string(taskID), customMetaContextIDKey: contextID},
					TurnComplete:   true,
				},
				LongRunningToolIDs: []string{"tool_lr"},
				Author:             agentName,
				Branch:             branch,
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

// TestToSessionEventWithParts_TransferToAgentAlwaysRedacted verifies that a
// remote A2A peer's transfer_to_agent metadata is never restored by this
// conversion layer, regardless of caller. A caller that has made its own
// trust decision about the remote peer restores it explicitly afterwards,
// using TransferToAgentFromMeta -- see
// remoteagent.A2AConfig.AllowTransferToAgent.
func TestToSessionEventWithParts_TransferToAgentAlwaysRedacted(t *testing.T) {
	branch, agentName := "main", "a2a agent"
	a2aAgent, err := agent.New(agent.Config{Name: agentName})
	if err != nil {
		t.Fatalf("failed to create an agent: %v", err)
	}
	ictx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{Branch: branch, Agent: a2aAgent})

	input := &a2a.Message{
		Parts: a2a.ContentParts{a2a.NewTextPart("foo")},
		Metadata: map[string]any{
			metadataTransferToAgentKey: "a-2",
			metadataEscalateKey:        true,
		},
	}

	got, err := ToSessionEventWithParts(ictx, input, nil)
	if err != nil {
		t.Fatalf("ToSessionEventWithParts() error = %v", err)
	}
	want := session.EventActions{Escalate: true}
	if diff := cmp.Diff(want, got.Actions); diff != "" {
		t.Errorf("Actions wrong result (+got,-want)\ndiff = %s", diff)
	}
}

func TestTransferToAgentFromMeta(t *testing.T) {
	if got, ok := TransferToAgentFromMeta(nil); ok || got != "" {
		t.Errorf("TransferToAgentFromMeta(nil) = (%q, %v), want (\"\", false)", got, ok)
	}
	if got, ok := TransferToAgentFromMeta(map[string]any{}); ok || got != "" {
		t.Errorf("TransferToAgentFromMeta(empty) = (%q, %v), want (\"\", false)", got, ok)
	}
	meta := map[string]any{metadataTransferToAgentKey: "a-2"}
	if got, ok := TransferToAgentFromMeta(meta); !ok || got != "a-2" {
		t.Errorf("TransferToAgentFromMeta(meta) = (%q, %v), want (\"a-2\", true)", got, ok)
	}
}

func TestToSessionEventWithPartsEquivalence(t *testing.T) {
	taskID, contextID := a2a.TaskID("task-eq-1"), "ctx-eq-1"
	a2aAgent, err := agent.New(agent.Config{Name: "eq_agent"})
	if err != nil {
		t.Fatalf("agent.New failed: %v", err)
	}

	inputs := []struct {
		name  string
		event a2a.Event
	}{
		{
			name: "message",
			event: &a2a.Message{
				TaskID:    taskID,
				ContextID: contextID,
				Role:      a2a.MessageRoleAgent,
				Parts:     a2a.ContentParts{a2a.NewTextPart("hello")},
				Metadata:  map[string]any{metadataEscalateKey: true},
			},
		},
		{
			name: "artifact update partial",
			event: &a2a.TaskArtifactUpdateEvent{
				TaskID:    taskID,
				ContextID: contextID,
				Artifact:  &a2a.Artifact{ID: "art-1", Parts: a2a.ContentParts{a2a.NewTextPart("delta")}},
				Append:    true,
			},
		},
		{
			name: "task completed",
			event: &a2a.Task{
				ID:        taskID,
				ContextID: contextID,
				Artifacts: []*a2a.Artifact{{ID: "art-1", Parts: a2a.ContentParts{a2a.NewTextPart("answer")}}},
				Status:    a2a.TaskStatus{State: a2a.TaskStateCompleted},
			},
		},
		{
			name: "working status update with message",
			event: &a2a.TaskStatusUpdateEvent{
				TaskID:    taskID,
				ContextID: contextID,
				Status: a2a.TaskStatus{
					State:   a2a.TaskStateWorking,
					Message: a2a.NewMessage(a2a.MessageRoleAgent, a2a.NewTextPart("thinking")),
				},
			},
		},
		{
			name: "final status update failed",
			event: &a2a.TaskStatusUpdateEvent{
				TaskID:    taskID,
				ContextID: contextID,
				Status: a2a.TaskStatus{
					State:   a2a.TaskStateFailed,
					Message: a2a.NewMessage(a2a.MessageRoleAgent, a2a.NewTextPart("boom")),
				},
			},
		},
	}

	for _, tc := range inputs {
		t.Run(tc.name, func(t *testing.T) {
			gotCtx, gotOrder := newRecordingInvocationContext(t, a2aAgent)
			wantCtx, wantOrder := newRecordingInvocationContext(t, a2aAgent)

			got, gotErr := ToSessionEventWithParts(gotCtx, tc.event, nil)
			want, wantErr := referenceToSessionEventWithParts(wantCtx, tc.event, nil)
			if (gotErr != nil) != (wantErr != nil) {
				t.Fatalf("error = %v, want %v", gotErr, wantErr)
			}
			if diff := cmp.Diff(want, got); diff != "" {
				t.Errorf("ToSessionEventWithParts mismatch (-want +got):\n%s", diff)
			}
			if diff := cmp.Diff(*wantOrder, *gotOrder); diff != "" {
				t.Errorf("platform call order mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func newRecordingInvocationContext(t *testing.T, agnt agent.Agent) (agent.InvocationContext, *[]string) {
	t.Helper()
	var calls []string
	seq := 0
	base := platform.WithUUIDProvider(t.Context(), func() string {
		seq++
		calls = append(calls, fmt.Sprintf("uuid:%d", seq))
		return fmt.Sprintf("uuid-%d", seq)
	})
	base = platform.WithTimeProvider(base, func() time.Time {
		seq++
		calls = append(calls, fmt.Sprintf("time:%d", seq))
		return time.Unix(1700000000+int64(seq), 0).UTC()
	})
	return icontext.NewInvocationContext(base, icontext.InvocationContextParams{
		Branch: "main",
		Agent:  agnt,
	}), &calls
}

func referenceToSessionEventWithParts(ctx agent.InvocationContext, event a2a.Event, partConverter A2APartConverter) (*session.Event, error) {
	if partConverter == nil {
		partConverter = func(_ context.Context, _ a2a.Event, part *a2a.Part) (*genai.Part, error) {
			return ToGenAIPart(part)
		}
	}
	switch v := event.(type) {
	case *a2a.Task:
		var parts []*genai.Part
		var lrtIDs []string
		for _, artifact := range v.Artifacts {
			allParts, err := convertParts(ctx, v, artifact.Parts, partConverter)
			if err != nil {
				return nil, err
			}
			parts = append(parts, filterNilParts(allParts)...)
			lrtIDs = append(lrtIDs, getLongRunningToolIDs(artifact.Parts, allParts)...)
		}
		ev := NewRemoteAgentEvent(ctx)
		statusMsg, err := convertStatusMessage(ctx, v, v.Status, partConverter)
		if err != nil {
			return nil, err
		}
		parts = append(parts, statusMsg.parts...)
		lrtIDs = append(lrtIDs, statusMsg.longRunningToolIDs...)
		if v.Status.State == a2a.TaskStateFailed {
			if statusMsg.errorMessage != "" {
				ev.ErrorMessage = statusMsg.errorMessage
			} else {
				ev.ErrorMessage = "a2a task failed"
			}
		}
		isTerminal := v.Status.State.Terminal() || v.Status.State == a2a.TaskStateInputRequired
		if len(parts) == 0 && !isTerminal {
			return nil, nil
		}
		if len(parts) > 0 {
			ev.Content = genai.NewContentFromParts(parts, genai.RoleModel)
		}
		if v.Status.State == a2a.TaskStateInputRequired {
			ev.LongRunningToolIDs = lrtIDs
		}
		if err := processA2AMeta(v, ev); err != nil {
			return nil, err
		}
		ev.TurnComplete = isTerminal
		return ev, nil

	case *a2a.Message:
		var parts []*genai.Part
		for _, part := range v.Parts {
			gp, err := partConverter(ctx, v, part)
			if err != nil {
				return nil, err
			}
			if gp != nil {
				parts = append(parts, gp)
			}
		}
		ev := NewRemoteAgentEvent(ctx)
		if len(parts) > 0 {
			ev.Content = genai.NewContentFromParts(parts, toGenAIRole(v.Role))
		}
		if err := processA2AMeta(v, ev); err != nil {
			return nil, err
		}
		ev.TurnComplete = true
		return ev, nil

	case *a2a.TaskArtifactUpdateEvent:
		allParts, err := convertParts(ctx, v, v.Artifact.Parts, partConverter)
		if err != nil {
			return nil, err
		}
		ev := NewRemoteAgentEvent(ctx)
		ev.Content = genai.NewContentFromParts(filterNilParts(allParts), genai.RoleModel)
		ev.LongRunningToolIDs = getLongRunningToolIDs(v.Artifact.Parts, allParts)
		if err := processA2AMeta(v, ev); err != nil {
			return nil, err
		}
		if partial, ok := v.Metadata[metadataPartialKey].(bool); ok {
			ev.Partial = partial
		} else {
			ev.Partial = v.Append || !v.LastChunk
		}
		return ev, nil

	case *a2a.TaskStatusUpdateEvent:
		if v.Status.State.Terminal() || v.Status.State == a2a.TaskStateInputRequired {
			ev := NewRemoteAgentEvent(ctx)
			statusMsg, err := convertStatusMessage(ctx, v, v.Status, partConverter)
			if err != nil {
				return nil, err
			}
			if len(statusMsg.parts) > 0 {
				ev.Content = genai.NewContentFromParts(statusMsg.parts, genai.RoleModel)
			}
			ev.LongRunningToolIDs = statusMsg.longRunningToolIDs
			if v.Status.State == a2a.TaskStateFailed {
				if statusMsg.errorMessage != "" {
					ev.ErrorMessage = statusMsg.errorMessage
				} else {
					ev.ErrorMessage = "a2a task failed"
				}
			}
			if err := processA2AMeta(v, ev); err != nil {
				return nil, err
			}
			ev.TurnComplete = true
			return ev, nil
		}
		ev, err := referenceToSessionEventWithParts(ctx, v.Status.Message, partConverter)
		if err != nil || ev == nil {
			return ev, err
		}
		ev.TurnComplete = false
		if err := processA2AMeta(v, ev); err != nil {
			return nil, err
		}
		for _, p := range ev.Content.Parts {
			if p.Text != "" {
				p.Thought = true
			}
		}
		ev.Partial = true
		return ev, nil
	default:
		return nil, fmt.Errorf("unknown event type: %T", v)
	}
}
