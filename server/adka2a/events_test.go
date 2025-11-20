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
	"testing"

	"github.com/a2aproject/a2a-go/a2a"
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
				Parts:     []a2a.Part{a2a.TextPart{Text: "foo"}},
				TaskID:    taskID,
				ContextID: contextID,
			},
			want: &session.Event{
				LLMResponse: model.LLMResponse{
					Content: genai.NewContentFromParts([]*genai.Part{{Text: "foo"}}, genai.RoleModel),
					CustomMetadata: map[string]any{
						customMetaTaskIDKey:    string(taskID),
						customMetaContextIDKey: contextID,
					},
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
			},
			want: &session.Event{
				LLMResponse: model.LLMResponse{
					CustomMetadata: map[string]any{
						customMetaTaskIDKey:    string(taskID),
						customMetaContextIDKey: contextID,
					},
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
						Parts: []a2a.Part{
							a2a.DataPart{
								Data:     map[string]any{"id": "get_weather", "args": map[string]any{"city": "Warsaw"}, "name": "GetWeather"},
								Metadata: map[string]any{a2aDataPartMetaTypeKey: a2aDataPartTypeFunctionCall, a2aDataPartMetaLongRunningKey: true},
							},
						},
					},
					{ID: a2a.NewArtifactID(), Parts: a2a.ContentParts{a2a.TextPart{Text: "foo"}}},
					{ID: a2a.NewArtifactID(), Parts: a2a.ContentParts{a2a.TextPart{Text: "bar"}}},
				},
				Status: a2a.TaskStatus{
					State:   a2a.TaskStateCompleted,
					Message: a2a.NewMessage(a2a.MessageRoleAgent, a2a.TextPart{Text: "done"}),
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
					CustomMetadata: map[string]any{
						customMetaTaskIDKey:    string(taskID),
						customMetaContextIDKey: contextID,
					},
				},
				Author: agentName,
				Branch: branch,
			},
		},
		{
			name: "task with no parts",
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
				},
				Author: agentName,
				Branch: branch,
			},
		},
		{
			name: "task in input required",
			input: &a2a.Task{
				ID:        taskID,
				ContextID: contextID,
				Artifacts: []*a2a.Artifact{
					{
						ID: a2a.NewArtifactID(),
						Parts: []a2a.Part{
							a2a.DataPart{
								Data:     map[string]any{"id": "get_weather", "args": map[string]any{"city": "Warsaw"}, "name": "GetWeather"},
								Metadata: map[string]any{a2aDataPartMetaTypeKey: a2aDataPartTypeFunctionCall, a2aDataPartMetaLongRunningKey: true},
							},
						},
					},
				},
				Status: a2a.TaskStatus{State: a2a.TaskStateInputRequired},
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
				},
				LongRunningToolIDs: []string{"get_weather"},
				Author:             agentName,
				Branch:             branch,
			},
		},
		{
			name: "artifact update",
			input: &a2a.TaskArtifactUpdateEvent{
				TaskID:    taskID,
				ContextID: contextID,
				Artifact: &a2a.Artifact{
					ID: a2a.NewArtifactID(), Parts: a2a.ContentParts{a2a.TextPart{Text: "foo"}, a2a.TextPart{Text: "bar"}},
				},
			},
			want: &session.Event{
				LLMResponse: model.LLMResponse{
					Content: genai.NewContentFromParts([]*genai.Part{
						{Text: "foo"},
						{Text: "bar"},
					}, genai.RoleModel),
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
			name: "artifact update with no parts is skipped",
			input: &a2a.TaskArtifactUpdateEvent{
				TaskID:    taskID,
				ContextID: contextID,
				Artifact: &a2a.Artifact{
					ID:    a2a.NewArtifactID(),
					Parts: []a2a.Part{},
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
					Parts: []a2a.Part{
						a2a.DataPart{
							Data:     map[string]any{"id": "get_weather", "args": map[string]any{"city": "Warsaw"}, "name": "GetWeather"},
							Metadata: map[string]any{a2aDataPartMetaTypeKey: a2aDataPartTypeFunctionCall, a2aDataPartMetaLongRunningKey: true},
						},
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
				Final:     true,
				Status: a2a.TaskStatus{
					Message: &a2a.Message{
						Parts: []a2a.Part{a2a.TextPart{Text: "foo"}},
					},
				},
			},
			want: &session.Event{
				LLMResponse: model.LLMResponse{
					Content: genai.NewContentFromParts([]*genai.Part{{Text: "foo"}}, genai.RoleModel),
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
			name:  "final task status update without message",
			input: &a2a.TaskStatusUpdateEvent{TaskID: taskID, ContextID: contextID, Final: true},
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
					State: a2a.TaskStateCompleted,
					Message: &a2a.Message{
						Parts: []a2a.Part{a2a.TextPart{Text: "foo"}},
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
