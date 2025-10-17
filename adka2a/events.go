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
	"fmt"

	"github.com/a2aproject/a2a-go/a2a"
	"google.golang.org/adk/agent"
	"google.golang.org/adk/model"
	"google.golang.org/adk/session"
	"google.golang.org/genai"
)

const CustomMetaTaskIDKey = "a2a:task_id"
const CustomMetaContextIDKey = "a2a:context_id"

func NewRemoteAgentEvent(ctx agent.InvocationContext) *session.Event {
	event := session.NewEvent(ctx.InvocationID())
	event.Author = ctx.Agent().Name()
	event.Branch = ctx.Branch()
	event.LLMResponse = &model.LLMResponse{}
	return event
}

func EventToMessage(ctx agent.InvocationContext, event *session.Event) (*a2a.Message, error) {
	if ctx == nil {
		return nil, fmt.Errorf("InvocationContext not provided")
	}
	if event == nil || event.Content == nil {
		return nil, nil
	}

	parts, err := ToA2AParts(event.Content.Parts, event.LongRunningToolIDs)
	if err != nil {
		return nil, fmt.Errorf("part conversion failed: %w", err)
	}
	if len(parts) == 0 {
		return nil, nil
	}

	var role a2a.MessageRole
	if event.Author == "user" {
		role = a2a.MessageRoleUser
	} else {
		role = a2a.MessageRoleAgent
	}

	return a2a.NewMessage(role, parts...), nil
}

func ToSessionEvent(ctx agent.InvocationContext, event a2a.Event) (*session.Event, error) {
	switch v := event.(type) {
	case *a2a.Task:
		return taskToEvent(ctx, v)

	case *a2a.Message:
		return messageToEvent(ctx, v)

	case *a2a.TaskArtifactUpdateEvent:
		if len(v.Artifact.Parts) == 0 {
			return nil, nil
		}
		event, err := artifactToEvent(ctx, v.Artifact)
		if err != nil {
			return nil, err
		}
		event.LongRunningToolIDs = getLongRunningToolIDs(v.Artifact.Parts, event.Content.Parts)
		event.CustomMetadata = map[string]any{
			CustomMetaTaskIDKey:    v.TaskID,
			CustomMetaContextIDKey: v.ContextID,
		}
		return event, nil

	case *a2a.TaskStatusUpdateEvent:
		if v.Final {
			return finalTaskStatusUpdateToEvent(ctx, v)
		}
		if v.Status.Message == nil {
			return nil, nil
		}
		event, err := messageToEvent(ctx, v.Status.Message)
		event.CustomMetadata = map[string]any{
			CustomMetaTaskIDKey:    v.TaskID,
			CustomMetaContextIDKey: v.ContextID,
		}
		if err != nil {
			return nil, err
		}
		if len(event.Content.Parts) == 0 {
			return nil, nil
		}
		for _, part := range event.Content.Parts {
			part.Thought = true
		}
		return event, nil

	default:
		return nil, fmt.Errorf("unknown event type: %T", v)
	}
}

func messageToEvent(ctx agent.InvocationContext, msg *a2a.Message) (*session.Event, error) {
	if ctx == nil {
		return nil, fmt.Errorf("InvocationContext not provided")
	}
	if msg == nil {
		return nil, nil
	}

	parts, err := toGenAIParts(msg.Parts)
	if err != nil {
		return nil, err
	}

	event := NewRemoteAgentEvent(ctx)
	if len(parts) > 0 {
		event.Content = genai.NewContentFromParts(parts, toGenAIRole(msg.Role))
	}
	if msg.TaskID != "" || msg.ContextID != "" {
		event.CustomMetadata = map[string]any{
			CustomMetaTaskIDKey:    msg.TaskID,
			CustomMetaContextIDKey: msg.ContextID,
		}
	}
	return event, nil
}

func artifactToEvent(ctx agent.InvocationContext, artifact *a2a.Artifact) (*session.Event, error) {
	if ctx == nil {
		return nil, fmt.Errorf("InvocationContext not provided")
	}

	parts, err := toGenAIParts(artifact.Parts)
	if err != nil {
		return nil, err
	}

	event := NewRemoteAgentEvent(ctx)
	event.Content = genai.NewContentFromParts(parts, genai.RoleModel)
	return event, nil
}

func taskToEvent(ctx agent.InvocationContext, task *a2a.Task) (*session.Event, error) {
	if ctx == nil {
		return nil, fmt.Errorf("InvocationContext not provided")
	}

	var parts []*genai.Part
	var longRunningToolIDs []string
	for _, artifact := range task.Artifacts {
		artifactParts, err := toGenAIParts(artifact.Parts)
		if err != nil {
			return nil, fmt.Errorf("failed to convert artifact parts: %w", err)
		}
		lrtIDs := getLongRunningToolIDs(artifact.Parts, artifactParts)

		parts = append(parts, artifactParts...)
		longRunningToolIDs = append(longRunningToolIDs, lrtIDs...)
	}

	if task.Status.Message != nil {
		msgParts, err := toGenAIParts(task.Status.Message.Parts)
		if err != nil {
			return nil, fmt.Errorf("failed to convert status message parts: %w", err)
		}
		lrtIDs := getLongRunningToolIDs(task.Status.Message.Parts, msgParts)

		parts = append(parts, msgParts...)
		longRunningToolIDs = append(longRunningToolIDs, lrtIDs...)
	}

	event := NewRemoteAgentEvent(ctx)
	if len(parts) > 0 {
		event.Content = genai.NewContentFromParts(parts, genai.RoleModel)
	}
	event.CustomMetadata = map[string]any{
		CustomMetaTaskIDKey:    task.ID,
		CustomMetaContextIDKey: task.ContextID,
	}
	if task.Status.State == a2a.TaskStateInputRequired {
		event.LongRunningToolIDs = longRunningToolIDs
	}
	return event, nil
}

func finalTaskStatusUpdateToEvent(ctx agent.InvocationContext, update *a2a.TaskStatusUpdateEvent) (*session.Event, error) {
	if update == nil {
		return nil, nil
	}

	var parts []*genai.Part
	if update.Status.Message != nil {
		localParts, err := toGenAIParts(update.Status.Message.Parts)
		if err != nil {
			return nil, err
		}
		parts = localParts
	}
	event := NewRemoteAgentEvent(ctx)
	if len(parts) > 0 {
		event.Content = genai.NewContentFromParts(parts, genai.RoleModel)
	}
	event.CustomMetadata = map[string]any{
		CustomMetaTaskIDKey:    update.TaskID,
		CustomMetaContextIDKey: update.ContextID,
	}
	return event, nil
}

func getLongRunningToolIDs(parts []a2a.Part, converted []*genai.Part) []string {
	var ids []string
	for i, part := range parts {
		dp, ok := part.(a2a.DataPart)
		if !ok {
			continue
		}
		if longRunning, ok := dp.Metadata[a2aDataPartMetaLongRunningKey].(bool); ok && longRunning {
			fnCall := converted[i]
			if fnCall.FunctionCall == nil {
				// TODO(yarolegovich): log a warning
				continue
			}
			ids = append(ids, fnCall.FunctionCall.ID)
		}
	}
	return ids
}

func toGenAIRole(role a2a.MessageRole) genai.Role {
	if role == a2a.MessageRoleUser {
		return genai.RoleUser
	} else {
		return genai.RoleModel
	}
}
