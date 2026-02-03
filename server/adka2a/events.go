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
	"maps"

	"github.com/a2aproject/a2a-go/a2a"
	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/session"
)

// NewRemoteAgentEvent create a new Event authored by the agent running in the provided invocation context.
func NewRemoteAgentEvent(ctx agent.InvocationContext) *session.Event {
	event := session.NewEvent(ctx.InvocationID())
	event.Author = ctx.Agent().Name()
	event.Branch = ctx.Branch()
	return event
}

// EventToMessage converts the provided session event to A2A message.
func EventToMessage(event *session.Event) (*a2a.Message, error) {
	if event == nil {
		return nil, nil
	}

	parts, err := ToA2AParts(event.Content.Parts, event.LongRunningToolIDs)
	if err != nil {
		return nil, fmt.Errorf("part conversion failed: %w", err)
	}
	eventMeta, err := toEventMeta(invocationMeta{}, event)
	if err != nil {
		return nil, fmt.Errorf("event metadata conversion failed: %w", err)
	}

	var role a2a.MessageRole
	if event.Author == "user" {
		role = a2a.MessageRoleUser
	} else {
		role = a2a.MessageRoleAgent
	}

	msg := a2a.NewMessage(role, parts...)
	msg.Metadata = setActionsMeta(msg.Metadata, event.Actions)
	maps.Copy(msg.Metadata, eventMeta)
	return msg, nil
}

// ToSessionEvent converts the provided a2a event to session event authored by the agent running in the provided invocation context.
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
			return nil, fmt.Errorf("artifact update event conversion failed: %w", err)
		}
		event.LongRunningToolIDs = getLongRunningToolIDs(v.Artifact.Parts, event.Content.Parts)
		if err := processA2AMeta(v, event); err != nil {
			return nil, fmt.Errorf("metadata processing failed: %w", err)
		}
		event.Partial = true
		return event, nil

	case *a2a.TaskStatusUpdateEvent:
		if v.Final {
			return finalTaskStatusUpdateToEvent(ctx, v)
		}
		if v.Status.Message == nil {
			return nil, nil
		}
		event, err := messageToEvent(ctx, v.Status.Message)
		if err != nil {
			return nil, fmt.Errorf("custom metadata conversion failed: %w", err)
		}
		if len(event.Content.Parts) == 0 {
			return nil, nil
		}
		if err := processA2AMeta(v, event); err != nil {
			return nil, fmt.Errorf("metadata processing failed: %w", err)
		}
		for _, part := range event.Content.Parts {
			part.Thought = true
		}
		event.Partial = true
		return event, nil

	default:
		return nil, fmt.Errorf("unknown event type: %T", v)
	}
}

// ToCustomMetadata creates a session event custom metadata with A2A task and context IDs in it.
func ToCustomMetadata(taskID a2a.TaskID, ctxID string) map[string]any {
	if taskID == "" && ctxID == "" {
		return nil
	}
	result := make(map[string]any)
	if taskID != "" {
		result[customMetaTaskIDKey] = string(taskID)
	}
	if ctxID != "" {
		result[customMetaContextIDKey] = ctxID
	}
	return result
}

// GetA2ATaskInfo returns A2A task and context IDs if they are present in session event custom metadata.
func GetA2ATaskInfo(event *session.Event) (a2a.TaskID, string) {
	var taskID a2a.TaskID
	var contextID string
	if event == nil || event.CustomMetadata == nil {
		return taskID, contextID
	}
	if tid, ok := event.CustomMetadata[customMetaTaskIDKey].(string); ok {
		taskID = a2a.TaskID(tid)
	}
	if ctxID, ok := event.CustomMetadata[customMetaContextIDKey].(string); ok {
		contextID = ctxID
	}
	return taskID, contextID
}

func messageToEvent(ctx agent.InvocationContext, msg *a2a.Message) (*session.Event, error) {
	if ctx == nil {
		return nil, fmt.Errorf("InvocationContext not provided")
	}
	if msg == nil {
		return nil, nil
	}

	parts, err := ToGenAIParts(msg.Parts)
	if err != nil {
		return nil, err
	}

	event := NewRemoteAgentEvent(ctx)
	if len(parts) > 0 {
		event.Content = genai.NewContentFromParts(parts, toGenAIRole(msg.Role))
	}
	if err := processA2AMeta(msg, event); err != nil {
		return nil, fmt.Errorf("metadata processing failed: %w", err)
	}
	return event, nil
}

func artifactToEvent(ctx agent.InvocationContext, artifact *a2a.Artifact) (*session.Event, error) {
	if ctx == nil {
		return nil, fmt.Errorf("InvocationContext not provided")
	}

	parts, err := ToGenAIParts(artifact.Parts)
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
		artifactParts, err := ToGenAIParts(artifact.Parts)
		if err != nil {
			return nil, fmt.Errorf("failed to convert artifact parts: %w", err)
		}
		lrtIDs := getLongRunningToolIDs(artifact.Parts, artifactParts)

		parts = append(parts, artifactParts...)
		longRunningToolIDs = append(longRunningToolIDs, lrtIDs...)
	}

	event := NewRemoteAgentEvent(ctx)

	if task.Status.Message != nil {
		msgParts, err := ToGenAIParts(task.Status.Message.Parts)
		if err != nil {
			return nil, fmt.Errorf("failed to convert status message parts: %w", err)
		}
		lrtIDs := getLongRunningToolIDs(task.Status.Message.Parts, msgParts)

		if task.Status.State == a2a.TaskStateFailed && len(msgParts) == 1 && msgParts[0].Text != "" {
			event.ErrorMessage = msgParts[0].Text
		} else {
			parts = append(parts, msgParts...)
		}
		longRunningToolIDs = append(longRunningToolIDs, lrtIDs...)
	}

	notTerminal := !task.Status.State.Terminal() && task.Status.State != a2a.TaskStateInputRequired
	if len(parts) == 0 && notTerminal {
		return nil, nil
	}

	if len(parts) > 0 {
		event.Content = genai.NewContentFromParts(parts, genai.RoleModel)
	}
	if task.Status.State == a2a.TaskStateInputRequired {
		event.LongRunningToolIDs = longRunningToolIDs
	}
	if err := processA2AMeta(task, event); err != nil {
		return nil, fmt.Errorf("metadata processing failed: %w", err)
	}
	return event, nil
}

func finalTaskStatusUpdateToEvent(ctx agent.InvocationContext, update *a2a.TaskStatusUpdateEvent) (*session.Event, error) {
	if update == nil {
		return nil, nil
	}

	event := NewRemoteAgentEvent(ctx)

	var parts []*genai.Part
	if update.Status.Message != nil {
		localParts, err := ToGenAIParts(update.Status.Message.Parts)
		if err != nil {
			return nil, err
		}
		parts = localParts
	}
	if update.Status.State == a2a.TaskStateFailed && len(parts) == 1 && parts[0].Text != "" {
		event.ErrorMessage = parts[0].Text
	} else if len(parts) > 0 {
		event.Content = genai.NewContentFromParts(parts, genai.RoleModel)
	}
	if err := processA2AMeta(update, event); err != nil {
		return nil, fmt.Errorf("metadata processing failed: %w", err)
	}
	if update.Status.Message != nil {
		event.LongRunningToolIDs = getLongRunningToolIDs(update.Status.Message.Parts, parts)
	}
	event.TurnComplete = true
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

func toEventActions(meta map[string]any) session.EventActions {
	if meta == nil {
		return session.EventActions{}
	}
	var result session.EventActions
	result.Escalate, _ = meta[metadataEscalateKey].(bool)
	result.TransferToAgent, _ = meta[metadataTransferToAgentKey].(string)
	return result
}
