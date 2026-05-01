// Copyright 2026 Google LLC
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
	"slices"

	"github.com/a2aproject/a2a-go/v2/a2a"
	"github.com/a2aproject/a2a-go/v2/a2asrv"
	"github.com/a2aproject/a2a-go/v2/log"
	"google.golang.org/genai"

	"google.golang.org/adk/internal/utils"
	"google.golang.org/adk/session"
)

type inputRequiredProcessor struct {
	reqCtx        *a2asrv.ExecutorContext
	event         *a2a.TaskStatusUpdateEvent
	partConverter GenAIPartConverter
	// handles possible duplication in partial and non-partial events
	addedParts []*genai.Part
}

func newInputRequiredProcessor(reqCtx *a2asrv.ExecutorContext, partConverter GenAIPartConverter) *inputRequiredProcessor {
	return &inputRequiredProcessor{reqCtx: reqCtx, partConverter: partConverter}
}

// process handles long-running function tool calls by accumulating them for the final task status update.
// If a part was incorporated into the final task status update the original event is modified to not include it,
// so that parts are not duplicated in the response.
func (p *inputRequiredProcessor) process(ctx context.Context, event *session.Event) (*session.Event, error) {
	resp := event.LLMResponse
	if resp.Content == nil {
		return event, nil
	}

	var longRunningCallIDs []string
	var inputRequiredParts []*genai.Part
	var remainingParts []*genai.Part
	for _, part := range resp.Content.Parts {
		callID := ""
		if part.FunctionCall != nil && slices.Contains(event.LongRunningToolIDs, part.FunctionCall.ID) {
			callID = part.FunctionCall.ID
		}
		if p.isLongRunningResponse(event, part) {
			callID = part.FunctionResponse.ID
		}
		if callID == "" {
			remainingParts = append(remainingParts, part)
			continue
		}
		added := slices.ContainsFunc(p.addedParts, func(p *genai.Part) bool {
			if part.FunctionCall != nil && p.FunctionCall != nil && part.FunctionCall.ID == p.FunctionCall.ID {
				return true
			}
			return part.FunctionResponse != nil && p.FunctionResponse != nil && part.FunctionResponse.ID == p.FunctionResponse.ID
		})
		if added {
			continue
		}
		p.addedParts = append(p.addedParts, part)
		inputRequiredParts = append(inputRequiredParts, part)
		longRunningCallIDs = append(longRunningCallIDs, callID)
	}

	if len(inputRequiredParts) > 0 {
		a2aParts, err := p.convertParts(ctx, event, inputRequiredParts, longRunningCallIDs)
		if err != nil {
			return nil, fmt.Errorf("failed to convert input required parts to A2A parts: %w", err)
		}

		if p.event != nil {
			p.event.Status.Message.Parts = append(p.event.Status.Message.Parts, a2aParts...)
		} else {
			msg := a2a.NewMessage(a2a.MessageRoleAgent, a2aParts...)
			ev := a2a.NewStatusUpdateEvent(p.reqCtx, a2a.TaskStateInputRequired, msg)
			p.event = ev
		}
	}

	if len(remainingParts) == len(resp.Content.Parts) {
		return event, nil
	}

	modifiedEvent := *event
	newContent := *resp.Content
	newContent.Parts = remainingParts
	modifiedEvent.LLMResponse.Content = &newContent

	return &modifiedEvent, nil
}

func (p *inputRequiredProcessor) convertParts(ctx context.Context, event *session.Event, parts []*genai.Part, longRunningCallIDs []string) ([]*a2a.Part, error) {
	if p.partConverter == nil {
		return ToA2AParts(parts, longRunningCallIDs)
	}
	converted := make([]*a2a.Part, 0, len(parts))
	for _, part := range parts {
		cp, err := p.partConverter(ctx, event, part)
		if err != nil {
			return nil, err
		}
		if cp == nil {
			continue
		}
		converted = append(converted, cp)
	}
	return converted, nil
}

func (p *inputRequiredProcessor) isLongRunningResponse(event *session.Event, part *genai.Part) bool {
	if part.FunctionResponse == nil {
		return false
	}
	id := part.FunctionResponse.ID
	if slices.Contains(event.LongRunningToolIDs, id) {
		return true
	}
	if p.event == nil {
		return false
	}
	for _, part := range p.event.Status.Message.Parts {
		if data, ok := part.Data().(map[string]any); ok {
			if typeVal, ok := part.Meta()[a2aDataPartMetaTypeKey]; ok && typeVal == a2aDataPartTypeFunctionCall {
				if callID, ok := data["id"].(string); ok && callID == id {
					return true
				}
			}
		}
	}
	return false
}

// HandleInputRequired checks if the input message contains responses to all function calls
// that happened during the previous invocation and were recorded in the Task input-required state message.
// If a non-nil event is returned the invoking code needs to use the event as the result of the execution
func HandleInputRequired(reqCtx *a2asrv.ExecutorContext, content *genai.Content) (*a2a.TaskStatusUpdateEvent, error) {
	if reqCtx.StoredTask == nil {
		return nil, nil
	}
	task, statusMsg := reqCtx.StoredTask, reqCtx.StoredTask.Status.Message
	if task.Status.State != a2a.TaskStateInputRequired || statusMsg == nil {
		return nil, nil
	}

	taskParts, err := ToGenAIParts(statusMsg.Parts)
	if err != nil {
		return nil, fmt.Errorf("failed to parse task status message: %w", err)
	}

	for _, statusPart := range taskParts {
		if statusPart.FunctionCall == nil {
			continue
		}
		hasMatchingResponse := slices.ContainsFunc(content.Parts, func(p *genai.Part) bool {
			return p.FunctionResponse != nil && p.FunctionResponse.ID == statusPart.FunctionCall.ID
		})
		if !hasMatchingResponse {
			parts := makeInputMissingErrorMessage(statusMsg.Parts, statusPart.FunctionCall.ID)
			msg := a2a.NewMessageForTask(a2a.MessageRoleAgent, reqCtx.StoredTask, parts...)
			event := a2a.NewStatusUpdateEvent(reqCtx, a2a.TaskStateInputRequired, msg)
			return event, nil
		}
	}
	return nil, nil
}

func getSubagentTasksToCancel(ctx context.Context, status a2a.TaskStatus, session session.Session) ([]subagentTask, error) {
	pendingCallIDs, err := getPendingLongRunningCallIDs(status)
	if err != nil {
		return nil, fmt.Errorf("malformed task status: %w", err)
	}
	if len(pendingCallIDs) == 0 {
		return nil, nil
	}

	foundCalls, foundTaskIDs := map[string]bool{}, 0
	var tasksToCancel []subagentTask
	events := session.Events()
	for i := events.Len() - 1; i >= 0; i-- {
		event := events.At(i)
		for _, call := range utils.FunctionCalls(event.Content) {
			if !slices.Contains(pendingCallIDs, call.ID) {
				continue
			}
			if foundCalls[call.ID] {
				log.Warn(ctx, "duplicate function call id", "id", call.ID)
				continue
			}
			foundCalls[call.ID] = true
			if taskID, _ := GetA2ATaskInfo(event); taskID != "" {
				tasksToCancel = append(tasksToCancel, subagentTask{agentName: event.Author, taskID: a2a.TaskID(taskID)})
				foundTaskIDs++
			}
		}
		if len(foundCalls) == len(pendingCallIDs) {
			break
		}
	}

	if len(foundCalls) < len(pendingCallIDs) {
		log.Warn(ctx, "could not find all function calls from status message", "found", len(foundCalls), "total", len(pendingCallIDs))
	}

	if foundTaskIDs < len(foundCalls) {
		log.Warn(ctx, "not all function calls had a taskID", "found_calls", len(foundCalls), "found_ids", foundTaskIDs)
	}

	return tasksToCancel, nil
}

func getPendingLongRunningCallIDs(status a2a.TaskStatus) ([]string, error) {
	statusMsg := status.Message
	if status.State != a2a.TaskStateInputRequired || statusMsg == nil {
		return nil, nil
	}
	taskParts, err := ToGenAIParts(statusMsg.Parts)
	if err != nil {
		return nil, fmt.Errorf("failed to parse task status message: %w", err)
	}
	var callIDs []string
	for _, statusPart := range taskParts {
		if statusPart.FunctionCall != nil {
			callIDs = append(callIDs, statusPart.FunctionCall.ID)
		}
	}
	return callIDs, nil
}

func makeInputMissingErrorMessage(inputRequiredParts []*a2a.Part, callID string) []*a2a.Part {
	errPart := a2a.NewTextPart(fmt.Sprintf("no input provided for function call ID %q", callID))
	errPart.SetMeta("validation_error", true)

	var preservedParts []*a2a.Part
	for _, p := range inputRequiredParts {
		if meta := p.Meta(); meta != nil {
			if v, ok := meta["validation_error"].(bool); ok && v {
				continue
			}
		}
		preservedParts = append(preservedParts, p)
	}
	return append(preservedParts, errPart)
}

type subagentTask struct {
	agentName string
	taskID    a2a.TaskID
}
