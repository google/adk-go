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
	"slices"

	"github.com/a2aproject/a2a-go/a2a"
	"github.com/a2aproject/a2a-go/a2asrv"
	"google.golang.org/genai"

	"google.golang.org/adk/model"
	"google.golang.org/adk/session"
)

type eventProcessor struct {
	reqCtx *a2asrv.RequestContext
	meta   invocationMeta

	// Created once the first TaskArtifactUpdateEvent is sent. Used for subsequent artifact updates.
	responseID a2a.ArtifactID

	// We don't send terminal events during processing because we don't want A2A server to stop reading from the queue
	// until the whole ADK response is saved as an A2A artifact.
	// The highest-priority terminal event from this map is going to be send as the final Task status update, in the order of priority:
	//  - failed
	//  - input_required
	terminalEvents map[a2a.TaskState]*a2a.TaskStatusUpdateEvent
}

func newEventProcessor(reqCtx *a2asrv.RequestContext, meta invocationMeta) *eventProcessor {
	return &eventProcessor{
		reqCtx:         reqCtx,
		meta:           meta,
		terminalEvents: make(map[a2a.TaskState]*a2a.TaskStatusUpdateEvent),
	}
}

func (p *eventProcessor) process(_ context.Context, event *session.Event) (*a2a.TaskArtifactUpdateEvent, error) {
	if event == nil {
		return nil, nil
	}

	eventMeta, err := toEventMeta(p.meta, event)
	if err != nil {
		return nil, err
	}

	resp := event.LLMResponse
	if resp.ErrorCode != "" {
		// TODO(yarolegovich): consider merging responses if multiple errors can be produced during an invocation
		if _, ok := p.terminalEvents[a2a.TaskStateFailed]; !ok {
			p.terminalEvents[a2a.TaskStateFailed] = toTaskFailedUpdateEvent(p.reqCtx, errorFromResponse(&resp), eventMeta)
		}
	}

	if resp.Content == nil || len(resp.Content.Parts) == 0 {
		return nil, nil
	}

	if isInputRequired(event, resp.Content.Parts) {
		ev := a2a.NewStatusUpdateEvent(p.reqCtx, a2a.TaskStateInputRequired, nil)
		ev.Final = true
		p.terminalEvents[a2a.TaskStateFailed] = ev
	}

	parts, err := ToA2AParts(resp.Content.Parts, event.LongRunningToolIDs)
	if err != nil {
		return nil, err
	}

	var result *a2a.TaskArtifactUpdateEvent
	if p.responseID == "" {
		result = a2a.NewArtifactEvent(p.reqCtx, parts...)
		p.responseID = result.Artifact.ID
	} else {
		result = a2a.NewArtifactUpdateEvent(p.reqCtx, p.responseID, parts...)
	}
	if len(eventMeta) > 0 {
		result.Metadata = eventMeta
	}

	return result, nil
}

func (p *eventProcessor) makeTerminalEvents() []a2a.Event {
	result := make([]a2a.Event, 0, 2)

	if p.responseID != "" {
		ev := a2a.NewArtifactUpdateEvent(p.reqCtx, p.responseID)
		ev.LastChunk = true
		result = append(result, ev)
	}

	for _, s := range []a2a.TaskState{a2a.TaskStateFailed, a2a.TaskStateInputRequired} {
		if ev, ok := p.terminalEvents[s]; ok {
			result = append(result, ev)
			return result
		}
	}

	ev := a2a.NewStatusUpdateEvent(p.reqCtx, a2a.TaskStateCompleted, nil)
	ev.Metadata = p.meta.eventMeta
	ev.Final = true
	result = append(result, ev)
	return result
}

func (p *eventProcessor) makeTaskFailedEvent(cause error, event *session.Event) *a2a.TaskStatusUpdateEvent {
	meta := p.meta.eventMeta
	if event != nil {
		if eventMeta, err := toEventMeta(p.meta, event); err != nil {
			// TODO(yarolegovich): log ignored error
		} else {
			meta = eventMeta
		}
	}
	return toTaskFailedUpdateEvent(p.reqCtx, cause, meta)
}

func toTaskFailedUpdateEvent(task a2a.TaskInfoProvider, cause error, meta map[string]any) *a2a.TaskStatusUpdateEvent {
	msg := a2a.NewMessageForTask(a2a.MessageRoleAgent, task, a2a.TextPart{Text: cause.Error()})
	ev := a2a.NewStatusUpdateEvent(task, a2a.TaskStateFailed, msg)
	ev.Metadata = meta
	ev.Final = true
	return ev
}

func isInputRequired(event *session.Event, parts []*genai.Part) bool {
	for _, p := range parts {
		if p.FunctionCall != nil && slices.Contains(event.LongRunningToolIDs, p.FunctionCall.ID) {
			return true
		}
	}
	return false
}

func errorFromResponse(resp *model.LLMResponse) error {
	return fmt.Errorf("llm error response: %q", resp.ErrorMessage)
}
