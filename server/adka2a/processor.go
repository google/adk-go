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
	"maps"

	"github.com/a2aproject/a2a-go/a2a"
	"github.com/a2aproject/a2a-go/a2asrv"

	"google.golang.org/adk/model"
	"google.golang.org/adk/session"
)

type eventToArtifactTransform interface {
	transform(event *session.Event, parts []a2a.Part, meta map[string]any) (*a2a.TaskArtifactUpdateEvent, error)
	makeFinalUpdate() *a2a.TaskArtifactUpdateEvent
}

type eventProcessor struct {
	reqCtx        *a2asrv.RequestContext
	meta          invocationMeta
	partConverter GenAIPartConverter

	// terminalActions is used to keep track of escalate and agent transfer actions on processed events.
	// It is then gets passed to caller through with metadata of a terminal event.
	// This is done to make sure the caller processes it, since intermediate events without parts might be ignored.
	terminalActions session.EventActions

	// failedEvent is used to postpone sending a terminal event until the whole ADK response is saved as an A2A artifact.
	// Will be sent as the final Task status update if not nil.
	failedEvent *a2a.TaskStatusUpdateEvent

	// inputRequiredProcessor is used to postpone sending input-required in response to long-running function tool calls.
	// inputRequiredProcessor.event will be sent as the final Task status update if failedEvent is nil.
	inputRequiredProcessor *inputRequiredProcessor

	// eventToArtifact is used to convert ADK events to A2A TaskArtifactUpdateEvents.
	eventToArtifact eventToArtifactTransform
}

func newEventProcessor(
	reqCtx *a2asrv.RequestContext,
	meta invocationMeta,
	converter GenAIPartConverter,
	transform eventToArtifactTransform,
) *eventProcessor {
	return &eventProcessor{
		inputRequiredProcessor: newInputRequiredProcessor(reqCtx),
		partConverter:          converter,
		reqCtx:                 reqCtx,
		meta:                   meta,
		eventToArtifact:        transform,
	}
}

func (p *eventProcessor) process(ctx context.Context, event *session.Event) (*a2a.TaskArtifactUpdateEvent, error) {
	if event == nil {
		return nil, nil
	}

	p.updateTerminalActions(event)

	eventMeta, err := toEventMeta(p.meta, event)
	if err != nil {
		return nil, err
	}

	resp := event.LLMResponse
	if resp.ErrorCode != "" || resp.ErrorMessage != "" {
		// TODO(yarolegovich): consider merging responses if multiple errors can be produced during an invocation
		if p.failedEvent == nil {
			// terminal event might add additional keys to its metadata when it's dispatched and these changes should
			// not be reflected in this event's metadata
			terminalEventMeta := maps.Clone(eventMeta)
			p.failedEvent = toTaskFailedUpdateEvent(p.reqCtx, errorFromResponse(&resp), terminalEventMeta)
		}
	}

	event, err = p.inputRequiredProcessor.process(event)
	if err != nil {
		return nil, fmt.Errorf("input required processing failed: %w", err)
	}

	parts, err := p.convertParts(ctx, event)
	if err != nil {
		return nil, err
	}
	if len(parts) == 0 {
		return nil, nil
	}

	result, err := p.eventToArtifact.transform(event, parts, eventMeta)
	if err != nil {
		return nil, err
	}

	return result, nil
}

func (p *eventProcessor) makeFinalArtifactUpdate() *a2a.TaskArtifactUpdateEvent {
	return p.eventToArtifact.makeFinalUpdate()
}

func (p *eventProcessor) makeFinalStatusUpdate() *a2a.TaskStatusUpdateEvent {
	for _, event := range []*a2a.TaskStatusUpdateEvent{p.failedEvent, p.inputRequiredProcessor.event} {
		if event != nil {
			event.Metadata = setActionsMeta(event.Metadata, p.terminalActions)
			return event
		}
	}

	ev := a2a.NewStatusUpdateEvent(p.reqCtx, a2a.TaskStateCompleted, nil)
	ev.Final = true
	// we're modifying base processor metadata which might have been sent with one of the previous events.
	// this update shouldn't be reflected in the sent events' metadata.
	baseMetaCopy := maps.Clone(p.meta.eventMeta)
	ev.Metadata = setActionsMeta(baseMetaCopy, p.terminalActions)
	return ev
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

func (p *eventProcessor) updateTerminalActions(event *session.Event) {
	p.terminalActions.Escalate = p.terminalActions.Escalate || event.Actions.Escalate
	if event.Actions.TransferToAgent != "" {
		p.terminalActions.TransferToAgent = event.Actions.TransferToAgent
	}
}

func (p *eventProcessor) convertParts(ctx context.Context, event *session.Event) ([]a2a.Part, error) {
	if event.Content == nil || len(event.Content.Parts) == 0 {
		return nil, nil
	}
	parts := event.Content.Parts
	if p.partConverter == nil {
		return ToA2AParts(parts, event.LongRunningToolIDs)
	}
	converted := make([]a2a.Part, 0, len(parts))
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

func toTaskFailedUpdateEvent(task a2a.TaskInfoProvider, cause error, meta map[string]any) *a2a.TaskStatusUpdateEvent {
	msg := a2a.NewMessageForTask(a2a.MessageRoleAgent, task, a2a.TextPart{Text: cause.Error()})
	ev := a2a.NewStatusUpdateEvent(task, a2a.TaskStateFailed, msg)
	ev.Metadata = meta
	ev.Final = true
	return ev
}

func errorFromResponse(resp *model.LLMResponse) error {
	return fmt.Errorf("llm error response: %q", resp.ErrorMessage)
}
