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
	"fmt"

	"github.com/a2aproject/a2a-go/a2a"
	"google.golang.org/genai"

	"github.com/sjzsdu/adk-go/agent"
	icontext "github.com/sjzsdu/adk-go/internal/context"
	"github.com/sjzsdu/adk-go/internal/converters"
	"github.com/sjzsdu/adk-go/server/adka2a"
	"github.com/sjzsdu/adk-go/session"
)

type a2aAgentRunProcessor struct {
	config A2AConfig

	request *a2a.MessageSendParams

	// partial event contents emitted before the terminal event
	aggregatedText     string
	aggregatedThoughts string
}

func newRunProcessor(config A2AConfig, request *a2a.MessageSendParams) *a2aAgentRunProcessor {
	return &a2aAgentRunProcessor{config: config, request: request}
}

// aggregatePartial stores contents of partial events to emit them with the terminal event.
// It can modify the original event or return a new event to emit before the provided event.
func (p *a2aAgentRunProcessor) aggregatePartial(ctx agent.InvocationContext, a2aEvent a2a.Event, event *session.Event) *session.Event {
	// ADK partial events should be aggregated by ADK and emitted as a non-partial artifact update.
	// That's why we skip them regardless of the actual isPartial value.

	if a2aEvent != nil && adka2a.IsPartialFlagSet(a2aEvent.Meta()) {
		return nil
	}

	// RemoteAgent event stream finished, emit any aggregated events data we have
	if statusUpdate, ok := a2aEvent.(*a2a.TaskStatusUpdateEvent); ok && statusUpdate.Final {
		return p.buildAggregatedEvent(ctx, event)
	}

	// RemoteAgent published a snapshot which should have all the data we potentially aggregated.
	// Reset the aggregation so that it is not published twice.
	if _, ok := a2aEvent.(*a2a.Task); ok {
		p.aggregatedText = ""
		p.aggregatedThoughts = ""
		return nil
	}

	if update, ok := a2aEvent.(*a2a.TaskArtifactUpdateEvent); ok && !update.Append {
		p.aggregatedText = ""
		p.aggregatedThoughts = ""
	}

	updatedAggregatedBlock := false
	if event.Partial {
		for _, part := range event.Content.Parts {
			if part.Text == "" {
				continue
			}
			if part.Thought {
				p.aggregatedThoughts += part.Text
			} else {
				p.aggregatedText += part.Text
			}
			updatedAggregatedBlock = true
		}
	}

	if updatedAggregatedBlock {
		return nil
	}

	// If a non-partial or non-text event is received we might need to publish the data we aggregated
	// before it so that it appears as a single block of text.
	return p.buildAggregatedEvent(ctx, event)
}

func (p *a2aAgentRunProcessor) buildAggregatedEvent(ctx agent.InvocationContext, event *session.Event) *session.Event {
	parts := []*genai.Part{}
	if p.aggregatedThoughts != "" {
		parts = append(parts, &genai.Part{Thought: true, Text: p.aggregatedThoughts})
		p.aggregatedThoughts = ""
	}
	if p.aggregatedText != "" {
		parts = append(parts, &genai.Part{Text: p.aggregatedText})
		p.aggregatedText = ""
	}
	if len(parts) == 0 {
		return nil
	}

	content := genai.NewContentFromParts(parts, genai.RoleModel)

	// Use the terminal event to emit aggregated content if it would be empty otherwise.
	if event.Content == nil {
		event.Content = content
		return nil
	}

	aggregatedEvent := adka2a.NewRemoteAgentEvent(ctx)
	aggregatedEvent.Content = content
	aggregatedEvent.CustomMetadata = map[string]any{adka2a.ToADKMetaKey("aggregated"): true}
	p.updateCustomMetadata(aggregatedEvent, nil)
	return aggregatedEvent
}

// convertToSessionEvent converts A2A client SendStreamingMessage result to a session event. Returns nil if nothing should be emitted.
func (p *a2aAgentRunProcessor) convertToSessionEvent(ctx agent.InvocationContext, a2aEvent a2a.Event, err error) (*session.Event, error) {
	if err != nil {
		event := toErrorEvent(ctx, err)
		p.updateCustomMetadata(event, nil)
		return event, nil
	}

	event, err := adka2a.ToSessionEvent(ctx, a2aEvent)
	if err != nil {
		event := toErrorEvent(ctx, fmt.Errorf("failed to convert a2aEvent: %w", err))
		p.updateCustomMetadata(event, nil)
		return event, nil
	}

	if event != nil {
		p.updateCustomMetadata(event, a2aEvent)
	}

	return event, nil
}

func (p *a2aAgentRunProcessor) runBeforeA2ARequestCallbacks(ctx agent.InvocationContext) (*session.Event, error) {
	cctx := icontext.NewCallbackContext(ctx)
	for _, callback := range p.config.BeforeRequestCallbacks {
		if cbResp, cbErr := callback(cctx, p.request); cbResp != nil || cbErr != nil {
			return cbResp, cbErr
		}
	}
	return nil, nil
}

func (p *a2aAgentRunProcessor) runAfterA2ARequestCallbacks(ctx agent.InvocationContext, resp *session.Event, err error) (*session.Event, error) {
	cctx := icontext.NewCallbackContext(ctx)
	for _, callback := range p.config.AfterRequestCallbacks {
		if cbEvent, cbErr := callback(cctx, p.request, resp, err); cbEvent != nil || cbErr != nil {
			return cbEvent, cbErr
		}
	}
	return nil, nil
}

func (p *a2aAgentRunProcessor) updateCustomMetadata(event *session.Event, response a2a.Event) {
	toAdd := map[string]any{}
	if p.request != nil && event.TurnComplete {
		// only add request to the final event to avoid massive data duplication during streaming
		toAdd["request"] = p.request
	}
	if response != nil {
		toAdd["response"] = response
	}
	if len(toAdd) == 0 {
		return
	}
	if event.CustomMetadata == nil {
		event.CustomMetadata = map[string]any{}
	}
	for k, v := range toAdd {
		if v == nil {
			continue
		}
		payload, err := converters.ToMapStructure(v)
		if err == nil {
			event.CustomMetadata[adka2a.ToADKMetaKey(k)] = payload
		} else {
			event.CustomMetadata[adka2a.ToADKMetaKey(k+"_codec_error")] = err.Error()
		}
	}
}
