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
	"maps"

	"github.com/a2aproject/a2a-go/a2a"
	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	icontext "google.golang.org/adk/internal/context"
	"google.golang.org/adk/internal/converters"
	"google.golang.org/adk/server/adka2a"
	"google.golang.org/adk/session"
)

type artifactAggregation struct {
	aggregatedText     string
	aggregatedThoughts string

	parts      []*genai.Part
	citations  *genai.CitationMetadata
	grounding  *genai.GroundingMetadata
	customMeta map[string]any
	usage      *genai.GenerateContentResponseUsageMetadata
}

type a2aAgentRunProcessor struct {
	config A2AConfig

	request *a2a.MessageSendParams

	// partial event contents emitted before the terminal event
	aggregations map[a2a.ArtifactID]*artifactAggregation
	// used to emit aggregations in the order of last update
	aggregationOrder []a2a.ArtifactID
}

func newRunProcessor(config A2AConfig, request *a2a.MessageSendParams) *a2aAgentRunProcessor {
	return &a2aAgentRunProcessor{
		config:       config,
		request:      request,
		aggregations: make(map[a2a.ArtifactID]*artifactAggregation),
	}
}

// aggregatePartial stores contents of partial events to emit them with the terminal event.
// It can modify the original event or return a new event to emit before the provided event.
func (p *a2aAgentRunProcessor) aggregatePartial(ctx agent.InvocationContext, a2aEvent a2a.Event, event *session.Event) []*session.Event {
	// ADK partial events should be aggregated by ADK and emitted as a non-partial artifact update.
	// That's why we skip them regardless of the actual isPartial value.
	// This is the legacy [adka2a.OutputMode].
	if a2aEvent != nil && adka2a.IsPartialFlagSet(a2aEvent.Meta()) {
		return []*session.Event{event}
	}

	// RemoteAgent event stream finished, emit any aggregated events data we have before the final event
	if statusUpdate, ok := a2aEvent.(*a2a.TaskStatusUpdateEvent); ok && statusUpdate.Final {
		var events []*session.Event
		for _, aid := range p.aggregationOrder {
			if agg, ok := p.aggregations[aid]; ok {
				events = append(events, p.buildNonPartialAggregation(ctx, agg))
			}
		}
		return append(events, event)
	}

	// RemoteAgent published a snapshot which should have all the data we potentially aggregated.
	// Reset the aggregation so that it is not published twice.
	if _, ok := a2aEvent.(*a2a.Task); ok {
		p.aggregations = map[a2a.ArtifactID]*artifactAggregation{}
		p.aggregationOrder = nil
		return []*session.Event{event}
	}

	update, ok := a2aEvent.(*a2a.TaskArtifactUpdateEvent)
	if !ok { // do not aggregate status updates
		return []*session.Event{event}
	}

	if !update.Append { // non-append event resets aggregation
		p.removeAggregation(update.Artifact.ID)
		if update.LastChunk { // non-append event which is the last chunk must already be non-partial
			event.Partial = false
			return []*session.Event{event}
		}
	}

	aggregation := p.aggregations[update.Artifact.ID]
	if aggregation == nil {
		aggregation = &artifactAggregation{}
		p.aggregations[update.Artifact.ID] = aggregation
	}

	p.updateAggregation(update.Artifact.ID, aggregation, event)

	if !update.LastChunk {
		return []*session.Event{event}
	}

	// emit partial last chunk and follow by the non-partial aggregated event
	p.removeAggregation(update.Artifact.ID)
	return []*session.Event{event, p.buildNonPartialAggregation(ctx, aggregation)}
}

func (p *a2aAgentRunProcessor) removeAggregation(aid a2a.ArtifactID) {
	delete(p.aggregations, aid)
	p.removeFromOrder(aid)
}

func (p *a2aAgentRunProcessor) removeFromOrder(aid a2a.ArtifactID) {
	for i, id := range p.aggregationOrder {
		if id == aid {
			p.aggregationOrder = append(p.aggregationOrder[:i], p.aggregationOrder[i+1:]...)
			return
		}
	}
}

func (p *a2aAgentRunProcessor) updateAggregation(aid a2a.ArtifactID, agg *artifactAggregation, event *session.Event) {
	for _, part := range event.Content.Parts {
		if part.Text != "" { // collapse small text-block parts to bigger text blocks
			if part.Thought {
				agg.aggregatedThoughts += part.Text
			} else {
				agg.aggregatedText += part.Text
			}
		} else {
			agg.promoteTextBlocksToParts()
			agg.parts = append(agg.parts, part)
		}
	}

	if event.CitationMetadata != nil {
		if agg.citations == nil {
			agg.citations = &genai.CitationMetadata{}
		}
		agg.citations.Citations = append(agg.citations.Citations, event.CitationMetadata.Citations...)
	}
	if event.CustomMetadata != nil {
		if agg.customMeta == nil {
			agg.customMeta = make(map[string]any)
		}
		maps.Copy(agg.customMeta, event.CustomMetadata)
	}
	if event.GroundingMetadata != nil {
		agg.grounding = event.GroundingMetadata
	}
	if event.UsageMetadata != nil { // cumulative
		agg.usage = event.UsageMetadata
	}

	p.removeFromOrder(aid)
	p.aggregationOrder = append(p.aggregationOrder, aid)
}

func (p *a2aAgentRunProcessor) buildNonPartialAggregation(ctx agent.InvocationContext, agg *artifactAggregation) *session.Event {
	agg.promoteTextBlocksToParts()
	parts := agg.parts
	result := adka2a.NewRemoteAgentEvent(ctx)
	result.Content = genai.NewContentFromParts(parts, genai.RoleModel)
	result.CustomMetadata = agg.customMeta
	result.GroundingMetadata = agg.grounding
	result.CitationMetadata = agg.citations
	result.UsageMetadata = agg.usage
	return result
}

func (a *artifactAggregation) promoteTextBlocksToParts() {
	if a.aggregatedThoughts != "" {
		a.parts = append(a.parts, &genai.Part{Thought: true, Text: a.aggregatedThoughts})
		a.aggregatedThoughts = ""
	}
	if a.aggregatedText != "" {
		a.parts = append(a.parts, &genai.Part{Text: a.aggregatedText})
		a.aggregatedText = ""
	}
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
