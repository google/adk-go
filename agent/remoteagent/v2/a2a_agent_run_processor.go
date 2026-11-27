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
	"slices"
	"strings"

	"github.com/a2aproject/a2a-go/v2/a2a"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/internal/converters"
	"google.golang.org/adk/v2/server/adka2a/v2"
	"google.golang.org/adk/v2/session"
)

type artifactAggregation struct {
	parts []*genai.Part
	// text holds the accumulated text of textPart, the part streamed chunks
	// are being appended to, so appending does not recopy it on every chunk.
	text       strings.Builder
	textPart   *genai.Part
	citations  *genai.CitationMetadata
	grounding  *genai.GroundingMetadata
	customMeta map[string]any
	usage      *genai.GenerateContentResponseUsageMetadata
}

type a2aAgentRunProcessor struct {
	config        A2AConfig
	partConverter adka2a.A2APartConverter
	request       *a2a.SendMessageRequest

	// adkExtension is set once the remote agent sends a response whose
	// [adka2a.ADKExtensionURI] metadata value is truthy: true from
	// [adka2a.Executor], an object from adk-python's executor. adk-python
	// filters thought parts only from responses without a truthy value there,
	// so thoughts beside an answer are dropped only while this is false.
	adkExtension bool

	// partial event contents emitted before the terminal event
	aggregations map[a2a.ArtifactID]*artifactAggregation
	// used to emit aggregations in the order of last update
	aggregationOrder []a2a.ArtifactID
}

func newRunProcessor(config A2AConfig, request *a2a.SendMessageRequest) *a2aAgentRunProcessor {
	return &a2aAgentRunProcessor{
		config:        config,
		request:       request,
		partConverter: config.A2APartConverter,
		aggregations:  make(map[a2a.ArtifactID]*artifactAggregation),
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
	if statusUpdate, ok := a2aEvent.(*a2a.TaskStatusUpdateEvent); ok && statusUpdate.Status.State.Terminal() {
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
	p.aggregationOrder = slices.DeleteFunc(p.aggregationOrder, func(id a2a.ArtifactID) bool {
		return id == aid
	})
}

func (p *a2aAgentRunProcessor) updateAggregation(aid a2a.ArtifactID, agg *artifactAggregation, event *session.Event) {
	if event.Content != nil {
		for _, part := range event.Content.Parts {
			if part.Text != "" { // collapse small text-block parts to bigger text blocks
				if len(agg.parts) > 0 {
					lastPart := agg.parts[len(agg.parts)-1]
					// check if last part is a text block of the same 'Thought' type
					if lastPart.Text != "" && lastPart.Thought == part.Thought {
						if agg.textPart != lastPart {
							agg.text.Reset()
							agg.text.WriteString(lastPart.Text)
							agg.textPart = lastPart
						}
						agg.text.WriteString(part.Text)
						lastPart.Text = agg.text.String()
						continue
					}
				}
				agg.parts = append(agg.parts, &genai.Part{
					Text:    part.Text,
					Thought: part.Thought,
				})
			} else {
				agg.parts = append(agg.parts, part)
			}
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
	parts := agg.parts
	result := adka2a.NewRemoteAgentEvent(ctx)
	result.Content = genai.NewContentFromParts(parts, genai.RoleModel)
	// A thought sent in a chunk of its own survives dropThoughtsBesideAnswer,
	// so the artifact is filtered again once it is whole. Events from a custom
	// Converter are never filtered, and neither is the artifact reassembled
	// from them.
	if p.config.Converter == nil && !p.adkExtension {
		dropThoughtParts(result.Content)
	}
	result.CustomMetadata = agg.customMeta
	result.GroundingMetadata = agg.grounding
	result.CitationMetadata = agg.citations
	result.UsageMetadata = agg.usage
	return result
}

// convertToSessionEvent converts A2A client SendStreamingMessage result to a session event. Returns nil if nothing should be emitted.
func (p *a2aAgentRunProcessor) convertToSessionEvent(ctx agent.InvocationContext, a2aEvent a2a.Event, err error) (*session.Event, error) {
	if err != nil {
		return nil, err
	}

	// The key is read before converting. A keyed response that fails to convert
	// can still be recovered by an AfterRequestCallback, and the run goes on
	// with a remote agent that has marked itself as an ADK server.
	if a2aEvent != nil && truthy(a2aEvent.Meta()[adka2a.ADKExtensionURI]) {
		p.adkExtension = true
	}

	event, err := adka2a.ToSessionEventWithParts(ctx, a2aEvent, p.partConverter)
	if err != nil {
		return nil, fmt.Errorf("failed to convert a2aEvent: %w", err)
	}

	if event != nil {
		p.updateCustomMetadata(event, a2aEvent)
		if !p.adkExtension {
			dropThoughtsBesideAnswer(a2aEvent, event)
		}
	}

	return event, nil
}

// truthy reports whether a metadata value counts as true the way Python's
// bool() reads a JSON value: false, null, 0, "" and empty arrays and objects
// do not.
func truthy(v any) bool {
	switch v := v.(type) {
	case nil:
		return false
	case bool:
		return v
	case float64:
		return v != 0
	case string:
		return v != ""
	case []any:
		return len(v) > 0
	case map[string]any:
		return len(v) > 0
	default:
		return true
	}
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
	if resp == nil && err == nil {
		return nil, nil
	}
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

// dropThoughtsBesideAnswer removes the thought parts of an event that also
// carries parts that are not thoughts, so only the answer reaches the user.
// An event made only of thoughts is kept as it is.
//
// Interim status updates are skipped: every state except the terminal ones and
// input-required. adka2a marks the text parts of such an update as thoughts to
// report progress, but not its function call and data parts, so filtering
// would drop that text. adk-python marks every part of a submitted or working
// update as a thought, so its filter keeps the update whole. It does filter
// auth-required and unspecified updates, which adk-go skips because adka2a
// marks their text parts the same way.
//
// Submitted and working task snapshots are kept whole to match adk-python,
// which marks every part of such a snapshot as a thought.
//
// Every artifact update is filtered on its own, streamed chunks included, as
// in adk-python. A thought sent in a chunk of its own stays in that chunk, so
// buildNonPartialAggregation filters the artifact it reassembles as well.
func dropThoughtsBesideAnswer(a2aEvent a2a.Event, event *session.Event) {
	switch e := a2aEvent.(type) {
	case *a2a.TaskStatusUpdateEvent:
		if !e.Status.State.Terminal() && e.Status.State != a2a.TaskStateInputRequired {
			return
		}
	case *a2a.Task:
		if e.Status.State == a2a.TaskStateSubmitted || e.Status.State == a2a.TaskStateWorking {
			return
		}
	}
	dropThoughtParts(event.Content)
}

// dropThoughtParts removes the thought parts of content that also has parts
// that are not thoughts. Content made only of thoughts is left as it is.
func dropThoughtParts(content *genai.Content) {
	if content == nil {
		return
	}
	answer := slices.DeleteFunc(slices.Clone(content.Parts), func(p *genai.Part) bool { return p.Thought })
	if len(answer) > 0 {
		content.Parts = answer
	}
}
