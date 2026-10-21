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

package telemetry

import (
	"context"
	"reflect"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/log"
	semconv "go.opentelemetry.io/otel/semconv/v1.36.0"
	"go.opentelemetry.io/otel/trace"

	"google.golang.org/adk/v2/model"
)

// operationDetails records the messages of one model request on its
// generate_content span and its gen_ai.client.inference.operation.details
// event, so that both describe the same request. span must be set; the legacy
// schema records nothing through it.
type operationDetails struct {
	span         trace.Span
	request      []contentAttr
	eventRequest []attribute.KeyValue
}

// noteLLMRequest records the messages of req, where content capture asks for
// them. A model may edit the request it is given, so call it again once the
// call has ended: the span and the event then describe the request as the
// model left it. Messages that did not change are not encoded again.
func (d *operationDetails) noteLLMRequest(req *model.LLMRequest) {
	if useLegacySchema() {
		return
	}
	// After the sampling decision, not before: converting a whole conversation
	// costs milliseconds, and on a span the sampler drops it buys nothing.
	onSpan := d.span.IsRecording() && captureContentOnSpans()
	onEvent := getGenAICaptureMessageContent()
	if !onSpan && !onEvent {
		return
	}
	content := requestContent(req)
	if reflect.DeepEqual(content, d.request) {
		return
	}
	d.request = content
	if onSpan {
		// ponytail: span attributes cannot be removed, so an edit that drops a
		// message key, or grows it past the size limit, leaves the earlier value.
		d.span.SetAttributes(spanContentAttributes(content)...)
	}
	if onEvent {
		d.eventRequest = eventContentAttributes(content)
	}
}

// emit emits the event, with messages only when capture is on.
func (d *operationDetails) emit(ctx context.Context, params GenerateContentParams, result generateContentResult) {
	if useLegacySchema() {
		return
	}
	attrs := []attribute.KeyValue{
		semconv.GenAIOperationNameGenerateContent,
		semconv.GenAIRequestModel(params.ModelName),
		gcpVertexAgentInvocationID.String(params.InvocationID),
	}
	if provider, ok := providerName(params.ModelName, params.Backend); ok {
		attrs = append(attrs, provider)
	}
	if result.Error != nil {
		attrs = append(attrs, errorType(result.Error))
	}
	attrs = append(attrs, generateContentResultAttributes(result)...)
	if getGenAICaptureMessageContent() {
		attrs = append(attrs, d.eventRequest...)
		attrs = append(attrs, eventContentAttributes(responseContent(result))...)
	}
	record := log.Record{}
	record.SetEventName(inferenceOperationDetailsEventName)
	// The spec makes this event opt-in; DEBUG lets consumers filter it out.
	record.SetSeverity(log.SeverityDebug)
	record.AddAttributes(attrs...)
	otelLogger().Emit(ctx, record)
}
