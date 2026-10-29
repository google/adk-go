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
	span trace.Span
	// logger is the logger the event goes to, and eventEnabled whether it
	// will take the event at all. Resolved once per call: with no
	// LoggerProvider installed the logger is a no-op, and converting a
	// conversation for it buys nothing.
	logger       log.Logger
	eventEnabled bool
	eventRequest []attribute.KeyValue
}

// newOperationDetails resolves where the details of the call on span go.
func newOperationDetails(ctx context.Context, span trace.Span) operationDetails {
	d := operationDetails{span: span}
	if useLegacySchema() {
		return d
	}
	d.logger = otelLogger()
	d.eventEnabled = d.logger.Enabled(ctx, log.EnabledParameters{
		Severity:  log.SeverityDebug,
		EventName: inferenceOperationDetailsEventName,
	})
	return d
}

// noteLLMRequest records the messages of req, where content capture asks for
// them. Called once the call has ended, not before it: a model may edit the
// request it is given (gemini appends a turn-alternation placeholder), and the
// span and the event describe the request as the model left it. Nothing reads
// the span's attributes before it ends, so recording them late loses nothing,
// and converting a conversation costs milliseconds, so it is done once. What
// the request held before an edit is not recorded.
func (d *operationDetails) noteLLMRequest(req *model.LLMRequest) {
	if useLegacySchema() {
		return
	}
	// After the sampling decision, not before: converting a whole conversation
	// costs milliseconds, and on a span the sampler drops it buys nothing.
	onSpan := d.span.IsRecording() && captureContentOnSpans()
	onEvent := d.eventEnabled && getGenAICaptureMessageContent()
	if !onSpan && !onEvent {
		return
	}
	content := requestContent(req)
	if onSpan {
		d.span.SetAttributes(spanContentAttributes(content)...)
	}
	if onEvent {
		d.eventRequest = eventContentAttributes(content)
	}
}

// emit emits the event, with messages only when capture is on, and not at
// all when no logger takes it.
func (d *operationDetails) emit(ctx context.Context, params GenerateContentParams, result generateContentResult) {
	if useLegacySchema() || !d.eventEnabled {
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
	d.logger.Emit(ctx, record)
}
