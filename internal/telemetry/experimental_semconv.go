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
	"bytes"
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
	span trace.Span
	// logger takes the event, and eventEnabled says whether it will, asked
	// once per call: with no LoggerProvider installed the logger is a no-op,
	// and converting a conversation for it buys nothing.
	logger       log.Logger
	eventEnabled bool
	request      []contentAttr
	eventRequest []attribute.KeyValue
}

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
	onEvent := d.eventEnabled && getGenAICaptureMessageContent()
	if !onSpan && !onEvent {
		return
	}
	content := requestContent(req)
	if contentAttrsEqual(content, d.request) {
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

// contentAttrsEqual is reflect.DeepEqual(a, b) without reflection for the
// values requestContent builds: DeepEqual records every non-nil slice, pointer
// and interface it passes in a map, at least one entry per message and one per
// part. Values of other types are left to reflect.DeepEqual.
func contentAttrsEqual(a, b []contentAttr) bool {
	return slicesDeepEqual(a, b, func(x, y contentAttr) bool {
		if x.key != y.key {
			return false
		}
		switch xv := x.value.(type) {
		case messageList:
			yv, ok := y.value.(messageList)
			return ok && slicesDeepEqual(xv, yv, messageEqual)
		case partList:
			yv, ok := y.value.(partList)
			return ok && slicesDeepEqual(xv, yv, partEqual)
		}
		return reflect.DeepEqual(x.value, y.value)
	})
}

func messageEqual(x, y chatMessage) bool {
	return x.Role == y.Role && slicesDeepEqual(x.Parts, y.Parts, partEqual) &&
		(x.FinishReason == y.FinishReason ||
			x.FinishReason != nil && y.FinishReason != nil && *x.FinishReason == *y.FinishReason)
}

func partEqual(x, y contentValue) bool {
	switch xv := x.(type) {
	case textPart:
		yv, ok := y.(textPart)
		return ok && xv == yv
	case blobPart:
		yv, ok := y.(blobPart)
		return ok && xv == yv
	case uriPart:
		yv, ok := y.(uriPart)
		return ok && xv == yv
	case toolCallPart:
		yv, ok := y.(toolCallPart)
		return ok && xv.Type == yv.Type && xv.ID == yv.ID && xv.Name == yv.Name && rawJSONEqual(xv.Arguments, yv.Arguments)
	case toolResponsePart:
		yv, ok := y.(toolResponsePart)
		return ok && xv.Type == yv.Type && xv.ID == yv.ID && xv.Name == yv.Name && rawJSONEqual(xv.Response, yv.Response)
	}
	return reflect.DeepEqual(x, y)
}

// rawJSONEqual tells nil from empty, as reflect.DeepEqual does.
func rawJSONEqual(x, y rawJSON) bool {
	return (x == nil) == (y == nil) && bytes.Equal(x, y)
}

// slicesDeepEqual is reflect.DeepEqual's rule for slices, with eq comparing
// elements: a nil slice differs from an empty one, and a slice equals itself
// whatever it holds.
func slicesDeepEqual[S ~[]E, E any](a, b S, eq func(x, y E) bool) bool {
	if (a == nil) != (b == nil) || len(a) != len(b) {
		return false
	}
	if len(a) == 0 || &a[0] == &b[0] {
		return true
	}
	for i := range a {
		if !eq(a[i], b[i]) {
			return false
		}
	}
	return true
}

// emit emits the event, with messages only when capture is on, and not at all
// when the logger does not take it.
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
