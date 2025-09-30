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

package telemetry

import (
	"fmt"
	"sync"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
	"google.golang.org/adk/agent"
	"google.golang.org/adk/llm"
	"google.golang.org/adk/session"

	sdktrace "go.opentelemetry.io/otel/sdk/trace"
)

var (
	once   sync.Once
	limits = sdktrace.SpanLimits{
		AttributeValueLengthLimit:   -1,
		AttributeCountLimit:         -1,
		EventCountLimit:             -1,
		LinkCountLimit:              -1,
		AttributePerEventCountLimit: -1,
		AttributePerLinkCountLimit:  -1,
	}
)

func RegisterTelemetry(processors []sdktrace.SpanProcessor) {
	once.Do(func() {
		traceProvider := sdktrace.NewTracerProvider(
			sdktrace.WithRawSpanLimits(limits),
		)
		for _, processor := range processors {
			traceProvider.RegisterSpanProcessor(processor)
		}
		otel.SetTracerProvider(traceProvider)
	})
}

func GetTracer() trace.Tracer {
	return otel.GetTracerProvider().Tracer("gcp.vertex.agent")
}

func TraceLLMCall(span trace.Span, agentCtx agent.Context, event *session.Event, llmRequest *llm.Request) {
	fmt.Printf("TraceLLMCall: %v\n", span)
	attributes := []attribute.KeyValue{
		attribute.String("gen_ai.system", "gcp.vertex.agent"),
		// attribute.String("gen_ai.request.model", agentCtx.Agent().Model),
		attribute.String("gcp.vertex.agent.invocation_id", event.InvocationID),
		attribute.String("gcp.vertex.agent.session_id", agentCtx.Session().ID().SessionID),
		attribute.String("gcp.vertex.agent.event_id", event.ID),
	}
	span.SetAttributes(attributes...)
	span.End()
}
