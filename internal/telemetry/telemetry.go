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

// Package telemetry sets up the open telemetry exporters to the ADK.
package telemetry

import (
	"context"
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

type tracerProviderHolder struct {
	tp trace.TracerProvider
}

type tracerProviderConfig struct {
	spanProcessors []sdktrace.SpanProcessor
	mu             *sync.RWMutex
}

var (
	once        sync.Once
	localTracer tracerProviderHolder
	limits      = sdktrace.SpanLimits{
		AttributeValueLengthLimit:   -1,
		AttributeCountLimit:         -1,
		EventCountLimit:             -1,
		LinkCountLimit:              -1,
		AttributePerEventCountLimit: -1,
		AttributePerLinkCountLimit:  -1,
	}
	localTracerConfig = tracerProviderConfig{
		spanProcessors: []sdktrace.SpanProcessor{},
		mu:             &sync.RWMutex{},
	}
)

// AddSpanProcessor adds a span processor to the local tracer config.
func AddSpanProcessor(processor sdktrace.SpanProcessor) {
	localTracerConfig.mu.Lock()
	defer localTracerConfig.mu.Unlock()
	localTracerConfig.spanProcessors = append(localTracerConfig.spanProcessors, processor)
}

// RegisterTelemetry sets up the local tracer that will be used to emit traces.
// We use local tracer to respect the global tracer configurations.
func RegisterTelemetry() {
	once.Do(func() {
		traceProvider := sdktrace.NewTracerProvider(
			sdktrace.WithRawSpanLimits(limits),
		)
		localTracerConfig.mu.RLock()
		spanProcessors := localTracerConfig.spanProcessors
		localTracerConfig.mu.RUnlock()
		for _, processor := range spanProcessors {
			traceProvider.RegisterSpanProcessor(processor)
		}
		localTracer = tracerProviderHolder{tp: traceProvider}
	})
}

// If the global tracer is not set, the default NoopTracerProvider will be used.
// That means that the spans are NOT recording/exporting
// If the local tracer is not set, we'll set up tracer with all registered span processors.
func getTracers() []trace.Tracer {
	if localTracer.tp == nil {
		RegisterTelemetry()
	}
	return []trace.Tracer{
		localTracer.tp.Tracer("gcp.vertex.agent"),
		otel.GetTracerProvider().Tracer("gcp.vertex.agent"),
	}
}

// StartTrace returns two spans to start emitting events, one from global tracer and second from the local.
func StartTrace(ctx context.Context, traceName string) []trace.Span {
	tracers := getTracers()
	spans := make([]trace.Span, len(tracers))
	for i, tracer := range tracers {
		_, span := tracer.Start(ctx, traceName)
		spans[i] = span
	}
	return spans
}

// TraceLLMCall fills the call_llm event details.
func TraceLLMCall(spans []trace.Span, agentCtx agent.Context, event *session.Event, llmRequest *llm.Request) {
	for _, span := range spans {
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
}
