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
	"encoding/json"
	"sync"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/trace"
	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/model"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
)

type tracerProviderHolder struct {
	tp trace.TracerProvider
}

type tracerProviderConfig struct {
	spanProcessors []sdktrace.SpanProcessor
	mu             *sync.RWMutex
}

var (
	once              sync.Once
	localTracer       tracerProviderHolder
	localTracerConfig = tracerProviderConfig{
		spanProcessors: []sdktrace.SpanProcessor{},
		mu:             &sync.RWMutex{},
	}
)

const (
	systemName            = "gcp.vertex.agent"
	genAiOperationName    = "gen_ai.operation.name"
	genAiToolDescription  = "gen_ai.tool.description"
	genAiToolName         = "gen_ai.tool.name"
	genAiToolCallID       = "gen_ai.tool.call.id"
	genAiSystemName       = "gen_ai.system"
	genAiRequestModelName = "gen_ai.request.model"

	gcpVertexAgentLLMRequestName   = "gcp.vertex.agent.llm_request"
	gcpVertexAgentToolCallArgsName = "gcp.vertex.agent.tool_call_args"
	gcpVertexAgentEventID          = "gcp.vertex.agent.event_id"
	gcpVertexAgentToolResponseName = "gcp.vertex.agent.tool_response"
	gcpVertexAgentLLMResponseName  = "gcp.vertex.agent.llm_response"
	gcpVertexAgentInvocationID     = "gcp.vertex.agent.invocation_id"
	gcpVertexAgentSessionID        = "gcp.vertex.agent.session_id"

	executeToolName = "execute_tool"
	mergeToolName   = "(merged tools)"
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
		traceProvider := sdktrace.NewTracerProvider()
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
		localTracer.tp.Tracer(systemName),
		otel.GetTracerProvider().Tracer(systemName),
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

// TraceMergedToolCalls traces the tool execution events.
func TraceMergedToolCalls(spans []trace.Span, fnResponseEvent *session.Event) {
	if fnResponseEvent == nil {
		return
	}
	for _, span := range spans {
		attributes := []attribute.KeyValue{
			attribute.String(genAiOperationName, executeToolName),
			attribute.String(genAiToolName, mergeToolName),
			attribute.String(genAiToolDescription, mergeToolName),
			// Setting empty llm request and response (as UI expect these) while not
			// applicable for tool_response.
			attribute.String(gcpVertexAgentLLMRequestName, "{}"),
			attribute.String(gcpVertexAgentLLMRequestName, "{}"),
			attribute.String(gcpVertexAgentToolCallArgsName, "N/A"),
			attribute.String(gcpVertexAgentEventID, fnResponseEvent.ID),
			attribute.String(gcpVertexAgentToolResponseName, safeSerialize(fnResponseEvent)),
		}
		span.SetAttributes(attributes...)
		span.End()
	}
}

// TraceToolCall traces the tool execution events.
func TraceToolCall(spans []trace.Span, tool tool.Tool, fnArgs map[string]any, fnResponseEvent *session.Event) {
	if fnResponseEvent == nil {
		return
	}
	for _, span := range spans {
		attributes := []attribute.KeyValue{
			attribute.String(genAiOperationName, executeToolName),
			attribute.String(genAiToolName, tool.Name()),
			attribute.String(genAiToolDescription, tool.Description()),
			// TODO: add tool type

			// Setting empty llm request and response (as UI expect these) while not
			// applicable for tool_response.
			attribute.String(gcpVertexAgentLLMRequestName, "{}"),
			attribute.String(gcpVertexAgentLLMRequestName, "{}"),
			attribute.String(gcpVertexAgentToolCallArgsName, safeSerialize(fnArgs)),
			attribute.String(gcpVertexAgentEventID, fnResponseEvent.ID),
		}

		toolCallID := "<not specified>"
		toolResponse := "<not specified>"

		if fnResponseEvent.LLMResponse.Content != nil {
			responseParts := fnResponseEvent.LLMResponse.Content.Parts

			if len(responseParts) > 0 {
				functionResponse := responseParts[0].FunctionResponse
				if functionResponse != nil {
					if functionResponse.ID != "" {
						toolCallID = functionResponse.ID
					}
					if functionResponse.Response != nil {
						toolResponse = safeSerialize(functionResponse.Response)
					}
				}
			}
		}

		attributes = append(attributes, attribute.String(genAiToolCallID, toolCallID))
		attributes = append(attributes, attribute.String(gcpVertexAgentToolResponseName, toolResponse))

		span.SetAttributes(attributes...)
		span.End()
	}
}

// TraceLLMCall fills the call_llm event details.
func TraceLLMCall(spans []trace.Span, agentCtx agent.InvocationContext, llmRequest *model.LLMRequest, event *session.Event) {
	for _, span := range spans {
		attributes := []attribute.KeyValue{
			attribute.String(genAiSystemName, systemName),
			attribute.String(genAiRequestModelName, llmRequest.Model),
			attribute.String(gcpVertexAgentInvocationID, event.InvocationID),
			attribute.String(gcpVertexAgentSessionID, agentCtx.Session().ID()),
			attribute.String(gcpVertexAgentEventID, event.ID),
			attribute.String(gcpVertexAgentLLMRequestName, safeSerialize(llmRequestToTrace(llmRequest))),
			attribute.String(gcpVertexAgentLLMResponseName, safeSerialize(event.LLMResponse)),
		}

		if llmRequest.Config.TopP != nil {
			attributes = append(attributes, attribute.Float64("gen_ai.request.top_p", float64(*llmRequest.Config.TopP)))
		}

		if llmRequest.Config.MaxOutputTokens != 0 {
			attributes = append(attributes, attribute.Int("gen_ai.request.max_tokens", int(llmRequest.Config.MaxOutputTokens)))
		}

		// TODO: add usage_metadata and finish_reason once ADK has them.

		span.SetAttributes(attributes...)
		span.End()
	}
}

func safeSerialize(obj any) string {
	dump, err := json.Marshal(obj)
	if err != nil {
		return "<not serializable>"
	}
	return string(dump)
}

func llmRequestToTrace(llmRequest *model.LLMRequest) map[string]any {
	result := map[string]any{
		"config":  llmRequest.Config,
		"model":   llmRequest.Model,
		"content": []*genai.Content{},
	}
	for _, content := range llmRequest.Contents {
		parts := []*genai.Part{}
		// filter out InlineData part
		for _, part := range content.Parts {
			if part.InlineData != nil {
				continue
			}
			parts = append(parts, part)
		}
		filteredContent := &genai.Content{
			Role:  content.Role,
			Parts: parts,
		}
		result["content"] = append(result["content"].([]*genai.Content), filteredContent)
	}
	return result
}
