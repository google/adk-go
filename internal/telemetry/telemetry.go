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

// Package telemetry implements telemetry for ADK.
//
// WARNING: telemetry provided by ADK (internaltelemetry package) may change (e.g. attributes and their names)
// because we're in process to standardize and unify telemetry across all ADKs.
package telemetry

import (
	"context"
	"encoding/json"
	"fmt"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	semconv "go.opentelemetry.io/otel/semconv/v1.36.0"
	"go.opentelemetry.io/otel/trace"
	"google.golang.org/genai"

	"google.golang.org/adk/internal/version"
	"google.golang.org/adk/model"
	"google.golang.org/adk/session"
)

const (
	systemName = "gcp.vertex.agent"

	executeToolName = "execute_tool"
	mergeToolName   = "(merged tools)"
)

var (
	gcpVertexAgentToolCallArgsName = attribute.Key("gcp.vertex.agent.tool_call_args")
	gcpVertexAgentEventID          = attribute.Key("gcp.vertex.agent.event_id")
	gcpVertexAgentToolResponseName = attribute.Key("gcp.vertex.agent.tool_response")
)

// tracer is the tracer instance for ADK go.
var tracer trace.Tracer = otel.GetTracerProvider().Tracer(
	systemName,
	trace.WithInstrumentationVersion(version.Version),
	trace.WithSchemaURL(semconv.SchemaURL),
)

type agent interface {
	Name() string
	Description() string
}

// StartInvokeAgentSpan starts a new semconv invoke_agent span.
// It returns a new context with the span and the span itself.
func StartInvokeAgentSpan(ctx context.Context, agent agent, sessionID string) (context.Context, trace.Span) {
	agentName := agent.Name()
	spanCtx, span := tracer.Start(ctx, fmt.Sprintf("invoke_agent %s", agentName), trace.WithAttributes(
		semconv.GenAIOperationNameInvokeAgent,
		semconv.GenAIAgentDescription(agent.Description()),
		semconv.GenAIAgentName(agentName),
		semconv.GenAIConversationID(sessionID),
	))

	return spanCtx, span
}

type TraceAgentResultParams struct {
	ResponseEvent *session.Event
	Error         error
}

// TraceAgentResult records the result of the agent invocation, including status and error.
func TraceAgentResult(span trace.Span, params TraceAgentResultParams) {
	recordErrorAndStatus(span, params.Error)
}

// StartGenerateContentSpanParams contains parameters for [StartGenerateContentSpan].
type StartGenerateContentSpanParams struct {
	// ModelName is the name of the model being used for generation.
	ModelName string
}

// StartGenerateContentSpan starts a new semconv generate_content span.
func StartGenerateContentSpan(ctx context.Context, params StartGenerateContentSpanParams) (context.Context, trace.Span) {
	modelName := params.ModelName
	spanCtx, span := tracer.Start(ctx, fmt.Sprintf("generate_content %s", modelName), trace.WithAttributes(
		semconv.GenAIOperationNameGenerateContent,
		semconv.GenAIRequestModel(modelName),
	))
	return spanCtx, span
}

type TraceGenerateContentResultParams struct {
	Response *model.LLMResponse
	Error    error
}

// TraceGenerateContentResult records the result of the generate_content operation, including token usage and finish reason.
func TraceGenerateContentResult(span trace.Span, params TraceGenerateContentResultParams) {
	recordErrorAndStatus(span, params.Error)
	// TODO(#479): set gcp.vertex.agent.event_id
	if params.Response == nil {
		return
	}
	span.SetAttributes(
		semconv.GenAIResponseFinishReasons(string(params.Response.FinishReason)),
	)
	if params.Response.UsageMetadata != nil {
		span.SetAttributes(
			semconv.GenAIUsageInputTokens(int(params.Response.UsageMetadata.PromptTokenCount)),
			semconv.GenAIUsageOutputTokens(int(params.Response.UsageMetadata.TotalTokenCount)),
		)
	}
}

// StartExecuteToolSpanParams contains parameters for [StartExecuteToolSpan].
type StartExecuteToolSpanParams struct {
	// ToolName is the name of the tool being executed.
	ToolName string
	// Args is the arguments of the tool call.
	Args map[string]any
}

// StartExecuteToolSpan starts a new semconv execute_tool span.
func StartExecuteToolSpan(ctx context.Context, params StartExecuteToolSpanParams) (context.Context, trace.Span) {
	toolName := params.ToolName
	spanCtx, span := tracer.Start(ctx, fmt.Sprintf("execute_tool %s", toolName), trace.WithAttributes(
		semconv.GenAIOperationNameExecuteTool,
		semconv.GenAIToolName(toolName),
		gcpVertexAgentToolCallArgsName.String(safeSerialize(params.Args))))
	return spanCtx, span
}

type TraceToolResultParams struct {
	// ToolDescription is a brief description of the tool's purpose.
	Description   string
	ResponseEvent *session.Event
	Error         error
}

// TraceToolResult records the tool execution events.
func TraceToolResult(span trace.Span, params TraceToolResultParams) {
	recordErrorAndStatus(span, params.Error)

	attributes := []attribute.KeyValue{
		semconv.GenAIOperationNameKey.String(executeToolName),
		semconv.GenAIToolDescriptionKey.String(params.Description),
	}

	toolCallID := "<not specified>"
	toolResponse := "<not specified>"

	if params.ResponseEvent != nil {
		attributes = append(attributes, gcpVertexAgentEventID.String(params.ResponseEvent.ID))
		if params.ResponseEvent.LLMResponse.Content != nil {
			responseParts := params.ResponseEvent.LLMResponse.Content.Parts

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
	}

	attributes = append(attributes, semconv.GenAIToolCallIDKey.String(toolCallID))
	attributes = append(attributes, gcpVertexAgentToolResponseName.String(toolResponse))

	span.SetAttributes(attributes...)
}

func recordErrorAndStatus(span trace.Span, err error) {
	if err == nil {
		return
	}
	span.RecordError(err)
	span.SetStatus(codes.Error, err.Error())
}

// WrapYield wraps a yield function to add tracing of values returned by iterators. Read [iter.Seq2] for more information about yield.
// Limitations:
// * if yield is called multiple times, then the span will be finalized with the values from the last call.
//
// Parameters:
//
//	span: The OpenTelemetry span to be managed.
//	yield: The original yield function `func(T, error) bool`.
//	finalizeSpan: A function `func(trace.Span, T, error)` called just before the span is ended to record final attributes.
//
// Returns:
//
//	wrapped: A wrapped yield function with the same signature as the original.
//	endSpan: A function to be called via `defer` to ensure the span is finalized with capture data and ended.
func WrapYield[T any](span trace.Span, yield func(T, error) bool, finalizeSpan func(trace.Span, T, error)) (wrapped func(T, error) bool, endSpan func()) {
	var val T
	var err error
	wrapped = func(v T, e error) bool {
		val = v
		err = e
		return yield(v, e)
	}
	endSpan = func() {
		finalizeSpan(span, val, err)
		span.End()
	}
	return wrapped, endSpan
}

// StartTrace starts a new span with the given name.
func StartTrace(ctx context.Context, traceName string) (context.Context, trace.Span) {
	return tracer.Start(ctx, traceName)
}

// TraceMergedToolCallsResult records the result of the merged tool calls, including status and tool execution events.
func TraceMergedToolCallsResult(span trace.Span, fnResponseEvent *session.Event, err error) {
	recordErrorAndStatus(span, err)
	attributes := []attribute.KeyValue{
		semconv.GenAIOperationNameKey.String(executeToolName),
		semconv.GenAIToolNameKey.String(mergeToolName),
		semconv.GenAIToolDescriptionKey.String(mergeToolName),
		gcpVertexAgentToolCallArgsName.String("N/A"),
		gcpVertexAgentToolResponseName.String(safeSerialize(fnResponseEvent)),
	}
	if fnResponseEvent != nil {
		attributes = append(attributes, gcpVertexAgentEventID.String(fnResponseEvent.ID))
	}
	span.SetAttributes(attributes...)
}

// TraceLLMCall fills the call_llm event details.
func TraceLLMCall(span trace.Span, sessionID string, llmRequest *model.LLMRequest, event *session.Event) {
	attributes := []attribute.KeyValue{
		attribute.String("gen_ai.system", systemName),
		attribute.String("gen_ai.request.model", llmRequest.Model),
		attribute.String("gcp.vertex.agent.invocation_id", event.InvocationID),
		attribute.String("gcp.vertex.agent.session_id", sessionID),
		attribute.String("gen_ai.conversation.id", sessionID),
		gcpVertexAgentEventID.String(event.ID),
		attribute.String("gcp.vertex.agent.llm_request", safeSerialize(llmRequestToTrace(llmRequest))),
		attribute.String("gcp.vertex.agent.llm_response", safeSerialize(event.LLMResponse)),
	}

	if llmRequest.Config.TopP != nil {
		attributes = append(attributes, attribute.Float64("gen_ai.request.top_p", float64(*llmRequest.Config.TopP)))
	}

	if llmRequest.Config.MaxOutputTokens != 0 {
		attributes = append(attributes, attribute.Int("gen_ai.request.max_tokens", int(llmRequest.Config.MaxOutputTokens)))
	}
	if event.FinishReason != "" {
		attributes = append(attributes, attribute.String("gen_ai.response.finish_reason", string(event.FinishReason)))
	}
	if event.UsageMetadata != nil {
		if event.UsageMetadata.PromptTokenCount > 0 {
			attributes = append(attributes, attribute.Int("gen_ai.response.prompt_token_count", int(event.UsageMetadata.PromptTokenCount)))
		}
		if event.UsageMetadata.CandidatesTokenCount > 0 {
			attributes = append(attributes, attribute.Int("gen_ai.response.candidates_token_count", int(event.UsageMetadata.CandidatesTokenCount)))
		}
		if event.UsageMetadata.CachedContentTokenCount > 0 {
			attributes = append(attributes, attribute.Int("gen_ai.response.cached_content_token_count", int(event.UsageMetadata.CachedContentTokenCount)))
		}
		if event.UsageMetadata.TotalTokenCount > 0 {
			attributes = append(attributes, attribute.Int("gen_ai.response.total_token_count", int(event.UsageMetadata.TotalTokenCount)))
		}
	}

	span.SetAttributes(attributes...)
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
