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
// WARNING: telemetry provided by ADK (internal/telemetry package) may change (e.g. attributes and their names)
// because we're in process to standardize and unify telemetry across all ADKs.
package telemetry

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strconv"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	semconv "go.opentelemetry.io/otel/semconv/v1.36.0"
	"go.opentelemetry.io/otel/trace"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/internal/version"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
)

const (
	systemName = "gcp.vertex.agent"

	executeToolName = "execute_tool"
	mergeToolName   = "(merged tools)"
)

var (
	gcpVertexAgentToolCallArgsName  = attribute.Key("gcp.vertex.agent.tool_call_args")
	gcpVertexAgentEventID           = attribute.Key("gcp.vertex.agent.event_id")
	gcpVertexAgentToolResponseName  = attribute.Key("gcp.vertex.agent.tool_response")
	gcpVertexAgentInvocationID      = attribute.Key("gcp.vertex.agent.invocation_id")
	genAIUsageCacheReadInputTokens  = attribute.Key("gen_ai.usage.cache_read.input_tokens")
	genAIUsageReasoningOutputTokens = attribute.Key("gen_ai.usage.reasoning.output_tokens")
	genAIToolCallArguments          = attribute.Key("gen_ai.tool.call.arguments")
	genAIToolCallResult             = attribute.Key("gen_ai.tool.call.result")
)

var tracerProvider = otel.GetTracerProvider()

// tracer is looked up per use, since the schema URL follows the env.
func tracer() trace.Tracer {
	return tracerProvider.Tracer(
		systemName,
		trace.WithInstrumentationVersion(version.Version),
		trace.WithSchemaURL(schemaURL()),
	)
}

func schemaURL() string {
	if useLegacySchema() {
		return semconv.SchemaURL
	}
	// TODO(maxind): use semconv/v1.44.0 SchemaURL once otel-go ships it.
	return "https://opentelemetry.io/schemas/1.44.0"
}

// OverrideTracerForTesting makes tracing use tp until the test ends.
func OverrideTracerForTesting(t interface{ Cleanup(func()) }, tp trace.TracerProvider) {
	original := tracerProvider
	tracerProvider = tp
	t.Cleanup(func() { tracerProvider = original })
}

type TraceAgentResultParams struct {
	ResponseEvent *session.Event
	Error         error
}

// TraceAgentResult records the result of the agent invocation, including status and error.
func TraceAgentResult(span trace.Span, params TraceAgentResultParams) {
	recordErrorAndStatus(span, params.Error)
}

type GenerateContentParams struct {
	// ModelName is the name of the model being used for generation.
	ModelName string
	// InvocationID is the ID of the invocation.
	InvocationID string
	// Request is required.
	Request *model.LLMRequest
	Backend genai.Backend
}

// startGenerateContentSpan starts a new semconv generate_content span.
func startGenerateContentSpan(ctx context.Context, params GenerateContentParams) (context.Context, trace.Span) {
	modelName := params.ModelName
	attrs := []attribute.KeyValue{
		// Used by adk-web, can be removed once it reads the invocation id from invoke_agent span.
		gcpVertexAgentInvocationID.String(params.InvocationID),
		semconv.GenAIOperationNameGenerateContent,
		semconv.GenAIRequestModel(modelName),
	}
	if provider, ok := providerName(params.ModelName, params.Backend); ok && !useLegacySchema() {
		attrs = append(attrs, provider)
	}
	spanCtx, span := tracer().Start(ctx, fmt.Sprintf("generate_content %s", modelName), trace.WithAttributes(attrs...))
	// After the sampling decision, not before: converting a whole conversation
	// costs milliseconds, and on a span the sampler drops it buys nothing. The
	// conventions list the attributes worth having at span creation for
	// sampling, and content is not among them. Still before the model call, so
	// the prompt is on the span even when the call fails and no response is
	// ever traced.
	if span.IsRecording() && captureContentOnSpans() {
		span.SetAttributes(spanContentAttributes(requestContent(params.Request))...)
	}
	return spanCtx, span
}

type generateContentResult struct {
	Response *model.LLMResponse
	EventID  string
	Error    error
}

// traceGenerateContentResult records the result of the generate_content operation, including token usage and finish reason.
func traceGenerateContentResult(span trace.Span, params generateContentResult) {
	recordErrorAndStatus(span, params.Error)
	span.SetAttributes(generateContentResultAttributes(params)...)
	if captureContentOnSpans() {
		span.SetAttributes(spanContentAttributes(responseContent(params.Response, params.Error))...)
	}
}

// generateContentResultAttributes returns content-free result attributes.
func generateContentResultAttributes(params generateContentResult) []attribute.KeyValue {
	var attrs []attribute.KeyValue
	// Record a finish reason when the call produced a response or an error; a
	// nil response and nil error means there is no result to describe.
	if params.Response != nil || params.Error != nil {
		attrs = append(attrs, semconv.GenAIResponseFinishReasons(schemaFinishReason(params.Response, params.Error)))
	}
	if params.Response == nil {
		return attrs
	}
	attrs = append(attrs, gcpVertexAgentEventID.String(params.EventID))
	if u := params.Response.UsageMetadata; u != nil {
		attrs = append(attrs,
			// Tool-use prompt tokens are reported separately from PromptTokenCount and
			// are billed as input, so they belong in gen_ai.usage.input_tokens. This
			// matches the semantic-conventions reference implementation for google-genai:
			// https://github.com/open-telemetry/semantic-conventions-genai/blob/main/reference/scenarios/google-genai/scenario.py
			semconv.GenAIUsageInputTokens(int(u.PromptTokenCount+u.ToolUsePromptTokenCount)),
			// According to OpenTelemetry Semantic Conventions:
			// https://github.com/open-telemetry/semantic-conventions/blob/v1.41.0/docs/registry/attributes/gen-ai.md
			// gen_ai.usage.reasoning.output_tokens (ThoughtsTokenCount) SHOULD be included in gen_ai.usage.output_tokens.
			semconv.GenAIUsageOutputTokens(int(u.CandidatesTokenCount+u.ThoughtsTokenCount)),
			genAIUsageCacheReadInputTokens.Int(int(u.CachedContentTokenCount)),
			genAIUsageReasoningOutputTokens.Int(int(u.ThoughtsTokenCount)),
		)
	}
	return attrs
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
	attrs := []attribute.KeyValue{
		semconv.GenAIOperationNameExecuteTool,
		semconv.GenAIToolName(toolName),
	}
	legacy := useLegacySchema()
	if legacy {
		attrs = append(attrs, gcpVertexAgentToolCallArgsName.String(safeSerialize(params.Args)))
	}
	spanCtx, span := tracer().Start(ctx, fmt.Sprintf("execute_tool %s", toolName), trace.WithAttributes(attrs...))
	if !legacy && params.Args != nil && span.IsRecording() && captureContentOnSpans() {
		span.SetAttributes(spanContentAttributes([]contentAttr{{genAIToolCallArguments, toolPayload(params.Args)}})...)
	}
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

	var functionResponse *genai.FunctionResponse
	if params.ResponseEvent != nil {
		attributes = append(attributes, gcpVertexAgentEventID.String(params.ResponseEvent.ID))
		if c := params.ResponseEvent.LLMResponse.Content; c != nil && len(c.Parts) > 0 {
			functionResponse = c.Parts[0].FunctionResponse
		}
	}

	if !useLegacySchema() {
		if functionResponse != nil {
			if functionResponse.ID != "" {
				attributes = append(attributes, semconv.GenAIToolCallID(functionResponse.ID))
			}
			if captureContentOnSpans() && functionResponse.Response != nil && params.Error == nil {
				attributes = append(attributes, spanContentAttributes([]contentAttr{{genAIToolCallResult, toolPayload(functionResponse.Response)}})...)
			}
		}
		span.SetAttributes(attributes...)
		return
	}

	toolCallID := "<not specified>"
	toolResponse := "<not specified>"
	if functionResponse != nil {
		if functionResponse.ID != "" {
			toolCallID = functionResponse.ID
		}
		if functionResponse.Response != nil {
			toolResponse = safeSerialize(functionResponse.Response)
		}
	}
	attributes = append(attributes,
		semconv.GenAIToolCallID(toolCallID),
		gcpVertexAgentToolResponseName.String(toolResponse))
	span.SetAttributes(attributes...)
}

func recordErrorAndStatus(span trace.Span, err error) {
	if err == nil {
		return
	}
	span.RecordError(err)
	span.SetStatus(codes.Error, err.Error())
	if !useLegacySchema() {
		span.SetAttributes(errorType(err))
	}
}

// errorType mirrors ADK Python: https://github.com/google/adk-python/blob/ed030cbf431ef6c12a2b3a75dcc7aff08378cb82/src/google/adk/telemetry/tracing.py#L183-L198
func errorType(err error) attribute.KeyValue {
	var apiErr genai.APIError
	if errors.As(err, &apiErr) {
		return semconv.ErrorTypeKey.String(strconv.Itoa(apiErr.Code))
	}
	return semconv.ErrorType(err)
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
	return tracer().Start(ctx, traceName)
}

// TraceMergedToolCallsResult records the result of the merged tool calls, including status and tool execution events.
func TraceMergedToolCallsResult(span trace.Span, fnResponseEvent *session.Event, err error) {
	recordErrorAndStatus(span, err)
	attributes := []attribute.KeyValue{
		semconv.GenAIOperationNameKey.String(executeToolName),
		semconv.GenAIToolNameKey.String(mergeToolName),
		semconv.GenAIToolDescriptionKey.String(mergeToolName),
	}
	if useLegacySchema() {
		attributes = append(attributes,
			gcpVertexAgentToolCallArgsName.String("N/A"),
			gcpVertexAgentToolResponseName.String(safeSerialize(fnResponseEvent)))
	}
	if fnResponseEvent != nil {
		attributes = append(attributes, gcpVertexAgentEventID.String(fnResponseEvent.ID))
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
