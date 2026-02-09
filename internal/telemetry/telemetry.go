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
	genAiResponsePromptTokenCount        = attribute.Key("gen_ai.response.prompt_token_count")
	genAiResponseCandidatesTokenCount    = attribute.Key("gen_ai.response.candidates_token_count")
	genAiResponseCachedContentTokenCount = attribute.Key("gen_ai.response.cached_content_token_count")
	genAiResponseTotalTokenCount         = attribute.Key("gen_ai.response.total_token_count")

	gcpVertexAgentLLMRequestName   = attribute.Key("gcp.vertex.agent.llm_request")
	gcpVertexAgentToolCallArgsName = attribute.Key("gcp.vertex.agent.tool_call_args")
	gcpVertexAgentEventID          = attribute.Key("gcp.vertex.agent.event_id")
	gcpVertexAgentToolResponseName = attribute.Key("gcp.vertex.agent.tool_response")
	gcpVertexAgentLLMResponseName  = attribute.Key("gcp.vertex.agent.llm_response")
	gcpVertexAgentInvocationID     = attribute.Key("gcp.vertex.agent.invocation_id")
	gcpVertexAgentSessionID        = attribute.Key("gcp.vertex.agent.session_id")
)

// tracer is the tracer instance for ADK go.
var tracer trace.Tracer = otel.GetTracerProvider().Tracer(
	systemName,
	trace.WithInstrumentationVersion(version.Version),
	trace.WithSchemaURL(semconv.SchemaURL),
)

// StartInvokeAgentParams contains parameters for [StartInvokeAgent].
type StartInvokeAgentParams struct {
	// AgentName is the name of the agent being invoked.
	AgentName string
	// AgentDescription is a brief description of the agent's purpose.
	AgentDescription string
	// SessionID is the unique identifier for the current session.
	SessionID string
}

// StartInvokeAgent starts a new semconv invoke_agent span.
// It returns a new context with the span and the span itself.
func StartInvokeAgent(ctx context.Context, params StartInvokeAgentParams) (context.Context, trace.Span) {
	agentName := params.AgentName
	spanCtx, span := tracer.Start(ctx, fmt.Sprintf("invoke_agent %s", agentName), trace.WithAttributes(
		semconv.GenAIOperationNameInvokeAgent,
		semconv.GenAIAgentDescription(params.AgentDescription),
		semconv.GenAIAgentName(agentName),
		semconv.GenAIConversationID(params.SessionID),
	))

	return spanCtx, span
}

type AfterInvokeAgentParams struct {
	ResponseEvent *session.Event
	Error         error
}

// AfterInvokeAgent records the result of the agent invocation, including status and error.
func AfterInvokeAgent(span trace.Span, params AfterInvokeAgentParams) {
	recordErrorAndStatus(span, params.Error)
}

// StartGenerateContentParams contains parameters for [StartGenerateContent].
type StartGenerateContentParams struct {
	// ModelName is the name of the model being used for generation.
	ModelName string
}

// StartGenerateContent starts a new semconv generate_content span.
func StartGenerateContent(ctx context.Context, params StartGenerateContentParams) (context.Context, trace.Span) {
	modelName := params.ModelName
	spanCtx, span := tracer.Start(ctx, fmt.Sprintf("generate_content %s", modelName), trace.WithAttributes(
		semconv.GenAIOperationNameGenerateContent,
		semconv.GenAIRequestModel(modelName),
		semconv.GenAIUsageInputTokens(123),
	))
	return spanCtx, span
}

type AfterGenerateContentParams struct {
	Response *model.LLMResponse
	Error    error
}

// AfterGenerateContent records the result of the generate_content operation, including token usage and finish reason.
func AfterGenerateContent(span trace.Span, params AfterGenerateContentParams) {
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

// StartExecuteToolParams contains parameters for [StartExecuteTool].
type StartExecuteToolParams struct {
	// ToolName is the name of the tool being executed.
	ToolName string
	// ToolDescription is a brief description of the tool's functionality.
	ToolDescription string
}

// StartExecuteTool starts a new semconv execute_tool span.
func StartExecuteTool(ctx context.Context, params StartExecuteToolParams) (context.Context, trace.Span) {
	toolName := params.ToolName
	spanCtx, span := tracer.Start(ctx, fmt.Sprintf("execute_tool %s", toolName), trace.WithAttributes(
		semconv.GenAIOperationNameExecuteTool,
		semconv.GenAIToolName(toolName),
		semconv.GenAIToolDescription(params.ToolDescription),
	))
	return spanCtx, span
}

type AfterExecuteToolParams struct {
	Name          string
	Description   string
	Args          map[string]any
	ResponseEvent *session.Event
	Error         error
}

// AfterExecuteTool records the tool execution events.
func AfterExecuteTool(span trace.Span, params AfterExecuteToolParams) {
	recordErrorAndStatus(span, params.Error)

	attributes := []attribute.KeyValue{
		semconv.GenAIOperationNameKey.String(executeToolName),
		semconv.GenAIToolNameKey.String(params.Name),
		semconv.GenAIToolDescriptionKey.String(params.Description),
		// TODO: add tool type

		// Setting empty llm request and response (as UI expect these) while not
		// applicable for tool_response.
		gcpVertexAgentLLMRequestName.String("{}"),
		gcpVertexAgentLLMResponseName.String("{}"),
		gcpVertexAgentToolCallArgsName.String(safeSerialize(params.Args)),
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
		span.SetStatus(codes.Ok, "")
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

// --- old tracing ---

// StartTrace starts a new span with the given name.
func StartTrace(ctx context.Context, traceName string) (context.Context, trace.Span) {
	return tracer.Start(ctx, traceName)
}

// AfterMergedToolCalls records the result of the merged tool calls, including status and tool execution events.
func AfterMergedToolCalls(span trace.Span, fnResponseEvent *session.Event, err error) {
	recordErrorAndStatus(span, err)
	attributes := []attribute.KeyValue{
		semconv.GenAIOperationNameKey.String(executeToolName),
		semconv.GenAIToolNameKey.String(mergeToolName),
		semconv.GenAIToolDescriptionKey.String(mergeToolName),
		// Setting empty llm request and response (as UI expect these) while not
		// applicable for tool_response.
		gcpVertexAgentLLMRequestName.String("{}"),
		gcpVertexAgentLLMResponseName.String("{}"),
		gcpVertexAgentToolCallArgsName.String("N/A"),
		gcpVertexAgentToolResponseName.String(safeSerialize(fnResponseEvent)),
	}
	if fnResponseEvent != nil {
		attributes = append(attributes, gcpVertexAgentEventID.String(fnResponseEvent.ID))
	}
	span.SetAttributes(attributes...)
}

// TraceLLMCallParams contains parameters for [TraceLLMCall].
type TraceLLMCallParams struct {
	SessionID  string
	LLMRequest *model.LLMRequest
	Event      *session.Event
	Error      error
}

// TraceLLMCall fills the call_llm event details.
func TraceLLMCall(span trace.Span, params TraceLLMCallParams) {
	recordErrorAndStatus(span, params.Error)
	attributes := []attribute.KeyValue{
		semconv.GenAISystemKey.String(systemName),
		semconv.GenAIRequestModelKey.String(params.LLMRequest.Model),
		gcpVertexAgentSessionID.String(params.SessionID),
		semconv.GenAIConversationIDKey.String(params.SessionID),
		gcpVertexAgentLLMRequestName.String(safeSerialize(llmRequestToTrace(params.LLMRequest))),
	}

	if params.Event != nil {
		attributes = append(attributes,
			gcpVertexAgentInvocationID.String(params.Event.InvocationID),
			gcpVertexAgentEventID.String(params.Event.ID),
			gcpVertexAgentLLMResponseName.String(safeSerialize(params.Event.LLMResponse)),
		)
		if params.Event.FinishReason != "" {
			attributes = append(attributes, semconv.GenAIResponseFinishReasonsKey.String(string(params.Event.FinishReason)))
		}
		if params.Event.UsageMetadata != nil {
			if params.Event.UsageMetadata.PromptTokenCount > 0 {
				attributes = append(attributes, genAiResponsePromptTokenCount.Int(int(params.Event.UsageMetadata.PromptTokenCount)))
			}
			if params.Event.UsageMetadata.CandidatesTokenCount > 0 {
				attributes = append(attributes, genAiResponseCandidatesTokenCount.Int(int(params.Event.UsageMetadata.CandidatesTokenCount)))
			}
			if params.Event.UsageMetadata.CachedContentTokenCount > 0 {
				attributes = append(attributes, genAiResponseCachedContentTokenCount.Int(int(params.Event.UsageMetadata.CachedContentTokenCount)))
			}
			if params.Event.UsageMetadata.TotalTokenCount > 0 {
				attributes = append(attributes, genAiResponseTotalTokenCount.Int(int(params.Event.UsageMetadata.TotalTokenCount)))
			}
		}
	} else {
		attributes = append(attributes, gcpVertexAgentLLMResponseName.String("{}"))
	}

	if params.LLMRequest.Config != nil {
		if params.LLMRequest.Config.TopP != nil {
			attributes = append(attributes, semconv.GenAIRequestTopPKey.Float64(float64(*params.LLMRequest.Config.TopP)))
		}

		if params.LLMRequest.Config.MaxOutputTokens != 0 {
			attributes = append(attributes, semconv.GenAIRequestMaxTokensKey.Int(int(params.LLMRequest.Config.MaxOutputTokens)))
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
