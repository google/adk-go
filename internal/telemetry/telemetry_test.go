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
	"errors"
	"fmt"
	"testing"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	semconv "go.opentelemetry.io/otel/semconv/v1.36.0"
	"go.opentelemetry.io/otel/trace"
	"go.opentelemetry.io/otel/trace/noop"
	"google.golang.org/genai"

	"google.golang.org/adk/model"
	"google.golang.org/adk/session"
)

func TestWrapYield(t *testing.T) {
	t.Parallel()

	var finalized bool
	finalizeFn := func(span trace.Span, val string, err error) {
		if val != "test" {
			t.Errorf("unexpected value in finalizeFn: got %q, want %q", val, "test")
		}
		if !errors.Is(err, errTest) {
			t.Errorf("unexpected error in finalizeFn: got %v, want %v", err, errTest)
		}
		finalized = true
	}

	yieldFn := func(val string, err error) bool {
		if val != "test" {
			t.Errorf("unexpected value in yieldFn: got %q, want %q", val, "test")
		}
		if !errors.Is(err, errTest) {
			t.Errorf("unexpected error in yieldFn: got %v, want %v", err, errTest)
		}
		return true
	}

	_, span := noop.NewTracerProvider().Tracer("test").Start(context.Background(), "test")
	wrappedYield, endSpan := WrapYield(span, yieldFn, finalizeFn)

	if !wrappedYield("test", errTest) {
		t.Error("wrappedYield should have returned true")
	}

	endSpan()

	if !finalized {
		t.Error("finalizeFn was not called")
	}
}

func TestWrapYield_MultipleCalls(t *testing.T) {
	t.Parallel()

	var finalized bool
	finalizeFn := func(span trace.Span, val string, err error) {
		if val != "last" {
			t.Errorf("unexpected value in finalizeFn: got %q, want %q", val, "last")
		}
		if !errors.Is(err, errTest) {
			t.Errorf("unexpected error in finalizeFn: got %v, want %v", err, errTest)
		}
		finalized = true
	}

	yieldFn := func(val string, err error) bool {
		return true
	}

	_, span := noop.NewTracerProvider().Tracer("test").Start(context.Background(), "test")
	wrappedYield, endSpan := WrapYield(span, yieldFn, finalizeFn)

	wrappedYield("first", nil)
	wrappedYield("", fmt.Errorf("some error"))
	wrappedYield("last", errTest)

	endSpan()

	if !finalized {
		t.Error("finalizeFn was not called")
	}
}

var errTest = errors.New("test error")

func TestInvokeAgent(t *testing.T) {
	tests := []struct {
		name        string
		startParams StartInvokeAgentParams
		afterParams AfterInvokeAgentParams
		wantName    string
		wantStatus  codes.Code
		wantAttrs   map[attribute.Key]string
	}{
		{
			name: "Success",
			startParams: StartInvokeAgentParams{
				AgentName:        "test-agent",
				AgentDescription: "test-description",
				SessionID:        "test-session",
			},
			afterParams: AfterInvokeAgentParams{
				ResponseEvent: session.NewEvent("test-invocation-id"),
			},
			wantName:   "invoke_agent test-agent",
			wantStatus: codes.Ok,
			wantAttrs: map[attribute.Key]string{
				semconv.GenAIOperationNameKey:    "invoke_agent",
				semconv.GenAIAgentNameKey:        "test-agent",
				semconv.GenAIAgentDescriptionKey: "test-description",
				semconv.GenAIConversationIDKey:   "test-session",
			},
		},
		{
			name: "Error",
			startParams: StartInvokeAgentParams{
				AgentName:        "test-agent",
				AgentDescription: "test-description",
				SessionID:        "test-session",
			},
			afterParams: AfterInvokeAgentParams{
				Error: errTest,
			},
			wantName:   "invoke_agent test-agent",
			wantStatus: codes.Error,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			exporter := setupTestTracer(t)
			ctx := t.Context()

			_, span := StartInvokeAgent(ctx, tc.startParams)
			AfterInvokeAgent(span, tc.afterParams)
			span.End()

			spans := exporter.GetSpans()
			if len(spans) != 1 {
				t.Fatalf("expected 1 span, got %d", len(spans))
			}
			gotSpan := spans[0]

			if gotSpan.Name != tc.wantName {
				t.Errorf("expected span name %q, got %q", tc.wantName, gotSpan.Name)
			}
			if gotSpan.Status.Code != tc.wantStatus {
				t.Errorf("expected status %v, got %v", tc.wantStatus, gotSpan.Status.Code)
			}
			if tc.afterParams.Error != nil {
				if gotSpan.Status.Description != tc.afterParams.Error.Error() {
					t.Errorf("expected status description %q, got %q", tc.afterParams.Error.Error(), gotSpan.Status.Description)
				}
			}

			if tc.wantAttrs != nil {
				gotAttrs := attributesToMap(gotSpan.Attributes)
				for k, v := range tc.wantAttrs {
					if gotAttrs[k] != v {
						t.Errorf("attribute %q: got %q, want %q", k, gotAttrs[k], v)
					}
				}
			}
		})
	}
}

func TestGenerateContent(t *testing.T) {
	tests := []struct {
		name        string
		startParams StartGenerateContentParams
		afterParams AfterGenerateContentParams
		wantName    string
		wantStatus  codes.Code
		wantAttrs   map[attribute.Key]string
	}{
		{
			name: "Success",
			startParams: StartGenerateContentParams{
				ModelName: "test-model",
			},
			afterParams: AfterGenerateContentParams{
				Response: &model.LLMResponse{
					UsageMetadata: &genai.GenerateContentResponseUsageMetadata{
						PromptTokenCount: 10,
						TotalTokenCount:  20,
					},
					FinishReason: genai.FinishReasonStop,
				},
			},
			wantName:   "generate_content test-model",
			wantStatus: codes.Ok,
			wantAttrs: map[attribute.Key]string{
				semconv.GenAIOperationNameKey:         "generate_content",
				semconv.GenAIRequestModelKey:          "test-model",
				semconv.GenAIUsageInputTokensKey:      "10",
				semconv.GenAIUsageOutputTokensKey:     "20",
				semconv.GenAIResponseFinishReasonsKey: "[\"STOP\"]",
			},
		},
		{
			name: "Error",
			startParams: StartGenerateContentParams{
				ModelName: "test-model",
			},
			afterParams: AfterGenerateContentParams{
				Error: errTest,
			},
			wantName:   "generate_content test-model",
			wantStatus: codes.Error,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			exporter := setupTestTracer(t)
			ctx := t.Context()

			_, span := StartGenerateContent(ctx, tc.startParams)
			AfterGenerateContent(span, tc.afterParams)
			span.End()

			spans := exporter.GetSpans()
			if len(spans) != 1 {
				t.Fatalf("expected 1 span, got %d", len(spans))
			}
			gotSpan := spans[0]

			if gotSpan.Name != tc.wantName {
				t.Errorf("expected span name %q, got %q", tc.wantName, gotSpan.Name)
			}
			if gotSpan.Status.Code != tc.wantStatus {
				t.Errorf("expected status %v, got %v", tc.wantStatus, gotSpan.Status.Code)
			}
			if tc.afterParams.Error != nil {
				if gotSpan.Status.Description != tc.afterParams.Error.Error() {
					t.Errorf("expected status description %q, got %q", tc.afterParams.Error.Error(), gotSpan.Status.Description)
				}
			}

			if tc.wantAttrs != nil {
				gotAttrs := attributesToMap(gotSpan.Attributes)
				for k, v := range tc.wantAttrs {
					if gotAttrs[k] != v {
						t.Errorf("attribute %q: got %q, want %q", k, gotAttrs[k], v)
					}
				}
			}
		})
	}
}

func TestExecuteTool(t *testing.T) {
	tests := []struct {
		name        string
		startParams StartExecuteToolParams
		afterParams AfterExecuteToolParams
		wantName    string
		wantStatus  codes.Code
		wantAttrs   map[attribute.Key]string
	}{
		{
			name: "Success",
			startParams: StartExecuteToolParams{
				ToolName:        "test-tool",
				ToolDescription: "test-tool-desc",
			},
			afterParams: AfterExecuteToolParams{
				Name:          "test-tool",
				Description:   "tool-description",
				Args:          map[string]any{"arg": "val"},
				ResponseEvent: &session.Event{ID: "test-event"},
			},
			wantName:   "execute_tool test-tool",
			wantStatus: codes.Ok,
			wantAttrs: map[attribute.Key]string{
				semconv.GenAIOperationNameKey:   "execute_tool",
				semconv.GenAIToolNameKey:        "test-tool",
				semconv.GenAIToolDescriptionKey: "tool-description",
			},
		},
		{
			name: "Error",
			startParams: StartExecuteToolParams{
				ToolName:        "test-tool",
				ToolDescription: "test-tool-desc",
			},
			afterParams: AfterExecuteToolParams{
				Name:        "test-tool",
				Description: "test-description",
				Error:       errTest,
			},
			wantName:   "execute_tool test-tool",
			wantStatus: codes.Error,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			exporter := setupTestTracer(t)
			ctx := t.Context()

			_, span := StartExecuteTool(ctx, tc.startParams)
			AfterExecuteTool(span, tc.afterParams)
			span.End()

			spans := exporter.GetSpans()
			if len(spans) != 1 {
				t.Fatalf("expected 1 span, got %d", len(spans))
			}
			gotSpan := spans[0]

			if gotSpan.Name != tc.wantName {
				t.Errorf("expected span name %q, got %q", tc.wantName, gotSpan.Name)
			}
			if gotSpan.Status.Code != tc.wantStatus {
				t.Errorf("expected status %v, got %v", tc.wantStatus, gotSpan.Status.Code)
			}
			if tc.afterParams.Error != nil {
				if gotSpan.Status.Description != tc.afterParams.Error.Error() {
					t.Errorf("expected status description %q, got %q", tc.afterParams.Error.Error(), gotSpan.Status.Description)
				}
			}

			if tc.wantAttrs != nil {
				gotAttrs := attributesToMap(gotSpan.Attributes)
				for k, v := range tc.wantAttrs {
					if gotAttrs[k] != v {
						t.Errorf("attribute %q: got %q, want %q", k, gotAttrs[k], v)
					}
				}
			}
		})
	}
}

func TestTraceLLMCall(t *testing.T) {
	tests := []struct {
		name       string
		params     TraceLLMCallParams
		wantAttrs  map[attribute.Key]string
		wantStatus codes.Code
	}{
		{
			name: "Success",
			params: TraceLLMCallParams{
				SessionID: "test-session-id",
				LLMRequest: &model.LLMRequest{
					Model: "test-model",
					Config: &genai.GenerateContentConfig{
						TopP:            float32Ptr(0.9),
						MaxOutputTokens: 100,
					},
					Contents: []*genai.Content{
						{Role: "user", Parts: []*genai.Part{{Text: "hello"}}},
					},
				},
				Event: &session.Event{
					ID:           "test-event-id",
					InvocationID: "test-invocation-id",
					LLMResponse: model.LLMResponse{
						FinishReason: genai.FinishReasonStop,
						UsageMetadata: &genai.GenerateContentResponseUsageMetadata{
							PromptTokenCount:     5,
							CandidatesTokenCount: 10,
							TotalTokenCount:      15,
						},
						Content: &genai.Content{
							Parts: []*genai.Part{{Text: "world"}},
						},
					},
				},
			},
			wantAttrs: map[attribute.Key]string{
				semconv.GenAISystemKey:                systemName,
				semconv.GenAIRequestModelKey:          "test-model",
				gcpVertexAgentInvocationID:            "test-invocation-id",
				gcpVertexAgentSessionID:               "test-session-id",
				semconv.GenAIConversationIDKey:        "test-session-id",
				gcpVertexAgentEventID:                 "test-event-id",
				semconv.GenAIRequestTopPKey:           "0.8999999761581421",
				semconv.GenAIRequestMaxTokensKey:      "100",
				semconv.GenAIResponseFinishReasonsKey: "STOP",
				genAiResponsePromptTokenCount:         "5",
				genAiResponseCandidatesTokenCount:     "10",
				genAiResponseTotalTokenCount:          "15",
			},
			wantStatus: codes.Ok,
		},
		{
			name: "Empty",
			params: TraceLLMCallParams{
				SessionID: "test-session-id",
				LLMRequest: &model.LLMRequest{
					Model: "test-model",
				},
				Event: &session.Event{},
			},
			wantAttrs: map[attribute.Key]string{
				semconv.GenAISystemKey:       systemName,
				semconv.GenAIRequestModelKey: "test-model",
			},
			wantStatus: codes.Ok,
		},
		{
			name: "Error",
			params: TraceLLMCallParams{
				SessionID: "test-session-id",
				LLMRequest: &model.LLMRequest{
					Model: "test-model",
				},
				Error: errTest,
			},
			wantAttrs: map[attribute.Key]string{
				semconv.GenAISystemKey:       systemName,
				semconv.GenAIRequestModelKey: "test-model",
			},
			wantStatus: codes.Error,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			exporter := setupTestTracer(t)
			ctx := t.Context()
			_, span := tracer.Start(ctx, "test-span")

			TraceLLMCall(span, tc.params)
			span.End()

			spans := exporter.GetSpans()
			if len(spans) != 1 {
				t.Fatalf("expected 1 span, got %d", len(spans))
			}
			gotSpan := spans[0]

			if gotSpan.Status.Code != tc.wantStatus {
				t.Errorf("expected status %v, got %v", tc.wantStatus, gotSpan.Status.Code)
			}

			if tc.params.Error != nil {
				if gotSpan.Status.Description != tc.params.Error.Error() {
					t.Errorf("expected status description %q, got %q", tc.params.Error.Error(), gotSpan.Status.Description)
				}
			}

			attrs := attributesToMap(gotSpan.Attributes)
			for k, v := range tc.wantAttrs {
				if attrs[k] != v {
					t.Errorf("attribute %q: got %q, want %q", k, attrs[k], v)
				}
			}

			// Check JSON serialized fields existence/validity
			if attrs[gcpVertexAgentLLMRequestName] == "" || attrs[gcpVertexAgentLLMRequestName] == "<not serializable>" {
				t.Errorf("invalid request serialization: %s", attrs[gcpVertexAgentLLMRequestName])
			}
			if attrs[gcpVertexAgentLLMResponseName] == "" || attrs[gcpVertexAgentLLMResponseName] == "<not serializable>" {
				t.Errorf("invalid response serialization: %s", attrs[gcpVertexAgentLLMResponseName])
			}
		})
	}
}

func float32Ptr(f float32) *float32 {
	return &f
}

func setupTestTracer(t *testing.T) *tracetest.InMemoryExporter {
	t.Helper()
	exporter := tracetest.NewInMemoryExporter()
	tp := sdktrace.NewTracerProvider(
		sdktrace.WithSyncer(exporter),
	)

	originalTracer := tracer
	tracer = tp.Tracer("test")
	t.Cleanup(func() {
		tracer = originalTracer
	})
	return exporter
}

func attributesToMap(attrs []attribute.KeyValue) map[attribute.Key]string {
	m := make(map[attribute.Key]string, len(attrs))
	for _, attr := range attrs {
		m[attr.Key] = attr.Value.Emit()
	}
	return m
}
