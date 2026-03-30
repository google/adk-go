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

type mockAgent struct{}

func (a *mockAgent) Name() string {
	return "test-agent"
}

func (a *mockAgent) Description() string {
	return "test-agent-description"
}

func TestInvokeAgent(t *testing.T) {
	sessionID := "test-session"
	invocationID := "test-invocation-id"
	agent := &mockAgent{}
	tests := []struct {
		name         string
		resultParams TraceAgentResultParams
		wantName     string
		wantStatus   codes.Code
		wantAttrs    map[attribute.Key]string
	}{
		{
			name: "Success",
			resultParams: TraceAgentResultParams{
				ResponseEvent: session.NewEvent("test-invocation-id"),
			},
			wantName:   "invoke_agent test-agent",
			wantStatus: codes.Unset,
			wantAttrs: map[attribute.Key]string{
				semconv.GenAIOperationNameKey:    "invoke_agent",
				semconv.GenAIAgentNameKey:        "test-agent",
				semconv.GenAIAgentDescriptionKey: "test-agent-description",
				semconv.GenAIConversationIDKey:   "test-session",
				gcpVertexAgentInvocationID:       invocationID,
			},
		},
		{
			name: "Error",
			resultParams: TraceAgentResultParams{
				Error: errTest,
			},
			wantName:   "invoke_agent test-agent",
			wantStatus: codes.Error,
			wantAttrs: map[attribute.Key]string{
				semconv.GenAIOperationNameKey:    "invoke_agent",
				semconv.GenAIAgentNameKey:        "test-agent",
				semconv.GenAIAgentDescriptionKey: "test-agent-description",
				semconv.GenAIConversationIDKey:   "test-session",
				gcpVertexAgentInvocationID:       invocationID,
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			exporter := setupTestTracer(t)
			ctx := t.Context()

			_, span := StartInvokeAgentSpan(ctx, agent, sessionID, invocationID)
			TraceAgentResult(span, tc.resultParams)
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
			if tc.resultParams.Error != nil {
				if gotSpan.Status.Description != tc.resultParams.Error.Error() {
					t.Errorf("expected status description %q, got %q", tc.resultParams.Error.Error(), gotSpan.Status.Description)
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
	invocationID := "test-invocation-id"
	tests := []struct {
		name         string
		startParams  StartGenerateContentSpanParams
		resultParams TraceGenerateContentResultParams
		wantName     string
		wantStatus   codes.Code
		wantAttrs    map[attribute.Key]string
	}{
		{
			name: "Success",
			startParams: StartGenerateContentSpanParams{
				ModelName:    "test-model",
				InvocationID: invocationID,
			},
			resultParams: TraceGenerateContentResultParams{
				Response: &model.LLMResponse{
					UsageMetadata: &genai.GenerateContentResponseUsageMetadata{
						PromptTokenCount:     10,
						CandidatesTokenCount: 20,
					},
					FinishReason: genai.FinishReasonStop,
				},
			},
			wantName:   "generate_content test-model",
			wantStatus: codes.Unset,
			wantAttrs: map[attribute.Key]string{
				semconv.GenAIOperationNameKey:         "generate_content",
				semconv.GenAIRequestModelKey:          "test-model",
				semconv.GenAIUsageInputTokensKey:      "10",
				semconv.GenAIUsageOutputTokensKey:     "20",
				semconv.GenAIResponseFinishReasonsKey: "[\"STOP\"]",
				gcpVertexAgentInvocationID:            invocationID,
			},
		},
		{
			name: "Error",
			startParams: StartGenerateContentSpanParams{
				ModelName:    "test-model",
				InvocationID: invocationID,
			},
			resultParams: TraceGenerateContentResultParams{
				Error: errTest,
			},
			wantName:   "generate_content test-model",
			wantStatus: codes.Error,
			wantAttrs: map[attribute.Key]string{
				semconv.GenAIOperationNameKey: "generate_content",
				semconv.GenAIRequestModelKey:  "test-model",
				gcpVertexAgentInvocationID:    invocationID,
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			exporter := setupTestTracer(t)
			ctx := t.Context()

			_, span := StartGenerateContentSpan(ctx, tc.startParams)
			TraceGenerateContentResult(span, tc.resultParams)
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
			if tc.resultParams.Error != nil {
				if gotSpan.Status.Description != tc.resultParams.Error.Error() {
					t.Errorf("expected status description %q, got %q", tc.resultParams.Error.Error(), gotSpan.Status.Description)
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
		name         string
		startParams  StartExecuteToolSpanParams
		resultParams TraceToolResultParams
		wantName     string
		wantStatus   codes.Code
		wantAttrs    map[attribute.Key]string
	}{
		{
			name: "Success",
			startParams: StartExecuteToolSpanParams{
				ToolName: "test-tool",
				Args:     map[string]any{"arg": "val"},
			},
			resultParams: TraceToolResultParams{
				Description:   "tool-description",
				ResponseEvent: &session.Event{ID: "test-event"},
			},
			wantName:   "execute_tool test-tool",
			wantStatus: codes.Unset,
			wantAttrs: map[attribute.Key]string{
				semconv.GenAIOperationNameKey:   "execute_tool",
				semconv.GenAIToolNameKey:        "test-tool",
				semconv.GenAIToolDescriptionKey: "tool-description",
			},
		},
		{
			name: "Error",
			startParams: StartExecuteToolSpanParams{
				ToolName: "test-tool",
			},
			resultParams: TraceToolResultParams{
				Description: "tool-description",
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

			_, span := StartExecuteToolSpan(ctx, tc.startParams)
			TraceToolResult(span, tc.resultParams)
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
			if tc.resultParams.Error != nil {
				if gotSpan.Status.Description != tc.resultParams.Error.Error() {
					t.Errorf("expected status description %q, got %q", tc.resultParams.Error.Error(), gotSpan.Status.Description)
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
