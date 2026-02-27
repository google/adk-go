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

package llminternal

import (
	"context"
	"errors"
	"iter"
	"sync"
	"testing"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/log/global"
	sdklog "go.opentelemetry.io/otel/sdk/log"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	semconv "go.opentelemetry.io/otel/semconv/v1.36.0"
	"go.opentelemetry.io/otel/trace"
	"google.golang.org/genai"

	icontext "google.golang.org/adk/internal/context"
	"google.golang.org/adk/model"
)

type mockModelForTest struct {
	name            string
	generateContent func(ctx context.Context, req *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error]
}

var _ model.LLM = (*mockModelForTest)(nil)

func (m *mockModelForTest) Name() string {
	return m.name
}

func (m *mockModelForTest) GenerateContent(ctx context.Context, req *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error] {
	if m.generateContent != nil {
		return m.generateContent(ctx, req, stream)
	}
	return func(yield func(*model.LLMResponse, error) bool) {}
}

func (m *mockModelForTest) Backend() genai.Backend {
	return genai.BackendGeminiAPI
}

var (
	testExporter *tracetest.InMemoryExporter
	initTracer   sync.Once
)

func TestGenerateContentTracing(t *testing.T) {
	setupTestTracer(t)

	modelMock := &mockModelForTest{
		name: "test-model",
		generateContent: func(ctx context.Context, req *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error] {
			return func(yield func(*model.LLMResponse, error) bool) {
				// Yield partial response.
				if !yield(&model.LLMResponse{
					UsageMetadata: &genai.GenerateContentResponseUsageMetadata{
						PromptTokenCount:     1,
						CandidatesTokenCount: 2,
					},
					Partial: true,
				}, nil) {
					return
				}
				// Verify span NOT ended.
				gotSpans := testExporter.GetSpans()
				if len(gotSpans) != 0 {
					t.Errorf("expected 0 spans after partial response, got %d", len(gotSpans))
				}

				// Yield final response.
				if !yield(&model.LLMResponse{
					UsageMetadata: &genai.GenerateContentResponseUsageMetadata{
						PromptTokenCount:     10,
						CandidatesTokenCount: 20,
					},
					Partial: false,
				}, nil) {
					return
				}
				// Verify span ENDED.
				gotSpans = testExporter.GetSpans()
				if len(gotSpans) != 1 {
					t.Errorf("expected 1 span after final response, got %d", len(gotSpans))
				}

				// Yield final response - should not panic.
				if !yield(&model.LLMResponse{
					UsageMetadata: &genai.GenerateContentResponseUsageMetadata{
						PromptTokenCount:     100,
						CandidatesTokenCount: 200,
					},
					Partial: false,
				}, nil) {
					return
				}
				// Verify there is no new span.
				gotSpans = testExporter.GetSpans()
				if len(gotSpans) != 1 {
					t.Errorf("expected 1 span after final response, got %d", len(gotSpans))
				}
			}
		},
	}

	ctx := icontext.NewInvocationContext(context.Background(), icontext.InvocationContextParams{})

	for range generateContent(ctx, modelMock, &model.LLMRequest{}, true) {
	}

	// Verify that there is only single span.
	gotSpans := testExporter.GetSpans()
	if len(gotSpans) != 1 {
		t.Fatalf("expected 1 span, got %d", len(gotSpans))
	}
	gotSpan := gotSpans[0]

	if gotSpan.Name != "generate_content test-model" {
		t.Errorf("expected span name %q, got %q", "generate_content test-model", gotSpan.Name)
	}

	// Verify span attributes.
	attrs := make(map[attribute.Key]string)
	for _, kv := range gotSpan.Attributes {
		attrs[kv.Key] = kv.Value.Emit()
	}

	if val := attrs[semconv.GenAIUsageInputTokensKey]; val != "10" {
		t.Errorf("expected input tokens 10, got %s", val)
	}
	if val := attrs[semconv.GenAIUsageOutputTokensKey]; val != "20" {
		t.Errorf("expected output tokens 20, got %s", val)
	}
	if val := attrs["gcp.vertexai.invocation_id"]; val != "" {
		t.Errorf("expected invocation id, got %s", val)
	}
}

func TestGenerateContentTracingNoFinalResponse(t *testing.T) {
	setupTestTracer(t)

	modelMock := &mockModelForTest{
		name: "test-model",
		generateContent: func(ctx context.Context, req *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error] {
			return func(yield func(*model.LLMResponse, error) bool) {
				// Yield partial response.
				if !yield(&model.LLMResponse{
					UsageMetadata: &genai.GenerateContentResponseUsageMetadata{
						PromptTokenCount:     10,
						CandidatesTokenCount: 20,
					},
					Partial: true,
				}, nil) {
					return
				}
				// Verify span NOT ended.
				gotSpans := testExporter.GetSpans()
				if len(gotSpans) != 0 {
					t.Errorf("expected 0 spans after partial response, got %d", len(gotSpans))
				}
			}
		},
	}

	ctx := icontext.NewInvocationContext(context.Background(), icontext.InvocationContextParams{})

	for range generateContent(ctx, modelMock, &model.LLMRequest{}, true) {
	}

	// Verify that there is only single span.
	gotSpans := testExporter.GetSpans()
	if len(gotSpans) != 1 {
		t.Fatalf("expected 1 span, got %d", len(gotSpans))
	}
	gotSpan := gotSpans[0]

	if gotSpan.Name != "generate_content test-model" {
		t.Errorf("expected span name %q, got %q", "generate_content test-model", gotSpan.Name)
	}

	// Verify span attributes.
	attrs := make(map[attribute.Key]string)
	for _, kv := range gotSpan.Attributes {
		attrs[kv.Key] = kv.Value.Emit()
	}

	if val := attrs[semconv.GenAIUsageInputTokensKey]; val != "10" {
		t.Errorf("expected input tokens 10, got %s", val)
	}
	if val := attrs[semconv.GenAIUsageOutputTokensKey]; val != "20" {
		t.Errorf("expected output tokens 20, got %s", val)
	}
}

func TestGenerateContentTracingError(t *testing.T) {
	setupTestTracer(t)

	modelMock := &mockModelForTest{
		name: "test-model",
		generateContent: func(ctx context.Context, req *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error] {
			return func(yield func(*model.LLMResponse, error) bool) {
				// Yield partial response.
				if !yield(&model.LLMResponse{
					UsageMetadata: &genai.GenerateContentResponseUsageMetadata{
						PromptTokenCount:     1,
						CandidatesTokenCount: 2,
					},
					Partial: true,
				}, nil) {
					return
				}

				// Yield error.
				yield(nil, errors.New("test error"))

				// Verify span ended.
				gotSpans := testExporter.GetSpans()
				if len(gotSpans) != 1 {
					t.Errorf("expected 1 span after error, got %d", len(gotSpans))
				}
			}
		},
	}

	ctx := icontext.NewInvocationContext(context.Background(), icontext.InvocationContextParams{})

	for range generateContent(ctx, modelMock, &model.LLMRequest{}, true) {
	}

	// Verify that there is only single span.
	gotSpans := testExporter.GetSpans()
	if len(gotSpans) != 1 {
		t.Fatalf("expected 1 span, got %d", len(gotSpans))
	}
	gotSpan := gotSpans[0]

	if gotSpan.Name != "generate_content test-model" {
		t.Errorf("expected span name %q, got %q", "generate_content test-model", gotSpan.Name)
	}

	if gotSpan.Status.Code != codes.Error {
		t.Errorf("expected span status %q, got %q", codes.Error, gotSpan.Status.Code)
	}

	if gotSpan.Status.Description != "test error" {
		t.Errorf("expected span status description %q, got %q", "test error", gotSpan.Status.Description)
	}
}

func setupTestTracer(t *testing.T) {
	t.Helper()
	initTracer.Do(func() {
		// internal/telemetry initializes the global tracer provider once at startup.
		// Subsequent calls to otel.SetTracerProvider don't update existing tracer providers, so we can override only once.
		testExporter = tracetest.NewInMemoryExporter()
		tp := sdktrace.NewTracerProvider(
			sdktrace.WithSyncer(testExporter),
		)
		otel.SetTracerProvider(tp)
	})
	// Reset the exporter before each test to avoid flakiness.
	testExporter.Reset()
	t.Cleanup(func() {
		testExporter.Reset()
	})
}

type inMemoryLogExporter struct {
	records []sdklog.Record
}

func (e *inMemoryLogExporter) Export(ctx context.Context, records []sdklog.Record) error {
	e.records = append(e.records, records...)
	return nil
}
func (e *inMemoryLogExporter) Shutdown(ctx context.Context) error   { return nil }
func (e *inMemoryLogExporter) ForceFlush(ctx context.Context) error { return nil }

func TestLoggingSpanIDPropagation(t *testing.T) {
	setupTestTracer(t)
	logExporter := setupLoggerProvider(t)

	var wantSpanID trace.SpanID
	modelMock := &mockModelForTest{
		name: "test-model",
		generateContent: func(ctx context.Context, req *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error] {
			// Capture the span ID.
			wantSpanID = trace.SpanFromContext(ctx).SpanContext().SpanID()
			if !wantSpanID.IsValid() {
				t.Fatalf("expected span ID to be valid, got %q", wantSpanID)
			}
			return func(yield func(*model.LLMResponse, error) bool) {
				yield(&model.LLMResponse{
					UsageMetadata: &genai.GenerateContentResponseUsageMetadata{
						PromptTokenCount:     1,
						CandidatesTokenCount: 2,
					},
					Content: &genai.Content{
						Role:  "model",
						Parts: []*genai.Part{{Text: "Response"}},
					},
				}, nil)
			}
		},
	}

	req := &model.LLMRequest{
		Config: &genai.GenerateContentConfig{
			SystemInstruction: &genai.Content{
				Role: "system",
				Parts: []*genai.Part{
					{Text: "You are a helpful assistant."},
				},
			},
		},
		Contents: []*genai.Content{
			{
				Role: "user",
				Parts: []*genai.Part{
					{Text: "Hello"},
				},
			},
		},
	}

	ctx := icontext.NewInvocationContext(context.Background(), icontext.InvocationContextParams{})
	for range generateContent(ctx, modelMock, req, true) {
	}

	if len(logExporter.records) != 3 {
		t.Fatalf("expected 3 log records, got %d", len(logExporter.records))
	}

	wantEvents := []string{
		"gen_ai.system.message",
		"gen_ai.user.message",
		"gen_ai.choice",
	}

	for i, record := range logExporter.records {
		if got := record.SpanID(); got != wantSpanID {
			t.Errorf("record[%d]: expected span ID %q, got %q", i, wantSpanID, got)
		}
		if got := record.EventName(); got != wantEvents[i] {
			t.Errorf("record[%d]: expected event name %q, got %q", i, wantEvents[i], got)
		}
	}
}

func setupLoggerProvider(t *testing.T) *inMemoryLogExporter {
	logExporter := &inMemoryLogExporter{}
	provider := sdklog.NewLoggerProvider(
		sdklog.WithProcessor(sdklog.NewSimpleProcessor(logExporter)),
	)
	originalProvider := global.GetLoggerProvider()
	global.SetLoggerProvider(provider)
	t.Cleanup(func() {
		global.SetLoggerProvider(originalProvider)
	})
	return logExporter
}
