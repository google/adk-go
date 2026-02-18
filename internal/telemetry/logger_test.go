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
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/log"
	sdklog "go.opentelemetry.io/otel/sdk/log"
	semconv "go.opentelemetry.io/otel/semconv/v1.36.0"
	"google.golang.org/genai"

	"google.golang.org/adk/model"
)

func TestLogRequest(t *testing.T) {
	type wantEvent struct {
		name  string
		body  any // can be map[string]any or string (for elided)
		attrs []log.KeyValue
	}
	tests := []struct {
		name                  string
		backend               genai.Backend
		captureMessageContent bool
		req                   *model.LLMRequest
		wantEvents            []wantEvent
	}{
		{
			name:                  "RequestWithSystemAndUserMessages",
			captureMessageContent: true,
			req: &model.LLMRequest{
				Config: &genai.GenerateContentConfig{
					SystemInstruction: &genai.Content{
						Role: "system",
						Parts: []*genai.Part{
							{Text: "System instruction part 1"},
							{Text: "System instruction part 2"},
						},
					},
				},
				Contents: []*genai.Content{
					// Messages from previous turns.
					{
						Role: "user", Parts: []*genai.Part{
							{Text: "Previous user message part 1"},
							{Text: "Previous user message part 2"},
						},
					},
					{
						Role: "agent",
						Parts: []*genai.Part{
							{Text: "Previous agent message part 1"},
							{Text: "Previous agent message part 2"},
						},
					},
					// New message.
					{
						Role: "user", Parts: []*genai.Part{
							{Text: "User message part 1"},
							{Text: "User message part 2"},
						},
					},
				},
			},
			wantEvents: []wantEvent{
				{
					name: "gen_ai.system.message",
					body: map[string]any{
						"content": "System instruction part 1\nSystem instruction part 2",
					},
				},
				{
					name: "gen_ai.user.message",
					body: map[string]any{
						"content": map[string]any{
							"role": "user",
							"parts": []any{
								map[string]any{"text": "Previous user message part 1"},
								map[string]any{"text": "Previous user message part 2"},
							},
						},
					},
				},
				{
					name: "gen_ai.user.message",
					body: map[string]any{
						"content": map[string]any{
							"role": "agent",
							"parts": []any{
								map[string]any{"text": "Previous agent message part 1"},
								map[string]any{"text": "Previous agent message part 2"},
							},
						},
					},
				},
				{
					name: "gen_ai.user.message",
					body: map[string]any{
						"content": map[string]any{
							"role": "user",
							"parts": []any{
								map[string]any{"text": "User message part 1"},
								map[string]any{"text": "User message part 2"},
							},
						},
					},
				},
			},
		},
		{
			name:                  "RequestWithNilConfigAndContents",
			captureMessageContent: true,
			req: &model.LLMRequest{
				Config:   nil,
				Contents: nil,
			},
			wantEvents: []wantEvent{
				{
					name: "gen_ai.system.message",
					body: map[string]any{
						"content": nil,
					},
				},
			},
		},
		{
			name:                  "RequestWithNilContentsGeminiBackend",
			captureMessageContent: true,
			backend:               genai.BackendGeminiAPI,
			req: &model.LLMRequest{
				Config:   nil,
				Contents: nil,
			},
			wantEvents: []wantEvent{
				{
					name: "gen_ai.system.message",
					body: map[string]any{
						"content": nil,
					},
					attrs: []log.KeyValue{
						log.KeyValueFromAttribute(semconv.GenAISystemGCPGemini),
					},
				},
			},
		},
		{
			name:                  "RequestWithNilContentsVertexBackend",
			captureMessageContent: true,
			backend:               genai.BackendVertexAI,
			req: &model.LLMRequest{
				Config:   nil,
				Contents: nil,
			},
			wantEvents: []wantEvent{
				{
					name: "gen_ai.system.message",
					body: map[string]any{
						"content": nil,
					},
					attrs: []log.KeyValue{
						log.KeyValueFromAttribute(semconv.GenAISystemGCPVertexAI),
					},
				},
			},
		},
		{
			name:                  "RequestWithEmptyConfigAndUserContentWithoutParts",
			captureMessageContent: true,
			req: &model.LLMRequest{
				Config: &genai.GenerateContentConfig{
					// Config without system instruction.
				},
				Contents: []*genai.Content{
					// Content without parts.
					{Role: "user"},
				},
			},
			wantEvents: []wantEvent{
				{
					name: "gen_ai.system.message",
					body: map[string]any{
						"content": nil,
					},
				},
				{
					name: "gen_ai.user.message",
					body: map[string]any{
						"content": map[string]any{
							"role": "user",
						},
					},
				},
			},
		},
		{
			name:                  "ElidedRequest",
			captureMessageContent: false,
			req: &model.LLMRequest{
				Config: &genai.GenerateContentConfig{
					SystemInstruction: &genai.Content{
						Role: "system",
						Parts: []*genai.Part{
							{Text: "System instruction"},
						},
					},
				},
				Contents: []*genai.Content{
					{Role: "user", Parts: []*genai.Part{{Text: "Hello"}}},
				},
			},
			wantEvents: []wantEvent{
				{
					name: "gen_ai.system.message",
					body: map[string]any{
						"content": "<elided>",
					},
				},
				{
					name: "gen_ai.user.message",
					body: map[string]any{
						"content": "<elided>",
					},
				},
			},
		},
		{
			name:                  "ElidedRequestWithNilConfigAndContents",
			captureMessageContent: false,
			req: &model.LLMRequest{
				Config:   nil,
				Contents: nil,
			},
			wantEvents: []wantEvent{
				{
					name: "gen_ai.system.message",
					body: map[string]any{
						"content": "<elided>",
					},
				},
			},
		},
		{
			name:                  "ElidedRequestWithEmptyConfigAndUserContentWithoutParts",
			captureMessageContent: false,
			req: &model.LLMRequest{
				Config: &genai.GenerateContentConfig{
					// Config without system instruction.
				},
				Contents: []*genai.Content{
					// Content without parts.
					{Role: "user"},
				},
			},
			wantEvents: []wantEvent{
				{
					name: "gen_ai.system.message",
					body: map[string]any{
						"content": "<elided>",
					},
				},
				{
					name: "gen_ai.user.message",
					body: map[string]any{
						"content": "<elided>",
					},
				},
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ctx := t.Context()
			exporter := setup(t, tc.captureMessageContent)

			LogRequest(ctx, tc.req, tc.backend)

			if len(exporter.records) != len(tc.wantEvents) {
				var records strings.Builder
				for _, r := range exporter.records {
					records.WriteString(r.EventName())
					records.WriteString("\n")
				}
				t.Fatalf("expected %d records, got %d, got events:\n%s", len(tc.wantEvents), len(exporter.records), records.String())
			}

			for i, want := range tc.wantEvents {
				gotRecord := exporter.records[i]
				if gotRecord.EventName() != want.name {
					t.Errorf("record[%d]: expected event %q, got %q", i, want.name, gotRecord.EventName())
				}
				gotBody := toGoValue(gotRecord.Body())

				if diff := cmp.Diff(want.body, gotBody); diff != "" {
					t.Errorf("record[%d] body mismatch (-want +got):\n%s", i, diff)
				}

				var gotAttrs []log.KeyValue
				gotRecord.WalkAttributes(func(kv log.KeyValue) bool {
					gotAttrs = append(gotAttrs, kv)
					return true
				})
				if diff := cmp.Diff(want.attrs, gotAttrs); diff != "" {
					t.Errorf("record[%d] attributes mismatch (-want +got):\n%s", i, diff)
				}
			}
		})
	}
}

func TestLogResponse(t *testing.T) {
	tests := []struct {
		name                  string
		resp                  *model.LLMResponse
		backend               genai.Backend
		captureMessageContent bool
		wantName              string
		wantBody              map[string]any
		wantAttrs             []log.KeyValue
	}{
		{
			name:                  "Response",
			captureMessageContent: true,
			resp: &model.LLMResponse{
				FinishReason: genai.FinishReasonStop,
				Content: &genai.Content{
					Role: "model",
					Parts: []*genai.Part{
						{Text: "Text part 1"},
						{Text: "Text part 2"},
					},
				},
			},
			wantName: "gen_ai.choice",
			wantBody: map[string]any{
				"index":         int64(0),
				"finish_reason": "STOP",
				"content": map[string]any{
					"role": "model",
					"parts": []any{
						map[string]any{"text": "Text part 1"},
						map[string]any{"text": "Text part 2"},
					},
				},
			},
		},
		{
			name:                  "ResponseGeminiBackend",
			captureMessageContent: true,
			backend:               genai.BackendGeminiAPI,
			resp: &model.LLMResponse{
				FinishReason: genai.FinishReasonStop,
				Content: &genai.Content{
					Role: "model",
					Parts: []*genai.Part{
						{Text: "Text"},
					},
				},
			},
			wantName: "gen_ai.choice",
			wantBody: map[string]any{
				"index":         int64(0),
				"finish_reason": "STOP",
				"content": map[string]any{
					"role": "model",
					"parts": []any{
						map[string]any{"text": "Text"},
					},
				},
			},
			wantAttrs: []log.KeyValue{
				log.KeyValueFromAttribute(semconv.GenAISystemGCPGemini),
			},
		},
		{
			name:                  "ResponseVertexBackend",
			captureMessageContent: true,
			backend:               genai.BackendVertexAI,
			resp: &model.LLMResponse{
				FinishReason: genai.FinishReasonStop,
				Content: &genai.Content{
					Role: "model",
					Parts: []*genai.Part{
						{Text: "Text part 1"},
						{Text: "Text part 2"},
					},
				},
			},
			wantName: "gen_ai.choice",
			wantBody: map[string]any{
				"index":         int64(0),
				"finish_reason": "STOP",
				"content": map[string]any{
					"role": "model",
					"parts": []any{
						map[string]any{"text": "Text part 1"},
						map[string]any{"text": "Text part 2"},
					},
				},
			},
			wantAttrs: []log.KeyValue{
				log.KeyValueFromAttribute(semconv.GenAISystemGCPVertexAI),
			},
		},
		{
			name:                  "ResponseWithFunctionCall",
			captureMessageContent: true,
			resp: &model.LLMResponse{
				FinishReason: genai.FinishReasonStop,
				Content: &genai.Content{
					Role: "model",
					Parts: []*genai.Part{
						{Thought: true, Text: "Call tools"},
						{FunctionCall: &genai.FunctionCall{Name: "myTool1", ID: "id1", Args: map[string]any{"arg1": "val1"}}},
						{FunctionCall: &genai.FunctionCall{Name: "myTool2", ID: "id2", Args: map[string]any{"arg2": "val2"}}},
					},
				},
			},
			wantName: "gen_ai.choice",
			wantBody: map[string]any{
				"index":         int64(0),
				"finish_reason": "STOP",
				"content": map[string]any{
					"role": "model",
					"parts": []any{
						map[string]any{
							"text":    "Call tools",
							"thought": true,
						},
						map[string]any{"functionCall": map[string]any{
							"name": "myTool1",
							"id":   "id1",
							"args": map[string]any{"arg1": "val1"},
						}},
						map[string]any{"functionCall": map[string]any{
							"name": "myTool2",
							"id":   "id2",
							"args": map[string]any{"arg2": "val2"},
						}},
					},
				},
			},
		},
		{
			name:                  "NilResponse",
			captureMessageContent: true,
			resp:                  nil,
			wantName:              "gen_ai.choice",
			wantBody: map[string]any{
				"index":   int64(0),
				"content": nil,
			},
		},
		{
			name:                  "ElidedResponse",
			captureMessageContent: false,
			resp: &model.LLMResponse{
				FinishReason: genai.FinishReasonStop,
				Content: &genai.Content{
					Role: "model",
					Parts: []*genai.Part{
						{Text: "Response part 1"},
						{Text: "Response part 2"},
					},
				},
			},
			wantName: "gen_ai.choice",
			wantBody: map[string]any{
				"index":         int64(0),
				"finish_reason": "STOP",
				"content":       "<elided>",
			},
		},
		{
			name:                  "ElidedNilResponse",
			captureMessageContent: false,
			resp:                  nil,
			wantName:              "gen_ai.choice",
			wantBody: map[string]any{
				"index":   int64(0),
				"content": "<elided>",
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			exporter := setup(t, tc.captureMessageContent)

			LogResponse(t.Context(), tc.resp, tc.backend)

			if len(exporter.records) != 1 {
				var records strings.Builder
				for _, r := range exporter.records {
					records.WriteString(r.EventName())
					records.WriteString("\n")
				}
				t.Fatalf("expected 1 record, got %d, got events:\n%s", len(exporter.records), records.String())
			}
			record := exporter.records[0]
			if record.EventName() != tc.wantName {
				t.Errorf("expected event %q, got %q", tc.wantName, record.EventName())
			}

			got := toGoValue(record.Body())
			if diff := cmp.Diff(tc.wantBody, got); diff != "" {
				t.Errorf("Body mismatch (-want +got):\n%s", diff)
			}

			var gotAttrs []log.KeyValue
			record.WalkAttributes(func(kv log.KeyValue) bool {
				gotAttrs = append(gotAttrs, kv)
				return true
			})
			if diff := cmp.Diff(tc.wantAttrs, gotAttrs); diff != "" {
				t.Errorf("attributes mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestSpanIDPropagation(t *testing.T) {
	ctx, span := otel.Tracer("test").Start(context.Background(), "test")
	defer span.End()

	exporter := setup(t, false)

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

	LogRequest(ctx, req, genai.BackendVertexAI)
	LogResponse(ctx, &model.LLMResponse{}, genai.BackendVertexAI)

	if len(exporter.records) != 3 {
		t.Fatalf("expected 3 records, got %d", len(exporter.records))
	}

	wantSpanID := span.SpanContext().SpanID()
	for _, record := range exporter.records {
		if got := record.SpanID(); got != wantSpanID {
			t.Errorf("expected span ID %q, got %q", wantSpanID, got)
		}
	}
}

func setup(t *testing.T, elided bool) *inMemoryExporter {
	exporter := &inMemoryExporter{}
	provider := sdklog.NewLoggerProvider(
		sdklog.WithProcessor(sdklog.NewSimpleProcessor(exporter)),
	)
	originalLogger := otelLogger
	otelLogger = provider.Logger("test")
	t.Cleanup(func() {
		otelLogger = originalLogger
	})

	original := getGenAICaptureMessageContent()
	SetGenAICaptureMessageContent(elided)
	t.Cleanup(func() {
		SetGenAICaptureMessageContent(original)
	})
	return exporter
}

type inMemoryExporter struct {
	records []sdklog.Record
}

func (e *inMemoryExporter) Export(ctx context.Context, records []sdklog.Record) error {
	e.records = append(e.records, records...)
	return nil
}

func (e *inMemoryExporter) Shutdown(ctx context.Context) error   { return nil }
func (e *inMemoryExporter) ForceFlush(ctx context.Context) error { return nil }

// toGoValue converts a log.Value to a Go value for easier testing.
// log.Value is not comparable by design, so we need to transform it to another form.
func toGoValue(v log.Value) any {
	switch v.Kind() {
	case log.KindBool:
		return v.AsBool()
	case log.KindFloat64:
		return v.AsFloat64()
	case log.KindInt64:
		return v.AsInt64()
	case log.KindString:
		return v.AsString()
	case log.KindBytes:
		return v.AsBytes()
	case log.KindSlice:
		var s []any
		for _, v := range v.AsSlice() {
			s = append(s, toGoValue(v))
		}
		return s
	case log.KindMap:
		m := make(map[string]any)
		for _, kv := range v.AsMap() {
			m[kv.Key] = toGoValue(kv.Value)
		}
		return m
	default:
		return nil
	}
}
