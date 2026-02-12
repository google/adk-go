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
	"go.opentelemetry.io/otel/log"
	sdklog "go.opentelemetry.io/otel/sdk/log"
	"google.golang.org/genai"

	"google.golang.org/adk/model"
)

func TestLogRequest(t *testing.T) {
	type wantEvent struct {
		name string
		body any // can be map[string]any or string (for elided)
	}
	tests := []struct {
		name       string
		req        *model.LLMRequest
		elide      bool
		wantEvents []wantEvent
	}{
		{
			name: "RequestWithSystemAndUserMessages",
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
			name: "RequestWithNilConfigAndContents",
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
			name: "RequestWithEmptyConfigAndUserContentWithoutParts",
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
			name: "ElidedRequest",
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
			elide: true,
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
			name: "ElidedRequestWithNilConfigAndContents",
			req: &model.LLMRequest{
				Config:   nil,
				Contents: nil,
			},
			elide: true,
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
			name: "ElidedRequestWithEmptyConfigAndUserContentWithoutParts",
			req: &model.LLMRequest{
				Config: &genai.GenerateContentConfig{
					// Config without system instruction.
				},
				Contents: []*genai.Content{
					// Content without parts.
					{Role: "user"},
				},
			},
			elide: true,
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
			exporter := setup(t, tc.elide)

			LogRequest(ctx, tc.req)

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
			}
		})
	}
}

func TestLogResponse(t *testing.T) {
	tests := []struct {
		name     string
		resp     *model.LLMResponse
		elide    bool
		wantName string
		wantBody map[string]any
	}{
		{
			name: "Response",
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
			name: "ResponseWithFunctionCall",
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
			name:     "NilResponse",
			resp:     nil,
			wantName: "gen_ai.choice",
			wantBody: map[string]any{
				"index":   int64(0),
				"content": nil,
			},
		},
		{
			name: "ElidedResponse",
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
			elide:    true,
			wantName: "gen_ai.choice",
			wantBody: map[string]any{
				"index":         int64(0),
				"finish_reason": "STOP",
				"content":       "<elided>",
			},
		},
		{
			name:     "ElidedNilResponse",
			resp:     nil,
			elide:    true,
			wantName: "gen_ai.choice",
			wantBody: map[string]any{
				"index":   int64(0),
				"content": "<elided>",
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			exporter := setup(t, tc.elide)

			LogResponse(t.Context(), tc.resp, nil)

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
		})
	}
}

func setup(t *testing.T, elided bool) *inMemoryExporter {
	exporter := &inMemoryExporter{}
	provider := sdklog.NewLoggerProvider(
		sdklog.WithProcessor(sdklog.NewSimpleProcessor(exporter)),
	)
	originalLogger := logger
	logger = provider.Logger("test")
	t.Cleanup(func() {
		logger = originalLogger
	})

	original := elideMessageContent
	elideMessageContent = elided
	t.Cleanup(func() {
		elideMessageContent = original
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
