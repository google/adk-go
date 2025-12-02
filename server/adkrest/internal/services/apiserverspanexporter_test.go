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

package services

import (
	"context"
	"testing"

	"go.opentelemetry.io/otel/attribute"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/trace"
)

// capturingExporter is a custom exporter that captures spans for testing.
type capturingExporter struct {
	spans []sdktrace.ReadOnlySpan
}

func (e *capturingExporter) ExportSpans(ctx context.Context, spans []sdktrace.ReadOnlySpan) error {
	e.spans = append(e.spans, spans...)
	return nil
}

func (e *capturingExporter) Shutdown(ctx context.Context) error {
	return nil
}

func TestNewAPIServerSpanExporter(t *testing.T) {
	exporter := NewAPIServerSpanExporter()
	if exporter == nil {
		t.Fatal("NewAPIServerSpanExporter returned nil")
	}
	if exporter.GetTraceDict() == nil {
		t.Error("traceDict should be non-nil")
	}
}

func TestAPIServerSpanExporterExportSpans(t *testing.T) {
	tests := []struct {
		name          string
		spanName      string
		attributes    []attribute.KeyValue
		expectedEvent bool
	}{
		{
			name:     "call_llm-with-event-id-saved",
			spanName: "call_llm",
			attributes: []attribute.KeyValue{
				attribute.String("gcp.vertex.agent.event_id", "event-id"),
			},
			expectedEvent: true,
		},
		{
			name:     "send_data-with-event-id-saved",
			spanName: "send_data",
			attributes: []attribute.KeyValue{
				attribute.String("gcp.vertex.agent.event_id", "event-id"),
			},
			expectedEvent: true,
		},
		{
			name:     "execute_tool-with-event-id-saved",
			spanName: "execute_tool_test",
			attributes: []attribute.KeyValue{
				attribute.String("gcp.vertex.agent.event_id", "event-id"),
			},
			expectedEvent: true,
		},
		{
			name:     "irrelevant_span-ignored",
			spanName: "irrelevant_span",
			attributes: []attribute.KeyValue{
				attribute.String("gcp.vertex.agent.event_id", "event-id"),
			},
			expectedEvent: false,
		},
		{
			name:          "call_llm-missing-event-id-ignored",
			spanName:      "call_llm",
			attributes:    []attribute.KeyValue{},
			expectedEvent: false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ctx := context.Background()
			capturer := &capturingExporter{}
			tp := sdktrace.NewTracerProvider(sdktrace.WithSyncer(capturer))
			tracer := tp.Tracer("test-tracer")

			_, span1 := tracer.Start(ctx, tc.spanName, trace.WithAttributes(tc.attributes...))
			span1.End()

			if err := tp.Shutdown(ctx); err != nil {
				t.Fatalf("failed to shutdown tracer provider: %v", err)
			}

			apiServerExporter := NewAPIServerSpanExporter()
			if err := apiServerExporter.ExportSpans(ctx, capturer.spans); err != nil {
				t.Fatalf("ExportSpans() error = %v", err)
			}

			traceDict := apiServerExporter.GetTraceDict()

			if !tc.expectedEvent {
				if len(traceDict) != 0 {
					t.Errorf("traceDict should be empty, but has %d items", len(traceDict))
				}
				return
			}

			if len(traceDict) != 1 {
				t.Fatalf("traceDict should have 1 item, but has %d", len(traceDict))
			}

			eventDict, ok := traceDict["event-id"]
			if !ok {
				t.Fatalf("traceDict should contain event ID event-id")
			}

			if _, ok := eventDict["span_id"]; !ok {
				t.Fatalf("traceDict should contain span_id")
			}

			if _, ok := eventDict["trace_id"]; !ok {
				t.Fatalf("traceDict should contain trace_id")
			}
		})
	}
}

func TestAPIServerSpanExporterShutdown(t *testing.T) {
	exporter := NewAPIServerSpanExporter()
	if err := exporter.Shutdown(context.Background()); err != nil {
		t.Errorf("Shutdown() error = %v, wantErr nil", err)
	}
}
