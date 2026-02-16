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
	"sync"
	"time"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/log"
	sdklog "go.opentelemetry.io/otel/sdk/log"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	semconv "go.opentelemetry.io/otel/semconv/v1.36.0"
)

// DebugTelemetry stores the in memory spans and logs, grouped by session and event.
type DebugTelemetry struct {
	spanExporter *tracetest.InMemoryExporter
	logExporter  *InMemoryLogExporter
}

// NewDebugTelemetry returns a DebugTelemetry instance
func NewDebugTelemetry() *DebugTelemetry {
	spanExporter := tracetest.NewInMemoryExporter()
	logExporter := &InMemoryLogExporter{
		records: make([]sdklog.Record, 0),
	}
	return &DebugTelemetry{
		spanExporter: spanExporter,
		logExporter:  logExporter,
	}
}

func (d *DebugTelemetry) SpanProcessor() sdktrace.SpanProcessor {
	return sdktrace.NewSimpleSpanProcessor(d.spanExporter)
}

func (d *DebugTelemetry) LogProcessor() sdklog.Processor {
	return sdklog.NewSimpleProcessor(d.logExporter)
}

// GetSpansByEventID returns stored event traces.
func (d *DebugTelemetry) GetSpansByEventID(eventID string) []DebugSpan {
	return d.getSpansFilterByAttrs(func(span tracetest.SpanStub, attrs map[string]string) bool {
		return attrs["gcp.vertex.agent.event_id"] == eventID
	})
}

// GetSpansBySessionID returns stored session traces.
func (d *DebugTelemetry) GetSpansBySessionID(sessionID string) []DebugSpan {
	sessionIDKey := string(semconv.GenAIConversationIDKey)
	return d.getSpansFilterByAttrs(func(span tracetest.SpanStub, attrs map[string]string) bool {
		return attrs[sessionIDKey] == sessionID
	})
}

func (d *DebugTelemetry) getSpansFilterByAttrs(filter func(span tracetest.SpanStub, attrs map[string]string) bool) []DebugSpan {
	var debugSpans []DebugSpan
	spans := d.spanExporter.GetSpans()

	logsBySpanID := make(map[string][]sdklog.Record)
	for _, log := range d.logExporter.GetRecords() {
		if !log.SpanID().IsValid() {
			continue
		}
		spanID := log.SpanID().String()
		prev, ok := logsBySpanID[spanID]
		if !ok {
			prev = nil
		}
		logsBySpanID[spanID] = append(prev, log)
	}

	for _, span := range spans {
		attrs := convertAttrs(span.Attributes)
		if filter(span, attrs) {
			debugSpan := convert(span, attrs)
			debugSpan.Logs = convertLogs(logsBySpanID[span.SpanContext.SpanID().String()])
			debugSpans = append(debugSpans, debugSpan)
		}
	}
	return debugSpans
}

func convertLogs(records []sdklog.Record) []DebugLog {
	var spanLogs []DebugLog
	for _, l := range records {
		spanLogs = append(spanLogs, DebugLog{
			Body:              convertLogValue(l.Body()),
			ObservedTimestamp: l.ObservedTimestamp().Format(time.RFC3339),
			TraceID:           l.TraceID().String(),
			SpanID:            l.SpanID().String(),
			EventName:         l.EventName(),
		})
	}
	return spanLogs
}

func convertLogValue(v log.Value) any {
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
		var res []any
		for _, item := range v.AsSlice() {
			res = append(res, convertLogValue(item))
		}
		return res
	case log.KindMap:
		res := make(map[string]any)
		for _, kv := range v.AsMap() {
			res[kv.Key] = convertLogValue(kv.Value)
		}
		return res
	default:
		return nil
	}
}

func convertAttrs(in []attribute.KeyValue) map[string]string {
	out := make(map[string]string)
	for _, attr := range in {
		out[string(attr.Key)] = attr.Value.Emit()
	}
	return out
}

func convert(span tracetest.SpanStub, attrs map[string]string) DebugSpan {
	return DebugSpan{
		Name:      span.Name,
		StartTime: span.StartTime.Format(time.RFC3339),
		EndTime:   span.EndTime.Format(time.RFC3339),
		Context: SpanContext{
			TraceID: span.SpanContext.TraceID().String(),
			SpanID:  span.SpanContext.SpanID().String(),
		},
		ParentSpanID: span.Parent.SpanID().String(),
		Attributes:   attrs,
	}
}

type SpanContext struct {
	TraceID string `json:"trace_id"`
	SpanID  string `json:"span_id"`
}

// Span represents a span in the trace.
type DebugSpan struct {
	Name         string            `json:"name"`
	StartTime    string            `json:"start_time"`
	EndTime      string            `json:"end_time"`
	Context      SpanContext       `json:"context"`
	ParentSpanID string            `json:"parent_span_id"`
	Attributes   map[string]string `json:"attributes"`
	Logs         []DebugLog        `json:"logs"`
}

// DebugLog represents a log in the span.
type DebugLog struct {
	Body Body `json:"body"`
	// RFC 3339 format timestamp  e.g. "2025-12-02T09:45:36.115239Z"
	ObservedTimestamp string `json:"observed_timestamp"`
	// base16 0x + 32 characters  e.g. "0x6bd725d0f21eb3117ae8cfaa709694b1"
	TraceID   string `json:"trace_id"`
	SpanID    string `json:"span_id"`
	EventName string `json:"event_name"`
}

type Body any

// InMemoryLogExporter stores logs in memory for debug telemetry.
type InMemoryLogExporter struct {
	mu      sync.Mutex
	records []sdklog.Record
}

// Export implements sdklog.Exporter.
func (e *InMemoryLogExporter) Export(ctx context.Context, records []sdklog.Record) error {
	e.mu.Lock()
	defer e.mu.Unlock()
	for _, r := range records {
		e.records = append(e.records, r.Clone())
	}
	return nil
}

// ForceFlush implements sdklog.Exporter.
func (e *InMemoryLogExporter) ForceFlush(ctx context.Context) error {
	return nil
}

// Shutdown implements sdklog.Exporter.
func (e *InMemoryLogExporter) Shutdown(ctx context.Context) error {
	return nil
}

// GetRecords returns a copy of the stored records.
func (e *InMemoryLogExporter) GetRecords() []sdklog.Record {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.records
}
