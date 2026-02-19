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
	"slices"
	"sync"
	"time"

	"go.opentelemetry.io/otel/attribute"
	sdklog "go.opentelemetry.io/otel/sdk/log"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	semconv "go.opentelemetry.io/otel/semconv/v1.36.0"
	"go.opentelemetry.io/otel/trace"

	"google.golang.org/adk/internal/telemetry"
)

// DebugTelemetry stores the in memory spans and logs, grouped by session and event.
type DebugTelemetry struct {
	store *spanStore
}

// NewDebugTelemetry returns a new DebugTelemetry instance.
func NewDebugTelemetry() *DebugTelemetry {
	return &DebugTelemetry{
		store: newSpanStore(),
	}
}

func (d *DebugTelemetry) SpanProcessor() sdktrace.SpanProcessor {
	return sdktrace.NewBatchSpanProcessor(d.store)
}

func (d *DebugTelemetry) LogProcessor() sdklog.Processor {
	return sdklog.NewBatchProcessor(d.store)
}

// GetSpansByEventID returns spans associated with the given event ID.
func (d *DebugTelemetry) GetSpansByEventID(eventID string) []DebugSpan {
	records := d.store.getByEventID(eventID)
	return convertRecords(records)
}

// GetSpansBySessionID returns spans associated with the given session ID.
func (d *DebugTelemetry) GetSpansBySessionID(sessionID string) []DebugSpan {
	records := d.store.getBySessionID(sessionID)
	return convertRecords(records)
}

func convertRecords(records []*spanRecord) []DebugSpan {
	debugSpans := make([]DebugSpan, len(records))
	for i, r := range records {
		debugSpans[i] = DebugSpan{
			Name:         r.Span.Name,
			StartTime:    r.Span.StartTime.UnixNano(),
			EndTime:      r.Span.EndTime.UnixNano(),
			TraceID:      r.Span.Context.TraceID().String(),
			SpanID:       r.Span.Context.SpanID().String(),
			ParentSpanID: r.Span.ParentSpanID.String(),
			Attributes:   r.Span.Attributes,
			Logs:         r.Logs,
		}
	}
	return debugSpans
}

func convertAttrs(in []attribute.KeyValue) map[string]string {
	out := make(map[string]string)
	for _, attr := range in {
		out[string(attr.Key)] = attr.Value.Emit()
	}
	return out
}

// SpanContext uniquely identifies a span.
type SpanContext struct {
	TraceID string `json:"trace_id"`
	SpanID  string `json:"span_id"`
}

// DebugSpan represents a span in the trace.
type DebugSpan struct {
	Name         string            `json:"name"`
	StartTime    int64             `json:"start_time"`
	EndTime      int64             `json:"end_time"`
	SpanID       string            `json:"span_id"`
	TraceID      string            `json:"trace_id"`
	ParentSpanID string            `json:"parent_span_id"`
	Attributes   map[string]string `json:"attributes"`
	Logs         []DebugLog        `json:"logs"`
}

// DebugLog represents a log in the span.
type DebugLog struct {
	Body              any    `json:"body"`
	ObservedTimestamp string `json:"observed_timestamp"`
	TraceID           string `json:"trace_id"`
	SpanID            string `json:"span_id"`
	EventName         string `json:"event_name"`
}

// spanRecord stores a span and its associated logs.
type spanRecord struct {
	Span *inMemorySpan
	Logs []DebugLog
}

// inMemorySpan stores spans in memory for debug telemetry.
type inMemorySpan struct {
	Name         string
	StartTime    time.Time
	EndTime      time.Time
	Context      trace.SpanContext
	ParentSpanID trace.SpanID
	Attributes   map[string]string
}

// spanStore stores spans and logs in memory for debug telemetry.
type spanStore struct {
	mu sync.RWMutex
	// recordsBySpanID stores spans indexed by span id.
	recordsBySpanID map[string]*spanRecord
	// traceIDsBySessionID stores trace ids indexed by session id for easy lookup.
	traceIDsBySessionID map[string]map[string]struct{}
	// recordsByEventID stores spans indexed by event id for easy lookup.
	recordsByEventID map[string][]*spanRecord
	// recordsByTraceID stores spans indexed by trace id for easy lookup.
	recordsByTraceID map[string][]*spanRecord
}

func newSpanStore() *spanStore {
	return &spanStore{
		recordsBySpanID:     make(map[string]*spanRecord),
		traceIDsBySessionID: make(map[string]map[string]struct{}),
		recordsByEventID:    make(map[string][]*spanRecord),
		recordsByTraceID:    make(map[string][]*spanRecord),
	}
}

func (s *spanStore) getByEventID(id string) []*spanRecord {
	s.mu.RLock()
	defer s.mu.RUnlock()
	// Create a copy of the slice to avoid race conditions.
	src := s.recordsByEventID[id]
	dst := make([]*spanRecord, len(src))
	copy(dst, src)
	return filterNilsAndSort(dst)
}

func (s *spanStore) getBySessionID(sessionID string) []*spanRecord {
	s.mu.RLock()
	defer s.mu.RUnlock()
	traces := s.traceIDsBySessionID[sessionID]
	var records []*spanRecord
	for traceID := range traces {
		if r, ok := s.recordsByTraceID[traceID]; ok {
			records = append(records, r...)
		}
	}
	return filterNilsAndSort(records)
}

func filterNilsAndSort(records []*spanRecord) []*spanRecord {
	filtered := slices.DeleteFunc(records, func(s *spanRecord) bool {
		// Logs are emitted before the span is closed and sent to the processor.
		// Skip them in the response.
		return s == nil || s.Span == nil
	})
	slices.SortFunc(filtered, func(a, b *spanRecord) int {
		return a.Span.StartTime.Compare(b.Span.StartTime)
	})
	return filtered
}

// Export implements sdklog.Exporter.
func (s *spanStore) Export(ctx context.Context, logRecords []sdklog.Record) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, log := range logRecords {
		if !log.SpanID().IsValid() {
			// Drop the logs without spanID - we'll never return them to the user.
			continue
		}
		spanID := log.SpanID().String()
		record, ok := s.recordsBySpanID[spanID]
		if !ok {
			record = &spanRecord{}
			s.recordsBySpanID[spanID] = record
		}
		record.Logs = append(record.Logs, DebugLog{
			Body:              telemetry.FromLogValue(log.Body()),
			ObservedTimestamp: log.ObservedTimestamp().Format(time.RFC3339Nano),
			TraceID:           log.TraceID().String(),
			SpanID:            log.SpanID().String(),
			EventName:         log.EventName(),
		})
	}
	return nil
}

// ExportSpans implements sdktrace.SpanProcessor.
func (s *spanStore) ExportSpans(ctx context.Context, spans []sdktrace.ReadOnlySpan) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, span := range spans {
		attrs := convertAttrs(span.Attributes())
		spanID := span.SpanContext().SpanID().String()
		record, ok := s.recordsBySpanID[spanID]
		if !ok {
			record = &spanRecord{}
			s.recordsBySpanID[spanID] = record
		}

		record.Span = &inMemorySpan{
			Name:         span.Name(),
			StartTime:    span.StartTime(),
			EndTime:      span.EndTime(),
			Context:      span.SpanContext(),
			ParentSpanID: span.Parent().SpanID(),
			Attributes:   attrs,
		}

		s.updateSpanIndexes(record.Span, record)
	}
	return nil
}

func (s *spanStore) updateSpanIndexes(span *inMemorySpan, records *spanRecord) {
	// Update session id -> trace id mapping.
	sessionIDKey := string(semconv.GenAIConversationIDKey)
	if sessionID, ok := span.Attributes[sessionIDKey]; ok {
		traces, ok := s.traceIDsBySessionID[sessionID]
		if !ok {
			traces = make(map[string]struct{})
			s.traceIDsBySessionID[sessionID] = traces
		}
		traceID := span.Context.TraceID().String()
		traces[traceID] = struct{}{}
	}
	// Update event id -> span id mapping.
	if eventID, ok := span.Attributes["gcp.vertex.agent.event_id"]; ok {
		s.recordsByEventID[eventID] = append(s.recordsByEventID[eventID], records)
	}
	// Update trace id -> span id mapping.
	traceID := span.Context.TraceID().String()
	s.recordsByTraceID[traceID] = append(s.recordsByTraceID[traceID], records)
}

// ForceFlush implements sdklog.Exporter and sdktrace.SpanProcessor.
func (s *spanStore) ForceFlush(ctx context.Context) error {
	return nil
}

// Shutdown implements sdklog.Exporter and sdktrace.SpanProcessor.
func (s *spanStore) Shutdown(ctx context.Context) error {
	return nil
}
