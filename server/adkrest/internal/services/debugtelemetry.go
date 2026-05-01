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

package services

import (
	"cmp"
	"context"
	"fmt"
	"slices"
	"sync"
	"sync/atomic"
	"time"

	lru "github.com/hashicorp/golang-lru/v2"

	"go.opentelemetry.io/otel/attribute"
	sdklog "go.opentelemetry.io/otel/sdk/log"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	semconv "go.opentelemetry.io/otel/semconv/v1.36.0"
	"go.opentelemetry.io/otel/trace"

	"google.golang.org/adk/internal/telemetry"
)

const (
	defaultTraceCapacity = 10_000
	eventIDKey           = "gcp.vertex.agent.event_id"
)

// DebugTelemetry stores the in memory spans and logs, grouped by session and event.
type DebugTelemetry struct {
	store *spanStore
}

type DebugTelemetryConfig struct {
	// Maximum number of traces to keep in memory.
	// If <= 0, default capacity (10_000) is used.
	TraceCapacity int
}

// NewDebugTelemetryWithConfig returns a new DebugTelemetry instance with custom capacity.
func NewDebugTelemetryWithConfig(cfg *DebugTelemetryConfig) (*DebugTelemetry, error) {
	capacity := defaultTraceCapacity
	if cfg != nil && cfg.TraceCapacity > 0 {
		capacity = cfg.TraceCapacity
	}
	store, err := newSpanStore(capacity)
	if err != nil {
		return nil, fmt.Errorf("failed to create span store: %w", err)
	}
	return &DebugTelemetry{
		store: store,
	}, nil
}

func (d *DebugTelemetry) SpanProcessor() sdktrace.SpanProcessor {
	// The store implements sdktrace.SpanProcessor directly so it can hook
	// OnStart and assign a monotonic startSeq used as a tie-breaker when
	// wall-clock StartTime collides (Darwin's microsecond-resolution
	// time.Now() makes this routine on adjacent tracer.Start calls).
	return d.store
}

func (d *DebugTelemetry) LogProcessor() sdklog.Processor {
	// Use simple processor to avoid the lag between logging and it appearing in adk-web.
	return sdklog.NewSimpleProcessor(d.store)
}

// GetSpansByEventID returns spans associated with the given event ID.
func (d *DebugTelemetry) GetSpansByEventID(eventID string) []DebugSpan {
	return d.store.getSpansByEventID(eventID)
}

// GetSpansBySessionID returns spans associated with the given session ID.
func (d *DebugTelemetry) GetSpansBySessionID(sessionID string) []DebugSpan {
	return d.store.getSpansBySessionID(sessionID)
}

func convertAttrs(in []attribute.KeyValue) map[string]string {
	out := make(map[string]string, len(in))
	for _, attr := range in {
		out[string(attr.Key)] = attr.Value.Emit()
	}
	return out
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
	Name         string
	StartTime    time.Time
	EndTime      time.Time
	Context      trace.SpanContext
	ParentSpanID trace.SpanID
	Attributes   map[string]string
	Logs         []DebugLog

	// startSeq is a monotonic sequence number assigned in OnStart. It
	// breaks ties when StartTime.UnixNano() collides — which happens on
	// platforms (e.g. some Darwin versions) where time.Now() has only
	// microsecond resolution and back-to-back tracer.Start calls return
	// identical wall-clock timestamps. Sorting by (StartTime, startSeq)
	// preserves the actual call order regardless of wall-clock granularity.
	startSeq uint64
}

// spanStore stores spans and logs in memory for debug telemetry.
type spanStore struct {
	mu sync.RWMutex
	// recordsByTraceID is the main store for spans, indexed by trace id.
	recordsByTraceID *lru.Cache[string, []*spanRecord]
	// recordsBySpanID stores spans indexed by span id.
	recordsBySpanID map[string]*spanRecord
	// traceIDsBySessionID stores trace ids indexed by session id for easy lookup.
	traceIDsBySessionID map[string]map[string]struct{}
	// recordsByEventID stores spans indexed by event id for easy lookup.
	recordsByEventID map[string][]*spanRecord

	// startSeqCounter is incremented atomically in OnStart and stamped
	// directly onto a stub spanRecord so we have a stable secondary
	// sort key. No separate per-span tracking map: an abandoned span
	// (started but never ended) leaves a stub in recordsBySpanID, which
	// is bounded by the LRU eviction path.
	startSeqCounter atomic.Uint64
}

func newSpanStore(capacity int) (*spanStore, error) {
	store := &spanStore{
		recordsBySpanID:     make(map[string]*spanRecord),
		traceIDsBySessionID: make(map[string]map[string]struct{}),
		recordsByEventID:    make(map[string][]*spanRecord),
	}
	var err error
	store.recordsByTraceID, err = lru.NewWithEvict(capacity, store.evict)
	if err != nil {
		return nil, fmt.Errorf("failed to create LRU cache: %w", err)
	}
	return store, nil
}

func (s *spanStore) getSpansByEventID(id string) []DebugSpan {
	s.mu.RLock()
	defer s.mu.RUnlock()
	// Create a copy of the slice to avoid race conditions.
	records := slices.Clone(s.recordsByEventID[id])
	s.touchTraces(records)
	return convertRecords(records)
}

// touchTraces marks traces as recently used. Required because fetching by event ID bypasses the trace LRU cache.
func (s *spanStore) touchTraces(records []*spanRecord) {
	// touchedTraces is used to avoid touching the same trace multiple times.
	touchedTraces := make(map[string]bool)
	for _, r := range records {
		traceIDStr := r.Context.TraceID().String()
		if traceIDStr != "" && !touchedTraces[traceIDStr] {
			touchedTraces[traceIDStr] = true
			// Get the trace to update its access time in the LRU cache, ignore the result.
			s.recordsByTraceID.Get(traceIDStr)
		}
	}
}

func (s *spanStore) getSpansBySessionID(sessionID string) []DebugSpan {
	s.mu.RLock()
	defer s.mu.RUnlock()
	traces := s.traceIDsBySessionID[sessionID]
	var records []*spanRecord
	for traceID := range traces {
		if traceRecords, ok := s.recordsByTraceID.Get(traceID); ok {
			records = append(records, traceRecords...)
		}
	}
	return convertRecords(records)
}

func convertRecords(records []*spanRecord) []DebugSpan {
	records = filterUnclosedAndSort(records)
	debugSpans := make([]DebugSpan, len(records))
	for i, r := range records {
		// Clone the logs to avoid race conditions.
		logs := slices.Clone(r.Logs)
		debugSpans[i] = DebugSpan{
			Name:         r.Name,
			StartTime:    r.StartTime.UnixNano(),
			EndTime:      r.EndTime.UnixNano(),
			SpanID:       r.Context.SpanID().String(),
			TraceID:      r.Context.TraceID().String(),
			ParentSpanID: r.ParentSpanID.String(),
			Attributes:   r.Attributes,
			Logs:         logs,
		}
	}
	return debugSpans
}

func filterUnclosedAndSort(records []*spanRecord) []*spanRecord {
	filtered := slices.DeleteFunc(records, func(s *spanRecord) bool {
		// Logs are emitted before the span is closed and sent to the processor.
		// Skip them in the response.
		return s == nil || !s.Context.TraceID().IsValid()
	})
	slices.SortStableFunc(filtered, func(a, b *spanRecord) int {
		// Primary key: wall-clock start time. Tie-broken by startSeq
		// (assigned monotonically in OnStart) to preserve actual call
		// order on platforms whose time.Now() resolution is too coarse
		// to distinguish back-to-back tracer.Start calls.
		if c := cmp.Compare(a.StartTime.UnixNano(), b.StartTime.UnixNano()); c != 0 {
			return c
		}
		return cmp.Compare(a.startSeq, b.startSeq)
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

// OnStart implements [sdktrace.SpanProcessor]. Stamps the per-store
// monotonic sequence number directly onto a stub spanRecord. Subsequent
// OnEnd / Export calls find the stub and fill in the remaining fields.
// This avoids a side map that could leak entries for abandoned spans.
func (s *spanStore) OnStart(_ context.Context, span sdktrace.ReadWriteSpan) {
	seq := s.startSeqCounter.Add(1)
	spanID := span.SpanContext().SpanID().String()
	s.mu.Lock()
	defer s.mu.Unlock()
	if existing, ok := s.recordsBySpanID[spanID]; ok {
		// Already created by an Export call (logs arrive before the span
		// is closed). Just stamp the seq.
		existing.startSeq = seq
		return
	}
	s.recordsBySpanID[spanID] = &spanRecord{startSeq: seq}
}

// OnEnd implements [sdktrace.SpanProcessor]. Persists the span fields
// onto the stub created in OnStart.
func (s *spanStore) OnEnd(span sdktrace.ReadOnlySpan) {
	s.mu.Lock()
	defer s.mu.Unlock()
	attrs := convertAttrs(span.Attributes())
	spanID := span.SpanContext().SpanID().String()
	record, ok := s.recordsBySpanID[spanID]
	if !ok {
		record = &spanRecord{}
		s.recordsBySpanID[spanID] = record
	}

	record.Name = span.Name()
	record.StartTime = span.StartTime()
	record.EndTime = span.EndTime()
	record.Context = span.SpanContext()
	record.ParentSpanID = span.Parent().SpanID()
	record.Attributes = attrs

	s.updateSpanIndexes(record)
}

func (s *spanStore) updateSpanIndexes(span *spanRecord) {
	traceIDStr := span.Context.TraceID().String()
	// Update session id -> trace id mapping.
	sessionIDKey := string(semconv.GenAIConversationIDKey)
	if sessionID, ok := span.Attributes[sessionIDKey]; ok {
		traces, ok := s.traceIDsBySessionID[sessionID]
		if !ok {
			traces = make(map[string]struct{})
			s.traceIDsBySessionID[sessionID] = traces
		}
		traces[traceIDStr] = struct{}{}
	}
	// Update event id -> span id mapping.
	if eventID, ok := span.Attributes[eventIDKey]; ok {
		s.recordsByEventID[eventID] = append(s.recordsByEventID[eventID], span)
	}

	// Update trace id -> span id mapping (LRU).
	records, _ := s.recordsByTraceID.Get(traceIDStr)
	s.recordsByTraceID.Add(traceIDStr, append(records, span))
}

func (s *spanStore) evict(traceID string, spans []*spanRecord) {
	for _, span := range spans {
		if span.Context.TraceID().IsValid() {
			delete(s.recordsBySpanID, span.Context.SpanID().String())

			if eventID, ok := span.Attributes[eventIDKey]; ok {
				s.evictRecordsByEventID(eventID, span)
			}

			if sessionID, ok := span.Attributes[string(semconv.GenAIConversationIDKey)]; ok {
				s.evictTraceIDsBySessionID(sessionID, traceID)
			}
		}
	}
}

func (s *spanStore) evictRecordsByEventID(eventID string, span *spanRecord) {
	records := s.recordsByEventID[eventID]
	records = slices.DeleteFunc(records, func(r *spanRecord) bool {
		return r.Context.SpanID() == span.Context.SpanID()
	})
	if len(records) == 0 {
		delete(s.recordsByEventID, eventID)
	} else {
		s.recordsByEventID[eventID] = records
	}
}

func (s *spanStore) evictTraceIDsBySessionID(sessionID, traceID string) {
	traces := s.traceIDsBySessionID[sessionID]
	if traces != nil {
		delete(traces, traceID)
		if len(traces) == 0 {
			delete(s.traceIDsBySessionID, sessionID)
		}
	}
}

// ForceFlush implements sdklog.Exporter and sdktrace.SpanProcessor.
func (s *spanStore) ForceFlush(ctx context.Context) error {
	return nil
}

// Shutdown implements sdklog.Exporter and sdktrace.SpanProcessor.
func (s *spanStore) Shutdown(ctx context.Context) error {
	return nil
}
