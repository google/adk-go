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
	"encoding/json"
	"fmt"
	"maps"
	"slices"
	"strings"
	"sync"
	"time"

	lru "github.com/hashicorp/golang-lru/v2"

	"go.opentelemetry.io/otel/attribute"
	sdklog "go.opentelemetry.io/otel/sdk/log"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	semconv "go.opentelemetry.io/otel/semconv/v1.36.0"
	"go.opentelemetry.io/otel/trace"

	"google.golang.org/adk/v2/internal/telemetry"
)

const (
	defaultTraceCapacity = 10_000
	eventIDKey           = "gcp.vertex.agent.event_id"
)

// Event names whose log body the ADK web UI validates as a structured message:
// body.content has to be an object holding a "parts" array. Every other event
// name is validated loosely, and gen_ai.system.message wants a plain string, so
// only these two are reshaped.
const (
	userMessageEventName = "gen_ai.user.message"
	choiceEventName      = "gen_ai.choice"
)

// Body field names shared with the web UI schema.
const (
	contentKey = "content"
	partsKey   = "parts"
	textKey    = "text"
	roleKey    = "role"
)

// Default roles for a message whose own role did not survive elision.
//
// The web UI's message schema pipes the body through a transform and then
// requires content.role, so a body without one is rejected and the whole
// Traces panel is discarded. These are the roles the corresponding events
// carry when content capture is on.
const (
	userRole  = "user"
	modelRole = "model"
)

// unrepresentableContent replaces a body that cannot be turned into JSON.
const unrepresentableContent = "<unrepresentable>"

// Span identity the web UI's schema depends on.
const (
	generateContentSpanPrefix = "generate_content"
	eventIDAttribute          = "gcp.vertex.agent.event_id"
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
	// Use simple processor to avoid the lag between ending the span and it appearing in adk-web.
	return sdktrace.NewSimpleSpanProcessor(d.store)
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
//
// Attributes and Logs are never nil in a span returned by [DebugTelemetry], so
// they marshal as {} and [] instead of JSON null.
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
//
// Body is normalised by [normalizeLogBody] when the record is stored, so it
// always marshals and always carries the shape the ADK web UI expects for the
// event name.
type DebugLog struct {
	Body              any    `json:"body"`
	ObservedTimestamp string `json:"observed_timestamp"`
	TraceID           string `json:"trace_id"`
	SpanID            string `json:"span_id"`
	EventName         string `json:"event_name"`
}

// normalizeLogBody returns a log body the ADK web UI can validate, staying as
// close to the original as it can.
//
// The UI validates the whole span array in one pass and discards all of it if
// any record fails, so a single odd body empties the Traces panel rather than
// its own row. Two bodies do that:
//
//   - A message whose content was elided. With content capture off — the
//     default — the server records body.content as the string "<elided>",
//     while the UI requires an object with a "parts" array for user messages
//     and choices. The elision moves inside the structure instead.
//   - A body that cannot be marshalled at all, a NaN float for example. The
//     handler encodes every span in one call, so one of those truncates the
//     response.
func normalizeLogBody(eventName string, body any) any {
	normalized := normalizeMessageShape(eventName, body)
	if _, err := json.Marshal(normalized); err != nil {
		return normalizeMessageShape(eventName, map[string]any{contentKey: unrepresentableContent})
	}
	return normalized
}

// normalizeMessageShape rewrites body.content into a message object for the
// event names the web UI validates as structured messages. Other event names
// are returned untouched.
func normalizeMessageShape(eventName string, body any) any {
	if eventName != userMessageEventName && eventName != choiceEventName {
		return body
	}
	fields, ok := body.(map[string]any)
	if !ok {
		return map[string]any{contentKey: messageContent(body, defaultRole(eventName), nil)}
	}
	out := maps.Clone(fields)
	out[contentKey] = messageContent(fields[contentKey], defaultRole(eventName), fields[roleKey])
	return out
}

// defaultRole is the role to assume for an event whose message lost its own.
func defaultRole(eventName string) string {
	if eventName == choiceEventName {
		return modelRole
	}
	return userRole
}

// messageContent coerces v into the {"parts": [...], "role": …} object the web
// UI's message schema requires, keeping any other fields v already has.
//
// outerRole is the role from the enclosing body, which the UI would otherwise
// hoist itself; it wins over fallback but not over a role already on the
// content.
func messageContent(v any, fallbackRole string, outerRole any) map[string]any {
	content, ok := v.(map[string]any)
	if !ok {
		content = map[string]any{partsKey: textParts(v)}
	} else {
		content = maps.Clone(content)
		if _, ok := content[partsKey].([]any); !ok {
			content[partsKey] = textParts(content[partsKey])
		}
	}
	if role, ok := content[roleKey].(string); !ok || role == "" {
		if role, ok := outerRole.(string); ok && role != "" {
			content[roleKey] = role
		} else {
			content[roleKey] = fallbackRole
		}
	}
	return content
}

// textParts renders v as a parts array: nothing at all for a missing value, one
// text part otherwise.
func textParts(v any) []any {
	if v == nil {
		return []any{}
	}
	text, ok := v.(string)
	if !ok {
		text = fmt.Sprint(v)
	}
	return []any{map[string]any{textKey: text}}
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
		// Build empty containers rather than nil ones: a nil slice marshals to
		// JSON null, and the ADK web UI validates these fields against array and
		// object schemas that a null fails, which makes it discard the whole
		// response and render an empty Traces panel.
		if logs == nil {
			logs = []DebugLog{}
		}
		attributes := eventIDAttributes(r)
		debugSpans[i] = DebugSpan{
			Name:         r.Name,
			StartTime:    r.StartTime.UnixNano(),
			EndTime:      r.EndTime.UnixNano(),
			SpanID:       r.Context.SpanID().String(),
			TraceID:      r.Context.TraceID().String(),
			ParentSpanID: r.ParentSpanID.String(),
			Attributes:   attributes,
			Logs:         logs,
		}
	}
	return debugSpans
}

// eventIDAttributes returns the span's attributes for the response, giving a
// generate_content span an event id when it has none so the web UI will render
// it.
//
// The UI validates the whole span array in one pass and discards all of it if
// any span fails, and its schema requires gcp.vertex.agent.event_id on a
// generate_content span. That attribute is only recorded once the model
// returns: TraceGenerateContentResult returns early when the call errored, so
// a failed turn produces a span without one and the entire Traces panel goes
// blank — exactly when a developer opens it to find out why the call failed.
//
// Dropping the span would satisfy the schema but lose the one span worth
// looking at, so the span id stands in. Every consumer of the attribute in the
// UI is a lookup that tolerates a miss — isEventRow, getUiEvent and the span
// filters all fall back to "no linked event" — so the span renders and is
// simply not tied to a session event, which is the truth: the call produced no
// event to tie it to.
//
// The result is always a fresh map. The records are shared with the store, so
// writing the attribute into r.Attributes would both race with a concurrent
// reader and leave the synthesized value behind in stored telemetry.
func eventIDAttributes(r *spanRecord) map[string]string {
	// Empty rather than nil: a nil map marshals to JSON null, which the UI's
	// object schema rejects.
	attributes := make(map[string]string, len(r.Attributes)+1)
	maps.Copy(attributes, r.Attributes)
	if _, ok := attributes[eventIDAttribute]; !ok && strings.HasPrefix(r.Name, generateContentSpanPrefix) {
		attributes[eventIDAttribute] = r.Context.SpanID().String()
	}
	return attributes
}

func filterUnclosedAndSort(records []*spanRecord) []*spanRecord {
	filtered := slices.DeleteFunc(records, func(s *spanRecord) bool {
		// Logs are emitted before the span is closed and sent to the processor.
		// Skip them in the response.
		return s == nil || !s.Context.TraceID().IsValid()
	})
	slices.SortStableFunc(filtered, func(a, b *spanRecord) int {
		return cmp.Compare(a.StartTime.UnixNano(), b.StartTime.UnixNano())
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
			Body:              normalizeLogBody(log.EventName(), telemetry.FromLogValue(log.Body())),
			ObservedTimestamp: log.ObservedTimestamp().Format(time.RFC3339Nano),
			TraceID:           log.TraceID().String(),
			SpanID:            log.SpanID().String(),
			EventName:         log.EventName(),
		})
	}
	return nil
}

// ExportSpans implements [sdktrace.SpanExporter].
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

		record.Name = span.Name()
		record.StartTime = span.StartTime()
		record.EndTime = span.EndTime()
		record.Context = span.SpanContext()
		record.ParentSpanID = span.Parent().SpanID()
		record.Attributes = attrs

		s.updateSpanIndexes(record)
	}
	return nil
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
