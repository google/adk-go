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
	"context"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/log"
	sdklog "go.opentelemetry.io/otel/sdk/log"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	semconv "go.opentelemetry.io/otel/semconv/v1.36.0"
	"go.opentelemetry.io/otel/trace"
)

func TestDebugTelemetryGetSpansBySessionID(t *testing.T) {
	ctx := context.Background()

	type testCase struct {
		name             string
		testSetup        func(ctx context.Context, tracer trace.Tracer, logger log.Logger)
		querySessionID   string
		wantSessionSpans []DebugSpan
	}

	tests := []testCase{
		{
			name: "root span with conversation id",
			testSetup: func(rootCtx context.Context, tracer trace.Tracer, logger log.Logger) {
				rootCtx, rootSpan := tracer.Start(rootCtx, "root-span", trace.WithAttributes(
					attribute.String(string(semconv.GenAIConversationIDKey), "session-1"),
				))
				defer rootSpan.End()

				childCtx, childSpan := tracer.Start(rootCtx, "child-span")
				childLog := log.Record{}
				childLog.SetBody(log.StringValue("child-log-body"))
				childLog.SetEventName("child-log-event")
				childLog.SetTimestamp(time.Now())
				logger.Emit(childCtx, childLog)
				childSpan.End()

				rootLog := log.Record{}
				rootLog.SetBody(log.StringValue("root-log-body"))
				rootLog.SetEventName("root-log-event")
				rootLog.SetTimestamp(time.Now())
				logger.Emit(rootCtx, rootLog)
			},
			querySessionID: "session-1",
			wantSessionSpans: []DebugSpan{
				{
					Name:         "root-span",
					ParentSpanID: trace.SpanID{}.String(),
					Attributes: map[string]string{
						string(semconv.GenAIConversationIDKey): "session-1",
					},
					Logs: []DebugLog{
						{
							Body:      "root-log-body",
							EventName: "root-log-event",
						},
					},
				},
				{
					Name:         "child-span",
					ParentSpanID: trace.SpanID{}.String(),
					Attributes:   map[string]string{},
					Logs: []DebugLog{
						{
							Body:      "child-log-body",
							EventName: "child-log-event",
						},
					},
				},
			},
		},
		{
			name: "child span with conversation id",
			testSetup: func(rootCtx context.Context, tracer trace.Tracer, logger log.Logger) {
				var rootSpan trace.Span
				rootCtx, rootSpan = tracer.Start(rootCtx, "root")
				childCtx, childSpan := tracer.Start(rootCtx, "child")
				_, secondChildSpan := tracer.Start(rootCtx, "child-2")
				_, thirdChildSpan := tracer.Start(childCtx, "grandchild", trace.WithAttributes(
					semconv.GenAIConversationID("test-session-id"),
				))
				thirdChildSpan.End()
				secondChildSpan.End()
				childSpan.End()
				rootSpan.End()

				// Create another trace with a different session ID (should not be returned).
				_, rootSpan3 := tracer.Start(context.Background(), "root-3", trace.WithAttributes(
					semconv.GenAIConversationID("test-session-id-1"),
				))
				rootSpan3.End()
			},
			querySessionID: "test-session-id",
			wantSessionSpans: []DebugSpan{
				{Name: "root", Attributes: map[string]string{}},
				{Name: "child", Attributes: map[string]string{}},
				{Name: "child-2", Attributes: map[string]string{}},
				{Name: "grandchild", Attributes: map[string]string{string(semconv.GenAIConversationIDKey): "test-session-id"}},
			},
		},
		{
			name: "multiple traces with same session id",
			testSetup: func(ctx context.Context, tracer trace.Tracer, logger log.Logger) {
				// Trace 1
				root1Ctx, root1Span := tracer.Start(ctx, "root-1", trace.WithAttributes(
					semconv.GenAIConversationID("session-1"),
				))
				_, child1 := tracer.Start(root1Ctx, "child-1")
				child1.End()
				root1Span.End()

				// Trace 2 (different trace ID, same session ID)
				// Session ID on child span
				root2Ctx, root2Span := tracer.Start(ctx, "root-2")
				_, child2 := tracer.Start(root2Ctx, "child-2", trace.WithAttributes(
					semconv.GenAIConversationID("session-1"),
				))
				child2.End()
				root2Span.End()
			},
			querySessionID: "session-1",
			wantSessionSpans: []DebugSpan{
				{Name: "root-1", Attributes: map[string]string{string(semconv.GenAIConversationIDKey): "session-1"}},
				{Name: "child-1", Attributes: map[string]string{}},
				{Name: "root-2", Attributes: map[string]string{}},
				{Name: "child-2", Attributes: map[string]string{string(semconv.GenAIConversationIDKey): "session-1"}},
			},
		},
		{
			name: "trace with spans with mixed session ids session-1",
			testSetup: func(ctx context.Context, tracer trace.Tracer, logger log.Logger) {
				rootCtx, rootSpan := tracer.Start(ctx, "mixed-root", trace.WithAttributes(
					semconv.GenAIConversationID("session-1"),
				))
				_, childSpan := tracer.Start(rootCtx, "mixed-child", trace.WithAttributes(
					semconv.GenAIConversationID("session-2"),
				))
				childSpan.End()
				rootSpan.End()
			},
			querySessionID: "session-1",
			wantSessionSpans: []DebugSpan{
				{Name: "mixed-root", Attributes: map[string]string{string(semconv.GenAIConversationIDKey): "session-1"}},
				{Name: "mixed-child", Attributes: map[string]string{string(semconv.GenAIConversationIDKey): "session-2"}},
			},
		},
		{
			name: "trace with spans with mixed session ids session-2",
			testSetup: func(ctx context.Context, tracer trace.Tracer, logger log.Logger) {
				rootCtx, rootSpan := tracer.Start(ctx, "mixed-root", trace.WithAttributes(
					semconv.GenAIConversationID("session-1"),
				))
				_, childSpan := tracer.Start(rootCtx, "mixed-child", trace.WithAttributes(
					semconv.GenAIConversationID("session-2"),
				))
				childSpan.End()
				rootSpan.End()
			},
			querySessionID: "session-2",
			wantSessionSpans: []DebugSpan{
				{Name: "mixed-root", Attributes: map[string]string{string(semconv.GenAIConversationIDKey): "session-1"}},
				{Name: "mixed-child", Attributes: map[string]string{string(semconv.GenAIConversationIDKey): "session-2"}},
			},
		},
		{
			name: "no matching session id",
			testSetup: func(ctx context.Context, tracer trace.Tracer, logger log.Logger) {
				_, rootSpan := tracer.Start(ctx, "root-1", trace.WithAttributes(
					attribute.String(string(semconv.GenAIConversationIDKey), "session-1"),
					attribute.String("gcp.vertex.agent.event_id", "event-1"),
				))
				rootSpan.End()
			},
			querySessionID:   "non-existent-session",
			wantSessionSpans: nil,
		},
		{
			name: "log without span",
			testSetup: func(ctx context.Context, tracer trace.Tracer, logger log.Logger) {
				var logRecord log.Record
				logRecord.SetBody(log.StringValue("test body"))
				logRecord.SetEventName("test_event")
				logRecord.SetTimestamp(time.Now())

				logger.Emit(ctx, logRecord)
			},
			querySessionID:   "session-1",
			wantSessionSpans: nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			debugTelemetry, tp, lp := setup(t)

			if tt.testSetup != nil {
				tt.testSetup(ctx, tp.Tracer("test-tracer"), lp.Logger("test-logger"))
			}
			if err := tp.ForceFlush(ctx); err != nil {
				t.Fatalf("Failed to flush spans: %v", err)
			}
			if err := lp.ForceFlush(ctx); err != nil {
				t.Fatalf("Failed to flush logs: %v", err)
			}

			cmpOpts := []cmp.Option{
				cmpopts.IgnoreUnexported(log.Value{}),
				cmpopts.IgnoreFields(DebugSpan{}, "StartTime", "EndTime", "TraceID", "SpanID", "ParentSpanID"),
				cmpopts.IgnoreFields(DebugLog{}, "ObservedTimestamp", "TraceID", "SpanID"),
				cmpopts.EquateEmpty(),
			}

			// Validate session spans
			gotSessionSpans := debugTelemetry.GetSpansBySessionID(tt.querySessionID)
			if diff := cmp.Diff(tt.wantSessionSpans, gotSessionSpans, cmpOpts...); diff != "" {
				t.Errorf("GetSpansBySessionID() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestDebugTelemetryGetSpansByEventID(t *testing.T) {
	ctx := context.Background()

	type testCase struct {
		name           string
		testSetup      func(ctx context.Context, tracer trace.Tracer, logger log.Logger)
		queryEventID   string
		wantEventSpans []DebugSpan
	}

	tests := []testCase{
		{
			name: "single span and log",
			testSetup: func(ctx context.Context, tracer trace.Tracer, logger log.Logger) {
				ctx, span := tracer.Start(ctx, "root-1", trace.WithAttributes(
					attribute.String("gcp.vertex.agent.event_id", "event-1"),
					attribute.String("genai.operation.name", "generate_content"),
				))
				defer span.End()

				var r log.Record
				r.SetBody(log.StringValue("test body"))
				r.SetEventName("test_event")
				r.SetTimestamp(time.Now())

				logger.Emit(ctx, r)
			},
			queryEventID: "event-1",
			wantEventSpans: []DebugSpan{
				{
					Name:         "root-1",
					ParentSpanID: trace.SpanID{}.String(),
					Attributes: map[string]string{
						"gcp.vertex.agent.event_id": "event-1",
						"genai.operation.name":      "generate_content",
					},
					Logs: []DebugLog{
						{
							Body:      "test body",
							EventName: "test_event",
						},
					},
				},
			},
		},
		{
			name: "multiple spans",
			testSetup: func(span1Ctx context.Context, tracer trace.Tracer, logger log.Logger) {
				span1Ctx, span1 := tracer.Start(span1Ctx, "root-1", trace.WithAttributes(
					attribute.String("gcp.vertex.agent.event_id", "event-1"),
					attribute.String("genai.operation.name", "generate_content"),
				))
				defer span1.End()

				_, span2 := tracer.Start(span1Ctx, "root-2", trace.WithAttributes(
					attribute.String("gcp.vertex.agent.event_id", "event-1"),
					attribute.String("genai.operation.name", "execute_tool"),
				))
				defer span2.End()
			},
			queryEventID: "event-1",
			wantEventSpans: []DebugSpan{
				{
					Name:         "root-1",
					ParentSpanID: trace.SpanID{}.String(),
					Attributes: map[string]string{
						"gcp.vertex.agent.event_id": "event-1",
						"genai.operation.name":      "generate_content",
					},
				},
				{
					Name:         "root-2",
					ParentSpanID: trace.SpanID{}.String(),
					Attributes: map[string]string{
						"gcp.vertex.agent.event_id": "event-1",
						"genai.operation.name":      "execute_tool",
					},
				},
			},
		},
		{
			name: "no matching span",
			testSetup: func(ctx context.Context, tracer trace.Tracer, logger log.Logger) {
				_, span := tracer.Start(ctx, "root-1", trace.WithAttributes(
					attribute.String("gcp.vertex.agent.event_id", "event-1"),
					attribute.String("genai.operation.name", "generate_content"),
				))
				span.End()
			},
			queryEventID:   "non-existent-event",
			wantEventSpans: nil,
		},
		{
			name: "log without span",
			testSetup: func(ctx context.Context, tracer trace.Tracer, logger log.Logger) {
				var r log.Record
				r.SetBody(log.StringValue("test body"))
				r.SetEventName("test_event")
				r.SetTimestamp(time.Now())

				logger.Emit(ctx, r)
			},
			queryEventID:   "event-1",
			wantEventSpans: nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			debugTelemetry, tp, lp := setup(t)

			if tt.testSetup != nil {
				tt.testSetup(ctx, tp.Tracer("test-tracer"), lp.Logger("test-logger"))
			}
			if err := tp.ForceFlush(ctx); err != nil {
				t.Fatalf("Failed to flush spans: %v", err)
			}
			if err := lp.ForceFlush(ctx); err != nil {
				t.Fatalf("Failed to flush logs: %v", err)
			}

			cmpOpts := []cmp.Option{
				cmpopts.IgnoreUnexported(log.Value{}),
				cmpopts.IgnoreFields(DebugSpan{}, "StartTime", "EndTime", "ParentSpanID", "TraceID", "SpanID"),
				cmpopts.IgnoreFields(DebugLog{}, "ObservedTimestamp", "TraceID", "SpanID"),
				cmpopts.EquateEmpty(),
			}

			// Validate event spans
			gotEventSpans := debugTelemetry.GetSpansByEventID(tt.queryEventID)
			if diff := cmp.Diff(tt.wantEventSpans, gotEventSpans, cmpOpts...); diff != "" {
				t.Errorf("GetSpansByEventID() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestDebugTelemetryLRU(t *testing.T) {
	ctx := t.Context()

	debugTelemetry, tp, lp := setupWithConfig(t, &DebugTelemetryConfig{TraceCapacity: 2})
	tracer := tp.Tracer("test-tracer")

	// 1. Add Trace 1.
	_, span1 := tracer.Start(ctx, "root-1", trace.WithAttributes(
		semconv.GenAIConversationID("session-1"),
		attribute.String("gcp.vertex.agent.event_id", "event-1"),
	))
	span1.End()

	// 2. Add Trace 2.
	_, span2 := tracer.Start(ctx, "root-2", trace.WithAttributes(
		semconv.GenAIConversationID("session-2"),
		attribute.String("gcp.vertex.agent.event_id", "event-2"),
	))
	span2.End()

	_ = tp.ForceFlush(ctx)
	_ = lp.ForceFlush(ctx)

	// 3. Verify both traces are present.
	if gotSpans := len(debugTelemetry.GetSpansBySessionID("session-1")); gotSpans != 1 {
		t.Errorf("expected 1 span for session-1, got %d", gotSpans)
	}
	if gotSpans := len(debugTelemetry.GetSpansBySessionID("session-2")); gotSpans != 1 {
		t.Errorf("expected 1 span for session-2, got %d", gotSpans)
	}

	// 4. Access session-2 making it the most recently used.
	_ = debugTelemetry.GetSpansBySessionID("session-2")

	// 5. Add Trace 3 - should evict Trace 1 because it's the least recently used.
	_, span3 := tracer.Start(ctx, "root-3", trace.WithAttributes(
		semconv.GenAIConversationID("session-3"),
		attribute.String("gcp.vertex.agent.event_id", "event-3"),
	))
	span3.End()

	// 6. Verify Trace 1 is evicted, Trace 2 and 3 are present.
	if gotSpans := len(debugTelemetry.GetSpansBySessionID("session-1")); gotSpans != 0 {
		t.Errorf("expected 0 spans for session-1, got %d", gotSpans)
	}
	if gotSpans := len(debugTelemetry.GetSpansBySessionID("session-2")); gotSpans != 1 {
		t.Errorf("expected 1 span for session-2, got %d", gotSpans)
	}
	if gotSpans := len(debugTelemetry.GetSpansBySessionID("session-3")); gotSpans != 1 {
		t.Errorf("expected 1 span for session-3, got %d", gotSpans)
	}

	// 7. Verify Trace 1 spans are removed from event index.
	if gotSpans := len(debugTelemetry.GetSpansByEventID("event-1")); gotSpans != 0 {
		t.Errorf("expected 0 spans for event-1, got %d", gotSpans)
	}

	// 8. Access Trace 2 via GetSpansByEventID, making it the most recently used.
	_ = debugTelemetry.GetSpansByEventID("event-2")

	// 9. Add Trace 4 - should evict Trace 3 because it's the least recently used.
	_, span4 := tracer.Start(ctx, "root-4", trace.WithAttributes(
		semconv.GenAIConversationID("session-4"),
		attribute.String("gcp.vertex.agent.event_id", "event-4"),
	))
	span4.End()

	// 10. Verify Trace 3 is evicted, Trace 2 and 4 are present.
	if gotSpans := len(debugTelemetry.GetSpansBySessionID("session-2")); gotSpans != 1 {
		t.Errorf("expected 1 span for session-2, got %d", gotSpans)
	}
	if gotSpans := len(debugTelemetry.GetSpansBySessionID("session-4")); gotSpans != 1 {
		t.Errorf("expected 1 span for session-4, got %d", gotSpans)
	}
	if gotSpans := len(debugTelemetry.GetSpansBySessionID("session-3")); gotSpans != 0 {
		t.Errorf("expected 0 spans for session-3, got %d", gotSpans)
	}
}

func setupWithConfig(t *testing.T, cfg *DebugTelemetryConfig) (*DebugTelemetry, *sdktrace.TracerProvider, *sdklog.LoggerProvider) {
	debugTelemetry, err := NewDebugTelemetryWithConfig(cfg)
	if err != nil {
		t.Fatalf("Failed to create debug telemetry: %v", err)
	}
	tp := sdktrace.NewTracerProvider(
		sdktrace.WithSpanProcessor(debugTelemetry.SpanProcessor()),
	)
	lp := sdklog.NewLoggerProvider(sdklog.WithProcessor(debugTelemetry.LogProcessor()))

	return debugTelemetry, tp, lp
}

func setup(t *testing.T) (*DebugTelemetry, *sdktrace.TracerProvider, *sdklog.LoggerProvider) {
	return setupWithConfig(t, nil)
}
