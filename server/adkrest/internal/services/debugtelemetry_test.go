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
			name: "nested-span-with-log",
			testSetup: func(ctx context.Context, tracer trace.Tracer, logger log.Logger) {
				ctx, span := tracer.Start(ctx, "parent-span", trace.WithAttributes(
					attribute.String(string(semconv.GenAIConversationIDKey), "session-1"),
					attribute.String("gcp.vertex.agent.event_id", "parent-event"),
				))
				defer span.End()

				childSpanCtx, childSpan := tracer.Start(ctx, "child-span", trace.WithAttributes(
					attribute.String(string(semconv.GenAIConversationIDKey), "session-1"),
					attribute.String("gcp.vertex.agent.event_id", "child-event"),
				))
				r2 := log.Record{}
				r2.SetBody(log.StringValue("child-log-body"))
				r2.SetEventName("child-log-event")
				r2.SetTimestamp(time.Now())
				logger.Emit(childSpanCtx, r2)
				childSpan.End()

				r1 := log.Record{}
				r1.SetBody(log.StringValue("parent-log-body"))
				r1.SetEventName("parent-log-event")
				r1.SetTimestamp(time.Now())

				logger.Emit(ctx, r1)
			},
			querySessionID: "session-1",
			wantSessionSpans: []DebugSpan{
				{
					Name:         "child-span",
					ParentSpanID: trace.SpanID{}.String(),
					Attributes: map[string]string{
						string(semconv.GenAIConversationIDKey): "session-1",
						"gcp.vertex.agent.event_id":            "child-event",
					},
					Logs: []DebugLog{
						{
							Body:      "child-log-body",
							EventName: "child-log-event",
						},
					},
				},
				{
					Name:         "parent-span",
					ParentSpanID: trace.SpanID{}.String(),
					Attributes: map[string]string{
						string(semconv.GenAIConversationIDKey): "session-1",
						"gcp.vertex.agent.event_id":            "parent-event",
					},
					Logs: []DebugLog{
						{
							Body:      "parent-log-body",
							EventName: "parent-log-event",
						},
					},
				},
			},
		},
		{
			name: "empty-results",
			testSetup: func(ctx context.Context, tracer trace.Tracer, logger log.Logger) {
				_, span := tracer.Start(ctx, "test-span-1", trace.WithAttributes(
					attribute.String(string(semconv.GenAIConversationIDKey), "session-1"),
					attribute.String("gcp.vertex.agent.event_id", "event-1"),
				))
				defer span.End()
			},
			querySessionID:   "non-existent-session",
			wantSessionSpans: nil,
		},
		{
			name: "log without span",
			testSetup: func(ctx context.Context, tracer trace.Tracer, logger log.Logger) {
				var rec1 log.Record
				rec1.SetBody(log.StringValue("test body"))
				rec1.SetEventName("test_event")
				rec1.SetTimestamp(time.Now())

				logger.Emit(ctx, rec1)
			},
			querySessionID:   "session-1",
			wantSessionSpans: nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			debugTelemetry, tp, lp := setup()

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
				cmpopts.IgnoreFields(DebugSpan{}, "StartTime", "EndTime", "Context", "ParentSpanID"),
				cmpopts.IgnoreFields(DebugLog{}, "ObservedTimestamp", "TraceID", "SpanID"),
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

	spanName := "test-span-1"

	type testCase struct {
		name           string
		testSetup      func(ctx context.Context, tracer trace.Tracer, logger log.Logger)
		queryEventID   string
		wantEventSpans []DebugSpan
	}

	tests := []testCase{
		{
			name: "single-span-and-log",
			testSetup: func(ctx context.Context, tracer trace.Tracer, logger log.Logger) {
				ctx, span := tracer.Start(ctx, "test-span-1", trace.WithAttributes(
					attribute.String(string(semconv.GenAIConversationIDKey), "session-1"),
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
					Name:         "test-span-1",
					ParentSpanID: trace.SpanID{}.String(),
					Attributes: map[string]string{
						string(semconv.GenAIConversationIDKey): "session-1",
						"gcp.vertex.agent.event_id":            "event-1",
						"genai.operation.name":                 "generate_content",
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
			name: "empty-results",
			testSetup: func(ctx context.Context, tracer trace.Tracer, logger log.Logger) {
				_, span := tracer.Start(ctx, spanName, trace.WithAttributes(
					attribute.String(string(semconv.GenAIConversationIDKey), "session-1"),
					attribute.String("gcp.vertex.agent.event_id", "event-1"),
					attribute.String("genai.operation.name", "generate_content"),
				))
				defer span.End()
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
			debugTelemetry, tp, lp := setup()

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
				cmpopts.IgnoreFields(DebugSpan{}, "StartTime", "EndTime", "Context", "ParentSpanID"),
				cmpopts.IgnoreFields(DebugLog{}, "ObservedTimestamp", "TraceID", "SpanID"),
			}

			// Validate event spans
			gotEventSpans := debugTelemetry.GetSpansByEventID(tt.queryEventID)
			if diff := cmp.Diff(tt.wantEventSpans, gotEventSpans, cmpOpts...); diff != "" {
				t.Errorf("GetSpansByEventID() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func setup() (*DebugTelemetry, *sdktrace.TracerProvider, *sdklog.LoggerProvider) {
	debugTelemetry := NewDebugTelemetry()
	tp := sdktrace.NewTracerProvider(
		sdktrace.WithSpanProcessor(debugTelemetry.SpanProcessor()),
	)
	lp := sdklog.NewLoggerProvider(sdklog.WithProcessor(debugTelemetry.LogProcessor()))

	return debugTelemetry, tp, lp
}
