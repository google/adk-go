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

package controllers_test

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/gorilla/mux"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/log"
	sdklog "go.opentelemetry.io/otel/sdk/log"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	semconv "go.opentelemetry.io/otel/semconv/v1.36.0"
	"go.opentelemetry.io/otel/trace"

	"google.golang.org/adk/server/adkrest/controllers"
	"google.golang.org/adk/server/adkrest/internal/services"
)

func TestSessionSpansHandler(t *testing.T) {
	tc := []struct {
		name          string
		sessionID     string
		reqSessionID  string
		wantStatus    int
		wantSpanCount int
	}{
		{
			name:          "spans found for session",
			sessionID:     "test-session",
			reqSessionID:  "test-session",
			wantStatus:    http.StatusOK,
			wantSpanCount: 1,
		},
		{
			name:          "spans not found for session",
			sessionID:     "test-session",
			reqSessionID:  "other-session",
			wantStatus:    http.StatusOK,
			wantSpanCount: 0,
		},
		{
			name:          "empty session id param",
			sessionID:     "test-session",
			reqSessionID:  "",
			wantStatus:    http.StatusBadRequest,
			wantSpanCount: 0,
		},
	}

	for _, tt := range tc {
		t.Run(tt.name, func(t *testing.T) {
			eventID := "test-event"
			opName := semconv.GenAIOperationNameExecuteTool.Value.AsString()
			testTelemetry := setupTestTelemetry()

			apiController := controllers.NewDebugAPIController(nil, nil, testTelemetry.dt)
			req, err := http.NewRequest(http.MethodGet, "/debug/sessions/"+tt.reqSessionID+"/spans", nil)
			if err != nil {
				t.Fatalf("new request: %v", err)
			}

			req = mux.SetURLVars(req, map[string]string{
				"session_id": tt.reqSessionID,
			})
			rr := httptest.NewRecorder()

			emitTestSignals(tt.sessionID, eventID, opName, testTelemetry.tp, testTelemetry.lp)
			apiController.SessionSpansHandler(rr, req)

			if gotStatus := rr.Code; gotStatus != tt.wantStatus {
				t.Fatalf("handler returned wrong status code: got %v want %v", gotStatus, tt.wantStatus)
			}

			if tt.wantStatus == http.StatusOK {
				var result controllers.SessionTelemetry
				err = json.NewDecoder(rr.Body).Decode(&result)
				if err != nil {
					t.Fatalf("decode response: %v", err)
				}

				if result.SchemaVersion != 2 {
					t.Errorf("got schema_version %d, want 2", result.SchemaVersion)
				}

				if len(result.Spans) != tt.wantSpanCount {
					t.Fatalf("got %d spans, want %d", len(result.Spans), tt.wantSpanCount)
				}
			}
		})
	}
}

func TestEventSpanHandler(t *testing.T) {
	tc := []struct {
		name       string
		eventID    string
		reqEventID string
		opName     string
		wantStatus int
	}{
		{
			name:       "span with generate content operation",
			eventID:    "test-event",
			reqEventID: "test-event",
			opName:     semconv.GenAIOperationNameGenerateContent.Value.AsString(),
			wantStatus: http.StatusOK,
		},
		{
			name:       "span with execute tool operation",
			eventID:    "test-event",
			reqEventID: "test-event",
			opName:     semconv.GenAIOperationNameExecuteTool.Value.AsString(),
			wantStatus: http.StatusOK,
		},
		{
			name:       "span not found for event id",
			eventID:    "test-event",
			reqEventID: "other-event",
			opName:     semconv.GenAIOperationNameExecuteTool.Value.AsString(),
			wantStatus: http.StatusNotFound,
		},
		{
			name:       "span with different operation name",
			eventID:    "test-event",
			reqEventID: "test-event",
			opName:     "other-op",
			wantStatus: http.StatusNotFound,
		},
		{
			name:       "empty event id param",
			eventID:    "test-event",
			reqEventID: "",
			opName:     semconv.GenAIOperationNameExecuteTool.Value.AsString(),
			wantStatus: http.StatusBadRequest,
		},
	}

	for _, tt := range tc {
		t.Run(tt.name, func(t *testing.T) {
			sessionID := "test-session"
			testTelemetry := setupTestTelemetry()

			apiController := controllers.NewDebugAPIController(nil, nil, testTelemetry.dt)
			req, err := http.NewRequest(http.MethodGet, "/debug/events/"+tt.reqEventID+"/span", nil)
			if err != nil {
				t.Fatalf("new request: %v", err)
			}

			req = mux.SetURLVars(req, map[string]string{
				"event_id": tt.reqEventID,
			})
			rr := httptest.NewRecorder()

			emitTestSignals(sessionID, tt.eventID, tt.opName, testTelemetry.tp, testTelemetry.lp)
			apiController.EventSpanHandler(rr, req)

			if status := rr.Code; status != tt.wantStatus {
				t.Fatalf("handler returned wrong status code: got %v want %v", status, tt.wantStatus)
			}

			if tt.wantStatus == http.StatusOK {
				var gotSpan services.DebugSpan
				err = json.NewDecoder(rr.Body).Decode(&gotSpan)
				if err != nil {
					t.Fatalf("decode response: %v", err)
				}

				if gotSpan.Name != "test-span" {
					t.Errorf("got span name %q, want %q", gotSpan.Name, "test-span")
				}
				if gotSpan.Attributes["gcp.vertex.agent.event_id"] != tt.eventID {
					t.Errorf("got event_id %q, want %q", gotSpan.Attributes["gcp.vertex.agent.event_id"], tt.eventID)
				}
				if len(gotSpan.Logs) != 1 {
					t.Fatalf("got %d logs, want 1", len(gotSpan.Logs))
				}
				if gotSpan.Logs[0].EventName != "test-log-event" {
					t.Errorf("got log event name %q, want %q", gotSpan.Logs[0].EventName, "test-log-event")
				}
				if gotSpan.Logs[0].Body != "test log message" {
					t.Errorf("got log body %v, want %q", gotSpan.Logs[0].Body, "test log message")
				}
			}
		})
	}
}

type testTelemetry struct {
	dt     *services.DebugTelemetry
	tracer trace.Tracer
	tp     *sdktrace.TracerProvider
	logger log.Logger
	lp     *sdklog.LoggerProvider
}

func setupTestTelemetry() *testTelemetry {
	dt := services.NewDebugTelemetry()

	tp := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(dt.SpanProcessor()))
	lp := sdklog.NewLoggerProvider(sdklog.WithProcessor(dt.LogProcessor()))

	tracer := tp.Tracer("test-tracer")
	logger := lp.Logger("test-logger")

	return &testTelemetry{
		dt:     dt,
		tracer: tracer,
		tp:     tp,
		logger: logger,
		lp:     lp,
	}
}

func emitTestSignals(sessionID, eventID, opName string, tp *sdktrace.TracerProvider, lp *sdklog.LoggerProvider) {
	tracer := tp.Tracer("test-tracer")
	logger := lp.Logger("test-logger")

	ctx, span := tracer.Start(context.Background(), "test-span", trace.WithAttributes(
		attribute.String("gcp.vertex.agent.event_id", eventID),
		attribute.String(string(semconv.GenAIConversationIDKey), sessionID),
		attribute.String(string(semconv.GenAIOperationNameKey), opName),
	))

	var record log.Record
	record.SetTimestamp(time.Now())
	record.SetObservedTimestamp(time.Now())
	record.SetEventName("test-log-event")
	record.SetBody(log.StringValue("test log message"))
	logger.Emit(ctx, record)

	span.End()

	_ = tp.ForceFlush(context.Background())
	_ = lp.ForceFlush(context.Background())
}
