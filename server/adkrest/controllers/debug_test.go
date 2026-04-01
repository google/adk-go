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

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
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
		name         string
		sessionID    string
		reqSessionID string
		wantStatus   int
		wantBody     []map[string]any
	}{
		{
			name:         "spans_found_for_session",
			sessionID:    "test-session",
			reqSessionID: "test-session",
			wantStatus:   http.StatusOK,
			wantBody: []map[string]any{
				{
					"name":           "test-span",
					"start_time":     "test-time",
					"end_time":       "test-time",
					"trace_id":       "test-trace-id",
					"span_id":        "test-span-id",
					"parent_span_id": "test-parent-span-id",
					"attributes": map[string]any{
						"gcp.vertex.agent.event_id": "test-event",
						"gen_ai.conversation.id":    "test-session",
						"gen_ai.operation.name":     "execute_tool",
					},
					"logs": []any{
						map[string]any{
							"event_name": "test-log-event",
							"body": map[string]any{
								"message": "test log message",
							},
						},
					},
				},
			},
		},
		{
			name:         "spans_not_found_for_session",
			sessionID:    "test-session",
			reqSessionID: "other-session",
			wantStatus:   http.StatusOK,
			wantBody:     []map[string]any{},
		},
		{
			name:         "empty_session_id_param",
			sessionID:    "test-session",
			reqSessionID: "",
			wantStatus:   http.StatusBadRequest,
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
				var result []map[string]any
				err = json.NewDecoder(rr.Body).Decode(&result)
				if err != nil {
					t.Fatalf("decode response: %v", err)
				}

				if diff := cmp.Diff(tt.wantBody, result, ignoreDynamicFields()); diff != "" {
					t.Errorf("handler returned unexpected body (-want +got):\n%s", diff)
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
		wantBody   map[string]any
	}{
		{
			name:       "span_with_generate_content_operation",
			eventID:    "test-event",
			reqEventID: "test-event",
			opName:     semconv.GenAIOperationNameGenerateContent.Value.AsString(),
			wantStatus: http.StatusOK,
			wantBody: map[string]any{
				"name":                      "test-span",
				"gcp.vertex.agent.event_id": "test-event",
				"gen_ai.conversation.id":    "test-session",
				"gen_ai.operation.name":     semconv.GenAIOperationNameGenerateContent.Value.AsString(),
				"logs": []any{
					map[string]any{
						"event_name": "test-log-event",
						"body": map[string]any{
							"message": "test log message",
						},
					},
				},
			},
		},
		{
			name:       "span_with_execute_tool_operation",
			eventID:    "test-event",
			reqEventID: "test-event",
			opName:     semconv.GenAIOperationNameExecuteTool.Value.AsString(),
			wantStatus: http.StatusOK,
			wantBody: map[string]any{
				"name":                      "test-span",
				"gcp.vertex.agent.event_id": "test-event",
				"gen_ai.conversation.id":    "test-session",
				"gen_ai.operation.name":     semconv.GenAIOperationNameExecuteTool.Value.AsString(),
				"logs": []any{
					map[string]any{
						"event_name": "test-log-event",
						"body": map[string]any{
							"message": "test log message",
						},
					},
				},
			},
		},
		{
			name:       "span_not_found_for_event_id",
			eventID:    "test-event",
			reqEventID: "other-event",
			opName:     semconv.GenAIOperationNameExecuteTool.Value.AsString(),
			wantStatus: http.StatusNotFound,
		},
		{
			name:       "span_with_different_operation_name",
			eventID:    "test-event",
			reqEventID: "test-event",
			opName:     "other-op",
			wantStatus: http.StatusNotFound,
		},
		{
			name:       "empty_event_id_param",
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
				var gotBody map[string]any
				err = json.NewDecoder(rr.Body).Decode(&gotBody)
				if err != nil {
					t.Fatalf("decode response: %v", err)
				}

				if diff := cmp.Diff(tt.wantBody, gotBody, ignoreDynamicFields()); diff != "" {
					t.Errorf("handler returned unexpected body (-want +got):\n%s", diff)
				}
			}
		})
	}
}

func ignoreDynamicFields() cmp.Option {
	return cmpopts.IgnoreMapEntries(func(k string, v any) bool {
		switch k {
		case "end_time", "observed_timestamp", "span_id", "start_time", "trace_id", "parent_span_id":
			return true
		default:
			return false
		}
	})
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
	record.SetBody(
		log.MapValue(
			log.KeyValue{
				Key:   "message",
				Value: log.StringValue("test log message"),
			},
		),
	)
	logger.Emit(ctx, record)

	span.End()

	_ = tp.ForceFlush(context.Background())
	_ = lp.ForceFlush(context.Background())
}
