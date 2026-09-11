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
	"math"
	"net/http"
	"net/http/httptest"
	"regexp"
	"strings"
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

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/server/adkrest/controllers"
	"google.golang.org/adk/v2/server/adkrest/internal/fakes"
	"google.golang.org/adk/v2/server/adkrest/internal/services"
	"google.golang.org/adk/v2/session"
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
			testTelemetry := setupTestTelemetry(t)

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
			testTelemetry := setupTestTelemetry(t)

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

func setupTestTelemetry(t *testing.T) *testTelemetry {
	dt, err := services.NewDebugTelemetryWithConfig(nil)
	if err != nil {
		t.Fatalf("failed to create debug telemetry: %v", err)
	}

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
		attribute.MapValue(
			attribute.KeyValue{
				Key:   "message",
				Value: attribute.StringValue("test log message"),
			},
		),
	)
	logger.Emit(ctx, record)

	span.End()

	_ = tp.ForceFlush(context.Background())
	_ = lp.ForceFlush(context.Background())
}

// emitSpanWithoutLogs records a span that emits no logs, leaving the span record's
// log slice nil.
func emitSpanWithoutLogs(sessionID, eventID, opName string, tp *sdktrace.TracerProvider) {
	tracer := tp.Tracer("test-tracer")
	_, span := tracer.Start(context.Background(), "test-span-without-logs", trace.WithAttributes(
		attribute.String("gcp.vertex.agent.event_id", eventID),
		attribute.String(string(semconv.GenAIConversationIDKey), sessionID),
		attribute.String(string(semconv.GenAIOperationNameKey), opName),
	))
	span.End()
	_ = tp.ForceFlush(context.Background())
}

// TestSessionSpansHandlerEmptyContainersSerializeAsJSONContainers asserts on the raw
// response bytes.
//
// The ADK web UI validates the trace response against array and object schemas. A
// Go nil slice marshals to JSON null, which fails those schemas, so the UI discards
// the whole response and the Traces panel renders nothing even though the request
// returned 200. Decoding the body into Go values hides the difference between null
// and [], so these assertions read the serialised bytes directly.
func TestSessionSpansHandlerEmptyContainersSerializeAsJSONContainers(t *testing.T) {
	tc := []struct {
		name         string
		emitSpan     bool
		wantContains []string
		wantExact    string
	}{
		{
			name:     "span_without_logs_serializes_logs_as_empty_array",
			emitSpan: true,
			wantContains: []string{
				`"logs":[]`,
				`"attributes":{`,
			},
		},
		{
			name:      "session_without_spans_serializes_as_empty_array",
			emitSpan:  false,
			wantExact: "[]",
		},
	}

	for _, tt := range tc {
		t.Run(tt.name, func(t *testing.T) {
			const sessionID = "test-session"
			testTelemetry := setupTestTelemetry(t)
			if tt.emitSpan {
				emitSpanWithoutLogs(sessionID, "test-event", semconv.GenAIOperationNameExecuteTool.Value.AsString(), testTelemetry.tp)
			}

			apiController := controllers.NewDebugAPIController(nil, nil, testTelemetry.dt)
			req := httptest.NewRequest(http.MethodGet, "/debug/trace/session/"+sessionID, nil)
			req = mux.SetURLVars(req, map[string]string{"session_id": sessionID})
			rr := httptest.NewRecorder()

			apiController.SessionSpansHandler(rr, req)

			if rr.Code != http.StatusOK {
				t.Fatalf("status = %d, want %d", rr.Code, http.StatusOK)
			}
			raw := strings.TrimSpace(rr.Body.String())

			if tt.wantExact != "" && raw != tt.wantExact {
				t.Errorf("raw body = %s, want %s", raw, tt.wantExact)
			}
			for _, want := range tt.wantContains {
				if !strings.Contains(raw, want) {
					t.Errorf("raw body does not contain %s:\n%s", want, raw)
				}
			}
			if strings.Contains(raw, "null") {
				t.Errorf("raw body contains a JSON null, which the web UI rejects:\n%s", raw)
			}
		})
	}
}

// TestEventSpanHandlerEmptyLogsSerializeAsJSONArray checks the other serialisation
// path: the flattened per-event span, built by convertEventSpan.
func TestEventSpanHandlerEmptyLogsSerializeAsJSONArray(t *testing.T) {
	const (
		sessionID = "test-session"
		eventID   = "test-event"
	)
	testTelemetry := setupTestTelemetry(t)
	emitSpanWithoutLogs(sessionID, eventID, semconv.GenAIOperationNameGenerateContent.Value.AsString(), testTelemetry.tp)

	apiController := controllers.NewDebugAPIController(nil, nil, testTelemetry.dt)
	req := httptest.NewRequest(http.MethodGet, "/debug/trace/"+eventID, nil)
	req = mux.SetURLVars(req, map[string]string{"event_id": eventID})
	rr := httptest.NewRecorder()

	apiController.EventSpanHandler(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d (body %q)", rr.Code, http.StatusOK, rr.Body.String())
	}
	raw := strings.TrimSpace(rr.Body.String())
	if !strings.Contains(raw, `"logs":[]`) {
		t.Errorf(`raw body does not contain "logs":[]:`+"\n%s", raw)
	}
	if strings.Contains(raw, "null") {
		t.Errorf("raw body contains a JSON null, which the web UI rejects:\n%s", raw)
	}
}

// emitElidedMessageLogs records a span carrying the two log records the server
// emits with OpenTelemetry content capture off, which is the default: both bodies
// hold the elided placeholder instead of the message.
func emitElidedMessageLogs(sessionID, eventID, opName string, tp *sdktrace.TracerProvider, lp *sdklog.LoggerProvider) {
	ctx, span := tp.Tracer("test-tracer").Start(context.Background(), "test-span", trace.WithAttributes(
		attribute.String("gcp.vertex.agent.event_id", eventID),
		attribute.String(string(semconv.GenAIConversationIDKey), sessionID),
		attribute.String(string(semconv.GenAIOperationNameKey), opName),
	))
	for _, eventName := range []string{"gen_ai.system.message", "gen_ai.user.message"} {
		var r log.Record
		r.SetEventName(eventName)
		r.SetBody(attribute.MapValue(attribute.String("content", "<elided>")))
		r.SetTimestamp(time.Now())
		r.SetObservedTimestamp(time.Now())
		lp.Logger("test-logger").Emit(ctx, r)
	}
	span.End()
	_ = tp.ForceFlush(context.Background())
	_ = lp.ForceFlush(context.Background())
}

// volatileJSONFields matches the response fields that change between runs, so a
// test can compare the rest of the raw bytes exactly.
var volatileJSONFields = regexp.MustCompile(`"(start_time|end_time)":\d+|"(trace_id|span_id|parent_span_id)":"[0-9a-f]*"|"observed_timestamp":"[^"]*"`)

func maskVolatileJSONFields(raw string) string {
	return volatileJSONFields.ReplaceAllStringFunc(raw, func(match string) string {
		name, _, _ := strings.Cut(strings.TrimPrefix(match, `"`), `"`)
		return `"` + name + `":"masked"`
	})
}

// TestDebugHandlersSerializeElidedContentAsObject asserts on the raw response bytes.
//
// With content capture off the server records body.content as the string
// "<elided>". The ADK web UI requires an object holding a "parts" array for
// gen_ai.user.message and gen_ai.choice records, and it validates the whole span
// array in one pass, so that one string makes it discard every trace and render
// an empty Traces panel. A gen_ai.system.message keeps a plain string, which is
// what the UI wants there. Decoding into Go values hides a string that should be
// an object, so these assertions read the bytes. encoding/json escapes "<" and
// ">", hence the \u003c and \u003e below.
func TestDebugHandlersSerializeElidedContentAsObject(t *testing.T) {
	const (
		sessionID = "test-session"
		eventID   = "test-event"
	)
	opName := semconv.GenAIOperationNameGenerateContent.Value.AsString()

	wantLogs := `"logs":[` +
		`{"body":{"content":"\u003celided\u003e"},"observed_timestamp":"masked","trace_id":"masked","span_id":"masked","event_name":"gen_ai.system.message"},` +
		`{"body":{"content":{"parts":[{"text":"\u003celided\u003e"}],"role":"user"}},"observed_timestamp":"masked","trace_id":"masked","span_id":"masked","event_name":"gen_ai.user.message"}]`

	tc := []struct {
		name    string
		handler func(*controllers.DebugAPIController, http.ResponseWriter, *http.Request)
		vars    map[string]string
		want    string
	}{
		{
			name: "session_spans",
			handler: func(c *controllers.DebugAPIController, rw http.ResponseWriter, req *http.Request) {
				c.SessionSpansHandler(rw, req)
			},
			vars: map[string]string{"session_id": sessionID},
			want: `[{"name":"test-span","start_time":"masked","end_time":"masked","span_id":"masked","trace_id":"masked","parent_span_id":"masked",` +
				`"attributes":{"gcp.vertex.agent.event_id":"test-event","gen_ai.conversation.id":"test-session","gen_ai.operation.name":"generate_content"},` +
				wantLogs + `}]`,
		},
		{
			name: "event_span",
			handler: func(c *controllers.DebugAPIController, rw http.ResponseWriter, req *http.Request) {
				c.EventSpanHandler(rw, req)
			},
			vars: map[string]string{"event_id": eventID},
			want: `{"end_time":"masked","gcp.vertex.agent.event_id":"test-event","gen_ai.conversation.id":"test-session","gen_ai.operation.name":"generate_content",` +
				wantLogs + `,"name":"test-span","parent_span_id":"masked","span_id":"masked","start_time":"masked","trace_id":"masked"}`,
		},
	}

	for _, tt := range tc {
		t.Run(tt.name, func(t *testing.T) {
			testTelemetry := setupTestTelemetry(t)
			emitElidedMessageLogs(sessionID, eventID, opName, testTelemetry.tp, testTelemetry.lp)

			apiController := controllers.NewDebugAPIController(nil, nil, testTelemetry.dt)
			req := httptest.NewRequest(http.MethodGet, "/debug/trace", nil)
			req = mux.SetURLVars(req, tt.vars)
			rr := httptest.NewRecorder()

			tt.handler(apiController, rr, req)

			if rr.Code != http.StatusOK {
				t.Fatalf("status = %d, want %d (body %q)", rr.Code, http.StatusOK, rr.Body.String())
			}
			if got := maskVolatileJSONFields(strings.TrimSpace(rr.Body.String())); got != tt.want {
				t.Errorf("raw body =\n%s\nwant\n%s", got, tt.want)
			}
		})
	}
}

// TestSessionSpansHandlerUnrepresentableLogKeepsOtherSpans asserts on the raw
// response bytes.
//
// The handler encodes every span of the session in one call, so a body that
// encoding/json refuses truncates the response and blanks every trace in the
// panel. The record is repaired into a placeholder instead, and the other spans
// come back untouched.
func TestSessionSpansHandlerUnrepresentableLogKeepsOtherSpans(t *testing.T) {
	const sessionID = "test-session"
	// Guard: the test is only meaningful while this body really is unmarshalable.
	if _, err := json.Marshal(math.NaN()); err == nil {
		t.Fatal("json.Marshal(NaN) succeeded, pick another unrepresentable body")
	}

	testTelemetry := setupTestTelemetry(t)
	emitElidedMessageLogs(sessionID, "good-event", semconv.GenAIOperationNameGenerateContent.Value.AsString(), testTelemetry.tp, testTelemetry.lp)

	ctx, badSpan := testTelemetry.tp.Tracer("test-tracer").Start(context.Background(), "bad-span", trace.WithAttributes(
		attribute.String(string(semconv.GenAIConversationIDKey), sessionID),
	))
	var r log.Record
	r.SetEventName("gen_ai.user.message")
	r.SetBody(attribute.MapValue(
		attribute.String("content", "<elided>"),
		attribute.Float64("index", math.NaN()),
	))
	r.SetTimestamp(time.Now())
	r.SetObservedTimestamp(time.Now())
	testTelemetry.lp.Logger("test-logger").Emit(ctx, r)
	badSpan.End()
	_ = testTelemetry.tp.ForceFlush(context.Background())
	_ = testTelemetry.lp.ForceFlush(context.Background())

	apiController := controllers.NewDebugAPIController(nil, nil, testTelemetry.dt)
	req := httptest.NewRequest(http.MethodGet, "/debug/trace/session/"+sessionID, nil)
	req = mux.SetURLVars(req, map[string]string{"session_id": sessionID})
	rr := httptest.NewRecorder()

	apiController.SessionSpansHandler(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d", rr.Code, http.StatusOK)
	}
	raw := strings.TrimSpace(rr.Body.String())
	var spans []map[string]any
	if err := json.Unmarshal([]byte(raw), &spans); err != nil {
		t.Fatalf("response is not valid JSON, the bad record truncated it: %v\n%s", err, raw)
	}
	if len(spans) != 2 {
		t.Errorf("got %d spans, want 2 (test-span and bad-span):\n%s", len(spans), raw)
	}
	wantGood := `"body":{"content":{"parts":[{"text":"\u003celided\u003e"}],"role":"user"}}`
	if !strings.Contains(raw, wantGood) {
		t.Errorf("the good span lost its body: want\n%s\ngot\n%s", wantGood, raw)
	}
	wantRepaired := `"body":{"content":{"parts":[{"text":"\u003cunrepresentable\u003e"}],"role":"user"}}`
	if !strings.Contains(raw, wantRepaired) {
		t.Errorf("the bad record was not repaired: want\n%s\ngot\n%s", wantRepaired, raw)
	}
}

// TestEventGraphHandlerResponseShape pins the event graph response, which is a
// single-key object of scalars and so is unaffected by the nil-container fixes.
func TestEventGraphHandlerResponseShape(t *testing.T) {
	const (
		appName   = "test-app"
		userID    = "test-user"
		sessionID = "test-session"
		eventID   = "test-event"
	)

	rootAgent, err := agent.New(agent.Config{Name: appName, Description: "graph test agent"})
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}
	key := fakes.SessionKey{AppName: appName, UserID: userID, SessionID: sessionID}
	sessionService := &fakes.FakeSessionService{
		Sessions: map[fakes.SessionKey]fakes.TestSession{
			key: {
				Id:           key,
				SessionState: fakes.TestState{},
				SessionEvents: fakes.TestEvents{
					&session.Event{ID: eventID, Author: appName},
				},
				UpdatedAt: time.Now(),
			},
		},
	}

	apiController := controllers.NewDebugAPIController(sessionService, agent.NewSingleLoader(rootAgent), nil)
	req := httptest.NewRequest(http.MethodGet, "/events/"+eventID+"/graph", nil)
	req = mux.SetURLVars(req, map[string]string{
		"app_name":   appName,
		"user_id":    userID,
		"session_id": sessionID,
		"event_id":   eventID,
	})
	rr := httptest.NewRecorder()

	apiController.EventGraphHandler(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d (body %q)", rr.Code, http.StatusOK, rr.Body.String())
	}
	var got map[string]any
	if err := json.Unmarshal(rr.Body.Bytes(), &got); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if len(got) != 1 {
		t.Errorf("response has %d keys, want exactly 1 (dotSrc): %v", len(got), got)
	}
	dotSrc, ok := got["dotSrc"].(string)
	if !ok {
		t.Fatalf("dotSrc = %v (%T), want a string", got["dotSrc"], got["dotSrc"])
	}
	if !strings.Contains(dotSrc, "digraph") {
		t.Errorf("dotSrc is not DOT source: %q", dotSrc)
	}
	if strings.Contains(rr.Body.String(), "null") {
		t.Errorf("raw body contains a JSON null:\n%s", rr.Body.String())
	}
}
