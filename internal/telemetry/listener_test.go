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

package telemetry

import (
	"context"
	"fmt"
	"io"
	"iter"
	stdlog "log"
	"math"
	"reflect"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/log"
	sdklog "go.opentelemetry.io/otel/sdk/log"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	semconv "go.opentelemetry.io/otel/semconv/v1.36.0"
	"go.opentelemetry.io/otel/trace"
	tracenoop "go.opentelemetry.io/otel/trace/noop"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
)

// The ref* functions are the telemetry as it was before it asked whether a
// logger takes its records and whether a span records: everything is built and
// handed over. The tests below hold the current functions to them for every
// consumer that is listening.

type refOperationDetails struct {
	span         trace.Span
	request      []contentAttr
	eventRequest []attribute.KeyValue
}

func (d *refOperationDetails) noteLLMRequest(req *model.LLMRequest) {
	if useLegacySchema() {
		return
	}
	onSpan := d.span.IsRecording() && captureContentOnSpans()
	onEvent := getGenAICaptureMessageContent()
	if !onSpan && !onEvent {
		return
	}
	content := requestContent(req)
	if reflect.DeepEqual(content, d.request) {
		return
	}
	d.request = content
	if onSpan {
		d.span.SetAttributes(spanContentAttributes(content)...)
	}
	if onEvent {
		d.eventRequest = eventContentAttributes(content)
	}
}

func (d *refOperationDetails) emit(ctx context.Context, params GenerateContentParams, result generateContentResult) {
	if useLegacySchema() {
		return
	}
	attrs := []attribute.KeyValue{
		semconv.GenAIOperationNameGenerateContent,
		semconv.GenAIRequestModel(params.ModelName),
		gcpVertexAgentInvocationID.String(params.InvocationID),
	}
	if provider, ok := providerName(params.ModelName, params.Backend); ok {
		attrs = append(attrs, provider)
	}
	if result.Error != nil {
		attrs = append(attrs, errorType(result.Error))
	}
	attrs = append(attrs, generateContentResultAttributes(result)...)
	if getGenAICaptureMessageContent() {
		attrs = append(attrs, d.eventRequest...)
		attrs = append(attrs, eventContentAttributes(responseContent(result))...)
	}
	record := log.Record{}
	record.SetEventName(inferenceOperationDetailsEventName)
	record.SetSeverity(log.SeverityDebug)
	record.AddAttributes(attrs...)
	otelLogger().Emit(ctx, record)
}

func refInstrumentGenerateContent[T any](ctx context.Context, params GenerateContentParams, response func(T) (*model.LLMResponse, string), call func(context.Context) iter.Seq2[T, error]) iter.Seq2[T, error] {
	return func(yield func(T, error) bool) {
		ctx, span := startGenerateContentSpan(ctx, params)
		refLogRequest(ctx, params.Request, params.Backend)
		details := refOperationDetails{span: span}
		details.noteLLMRequest(params.Request)

		result := generateContentResult{RequestModel: params.ModelName}
		ended := false
		end := func() {
			if ended {
				return
			}
			ended = true
			refTraceGenerateContentResult(span, result)
			details.noteLLMRequest(params.Request)
			details.emit(ctx, params, result)
			span.End()
		}
		defer end()
		legacy := useLegacySchema()
		for v, err := range call(ctx) {
			resp, eventID := response(v)
			switch {
			case legacy:
				result = generateContentResult{Response: resp, EventID: eventID, Error: err, RequestModel: params.ModelName}
				if err != nil {
					end()
				} else if !resp.Partial {
					refLogResponse(ctx, resp, params.Backend)
					end()
				}
			case ended || (resp == nil && err == nil):
			default:
				result.Error = err
				if resp != nil {
					result.Response, result.EventID = resp, eventID
					result.Responses = append(result.Responses, resp)
				}
				if err != nil || reportedFinishReason(resp, nil) != "" || !params.Stream {
					end()
				}
			}
			if !yield(v, err) {
				return
			}
		}
	}
}

func refTraceGenerateContentResult(span trace.Span, params generateContentResult) {
	RecordError(span, params.Error)
	span.SetAttributes(generateContentResultAttributes(params)...)
	if captureContentOnSpans() {
		span.SetAttributes(spanContentAttributes(responseContent(params))...)
	}
}

func refTraceToolResult(span trace.Span, params TraceToolResultParams) {
	RecordError(span, params.Error)
	attributes := []attribute.KeyValue{
		semconv.GenAIOperationNameKey.String(executeToolName),
		semconv.GenAIToolDescriptionKey.String(params.Description),
	}
	var functionResponse *genai.FunctionResponse
	if params.ResponseEvent != nil {
		attributes = append(attributes, gcpVertexAgentEventID.String(params.ResponseEvent.ID))
		if c := params.ResponseEvent.LLMResponse.Content; c != nil && len(c.Parts) > 0 {
			functionResponse = c.Parts[0].FunctionResponse
		}
	}
	if !useLegacySchema() {
		if functionResponse != nil {
			if functionResponse.ID != "" {
				attributes = append(attributes, semconv.GenAIToolCallID(functionResponse.ID))
			}
			if captureContentOnSpans() && functionResponse.Response != nil && params.Error == nil {
				attributes = append(attributes, spanContentAttributes([]contentAttr{{genAIToolCallResult, toolPayload(functionResponse.Response)}})...)
			}
		}
		span.SetAttributes(attributes...)
		return
	}
	toolCallID := "<not specified>"
	toolResponse := "<not specified>"
	if functionResponse != nil {
		if functionResponse.ID != "" {
			toolCallID = functionResponse.ID
		}
		if functionResponse.Response != nil {
			toolResponse = safeSerialize(functionResponse.Response)
		}
	}
	attributes = append(attributes,
		semconv.GenAIToolCallID(toolCallID),
		gcpVertexAgentToolResponseName.String(toolResponse))
	span.SetAttributes(attributes...)
}

func refTraceMergedToolCallsResult(span trace.Span, fnResponseEvent *session.Event, err error) {
	RecordError(span, err)
	attributes := []attribute.KeyValue{
		semconv.GenAIOperationNameKey.String(executeToolName),
		semconv.GenAIToolNameKey.String(mergeToolName),
		semconv.GenAIToolDescriptionKey.String(mergeToolName),
	}
	if useLegacySchema() {
		attributes = append(attributes,
			gcpVertexAgentToolCallArgsName.String("N/A"),
			gcpVertexAgentToolResponseName.String(safeSerialize(fnResponseEvent)))
	}
	if fnResponseEvent != nil {
		attributes = append(attributes, gcpVertexAgentEventID.String(fnResponseEvent.ID))
	}
	span.SetAttributes(attributes...)
}

func refLogRequest(ctx context.Context, req *model.LLMRequest, backend genai.Backend) {
	if !useLegacySchema() {
		return
	}
	genAISystem := variantToGenAISystem(backend)
	record := log.Record{}
	record.SetEventName("gen_ai.system.message")
	record.SetBody(attribute.MapValue(attribute.KeyValue{Key: "content", Value: extractSystemMessage(req)}))
	if genAISystem != nil {
		record.AddAttributes(*genAISystem)
	}
	otelLogger().Emit(ctx, record)
	for _, content := range req.Contents {
		record := log.Record{}
		record.SetEventName("gen_ai.user.message")
		record.SetBody(attribute.MapValue(attribute.KeyValue{Key: "content", Value: toLogValue(contentToJSONLikeValue(content))}))
		if genAISystem != nil {
			record.AddAttributes(*genAISystem)
		}
		otelLogger().Emit(ctx, record)
	}
}

func refLogResponse(ctx context.Context, resp *model.LLMResponse, backend genai.Backend) {
	if !useLegacySchema() {
		return
	}
	record := log.Record{}
	record.SetEventName("gen_ai.choice")
	var finishReason string
	var content *genai.Content
	if resp != nil {
		finishReason = string(resp.FinishReason)
		if resp.Content != nil {
			content = resp.Content
		}
	}
	kvs := []attribute.KeyValue{
		attribute.Int("index", 0),
		{Key: "content", Value: contentToLogValue(content)},
	}
	if finishReason != "" {
		kvs = append(kvs, attribute.String("finish_reason", finishReason))
	}
	record.SetBody(attribute.MapValue(kvs...))
	if genAISystem := variantToGenAISystem(backend); genAISystem != nil {
		record.AddAttributes(*genAISystem)
	}
	otelLogger().Emit(ctx, record)
}

// equivalenceParts covers every part kind semconvPart maps, with
// unicode, thought and empty text, and nested tool payloads.
func equivalenceParts() []*genai.Part {
	return []*genai.Part{
		{Text: "Zażółć gęślą jaźń — 東京 🚀 \u2028 <b>&amp;</b>"},
		{Text: "Let me think.", Thought: true, ThoughtSignature: []byte{0, 1, 2, 255}},
		{Text: ""},
		{},
		nil,
		{FunctionCall: &genai.FunctionCall{ID: "call-1", Name: "lookup", Args: map[string]any{
			"q": "weather", "n": 3, "f": 0.1, "big": int64(1) << 60, "ok": true, "none": nil,
			"nested": map[string]any{"list": []any{1, "two", map[string]any{"三": 3.5}}},
		}}},
		{FunctionResponse: &genai.FunctionResponse{ID: "call-1", Name: "lookup", Response: map[string]any{
			"result": []any{map[string]any{"city": "Kraków", "temp": -2.5}}, "error": nil,
		}}},
		{InlineData: &genai.Blob{MIMEType: "image/png", Data: []byte("\x89PNG\r\n\x1a\n")}},
		{FileData: &genai.FileData{MIMEType: "application/pdf", FileURI: "gs://bucket/doc.pdf"}},
	}
}

func equivalenceRequests() map[string]func() *model.LLMRequest {
	return map[string]func() *model.LLMRequest{
		"empty":          func() *model.LLMRequest { return &model.LLMRequest{} },
		"empty contents": func() *model.LLMRequest { return &model.LLMRequest{Contents: []*genai.Content{}} },
		"system instruction only": func() *model.LLMRequest {
			return &model.LLMRequest{Config: &genai.GenerateContentConfig{SystemInstruction: &genai.Content{Parts: []*genai.Part{{Text: "Be brief."}, {Text: ""}, {Text: "Ünïcode ✓", Thought: true}}}}}
		},
		"nil content": func() *model.LLMRequest {
			return &model.LLMRequest{Contents: []*genai.Content{nil, genai.NewContentFromText("after nil", genai.RoleUser)}}
		},
		"no parts": func() *model.LLMRequest {
			return &model.LLMRequest{Contents: []*genai.Content{{Role: genai.RoleUser}, {Role: genai.RoleModel, Parts: []*genai.Part{}}}}
		},
		"every part kind": func() *model.LLMRequest {
			return &model.LLMRequest{
				Config:   &genai.GenerateContentConfig{SystemInstruction: genai.NewContentFromText("You are terse.", genai.RoleUser)},
				Contents: []*genai.Content{{Role: genai.RoleUser, Parts: equivalenceParts()}, {Role: genai.RoleModel, Parts: equivalenceParts()}},
			}
		},
		"over the attribute limit": func() *model.LLMRequest { return textRequest(strings.Repeat("ö", 40_000), genai.RoleUser) },
		"not serializable": func() *model.LLMRequest {
			return partsRequest(genai.RoleModel, &genai.Part{FunctionCall: &genai.FunctionCall{Name: "f", Args: map[string]any{"x": math.NaN()}}})
		},
		"history": func() *model.LLMRequest { return benchHistory(16) },
	}
}

func equivalenceResponses() map[string][]*model.LLMResponse {
	return map[string][]*model.LLMResponse{
		"text":            {{Content: genai.NewContentFromText("Done — ✓", genai.RoleModel), FinishReason: genai.FinishReasonStop, ModelVersion: "v1"}},
		"every part kind": {{Content: &genai.Content{Role: genai.RoleModel, Parts: equivalenceParts()}, FinishReason: genai.FinishReasonMaxTokens}},
		"no content":      {{FinishReason: genai.FinishReasonSafety, UsageMetadata: &genai.GenerateContentResponseUsageMetadata{PromptTokenCount: 3}}},
		"stream": {
			{Content: genai.NewContentFromText("Do", genai.RoleModel)},
			{Content: genai.NewContentFromText("ne", genai.RoleModel), FinishReason: genai.FinishReasonStop},
		},
		"error": nil,
	}
}

// recordSnapshot is what an exporter receives of a record, less the observed
// timestamp, which the SDK sets to the time of the call. Values keep their
// types and their order, so snapshots compare equal only for identical records.
type recordSnapshot struct {
	EventName         string
	Severity          log.Severity
	SeverityText      string
	Timestamp         time.Time
	Body              any
	Attributes        []any
	DroppedAttributes int
	TraceID           trace.TraceID
	SpanID            trace.SpanID
	TraceFlags        trace.TraceFlags
	Scope             [3]string
}

func snapshotRecords(records []sdklog.Record) []recordSnapshot {
	var out []recordSnapshot
	for _, r := range records {
		var attrs []any
		r.WalkAttributes(func(kv attribute.KeyValue) bool {
			attrs = append(attrs, typedKeyValue(kv))
			return true
		})
		scope := r.InstrumentationScope()
		out = append(out, recordSnapshot{
			EventName:         r.EventName(),
			Severity:          r.Severity(),
			SeverityText:      r.SeverityText(),
			Timestamp:         r.Timestamp(),
			Body:              typedValue(r.Body()),
			Attributes:        attrs,
			DroppedAttributes: r.DroppedAttributes(),
			TraceID:           r.TraceID(),
			SpanID:            r.SpanID(),
			TraceFlags:        r.TraceFlags(),
			Scope:             [3]string{scope.Name, scope.Version, scope.SchemaURL},
		})
	}
	return out
}

// spanSnapshot is what an exporter receives of a span, less its times, which
// differ between any two spans, and its IDs, which are checked apart.
type spanSnapshot struct {
	Name              string
	Kind              trace.SpanKind
	Attributes        []any
	Status            sdktrace.Status
	Events            []any
	DroppedAttributes int
	DroppedEvents     int
	Scope             [3]string
}

func snapshotSpans(spans tracetest.SpanStubs) []spanSnapshot {
	typed := func(kvs []attribute.KeyValue) []any {
		var out []any
		for _, kv := range kvs {
			out = append(out, typedKeyValue(kv))
		}
		return out
	}
	var out []spanSnapshot
	for _, s := range spans {
		var events []any
		for _, e := range s.Events {
			events = append(events, []any{e.Name, typed(e.Attributes), e.DroppedAttributeCount})
		}
		out = append(out, spanSnapshot{
			Name:              s.Name,
			Kind:              s.SpanKind,
			Attributes:        typed(s.Attributes),
			Status:            s.Status,
			Events:            events,
			DroppedAttributes: s.DroppedAttributes,
			DroppedEvents:     s.DroppedEvents,
			Scope:             [3]string{s.InstrumentationScope.Name, s.InstrumentationScope.Version, s.InstrumentationScope.SchemaURL},
		})
	}
	return out
}

// typedValue renders v with its type at every level and map entries in their
// stored order.
func typedValue(v attribute.Value) any {
	switch v.Type() {
	case attribute.SLICE:
		s := []any{}
		for _, e := range v.AsSlice() {
			s = append(s, typedValue(e))
		}
		return s
	case attribute.MAP:
		m := []any{}
		for _, kv := range v.AsMap() {
			m = append(m, typedKeyValue(kv))
		}
		return m
	default:
		return fmt.Sprintf("%s(%s)", v.Type(), v.Emit())
	}
}

func typedKeyValue(kv attribute.KeyValue) any {
	return [2]any{string(kv.Key), typedValue(kv.Value)}
}

// listeners says which signals have an SDK consumer. The other is the OTel
// no-op: a span that does not record, or a logger that takes nothing.
type listeners struct {
	name           string
	tracer, logger bool
}

var listenerCases = []listeners{
	{"tracer and logger", true, true},
	{"tracer only", true, false},
	{"logger only", false, true},
}

// capture records what one run hands to the exporters.
type capture struct {
	spans   *tracetest.InMemoryExporter
	records *inMemoryExporter
}

func (l listeners) install(t *testing.T) capture {
	t.Helper()
	c := capture{spans: tracetest.NewInMemoryExporter(), records: &inMemoryExporter{}}
	if l.tracer {
		OverrideTracerForTesting(t, sdktrace.NewTracerProvider(sdktrace.WithSyncer(c.spans)))
	} else {
		OverrideTracerForTesting(t, tracenoop.NewTracerProvider())
	}
	if l.logger {
		OverrideLoggerForTesting(t, sdklog.NewLoggerProvider(sdklog.WithProcessor(sdklog.NewSimpleProcessor(c.records))))
	} else {
		OverrideLoggerForTesting(t, defaultLoggerProvider)
	}
	return c
}

// run returns what f hands to the exporters, its records linked to its spans
// by position rather than by ID, since IDs differ between two runs.
func (c capture) run(t *testing.T, f func()) ([]spanSnapshot, []recordSnapshot) {
	t.Helper()
	c.spans.Reset()
	c.records.records = nil
	f()
	stubs := c.spans.GetSpans()
	records := snapshotRecords(c.records.records)
	for i := range records {
		spanIndex := slices.IndexFunc(stubs, func(s tracetest.SpanStub) bool { return s.SpanContext.SpanID() == records[i].SpanID })
		records[i].TraceID, records[i].SpanID = trace.TraceID{}, trace.SpanID{byte(spanIndex + 1)}
	}
	return snapshotSpans(stubs), records
}

// checkSame stops the test at the first span or record that differs, and
// diffs only that one: a diff of every snapshot of a run takes minutes.
func checkSame(t *testing.T, what string, wantSpans, gotSpans []spanSnapshot, wantRecords, gotRecords []recordSnapshot) {
	t.Helper()
	if diff := firstDiff(wantSpans, gotSpans); diff != "" {
		t.Fatalf("%s: spans differ (-reference +current):\n%s", what, diff)
	}
	if diff := firstDiff(wantRecords, gotRecords); diff != "" {
		t.Fatalf("%s: records differ (-reference +current):\n%s", what, diff)
	}
}

func firstDiff[T any](want, got []T) string {
	if len(want) != len(got) {
		return fmt.Sprintf("%d of them, want %d", len(got), len(want))
	}
	for i := range want {
		if !reflect.DeepEqual(want[i], got[i]) {
			return fmt.Sprintf("at %d:\n%s", i, cmp.Diff(want[i], got[i]))
		}
	}
	return ""
}

var captureSettings = []string{"", "true", "EVENT_ONLY", "SPAN_ONLY", "SPAN_AND_EVENT"}

// TestInstrumentGenerateContentMatchesReference holds a model call's spans and
// log records to the reference for every listening consumer, schema and
// capture setting, including a model that edits the request it is given.
func TestInstrumentGenerateContentMatchesReference(t *testing.T) {
	// The cases over the attribute limit report to the OpenTelemetry error
	// handler, which logs.
	defer stdlog.SetOutput(stdlog.Writer())
	stdlog.SetOutput(io.Discard)

	for _, schema := range []string{otelSemconv136, "otel_semconv_1_44"} {
		for _, capture := range captureSettings {
			for _, l := range listenerCases {
				t.Run(fmt.Sprintf("%s/capture=%q/%s", schema, capture, l.name), func(t *testing.T) {
					setEnvForTesting(t, map[string]string{adkTelemetrySchemaVersionOptIn: schema, captureMessageContentEnvVar: capture})
					c := l.install(t)
					var sawRecords, sawSpans bool
					for reqName, buildReq := range equivalenceRequests() {
						for respName, responses := range equivalenceResponses() {
							for _, edit := range []bool{false, true} {
								what := fmt.Sprintf("request %s, response %s, edit %t", reqName, respName, edit)
								drive := func(instrument func(GenerateContentParams, func(*model.LLMResponse) (*model.LLMResponse, string), func(context.Context) iter.Seq2[*model.LLMResponse, error]) iter.Seq2[*model.LLMResponse, error]) func() {
									return func() {
										req := buildReq()
										call := func(context.Context) iter.Seq2[*model.LLMResponse, error] {
											return func(yield func(*model.LLMResponse, error) bool) {
												if edit {
													req.Contents = append(req.Contents, genai.NewContentFromText("appended by the model", genai.RoleUser))
												}
												if responses == nil {
													yield(nil, errTest)
													return
												}
												for _, r := range responses {
													if !yield(r, nil) {
														return
													}
												}
											}
										}
										params := GenerateContentParams{ModelName: "gemini-2.5-flash", InvocationID: "inv", Request: req, Backend: genai.BackendVertexAI, Stream: len(responses) > 1}
										for _, err := range instrument(params, func(r *model.LLMResponse) (*model.LLMResponse, string) { return r, "ev" }, call) {
											if err != nil && responses != nil {
												t.Fatal(err)
											}
										}
									}
								}
								wantSpans, wantRecords := c.run(t, drive(func(p GenerateContentParams, r func(*model.LLMResponse) (*model.LLMResponse, string), call func(context.Context) iter.Seq2[*model.LLMResponse, error]) iter.Seq2[*model.LLMResponse, error] {
									return refInstrumentGenerateContent(t.Context(), p, r, call)
								}))
								gotSpans, gotRecords := c.run(t, drive(func(p GenerateContentParams, r func(*model.LLMResponse) (*model.LLMResponse, string), call func(context.Context) iter.Seq2[*model.LLMResponse, error]) iter.Seq2[*model.LLMResponse, error] {
									return InstrumentGenerateContent(t.Context(), p, r, call)
								}))
								checkSame(t, what, wantSpans, gotSpans, wantRecords, gotRecords)
								sawSpans = sawSpans || len(wantSpans) > 0
								sawRecords = sawRecords || len(wantRecords) > 0
							}
						}
					}
					if sawSpans != l.tracer || sawRecords != l.logger {
						t.Errorf("reference handed over spans %t and records %t, want %t and %t", sawSpans, sawRecords, l.tracer, l.logger)
					}
				})
			}
		}
	}
}

func toolResultCases() map[string]TraceToolResultParams {
	return map[string]TraceToolResultParams{
		"no event":           {Description: "d"},
		"no event, error":    {Description: "d", Error: errTest},
		"no content":         {Description: "d", ResponseEvent: &session.Event{ID: "e"}},
		"no parts":           {ResponseEvent: &session.Event{ID: "e", LLMResponse: model.LLMResponse{Content: &genai.Content{Role: genai.RoleUser}}}},
		"text before result": {ResponseEvent: &session.Event{ID: "e", LLMResponse: model.LLMResponse{Content: &genai.Content{Parts: []*genai.Part{{Text: "x"}, {FunctionResponse: &genai.FunctionResponse{ID: "c", Name: "t"}}}}}}},
		"no call ID":         {Description: "d", ResponseEvent: toolResponseEvent("", map[string]any{"ok": true})},
		"unicode":            {Description: "Zażółć ✓", ResponseEvent: toolResponseEvent("c", map[string]any{"result": "Zażółć gęślą jaźń — 東京 🚀 <&>\u2028"})},
		"50 KB":              {Description: "d", ResponseEvent: toolResponseEvent("c", benchToolResponse(128))},
		"over the limit":     {Description: "d", ResponseEvent: toolResponseEvent("c", benchToolResponse(200))},
		"not serializable":   {Description: "d", ResponseEvent: toolResponseEvent("c", map[string]any{"ch": make(chan int)})},
		"error and response": {Description: "d", ResponseEvent: toolResponseEvent("c", map[string]any{"error": "boom"}), Error: genai.APIError{Code: 429, Message: "slow down"}},
	}
}

func mergedToolCallsCases() map[string]struct {
	event *session.Event
	err   error
} {
	merged := &session.Event{ID: "ev-merged", LLMResponse: model.LLMResponse{Content: &genai.Content{Role: genai.RoleUser}}}
	merged.Actions.StateDelta = map[string]any{"last_query": "Kraków ✓"}
	for _, id := range []string{"c0", "c1", "c2"} {
		merged.LLMResponse.Content.Parts = append(merged.LLMResponse.Content.Parts, toolResponseEvent(id, benchToolResponse(26)).LLMResponse.Content.Parts...)
	}
	return map[string]struct {
		event *session.Event
		err   error
	}{
		"no event":        {},
		"no event, error": {err: errTest},
		"three results":   {event: merged},
		"api error":       {event: merged, err: genai.APIError{Code: 503}},
		"no content":      {event: &session.Event{ID: "e"}},
	}
}

// TestToolSpansMatchReference holds the tool result spans to the reference on
// recording spans, for both schemas and every capture setting.
func TestToolSpansMatchReference(t *testing.T) {
	defer stdlog.SetOutput(stdlog.Writer())
	stdlog.SetOutput(io.Discard)

	for _, schema := range []string{otelSemconv136, "otel_semconv_1_44"} {
		for _, capture := range captureSettings {
			t.Run(fmt.Sprintf("%s/capture=%q", schema, capture), func(t *testing.T) {
				setEnvForTesting(t, map[string]string{adkTelemetrySchemaVersionOptIn: schema, captureMessageContentEnvVar: capture})
				c := listeners{tracer: true}.install(t)
				for name, params := range toolResultCases() {
					traced := func(f func(trace.Span, TraceToolResultParams)) func() {
						return func() {
							_, span := StartExecuteToolSpan(t.Context(), StartExecuteToolSpanParams{ToolName: "lookup", Args: map[string]any{"q": "weather"}})
							f(span, params)
							span.End()
						}
					}
					wantSpans, _ := c.run(t, traced(refTraceToolResult))
					gotSpans, _ := c.run(t, traced(TraceToolResult))
					checkSame(t, "tool result "+name, wantSpans, gotSpans, nil, nil)
				}
				for name, tc := range mergedToolCallsCases() {
					traced := func(f func(trace.Span, *session.Event, error)) func() {
						return func() {
							_, span := StartTrace(t.Context(), "execute_tool (merged)")
							f(span, tc.event, tc.err)
							span.End()
						}
					}
					wantSpans, _ := c.run(t, traced(refTraceMergedToolCallsResult))
					gotSpans, _ := c.run(t, traced(TraceMergedToolCallsResult))
					checkSame(t, "merged tool calls "+name, wantSpans, gotSpans, nil, nil)
				}
			})
		}
	}
}

// countingMarshaler counts how often encoding/json encodes it. Converting
// content that holds it as a tool payload encodes it.
type countingMarshaler struct {
	mu *sync.Mutex
	n  *int
}

func newCountingMarshaler() countingMarshaler { return countingMarshaler{&sync.Mutex{}, new(int)} }

func (m countingMarshaler) MarshalJSON() ([]byte, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	*m.n++
	return []byte(`"x"`), nil
}

func (m countingMarshaler) count() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return *m.n
}

// enabledRecorder is a log processor that takes the event names in enable,
// and remembers what it was asked and given.
type enabledRecorder struct {
	enable  map[string]bool
	asked   []sdklog.EnabledParameters
	emitted []string
}

func (p *enabledRecorder) Enabled(_ context.Context, param sdklog.EnabledParameters) bool {
	p.asked = append(p.asked, param)
	return p.enable[param.EventName]
}

func (p *enabledRecorder) OnEmit(_ context.Context, r *sdklog.Record) error {
	p.emitted = append(p.emitted, r.EventName())
	return nil
}

func (p *enabledRecorder) Shutdown(context.Context) error   { return nil }
func (p *enabledRecorder) ForceFlush(context.Context) error { return nil }

// driveTelemetry runs a model call and a tool call, the merged span of a step
// included, whose request, response and tool result each carry m.
func driveTelemetry(t *testing.T, m countingMarshaler) {
	ctx := t.Context()
	payload := map[string]any{"v": m}
	req := benchHistory(3)
	req.Contents = append(req.Contents, &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{Name: "f", Args: payload}}}})
	resp := &model.LLMResponse{Content: &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{Name: "f", Args: payload}}}}, FinishReason: genai.FinishReasonStop}
	call := func(context.Context) iter.Seq2[*model.LLMResponse, error] {
		return func(yield func(*model.LLMResponse, error) bool) { yield(resp, nil) }
	}
	params := GenerateContentParams{ModelName: "gemini-2.5-flash", Request: req, Backend: genai.BackendVertexAI}
	for _, err := range InstrumentGenerateContent(ctx, params, func(r *model.LLMResponse) (*model.LLMResponse, string) { return r, "ev" }, call) {
		if err != nil {
			t.Fatal(err)
		}
	}
	ev := toolResponseEvent("c", payload)
	_, span := StartExecuteToolSpan(ctx, StartExecuteToolSpanParams{ToolName: "f"})
	TraceToolResult(span, TraceToolResultParams{ResponseEvent: ev})
	span.End()
	_, merged := StartTrace(ctx, "execute_tool (merged)")
	TraceMergedToolCallsResult(merged, ev, nil)
	merged.End()
}

// TestContentConvertedOnlyForAListener checks that with content capture on, a
// model call and a tool call convert no content when no logger takes the
// records and no span records: with the OTel no-op providers, with a span the
// sampler dropped, and with a logger whose processor is not enabled.
func TestContentConvertedOnlyForAListener(t *testing.T) {
	notEnabled := func() log.LoggerProvider {
		return sdklog.NewLoggerProvider(sdklog.WithProcessor(&enabledRecorder{}))
	}
	for _, schema := range []string{otelSemconv136, "otel_semconv_1_44"} {
		for _, tc := range []struct {
			name      string
			tracer    trace.TracerProvider
			logger    log.LoggerProvider
			converted bool
		}{
			{"no-op providers", tracenoop.NewTracerProvider(), defaultLoggerProvider, false},
			{"sampled out, logger not enabled", sdktrace.NewTracerProvider(sdktrace.WithSampler(sdktrace.NeverSample())), notEnabled(), false},
			// Controls: each listener alone converts, so the counter can see
			// a conversion.
			{"recording span", sdktrace.NewTracerProvider(), defaultLoggerProvider, true},
			{"enabled logger", tracenoop.NewTracerProvider(), sdklog.NewLoggerProvider(sdklog.WithProcessor(&enabledRecorder{enable: map[string]bool{
				inferenceOperationDetailsEventName: true, systemMessageEventName: true, userMessageEventName: true, choiceEventName: true,
			}})), true},
		} {
			t.Run(schema+"/"+tc.name, func(t *testing.T) {
				setEnvForTesting(t, map[string]string{adkTelemetrySchemaVersionOptIn: schema, captureMessageContentEnvVar: "SPAN_AND_EVENT"})
				OverrideTracerForTesting(t, tc.tracer)
				OverrideLoggerForTesting(t, tc.logger)
				m := newCountingMarshaler()
				driveTelemetry(t, m)
				if got := m.count() > 0; got != tc.converted {
					t.Errorf("content encoded %d times, want converted = %t", m.count(), tc.converted)
				}
			})
		}
	}
}

// question is what Enabled was asked, less the instrumentation scope.
type question struct {
	EventName string
	Severity  log.Severity
}

// TestLoggerAskedForEachEvent checks the question each record is gated on: a
// logger that is installed but not enabled for an event gets no record of it,
// and the inference event is asked about at the severity it is emitted with.
func TestLoggerAskedForEachEvent(t *testing.T) {
	for _, tc := range []struct {
		name   string
		schema string
		enable map[string]bool
		asked  []question
		want   []string
	}{
		{
			name:   "experimental, not enabled",
			schema: "otel_semconv_1_44",
			asked:  []question{{EventName: inferenceOperationDetailsEventName, Severity: log.SeverityDebug}},
		},
		{
			name:   "experimental, enabled",
			schema: "otel_semconv_1_44",
			enable: map[string]bool{inferenceOperationDetailsEventName: true},
			asked:  []question{{EventName: inferenceOperationDetailsEventName, Severity: log.SeverityDebug}},
			want:   []string{inferenceOperationDetailsEventName},
		},
		{
			name:   "legacy, not enabled",
			schema: otelSemconv136,
			asked:  []question{{EventName: systemMessageEventName}, {EventName: userMessageEventName}, {EventName: choiceEventName}},
		},
		{
			name:   "legacy, user messages only",
			schema: otelSemconv136,
			enable: map[string]bool{userMessageEventName: true},
			asked:  []question{{EventName: systemMessageEventName}, {EventName: userMessageEventName}, {EventName: choiceEventName}},
			want:   []string{userMessageEventName, userMessageEventName},
		},
		{
			name:   "legacy, system and choice",
			schema: otelSemconv136,
			enable: map[string]bool{systemMessageEventName: true, choiceEventName: true},
			asked:  []question{{EventName: systemMessageEventName}, {EventName: userMessageEventName}, {EventName: choiceEventName}},
			want:   []string{systemMessageEventName, choiceEventName},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			setEnvForTesting(t, map[string]string{adkTelemetrySchemaVersionOptIn: tc.schema, captureMessageContentEnvVar: "true"})
			OverrideTracerForTesting(t, tracenoop.NewTracerProvider())
			p := &enabledRecorder{enable: tc.enable}
			OverrideLoggerForTesting(t, sdklog.NewLoggerProvider(sdklog.WithProcessor(p)))
			call := func(context.Context) iter.Seq2[*model.LLMResponse, error] {
				return func(yield func(*model.LLMResponse, error) bool) {
					yield(&model.LLMResponse{Content: genai.NewContentFromText("ok", genai.RoleModel), FinishReason: genai.FinishReasonStop}, nil)
				}
			}
			params := GenerateContentParams{ModelName: "m", Request: benchHistory(2)}
			for _, err := range InstrumentGenerateContent(t.Context(), params, func(r *model.LLMResponse) (*model.LLMResponse, string) { return r, "" }, call) {
				if err != nil {
					t.Fatal(err)
				}
			}
			var asked []question
			for _, a := range p.asked {
				asked = append(asked, question{EventName: a.EventName, Severity: a.Severity})
			}
			if diff := cmp.Diff(tc.asked, asked); diff != "" {
				t.Errorf("Enabled questions mismatch (-want +got):\n%s", diff)
			}
			if diff := cmp.Diff(tc.want, p.emitted); diff != "" {
				t.Errorf("emitted records mismatch (-want +got):\n%s", diff)
			}
		})
	}
}
