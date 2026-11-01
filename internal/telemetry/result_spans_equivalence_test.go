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
	"io"
	"log"
	"reflect"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"go.opentelemetry.io/otel/attribute"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	semconv "go.opentelemetry.io/otel/semconv/v1.36.0"
	"go.opentelemetry.io/otel/trace"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
)

// refTraceGenerateContentResult, refTraceToolResult and
// refTraceMergedToolCallsResult are the span result recorders as they were
// before they asked whether the span records: everything is computed and set
// whatever the span does with it. The tests below hold the current functions
// to them.

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

func toolResultCases() map[string]TraceToolResultParams {
	apiErr := genai.APIError{Code: 429, Message: "slow down"}
	return map[string]TraceToolResultParams{
		"no event":           {Description: "d"},
		"no event, error":    {Description: "d", Error: errTest},
		"no content":         {Description: "d", ResponseEvent: &session.Event{ID: "e"}},
		"no parts":           {ResponseEvent: &session.Event{ID: "e", LLMResponse: model.LLMResponse{Content: &genai.Content{Role: genai.RoleUser}}}},
		"text before result": {ResponseEvent: &session.Event{ID: "e", LLMResponse: model.LLMResponse{Content: &genai.Content{Parts: []*genai.Part{{Text: "x"}, {FunctionResponse: &genai.FunctionResponse{ID: "c", Name: "t"}}}}}}},
		"no call ID":         {Description: "d", ResponseEvent: toolResponseEvent("", map[string]any{"ok": true})},
		"no response":        {Description: "d", ResponseEvent: &session.Event{ID: "e", LLMResponse: model.LLMResponse{Content: &genai.Content{Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{ID: "c", Name: "t"}}}}}}},
		"unicode":            {Description: "Zażółć ✓", ResponseEvent: toolResponseEvent("c", map[string]any{"result": "Zażółć gęślą jaźń — 東京 🚀 <&>\u2028"})},
		"50 KB":              {Description: "d", ResponseEvent: toolResponseEvent("c", benchToolResponse(128))},
		"over the limit":     {Description: "d", ResponseEvent: toolResponseEvent("c", benchToolResponse(200))},
		"not serializable":   {Description: "d", ResponseEvent: toolResponseEvent("c", map[string]any{"ch": make(chan int)})},
		"error and response": {Description: "d", ResponseEvent: toolResponseEvent("c", map[string]any{"error": "boom"}), Error: apiErr},
	}
}

func mergedToolCallsCases() map[string]struct {
	event *session.Event
	err   error
} {
	merged := &session.Event{ID: "ev-merged", LLMResponse: model.LLMResponse{Content: &genai.Content{Role: genai.RoleUser}}}
	merged.Actions.StateDelta = map[string]any{"last_query": "Kraków ✓"}
	for _, id := range []string{"c0", "c1", "c2", "c3", "c4"} {
		merged.LLMResponse.Content.Parts = append(merged.LLMResponse.Content.Parts, toolResponseEvent(id, benchToolResponse(26)).LLMResponse.Content.Parts...)
	}
	unserializable := toolResponseEvent("c", map[string]any{"ok": true})
	unserializable.Actions.StateDelta = map[string]any{"f": func() {}}
	return map[string]struct {
		event *session.Event
		err   error
	}{
		"no event":         {},
		"no event, error":  {err: errTest},
		"five results":     {event: merged},
		"five, api error":  {event: merged, err: genai.APIError{Code: 503}},
		"no content":       {event: &session.Event{ID: "e"}},
		"not serializable": {event: unserializable},
	}
}

func generateContentResultCases() map[string]generateContentResult {
	usage := &genai.GenerateContentResponseUsageMetadata{PromptTokenCount: 10, CandidatesTokenCount: 5, ThoughtsTokenCount: 2, CachedContentTokenCount: 1, ToolUsePromptTokenCount: 3}
	final := &model.LLMResponse{Content: genai.NewContentFromText("Done — ✓", genai.RoleModel), FinishReason: genai.FinishReasonStop, UsageMetadata: usage}
	calls := &model.LLMResponse{Content: &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{
		{Text: "thinking", Thought: true},
		{FunctionCall: &genai.FunctionCall{ID: "c", Name: "lookup", Args: map[string]any{"q": "x"}}},
	}}}
	chunk := &model.LLMResponse{Content: genai.NewContentFromText("part", genai.RoleModel), Partial: true}
	return map[string]generateContentResult{
		"nothing":       {},
		"error":         {Error: errTest},
		"api error":     {Error: genai.APIError{Code: 503}},
		"final":         {Response: final, EventID: "ev"},
		"calls":         {Response: calls, EventID: "ev"},
		"partial":       {Response: chunk, EventID: "ev"},
		"stream":        {Response: final, EventID: "ev", Responses: []*model.LLMResponse{chunk, chunk, final}},
		"large":         {Response: &model.LLMResponse{Content: genai.NewContentFromText(strings.Repeat("ö", 20_000), genai.RoleModel)}, EventID: "ev"},
		"error, chunks": {Response: chunk, EventID: "ev", Error: errTest, Responses: []*model.LLMResponse{chunk}},
	}
}

// spanSnapshot is what an exporter receives of a span, less its IDs and
// times, which differ between any two spans. Attribute values keep their types
// and their order.
type spanSnapshot struct {
	Name              string
	Kind              trace.SpanKind
	Attributes        []any
	Status            sdktrace.Status
	Events            []any
	Links             int
	DroppedAttributes int
	DroppedEvents     int
	DroppedLinks      int
	Scope             [3]string
}

func snapshotSpan(s tracetest.SpanStub) spanSnapshot {
	typed := func(kvs []attribute.KeyValue) []any {
		var out []any
		for _, kv := range kvs {
			out = append(out, typedKeyValue(kv))
		}
		return out
	}
	var events []any
	for _, e := range s.Events {
		events = append(events, []any{e.Name, typed(e.Attributes), e.DroppedAttributeCount})
	}
	return spanSnapshot{
		Name:              s.Name,
		Kind:              s.SpanKind,
		Attributes:        typed(s.Attributes),
		Status:            s.Status,
		Events:            events,
		Links:             len(s.Links),
		DroppedAttributes: s.DroppedAttributes,
		DroppedEvents:     s.DroppedEvents,
		DroppedLinks:      s.DroppedLinks,
		Scope:             [3]string{s.InstrumentationScope.Name, s.InstrumentationScope.Version, s.InstrumentationScope.SchemaURL},
	}
}

// TestResultSpansMatchReference holds TraceToolResult,
// TraceMergedToolCallsResult and traceGenerateContentResult to the reference
// above on recording spans, for both schemas and every capture setting.
func TestResultSpansMatchReference(t *testing.T) {
	// The over-the-limit cases report to the OpenTelemetry error handler,
	// which logs.
	defer log.SetOutput(log.Writer())
	log.SetOutput(io.Discard)

	for _, schema := range []string{otelSemconv136, otelSemconv144} {
		for _, capture := range []string{"", "EVENT_ONLY", "SPAN_ONLY", "SPAN_AND_EVENT"} {
			t.Run(schema+"/capture="+capture, func(t *testing.T) {
				setEnvForTesting(t, map[string]string{adkTelemetrySchemaVersionOptIn: schema, captureMessageContentEnvVar: capture})
				exporter := setupTestTracer(t)
				// recorded returns what an exporter gets of a span started by
				// start and finished by f.
				recorded := func(start func() trace.Span, f func(trace.Span)) spanSnapshot {
					t.Helper()
					exporter.Reset()
					span := start()
					if !span.IsRecording() {
						t.Fatal("span is not recording")
					}
					f(span)
					span.End()
					spans := exporter.GetSpans()
					if len(spans) != 1 {
						t.Fatalf("got %d spans, want 1", len(spans))
					}
					return snapshotSpan(spans[0])
				}
				check := func(what string, want, got spanSnapshot) {
					t.Helper()
					if !reflect.DeepEqual(want, got) {
						t.Errorf("%s: span differs (-reference +current):\n%s", what, cmp.Diff(want, got))
					}
				}

				toolSpan := func() trace.Span {
					_, span := StartExecuteToolSpan(t.Context(), StartExecuteToolSpanParams{ToolName: "lookup", Args: map[string]any{"q": "weather"}})
					return span
				}
				for name, params := range toolResultCases() {
					want := recorded(toolSpan, func(s trace.Span) { refTraceToolResult(s, params) })
					got := recorded(toolSpan, func(s trace.Span) { TraceToolResult(s, params) })
					check("tool result "+name, want, got)
				}

				mergedSpan := func() trace.Span {
					_, span := StartTrace(t.Context(), "execute_tool (merged)")
					return span
				}
				for name, c := range mergedToolCallsCases() {
					want := recorded(mergedSpan, func(s trace.Span) { refTraceMergedToolCallsResult(s, c.event, c.err) })
					got := recorded(mergedSpan, func(s trace.Span) { TraceMergedToolCallsResult(s, c.event, c.err) })
					check("merged tool calls "+name, want, got)
				}

				generateSpan := func() trace.Span {
					_, span := startGenerateContentSpan(t.Context(), GenerateContentParams{ModelName: "gemini", InvocationID: "inv", Request: benchHistory(3)})
					return span
				}
				for name, result := range generateContentResultCases() {
					want := recorded(generateSpan, func(s trace.Span) { refTraceGenerateContentResult(s, result) })
					got := recorded(generateSpan, func(s trace.Span) { traceGenerateContentResult(s, result) })
					check("generate content "+name, want, got)
				}
			})
		}
	}
}

// countingMarshaler counts how often encoding/json encodes it.
type countingMarshaler struct{ n *int }

func (m countingMarshaler) MarshalJSON() ([]byte, error) {
	*m.n++
	return []byte(`"x"`), nil
}

// TestToolSpansSerializeOnlyWhenRecording checks that the legacy schema
// serializes the tool response and the merged event once each for a span that
// records, and not at all for one that does not.
func TestToolSpansSerializeOnlyWhenRecording(t *testing.T) {
	for _, tc := range []struct {
		name     string
		provider trace.TracerProvider
		want     int
	}{
		{"recording", sdktrace.NewTracerProvider(), 2},
		{"sampled out", sdktrace.NewTracerProvider(sdktrace.WithSampler(sdktrace.NeverSample())), 0},
		{"no provider", nil, 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			setEnvForTesting(t, map[string]string{adkTelemetrySchemaVersionOptIn: otelSemconv136})
			if tc.provider != nil {
				OverrideTracerForTesting(t, tc.provider)
			}
			var n int
			ev := toolResponseEvent("c", map[string]any{"v": countingMarshaler{&n}})

			_, span := StartExecuteToolSpan(t.Context(), StartExecuteToolSpanParams{ToolName: "t"})
			TraceToolResult(span, TraceToolResultParams{ResponseEvent: ev})
			span.End()
			_, merged := StartTrace(t.Context(), "execute_tool (merged)")
			TraceMergedToolCallsResult(merged, ev, nil)
			merged.End()

			if n != tc.want {
				t.Errorf("tool payload encoded %d times, want %d", n, tc.want)
			}
		})
	}
}
