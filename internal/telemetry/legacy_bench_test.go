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
	"iter"
	"strings"
	"sync/atomic"
	"testing"

	sdklog "go.opentelemetry.io/otel/sdk/log"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
)

// benchText is the body of every synthetic text turn, ~300 bytes.
var benchText = strings.Repeat("The quick brown fox jumps over the lazy dog. ", 7)

// benchHistory is the prompt of a turn with n history contents, shaped like a
// long chat: text turns, with a tool call and its result every eighth.
func benchHistory(n int) *model.LLMRequest {
	req := &model.LLMRequest{
		Config: &genai.GenerateContentConfig{SystemInstruction: genai.NewContentFromText("You are terse.", genai.RoleUser)},
	}
	for i := range n {
		var c *genai.Content
		switch {
		case i%2 == 0:
			c = genai.NewContentFromText(benchText, genai.RoleUser)
		case i%8 == 1:
			c = &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{{
				FunctionCall: &genai.FunctionCall{ID: fmt.Sprintf("call-%06d", i), Name: "lookup", Args: map[string]any{"q": "weather"}},
			}}}
		case i%8 == 3:
			c = &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{{
				FunctionResponse: &genai.FunctionResponse{ID: fmt.Sprintf("call-%06d", i-2), Name: "lookup", Response: map[string]any{"result": benchText}},
			}}}
		default:
			c = genai.NewContentFromText(benchText, genai.RoleModel)
		}
		req.Contents = append(req.Contents, c)
	}
	return req
}

// benchToolResponse is a search-style tool result of n records, ~400 bytes of
// JSON each.
func benchToolResponse(n int) map[string]any {
	items := make([]any, n)
	for i := range items {
		items[i] = map[string]any{
			"id":      i,
			"title":   fmt.Sprintf("Result %d", i),
			"url":     fmt.Sprintf("https://example.com/items/%d", i),
			"snippet": benchText,
			"score":   0.5 + float64(i)/1000,
		}
	}
	return map[string]any{"results": items, "total": n}
}

// toolResponseEvent is the event a tool call with response resp produces.
func toolResponseEvent(id string, resp map[string]any) *session.Event {
	return &session.Event{ID: "ev-" + id, LLMResponse: model.LLMResponse{Content: &genai.Content{
		Role:  genai.RoleUser,
		Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{ID: id, Name: "lookup", Response: resp}}},
	}}}
}

// countingSpanExporter counts the spans it is given and keeps none.
type countingSpanExporter struct{ spans atomic.Int64 }

func (e *countingSpanExporter) ExportSpans(_ context.Context, spans []sdktrace.ReadOnlySpan) error {
	e.spans.Add(int64(len(spans)))
	return nil
}

func (e *countingSpanExporter) Shutdown(context.Context) error { return nil }

// countingLogExporter counts the log records it is given and keeps none.
type countingLogExporter struct{ records atomic.Int64 }

func (e *countingLogExporter) Export(_ context.Context, records []sdklog.Record) error {
	e.records.Add(int64(len(records)))
	return nil
}

func (e *countingLogExporter) Shutdown(context.Context) error   { return nil }
func (e *countingLogExporter) ForceFlush(context.Context) error { return nil }

// benchTelemetry is a telemetry setting: the schema and content-capture
// environment, and the SDK providers consuming the signals. Without providers
// spans do not record and the logger is the global no-op one, which is what a
// process without an exporter runs with.
type benchTelemetry struct {
	schema, capture string
	tracer, logger  bool
}

func (c benchTelemetry) providers() string {
	switch {
	case c.tracer && c.logger:
		return "providers=tracer+logger"
	case c.tracer:
		return "providers=tracer"
	case c.logger:
		return "providers=logger"
	}
	return "providers=none"
}

// install applies the setting for the rest of the benchmark and returns the
// exporters its providers feed.
func (c benchTelemetry) install(b *testing.B) (*countingSpanExporter, *countingLogExporter) {
	b.Helper()
	// Runs after b.Setenv restores the environment.
	b.Cleanup(ApplyEnv)
	b.Setenv(adkTelemetrySchemaVersionOptIn, c.schema)
	b.Setenv(captureMessageContentEnvVar, c.capture)
	ApplyEnv()
	spans, records := &countingSpanExporter{}, &countingLogExporter{}
	if c.tracer {
		OverrideTracerForTesting(b, sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(sdktrace.NewSimpleSpanProcessor(spans))))
	}
	if c.logger {
		OverrideLoggerForTesting(b, sdklog.NewLoggerProvider(sdklog.WithProcessor(sdklog.NewSimpleProcessor(records))))
	}
	return spans, records
}

const otelSemconv144 = "otel_semconv_1_44"

// BenchmarkGenerateContentTelemetry measures the telemetry of one unstreamed
// model call over a prompt of n history contents. The legacy schema logs one
// record per content; capture=event puts the content into those records.
func BenchmarkGenerateContentTelemetry(b *testing.B) {
	resp := &model.LLMResponse{Content: genai.NewContentFromText(benchText, genai.RoleModel), FinishReason: genai.FinishReasonStop, TurnComplete: true}
	call := func(context.Context) iter.Seq2[*model.LLMResponse, error] {
		return func(yield func(*model.LLMResponse, error) bool) { yield(resp, nil) }
	}
	response := func(r *model.LLMResponse) (*model.LLMResponse, string) { return r, "ev" }

	type benchCase struct {
		name string
		n    int
		benchTelemetry
	}
	var cases []benchCase
	for _, capture := range []struct{ name, value string }{{"off", ""}, {"event", "EVENT_ONLY"}} {
		for _, n := range []int{100, 1000} {
			for _, p := range []benchTelemetry{{}, {tracer: true}, {tracer: true, logger: true}} {
				p.schema, p.capture = otelSemconv136, capture.value
				cases = append(cases, benchCase{fmt.Sprintf("legacy/capture=%s/n=%d/%s", capture.name, n, p.providers()), n, p})
			}
		}
	}
	for _, p := range []benchTelemetry{{}, {tracer: true}} {
		p.schema = otelSemconv144
		cases = append(cases, benchCase{fmt.Sprintf("experimental/capture=off/n=100/%s", p.providers()), 100, p})
	}

	for _, tc := range cases {
		b.Run(tc.name, func(b *testing.B) {
			spans, records := tc.install(b)
			params := GenerateContentParams{ModelName: "gemini-2.5-flash", InvocationID: "inv", Request: benchHistory(tc.n), Backend: genai.BackendGeminiAPI}
			b.ReportAllocs()
			for b.Loop() {
				for range InstrumentGenerateContent(b.Context(), params, response, call) {
				}
			}
			b.ReportMetric(float64(spans.spans.Load())/float64(b.N), "spans/op")
			b.ReportMetric(float64(records.records.Load())/float64(b.N), "records/op")
		})
	}
}

// toolBenchSettings are the settings the tool benchmarks run under. Capture is
// off: the legacy schema records tool payloads on spans regardless.
var toolBenchSettings = []struct {
	name string
	benchTelemetry
}{
	{"legacy/providers=none", benchTelemetry{schema: otelSemconv136}},
	{"legacy/providers=tracer", benchTelemetry{schema: otelSemconv136, tracer: true}},
	{"experimental/providers=none", benchTelemetry{schema: otelSemconv144}},
	{"experimental/providers=tracer", benchTelemetry{schema: otelSemconv144, tracer: true}},
}

// BenchmarkToolResultTelemetry measures the execute_tool span of one tool call
// whose result is ~50 KB of JSON, as the flow drives it.
func BenchmarkToolResultTelemetry(b *testing.B) {
	args := map[string]any{"q": "weather"}
	ev := toolResponseEvent("call-1", benchToolResponse(128))
	for _, tc := range toolBenchSettings {
		b.Run(tc.name, func(b *testing.B) {
			spans, _ := tc.install(b)
			b.ReportAllocs()
			for b.Loop() {
				_, span := StartExecuteToolSpan(b.Context(), StartExecuteToolSpanParams{ToolName: "lookup", Args: args})
				TraceToolResult(span, TraceToolResultParams{Description: "Looks things up.", ResponseEvent: ev})
				span.End()
			}
			b.ReportMetric(float64(spans.spans.Load())/float64(b.N), "spans/op")
		})
	}
}

// BenchmarkMergedToolCallsTelemetry measures the merged span of a step that
// ran five tool calls with ~10 KB results each, as the flow drives it.
func BenchmarkMergedToolCallsTelemetry(b *testing.B) {
	merged := &session.Event{ID: "ev-merged", LLMResponse: model.LLMResponse{Content: &genai.Content{Role: genai.RoleUser}}}
	merged.Actions.StateDelta = map[string]any{"last_query": "weather"}
	for i := range 5 {
		ev := toolResponseEvent(fmt.Sprintf("call-%d", i), benchToolResponse(26))
		merged.LLMResponse.Content.Parts = append(merged.LLMResponse.Content.Parts, ev.LLMResponse.Content.Parts...)
	}
	for _, tc := range toolBenchSettings {
		b.Run(tc.name, func(b *testing.B) {
			spans, _ := tc.install(b)
			b.ReportAllocs()
			for b.Loop() {
				_, span := StartTrace(b.Context(), "execute_tool (merged)")
				TraceMergedToolCallsResult(span, merged, nil)
				span.End()
			}
			b.ReportMetric(float64(spans.spans.Load())/float64(b.N), "spans/op")
		})
	}
}
