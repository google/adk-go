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

type countingSpanExporter struct{ spans atomic.Int64 }

func (e *countingSpanExporter) ExportSpans(_ context.Context, spans []sdktrace.ReadOnlySpan) error {
	e.spans.Add(int64(len(spans)))
	return nil
}

func (e *countingSpanExporter) Shutdown(context.Context) error { return nil }

type countingLogExporter struct{ records atomic.Int64 }

func (e *countingLogExporter) Export(_ context.Context, records []sdklog.Record) error {
	e.records.Add(int64(len(records)))
	return nil
}

func (e *countingLogExporter) Shutdown(context.Context) error   { return nil }
func (e *countingLogExporter) ForceFlush(context.Context) error { return nil }

// benchTelemetry is the schema and content-capture environment, and whether
// SDK providers consume the signals. Without them spans do not record and the
// logger is the global no-op one, which is what a process without an exporter
// runs with.
type benchTelemetry struct {
	name            string
	schema, capture string
	providers       bool
}

// install applies c for the rest of the benchmark and returns the exporters
// its providers feed.
func (c benchTelemetry) install(b *testing.B) (*countingSpanExporter, *countingLogExporter) {
	b.Helper()
	// Runs after b.Setenv restores the environment.
	b.Cleanup(ApplyEnv)
	b.Setenv(adkTelemetrySchemaVersionOptIn, c.schema)
	b.Setenv(captureMessageContentEnvVar, c.capture)
	ApplyEnv()
	spans, records := &countingSpanExporter{}, &countingLogExporter{}
	if c.providers {
		OverrideTracerForTesting(b, sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(sdktrace.NewSimpleSpanProcessor(spans))))
		OverrideLoggerForTesting(b, sdklog.NewLoggerProvider(sdklog.WithProcessor(sdklog.NewSimpleProcessor(records))))
	}
	return spans, records
}

// BenchmarkInstrumentGenerateContent measures the telemetry of one unstreamed
// model call over a 1000-content prompt, and reports the spans and log records
// the providers consumed.
func BenchmarkInstrumentGenerateContent(b *testing.B) {
	resp := &model.LLMResponse{Content: genai.NewContentFromText(benchText, genai.RoleModel), FinishReason: genai.FinishReasonStop, TurnComplete: true}
	call := func(context.Context) iter.Seq2[*model.LLMResponse, error] {
		return func(yield func(*model.LLMResponse, error) bool) { yield(resp, nil) }
	}
	response := func(r *model.LLMResponse) (*model.LLMResponse, string) { return r, "ev" }
	for _, tc := range []benchTelemetry{
		{name: "capture=span_and_event/providers=none", capture: "SPAN_AND_EVENT"},
		{name: "capture=off/providers=sdk", providers: true},
		{name: "capture=span_and_event/providers=sdk", capture: "SPAN_AND_EVENT", providers: true},
		{name: "legacy/capture=event/providers=none", schema: otelSemconv136, capture: "EVENT_ONLY"},
		{name: "legacy/capture=event/providers=sdk", schema: otelSemconv136, capture: "EVENT_ONLY", providers: true},
	} {
		b.Run(tc.name, func(b *testing.B) {
			spans, records := tc.install(b)
			params := GenerateContentParams{ModelName: "gemini-2.5-flash", InvocationID: "inv", Request: benchHistory(1000), Backend: genai.BackendGeminiAPI}
			b.ReportAllocs()
			for b.Loop() {
				for _, err := range InstrumentGenerateContent(b.Context(), params, response, call) {
					if err != nil {
						b.Fatal(err)
					}
				}
			}
			b.ReportMetric(float64(spans.spans.Load())/float64(b.N), "spans/op")
			b.ReportMetric(float64(records.records.Load())/float64(b.N), "records/op")
		})
	}
}

// BenchmarkNoteLLMRequest measures noting a 1000-content request a second
// time, as a model call does once it ends, when the model left it unchanged.
func BenchmarkNoteLLMRequest(b *testing.B) {
	benchTelemetry{capture: "SPAN_AND_EVENT", providers: true}.install(b)
	req := benchHistory(1000)
	_, span := tracer().Start(b.Context(), "generate_content")
	defer span.End()
	d := newOperationDetails(b.Context(), span)
	d.noteLLMRequest(req)
	b.ReportAllocs()
	for b.Loop() {
		d.noteLLMRequest(req)
	}
}

// BenchmarkToolResultTelemetry measures the execute_tool span of one tool call
// whose result is ~50 KB of JSON.
func BenchmarkToolResultTelemetry(b *testing.B) {
	args := map[string]any{"q": "weather"}
	ev := toolResponseEvent("call-1", benchToolResponse(128))
	for _, tc := range []benchTelemetry{
		{name: "legacy/providers=none", schema: otelSemconv136},
		{name: "legacy/providers=sdk", schema: otelSemconv136, providers: true},
		{name: "capture=span/providers=none", capture: "SPAN_ONLY"},
		{name: "capture=span/providers=sdk", capture: "SPAN_ONLY", providers: true},
	} {
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
