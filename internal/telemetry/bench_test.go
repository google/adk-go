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

	"go.opentelemetry.io/otel"
	sdklog "go.opentelemetry.io/otel/sdk/log"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/trace"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"
)

// benchText is the body of every text turn: ~300 bytes, so 1000 turns are a
// long chat of ~300 KiB.
var benchText = strings.Repeat("The quick brown fox jumps over the lazy dog. ", 7)

// benchHistories are the conversation lengths, in turns, of the prompts.
var benchHistories = []int{100, 1000}

// benchRequest is the prompt of a turn after n earlier turns: text turns
// alternating between the user and the model, with a tool call and its result
// every eighth turn.
func benchRequest(n int) *model.LLMRequest {
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

// countingSpanExporter counts the spans it is given and keeps none.
type countingSpanExporter struct {
	spans atomic.Int64
}

func (e *countingSpanExporter) ExportSpans(_ context.Context, spans []sdktrace.ReadOnlySpan) error {
	e.spans.Add(int64(len(spans)))
	return nil
}

func (e *countingSpanExporter) Shutdown(context.Context) error { return nil }

// countingLogExporter counts the log records it is given and keeps none.
type countingLogExporter struct {
	records atomic.Int64
}

func (e *countingLogExporter) Export(_ context.Context, records []sdklog.Record) error {
	e.records.Add(int64(len(records)))
	return nil
}

func (e *countingLogExporter) Shutdown(context.Context) error   { return nil }
func (e *countingLogExporter) ForceFlush(context.Context) error { return nil }

// benchCapture is a content-capture setting and the providers consuming it.
type benchCapture struct {
	name string
	// capture is the OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT value.
	capture string
	// tracer and logger install SDK providers over counting exporters. Without
	// them the span does not record and the logger is the global no-op one,
	// as in a process with no exporter configured.
	tracer, logger bool
}

var benchCaptures = []benchCapture{
	{name: "off"},
	{name: "event/no-provider", capture: "EVENT_ONLY"},
	{name: "event/sdk-logger", capture: "EVENT_ONLY", logger: true},
	{name: "span/sdk-tracer", capture: "SPAN_ONLY", tracer: true},
	{name: "span_and_event/sdk-both", capture: "SPAN_AND_EVENT", tracer: true, logger: true},
}

// install applies c to the package for the rest of the benchmark, under the
// default schema, and returns the exporters its providers feed.
func (c benchCapture) install(b *testing.B) (*countingSpanExporter, *countingLogExporter) {
	b.Helper()
	// Runs after b.Setenv restores the environment.
	b.Cleanup(ApplyEnv)
	b.Setenv(adkTelemetrySchemaVersionOptIn, "")
	b.Setenv(captureMessageContentEnvVar, c.capture)
	ApplyEnv()
	// A conversation over the span attribute limit is reported to the
	// OpenTelemetry error handler on every call, which logs to stderr.
	previous := otel.GetErrorHandler()
	otel.SetErrorHandler(otel.ErrorHandlerFunc(func(error) {}))
	b.Cleanup(func() { otel.SetErrorHandler(previous) })
	spans := &countingSpanExporter{}
	records := &countingLogExporter{}
	if c.tracer {
		OverrideTracerForTesting(b, sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(sdktrace.NewSimpleSpanProcessor(spans))))
	}
	if c.logger {
		OverrideLoggerForTesting(b, sdklog.NewLoggerProvider(sdklog.WithProcessor(sdklog.NewSimpleProcessor(records))))
	}
	return spans, records
}

// BenchmarkInstrumentGenerateContent measures the telemetry of one unstreamed
// model call that does no work, under each content-capture setting. It
// reports the spans and log records the providers consumed.
func BenchmarkInstrumentGenerateContent(b *testing.B) {
	for _, n := range benchHistories {
		for _, tc := range benchCaptures {
			b.Run(fmt.Sprintf("history=%d/%s", n, tc.name), func(b *testing.B) {
				spans, records := tc.install(b)
				params := GenerateContentParams{ModelName: "gemini-2.5-flash", InvocationID: "inv", Request: benchRequest(n), Backend: genai.BackendGeminiAPI}
				resp := &model.LLMResponse{Content: genai.NewContentFromText(benchText, genai.RoleModel), FinishReason: genai.FinishReasonStop, TurnComplete: true}
				call := func(context.Context) iter.Seq2[*model.LLMResponse, error] {
					return func(yield func(*model.LLMResponse, error) bool) { yield(resp, nil) }
				}
				response := func(r *model.LLMResponse) (*model.LLMResponse, string) { return r, "ev" }
				b.ReportAllocs()
				for b.Loop() {
					for range InstrumentGenerateContent(b.Context(), params, response, call) {
					}
				}
				b.ReportMetric(float64(spans.spans.Load())/float64(b.N), "spans/call")
				b.ReportMetric(float64(records.records.Load())/float64(b.N), "records/call")
			})
		}
	}
}

// BenchmarkNoteLLMRequestUnchanged measures noting the request once more after
// a model call that left it unchanged, which every model call does when
// content capture is on.
func BenchmarkNoteLLMRequestUnchanged(b *testing.B) {
	for _, n := range benchHistories {
		b.Run(fmt.Sprintf("history=%d", n), func(b *testing.B) {
			benchCapture{capture: "EVENT_ONLY"}.install(b)
			req := benchRequest(n)
			d := operationDetails{span: trace.SpanFromContext(b.Context())}
			d.noteLLMRequest(req)
			encoded := d.eventRequest
			b.ReportAllocs()
			for b.Loop() {
				d.noteLLMRequest(req)
			}
			if &d.eventRequest[0] != &encoded[0] {
				b.Fatal("noting an unchanged request encoded it again")
			}
		})
	}
}
