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
	"iter"
	"testing"
	"time"

	sdklog "go.opentelemetry.io/otel/sdk/log"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/internal/benchutil"
	"google.golang.org/adk/v2/model"
)

// benchRequest is the prompt of a turn with n prior events, as the runner
// benchmark assembles it: text turns with a tool call and result every eighth.
func benchRequest(n int) *model.LLMRequest {
	req := &model.LLMRequest{
		Config: &genai.GenerateContentConfig{SystemInstruction: genai.NewContentFromText("You are terse.", genai.RoleUser)},
	}
	for _, ev := range benchutil.BenchEvents(n, "agent", time.Unix(1_700_000_000, 0)) {
		req.Contents = append(req.Contents, ev.Content)
	}
	return req
}

// captureCase is one content-capture setting and the providers consuming it.
type captureCase struct {
	name string
	// capture is the OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT value.
	capture string
	// tracer and logger install SDK providers over counting exporters. Without
	// them the span is not recording and the logger is the global no-op one,
	// which is what a process without an exporter runs with.
	tracer, logger bool
}

var captureCases = []captureCase{
	{name: "off"},
	{name: "event/no-provider", capture: "EVENT_ONLY"},
	{name: "event/sdk-logger", capture: "EVENT_ONLY", logger: true},
	{name: "span/no-provider", capture: "SPAN_ONLY"},
	{name: "span/sdk-tracer", capture: "SPAN_ONLY", tracer: true},
	{name: "span_and_event/sdk-both", capture: "SPAN_AND_EVENT", tracer: true, logger: true},
}

// install applies the case to the package for the rest of the benchmark and
// returns the exporters its providers feed, which count what was consumed.
func (c captureCase) install(b *testing.B) (*benchutil.CountingSpanExporter, *benchutil.CountingLogExporter) {
	b.Helper()
	// Runs after b.Setenv restores the environment.
	b.Cleanup(ApplyEnv)
	b.Setenv(captureMessageContentEnvVar, c.capture)
	ApplyEnv()
	spans := &benchutil.CountingSpanExporter{}
	records := &benchutil.CountingLogExporter{}
	if c.tracer {
		OverrideTracerForTesting(b, sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(sdktrace.NewSimpleSpanProcessor(spans))))
	}
	if c.logger {
		OverrideLoggerForTesting(b, sdklog.NewLoggerProvider(sdklog.WithProcessor(sdklog.NewSimpleProcessor(records))))
	}
	return spans, records
}

// BenchmarkInstrumentGenerateContent measures the telemetry of one unstreamed
// model call over a 1000-event prompt, on its own, under each content-capture
// setting. It reports the spans and log records the providers consumed.
func BenchmarkInstrumentGenerateContent(b *testing.B) {
	for _, tc := range captureCases {
		b.Run(tc.name, func(b *testing.B) {
			spans, records := tc.install(b)
			params := GenerateContentParams{ModelName: "gemini-2.5-flash", InvocationID: "inv", Request: benchRequest(1000), Backend: genai.BackendGeminiAPI}
			resp := &model.LLMResponse{Content: genai.NewContentFromText(benchutil.BenchText, genai.RoleModel), FinishReason: genai.FinishReasonStop, TurnComplete: true}
			call := func(context.Context) iter.Seq2[*model.LLMResponse, error] {
				return func(yield func(*model.LLMResponse, error) bool) { yield(resp, nil) }
			}
			response := func(r *model.LLMResponse) (*model.LLMResponse, string) { return r, "ev" }
			b.ReportAllocs()
			for b.Loop() {
				for range InstrumentGenerateContent(b.Context(), params, response, call) {
				}
			}
			b.ReportMetric(float64(spans.Spans.Load())/float64(b.N), "spans/call")
			b.ReportMetric(float64(records.Records.Load())/float64(b.N), "records/call")
		})
	}
}
