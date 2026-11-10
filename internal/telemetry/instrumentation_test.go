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
	"errors"
	"fmt"
	"iter"
	"slices"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/log"
	sdklog "go.opentelemetry.io/otel/sdk/log"
	semconv "go.opentelemetry.io/otel/semconv/v1.36.0"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/model/anthropicmodel"
	"google.golang.org/adk/v2/session"
)

type testAgent struct{}

func (testAgent) Name() string        { return "root" }
func (testAgent) Description() string { return "" }

func TestInstrumentInvocation_FirstErrorAndEarlyStop(t *testing.T) {
	t.Setenv(adkTelemetrySchemaVersionOptIn, "otel_semconv_1_44")
	exporter := setupTestTracer(t)
	first, second := errors.New("first"), errors.New("second")
	run := func(context.Context) iter.Seq2[*session.Event, error] {
		return func(yield func(*session.Event, error) bool) {
			_ = yield(nil, first) && yield(nil, second) && yield(&session.Event{}, nil)
		}
	}

	seen := 0
	for range InstrumentInvocation(t.Context(), testAgent{}, "s", run) { //nolint:errcheck // the run yields errors by design; the span status below carries the first one
		if seen++; seen == 2 {
			break
		}
	}

	spans := exporter.GetSpans()
	if len(spans) != 1 {
		t.Fatalf("got %d ended spans, want 1", len(spans))
	}
	if got := spans[0].Status.Description; got != "first" {
		t.Errorf("status description = %q, want %q", got, "first")
	}
}

func TestInstrumentInvocation_NestedOpensNoSpan(t *testing.T) {
	t.Setenv(adkTelemetrySchemaVersionOptIn, "otel_semconv_1_44")
	exporter := setupTestTracer(t)
	run := func(context.Context) iter.Seq2[*session.Event, error] {
		return func(yield func(*session.Event, error) bool) { yield(&session.Event{}, nil) }
	}

	ctx, outer := startInvokeWorkflowSpan(t.Context(), "outer", "s")
	for _, err := range InstrumentInvocation(ctx, testAgent{}, "s", run) {
		if err != nil {
			t.Fatalf("InstrumentInvocation() error = %v", err)
		}
	}
	outer.End()

	if got := len(exporter.GetSpans()); got != 1 {
		t.Errorf("got %d spans, want only the outer one", got)
	}
}

func TestInstrumentGenerateContent_EarlyStop(t *testing.T) {
	t.Setenv(adkTelemetrySchemaVersionOptIn, "otel_semconv_1_44")
	spans := setupTestTracer(t)
	logs := &inMemoryExporter{}
	OverrideLoggerForTesting(t, sdklog.NewLoggerProvider(sdklog.WithProcessor(sdklog.NewSimpleProcessor(logs))))
	partial := &model.LLMResponse{Content: genai.NewContentFromText("par", genai.RoleModel), Partial: true}
	final := &model.LLMResponse{Content: genai.NewContentFromText("partial", genai.RoleModel)}
	call := func(context.Context) iter.Seq2[*model.LLMResponse, error] {
		return func(yield func(*model.LLMResponse, error) bool) {
			_ = yield(partial, nil) && yield(final, nil)
		}
	}
	params := GenerateContentParams{ModelName: "m", Request: &model.LLMRequest{}, Stream: true}

	for _, err := range InstrumentGenerateContent(t.Context(), params, func(r *model.LLMResponse) (*model.LLMResponse, string) { return r, "" }, call) {
		if err != nil {
			t.Fatalf("InstrumentGenerateContent() error = %v", err)
		}
		break
	}

	if got := len(spans.GetSpans()); got != 1 {
		t.Fatalf("got %d ended spans, want 1", got)
	}
	if got := len(logs.records); got != 1 {
		t.Fatalf("got %d log records, want 1", got)
	}
	if got := logs.records[0].Severity(); got != log.SeverityDebug {
		t.Errorf("severity = %v, want %v", got, log.SeverityDebug)
	}
	// An abandoned stream reported no finish reason, so none is published.
	for _, kv := range spans.GetSpans()[0].Attributes {
		if kv.Key == semconv.GenAIResponseFinishReasonsKey {
			t.Errorf("span finish reasons = %v, want none", kv.Value.Emit())
		}
	}
	logs.records[0].WalkAttributes(func(kv attribute.KeyValue) bool {
		if kv.Key == semconv.GenAIResponseFinishReasonsKey {
			t.Errorf("event finish reasons = %v, want none", kv.Value.Emit())
		}
		return true
	})
}

// A stream may report usage on one chunk and the finish reason on another, and
// an aggregate arriving after the end is not part of the record.
func TestInstrumentGenerateContent_UsageFromLatestReport(t *testing.T) {
	t.Setenv(adkTelemetrySchemaVersionOptIn, "otel_semconv_1_44")
	spans := setupTestTracer(t)
	usage := func(n int32) *genai.GenerateContentResponseUsageMetadata {
		return &genai.GenerateContentResponseUsageMetadata{PromptTokenCount: n}
	}
	call := func(context.Context) iter.Seq2[*model.LLMResponse, error] {
		return func(yield func(*model.LLMResponse, error) bool) {
			_ = yield(&model.LLMResponse{UsageMetadata: usage(10), Partial: true}, nil) &&
				yield(&model.LLMResponse{FinishReason: genai.FinishReasonStop, Partial: true}, nil) &&
				yield(&model.LLMResponse{UsageMetadata: usage(99), FinishReason: genai.FinishReasonStop}, nil)
		}
	}
	params := GenerateContentParams{ModelName: "m", Request: &model.LLMRequest{}, Stream: true}
	for _, err := range InstrumentGenerateContent(t.Context(), params, func(r *model.LLMResponse) (*model.LLMResponse, string) { return r, "" }, call) {
		if err != nil {
			t.Fatalf("InstrumentGenerateContent() error = %v", err)
		}
	}

	var got int64
	for _, kv := range spans.GetSpans()[0].Attributes {
		if kv.Key == semconv.GenAIUsageInputTokensKey {
			got = kv.Value.AsInt64()
		}
	}
	if got != 10 {
		t.Errorf("input tokens = %d, want 10", got)
	}
}

// modelStep is one value a fake model yields, after a delay.
type modelStep struct {
	resp  *model.LLMResponse
	err   error
	delay time.Duration
}

// generate drives one instrumented model call that yields the steps in order.
// It stops consuming after the given number of values, or drains the call when
// that number is zero.
func generate(t *testing.T, params GenerateContentParams, stopAfter int, steps ...modelStep) {
	t.Helper()
	call := func(context.Context) iter.Seq2[*model.LLMResponse, error] {
		return func(yield func(*model.LLMResponse, error) bool) {
			for _, s := range steps {
				time.Sleep(s.delay)
				if !yield(s.resp, s.err) {
					return
				}
			}
		}
	}
	n := 0
	for range InstrumentGenerateContent(t.Context(), params, func(r *model.LLMResponse) (*model.LLMResponse, string) { return r, "" }, call) { //nolint:errcheck // tests check the telemetry, not the error
		if n++; n == stopAfter {
			break
		}
	}
}

// sortPoints orders metric points by their attributes, which fmt prints with
// sorted keys.
var sortPoints = cmpopts.SortSlices(func(a, b metricPoint) bool { return fmt.Sprint(a.Attrs) < fmt.Sprint(b.Attrs) })

func TestInstrumentGenerateContent_Metrics(t *testing.T) {
	t.Setenv(adkTelemetrySchemaVersionOptIn, "otel_semconv_1_44")
	usage := func(prompt, candidates int32) *genai.GenerateContentResponseUsageMetadata {
		return &genai.GenerateContentResponseUsageMetadata{PromptTokenCount: prompt, CandidatesTokenCount: candidates}
	}
	text := func(n int32) *genai.ModalityTokenCount {
		return &genai.ModalityTokenCount{Modality: genai.MediaModalityText, TokenCount: n}
	}
	streamed := GenerateContentParams{ModelName: "gemini-x", Backend: genai.BackendGeminiAPI, Request: &model.LLMRequest{}, Stream: true}
	notStreamed := streamed
	notStreamed.Stream = false
	noRequestModel := notStreamed
	noRequestModel.ModelName = ""
	responseModel := semconv.GenAIResponseModel("gemini-x")
	unknown := func(n float64) map[string]float64 { return map[string]float64{"unknown": n} }

	for _, tc := range []struct {
		name      string
		params    GenerateContentParams
		env       map[string]string
		steps     []modelStep
		stopAfter int
		// provider defaults to gcp.gemini.
		provider string
		// histogramAttrs and counterAttrs follow the operation name, provider
		// and request model, which every point carries.
		histogramAttrs, counterAttrs        []attribute.KeyValue
		opInput, opOutput                   float64
		input, output, cacheRead, reasoning map[string]float64
		// cacheWrite defaults to zero under the unknown modality.
		cacheWrite map[string]float64
	}{
		{
			name:   "not streamed",
			params: notStreamed,
			steps: []modelStep{{resp: &model.LLMResponse{
				FinishReason: genai.FinishReasonStop,
				ModelVersion: "gemini-x-001",
				UsageMetadata: &genai.GenerateContentResponseUsageMetadata{
					PromptTokenCount: 10, CachedContentTokenCount: 2, CandidatesTokenCount: 4, ThoughtsTokenCount: 1,
				},
			}}},
			histogramAttrs: []attribute.KeyValue{semconv.GenAIResponseModel("gemini-x-001")},
			counterAttrs:   []attribute.KeyValue{semconv.GenAIResponseModel("gemini-x-001")},
			opInput:        10, opOutput: 5,
			input: unknown(10), output: unknown(5), cacheRead: unknown(2), reasoning: unknown(1),
		},
		{
			// Usage is cumulative, so the newest report is the whole figure,
			// and a final chunk without usage does not hide it.
			name:   "newest usage",
			params: streamed,
			steps: []modelStep{
				{resp: &model.LLMResponse{UsageMetadata: usage(10, 1), Partial: true}},
				{resp: &model.LLMResponse{UsageMetadata: usage(10, 3), Partial: true}},
				{resp: &model.LLMResponse{FinishReason: genai.FinishReasonStop, Partial: true}},
				{resp: &model.LLMResponse{FinishReason: genai.FinishReasonStop}},
			},
			histogramAttrs: []attribute.KeyValue{responseModel},
			counterAttrs:   []attribute.KeyValue{responseModel},
			opInput:        10, opOutput: 3,
			input: unknown(10), output: unknown(3), cacheRead: unknown(0), reasoning: unknown(0),
		},
		{
			// Without a finish reason the span stays open to the end of the
			// stream, which is a normal end, not a panic.
			name:           "stream without a finish reason",
			params:         streamed,
			steps:          []modelStep{{resp: &model.LLMResponse{UsageMetadata: usage(10, 2), Partial: true}}},
			histogramAttrs: []attribute.KeyValue{responseModel},
			counterAttrs:   []attribute.KeyValue{responseModel},
			opInput:        10, opOutput: 2,
			input: unknown(10), output: unknown(2), cacheRead: unknown(0), reasoning: unknown(0),
		},
		{
			name:   "no request model",
			params: noRequestModel,
			steps: []modelStep{{resp: &model.LLMResponse{
				FinishReason:  genai.FinishReasonStop,
				ModelVersion:  "gemini-x-001",
				UsageMetadata: usage(3, 1),
			}}},
			histogramAttrs: []attribute.KeyValue{semconv.GenAIResponseModel("gemini-x-001")},
			counterAttrs:   []attribute.KeyValue{semconv.GenAIResponseModel("gemini-x-001")},
			opInput:        3, opOutput: 1,
			input: unknown(3), output: unknown(1), cacheRead: unknown(0), reasoning: unknown(0),
		},
		{
			name:   "cache writes",
			params: notStreamed,
			steps: []modelStep{{resp: &model.LLMResponse{
				FinishReason:   genai.FinishReasonStop,
				UsageMetadata:  usage(10, 4),
				CustomMetadata: map[string]any{anthropicmodel.CacheCreationInputTokensMetadataKey: int64(6)},
			}}},
			histogramAttrs: []attribute.KeyValue{responseModel},
			counterAttrs:   []attribute.KeyValue{responseModel},
			opInput:        10, opOutput: 4,
			input: unknown(10), output: unknown(4), cacheRead: unknown(0), reasoning: unknown(0), cacheWrite: unknown(6),
		},
		{
			// The aggregate arrives after the span ended, so the span leaves
			// its usage out, but the metrics count it.
			name:   "usage after the span ended",
			params: streamed,
			steps: []modelStep{
				{resp: &model.LLMResponse{Partial: true}},
				{resp: &model.LLMResponse{FinishReason: genai.FinishReasonStop, Partial: true}},
				{resp: &model.LLMResponse{FinishReason: genai.FinishReasonStop, UsageMetadata: usage(7, 2)}},
			},
			histogramAttrs: []attribute.KeyValue{responseModel},
			counterAttrs:   []attribute.KeyValue{responseModel},
			opInput:        7, opOutput: 2,
			input: unknown(7), output: unknown(2), cacheRead: unknown(0), reasoning: unknown(0),
		},
		{
			name:           "API error",
			params:         streamed,
			steps:          []modelStep{{err: genai.APIError{Code: 429}}},
			histogramAttrs: []attribute.KeyValue{semconv.ErrorTypeKey.String("429")},
			input:          unknown(0), output: unknown(0), cacheRead: unknown(0), reasoning: unknown(0),
		},
		{
			name:   "error after a response",
			params: streamed,
			steps: []modelStep{
				{resp: &model.LLMResponse{UsageMetadata: usage(10, 1), ModelVersion: "gemini-x-001", Partial: true}},
				{err: errors.New("boom")},
			},
			histogramAttrs: []attribute.KeyValue{semconv.GenAIResponseModel("gemini-x-001"), semconv.ErrorTypeKey.String("*errors.errorString")},
			counterAttrs:   []attribute.KeyValue{semconv.GenAIResponseModel("gemini-x-001")},
			opInput:        10, opOutput: 1,
			input: unknown(10), output: unknown(1), cacheRead: unknown(0), reasoning: unknown(0),
		},
		{
			// The span ends at the cancellation, but the metrics still name the
			// failure that follows it.
			name:   "real error after a cancellation",
			params: streamed,
			steps: []modelStep{
				{err: fmt.Errorf("stream: %w", context.Canceled)},
				{err: errors.New("boom")},
			},
			histogramAttrs: []attribute.KeyValue{semconv.ErrorTypeKey.String("*errors.errorString")},
			input:          unknown(0), output: unknown(0), cacheRead: unknown(0), reasoning: unknown(0),
		},
		{
			name:   "canceled",
			params: streamed,
			steps:  []modelStep{{err: fmt.Errorf("stream: %w", context.Canceled)}},
			input:  unknown(0), output: unknown(0), cacheRead: unknown(0), reasoning: unknown(0),
		},
		{
			name:   "consumer stops early",
			params: streamed,
			steps: []modelStep{
				{resp: &model.LLMResponse{UsageMetadata: usage(10, 1), Partial: true}},
				{resp: &model.LLMResponse{UsageMetadata: usage(10, 4), FinishReason: genai.FinishReasonStop, Partial: true}},
			},
			stopAfter:      1,
			histogramAttrs: []attribute.KeyValue{responseModel},
			counterAttrs:   []attribute.KeyValue{responseModel},
			opInput:        10, opOutput: 1,
			input: unknown(10), output: unknown(1), cacheRead: unknown(0), reasoning: unknown(0),
		},
		{
			// Ported from https://github.com/google/adk-python/blob/16898f30c9a5c6b2ae1ecf099424afff8fd28e77/tests/unittests/telemetry/test_metrics.py#L558-L603
			// without its cache write count.
			name:   "modality split",
			params: notStreamed,
			steps: []modelStep{{resp: &model.LLMResponse{
				FinishReason: genai.FinishReasonStop,
				UsageMetadata: &genai.GenerateContentResponseUsageMetadata{
					PromptTokenCount:           20,
					PromptTokensDetails:        []*genai.ModalityTokenCount{text(15), {Modality: genai.MediaModalityImage, TokenCount: 5}},
					ToolUsePromptTokenCount:    5,
					ToolUsePromptTokensDetails: []*genai.ModalityTokenCount{text(3)},
					CandidatesTokenCount:       30,
					CandidatesTokensDetails:    []*genai.ModalityTokenCount{text(30)},
					ThoughtsTokenCount:         10,
					CachedContentTokenCount:    8,
					CacheTokensDetails:         []*genai.ModalityTokenCount{{Modality: genai.MediaModalityUnspecified, TokenCount: 8}},
				},
			}}},
			histogramAttrs: []attribute.KeyValue{responseModel},
			counterAttrs:   []attribute.KeyValue{responseModel},
			opInput:        25, opOutput: 40,
			input:     map[string]float64{"text": 18, "image": 5, "unknown": 2},
			output:    map[string]float64{"text": 30, "unknown": 10},
			cacheRead: map[string]float64{"text": 8},
			reasoning: unknown(10),
		},
		{
			name:           "provider from the environment",
			params:         GenerateContentParams{ModelName: "gemini-x", Request: &model.LLMRequest{}},
			env:            map[string]string{"GOOGLE_GENAI_USE_ENTERPRISE": "true"},
			steps:          []modelStep{{resp: &model.LLMResponse{FinishReason: genai.FinishReasonStop}}},
			provider:       "gcp.vertex_ai",
			histogramAttrs: []attribute.KeyValue{responseModel},
			counterAttrs:   []attribute.KeyValue{responseModel},
			input:          unknown(0), output: unknown(0), cacheRead: unknown(0), reasoning: unknown(0),
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			for k, v := range tc.env {
				t.Setenv(k, v)
			}
			setupTestTracer(t)
			reader := setupTestMeter(t)
			generate(t, tc.params, tc.stopAfter, tc.steps...)

			provider := tc.provider
			if provider == "" {
				provider = "gcp.gemini"
			}
			with := func(extra []attribute.KeyValue, more ...attribute.KeyValue) map[attribute.Key]string {
				base := []attribute.KeyValue{semconv.GenAIOperationNameGenerateContent, genAIProviderName.String(provider)}
				if tc.params.ModelName != "" {
					base = append(base, semconv.GenAIRequestModel(tc.params.ModelName))
				}
				return attributesToMap(slices.Concat(base, extra, more))
			}
			byModality := func(split map[string]float64) []metricPoint {
				var out []metricPoint
				for modality, n := range split {
					out = append(out, metricPoint{Attrs: with(tc.counterAttrs, genAITokenModality.String(modality)), Value: n})
				}
				return out
			}
			cacheWrite := tc.cacheWrite
			if cacheWrite == nil {
				cacheWrite = unknown(0)
			}
			want := map[string][]metricPoint{
				"gen_ai.client.inference.duration":                       {{Attrs: with(tc.histogramAttrs), Count: 1}},
				"gen_ai.client.inference.operation.input_tokens":         {{Attrs: with(tc.histogramAttrs), Value: tc.opInput, Count: 1}},
				"gen_ai.client.inference.operation.output_tokens":        {{Attrs: with(tc.histogramAttrs), Value: tc.opOutput, Count: 1}},
				"gen_ai.client.inference.usage.input_tokens":             byModality(tc.input),
				"gen_ai.client.inference.usage.output_tokens":            byModality(tc.output),
				"gen_ai.client.inference.usage.cache_read.input_tokens":  byModality(tc.cacheRead),
				"gen_ai.client.inference.usage.cache_write.input_tokens": byModality(cacheWrite),
				"gen_ai.client.inference.usage.reasoning.output_tokens":  byModality(tc.reasoning),
			}
			got := collectMetrics(t, reader)
			// Timings vary run to run. The chunk timing tests and the test that
			// compares the duration with the span check them.
			delete(got, "gen_ai.client.inference.time_to_first_chunk")
			delete(got, "gen_ai.client.inference.time_per_output_chunk")
			for i := range got["gen_ai.client.inference.duration"] {
				got["gen_ai.client.inference.duration"][i].Value = 0
			}
			if diff := cmp.Diff(want, got, sortPoints); diff != "" {
				t.Errorf("metrics mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// A canceled call did not fail, so neither its span, its event nor its
// metrics name an error, under either schema.
func TestInstrumentGenerateContent_CanceledIsNotAnError(t *testing.T) {
	for _, schema := range []string{otelSemconv136, "otel_semconv_1_44"} {
		t.Run(schema, func(t *testing.T) {
			t.Setenv(adkTelemetrySchemaVersionOptIn, schema)
			spans := setupTestTracer(t)
			reader := setupTestMeter(t)
			logs := &inMemoryExporter{}
			OverrideLoggerForTesting(t, sdklog.NewLoggerProvider(sdklog.WithProcessor(sdklog.NewSimpleProcessor(logs))))
			params := GenerateContentParams{ModelName: "m", Request: &model.LLMRequest{}, Stream: true}

			generate(t, params, 0, modelStep{resp: &model.LLMResponse{Partial: true}}, modelStep{err: fmt.Errorf("stream: %w", context.Canceled)})

			span := spans.GetSpans()[0]
			if span.Status.Code != codes.Unset {
				t.Errorf("span status = %v, want %v", span.Status.Code, codes.Unset)
			}
			for _, kv := range span.Attributes {
				if kv.Key == semconv.ErrorTypeKey {
					t.Errorf("span error.type = %q, want none", kv.Value.Emit())
				}
				if kv.Key == semconv.GenAIResponseFinishReasonsKey && slices.Contains(kv.Value.AsStringSlice(), "error") {
					t.Errorf("span finish reasons = %v, want no error", kv.Value.AsStringSlice())
				}
			}
			for _, r := range logs.records {
				r.WalkAttributes(func(kv attribute.KeyValue) bool {
					if kv.Key == semconv.ErrorTypeKey {
						t.Errorf("event %q error.type = %v, want none", r.EventName(), kv.Value.Emit())
					}
					return true
				})
			}
			for name, points := range collectMetrics(t, reader) {
				for _, p := range points {
					if v, ok := p.Attrs[semconv.ErrorTypeKey]; ok {
						t.Errorf("%s error.type = %q, want none", name, v)
					}
				}
			}
		})
	}
}

// A panic that unwinds the call while its span is open keeps propagating, and
// the span and the metrics both record it as an error of type _OTHER.
func TestInstrumentGenerateContent_Panic(t *testing.T) {
	t.Setenv(adkTelemetrySchemaVersionOptIn, "otel_semconv_1_44")
	spans := setupTestTracer(t)
	reader := setupTestMeter(t)
	params := GenerateContentParams{ModelName: "m", Backend: genai.BackendGeminiAPI, Request: &model.LLMRequest{}, Stream: true}
	call := func(context.Context) iter.Seq2[*model.LLMResponse, error] {
		return func(yield func(*model.LLMResponse, error) bool) {
			if yield(&model.LLMResponse{Partial: true}, nil) {
				panic("model bug")
			}
		}
	}

	func() {
		defer func() {
			if r := recover(); r != "model bug" {
				t.Errorf("recovered %v, want the model's panic", r)
			}
		}()
		for range InstrumentGenerateContent(t.Context(), params, func(r *model.LLMResponse) (*model.LLMResponse, string) { return r, "" }, call) { //nolint:errcheck // the call panics
		}
	}()

	got := spans.GetSpans()
	if len(got) != 1 {
		t.Fatalf("got %d ended spans, want 1", len(got))
	}
	if got[0].Status.Code != codes.Error {
		t.Errorf("span status = %v, want %v", got[0].Status.Code, codes.Error)
	}
	if v := attributesToMap(got[0].Attributes)[semconv.ErrorTypeKey]; v != "_OTHER" {
		t.Errorf("span error.type = %q, want _OTHER", v)
	}
	for _, name := range []string{
		"gen_ai.client.inference.duration",
		"gen_ai.client.inference.operation.input_tokens",
		"gen_ai.client.inference.operation.output_tokens",
	} {
		points := collectMetrics(t, reader)[name]
		if len(points) != 1 || points[0].Attrs[semconv.ErrorTypeKey] != "_OTHER" {
			t.Errorf("%s = %+v, want one point with error.type _OTHER", name, points)
		}
	}
}

// A panic after the span ended keeps propagating, but the call had already
// finished, so neither the span nor the metrics record an error.
func TestInstrumentGenerateContent_PanicAfterSpanEnded(t *testing.T) {
	t.Setenv(adkTelemetrySchemaVersionOptIn, "otel_semconv_1_44")
	spans := setupTestTracer(t)
	reader := setupTestMeter(t)
	params := GenerateContentParams{ModelName: "m", Backend: genai.BackendGeminiAPI, Request: &model.LLMRequest{}, Stream: true}
	call := func(context.Context) iter.Seq2[*model.LLMResponse, error] {
		return func(yield func(*model.LLMResponse, error) bool) {
			if yield(&model.LLMResponse{Partial: true, FinishReason: genai.FinishReasonStop}, nil) {
				panic("model bug")
			}
		}
	}

	func() {
		defer func() {
			if r := recover(); r != "model bug" {
				t.Errorf("recovered %v, want the model's panic", r)
			}
		}()
		for range InstrumentGenerateContent(t.Context(), params, func(r *model.LLMResponse) (*model.LLMResponse, string) { return r, "" }, call) { //nolint:errcheck // the call panics
		}
	}()

	got := spans.GetSpans()
	if len(got) != 1 {
		t.Fatalf("got %d ended spans, want 1", len(got))
	}
	if got[0].Status.Code != codes.Unset {
		t.Errorf("span status = %v, want %v", got[0].Status.Code, codes.Unset)
	}
	points := collectMetrics(t, reader)["gen_ai.client.inference.duration"]
	if len(points) != 1 {
		t.Fatalf("duration points = %+v, want one", points)
	}
	if v, ok := points[0].Attrs[semconv.ErrorTypeKey]; ok {
		t.Errorf("duration error.type = %q, want none", v)
	}
}

func TestInstrumentGenerateContent_ChunkTiming(t *testing.T) {
	t.Setenv(adkTelemetrySchemaVersionOptIn, "otel_semconv_1_44")
	partial := func(r model.LLMResponse) modelStep {
		r.Partial = true
		return modelStep{resp: &r}
	}
	whole := func(r model.LLMResponse) modelStep { return modelStep{resp: &r} }
	stop := model.LLMResponse{FinishReason: genai.FinishReasonStop}
	for _, tc := range []struct {
		name               string
		notStreamed        bool
		steps              []modelStep
		wantFirst, wantPer uint64
	}{
		{
			name:      "stream",
			steps:     []modelStep{partial(model.LLMResponse{}), partial(model.LLMResponse{}), partial(stop), whole(stop)},
			wantFirst: 1, wantPer: 2,
		},
		{
			// A model whose aggregate carries the finish reason, such as
			// adk-python's LiteLLM integration: the aggregate ends the span
			// untimed.
			name:      "aggregate ends the call",
			steps:     []modelStep{partial(model.LLMResponse{}), partial(model.LLMResponse{}), whole(stop)},
			wantFirst: 1, wantPer: 1,
		},
		{
			name:      "model never marks partial",
			steps:     []modelStep{whole(model.LLMResponse{}), whole(model.LLMResponse{}), whole(stop)},
			wantFirst: 1, wantPer: 2,
		},
		{
			name:      "chunk after the span ended",
			steps:     []modelStep{partial(stop), partial(model.LLMResponse{})},
			wantFirst: 1,
		},
		{
			name:  "error before any chunk",
			steps: []modelStep{{err: errors.New("boom")}},
		},
		{
			name:      "error",
			steps:     []modelStep{partial(model.LLMResponse{}), {resp: &model.LLMResponse{Partial: true}, err: errors.New("boom")}},
			wantFirst: 1,
		},
		{
			name:        "not streamed",
			notStreamed: true,
			steps:       []modelStep{whole(stop)},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			setupTestTracer(t)
			reader := setupTestMeter(t)
			params := GenerateContentParams{ModelName: "m", Backend: genai.BackendGeminiAPI, Request: &model.LLMRequest{}, Stream: !tc.notStreamed}

			generate(t, params, 0, tc.steps...)

			got := collectMetrics(t, reader)
			count := func(name string) uint64 {
				var n uint64
				for _, p := range got[name] {
					n += p.Count
				}
				return n
			}
			if n := count("gen_ai.client.inference.time_to_first_chunk"); n != tc.wantFirst {
				t.Errorf("time_to_first_chunk count = %d, want %d", n, tc.wantFirst)
			}
			if n := count("gen_ai.client.inference.time_per_output_chunk"); n != tc.wantPer {
				t.Errorf("time_per_output_chunk count = %d, want %d", n, tc.wantPer)
			}
		})
	}
}

// The first chunk is timed from the start of the call, each later one from
// the chunk before it, and each carries the response model it reports.
func TestInstrumentGenerateContent_ChunkTimingValues(t *testing.T) {
	t.Setenv(adkTelemetrySchemaVersionOptIn, "otel_semconv_1_44")
	setupTestTracer(t)
	reader := setupTestMeter(t)
	const wait = 50 * time.Millisecond
	params := GenerateContentParams{ModelName: "m", Backend: genai.BackendGeminiAPI, Request: &model.LLMRequest{}, Stream: true}

	generate(t, params, 0,
		modelStep{resp: &model.LLMResponse{Partial: true, ModelVersion: "m-001"}, delay: wait},
		modelStep{resp: &model.LLMResponse{Partial: true}},
		modelStep{resp: &model.LLMResponse{Partial: true, FinishReason: genai.FinishReasonStop}},
	)

	got := collectMetrics(t, reader)
	base := []attribute.KeyValue{semconv.GenAIOperationNameGenerateContent, genAIProviderName.String("gcp.gemini"), semconv.GenAIRequestModel("m")}
	first, per := got["gen_ai.client.inference.time_to_first_chunk"], got["gen_ai.client.inference.time_per_output_chunk"]
	if len(first) != 1 || len(per) != 1 {
		t.Fatalf("got %d first-chunk and %d later-chunk points, want 1 each", len(first), len(per))
	}
	if diff := cmp.Diff(attributesToMap(append(slices.Clip(base), semconv.GenAIResponseModel("m-001"))), first[0].Attrs); diff != "" {
		t.Errorf("first chunk attributes mismatch (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff(attributesToMap(base), per[0].Attrs); diff != "" {
		t.Errorf("later chunk attributes mismatch (-want +got):\n%s", diff)
	}
	if first[0].Value < wait.Seconds() {
		t.Errorf("time to first chunk = %vs, want at least %vs", first[0].Value, wait.Seconds())
	}
	// Timed from the start of the call, the two later chunks would take at
	// least twice as long as the first.
	if per[0].Value >= first[0].Value {
		t.Errorf("time per output chunk sums to %vs, want less than the %vs to the first chunk", per[0].Value, first[0].Value)
	}
}

// Under either schema, the duration is the span's, and the metrics have the
// same names and attributes.
func TestInstrumentGenerateContent_MetricsFollowTheSpan(t *testing.T) {
	shapes := map[string]map[string][]map[attribute.Key]string{}
	for _, schema := range []string{otelSemconv136, "otel_semconv_1_44"} {
		t.Run(schema, func(t *testing.T) {
			t.Setenv(adkTelemetrySchemaVersionOptIn, schema)
			spans := setupTestTracer(t)
			reader := setupTestMeter(t)
			params := GenerateContentParams{ModelName: "m", Backend: genai.BackendGeminiAPI, Request: &model.LLMRequest{}, Stream: true}

			generate(t, params, 0,
				modelStep{resp: &model.LLMResponse{Partial: true}},
				modelStep{resp: &model.LLMResponse{Partial: true, FinishReason: genai.FinishReasonStop}, delay: 10 * time.Millisecond},
				modelStep{resp: &model.LLMResponse{FinishReason: genai.FinishReasonStop}, delay: 10 * time.Millisecond},
			)

			span := spans.GetSpans()[0]
			got := collectMetrics(t, reader)
			// Exporters send a span's timestamps as Unix nanoseconds.
			spanDuration := time.Duration(span.EndTime.UnixNano() - span.StartTime.UnixNano()).Seconds()
			if d := got["gen_ai.client.inference.duration"]; len(d) != 1 || d[0].Value != spanDuration {
				t.Errorf("duration = %+v, want one point of the span's %vs", d, spanDuration)
			}
			shape := map[string][]map[attribute.Key]string{}
			for name, points := range got {
				for _, p := range points {
					shape[name] = append(shape[name], p.Attrs)
				}
			}
			shapes[schema] = shape
		})
	}
	if diff := cmp.Diff(shapes[otelSemconv136], shapes["otel_semconv_1_44"]); diff != "" {
		t.Errorf("legacy metrics differ (-legacy +default):\n%s", diff)
	}
}
