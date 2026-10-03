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
	"iter"
	"testing"

	sdklog "go.opentelemetry.io/otel/sdk/log"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
)

type testAgent struct{}

func (testAgent) Name() string        { return "root" }
func (testAgent) Description() string { return "" }

// TestInstrumentInvocation_FirstErrorAndEarlyStop pins that the entrypoint
// span records the first error, not the last, and still ends when the caller
// stops ranging.
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
	for range InstrumentInvocation(t.Context(), testAgent{}, "s", run) {
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

// TestInstrumentGenerateContent_EarlyStop pins that a caller that stops
// ranging before the final response still gets the span ended and the event
// emitted, once.
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
	params := GenerateContentParams{ModelName: "m", Request: &model.LLMRequest{}}

	for range InstrumentGenerateContent(t.Context(), params, func(r *model.LLMResponse) (*model.LLMResponse, string) { return r, "" }, call) {
		break
	}

	if got := len(spans.GetSpans()); got != 1 {
		t.Errorf("got %d ended spans, want 1", got)
	}
	if got := len(logs.records); got != 1 {
		t.Errorf("got %d log records, want 1", got)
	}
}
