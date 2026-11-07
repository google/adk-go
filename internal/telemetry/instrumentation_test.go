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

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/log"
	sdklog "go.opentelemetry.io/otel/sdk/log"
	semconv "go.opentelemetry.io/otel/semconv/v1.36.0"
	"go.opentelemetry.io/otel/trace"
	"google.golang.org/genai"

	agentinternal "google.golang.org/adk/v2/internal/agent"
	"google.golang.org/adk/v2/model"
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

func TestInstrumentInvocation_NestedOpensNoSpan(t *testing.T) {
	t.Setenv(adkTelemetrySchemaVersionOptIn, "otel_semconv_1_44")
	exporter := setupTestTracer(t)
	run := func(context.Context) iter.Seq2[*session.Event, error] {
		return func(yield func(*session.Event, error) bool) { yield(&session.Event{}, nil) }
	}

	ctx, outer := startInvokeWorkflowSpan(t.Context(), "outer", "s")
	for range InstrumentInvocation(ctx, testAgent{}, "s", run) {
	}
	if !outer.IsRecording() {
		t.Error("nested invocation ended the enclosing workflow span")
	}
	outer.End()

	if got := len(exporter.GetSpans()); got != 1 {
		t.Errorf("got %d spans, want only the outer one", got)
	}
}

func TestInvocationWorkflowSpanOwnership(t *testing.T) {
	for _, resumeChild := range []bool{false, true} {
		name := "root"
		if resumeChild {
			name = "resumed_child"
		}
		t.Run(name, func(t *testing.T) {
			t.Setenv(adkTelemetrySchemaVersionOptIn, "otel_semconv_1_44")
			exporter := setupTestTracer(t)
			root := &fakeAgent{State: agentinternal.State{AgentType: agentinternal.TypeWorkflowAgent}, name: "root"}
			child := &fakeAgent{State: agentinternal.State{AgentType: agentinternal.TypeWorkflowAgent}, name: "child"}
			ictx := &fakeInvocationContext{t: t, sessionID: "s", invocationID: "i"}
			var invocation trace.Span
			for range InstrumentInvocation(t.Context(), root, "s", func(ctx context.Context) iter.Seq2[*session.Event, error] {
				return func(func(*session.Event, error) bool) {
					invocation = trace.SpanFromContext(ctx)
					if !resumeChild {
						var owned trace.Span
						ctx, owned = StartNodeSpan(ctx, ictx, OperationAgent{Agent: root})
						owned.End()
						if !trace.SpanFromContext(ctx).IsRecording() {
							t.Fatal("root workflow ended the runner's span")
						}
						if !trace.SpanContextFromContext(ctx).Equal(invocation.SpanContext()) {
							t.Fatal("root workflow opened a duplicate span")
						}
					}
					childCtx, childSpan := StartNodeSpan(ctx, ictx, OperationAgent{Agent: child})
					if trace.SpanContextFromContext(childCtx).Equal(invocation.SpanContext()) {
						t.Fatal("child workflow reused the root span")
					}
					childSpan.End()
					if !invocation.IsRecording() {
						t.Error("child workflow ended the runner's span")
					}
				}
			}) {
			}
			spans := exporter.GetSpans()
			if len(spans) != 2 {
				t.Fatalf("ended spans = %d, want 2", len(spans))
			}
			if !spans[0].Parent.Equal(invocation.SpanContext()) {
				t.Error("child workflow lost its invocation parent")
			}
			var nested bool
			for _, attr := range spans[0].Attributes {
				if attr.Key == genAIWorkflowNested {
					nested = attr.Value.AsBool()
				}
			}
			if !nested {
				t.Error("child workflow is not marked nested")
			}
		})
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

	for range InstrumentGenerateContent(t.Context(), params, func(r *model.LLMResponse) (*model.LLMResponse, string) { return r, "" }, call) {
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
	for range InstrumentGenerateContent(t.Context(), params, func(r *model.LLMResponse) (*model.LLMResponse, string) { return r, "" }, call) {
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
