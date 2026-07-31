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

package compactioninternal

import (
	"context"
	"errors"
	"testing"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"

	"google.golang.org/adk/v2/internal/telemetry"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/session/compaction"
)

// spanRecorder installs an in-memory tracer for the calling test.
func spanRecorder(t *testing.T) *tracetest.InMemoryExporter {
	t.Helper()
	exp := tracetest.NewInMemoryExporter()
	telemetry.OverrideTracerForTesting(t, sdktrace.NewTracerProvider(sdktrace.WithSyncer(exp)))
	return exp
}

// attrs flattens a span's attributes for lookup by key.
func attrs(kvs []attribute.KeyValue) map[string]attribute.Value {
	out := make(map[string]attribute.Value, len(kvs))
	for _, kv := range kvs {
		out[string(kv.Key)] = kv.Value
	}
	return out
}

func TestSlidingWindowEmitsSpan(t *testing.T) {
	exp := spanRecorder(t)

	events := []*session.Event{
		textEvent("a", "inv1", 1, "q1"), modelTextEvent("b", "inv1", 2, "a1"),
		textEvent("c", "inv2", 3, "q2"), modelTextEvent("d", "inv2", 4, "a2"),
	}
	cfg := &compaction.Config{CompactionInterval: 2, OverlapSize: 1, Summarizer: &fakeSummarizer{summary: "sum"}}

	got, err := SlidingWindow(context.Background(), cfg, &staticSession{events: events})
	if err != nil {
		t.Fatalf("SlidingWindow() error = %v", err)
	}
	if got == nil {
		t.Fatal("SlidingWindow() produced no summary")
	}

	spans := exp.GetSpans()
	if len(spans) != 1 {
		t.Fatalf("got %d spans, want 1", len(spans))
	}
	span := spans[0]
	if want := "compact_events sliding_window"; span.Name != want {
		t.Errorf("span name = %q, want %q", span.Name, want)
	}
	if span.Status.Code == codes.Error {
		t.Errorf("span status = error, want unset: %v", span.Status)
	}

	a := attrs(span.Attributes)
	for key, want := range map[string]string{
		"gen_ai.operation.name":             "compact_events",
		"gen_ai.conversation.id":            "sess",
		"gen_ai.compaction.trigger":         "sliding_window",
		"gen_ai.compaction.summarizer_type": "*compactioninternal.fakeSummarizer",
		"gen_ai.compaction.result_event_id": got.ID,
	} {
		if a[key].AsString() != want {
			t.Errorf("attribute %s = %q, want %q", key, a[key].AsString(), want)
		}
	}
	if a["gen_ai.compaction.event_count"].AsInt64() != 4 {
		t.Errorf("event_count = %d, want 4", a["gen_ai.compaction.event_count"].AsInt64())
	}
	if a["gen_ai.compaction.compaction_interval"].AsInt64() != 2 {
		t.Errorf("compaction_interval = %d, want 2", a["gen_ai.compaction.compaction_interval"].AsInt64())
	}
	if a["gen_ai.compaction.overlap_size"].AsInt64() != 1 {
		t.Errorf("overlap_size = %d, want 1", a["gen_ai.compaction.overlap_size"].AsInt64())
	}
	// Only the strategy that fired should be described.
	if _, ok := a["gen_ai.compaction.token_threshold"]; ok {
		t.Error("token_threshold attribute is present on a sliding-window span, want it omitted")
	}
	// The range must be recorded so a trace shows what the summary replaced.
	if a["gen_ai.compaction.start_timestamp"].AsString() == "" {
		t.Error("start_timestamp attribute is empty")
	}
	if a["gen_ai.compaction.end_timestamp"].AsString() == "" {
		t.Error("end_timestamp attribute is empty")
	}
}

func TestCompactionSpanRecordsFailure(t *testing.T) {
	exp := spanRecorder(t)

	events := []*session.Event{
		textEvent("a", "inv1", 1, "q1"), modelTextEvent("b", "inv1", 2, "a1"),
		textEvent("c", "inv2", 3, "q2"), modelTextEvent("d", "inv2", 4, "a2"),
	}
	cfg := &compaction.Config{CompactionInterval: 2, Summarizer: &fakeSummarizer{err: errors.New("boom")}}

	if _, err := SlidingWindow(context.Background(), cfg, &staticSession{events: events}); err == nil {
		t.Fatal("SlidingWindow() succeeded, want an error")
	}

	spans := exp.GetSpans()
	if len(spans) != 1 {
		t.Fatalf("got %d spans, want 1", len(spans))
	}
	if spans[0].Status.Code != codes.Error {
		t.Errorf("span status = %v, want %v", spans[0].Status.Code, codes.Error)
	}
	if len(spans[0].Events) == 0 {
		t.Error("span records no exception event, so the failure reason is lost")
	}
}

// TestNoSpanWhenNothingToCompact pins that evaluating a trigger and declining is
// silent, so the presence of a span in a trace means compaction really ran.
func TestNoSpanWhenNothingToCompact(t *testing.T) {
	exp := spanRecorder(t)

	events := []*session.Event{textEvent("a", "inv1", 1, "q1")}
	cfg := &compaction.Config{CompactionInterval: 5, Summarizer: &fakeSummarizer{summary: "sum"}}

	got, err := SlidingWindow(context.Background(), cfg, &staticSession{events: events})
	if err != nil || got != nil {
		t.Fatalf("SlidingWindow() = (%v, %v), want (nil, nil)", got, err)
	}
	if n := len(exp.GetSpans()); n != 0 {
		t.Errorf("got %d spans when the interval was not reached, want 0", n)
	}
}

// TestSpanRecordsDecliningSummarizer distinguishes "ran and produced nothing"
// from "ran and failed": the span exists and is successful, but carries no
// result attributes.
func TestSpanRecordsDecliningSummarizer(t *testing.T) {
	exp := spanRecorder(t)

	events := []*session.Event{
		textEvent("a", "inv1", 1, "q1"), modelTextEvent("b", "inv1", 2, "a1"),
		textEvent("c", "inv2", 3, "q2"), modelTextEvent("d", "inv2", 4, "a2"),
	}
	cfg := &compaction.Config{CompactionInterval: 2, Summarizer: &fakeSummarizer{}}

	if _, err := SlidingWindow(context.Background(), cfg, &staticSession{events: events}); err != nil {
		t.Fatalf("SlidingWindow() error = %v", err)
	}

	spans := exp.GetSpans()
	if len(spans) != 1 {
		t.Fatalf("got %d spans, want 1", len(spans))
	}
	if spans[0].Status.Code == codes.Error {
		t.Errorf("span status = error, want success for a summarizer that merely declined")
	}
	if _, ok := attrs(spans[0].Attributes)["gen_ai.compaction.result_event_id"]; ok {
		t.Error("result_event_id is set although no summary was produced")
	}
}
