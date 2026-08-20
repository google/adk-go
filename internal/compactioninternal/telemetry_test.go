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
	"slices"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/internal/telemetry"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/session/compaction"
)

// spanRecorder installs an in-memory tracer for the calling test.
// spanRecorder installs an in-memory tracer for the duration of a test.
//
// It replaces a package-level tracer, so no test in this file may call
// t.Parallel: a parallel test would swap the tracer while another test is
// reading it, which the race detector reports and which silently sends spans to
// the wrong exporter even when it does not.
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

	got, err := slidingWindowStored(context.Background(), cfg, &staticSession{events: events})
	if err != nil {
		t.Fatalf("slidingWindowStored() error = %v", err)
	}
	if got == nil {
		t.Fatal("slidingWindowStored() produced no summary")
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
		"gen_ai.compaction.summarizer_type": "fakeSummarizer",
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
	// Only the knobs of the configured strategy appear. This says nothing
	// about which strategy produced the span; the trigger attribute does.
	if _, ok := a["gen_ai.compaction.token_threshold"]; ok {
		t.Error("token_threshold attribute is present on a sliding-window span, want it omitted")
	}
	// The range must be recorded so a trace shows what the summary replaced,
	// and it must be the right range in the right layout. Asserting only that
	// the attributes are non-empty left the layout, the bound each one is
	// sourced from, and the timestamps themselves all unprotected.
	// Epoch seconds as a float, matching the reference implementation. The type
	// is asserted as well as the value, because emitting these as strings is the
	// defect this pins and a string attribute reads back as zero here.
	wantStart := float64(at(1).UnixNano()) / float64(time.Second)
	wantEnd := float64(at(4).UnixNano()) / float64(time.Second)
	if got := a["gen_ai.compaction.start_timestamp"]; got.Type() != attribute.FLOAT64 || got.AsFloat64() != wantStart {
		t.Errorf("start_timestamp = %v (%v), want %v (FLOAT64)", got.Emit(), got.Type(), wantStart)
	}
	if got := a["gen_ai.compaction.end_timestamp"]; got.Type() != attribute.FLOAT64 || got.AsFloat64() != wantEnd {
		t.Errorf("end_timestamp = %v (%v), want %v (FLOAT64)", got.Emit(), got.Type(), wantEnd)
	}
	if got := a["gen_ai.compaction.result_event_id"].AsString(); got == "" {
		t.Error("result_event_id is empty, so a trace cannot be joined to the stored summary")
	}
}

func TestCompactionSpanRecordsFailure(t *testing.T) {
	exp := spanRecorder(t)

	events := []*session.Event{
		textEvent("a", "inv1", 1, "q1"), modelTextEvent("b", "inv1", 2, "a1"),
		textEvent("c", "inv2", 3, "q2"), modelTextEvent("d", "inv2", 4, "a2"),
	}
	cfg := &compaction.Config{CompactionInterval: 2, Summarizer: &fakeSummarizer{err: errors.New("boom")}}

	if _, err := slidingWindowStored(context.Background(), cfg, &staticSession{events: events}); err == nil {
		t.Fatal("slidingWindowStored() succeeded, want an error")
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
	// A failed compaction has no result. Recording one would leave a span that
	// is at once an error and a success, naming an event nothing ever stored.
	a := attrs(spans[0].Attributes)
	for _, key := range []string{
		"gen_ai.compaction.result_event_id",
		"gen_ai.compaction.start_timestamp",
		"gen_ai.compaction.end_timestamp",
	} {
		if _, ok := a[key]; ok {
			t.Errorf("%s is present on a failed compaction span, want it omitted", key)
		}
	}
}

// TestNoSpanWhenNothingToCompact pins that evaluating a trigger and declining is
// silent, so the presence of a span in a trace means compaction really ran.
func TestNoSpanWhenNothingToCompact(t *testing.T) {
	exp := spanRecorder(t)

	events := []*session.Event{textEvent("a", "inv1", 1, "q1")}
	cfg := &compaction.Config{CompactionInterval: 5, Summarizer: &fakeSummarizer{summary: "sum"}}

	got, err := slidingWindowStored(context.Background(), cfg, &staticSession{events: events})
	if err != nil || got != nil {
		t.Fatalf("slidingWindowStored() = (%v, %v), want (nil, nil)", got, err)
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

	if _, err := slidingWindowStored(context.Background(), cfg, &staticSession{events: events}); err != nil {
		t.Fatalf("slidingWindowStored() error = %v", err)
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

// bothSummarizer returns a usable compaction event alongside an error, which a
// third-party Summarizer is free to do.
type bothSummarizer struct{}

func (s *bothSummarizer) SummarizeEvents(_ context.Context, events []*session.Event) (*genai.Content, *genai.GenerateContentResponseUsageMetadata, error) {
	// Content alongside an error. The framework must discard the content.
	return genai.NewContentFromText("SUM", "model"), nil, errors.New("boom")
}

// TestCompactionSpanOmitsResultWhenSummarizerAlsoErrors pins that a span is
// never both an error and a success.
//
// A Summarizer may return an event and an error together. The caller discards
// the event, so recording its identity would name something no session holds,
// and the span would report a failure while carrying the attributes of a
// success.
func TestCompactionSpanOmitsResultWhenSummarizerAlsoErrors(t *testing.T) {
	exp := spanRecorder(t)

	events := []*session.Event{
		textEvent("a", "inv1", 1, "q1"), modelTextEvent("b", "inv1", 2, "a1"),
		textEvent("c", "inv2", 3, "q2"), modelTextEvent("d", "inv2", 4, "a2"),
	}
	cfg := &compaction.Config{CompactionInterval: 2, Summarizer: &bothSummarizer{}}

	if _, err := slidingWindowStored(context.Background(), cfg, &staticSession{events: events}); err == nil {
		t.Fatal("slidingWindowStored() succeeded, want an error")
	}

	spans := exp.GetSpans()
	if len(spans) != 1 {
		t.Fatalf("got %d spans, want 1", len(spans))
	}
	if spans[0].Status.Code != codes.Error {
		t.Errorf("span status = %v, want %v", spans[0].Status.Code, codes.Error)
	}
	a := attrs(spans[0].Attributes)
	for _, key := range []string{
		"gen_ai.compaction.result_event_id",
		"gen_ai.compaction.start_timestamp",
		"gen_ai.compaction.end_timestamp",
	} {
		if v, ok := a[key]; ok {
			t.Errorf("%s = %q on a failed compaction span, want it omitted", key, v.AsString())
		}
	}
}

// panickingSummarizer models third-party code that blows up.
type panickingSummarizer struct{}

func (s *panickingSummarizer) SummarizeEvents(_ context.Context, _ []*session.Event) (*genai.Content, *genai.GenerateContentResponseUsageMetadata, error) {
	panic("summarizer exploded")
}

// TestCompactionSpanMarksAPanic pins that a panicking summarizer does not leave
// a span that reads as success.
//
// The OTel SDK records an exception event on the way out but leaves the status
// Unset, and Unset is indistinguishable from a healthy compaction that produced
// nothing. The panic itself still propagates.
func TestCompactionSpanMarksAPanic(t *testing.T) {
	exp := spanRecorder(t)

	events := []*session.Event{
		textEvent("a", "inv1", 1, "q1"), modelTextEvent("b", "inv1", 2, "a1"),
		textEvent("c", "inv2", 3, "q2"), modelTextEvent("d", "inv2", 4, "a2"),
	}
	cfg := &compaction.Config{CompactionInterval: 2, Summarizer: &panickingSummarizer{}}

	func() {
		defer func() {
			if r := recover(); r == nil {
				t.Error("the panic did not propagate; compaction must not swallow it")
			}
		}()
		_, _ = slidingWindowStored(context.Background(), cfg, &staticSession{events: events})
	}()

	spans := exp.GetSpans()
	if len(spans) != 1 {
		t.Fatalf("got %d spans, want 1", len(spans))
	}
	if spans[0].Status.Code != codes.Error {
		t.Errorf("span status = %v, want %v: a panicking summarizer must not look healthy", spans[0].Status.Code, codes.Error)
	}
}

// geminiSummarizer reports a backend the way the real summarizer does.
type geminiSummarizer struct {
	fakeSummarizer
	backend genai.Backend
}

func (s *geminiSummarizer) GetGoogleLLMVariant() genai.Backend { return s.backend }

// TestCompactionSpanRecordsGenAISystem pins gen_ai.system on the span.
//
// It names the system that produced the summary. Two deliberate divergences
// from the reference implementation, both repo-wide rather than compaction's,
// and both asserted here so a repo-wide change has to update this test rather
// than discover it in a dashboard.
//
// The values carry this repo's semconv prefix, "gcp.vertex_ai" against the
// reference's bare "vertex_ai". And a backend this repo cannot name leaves the
// attribute off, where the reference always emits one: naming a provider we
// have not identified is worse than saying nothing. Whatever the mapping is, it
// is shared with the rest of telemetry rather than restated here.
func TestCompactionSpanRecordsGenAISystem(t *testing.T) {
	events := []*session.Event{
		textEvent("a", "inv1", 1, "q1"), modelTextEvent("b", "inv1", 2, "a1"),
		textEvent("c", "inv2", 3, "q2"), modelTextEvent("d", "inv2", 4, "a2"),
	}

	tests := []struct {
		name    string
		backend genai.Backend
		want    string // "" means the attribute must be absent
	}{
		{name: "vertex ai", backend: genai.BackendVertexAI, want: "gcp.vertex_ai"},
		{name: "gemini api", backend: genai.BackendGeminiAPI, want: "gcp.gemini"},
		{name: "summarizer that does not say", backend: genai.BackendUnspecified, want: ""},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			exp := spanRecorder(t)
			cfg := &compaction.Config{
				CompactionInterval: 2,
				Summarizer: &geminiSummarizer{
					fakeSummarizer: fakeSummarizer{summary: "SUM"},
					backend:        tc.backend,
				},
			}
			if _, err := slidingWindowStored(context.Background(), cfg, &staticSession{events: events}); err != nil {
				t.Fatalf("slidingWindowStored() error = %v", err)
			}
			spans := exp.GetSpans()
			if len(spans) != 1 {
				t.Fatalf("got %d spans, want 1", len(spans))
			}
			// The expectation comes from the shared mapping, so this test
			// tracks it rather than freezing a second copy of it.
			if want, ok := telemetry.GenAISystemAttr(tc.backend); ok != (tc.want != "") ||
				(ok && want.Value.AsString() != tc.want) {
				t.Fatalf("the shared mapping now returns (%v, %t) for %v, so this table is stale", want.Value.AsString(), ok, tc.backend)
			}
			got, ok := attrs(spans[0].Attributes)["gen_ai.system"]
			if tc.want == "" {
				if ok {
					t.Errorf("gen_ai.system = %q, want it omitted", got.AsString())
				}
				return
			}
			if !ok {
				t.Fatal("gen_ai.system is absent")
			}
			if got.AsString() != tc.want {
				t.Errorf("gen_ai.system = %q, want %q", got.AsString(), tc.want)
			}
		})
	}
}

// TestCompactionSpanCarriesInvocationAndUsage pins the two attributes that let
// a compaction span be joined to the turn that caused it and costed.
//
// The span is not a child of the turn's span, so without the invocation id
// there is no way to ask which turn a compaction belonged to. And compaction
// spends a model call in order to save tokens later, so a span that does not
// record what it spent cannot show whether it paid for itself.
func TestCompactionSpanCarriesInvocationAndUsage(t *testing.T) {
	exp := spanRecorder(t)

	events := []*session.Event{
		textEvent("a", "inv1", 1, "q1"), modelTextEvent("b", "inv1", 2, "a1"),
		textEvent("c", "inv2", 3, "q2"), modelTextEvent("d", "inv2", 4, "a2"),
	}
	cfg := &compaction.Config{
		CompactionInterval: 2,
		Summarizer: &usageSummarizer{
			fakeSummarizer: fakeSummarizer{summary: "SUM"},
			prompt:         1234,
			output:         56,
		},
	}

	if _, err := slidingWindowStored(context.Background(), cfg, &staticSession{events: events}); err != nil {
		t.Fatalf("slidingWindowStored() error = %v", err)
	}
	a := attrs(exp.GetSpans()[0].Attributes)

	if got := a["gcp.vertex.agent.invocation_id"].AsString(); got != "inv2" {
		t.Errorf("invocation_id = %q, want the turn that triggered compaction (%q)", got, "inv2")
	}
	if got := a["gen_ai.usage.input_tokens"].AsInt64(); got != 1234 {
		t.Errorf("input_tokens = %d, want 1234", got)
	}
	if got := a["gen_ai.usage.output_tokens"].AsInt64(); got != 56 {
		t.Errorf("output_tokens = %d, want 56", got)
	}
}

// usageSummarizer reports token usage the way a real one does.
type usageSummarizer struct {
	fakeSummarizer
	prompt int32
	output int32
}

func (s *usageSummarizer) SummarizeEvents(ctx context.Context, events []*session.Event) (*genai.Content, *genai.GenerateContentResponseUsageMetadata, error) {
	content, _, err := s.fakeSummarizer.SummarizeEvents(ctx, events)
	if err != nil || content == nil {
		return content, nil, err
	}
	return content, &genai.GenerateContentResponseUsageMetadata{
		PromptTokenCount:     s.prompt,
		CandidatesTokenCount: s.output,
	}, nil
}

// TestCompactionSpanAttributeKeySet pins the exact set of attribute keys.
//
// The keys are a contract shared with adk-python, and the individual assertions
// elsewhere only check the keys they name. Adding, renaming or dropping one
// would otherwise pass unnoticed until a dashboard written against the other
// implementation stopped matching.
func TestCompactionSpanAttributeKeySet(t *testing.T) {
	exp := spanRecorder(t)

	events := []*session.Event{
		textEvent("a", "inv1", 1, "q1"), modelTextEvent("b", "inv1", 2, "a1"),
		textEvent("c", "inv2", 3, "q2"), modelTextEvent("d", "inv2", 4, "a2"),
	}
	cfg := &compaction.Config{CompactionInterval: 2, OverlapSize: 1, Summarizer: &fakeSummarizer{summary: "SUM"}}

	if _, err := slidingWindowStored(context.Background(), cfg, &staticSession{events: events}); err != nil {
		t.Fatalf("slidingWindowStored() error = %v", err)
	}

	want := []string{
		"gcp.vertex.agent.invocation_id",
		"gen_ai.compaction.compaction_interval",
		"gen_ai.compaction.end_timestamp",
		"gen_ai.compaction.event_count",
		"gen_ai.compaction.overlap_size",
		"gen_ai.compaction.result_event_id",
		"gen_ai.compaction.start_timestamp",
		"gen_ai.compaction.summarizer_type",
		"gen_ai.compaction.trigger",
		"gen_ai.conversation.id",
		"gen_ai.operation.name",
	}
	var got []string
	for k := range attrs(exp.GetSpans()[0].Attributes) {
		got = append(got, k)
	}
	slices.Sort(got)
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("attribute key set mismatch (-want +got):\n%s\nthese keys are shared with adk-python; change them together", diff)
	}
}

func TestTailRetentionEmitsSpan(t *testing.T) {
	exp := spanRecorder(t)

	events := []*session.Event{
		textEvent("a", "inv1", 1, "q1"),
		withUsage(modelTextEvent("b", "inv1", 2, "a1"), 900),
	}
	cfg := &compaction.Config{TokenThreshold: 100, EventRetentionSize: 0, Summarizer: &fakeSummarizer{summary: "sum"}}

	if _, err := tailRetentionStored(context.Background(), cfg, &staticSession{events: events}, TurnScope{}, nil, nil); err != nil {
		t.Fatalf("tailRetentionStored() error = %v", err)
	}

	spans := exp.GetSpans()
	if len(spans) != 1 {
		t.Fatalf("got %d spans, want 1", len(spans))
	}
	if want := "compact_events token_threshold"; spans[0].Name != want {
		t.Errorf("span name = %q, want %q", spans[0].Name, want)
	}
	a := attrs(spans[0].Attributes)
	if a["gen_ai.compaction.token_threshold"].AsInt64() != 100 {
		t.Errorf("token_threshold = %d, want 100", a["gen_ai.compaction.token_threshold"].AsInt64())
	}
	if _, ok := a["gen_ai.compaction.compaction_interval"]; ok {
		t.Error("compaction_interval attribute is present on a tail-retention span, want it omitted")
	}
}

// TestCompactionSpanRecordsTailRetentionThresholds pins the two attributes only
// a tail-retention span carries.
//
// They are declared in the telemetry commit, where nothing can exercise them
// because tail retention does not exist yet, so they were unprotected: renaming
// either one left the suite green. This is the first commit with a producer.
func TestCompactionSpanRecordsTailRetentionThresholds(t *testing.T) {
	exp := spanRecorder(t)

	events := []*session.Event{
		textEvent("a", "inv1", 1, "q1"), modelTextEvent("b", "inv1", 2, "a1"),
		textEvent("c", "inv2", 3, "q2"), modelTextEvent("d", "inv2", 4, "a2"),
	}
	cfg := &compaction.Config{
		TokenThreshold:     10,
		EventRetentionSize: 1,
		Summarizer:         &fakeSummarizer{summary: "SUM"},
	}

	if _, err := tailRetentionStored(context.Background(), cfg, &staticSession{events: events}, TurnScope{}, func([]*session.Event) int { return 1000 }, nil); err != nil {
		t.Fatalf("tailRetentionStored() error = %v", err)
	}

	spans := exp.GetSpans()
	if len(spans) != 1 {
		t.Fatalf("got %d spans, want 1", len(spans))
	}
	a := attrs(spans[0].Attributes)
	if got := a["gen_ai.compaction.token_threshold"].AsInt64(); got != 10 {
		t.Errorf("token_threshold = %d, want 10", got)
	}
	if got := a["gen_ai.compaction.event_retention_size"].AsInt64(); got != 1 {
		t.Errorf("event_retention_size = %d, want 1", got)
	}
	// The knobs of the strategy that is not configured stay off the span.
	if _, ok := a["gen_ai.compaction.compaction_interval"]; ok {
		t.Error("compaction_interval is present on a tail-retention span, want it omitted")
	}
}

// TestCompactionSpanRecordsADecline pins the difference between a trigger that
// never fired and one that fired and could do nothing.
//
// The first stays silent, so a span in a trace still means compaction was
// wanted. The second used to be silent too, which made a session whose prompt
// grows on every turn look exactly like an idle one.
func TestCompactionSpanRecordsADecline(t *testing.T) {
	exp := spanRecorder(t)

	// Threshold crossed, but the retained tail is the entire history, so there
	// is nothing the compactor may summarize.
	events := []*session.Event{
		textEvent("a", "inv1", 1, "q1"), modelTextEvent("b", "inv1", 2, "a1"),
	}
	cfg := &compaction.Config{
		TokenThreshold:     10,
		EventRetentionSize: 50,
		Summarizer:         &fakeSummarizer{summary: "SUM"},
	}

	got, err := tailRetentionStored(context.Background(), cfg, &staticSession{events: events}, TurnScope{}, func([]*session.Event) int { return 1000 }, nil)
	if err != nil || got != nil {
		t.Fatalf("tailRetentionStored() = (%v, %v), want (nil, nil)", got, err)
	}

	spans := exp.GetSpans()
	if len(spans) != 1 {
		t.Fatalf("got %d spans for a declined compaction, want 1", len(spans))
	}
	reason, ok := attrs(spans[0].Attributes)["gen_ai.compaction.declined"]
	if !ok {
		t.Fatal("the span does not say it declined, so it is indistinguishable from one that compacted")
	}
	if reason.AsString() == "" {
		t.Error("the decline reason is empty")
	}
	if n := attrs(spans[0].Attributes)["gen_ai.compaction.event_count"].AsInt64(); n != 0 {
		t.Errorf("event_count = %d on a declined compaction, want 0", n)
	}
}

// TestCompactionSpanOmitsAbsentTimestamps pins that a range with no bounds is
// reported as absent rather than as the year 1754.
//
// Epoch seconds of a zero time is -6.795e+09, so a compaction covering three
// seconds of history was published as a range 271 years wide, on a span that
// otherwise reported success. An absent attribute is the only form a consumer
// can tell apart from a real reading.
//
// The range is derived from the covered events, so the way to reach an unset
// bound is a covered event that was never stamped, which is what these events
// are. Nothing on the append path used to fill a missing timestamp in.
func TestCompactionSpanOmitsAbsentTimestamps(t *testing.T) {
	exp := spanRecorder(t)

	// Only the oldest event is unstamped, so the window still has a real upper
	// bound and the selection logic, which compares against the previous
	// compaction's end, still sees the later invocations as new.
	first := textEvent("a", "inv1", 1, "q1")
	first.Timestamp = time.Time{}
	events := []*session.Event{
		first, modelTextEvent("b", "inv1", 2, "a1"),
		textEvent("c", "inv2", 3, "q2"), modelTextEvent("d", "inv2", 4, "a2"),
	}
	cfg := &compaction.Config{CompactionInterval: 2, Summarizer: &fakeSummarizer{summary: "SUM"}}

	if _, err := slidingWindowStored(context.Background(), cfg, &staticSession{events: events}); err != nil {
		t.Fatalf("slidingWindowStored() error = %v", err)
	}
	a := attrs(exp.GetSpans()[0].Attributes)

	if v, ok := a["gen_ai.compaction.start_timestamp"]; ok {
		t.Errorf("start_timestamp = %v, want the key to be absent for an unset bound", v.AsFloat64())
	}
	// The bound that does exist is still reported, so absence means absence
	// rather than the attribute simply being dropped.
	if _, ok := a["gen_ai.compaction.end_timestamp"]; !ok {
		t.Error("end_timestamp is missing, want the bound that was recorded")
	}
}

// TestCompactionSpanReportsADiscardedSummary pins that a summary the caller
// threw away is not reported as a stored one.
//
// The span used to end when the summarizer returned, before any of the reasons
// a caller discards a result: a cancelled turn, a failed re-read, a competing
// compaction, a plugin rejecting it, a failed append. Every one of those left a
// span saying the compaction succeeded, carrying a result_event_id that exists
// in no session, so a trace could not distinguish a compaction that shrank a
// prompt from one that spent a model call and changed nothing.
func TestCompactionSpanReportsADiscardedSummary(t *testing.T) {
	exp := spanRecorder(t)

	events := []*session.Event{
		textEvent("a", "inv1", 1, "q1"), modelTextEvent("b", "inv1", 2, "a1"),
		textEvent("c", "inv2", 3, "q2"), modelTextEvent("d", "inv2", 4, "a2"),
	}
	cfg := &compaction.Config{CompactionInterval: 2, Summarizer: &fakeSummarizer{summary: "SUM"}}

	summary, finish, err := SlidingWindow(context.Background(), cfg, &staticSession{events: events}, "")
	if err != nil || summary == nil {
		t.Fatalf("SlidingWindow() = %v, %v, want a summary", summary, err)
	}
	// The caller decides not to keep it.
	finish(nil, "another compaction covering the same events landed while summarizing")

	spans := exp.GetSpans()
	if len(spans) != 1 {
		t.Fatalf("recorded %d spans, want 1", len(spans))
	}
	a := attrs(spans[0].Attributes)
	if got, ok := a["gen_ai.compaction.declined"]; !ok {
		t.Error("the span does not say the summary was discarded")
	} else if got.AsString() == "" {
		t.Error("the discard reason is empty")
	}
	if _, ok := a["gen_ai.compaction.result_event_id"]; ok {
		t.Error("the span names a result event, but nothing reached the session")
	}
}

// TestCompactionSpanRecordsAPanicAsAnException pins that a panicking summarizer
// is visible to an alert keyed on exception.type.
//
// The status was set to Error, which a dashboard sees, but no exception event
// was recorded, so the panic itself was invisible to the usual alerting path.
func TestCompactionSpanRecordsAPanicAsAnException(t *testing.T) {
	exp := spanRecorder(t)

	events := []*session.Event{
		textEvent("a", "inv1", 1, "q1"), modelTextEvent("b", "inv1", 2, "a1"),
		textEvent("c", "inv2", 3, "q2"), modelTextEvent("d", "inv2", 4, "a2"),
	}
	cfg := &compaction.Config{CompactionInterval: 2, Summarizer: &panickingSummarizer{}}

	func() {
		defer func() { _ = recover() }()
		_, _, _ = SlidingWindow(context.Background(), cfg, &staticSession{events: events}, "")
	}()

	spans := exp.GetSpans()
	if len(spans) != 1 {
		t.Fatalf("recorded %d spans, want 1", len(spans))
	}
	var sawException bool
	for _, e := range spans[0].Events {
		if e.Name == "exception" {
			sawException = true
		}
	}
	if !sawException {
		t.Error("no exception event was recorded, so an alert keyed on exception.type misses a panicking summarizer")
	}
}
