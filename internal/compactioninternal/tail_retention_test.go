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
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/internal/utils"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/session/compaction"
)

// withUsage tags an event with an observed prompt token count.
func withUsage(ev *session.Event, promptTokens int32) *session.Event {
	ev.LLMResponse.UsageMetadata = &genai.GenerateContentResponseUsageMetadata{
		PromptTokenCount: promptTokens,
	}
	return ev
}

func TestSelectTailRetentionWindow(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name      string
		events    []*session.Event
		retention int
		want      []string
	}{
		{
			name:      "fewer events than the retention size",
			events:    []*session.Event{textEvent("a", "inv1", 1, "q1"), textEvent("b", "inv1", 2, "a1")},
			retention: 5,
			want:      nil,
		},
		{
			name:      "exactly the retention size keeps everything raw",
			events:    []*session.Event{textEvent("a", "inv1", 1, "q1"), textEvent("b", "inv1", 2, "a1")},
			retention: 2,
			want:      nil,
		},
		{
			name: "older events are compacted, the tail stays raw",
			events: []*session.Event{
				textEvent("a", "inv1", 1, "q1"), modelTextEvent("b", "inv1", 2, "a1"),
				textEvent("c", "inv2", 3, "q2"), modelTextEvent("d", "inv2", 4, "a2"),
			},
			retention: 2,
			want:      []string{"a", "b"},
		},
		{
			name: "zero retention compacts everything",
			events: []*session.Event{
				textEvent("a", "inv1", 1, "q1"), modelTextEvent("b", "inv1", 2, "a1"),
			},
			retention: 0,
			want:      []string{"a", "b"},
		},
		{
			name: "the cut moves back past a same-timestamp group",
			events: []*session.Event{
				textEvent("a", "inv1", 1, "q1"),
				// b, c and d all share timestamp 2. Cutting between them would
				// give the summary an EndTimestamp that also covers a retained
				// event, silently dropping it from the prompt.
				modelTextEvent("b", "inv1", 2, "a1"),
				modelTextEvent("c", "inv1", 2, "a2"),
				modelTextEvent("d", "inv1", 2, "a3"),
			},
			retention: 2,
			want:      []string{"a"},
		},
		{
			name: "a whole same-timestamp tail leaves nothing to compact",
			events: []*session.Event{
				modelTextEvent("a", "inv1", 2, "a1"),
				modelTextEvent("b", "inv1", 2, "a2"),
				modelTextEvent("c", "inv1", 2, "a3"),
			},
			retention: 1,
			want:      nil,
		},
		{
			name: "window is trimmed so a call is not split from its response",
			events: []*session.Event{
				textEvent("a", "inv1", 1, "q1"),
				callEvent("b", "inv1", 2, "c1"),
				responseEvent("c", "inv1", 3, "c1"),
				modelTextEvent("d", "inv1", 4, "a1"),
			},
			// Cutting at 3 would compact [a, b] and strand the response.
			retention: 1,
			want:      []string{"a", "b", "c"},
		},
		{
			name: "nil when the compactable prefix is entirely an open call",
			events: []*session.Event{
				callEvent("a", "inv1", 1, "c1"),
				responseEvent("b", "inv1", 2, "c1"),
				modelTextEvent("c", "inv1", 3, "a1"),
			},
			retention: 2,
			want:      nil,
		},
		{
			name: "only events after the previous compaction are candidates",
			events: []*session.Event{
				textEvent("a", "inv1", 1, "q1"), modelTextEvent("b", "inv1", 2, "a1"),
				compactionEvent("s1", 3, 1, 2, "earlier summary"),
				textEvent("c", "inv2", 4, "q2"), modelTextEvent("d", "inv2", 5, "a2"),
				textEvent("e", "inv3", 6, "q3"), modelTextEvent("f", "inv3", 7, "a3"),
			},
			retention: 2,
			// The prior summary is seeded in as "" (a synthetic event with no
			// ID) so the new compaction supersedes it.
			want: []string{"", "c", "d"},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got := ids(selectTailRetentionWindow(tc.events, tc.retention))
			if diff := cmp.Diff(tc.want, got); diff != "" {
				t.Errorf("selectTailRetentionWindow(retention=%d) mismatch (-want +got):\n%s", tc.retention, diff)
			}
		})
	}
}

// TestSelectTailRetentionWindowSeedsPreviousSummary checks the rolling-summary
// seed: the new window opens with the previous summary, timestamped at the
// start of the range that summary covered, so the new compaction subsumes it.
func TestSelectTailRetentionWindowSeedsPreviousSummary(t *testing.T) {
	t.Parallel()

	events := []*session.Event{
		textEvent("a", "inv1", 1, "q1"), modelTextEvent("b", "inv1", 2, "a1"),
		compactionEvent("s1", 3, 1, 2, "earlier summary"),
		textEvent("c", "inv2", 4, "q2"), modelTextEvent("d", "inv2", 5, "a2"),
		textEvent("e", "inv3", 6, "q3"), modelTextEvent("f", "inv3", 7, "a3"),
	}

	window := selectTailRetentionWindow(events, 2)
	if len(window) == 0 {
		t.Fatal("selectTailRetentionWindow() returned nothing")
	}

	seed := window[0]
	if !seed.Timestamp.Equal(at(1)) {
		t.Errorf("seed timestamp = %v, want the previous compaction's start %v", seed.Timestamp, at(1))
	}
	if seed.Author != "model" {
		t.Errorf("seed author = %q, want %q", seed.Author, "model")
	}
	if got := utils.TextParts(utils.Content(seed)); len(got) != 1 || got[0] != "earlier summary" {
		t.Errorf("seed text = %v, want the previous summary", got)
	}

	// Summarizing this window must produce a range that strictly contains the
	// old one, so Apply treats the old summary as subsumed.
	summary, err := compaction.NewSummaryEvent(window, genai.NewContentFromText("new summary", "model"), nil)
	if err != nil {
		t.Fatalf("compaction.NewSummaryEvent() error = %v", err)
	}
	summary.ID, summary.Timestamp = "s2", at(8)
	if !summary.Actions.Compaction.StartTimestamp.Equal(at(1)) {
		t.Errorf("new summary starts at %v, want %v so it covers the old range",
			summary.Actions.Compaction.StartTimestamp, at(1))
	}

	got := ids(Apply(append(events, summary)))
	if diff := cmp.Diff([]string{"s2", "e", "f"}, got); diff != "" {
		t.Errorf("after the rolling compaction, prompt events mismatch (-want +got):\n%s", diff)
	}
}

func TestPromptTokenCount(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		events   []*session.Event
		estimate TokenCounter
		want     int
		wantOK   bool
	}{
		{
			name:   "no events and no estimator",
			want:   0,
			wantOK: false,
		},
		{
			name:     "estimator used when nothing reported a count",
			events:   []*session.Event{textEvent("a", "inv1", 1, "q1")},
			estimate: func([]*session.Event) int { return 123 },
			want:     123,
			wantOK:   true,
		},
		{
			name:     "estimator returning zero means unknown",
			events:   []*session.Event{textEvent("a", "inv1", 1, "q1")},
			estimate: func([]*session.Event) int { return 0 },
			want:     0,
			wantOK:   false,
		},
		{
			name: "observed count wins over the estimator",
			events: []*session.Event{
				withUsage(modelTextEvent("a", "inv1", 1, "a1"), 500),
			},
			estimate: func([]*session.Event) int { return 123 },
			want:     500,
			wantOK:   true,
		},
		{
			name: "the most recent observed count wins",
			events: []*session.Event{
				withUsage(modelTextEvent("a", "inv1", 1, "a1"), 500),
				textEvent("b", "inv2", 2, "q2"),
				withUsage(modelTextEvent("c", "inv2", 3, "a2"), 900),
			},
			want:   900,
			wantOK: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got, ok := promptTokenCount(tc.events, tc.estimate)
			if got != tc.want || ok != tc.wantOK {
				t.Errorf("promptTokenCount() = (%d, %t), want (%d, %t)", got, ok, tc.want, tc.wantOK)
			}
		})
	}
}

func TestEstimateTokensFromContents(t *testing.T) {
	t.Parallel()

	text := func(n int) *genai.Content {
		return &genai.Content{Parts: []*genai.Part{{Text: strings.Repeat("x", n)}}}
	}

	tests := []struct {
		name     string
		contents []*genai.Content
		want     int
	}{
		{name: "nil", contents: nil, want: 0},
		{name: "empty text", contents: []*genai.Content{text(0)}, want: 0},
		{name: "below one token", contents: []*genai.Content{text(3)}, want: 0},
		{name: "exactly one token", contents: []*genai.Content{text(4)}, want: 1},
		{name: "summed across contents", contents: []*genai.Content{text(2000), text(2000)}, want: 1000},
		{name: "nil content is skipped", contents: []*genai.Content{nil, text(4)}, want: 1},
		{name: "nil part is skipped", contents: []*genai.Content{{Parts: []*genai.Part{nil, {Text: "xxxx"}}}}, want: 1},
		{
			// Non-text parts are invisible to the estimate, which is why it is
			// only a floor until real usage metadata arrives.
			name:     "function call contributes nothing",
			contents: []*genai.Content{{Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{Name: "search"}}}}},
			want:     0,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			if got := EstimateTokensFromContents(tc.contents); got != tc.want {
				t.Errorf("EstimateTokensFromContents() = %d, want %d", got, tc.want)
			}
		})
	}
}

func TestTailRetention(t *testing.T) {
	t.Parallel()

	fourEvents := []*session.Event{
		textEvent("a", "inv1", 1, "q1"), modelTextEvent("b", "inv1", 2, "a1"),
		textEvent("c", "inv2", 3, "q2"), withUsage(modelTextEvent("d", "inv2", 4, "a2"), 900),
	}

	tests := []struct {
		name        string
		cfg         *compaction.Config
		events      []*session.Event
		summarizer  *fakeSummarizer
		wantSummary bool
		wantWindow  []string
		wantErr     bool
	}{
		{
			name:       "nil config does nothing",
			cfg:        nil,
			events:     fourEvents,
			summarizer: &fakeSummarizer{summary: "sum"},
		},
		{
			name:       "sliding-window-only config does nothing",
			cfg:        &compaction.Config{CompactionInterval: 2},
			events:     fourEvents,
			summarizer: &fakeSummarizer{summary: "sum"},
		},
		{
			name:       "below the threshold",
			cfg:        &compaction.Config{TokenThreshold: 1000, EventRetentionSize: 2},
			events:     fourEvents,
			summarizer: &fakeSummarizer{summary: "sum"},
		},
		{
			name:        "at the threshold",
			cfg:         &compaction.Config{TokenThreshold: 900, EventRetentionSize: 2},
			events:      fourEvents,
			summarizer:  &fakeSummarizer{summary: "sum"},
			wantSummary: true,
			wantWindow:  []string{"a", "b"},
		},
		{
			name:        "above the threshold",
			cfg:         &compaction.Config{TokenThreshold: 100, EventRetentionSize: 2},
			events:      fourEvents,
			summarizer:  &fakeSummarizer{summary: "sum"},
			wantSummary: true,
			wantWindow:  []string{"a", "b"},
		},
		{
			name:       "threshold reached but the tail retains everything",
			cfg:        &compaction.Config{TokenThreshold: 100, EventRetentionSize: 10},
			events:     fourEvents,
			summarizer: &fakeSummarizer{summary: "sum"},
		},
		{
			name:        "summarizer declines",
			cfg:         &compaction.Config{TokenThreshold: 100, EventRetentionSize: 2},
			events:      fourEvents,
			summarizer:  &fakeSummarizer{},
			wantSummary: false,
			wantWindow:  []string{"a", "b"},
		},
		{
			name:       "summarizer fails",
			cfg:        &compaction.Config{TokenThreshold: 100, EventRetentionSize: 2},
			events:     fourEvents,
			summarizer: &fakeSummarizer{err: errors.New("boom")},
			wantWindow: []string{"a", "b"},
			wantErr:    true,
		},
		{
			name:       "no observed token count and no estimate",
			cfg:        &compaction.Config{TokenThreshold: 1, EventRetentionSize: 1},
			events:     []*session.Event{textEvent("a", "inv1", 1, "q1"), textEvent("b", "inv1", 2, "q2")},
			summarizer: &fakeSummarizer{summary: "sum"},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			cfg := tc.cfg
			if cfg != nil {
				copied := *cfg
				copied.Summarizer = tc.summarizer
				cfg = &copied
			}

			got, err := TailRetention(context.Background(), cfg, &staticSession{events: tc.events}, nil)
			if gotErr := err != nil; gotErr != tc.wantErr {
				t.Fatalf("TailRetention() error = %v, wantErr %t", err, tc.wantErr)
			}
			if gotSummary := got != nil; gotSummary != tc.wantSummary {
				t.Errorf("TailRetention() returned event = %t, want %t", gotSummary, tc.wantSummary)
			}
			var gotWindow []string
			if len(tc.summarizer.windows) > 0 {
				gotWindow = tc.summarizer.windows[0]
			}
			if diff := cmp.Diff(tc.wantWindow, gotWindow); diff != "" {
				t.Errorf("summarizer window mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestTailRetentionUsesTheEstimator(t *testing.T) {
	t.Parallel()

	// No event carries usage metadata, so the estimator decides.
	events := []*session.Event{
		textEvent("a", "inv1", 1, "q1"), modelTextEvent("b", "inv1", 2, "a1"),
		textEvent("c", "inv2", 3, "q2"), modelTextEvent("d", "inv2", 4, "a2"),
	}
	summarizer := &fakeSummarizer{summary: "sum"}
	cfg := &compaction.Config{TokenThreshold: 500, EventRetentionSize: 2, Summarizer: summarizer}

	got, err := TailRetention(context.Background(), cfg, &staticSession{events: events},
		func([]*session.Event) int { return 100 })
	if err != nil {
		t.Fatalf("TailRetention() error = %v", err)
	}
	if got != nil {
		t.Error("TailRetention() compacted despite an estimate below the threshold")
	}

	got, err = TailRetention(context.Background(), cfg, &staticSession{events: events},
		func([]*session.Event) int { return 700 })
	if err != nil {
		t.Fatalf("TailRetention() error = %v", err)
	}
	if got == nil {
		t.Error("TailRetention() did not compact despite an estimate above the threshold")
	}
}

func TestTailRetentionRequiresSummarizer(t *testing.T) {
	t.Parallel()

	_, err := TailRetention(context.Background(), &compaction.Config{TokenThreshold: 1, EventRetentionSize: 0},
		&staticSession{events: []*session.Event{withUsage(modelTextEvent("a", "inv1", 1, "a"), 10)}}, nil)
	if err == nil {
		t.Fatal("TailRetention() with no Summarizer returned nil error, want an error")
	}
}

func TestTailRetentionStampsTheSummary(t *testing.T) {
	t.Parallel()

	events := []*session.Event{
		textEvent("a", "inv1", 1, "q1"),
		withUsage(modelTextEvent("b", "inv1", 2, "a1"), 900),
	}
	cfg := &compaction.Config{TokenThreshold: 100, EventRetentionSize: 0, Summarizer: &fakeSummarizer{summary: "sum"}}

	got, err := TailRetention(context.Background(), cfg, &staticSession{events: events}, nil)
	if err != nil {
		t.Fatalf("TailRetention() error = %v", err)
	}
	if got == nil {
		t.Fatal("TailRetention() produced no summary")
	}
	// The event must be ready to append without the caller filling anything in.
	if got.ID == "" {
		t.Error("summary has no ID")
	}
	if got.InvocationID == "" {
		t.Error("summary has no InvocationID")
	}
	if got.Timestamp.IsZero() {
		t.Error("summary has no Timestamp")
	}
	for _, ev := range events {
		if got.InvocationID == ev.InvocationID {
			t.Errorf("summary reuses invocation ID %q from a covered event; window selection counts invocations, so it must be fresh", got.InvocationID)
		}
	}
}

// TestTailRetentionThenApplyShrinksHistory is the round trip: compact, then
// build the prompt, and confirm the covered events are gone.
func TestTailRetentionThenApplyShrinksHistory(t *testing.T) {
	t.Parallel()

	events := []*session.Event{
		textEvent("a", "inv1", 1, "q1"), modelTextEvent("b", "inv1", 2, "a1"),
		textEvent("c", "inv1", 3, "q2"), withUsage(modelTextEvent("d", "inv1", 4, "a2"), 5000),
	}
	cfg := &compaction.Config{TokenThreshold: 1000, EventRetentionSize: 2, Summarizer: &fakeSummarizer{summary: "SUMMARY"}}

	summary, err := TailRetention(context.Background(), cfg, &staticSession{events: events}, nil)
	if err != nil {
		t.Fatalf("TailRetention() error = %v", err)
	}
	if summary == nil {
		t.Fatal("TailRetention() produced no summary")
	}
	summary.ID = "s1"

	got := Apply(append(events, summary))
	if diff := cmp.Diff([]string{"s1", "c", "d"}, ids(got)); diff != "" {
		t.Errorf("post-compaction prompt events mismatch (-want +got):\n%s", diff)
	}
	if texts := utils.TextParts(utils.Content(got[0])); len(texts) != 1 || texts[0] != "SUMMARY" {
		t.Errorf("first prompt event = %v, want the summary text", texts)
	}
}
