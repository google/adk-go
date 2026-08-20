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
			// The prior summary is seeded in under its own ID, so the new
			// compaction inherits what it covered and supersedes it.
			want: []string{"s1", "c", "d"},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got := ids(selectTailRetentionWindow(tc.events, tc.retention, TurnScope{}))
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

	window := selectTailRetentionWindow(events, 2, TurnScope{})
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
	//
	// The whole event list is passed as the second argument, which is what the
	// compactor does. Passing the window there instead lets the test agree with
	// itself: holes are found by scanning everything in the range the window
	// left out, and if the scan only sees the window there is nothing to find.
	// Events a and b are the ones that matter, covered by s1 and therefore
	// absent from the window that rolls s1 up.
	summary, err := newSummaryEvent(window, events, genai.NewContentFromText("new summary", "model"), nil)
	if err != nil {
		t.Fatalf("newSummaryEvent() error = %v", err)
	}
	summary.ID, summary.Timestamp = "s2", at(8)
	if !summary.Actions.Compaction.StartTimestamp.Equal(at(1)) {
		t.Errorf("new summary starts at %v, want %v so it covers the old range",
			summary.Actions.Compaction.StartTimestamp, at(1))
	}
	// Nothing in the range is a hole. a and b are represented by the summary
	// the window rolled up, and the rest of the range is the window itself.
	if got := summary.Actions.Compaction.ExcludedEvents; len(got) != 0 {
		t.Errorf("new summary excludes %v, want nothing: an event an earlier summary covers is covered by this one too", got)
	}

	// s1 is gone rather than sitting beside s2. A rolling summary that cannot
	// subsume the one it was built from leaves both in the prompt, and the pass
	// after that leaves three, which is growth proportional to the length of
	// the conversation.
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
			got, ok := promptTokenCount(tc.events, TurnScope{}, tc.estimate)
			if got != tc.want || ok != tc.wantOK {
				t.Errorf("promptTokenCount() = (%d, TurnScope{}, %t), want (%d, %t)", got, ok, tc.want, tc.wantOK)
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
			// Tool traffic counts. It is what a long turn grows by, and the
			// estimate exists to notice a long turn growing: "search" is six
			// characters, so it is a token and a half's worth on its own.
			name:     "a function call counts its name",
			contents: []*genai.Content{{Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{Name: "search"}}}}},
			want:     1,
		},
		{
			// The payload dominates, and counting only Text saw none of it.
			name: "a function response counts its payload",
			contents: []*genai.Content{{Parts: []*genai.Part{{
				FunctionResponse: &genai.FunctionResponse{
					Name:     "search",
					Response: map[string]any{"result": strings.Repeat("y", 4000)},
				},
			}}}},
			// 4000 characters of payload, so a thousand tokens give or take the
			// JSON punctuation and the name.
			want: 1004,
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

			got, err := tailRetentionStored(context.Background(), cfg, &staticSession{events: tc.events}, TurnScope{}, nil, nil)
			if gotErr := err != nil; gotErr != tc.wantErr {
				t.Fatalf("tailRetentionStored() error = %v, wantErr %t", err, tc.wantErr)
			}
			if gotSummary := got != nil; gotSummary != tc.wantSummary {
				t.Errorf("tailRetentionStored() returned event = %t, want %t", gotSummary, tc.wantSummary)
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

	got, err := tailRetentionStored(context.Background(), cfg, &staticSession{events: events}, TurnScope{},
		func([]*session.Event) int { return 100 }, nil)
	if err != nil {
		t.Fatalf("tailRetentionStored() error = %v", err)
	}
	if got != nil {
		t.Error("tailRetentionStored() compacted despite an estimate below the threshold")
	}

	got, err = tailRetentionStored(context.Background(), cfg, &staticSession{events: events}, TurnScope{},
		func([]*session.Event) int { return 700 }, nil)
	if err != nil {
		t.Fatalf("tailRetentionStored() error = %v", err)
	}
	if got == nil {
		t.Error("tailRetentionStored() did not compact despite an estimate above the threshold")
	}
}

func TestTailRetentionRequiresSummarizer(t *testing.T) {
	t.Parallel()

	_, err := tailRetentionStored(context.Background(), &compaction.Config{TokenThreshold: 1, EventRetentionSize: 0},
		&staticSession{events: []*session.Event{withUsage(modelTextEvent("a", "inv1", 1, "a"), 10)}}, TurnScope{}, nil, nil)
	if err == nil {
		t.Fatal("tailRetentionStored() with no Summarizer returned nil error, want an error")
	}
}

func TestTailRetentionStampsTheSummary(t *testing.T) {
	t.Parallel()

	events := []*session.Event{
		textEvent("a", "inv1", 1, "q1"),
		withUsage(modelTextEvent("b", "inv1", 2, "a1"), 900),
	}
	cfg := &compaction.Config{TokenThreshold: 100, EventRetentionSize: 0, Summarizer: &fakeSummarizer{summary: "sum"}}

	got, err := tailRetentionStored(context.Background(), cfg, &staticSession{events: events}, TurnScope{}, nil, nil)
	if err != nil {
		t.Fatalf("tailRetentionStored() error = %v", err)
	}
	if got == nil {
		t.Fatal("tailRetentionStored() produced no summary")
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

	summary, err := tailRetentionStored(context.Background(), cfg, &staticSession{events: events}, TurnScope{}, nil, nil)
	if err != nil {
		t.Fatalf("tailRetentionStored() error = %v", err)
	}
	if summary == nil {
		t.Fatal("tailRetentionStored() produced no summary")
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

// TestSelectTailRetentionWindowStaysInOneScope checks that the tail window stops
// at the first branch or isolation-scope change.
//
// A summary inherits the branch and isolation scope of what it covers, so a
// window spanning two of them produces one summary that necessarily misattributes
// half its content. Stamped with the first event's scope, it becomes readable by
// agents the filters exist to keep the rest away from.
func TestSelectTailRetentionWindowStaysInOneScope(t *testing.T) {
	t.Parallel()

	root1 := textEvent("a", "inv1", 1, "q1")
	root2 := modelTextEvent("b", "inv1", 2, "a1")
	sub := textEvent("c", "inv2", 3, "SUB-AGENT-SECRET")
	sub.Branch = "root.sub"
	sub.IsolationScope = "scope-1"
	tail1 := textEvent("d", "inv3", 4, "q3")
	tail2 := modelTextEvent("e", "inv3", 5, "a3")

	events := []*session.Event{root1, root2, sub, tail1, tail2}

	window := selectTailRetentionWindow(events, 2, TurnScope{})
	if diff := cmp.Diff([]string{"a", "b"}, ids(window)); diff != "" {
		t.Errorf("selectTailRetentionWindow() mismatch (-want +got):\n%s\nthe window must stop at the scope change", diff)
	}
	for _, ev := range window {
		if ev.Branch != "" || ev.IsolationScope != "" {
			t.Errorf("event %q carries branch %q scope %q, so the window is not homogeneous", ev.ID, ev.Branch, ev.IsolationScope)
		}
	}
}

// TestSelectTailRetentionWindowKeepsATiedBoundaryEvent checks that an event
// stamped exactly at the previous compaction's end is not lost.
//
// The candidate filter used to exclude anything not strictly after that
// instant, while the new range, seeded with the previous summary, starts back
// at the previous start and so covers it. An event on that boundary therefore
// went into no window and inside the next recorded range: summarized by
// nothing, and dropped from every prompt afterwards.
func TestSelectTailRetentionWindowKeepsATiedBoundaryEvent(t *testing.T) {
	t.Parallel()

	prior := compactionEvent("s1", 3, 1, 3, "EARLIER")
	// Appended after the compaction, but stamped on its end instant.
	tied := textEvent("tied", "inv2", 3, "NEVER-SUMMARIZED")
	events := []*session.Event{
		textEvent("a", "inv1", 1, "q1"),
		modelTextEvent("b", "inv1", 2, "a1"),
		prior,
		tied,
		textEvent("c", "inv3", 4, "q3"),
		modelTextEvent("d", "inv3", 5, "a3"),
		textEvent("e", "inv4", 6, "q4"),
	}

	window := selectTailRetentionWindow(events, 1, TurnScope{})
	if !slices.Contains(ids(window), "tied") {
		t.Errorf("window %v does not include the boundary event, so it is covered by the next range without being summarized", ids(window))
	}
}

// TestPromptTokenCountAddsEventsSinceTheLastReport checks that the count is not
// stale by a whole turn.
//
// A reported count describes the prompt of an earlier call. Returning it
// unchanged means everything appended since is invisible, so the call that
// first crosses the threshold is missed and compaction reacts one call late.
func TestPromptTokenCountAddsEventsSinceTheLastReport(t *testing.T) {
	t.Parallel()

	reported := modelTextEvent("a", "inv1", 1, "answer")
	reported.LLMResponse.UsageMetadata = &genai.GenerateContentResponseUsageMetadata{PromptTokenCount: 100}
	events := []*session.Event{
		reported,
		textEvent("b", "inv2", 2, strings.Repeat("x", 400)),
	}

	// The estimator stands in for the real one: four characters per token.
	estimate := func(evs []*session.Event) int {
		n := 0
		for _, ev := range evs {
			for _, p := range utils.Content(ev).Parts {
				n += len(p.Text)
			}
		}
		return n / 4
	}

	got, ok := promptTokenCount(events, TurnScope{}, estimate)
	if !ok {
		t.Fatal("promptTokenCount() reported nothing")
	}
	if got <= 100 {
		t.Errorf("promptTokenCount() = %d, TurnScope{}, want more than the reported 100: the 400 characters appended since are not counted", got)
	}
}

// recordingGate captures the [ProgressGate] calls TailRetention makes.
type recordingGate struct {
	allow     bool
	recorded  []int
	recovered int
	failed    int
}

func (g *recordingGate) AllowAt(int) bool { return g.allow }
func (g *recordingGate) RecordAt(t int)   { g.recorded = append(g.recorded, t) }
func (g *recordingGate) Recovered()       { g.recovered++ }
func (g *recordingGate) Failed()          { g.failed++ }

// TestTailRetentionReArmsTheGateBelowTheThreshold pins that a prompt back under
// the threshold re-arms the gate.
//
// Without this the gate closes on the first compaction of a turn and never
// reopens, so a long turn that keeps growing never compacts again.
func TestTailRetentionReArmsTheGateBelowTheThreshold(t *testing.T) {
	t.Parallel()

	events := []*session.Event{
		textEvent("a", "inv1", 1, "q1"), modelTextEvent("b", "inv1", 2, "a1"),
		textEvent("c", "inv2", 3, "q2"), withUsage(modelTextEvent("d", "inv2", 4, "a2"), 100),
	}
	gate := &recordingGate{allow: true}
	cfg := &compaction.Config{TokenThreshold: 1000, EventRetentionSize: 2, Summarizer: &fakeSummarizer{summary: "sum"}}

	got, err := tailRetentionStored(context.Background(), cfg, &staticSession{events: events}, TurnScope{}, nil, gate)
	if err != nil {
		t.Fatalf("tailRetentionStored() error = %v", err)
	}
	if got != nil {
		t.Fatalf("tailRetentionStored() returned a summary at 100 tokens against a 1000 threshold")
	}
	if gate.recovered != 1 {
		t.Errorf("Recovered() called %d times, want 1: a prompt under the threshold means the last compaction worked", gate.recovered)
	}
}

// TestTailRetentionDoesNotRecordAFailedAttempt pins that a summarizer failure
// leaves the progress gate as it found it.
//
// Recording the attempt rather than the result let one transient error disarm
// compaction for the whole invocation with nothing stored in exchange, and the
// prompt then grew unchecked behind a gate that had stopped retrying.
func TestTailRetentionDoesNotRecordAFailedAttempt(t *testing.T) {
	t.Parallel()

	events := []*session.Event{
		textEvent("a", "inv1", 1, "q1"), modelTextEvent("b", "inv1", 2, "a1"),
		textEvent("c", "inv2", 3, "q2"), withUsage(modelTextEvent("d", "inv2", 4, "a2"), 900),
	}
	gate := &recordingGate{allow: true}
	cfg := &compaction.Config{TokenThreshold: 100, EventRetentionSize: 2, Summarizer: &fakeSummarizer{err: errors.New("boom")}}

	if _, err := tailRetentionStored(context.Background(), cfg, &staticSession{events: events}, TurnScope{}, nil, gate); err == nil {
		t.Fatal("tailRetentionStored() error = nil, want the summarizer failure")
	}
	if len(gate.recorded) != 0 {
		t.Errorf("RecordAt called %v after a failed summarization, want no calls", gate.recorded)
	}
}

// TestTailRetentionRecordsASuccessfulCompaction is the counterpart: a summary
// that was produced must close the gate.
func TestTailRetentionRecordsASuccessfulCompaction(t *testing.T) {
	t.Parallel()

	events := []*session.Event{
		textEvent("a", "inv1", 1, "q1"), modelTextEvent("b", "inv1", 2, "a1"),
		textEvent("c", "inv2", 3, "q2"), withUsage(modelTextEvent("d", "inv2", 4, "a2"), 900),
	}
	gate := &recordingGate{allow: true}
	cfg := &compaction.Config{TokenThreshold: 100, EventRetentionSize: 2, Summarizer: &fakeSummarizer{summary: "sum"}}

	got, err := tailRetentionStored(context.Background(), cfg, &staticSession{events: events}, TurnScope{}, nil, gate)
	if err != nil || got == nil {
		t.Fatalf("tailRetentionStored() = %v, %v, want a summary and no error", got, err)
	}
	if diff := cmp.Diff([]int{900}, gate.recorded); diff != "" {
		t.Errorf("RecordAt calls mismatch (-want +got):\n%s", diff)
	}
}

// TestSelectTailRetentionWindowKeepsTheLiveQuestion pins that the turn being
// answered keeps its own question.
//
// EventRetentionSize counts events and a turn is not a fixed number of them, so
// at every size Validate accepts the question can scroll out of the retained
// tail and be summarized into a paraphrase of the instruction being carried
// out. It is held back separately.
//
// The traffic after it stays eligible, which is the point: excluding the whole
// live invocation would stop a long tool loop compacting itself, and that is
// the case this strategy exists for.
func TestSelectTailRetentionWindowKeepsTheLiveQuestion(t *testing.T) {
	t.Parallel()

	events := []*session.Event{
		textEvent("q1", "inv1", 1, "older question"),
		modelTextEvent("a1", "inv1", 2, "older answer"),
		// The turn in flight: its question, then a long tool loop.
		textEvent("q2", "inv2", 3, "the question being answered"),
		modelTextEvent("t1", "inv2", 4, "tool step 1"),
		modelTextEvent("t2", "inv2", 5, "tool step 2"),
		modelTextEvent("t3", "inv2", 6, "tool step 3"),
	}

	got := ids(selectTailRetentionWindow(events, 2, TurnScope{InvocationID: "inv2"}))

	if slices.Contains(got, "q2") {
		t.Error("the window covers the question the turn is answering")
	}
	// The loop's own older traffic is still compactable, skipping over the
	// question, which only a covered set can express.
	if diff := cmp.Diff([]string{"q1", "a1", "t1"}, got); diff != "" {
		t.Errorf("window mismatch (-want +got):\n%s", diff)
	}
}

// TestSelectTailRetentionWindowStepsPastABlockedHead pins that one unanswered
// tool call does not stop tail retention for the rest of the session.
//
// The window is anchored to the last compaction boundary, so a call awaiting
// human approval, or one whose backend died, sits at the head of every later
// attempt. The sliding window already steps past it; tail retention gave up
// instead, and gave up silently, since "no self-contained prefix" and "nothing
// to do" both come back as nil. Long tool-using sessions are exactly the ones
// this strategy exists for.
func TestSelectTailRetentionWindowStepsPastABlockedHead(t *testing.T) {
	t.Parallel()

	events := []*session.Event{
		// A call at the head that nothing ever answers.
		callEvent("blocked", "inv1", 1, "c-pending"),
		// A complete exchange behind it, which is compactable.
		callEvent("call", "inv2", 2, "c-done"),
		responseEvent("resp", "inv2", 3, "c-done"),
		textEvent("q", "inv3", 4, "another question"),
		modelTextEvent("a", "inv3", 5, "another answer"),
	}

	got := ids(selectTailRetentionWindow(events, 2, TurnScope{}))
	if len(got) == 0 {
		t.Fatal("selectTailRetentionWindow() gave up because the head is blocked")
	}
	if slices.Contains(got, "blocked") {
		t.Error("the window covers the pending call, which must stay raw and visible")
	}
	if diff := cmp.Diff([]string{"call", "resp"}, got); diff != "" {
		t.Errorf("window mismatch (-want +got):\n%s", diff)
	}
}

// TestSkipBlockedHeadKeepsACallWithItsResponse pins that stepping past a
// blocked head never summarizes a response whose call stays raw.
//
// longestSelfContainedPrefix only tracks obligations opened inside the slice it
// is handed, so a response whose call sits in the skipped head looked
// unremarkable: the response was summarized, the call stayed raw, and the model
// was shown a call it had already answered with the answer gone.
func TestSkipBlockedHeadKeepsACallWithItsResponse(t *testing.T) {
	t.Parallel()

	window := []*session.Event{
		// One event opening two calls: the head is blocked on c-pending, and
		// c-two is answered below. Any resume point is therefore past both
		// calls, so the response is the first thing the tail sees.
		multiCallEvent("head", "inv1", 1, "c-pending", "c-two"),
		responseEvent("resp2", "inv1", 2, "c-two"),
		textEvent("q", "inv2", 3, "later question"),
		modelTextEvent("a", "inv2", 4, "later answer"),
	}

	got := ids(skipBlockedHead(window))
	if slices.Contains(got, "resp2") {
		t.Errorf("window %v summarizes a response whose call stays raw in the skipped head", got)
	}
}

// TestPromptTokenCountIgnoresOtherBranches pins that a turn reads a token count
// describing its own prompt.
//
// The count decides whether this turn's prompt is too large, and that prompt is
// assembled with branch and isolation-scope filtering. Reading the newest count
// from anywhere in the session meant a sub-agent whose own prompt is a few
// tokens inherited its parent's, and compacted history it had no business
// compacting.
func TestPromptTokenCountIgnoresOtherBranches(t *testing.T) {
	t.Parallel()

	onBranch := func(ev *session.Event, branch string) *session.Event {
		ev.Branch = branch
		return ev
	}
	events := []*session.Event{
		withUsage(onBranch(modelTextEvent("mine", "inv1", 1, "small"), "parent.child"), 40),
		// A sibling's turn, invisible to parent.child, reporting a huge prompt.
		withUsage(onBranch(modelTextEvent("sibling", "inv2", 2, "huge"), "parent.other"), 200000),
	}

	got, ok := promptTokenCount(events, TurnScope{Branch: "parent.child"}, nil)
	if !ok {
		t.Fatal("promptTokenCount() found no count at all")
	}
	if got != 40 {
		t.Errorf("promptTokenCount() = %d, want 40: the reading came from another branch", got)
	}
}

// TestPromptTokenCountIgnoresOtherIsolationScopes is the same property for
// isolation scope, which is an exact match rather than an ancestor one.
func TestPromptTokenCountIgnoresOtherIsolationScopes(t *testing.T) {
	t.Parallel()

	scoped := func(ev *session.Event, scope string) *session.Event {
		ev.IsolationScope = scope
		return ev
	}
	events := []*session.Event{
		withUsage(scoped(modelTextEvent("mine", "inv1", 1, "small"), "task-a"), 40),
		withUsage(scoped(modelTextEvent("other", "inv2", 2, "huge"), "task-b"), 200000),
	}

	got, ok := promptTokenCount(events, TurnScope{IsolationScope: "task-a"}, nil)
	if !ok {
		t.Fatal("promptTokenCount() found no count at all")
	}
	if got != 40 {
		t.Errorf("promptTokenCount() = %d, want 40: the reading came from another isolation scope", got)
	}
}

// TestTailRetentionDoesNotRecordADiscardedSummary pins that a summary the
// caller throws away leaves the progress gate as it found it.
//
// Recording the attempt rather than the result disarmed compaction for the rest
// of an invocation with nothing stored. Moving the call past the summarizer
// fixed the transient-error case and left four others, because the caller can
// still discard a perfectly good summary: a cancelled turn, a failed re-read, a
// competing compaction, a failed append. Each closed the gate on a summary that
// never existed, and Recovered cannot reopen it because the prompt never drops.
func TestTailRetentionDoesNotRecordADiscardedSummary(t *testing.T) {
	t.Parallel()

	events := []*session.Event{
		textEvent("a", "inv1", 1, "q1"), modelTextEvent("b", "inv1", 2, "a1"),
		textEvent("c", "inv2", 3, "q2"), withUsage(modelTextEvent("d", "inv2", 4, "a2"), 900),
	}
	tests := []struct {
		name          string
		finishErr     error
		discardReason string
		wantRecorded  bool
	}{
		{name: "stored", wantRecorded: true},
		{name: "discarded by the caller", discardReason: "a competing compaction landed"},
		{name: "failed on the way to the session", finishErr: errors.New("append failed")},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			// Per subtest: the summarizer counts its calls, so sharing one
			// across parallel subtests races.
			cfg := &compaction.Config{
				TokenThreshold: 100, EventRetentionSize: 2,
				Summarizer: &fakeSummarizer{summary: "sum"},
			}
			gate := &recordingGate{allow: true}
			summary, finish, err := TailRetention(context.Background(), cfg,
				&staticSession{events: events}, TurnScope{}, nil, gate)
			if err != nil || summary == nil {
				t.Fatalf("TailRetention() = %v, %v, want a summary", summary, err)
			}
			finish(tt.finishErr, tt.discardReason)

			if gotRecorded := len(gate.recorded) > 0; gotRecorded != tt.wantRecorded {
				t.Errorf("gate recorded = %t, want %t: %v", gotRecorded, tt.wantRecorded, gate.recorded)
			}
		})
	}
}

// TestSkipBlockedHeadStillCompactsPastAnAnsweredSibling pins the positive half
// of TestSkipBlockedHeadKeepsACallWithItsResponse, which only asserts that a
// response is not summarized and is therefore satisfied by giving up entirely.
//
// One model turn emitting two calls, one to an ordinary tool and one to a
// long-running tool that never produces a response, is the standard
// long-running shape. The answered sibling's response necessarily sits after
// both calls, so every resume point the scan was willing to consider had that
// response in the tail and a call for it still open in the head, and all of
// them were refused. Nothing after the blockage was ever compacted again, for
// the rest of the session, and because "no window" and "nothing to do yet" are
// both nil it was silent.
//
// The resume point that works is the one just after the response: the head then
// holds the call and its answer, only the long-running call is still open, and
// the tail answers nothing. The scan skipped it because it only resumed after
// an event that opened an obligation.
func TestSkipBlockedHeadStillCompactsPastAnAnsweredSibling(t *testing.T) {
	t.Parallel()

	window := []*session.Event{
		multiCallEvent("head", "inv1", 1, "c-longrunning", "c-two"),
		responseEvent("resp2", "inv1", 2, "c-two"),
		textEvent("q", "inv2", 3, "later question"),
		modelTextEvent("a", "inv2", 4, "later answer"),
		textEvent("q2", "inv3", 5, "later question 2"),
		modelTextEvent("a2", "inv3", 6, "later answer 2"),
	}

	got := ids(skipBlockedHead(window))
	if len(got) == 0 {
		t.Fatal("skipBlockedHead() = nil: one unanswered call in a parallel pair stalls compaction for the rest of the session")
	}
	if slices.Contains(got, "resp2") {
		t.Errorf("window %v summarizes a response whose call stays raw in the skipped head", got)
	}
	if diff := cmp.Diff([]string{"q", "a", "q2", "a2"}, got); diff != "" {
		t.Errorf("window mismatch (-want +got):\n%s", diff)
	}
}

// TestTailRetentionClosesTheGateOnlyWhenTryingAgainIsPointless pins which
// outcomes stop this invocation attempting again, and which do not.
//
// The gate has to sit between two failure modes that pull opposite ways.
// Closing it on every attempt disarms compaction on a transient error and
// Recovered cannot reopen it, because the prompt never drops. Closing it only
// on a stored summary means a persistently failing summarizer is retried before
// every model call for the life of the turn.
//
// A discard is neither. The commonest one is a competing compaction landing in
// the range, which means a summary was stored and the picture changed, so the
// next call should look again.
func TestTailRetentionClosesTheGateOnlyWhenTryingAgainIsPointless(t *testing.T) {
	t.Parallel()

	events := []*session.Event{
		withUsage(modelTextEvent("a", "inv1", 1, "a1"), 900),
		textEvent("b", "inv1", 2, "q2"),
		modelTextEvent("c", "inv1", 3, "a2"),
		textEvent("d", "inv1", 4, "q3"),
	}
	run := func(t *testing.T, sum compaction.Summarizer, finishWith func(finish Finish)) *recordingGate {
		t.Helper()
		gate := &recordingGate{allow: true}
		cfg := &compaction.Config{TokenThreshold: 100, EventRetentionSize: 1, Summarizer: sum}
		_, finish, err := TailRetention(context.Background(), cfg, &staticSession{events: events},
			TurnScope{}, nil, gate)
		if err == nil && finishWith != nil {
			finishWith(finish)
		}
		return gate
	}

	t.Run("the summarizer errored", func(t *testing.T) {
		t.Parallel()
		// The caller never gets a Finish on this path, so closing the gate only
		// in that callback did not reach here at all.
		g := run(t, &fakeSummarizer{err: errTestSummarizer}, nil)
		if g.failed != 1 {
			t.Errorf("gate.failed = %d, want 1: a failing summarizer must not be retried every model call", g.failed)
		}
	})
	t.Run("the summarizer declined", func(t *testing.T) {
		t.Parallel()
		g := run(t, &fakeSummarizer{}, nil)
		if g.failed != 1 {
			t.Errorf("gate.failed = %d, want 1: a decline gets no Finish either", g.failed)
		}
	})
	t.Run("the append failed", func(t *testing.T) {
		t.Parallel()
		g := run(t, &fakeSummarizer{summary: "a summary"}, func(f Finish) { f(errTestAppend, "") })
		if g.failed != 1 {
			t.Errorf("gate.failed = %d, want 1", g.failed)
		}
		if len(g.recorded) != 0 {
			t.Errorf("gate recorded %v for a summary that was never stored", g.recorded)
		}
	})
	t.Run("the caller discarded it", func(t *testing.T) {
		t.Parallel()
		g := run(t, &fakeSummarizer{summary: "a summary"}, func(f Finish) { f(nil, "a competing compaction landed") })
		if g.failed != 0 {
			t.Errorf("gate.failed = %d, want 0: a discard means the picture changed, not that trying again is pointless", g.failed)
		}
		if len(g.recorded) != 0 {
			t.Errorf("gate recorded %v for a summary that was never stored", g.recorded)
		}
	})
	t.Run("it was stored", func(t *testing.T) {
		t.Parallel()
		g := run(t, &fakeSummarizer{summary: "a summary"}, func(f Finish) { f(nil, "") })
		if len(g.recorded) != 1 {
			t.Errorf("gate.recorded = %v, want one entry", g.recorded)
		}
		if g.failed != 0 {
			t.Errorf("gate.failed = %d, want 0", g.failed)
		}
	})
}

var (
	errTestAppend     = errors.New("append failed")
	errTestSummarizer = errors.New("summarizer unavailable")
)

// TestSelectTailRetentionWindowSeedsWhileHoldingBackTheLiveHead exercises the
// two mechanisms together, which is the combination production always runs and
// the suite never covered.
//
// The seed keeps history to one rolling summary. The live-head hold-back keeps
// the current turn's question out of the window. Every other case in this file
// passes an empty TurnScope, so the hold-back is inert in all of them, while
// the processor always supplies a real InvocationID. The two were each
// exercised alone and never together, which is how a defect in their
// interaction could sit here unnoticed.
func TestSelectTailRetentionWindowSeedsWhileHoldingBackTheLiveHead(t *testing.T) {
	t.Parallel()

	events := []*session.Event{
		textEvent("a", "inv1", 1, "q1"), modelTextEvent("b", "inv1", 2, "a1"),
		compactionEvent("s1", 3, 1, 2, "earlier summary"),
		textEvent("c", "inv2", 4, "q2"), modelTextEvent("d", "inv2", 5, "a2"),
		// The live turn: its question must stay out, and the events after it
		// are fair game.
		textEvent("live-q", "inv3", 6, "the question being answered now"),
		modelTextEvent("live-a", "inv3", 7, "partial work"),
		textEvent("live-b", "inv3", 8, "more partial work"),
	}

	window := selectTailRetentionWindow(events, 1, TurnScope{InvocationID: "inv3"})
	got := ids(window)

	if slices.Contains(got, "live-q") {
		t.Errorf("window %v summarizes the question the turn is answering", got)
	}
	if len(got) == 0 || got[0] != "s1" {
		t.Errorf("window %v does not open with the previous summary, so it cannot supersede it", got)
	}

	// And the record built from it supersedes the one it was seeded with.
	summary, err := newSummaryEvent(window, events, genai.NewContentFromText("new summary", "model"), nil)
	if err != nil {
		t.Fatalf("newSummaryEvent() error = %v", err)
	}
	summary.ID, summary.Timestamp = "s2", at(9)
	after := ids(Apply(append(events, summary)))
	if slices.Contains(after, "s1") {
		t.Errorf("prompt %v still carries the superseded summary alongside its replacement", after)
	}
	if !slices.Contains(after, "live-q") {
		t.Errorf("prompt %v lost the question the turn is answering", after)
	}
}

// TestTailRetentionEmptyWindowIsNotYetNotNever pins that finding nothing to
// summarize does not stop the invocation trying later.
//
// The window is empty when the retained tail is still the whole history, or
// when a tool call at its head is unanswered. A long tool-calling turn keeps
// appending, so a later model call in the same invocation can have a window
// when this one does not. Treating the empty window as a spent attempt stopped
// compaction for the whole turn on the first cheap check, before the summarizer
// had been asked anything.
//
// The distinction that matters: this path makes no model call, so re-checking
// is free, where retrying a failing summarizer is not.
func TestTailRetentionEmptyWindowIsNotYetNotNever(t *testing.T) {
	t.Parallel()

	gate := &recordingGate{allow: true}
	// One event over the threshold and a retention size that keeps it, so
	// there is nothing left to summarize.
	events := []*session.Event{withUsage(modelTextEvent("a", "inv1", 1, "a1"), 900)}
	cfg := &compaction.Config{TokenThreshold: 100, EventRetentionSize: 4, Summarizer: &fakeSummarizer{summary: "s"}}

	summary, _, err := TailRetention(context.Background(), cfg, &staticSession{events: events},
		TurnScope{}, nil, gate)
	if err != nil || summary != nil {
		t.Fatalf("TailRetention() = %v, %v, want no summary and no error", summary, err)
	}
	if gate.failed != 0 {
		t.Errorf("gate.failed = %d, want 0: an empty window is not a spent attempt", gate.failed)
	}
}
