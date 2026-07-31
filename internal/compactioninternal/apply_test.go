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
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/internal/utils"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/session/compaction"
)

func TestApply(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name   string
		events []*session.Event
		want   []string // event IDs in the order Apply returns them
	}{
		{
			name:   "no compaction events is a passthrough",
			events: []*session.Event{textEvent("a", "inv1", 1, "hi"), modelTextEvent("b", "inv1", 2, "hello")},
			want:   []string{"a", "b"},
		},
		{
			name: "covered events are replaced by the summary",
			events: []*session.Event{
				textEvent("a", "inv1", 1, "q1"),
				modelTextEvent("b", "inv1", 2, "a1"),
				textEvent("c", "inv2", 3, "q2"),
				modelTextEvent("d", "inv2", 4, "a2"),
				compactionEvent("s1", 5, 1, 4, "summary"),
				textEvent("e", "inv3", 6, "q3"),
			},
			want: []string{"s1", "e"},
		},
		{
			name: "events after the range survive",
			events: []*session.Event{
				textEvent("a", "inv1", 1, "q1"),
				modelTextEvent("b", "inv1", 2, "a1"),
				compactionEvent("s1", 3, 1, 2, "summary"),
				textEvent("c", "inv2", 4, "q2"),
				modelTextEvent("d", "inv2", 5, "a2"),
			},
			want: []string{"s1", "c", "d"},
		},
		{
			name: "an event predating the summary but outside its range survives",
			events: []*session.Event{
				textEvent("a", "inv1", 1, "before the range"),
				textEvent("b", "inv2", 3, "q2"),
				modelTextEvent("c", "inv2", 4, "a2"),
				compactionEvent("s1", 5, 3, 4, "summary of inv2"),
			},
			want: []string{"a", "s1"},
		},
		{
			name: "a subsumed compaction is dropped along with its summary",
			events: []*session.Event{
				textEvent("a", "inv1", 1, "q1"),
				modelTextEvent("b", "inv1", 2, "a1"),
				compactionEvent("s1", 3, 1, 2, "narrow"),
				textEvent("c", "inv2", 4, "q2"),
				modelTextEvent("d", "inv2", 5, "a2"),
				compactionEvent("s2", 6, 1, 5, "wide"),
			},
			want: []string{"s2"},
		},
		{
			name: "partially overlapping compactions both survive",
			events: []*session.Event{
				textEvent("a", "inv1", 1, "q1"),
				textEvent("b", "inv2", 2, "q2"),
				compactionEvent("s1", 3, 1, 2, "left"),
				textEvent("c", "inv3", 4, "q3"),
				compactionEvent("s2", 5, 2, 4, "right"),
			},
			want: []string{"s1", "s2"},
		},
		{
			name: "an event tying the end timestamp counts as covered",
			events: []*session.Event{
				textEvent("a", "inv1", 1, "q1"),
				textEvent("b", "inv1", 2, "also at 2"),
				compactionEvent("s1", 3, 1, 2, "summary"),
			},
			want: []string{"s1"},
		},
		{
			name: "a compaction with no content is ignored entirely",
			events: []*session.Event{
				textEvent("a", "inv1", 1, "q1"),
				{
					ID:        "s1",
					Timestamp: at(2),
					Actions:   session.EventActions{Compaction: &session.EventCompaction{StartTimestamp: at(1), EndTimestamp: at(1)}},
				},
			},
			want: []string{"a"},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got := ids(Apply(tc.events))
			if diff := cmp.Diff(tc.want, got); diff != "" {
				t.Errorf("Apply() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestApplyMaterializesSummaryAsModelContent(t *testing.T) {
	t.Parallel()

	events := []*session.Event{
		textEvent("a", "inv1", 1, "q1"),
		compactionEvent("s1", 3, 1, 1, "the summary text"),
	}

	got := Apply(events)
	if len(got) != 1 {
		t.Fatalf("Apply() returned %d events, want 1: %v", len(got), ids(got))
	}
	summary := got[0]

	if summary.Author != "model" {
		t.Errorf("summary Author = %q, want %q", summary.Author, "model")
	}
	if !summary.Timestamp.Equal(at(1)) {
		t.Errorf("summary Timestamp = %v, want the compaction end timestamp %v", summary.Timestamp, at(1))
	}
	texts := utils.TextParts(utils.Content(summary))
	if diff := cmp.Diff([]string{"the summary text"}, texts); diff != "" {
		t.Errorf("summary text mismatch (-want +got):\n%s", diff)
	}
}

func TestApplyDoesNotMutateInput(t *testing.T) {
	t.Parallel()

	stored := compactionEvent("s1", 3, 1, 2, "summary")
	events := []*session.Event{textEvent("a", "inv1", 1, "q1"), stored}

	Apply(events)

	// The stored event is what lives in the session; rewriting it in place
	// would corrupt history and make the next Apply see a bogus author.
	if stored.Author != "user" {
		t.Errorf("stored compaction Author = %q, want it left as %q", stored.Author, "user")
	}
	if !stored.Timestamp.Equal(at(3)) {
		t.Errorf("stored compaction Timestamp = %v, want it left at %v", stored.Timestamp, at(3))
	}
	if stored.LLMResponse.Content != nil {
		t.Errorf("stored compaction Content = %v, want it left nil", stored.LLMResponse.Content)
	}
}

func TestApplyRecoversCompactedLongRunningCall(t *testing.T) {
	t.Parallel()

	// A long-running call and its placeholder response are compacted away, and
	// the real result lands afterwards. Without recovery the surviving response
	// would be orphaned, which prompt assembly rejects.
	call := callEvent("call", "inv1", 2, "c1")
	call.LongRunningToolIDs = []string{"c1"}
	placeholder := responseEvent("placeholder", "inv1", 3, "c1")
	result := responseEvent("result", "inv2", 6, "c1")

	events := []*session.Event{
		textEvent("a", "inv1", 1, "please start the job"),
		call,
		placeholder,
		compactionEvent("s1", 5, 1, 3, "summary"),
		result,
	}

	got := Apply(events)
	if diff := cmp.Diff([]string{"s1", "call", "result"}, ids(got)); diff != "" {
		t.Errorf("Apply() mismatch (-want +got):\n%s", diff)
	}
}

func TestApplyRecoversParallelSiblingResponse(t *testing.T) {
	t.Parallel()

	// Two parallel long-running calls in one event. Only one response survives
	// compaction; the sibling's final response must be re-injected so it does
	// not look like a still-pending call.
	call := multiCallEvent("call", "inv1", 2, "c1", "c2")
	call.LongRunningToolIDs = []string{"c1", "c2"}

	events := []*session.Event{
		textEvent("a", "inv1", 1, "start both"),
		call,
		responseEvent("ph1", "inv1", 3, "c1"),
		responseEvent("done2", "inv1", 4, "c2"),
		compactionEvent("s1", 6, 1, 4, "summary"),
		responseEvent("done1", "inv2", 7, "c1"),
	}

	got := Apply(events)
	if diff := cmp.Diff([]string{"s1", "call", "done2", "done1"}, ids(got)); diff != "" {
		t.Errorf("Apply() mismatch (-want +got):\n%s", diff)
	}
}

func TestApplyLeavesNonLongRunningOrphanAlone(t *testing.T) {
	t.Parallel()

	// A response whose call was compacted but was never long-running signals a
	// genuine inconsistency. Recovery deliberately does not paper over it, so
	// downstream prompt assembly can surface the problem.
	events := []*session.Event{
		textEvent("a", "inv1", 1, "q1"),
		callEvent("call", "inv1", 2, "c1"), // no LongRunningToolIDs
		compactionEvent("s1", 4, 1, 2, "summary"),
		responseEvent("result", "inv2", 5, "c1"),
	}

	got := Apply(events)
	if diff := cmp.Diff([]string{"s1", "result"}, ids(got)); diff != "" {
		t.Errorf("Apply() mismatch (-want +got):\n%s", diff)
	}
}

func TestNewSummaryEvent(t *testing.T) {
	t.Parallel()

	events := []*session.Event{
		textEvent("a", "inv1", 3, "q1"),
		modelTextEvent("b", "inv1", 7, "a1"),
	}
	summaryContent := utils.Content(modelTextEvent("x", "inv1", 0, "the summary"))

	got, err := compaction.NewSummaryEvent(events, summaryContent, nil)
	if err != nil {
		t.Fatalf("compaction.NewSummaryEvent() error = %v", err)
	}

	if got.Author != "user" {
		t.Errorf("Author = %q, want %q", got.Author, "user")
	}
	if got.Actions.Compaction == nil {
		t.Fatal("Actions.Compaction is nil, want a compaction range")
	}
	if !got.Actions.Compaction.StartTimestamp.Equal(at(3)) {
		t.Errorf("StartTimestamp = %v, want %v", got.Actions.Compaction.StartTimestamp, at(3))
	}
	if !got.Actions.Compaction.EndTimestamp.Equal(at(7)) {
		t.Errorf("EndTimestamp = %v, want %v", got.Actions.Compaction.EndTimestamp, at(7))
	}
	if role := got.Actions.Compaction.CompactedContent.Role; role != "model" {
		t.Errorf("CompactedContent.Role = %q, want %q", role, "model")
	}
	// The caller's content must not be re-roled underneath them.
	if summaryContent.Role != "model" {
		t.Logf("input content role was already %q", summaryContent.Role)
	}
}

func TestNewSummaryEventRejectsBadInput(t *testing.T) {
	t.Parallel()

	ordered := []*session.Event{textEvent("a", "inv1", 1, "q1"), modelTextEvent("b", "inv1", 4, "a1")}
	content := genai.NewContentFromText("summary", "model")

	tests := []struct {
		name    string
		events  []*session.Event
		summary *genai.Content
		wantErr bool
	}{
		{name: "ok", events: ordered, summary: content},
		{name: "single event is a valid degenerate range", events: ordered[:1], summary: content},
		{name: "no events", events: nil, summary: content, wantErr: true},
		{name: "nil summary", events: ordered, summary: nil, wantErr: true},
		{
			// An inverted range covers nothing, so the compacted turns would
			// stay in every future prompt while a summary was still paid for.
			name:    "events out of chronological order",
			events:  []*session.Event{modelTextEvent("b", "inv1", 4, "a1"), textEvent("a", "inv1", 1, "q1")},
			summary: content,
			wantErr: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			_, err := compaction.NewSummaryEvent(tc.events, tc.summary, nil)
			if gotErr := err != nil; gotErr != tc.wantErr {
				t.Errorf("compaction.NewSummaryEvent() error = %v, wantErr %t", err, tc.wantErr)
			}
		})
	}
}

func TestApplyIgnoresInvertedRange(t *testing.T) {
	t.Parallel()

	// session.EventCompaction is a plain struct, so a caller can build an
	// inverted range directly, bypassing NewSummaryEvent. Apply must not
	// materialize it, or the summary would duplicate raw events it never
	// covered.
	inverted := compactionEvent("s1", 5, 4, 1, "bogus summary")
	events := []*session.Event{
		textEvent("a", "inv1", 1, "q1"),
		modelTextEvent("b", "inv1", 2, "a1"),
		inverted,
	}

	got := ids(Apply(events))
	if diff := cmp.Diff([]string{"a", "b"}, got); diff != "" {
		t.Errorf("Apply() mismatch (-want +got):\n%s", diff)
	}
}

// TestContentlessCompactionIsNeverConversation guards the predicate split. An
// event declaring a compaction but carrying no content is bookkeeping, and must
// never be counted as a real turn by window selection.
func TestContentlessCompactionIsNeverConversation(t *testing.T) {
	t.Parallel()

	contentless := &session.Event{
		ID:           "s1",
		InvocationID: "e-compaction",
		Timestamp:    at(5),
		Actions: session.EventActions{
			Compaction: &session.EventCompaction{StartTimestamp: at(1), EndTimestamp: at(4)},
		},
	}

	if compaction.IsCompactionEvent(contentless) {
		t.Error("compaction.IsCompactionEvent() = true for a contentless compaction, want false (nothing to show a model)")
	}
	if !hasCompaction(contentless) {
		t.Error("hasCompaction() = false for a contentless compaction, want true (it is still bookkeeping)")
	}

	events := []*session.Event{
		textEvent("a", "inv1", 1, "q1"), modelTextEvent("b", "inv1", 2, "a1"),
		textEvent("c", "inv2", 3, "q2"), modelTextEvent("d", "inv2", 4, "a2"),
		contentless,
	}

	// Its own invocation ID must not be counted toward the interval, and its
	// range must still act as the compaction boundary.
	if got := ids(selectSlidingWindow(events, 3, 0)); got != nil {
		t.Errorf("selectSlidingWindow() = %v, want nil: only 2 real invocations exist, so the interval of 3 is unmet", got)
	}
	if got := LatestCompactionEvent(events); got != contentless {
		t.Errorf("LatestCompactionEvent() = %v, want the contentless compaction (it still marks the boundary)", got)
	}
}

// TestApplyRecoveryBoundary pins exactly which orphans are recovered.
//
// The two cases differ only in whether the call was long-running, which is the
// whole basis of the gate. Recovery is deliberately not widened: an orphan with
// no long-running call is a genuine inconsistency, and guessing at it would hide
// a bug rather than surface one. Note that such an orphan is later dropped from
// the prompt silently by rearrangeEventsForFunctionResponsesInHistory.
func TestApplyRecoveryBoundary(t *testing.T) {
	t.Parallel()

	build := func(longRunning bool) []*session.Event {
		call := callEvent("call", "inv1", 2, "c1")
		if longRunning {
			call.LongRunningToolIDs = []string{"c1"}
		}
		return []*session.Event{
			textEvent("a", "inv1", 1, "start"),
			call,
			responseEvent("placeholder", "inv1", 3, "c1"),
			compactionEvent("s1", 5, 1, 3, "summary"),
			responseEvent("result", "inv2", 6, "c1"),
		}
	}

	tests := []struct {
		name        string
		longRunning bool
		want        []string
	}{
		{
			name:        "long-running call is restored so the response stays paired",
			longRunning: true,
			want:        []string{"s1", "call", "result"},
		},
		{
			name:        "non long-running call is not restored",
			longRunning: false,
			want:        []string{"s1", "result"},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			if diff := cmp.Diff(tc.want, ids(Apply(build(tc.longRunning)))); diff != "" {
				t.Errorf("Apply() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}
