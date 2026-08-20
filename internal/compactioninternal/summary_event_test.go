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
	"slices"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/internal/utils"
	"google.golang.org/adk/v2/session"
)

func TestNewSummaryEvent(t *testing.T) {
	t.Parallel()

	events := []*session.Event{
		textEvent("a", "inv1", 3, "q1"),
		modelTextEvent("b", "inv1", 7, "a1"),
	}
	summaryContent := utils.Content(modelTextEvent("x", "inv1", 0, "the summary"))

	got, err := newSummaryEvent(events, events, summaryContent, nil)
	if err != nil {
		t.Fatalf("newSummaryEvent() error = %v", err)
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
			// Not an error any more: the box is a true minimum and maximum, and
			// the covered set names the events regardless of their order.
			name:    "events out of chronological order",
			events:  []*session.Event{modelTextEvent("b", "inv1", 4, "a1"), textEvent("a", "inv1", 1, "q1")},
			summary: content,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			_, err := newSummaryEvent(tc.events, tc.events, tc.summary, nil)
			if gotErr := err != nil; gotErr != tc.wantErr {
				t.Errorf("newSummaryEvent() error = %v, wantErr %t", err, tc.wantErr)
			}
		})
	}
}

// TestNewSummaryEventKeepsPartMetadata checks that a surviving text part is
// copied whole rather than rebuilt from its text.
//
// A text part can carry metadata that belongs with it, a thought signature
// above all, which the model expects to get back alongside the text it
// accompanies. Rebuilding the part would drop that silently.
func TestNewSummaryEventKeepsPartMetadata(t *testing.T) {
	t.Parallel()

	events := []*session.Event{
		{Timestamp: time.Unix(1, 0)},
		{Timestamp: time.Unix(2, 0)},
	}
	summary := &genai.Content{Role: "model", Parts: []*genai.Part{{
		Text:             "the summary",
		ThoughtSignature: []byte("opaque-signature"),
	}}}

	got, err := newSummaryEvent(events, events, summary, nil)
	if err != nil {
		t.Fatalf("newSummaryEvent() error = %v", err)
	}
	parts := got.Actions.Compaction.CompactedContent.Parts
	if len(parts) != 1 {
		t.Fatalf("got %d parts, want 1", len(parts))
	}
	if diff := cmp.Diff([]byte("opaque-signature"), parts[0].ThoughtSignature); diff != "" {
		t.Errorf("ThoughtSignature mismatch (-want +got):\n%s", diff)
	}
}

// TestNewSummaryEventRejectsProselessSummary checks that a summary whose only
// text rides on an actionable part is refused rather than stored empty.
func TestNewSummaryEventRejectsProselessSummary(t *testing.T) {
	t.Parallel()

	events := []*session.Event{
		{Timestamp: time.Unix(1, 0)},
		{Timestamp: time.Unix(2, 0)},
	}
	summary := &genai.Content{Role: "model", Parts: []*genai.Part{{
		Text:         "transferring now",
		FunctionCall: &genai.FunctionCall{Name: "transfer_funds"},
	}}}

	if _, err := newSummaryEvent(events, events, summary, nil); err == nil {
		t.Error("newSummaryEvent() accepted a summary with no prose, want an error rather than an empty summary")
	}
}

// TestCompactionEventIsNotAFinalResponse checks that a stored summary does not
// present itself to streaming consumers as an agent's final response.
//
// A compaction event carries a record and no content, which satisfies every
// other clause of IsFinalResponse, so a client deciding what to show a user
// would surface an empty final response every time compaction ran.
func TestCompactionEventIsNotAFinalResponse(t *testing.T) {
	t.Parallel()

	events := []*session.Event{
		{Timestamp: time.Unix(1, 0)},
		{Timestamp: time.Unix(2, 0)},
	}
	got, err := newSummaryEvent(events, events, genai.NewContentFromText("the summary", "model"), nil)
	if err != nil {
		t.Fatalf("newSummaryEvent() error = %v", err)
	}

	if got.IsFinalResponse() {
		t.Error("a compaction event reports IsFinalResponse() = true; streaming clients would show it as an empty reply")
	}
}

// TestNewSummaryEventBoundsAnOutOfOrderWindow checks that a window whose
// timestamps are not sorted is summarized rather than refused.
//
// A stored event list is in append order while timestamps are stamped at
// creation, so two invocations in flight on one session leave it non-monotonic
// with one clock and no skew. Refusing those windows stopped the session
// compacting for good: nothing was recorded, so the same window was re-selected
// and re-refused on every later turn.
//
// The recorded box has to be a true minimum and maximum, or it would not bound
// the events the summary names.
func TestNewSummaryEventBoundsAnOutOfOrderWindow(t *testing.T) {
	t.Parallel()

	events := []*session.Event{
		{ID: "a", Timestamp: time.Unix(1, 0)},
		{ID: "b", Timestamp: time.Unix(9, 0)}, // past the last one
		{ID: "c", Timestamp: time.Unix(5, 0)},
	}

	got, err := newSummaryEvent(events, events, genai.NewContentFromText("s", "model"), nil)
	if err != nil {
		t.Fatalf("newSummaryEvent() error = %v", err)
	}
	c := got.Actions.Compaction
	if !c.StartTimestamp.Equal(time.Unix(1, 0)) || !c.EndTimestamp.Equal(time.Unix(9, 0)) {
		t.Errorf("range = [%v, %v], want the true bounds [%v, %v]",
			c.StartTimestamp, c.EndTimestamp, time.Unix(1, 0), time.Unix(9, 0))
	}
	// Every event in the window was summarized, so there are no holes to name.
	if len(c.ExcludedEvents) != 0 {
		t.Errorf("ExcludedEventIDs = %v, want none: the window has no holes", c.ExcludedEvents)
	}
}

// TestNewSummaryEventRejectsThoughtOnlySummary checks that a summary made only
// of reasoning is refused.
//
// The transcript builder skips thought parts of a stored summary, so one would
// render as nothing: the covered turns get deleted and replaced by an empty
// line.
func TestNewSummaryEventRejectsThoughtOnlySummary(t *testing.T) {
	t.Parallel()

	events := []*session.Event{{Timestamp: time.Unix(1, 0)}, {Timestamp: time.Unix(2, 0)}}
	summary := &genai.Content{Role: "model", Parts: []*genai.Part{{Text: "thinking about it", Thought: true}}}

	if _, err := newSummaryEvent(events, events, summary, nil); err == nil {
		t.Error("newSummaryEvent() accepted a thought-only summary")
	}
}

// TestNewSummaryEventDropsThoughtsFromAMixedSummary pins that a thinking
// model's reasoning does not reach the stored summary alongside real prose.
//
// The gate rejected a thought-only summary, but the part filter admitted
// thoughts, so a summary carrying one real sentence and the reasoning behind it
// stored both. A stored summary is replayed into every later prompt as
// something the model said, and its private reasoning is not that.
func TestNewSummaryEventDropsThoughtsFromAMixedSummary(t *testing.T) {
	t.Parallel()

	events := []*session.Event{{Timestamp: time.Unix(1, 0)}, {Timestamp: time.Unix(2, 0)}}
	summary := &genai.Content{Role: "model", Parts: []*genai.Part{
		{Text: "the user asked about the weather", Thought: true},
		{Text: "The user asked about the weather in Zurich."},
	}}

	got, err := newSummaryEvent(events, events, summary, nil)
	if err != nil {
		t.Fatalf("newSummaryEvent() error = %v", err)
	}
	stored := got.Actions.Compaction.CompactedContent.Parts
	if len(stored) != 1 {
		t.Fatalf("stored %d parts, want 1: the thought must be dropped", len(stored))
	}
	if stored[0].Thought {
		t.Error("the stored part is the thought, want the prose")
	}
	if stored[0].Text != "The user asked about the weather in Zurich." {
		t.Errorf("stored text = %q, want the prose part", stored[0].Text)
	}
}

// TestNewSummaryEventRecordsAHoleThatCollidesWithTheWindow pins that window
// membership is decided by identity, not by the reference key.
//
// Two events of one invocation can share a timestamp, which the key cannot tell
// apart, and EventRef's own documentation says so. When one of the pair is in
// the window and the other is not, reading membership from the key said the
// second was summarized as well. No hole was recorded, so the range covered it,
// and a summary that never saw it stood in for it: conversation deleted, which
// is the failure the exclusion list exists to prevent.
//
// Recording the hole costs the over-naming case instead. The reference matches
// both events of the pair, so the one that was summarized is also left raw
// beside a summary of it. That is visible and recoverable where the deletion is
// not.
func TestNewSummaryEventRecordsAHoleThatCollidesWithTheWindow(t *testing.T) {
	t.Parallel()

	inWindow := textEvent("a", "inv1", 1, "summarized")
	collides := textEvent("x", "inv1", 1, "never summarized, same invocation and timestamp")
	tail := modelTextEvent("b", "inv1", 3, "a1")

	window := []*session.Event{inWindow, tail}
	all := []*session.Event{collides, inWindow, tail}

	summary, err := newSummaryEvent(window, all, genai.NewContentFromText("summary", "model"), nil)
	if err != nil {
		t.Fatalf("newSummaryEvent() error = %v", err)
	}
	rec := summary.Actions.Compaction
	if len(rec.ExcludedEvents) != 1 {
		t.Fatalf("ExcludedEvents = %v, want one hole for the event no summary covers", rec.ExcludedEvents)
	}
	want := session.EventRef{InvocationID: "inv1", Timestamp: at(1)}
	if rec.ExcludedEvents[0] != want {
		t.Errorf("ExcludedEvents[0] = %v, want %v", rec.ExcludedEvents[0], want)
	}

	// End to end: the event that was never summarized survives into the prompt.
	summary.ID, summary.Timestamp = "s1", at(4)
	got := ids(Apply(append(all, summary)))
	if !slices.Contains(got, "x") {
		t.Errorf("prompt = %v, want it to still hold %q, which no summary stands in for", got, "x")
	}
}
