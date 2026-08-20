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
	"fmt"
	"slices"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/session/compaction"
	"google.golang.org/adk/v2/tool/toolconfirmation"
)

func TestLongestSelfContainedPrefix(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name   string
		events []*session.Event
		want   []string // event IDs of the returned prefix
	}{
		{
			name:   "empty",
			events: nil,
			want:   nil,
		},
		{
			name:   "plain text events are all self contained",
			events: []*session.Event{textEvent("a", "inv1", 1, "hi"), textEvent("b", "inv1", 2, "hello")},
			want:   []string{"a", "b"},
		},
		{
			name: "call and response in range",
			events: []*session.Event{
				textEvent("a", "inv1", 1, "hi"),
				callEvent("b", "inv1", 2, "c1"),
				responseEvent("c", "inv1", 3, "c1"),
			},
			want: []string{"a", "b", "c"},
		},
		{
			name: "dangling call truncates the prefix",
			events: []*session.Event{
				textEvent("a", "inv1", 1, "hi"),
				callEvent("b", "inv1", 2, "c1"),
			},
			want: []string{"a"},
		},
		{
			name: "trailing events after a dangling call are also dropped",
			events: []*session.Event{
				textEvent("a", "inv1", 1, "hi"),
				callEvent("b", "inv1", 2, "c1"),
				textEvent("c", "inv1", 3, "still thinking"),
			},
			want: []string{"a"},
		},
		{
			name: "parallel calls need every response",
			events: []*session.Event{
				multiCallEvent("a", "inv1", 1, "c1", "c2"),
				responseEvent("b", "inv1", 2, "c1"),
				responseEvent("c", "inv1", 3, "c2"),
			},
			want: []string{"a", "b", "c"},
		},
		{
			name: "parallel calls missing one response",
			events: []*session.Event{
				textEvent("z", "inv1", 1, "hi"),
				multiCallEvent("a", "inv1", 2, "c1", "c2"),
				responseEvent("b", "inv1", 3, "c1"),
			},
			want: []string{"z"},
		},
		{
			name: "unresolved tool confirmation blocks the prefix",
			events: []*session.Event{
				textEvent("a", "inv1", 1, "hi"),
				confirmationEvent("b", "inv1", 2, "c1"),
			},
			want: []string{"a"},
		},
		{
			name: "resolved tool confirmation is fine",
			events: []*session.Event{
				textEvent("a", "inv1", 1, "hi"),
				confirmationEvent("b", "inv1", 2, "c1"),
				responseEvent("c", "inv1", 3, "c1"),
			},
			want: []string{"a", "b", "c"},
		},
		{
			name: "response within the same event as its call still opens the obligation",
			events: []*session.Event{
				callAndResponseEvent("a", "inv1", 1, "c1"),
			},
			// Responses are applied before calls within an event, so the call
			// in this same event is still open at the end of it.
			want: nil,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got := ids(longestSelfContainedPrefix(tc.events))
			if diff := cmp.Diff(tc.want, got); diff != "" {
				t.Errorf("longestSelfContainedPrefix() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestSelectSlidingWindow(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		events   []*session.Event
		interval int
		overlap  int
		want     []string
	}{
		{
			name:     "interval not reached",
			events:   []*session.Event{textEvent("a", "inv1", 1, "hi"), textEvent("b", "inv1", 2, "hello")},
			interval: 2,
			want:     nil,
		},
		{
			name: "first compaction covers both invocations",
			events: []*session.Event{
				textEvent("a", "inv1", 1, "q1"), textEvent("b", "inv1", 2, "a1"),
				textEvent("c", "inv2", 3, "q2"), textEvent("d", "inv2", 4, "a2"),
			},
			interval: 2,
			want:     []string{"a", "b", "c", "d"},
		},
		{
			name: "interval zero disables selection",
			events: []*session.Event{
				textEvent("a", "inv1", 1, "q1"), textEvent("b", "inv2", 2, "q2"),
			},
			interval: 0,
			want:     nil,
		},
		{
			name: "only one new invocation since the last compaction",
			events: []*session.Event{
				textEvent("a", "inv1", 1, "q1"), textEvent("b", "inv1", 2, "a1"),
				textEvent("c", "inv2", 3, "q2"), textEvent("d", "inv2", 4, "a2"),
				compactionEvent("s1", 5, 1, 4, "summary of 1-2"),
				textEvent("e", "inv3", 6, "q3"), textEvent("f", "inv3", 7, "a3"),
			},
			interval: 2,
			overlap:  1,
			want:     nil,
		},
		{
			name: "second compaction pulls one invocation back via overlap",
			events: []*session.Event{
				textEvent("a", "inv1", 1, "q1"), textEvent("b", "inv1", 2, "a1"),
				textEvent("c", "inv2", 3, "q2"), textEvent("d", "inv2", 4, "a2"),
				compactionEvent("s1", 5, 1, 4, "summary of 1-2"),
				textEvent("e", "inv3", 6, "q3"), textEvent("f", "inv3", 7, "a3"),
				textEvent("g", "inv4", 8, "q4"), textEvent("h", "inv4", 9, "a4"),
			},
			interval: 2,
			overlap:  1,
			want:     []string{"c", "d", "e", "f", "g", "h"},
		},
		{
			name: "zero overlap starts after the previous compaction",
			events: []*session.Event{
				textEvent("a", "inv1", 1, "q1"), textEvent("b", "inv1", 2, "a1"),
				textEvent("c", "inv2", 3, "q2"), textEvent("d", "inv2", 4, "a2"),
				compactionEvent("s1", 5, 1, 4, "summary of 1-2"),
				textEvent("e", "inv3", 6, "q3"), textEvent("f", "inv3", 7, "a3"),
				textEvent("g", "inv4", 8, "q4"), textEvent("h", "inv4", 9, "a4"),
			},
			interval: 2,
			overlap:  0,
			want:     []string{"e", "f", "g", "h"},
		},
		{
			name: "window is trimmed so an open call is never summarized alone",
			events: []*session.Event{
				textEvent("a", "inv1", 1, "q1"), textEvent("b", "inv1", 2, "a1"),
				textEvent("c", "inv2", 3, "q2"), callEvent("d", "inv2", 4, "c1"),
			},
			interval: 2,
			want:     []string{"a", "b", "c"},
		},
		{
			name: "nil when the whole window is one open call",
			events: []*session.Event{
				callEvent("a", "inv1", 1, "c1"),
				callEvent("b", "inv2", 2, "c2"),
			},
			interval: 2,
			want:     nil,
		},
		{
			name: "events without an invocation ID are ignored",
			events: []*session.Event{
				textEvent("a", "", 1, "orphan"),
				textEvent("b", "inv1", 2, "q1"),
				textEvent("c", "inv2", 3, "q2"),
			},
			interval: 2,
			want:     []string{"b", "c"},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got := ids(selectSlidingWindow(tc.events, tc.interval, tc.overlap))
			if diff := cmp.Diff(tc.want, got); diff != "" {
				t.Errorf("selectSlidingWindow(interval=%d, overlap=%d) mismatch (-want +got):\n%s", tc.interval, tc.overlap, diff)
			}
		})
	}
}

func TestLatestCompactionEvent(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name   string
		events []*session.Event
		want   string // event ID, "" for nil
	}{
		{
			name:   "no compactions",
			events: []*session.Event{textEvent("a", "inv1", 1, "hi")},
			want:   "",
		},
		{
			name:   "single compaction",
			events: []*session.Event{compactionEvent("s1", 5, 1, 4, "sum")},
			want:   "s1",
		},
		{
			name: "wider compaction wins over the narrower one it contains",
			events: []*session.Event{
				compactionEvent("s1", 5, 1, 4, "narrow"),
				compactionEvent("s2", 9, 1, 8, "wide"),
			},
			want: "s2",
		},
		{
			name: "a later compaction does not win when an earlier one is wider",
			events: []*session.Event{
				compactionEvent("s1", 9, 1, 8, "wide"),
				compactionEvent("s2", 10, 3, 6, "narrow"),
			},
			want: "s1",
		},
		{
			name: "identical ranges keep the later event",
			events: []*session.Event{
				compactionEvent("s1", 5, 1, 4, "first"),
				compactionEvent("s2", 6, 1, 4, "second"),
			},
			want: "s2",
		},
		{
			name: "partially overlapping compactions both survive, latest wins",
			events: []*session.Event{
				compactionEvent("s1", 5, 1, 4, "left"),
				compactionEvent("s2", 9, 3, 8, "right"),
			},
			want: "s2",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got := LatestCompactionEvent(tc.events)
			gotID := ""
			if got != nil {
				gotID = got.ID
			}
			if gotID != tc.want {
				t.Errorf("LatestCompactionEvent() = %q, want %q", gotID, tc.want)
			}
		})
	}
}

func TestConfigValidate(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		cfg     *compaction.Config
		wantErr bool
	}{
		{name: "nil is valid", cfg: nil},
		// nil means "disabled"; an allocated-but-empty config means the
		// caller intended something and configured nothing.
		{name: "empty but non-nil is a mistake", cfg: &compaction.Config{}, wantErr: true},
		{name: "sliding window", cfg: &compaction.Config{CompactionInterval: 3, OverlapSize: 1}},
		{name: "sliding window with zero overlap", cfg: &compaction.Config{CompactionInterval: 3}},
		{name: "tail retention", cfg: &compaction.Config{TokenThreshold: 1000, EventRetentionSize: 5}},
		{name: "both strategies", cfg: &compaction.Config{CompactionInterval: 3, OverlapSize: 1, TokenThreshold: 1000, EventRetentionSize: 5}},
		{name: "negative interval", cfg: &compaction.Config{CompactionInterval: -1}, wantErr: true},
		{name: "negative overlap", cfg: &compaction.Config{CompactionInterval: 1, OverlapSize: -1}, wantErr: true},
		{name: "negative token threshold", cfg: &compaction.Config{TokenThreshold: -1}, wantErr: true},
		{name: "negative retention size", cfg: &compaction.Config{TokenThreshold: 1, EventRetentionSize: -1}, wantErr: true},
		{name: "overlap without interval", cfg: &compaction.Config{OverlapSize: 2, TokenThreshold: 10}, wantErr: true},
		{name: "retention without threshold", cfg: &compaction.Config{EventRetentionSize: 2, CompactionInterval: 1}, wantErr: true},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			err := tc.cfg.Validate()
			if gotErr := err != nil; gotErr != tc.wantErr {
				t.Errorf("Validate() error = %v, wantErr %t", err, tc.wantErr)
			}
		})
	}
}

func TestHasSlidingWindow(t *testing.T) {
	t.Parallel()

	var nilCfg *compaction.Config
	if HasSlidingWindow(nilCfg) {
		t.Error("a nil Config must report sliding window disabled")
	}
	if !HasSlidingWindow(&compaction.Config{CompactionInterval: 2}) {
		t.Error("HasSlidingWindow() = false, want true when CompactionInterval > 0")
	}
	if HasSlidingWindow(&compaction.Config{TokenThreshold: 10}) {
		t.Error("HasSlidingWindow() = true, want false when CompactionInterval is 0")
	}
}

func TestHasTailRetention(t *testing.T) {
	t.Parallel()

	var nilCfg *compaction.Config
	if HasTailRetention(nilCfg) {
		t.Error("a nil Config must report tail retention disabled")
	}
	if !HasTailRetention(&compaction.Config{TokenThreshold: 10}) {
		t.Error("HasTailRetention() = false, want true when TokenThreshold > 0")
	}
	if HasTailRetention(&compaction.Config{CompactionInterval: 2}) {
		t.Error("HasTailRetention() = true, want false when TokenThreshold is 0")
	}
}

func TestHasUsableSummary(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name  string
		event *session.Event
		want  bool
	}{
		{name: "nil", event: nil, want: false},
		{name: "plain event", event: textEvent("a", "inv1", 1, "hi"), want: false},
		{name: "compaction", event: compactionEvent("s1", 5, 1, 4, "sum"), want: true},
		{
			name: "compaction with no content is not usable",
			event: &session.Event{
				ID:      "s1",
				Actions: session.EventActions{Compaction: &session.EventCompaction{StartTimestamp: at(1), EndTimestamp: at(4)}},
			},
			want: false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			if got := HasUsableSummary(tc.event); got != tc.want {
				t.Errorf("HasUsableSummary() = %t, want %t", got, tc.want)
			}
		})
	}
}

func TestConfirmationEventOpensObligation(t *testing.T) {
	t.Parallel()

	// Guard against the helper silently producing an event with no
	// confirmation, which would make TestLongestSelfContainedPrefix vacuous.
	ev := confirmationEvent("b", "inv1", 2, "c1")
	if _, ok := ev.Actions.RequestedToolConfirmations["c1"]; !ok {
		t.Fatalf("confirmationEvent() produced no RequestedToolConfirmations entry, got %v", ev.Actions.RequestedToolConfirmations)
	}
	if _, ok := any(ev.Actions.RequestedToolConfirmations["c1"]).(toolconfirmation.ToolConfirmation); !ok {
		t.Fatal("RequestedToolConfirmations entry has an unexpected type")
	}
}

// assertWindowCoversItsRange checks the invariant the interval model depends
// on: the set of events a summary covers must equal the set it summarized.
//
// Coverage is recorded as an inclusive timestamp range and the prompt builder
// drops everything inside it, so any event that falls in the range but is
// missing from the window would be dropped without ever being summarized.
func assertWindowCoversItsRange(t *testing.T, all, window []*session.Event) {
	t.Helper()
	if len(window) == 0 {
		return
	}
	start, end := window[0].Timestamp, window[len(window)-1].Timestamp
	inWindow := make(map[*session.Event]bool, len(window))
	for _, ev := range window {
		inWindow[ev] = true
	}
	for _, ev := range all {
		if hasCompaction(ev) || inWindow[ev] {
			continue
		}
		if !ev.Timestamp.Before(start) && !ev.Timestamp.After(end) {
			t.Errorf("event %q at %v lies inside the summarized range [%v, %v] but was not summarized, so it would vanish from the prompt",
				ev.ID, ev.Timestamp, start, end)
		}
	}
}

func TestSelectSlidingWindowCoversEverythingInItsRange(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name   string
		events []*session.Event
	}{
		{
			name: "event with no invocation ID sits between two invocations",
			events: []*session.Event{
				textEvent("a", "inv1", 1, "q1"), modelTextEvent("b", "inv1", 2, "a1"),
				// Appended directly to the session rather than by an
				// invocation, so it carries no invocation ID.
				textEvent("orphan", "", 3, "side note"),
				textEvent("c", "inv2", 4, "q2"), modelTextEvent("d", "inv2", 5, "a2"),
			},
		},
		{
			name: "several ID-less events interleaved",
			events: []*session.Event{
				textEvent("x", "", 1, "before"),
				textEvent("a", "inv1", 2, "q1"),
				textEvent("y", "", 3, "middle"),
				modelTextEvent("b", "inv1", 4, "a1"),
				textEvent("c", "inv2", 5, "q2"),
				textEvent("z", "", 6, "later"),
				modelTextEvent("d", "inv2", 7, "a2"),
			},
		},
		{
			name: "trim boundary lands on a timestamp tie",
			events: []*session.Event{
				textEvent("a", "inv1", 1, "q1"),
				modelTextEvent("b", "inv1", 2, "a1"),
				textEvent("c", "inv2", 3, "q2"),
				// These three share a timestamp, and the open call forces a
				// trim right in the middle of the group.
				modelTextEvent("d", "inv2", 4, "a2"),
				callEvent("e", "inv2", 4, "c1"),
				modelTextEvent("f", "inv2", 4, "trailing"),
			},
		},
		{
			name: "overlap reaches back across an ID-less event",
			events: []*session.Event{
				textEvent("a", "inv1", 1, "q1"),
				textEvent("orphan", "", 2, "side note"),
				textEvent("b", "inv2", 3, "q2"),
				compactionEvent("s1", 4, 1, 3, "earlier summary"),
				textEvent("c", "inv3", 5, "q3"),
				textEvent("d", "inv4", 6, "q4"),
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			for _, overlap := range []int{0, 1, 2} {
				window := selectSlidingWindow(tc.events, 2, overlap)
				assertWindowCoversItsRange(t, tc.events, window)
			}
		})
	}
}

// TestSelectSlidingWindowIncludesIDlessEvents pins the specific behaviour the
// invariant depends on, so a future refactor that starts filtering again fails
// loudly rather than silently dropping events.
func TestSelectSlidingWindowIncludesIDlessEvents(t *testing.T) {
	t.Parallel()

	events := []*session.Event{
		textEvent("a", "inv1", 1, "q1"), modelTextEvent("b", "inv1", 2, "a1"),
		textEvent("orphan", "", 3, "side note"),
		textEvent("c", "inv2", 4, "q2"), modelTextEvent("d", "inv2", 5, "a2"),
	}

	got := ids(selectSlidingWindow(events, 2, 0))
	if diff := cmp.Diff([]string{"a", "b", "orphan", "c", "d"}, got); diff != "" {
		t.Errorf("selectSlidingWindow() mismatch (-want +got):\n%s", diff)
	}
}

// TestSelectSlidingWindowSurvivesBlockedHead pins that a tool call which never
// gets a response does not stop compaction for the rest of the session.
//
// The window is anchored to the last compaction boundary, so an unanswered call
// at the head stays at the head forever. Returning nil there would silently
// disable compaction on exactly the long tool-using sessions that need it.
func TestSelectSlidingWindowSurvivesBlockedHead(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		// interval is chosen per case so the window cap covers every
		// invocation the case sets up. The subject here is the blocked head,
		// not the cap.
		interval int
		events   []*session.Event
		want     []string
	}{
		{
			name:     "unanswered call at the head is stepped over",
			interval: 3,
			events: []*session.Event{
				// inv1 asks a tool something that never answers.
				callEvent("stuck", "inv1", 1, "c1"),
				textEvent("a", "inv2", 2, "q2"), modelTextEvent("b", "inv2", 3, "a2"),
				textEvent("c", "inv3", 4, "q3"), modelTextEvent("d", "inv3", 5, "a3"),
			},
			want: []string{"a", "b", "c", "d"},
		},
		{
			name:     "unanswered confirmation at the head is stepped over",
			interval: 3,
			events: []*session.Event{
				confirmationEvent("stuck", "inv1", 1, "c1"),
				textEvent("a", "inv2", 2, "q2"),
				textEvent("b", "inv3", 3, "q3"),
			},
			want: []string{"a", "b"},
		},
		{
			name: "still nil when nothing after the blockage is self-contained",
			events: []*session.Event{
				callEvent("stuck1", "inv1", 1, "c1"),
				callEvent("stuck2", "inv2", 2, "c2"),
			},
			want: nil,
		},
		{
			name: "a resolvable call is trimmed normally, not stepped over",
			events: []*session.Event{
				textEvent("a", "inv1", 1, "q1"),
				modelTextEvent("b", "inv1", 2, "a1"),
				textEvent("c", "inv2", 3, "q2"),
				callEvent("pending", "inv2", 4, "c1"),
			},
			want: []string{"a", "b", "c"},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			interval := tc.interval
			if interval == 0 {
				interval = 2
			}
			window := selectSlidingWindow(tc.events, interval, 0)
			if diff := cmp.Diff(tc.want, ids(window)); diff != "" {
				t.Errorf("selectSlidingWindow() mismatch (-want +got):\n%s", diff)
			}
			// Stepping past a blockage must not break the coverage invariant.
			assertWindowCoversItsRange(t, tc.events, window)
		})
	}
}

// TestLongestSelfContainedPrefixIDlessCall pins that a call with no ID is
// treated as an obligation. Pairing is keyed on the ID, which is optional, so
// keying an ID-less call on "" would let the trim that protects every other
// call silently not fire and split it from its response.
func TestLongestSelfContainedPrefixIDlessCall(t *testing.T) {
	t.Parallel()

	events := []*session.Event{
		textEvent("a", "inv1", 1, "q1"),
		newEvent("idless", "inv1", 2, "model", &genai.Part{
			FunctionCall: &genai.FunctionCall{Name: "tool_without_id"},
		}),
		responseEvent("resp", "inv1", 3, ""),
		modelTextEvent("d", "inv1", 4, "done"),
	}

	// The call must block the prefix rather than sail through it.
	if diff := cmp.Diff([]string{"a"}, ids(longestSelfContainedPrefix(events))); diff != "" {
		t.Errorf("longestSelfContainedPrefix() mismatch (-want +got):\n%s", diff)
	}
}

// TestSelectSlidingWindowIsBoundedByInterval pins that the window covers at
// most interval new invocations rather than running to the end of the session.
//
// Without the cap the window is O(session): enabling compaction on an existing
// deployment would hand a whole live conversation to a single model call.
func TestSelectSlidingWindowIsBoundedByInterval(t *testing.T) {
	t.Parallel()

	// Ten invocations of one turn each, no prior compaction: the entire
	// backlog is new.
	var events []*session.Event
	for i := range 10 {
		events = append(events, textEvent(fmt.Sprintf("q%d", i), fmt.Sprintf("inv%d", i), i+1, "q"))
	}

	window := selectSlidingWindow(events, 3, 0)
	if diff := cmp.Diff([]string{"q0", "q1", "q2"}, ids(window)); diff != "" {
		t.Errorf("selectSlidingWindow() mismatch (-want +got):\n%s\nthe window must not run to the end of the session", diff)
	}
}

// TestSelectSlidingWindowRetryDoesNotGrow pins that a failed attempt comes back
// to a window of the same size rather than a larger one.
//
// A summarizer error records no compaction, so the next turn recomputes from
// the same start. If the window grew with the session, a transient failure
// would leave a window that is more likely to fail again, and the session would
// never recover.
func TestSelectSlidingWindowRetryDoesNotGrow(t *testing.T) {
	t.Parallel()

	var events []*session.Event
	for i := range 3 {
		events = append(events, textEvent(fmt.Sprintf("q%d", i), fmt.Sprintf("inv%d", i), i+1, "q"))
	}
	first := selectSlidingWindow(events, 2, 0)

	// The attempt failed, so nothing was recorded. Two more turns arrive.
	for i := 3; i < 5; i++ {
		events = append(events, textEvent(fmt.Sprintf("q%d", i), fmt.Sprintf("inv%d", i), i+1, "q"))
	}
	retry := selectSlidingWindow(events, 2, 0)

	if len(retry) != len(first) {
		t.Errorf("retry window has %d events, want the same %d as the attempt that failed: %v then %v",
			len(retry), len(first), ids(first), ids(retry))
	}
	if diff := cmp.Diff(ids(first), ids(retry)); diff != "" {
		t.Errorf("retry window mismatch (-first +retry):\n%s", diff)
	}
}

// TestSlidingWindowMakesProgressAcrossABranchChange pins that a branch change
// inside an invocation does not stall compaction.
//
// The window is trimmed to one branch and one isolation scope, so when the
// branch changes inside an invocation the cut stops short of that invocation's
// last event. Progress was then measured against the newest compaction's end
// timestamp, and the slice was taken from the invocation's first event however
// much of it was already summarized, so every later turn recomputed a
// byte-identical window and paid for a model call that changed nothing.
// Forking a child branch inside one invocation is the ordinary multi-agent
// shape, so this was not an edge case.
func TestSlidingWindowMakesProgressAcrossABranchChange(t *testing.T) {
	t.Parallel()

	branched := func(id, inv string, ts int, branch, text string) *session.Event {
		ev := textEvent(id, inv, ts, text)
		ev.Branch = branch
		return ev
	}
	all := []*session.Event{
		textEvent("a", "inv1", 1, "q1"),
		modelTextEvent("b", "inv1", 2, "a1"),
		// The branch forks partway through inv2, so the cut lands inside it.
		textEvent("c", "inv2", 3, "q2"),
		branched("d", "inv2", 4, "child", "sub-agent work"),
		branched("e", "inv2", 5, "child", "more sub-agent work"),
	}

	var chosen [][]string
	for pass := 1; pass <= 4; pass++ {
		w := selectSlidingWindow(all, 1, 0)
		if len(w) == 0 {
			break
		}
		chosen = append(chosen, ids(w))

		summary, err := newSummaryEvent(w, w, genai.NewContentFromText("summary", "model"), nil)
		if err != nil {
			t.Fatalf("pass %d: newSummaryEvent() error = %v", pass, err)
		}
		summary.ID = fmt.Sprintf("s%d", pass)
		summary.InvocationID = fmt.Sprintf("e-compaction-%d", pass)
		summary.Timestamp = at(10 + pass)
		all = append(all, summary)
	}

	// Every pass moves on, and the session runs out of things to summarize
	// rather than re-offering the same slice for ever.
	want := [][]string{{"a", "b"}, {"c"}, {"d", "e"}}
	if diff := cmp.Diff(want, chosen); diff != "" {
		t.Errorf("windows chosen across passes mismatch (-want +got):\n%s", diff)
	}
}

// TestSlidingWindowRecoversFromABlockedInvocation pins that the window keeps
// advancing when an invocation stays partly uncompacted.
//
// endID resolved from firstNew, and firstNew does not move while nothing is
// compacted, so once endID's invocation was fully covered the slice bounds
// inverted and selection returned nil on every later turn. Silently, since an
// empty window and "nothing to do yet" are the same answer.
//
// The trigger is ordinary: a paused run reuses its invocation ID, so a pending
// tool confirmation produces exactly this shape, and it did not recover when
// the tool finally answered.
func TestSlidingWindowRecoversFromABlockedInvocation(t *testing.T) {
	t.Parallel()

	all := []*session.Event{
		// inv1 opens a call nothing has answered yet.
		callEvent("blocked", "inv1", 1, "c-pending"),
		textEvent("q2", "inv2", 2, "q2"),
		modelTextEvent("a2", "inv2", 3, "a2"),
		textEvent("q3", "inv3", 4, "q3"),
		modelTextEvent("a3", "inv3", 5, "a3"),
	}

	var chosen [][]string
	for pass := 1; pass <= 4; pass++ {
		w := selectSlidingWindow(all, 2, 0)
		if len(w) == 0 {
			break
		}
		chosen = append(chosen, ids(w))
		summary, err := newSummaryEvent(w, all, genai.NewContentFromText("summary", "model"), nil)
		if err != nil {
			t.Fatalf("pass %d: newSummaryEvent() error = %v", pass, err)
		}
		summary.ID = fmt.Sprintf("s%d", pass)
		summary.InvocationID = fmt.Sprintf("e-compaction-%d", pass)
		summary.Timestamp = at(10 + pass)
		all = append(all, summary)
	}

	if len(chosen) == 0 {
		t.Fatal("selectSlidingWindow() never chose a window, so compaction is stalled")
	}
	// The pending call stays raw and visible, which is what a pending call
	// needs, and everything behind it is summarized rather than accumulating.
	for _, w := range chosen {
		if slices.Contains(w, "blocked") {
			t.Errorf("window %v covers the pending call", w)
		}
	}
	var covered []string
	for _, w := range chosen {
		covered = append(covered, w...)
	}
	for _, want := range []string{"q2", "a2", "q3", "a3"} {
		if !slices.Contains(covered, want) {
			t.Errorf("event %q was never summarized across %v", want, chosen)
		}
	}
}

// TestCoversAllOfAbsorbsAGenuineSuperset pins that a strictly wider record
// absorbs a narrower one even when it names a hole the narrower one never
// spanned.
//
// A hole outside b's range is not a disagreement. b never covered that event,
// so it had nothing to say about it, and requiring it to say so made the test
// impossible to satisfy exactly when absorption is most obviously right. The
// consequence was not a missed optimisation: neither record was ever discarded,
// so Apply materialized both summaries and the prompt grew instead of shrinking.
func TestCoversAllOfAbsorbsAGenuineSuperset(t *testing.T) {
	t.Parallel()

	a := &session.EventCompaction{
		StartTimestamp: at(10), EndTimestamp: at(50),
		ExcludedEvents: []session.EventRef{{InvocationID: "inv-x", Timestamp: at(15)}},
	}
	b := &session.EventCompaction{StartTimestamp: at(20), EndTimestamp: at(40)}

	if !coversAllOf(a, b) {
		t.Error("coversAllOf() = false, want true: a spans b, and its hole is outside b entirely")
	}
	// A hole inside b's range that b does not name is still a real
	// disagreement, and must still block absorption.
	c := &session.EventCompaction{
		StartTimestamp: at(10), EndTimestamp: at(50),
		ExcludedEvents: []session.EventRef{{InvocationID: "inv-y", Timestamp: at(30)}},
	}
	if coversAllOf(c, b) {
		t.Error("coversAllOf() = true, want false: the hole is inside b and b does not name it")
	}
}

// TestCoversAllOfComparesInstantsNotClockReadings pins that a hole is matched
// the way excludes matches one.
//
// slices.Contains compares EventRef with ==, and == on the time.Time inside it
// compares the wall clock, the monotonic reading and the *Location pointer. Two
// references naming one instant stop matching once either has been through a
// store, because a round trip drops the monotonic reading and a backend may
// return a different timezone. Neither is wrong, and treating them as different
// left both summaries in the prompt.
func TestCoversAllOfComparesInstantsNotClockReadings(t *testing.T) {
	t.Parallel()

	utc := time.Date(2026, 1, 1, 12, 0, 0, 0, time.UTC)
	loc, err := time.LoadLocation("Europe/Warsaw")
	if err != nil {
		t.Skipf("no tzdata available: %v", err)
	}
	same := utc.In(loc)

	a := &session.EventCompaction{
		StartTimestamp: utc.Add(-time.Hour), EndTimestamp: utc.Add(time.Hour),
		ExcludedEvents: []session.EventRef{{InvocationID: "inv", Timestamp: utc}},
	}
	b := &session.EventCompaction{
		StartTimestamp: utc.Add(-time.Minute), EndTimestamp: utc.Add(time.Minute),
		ExcludedEvents: []session.EventRef{{InvocationID: "inv", Timestamp: same}},
	}
	if !coversAllOf(a, b) {
		t.Error("coversAllOf() = false, want true: the two references name the same instant")
	}
}

// TestApplyKeepsOneSummaryWhenARecordIsSuperseded is the symptom the two tests
// above are the cause of, asserted where a user would feel it.
//
// A superseded record that is not discarded is materialized alongside the one
// that replaced it, so the model is shown the same turns described twice and
// the prompt is larger after compaction than before.
func TestApplyKeepsOneSummaryWhenARecordIsSuperseded(t *testing.T) {
	t.Parallel()

	events := []*session.Event{
		textEvent("a", "inv1", 1, "q1"),
		modelTextEvent("b", "inv1", 2, "a1"),
		compactionEvent("s1", 3, 1, 2, "older summary"),
		textEvent("c", "inv2", 4, "q2"),
		modelTextEvent("d", "inv2", 5, "a2"),
		// Spans everything s1 covered and more, and names a hole of its own
		// that lies outside s1's range, which is what the live-head hold-back
		// produces on every tail-retention pass.
		compactionEvent("s2", 6, 1, 5, "newer summary", excl("inv2", 4)),
	}

	got := ids(Apply(events))
	for _, id := range got {
		if id == "s1" {
			t.Errorf("prompt = %v, want the superseded summary gone: both summaries describe the same turns", got)
		}
	}
}
