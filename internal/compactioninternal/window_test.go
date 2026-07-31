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

func TestIsCompactionEvent(t *testing.T) {
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
			if got := compaction.IsCompactionEvent(tc.event); got != tc.want {
				t.Errorf("compaction.IsCompactionEvent() = %t, want %t", got, tc.want)
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
