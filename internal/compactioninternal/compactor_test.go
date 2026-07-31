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
	"iter"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"

	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/session/compaction"
)

// staticSession is a minimal session.Session over a fixed event list, so the
// compactor can be exercised without a session service.
type staticSession struct {
	events []*session.Event
}

func (s *staticSession) ID() string                    { return "sess" }
func (s *staticSession) AppName() string               { return "app" }
func (s *staticSession) UserID() string                { return "user" }
func (s *staticSession) State() session.State          { return nil }
func (s *staticSession) LastUpdateTime() (t time.Time) { return t }
func (s *staticSession) Events() session.Events        { return &staticEvents{events: s.events} }

var _ session.Session = (*staticSession)(nil)

type staticEvents struct{ events []*session.Event }

func (e *staticEvents) Len() int                { return len(e.events) }
func (e *staticEvents) At(i int) *session.Event { return e.events[i] }
func (e *staticEvents) All() iter.Seq[*session.Event] {
	return func(yield func(*session.Event) bool) {
		for _, ev := range e.events {
			if !yield(ev) {
				return
			}
		}
	}
}

func TestSlidingWindow(t *testing.T) {
	t.Parallel()

	twoInvocations := []*session.Event{
		textEvent("a", "inv1", 1, "q1"), modelTextEvent("b", "inv1", 2, "a1"),
		textEvent("c", "inv2", 3, "q2"), modelTextEvent("d", "inv2", 4, "a2"),
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
			name:       "disabled config does nothing",
			cfg:        &compaction.Config{TokenThreshold: 100, EventRetentionSize: 1},
			events:     twoInvocations,
			summarizer: &fakeSummarizer{summary: "sum"},
		},
		{
			name:       "nil config does nothing",
			cfg:        nil,
			events:     twoInvocations,
			summarizer: &fakeSummarizer{summary: "sum"},
		},
		{
			name:       "interval not reached",
			cfg:        &compaction.Config{CompactionInterval: 3},
			events:     twoInvocations,
			summarizer: &fakeSummarizer{summary: "sum"},
		},
		{
			name:        "interval reached",
			cfg:         &compaction.Config{CompactionInterval: 2},
			events:      twoInvocations,
			summarizer:  &fakeSummarizer{summary: "sum"},
			wantSummary: true,
			wantWindow:  []string{"a", "b", "c", "d"},
		},
		{
			name:        "summarizer declines",
			cfg:         &compaction.Config{CompactionInterval: 2},
			events:      twoInvocations,
			summarizer:  &fakeSummarizer{},
			wantSummary: false,
			wantWindow:  []string{"a", "b", "c", "d"},
		},
		{
			name:       "summarizer fails",
			cfg:        &compaction.Config{CompactionInterval: 2},
			events:     twoInvocations,
			summarizer: &fakeSummarizer{err: errors.New("boom")},
			wantWindow: []string{"a", "b", "c", "d"},
			wantErr:    true,
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

			got, err := SlidingWindow(context.Background(), cfg, &staticSession{events: tc.events})
			if gotErr := err != nil; gotErr != tc.wantErr {
				t.Fatalf("SlidingWindow() error = %v, wantErr %t", err, tc.wantErr)
			}
			if gotSummary := got != nil; gotSummary != tc.wantSummary {
				t.Errorf("SlidingWindow() returned event = %t, want %t", gotSummary, tc.wantSummary)
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

func TestSlidingWindowRequiresSummarizer(t *testing.T) {
	t.Parallel()

	// The runner resolves a default summarizer at construction, so reaching the
	// compactor without one is a programming error worth surfacing loudly
	// rather than silently skipping every compaction.
	_, err := SlidingWindow(context.Background(), &compaction.Config{CompactionInterval: 1}, &staticSession{})
	if err == nil {
		t.Fatal("SlidingWindow() with no Summarizer returned nil error, want an error")
	}
}

func TestSlidingWindowNilSession(t *testing.T) {
	t.Parallel()

	got, err := SlidingWindow(context.Background(), &compaction.Config{CompactionInterval: 1, Summarizer: &fakeSummarizer{}}, nil)
	if err != nil {
		t.Fatalf("SlidingWindow() error = %v", err)
	}
	if got != nil {
		t.Errorf("SlidingWindow() = %v, want nil for a nil session", got)
	}
}

func TestSlidingWindowSucceedingCompactions(t *testing.T) {
	t.Parallel()

	// Walk two consecutive compactions to confirm the overlap pulls exactly one
	// prior invocation into the second window.
	summarizer := &fakeSummarizer{summary: "sum"}
	cfg := &compaction.Config{CompactionInterval: 2, OverlapSize: 1, Summarizer: summarizer}

	events := []*session.Event{
		textEvent("a", "inv1", 1, "q1"), modelTextEvent("b", "inv1", 2, "a1"),
		textEvent("c", "inv2", 3, "q2"), modelTextEvent("d", "inv2", 4, "a2"),
	}

	first, err := SlidingWindow(context.Background(), cfg, &staticSession{events: events})
	if err != nil {
		t.Fatalf("first SlidingWindow() error = %v", err)
	}
	if first == nil {
		t.Fatal("first SlidingWindow() produced no summary")
	}
	first.ID = "s1"
	first.Timestamp = at(5)
	events = append(events, first)

	// One more invocation is not enough.
	events = append(events, textEvent("e", "inv3", 6, "q3"), modelTextEvent("f", "inv3", 7, "a3"))
	mid, err := SlidingWindow(context.Background(), cfg, &staticSession{events: events})
	if err != nil {
		t.Fatalf("second SlidingWindow() error = %v", err)
	}
	if mid != nil {
		t.Errorf("SlidingWindow() compacted after only one new invocation, want nil")
	}

	// The second invocation crosses the interval again.
	events = append(events, textEvent("g", "inv4", 8, "q4"), modelTextEvent("h", "inv4", 9, "a4"))
	third, err := SlidingWindow(context.Background(), cfg, &staticSession{events: events})
	if err != nil {
		t.Fatalf("third SlidingWindow() error = %v", err)
	}
	if third == nil {
		t.Fatal("third SlidingWindow() produced no summary")
	}

	want := [][]string{
		{"a", "b", "c", "d"},
		{"c", "d", "e", "f", "g", "h"},
	}
	if diff := cmp.Diff(want, summarizer.windows); diff != "" {
		t.Errorf("summarizer windows mismatch (-want +got):\n%s", diff)
	}
}
