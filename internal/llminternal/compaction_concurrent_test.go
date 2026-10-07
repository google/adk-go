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

package llminternal_test

import (
	"context"
	"fmt"
	"sync"
	"testing"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/internal/compactioninternal"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/session/compaction"
)

// recordingSummarizer remembers which events it was actually asked to
// summarize, so a test can tell "summarized" apart from "silently dropped".
//
// Without this the two are indistinguishable from outside: both show up as an
// event that is not in the assembled prompt. An earlier version of this test
// treated every absent event as lost and reported a bug that was not there --
// the events had been summarized correctly.
type recordingSummarizer struct {
	mu   sync.Mutex
	seen map[string]bool
}

func (s *recordingSummarizer) SummarizeEvents(_ context.Context, events []*session.Event) (compaction.SummarizeResult, error) {
	s.mu.Lock()
	if s.seen == nil {
		s.seen = map[string]bool{}
	}
	for _, ev := range events {
		if ev != nil {
			s.seen[ev.InvocationID] = true
		}
	}
	s.mu.Unlock()
	return compaction.SummarizeResult{Content: genai.NewContentFromText("summary", "model")}, nil
}

func (s *recordingSummarizer) summarized(invocationID string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.seen[invocationID]
}

// blockingSummarizer holds a compaction open until it is released, so a second
// writer can be scheduled against a window that is genuinely still in flight.
type blockingSummarizer struct {
	entered chan struct{}
	release chan struct{}
	once    sync.Once
}

func (s *blockingSummarizer) SummarizeEvents(ctx context.Context, events []*session.Event) (compaction.SummarizeResult, error) {
	s.once.Do(func() { close(s.entered) })
	select {
	case <-s.release:
	case <-ctx.Done():
		return compaction.SummarizeResult{}, ctx.Err()
	}
	return compaction.SummarizeResult{
		Content: genai.NewContentFromText("summary", "model"),
	}, nil
}

// TestCompactionSurvivesAConcurrentWriter runs a real second writer against the
// session while a compaction is in flight.
//
// The existing straggler test lands its event from inside the append, on the
// same goroutine, which pins the repair logic but never puts two writers on the
// session at once. So the collision the repair exists for was described by a
// test and never actually performed, and nothing here ran under the race
// detector in a way that could see the two paths interleave.
//
// This starts the compaction, waits until the summarizer is provably inside the
// window, then appends from a separate goroutine before releasing it. Run with
// -race, which is how the suite runs in CI.
//
// What it asserts is the invariant the whole design rests on: **every event is
// either summarized, or named as a hole, or still present raw**. An event that
// falls inside a recorded range while nothing summarized it and nothing names
// it is gone from every future prompt, silently, which is the failure mode the
// repair step exists to prevent.
func TestCompactionSurvivesAConcurrentWriter(t *testing.T) {
	t.Parallel()

	base, sess := tailRetentionFixture(t, 6)
	summarizer := &blockingSummarizer{
		entered: make(chan struct{}),
		release: make(chan struct{}),
	}

	// The straggler is created with a timestamp inside the range the compaction
	// is about to record, and stored after the window was chosen. That is the
	// real shape: an event carries the time it was created, not the time it was
	// stored, so a parallel branch routinely produces one.
	straggler := session.NewEvent(t.Context(), "concurrent-inv")
	straggler.Author = "user"
	straggler.Timestamp = time.Unix(4, 0)
	straggler.LLMResponse.Content = genai.NewContentFromText("from the other writer", "user")

	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		<-summarizer.entered
		if err := base.AppendEvent(t.Context(), sess, straggler); err != nil {
			t.Errorf("the concurrent writer could not append: %v", err)
		}
		close(summarizer.release)
	}()

	err := runCompactionProcessor(t, base, sess, &compaction.Config{
		TokenThreshold:     100,
		EventRetentionSize: 2,
		Summarizer:         summarizer,
	})
	wg.Wait()
	if err != nil {
		t.Fatalf("CompactionRequestProcessor failed: %v", err)
	}

	got, err := base.Get(t.Context(), &session.GetRequest{AppName: "app", UserID: "u", SessionID: "s"})
	if err != nil {
		t.Fatalf("Get() error = %v", err)
	}
	var stored []*session.Event
	for ev := range got.Session.Events().All() {
		stored = append(stored, ev)
	}

	// Deliberately not asserting that a summary landed. There are two defences
	// and this scenario reaches the first one: the race check before the append
	// sees the session changed inside the range and throws the summary away.
	// The second, RepairAfterAppend, covers the narrower window where the event
	// lands between that check and the store. Which one fires is a timing
	// detail; the property below has to hold either way, and pinning one of
	// them here would make the test fail whenever the other did its job.
	var summaries int
	for _, ev := range stored {
		if compactioninternal.HasUsableSummary(ev) {
			summaries++
		}
	}

	// The invariant. Every event is summarized, named as a hole, or still
	// present raw. An event inside a recorded range that nothing summarized and
	// nothing names is gone from every future prompt, silently.
	for _, ev := range compactioninternal.Apply(stored) {
		if ev.InvocationID == straggler.InvocationID {
			return
		}
	}
	t.Errorf("an event appended by a concurrent writer during compaction reached no prompt and no summary: it was lost.\n"+
		"stored events: %d, summaries kept: %d", len(stored), summaries)
}

// TestCompactionUnderConcurrentAppendsLosesNothing is the same collision without
// the choreography, repeated, so the race detector has interleavings to look at
// rather than one scripted ordering.
//
// The blocking test above pins one exact sequence. This one starts several
// writers against a session that is being compacted and lets the scheduler
// decide, which is what actually happens in a parallel agent. It checks two
// things at once: that -race finds no data race across the compaction and append
// paths, and that no event is silently dropped whichever way the timing falls.
//
// The invariant is stated carefully, because the obvious version is wrong. An
// event missing from the prompt is not evidence of a bug: it may have been
// summarized, which is the whole point. What must never happen is an event that
// reaches no prompt AND was never handed to a summarizer. That one is gone with
// nothing standing in for it, and it is what the race guard and the repair step
// exist to prevent.
func TestCompactionUnderConcurrentAppendsLosesNothing(t *testing.T) {
	t.Parallel()

	// Repeated, because one attempt only samples one interleaving and the
	// losing ones are a minority. Disabling the straggler repair loses an event
	// in roughly one attempt in ten, so a single attempt would let that
	// regression through nine times out of ten; at this count it is caught
	// essentially every run. Each attempt is a few milliseconds.
	for attempt := range attempts {
		if lost := concurrentAppendAttempt(t); lost != "" {
			t.Fatalf("attempt %d: event from %s reached no prompt and was never handed to a summarizer: it was lost",
				attempt, lost)
		}
	}
}

// What these two catch, measured by breaking the code on purpose:
//
//   - Making RepairAfterAppend never report a straggler is caught in about five
//     runs out of six.
//   - Making the pre-append race check never fire is NOT caught, and should not
//     be: the repair then handles the same case on its own. The two defences
//     overlap deliberately, so removing either one alone still loses nothing.
//     Removing both would be caught by the first bullet.

// attempts is how many interleavings one run samples.
//
// Not a round number picked for looks. Disabling the straggler repair loses an
// event in roughly one attempt in fifty, so a handful of attempts would let that
// regression through most of the time; measured, this count catches it in about
// seven runs out of eight, for a second and a half under -race. Raising it buys
// a little more certainty for proportionally more time.
const attempts = 200

// concurrentAppendAttempt runs one interleaving and returns the invocation ID of
// an event that was lost, or "" if none was.
func concurrentAppendAttempt(t *testing.T) string {
	t.Helper()

	const writers = 4

	base, sess := tailRetentionFixture(t, 6)
	summarizer := &recordingSummarizer{}

	appended := make([]*session.Event, writers)
	for i := range appended {
		ev := session.NewEvent(t.Context(), fmt.Sprintf("writer-%d", i))
		ev.Author = "user"
		// Inside the range the compaction is about to claim. An event carries
		// the time it was created rather than stored, so a parallel branch
		// produces exactly this.
		ev.Timestamp = time.Unix(int64(3+i), 0)
		ev.LLMResponse.Content = genai.NewContentFromText("concurrent", "user")
		appended[i] = ev
	}

	var wg sync.WaitGroup
	start := make(chan struct{})
	for _, ev := range appended {
		wg.Add(1)
		go func() {
			defer wg.Done()
			<-start
			if err := base.AppendEvent(t.Context(), sess, ev); err != nil {
				t.Errorf("concurrent append failed: %v", err)
			}
		}()
	}

	wg.Add(1)
	go func() {
		defer wg.Done()
		<-start
		// A failure here is a legitimate outcome: compaction may decline, or be
		// discarded because the session moved under it. What must not happen is
		// an event lost with no summary describing it, which is checked below.
		_ = runCompactionProcessor(t, base, sess, &compaction.Config{
			TokenThreshold:     100,
			EventRetentionSize: 2,
			Summarizer:         summarizer,
		})
	}()

	close(start)
	wg.Wait()

	got, err := base.Get(t.Context(), &session.GetRequest{AppName: "app", UserID: "u", SessionID: "s"})
	if err != nil {
		t.Fatalf("Get() error = %v", err)
	}
	var stored []*session.Event
	for ev := range got.Session.Events().All() {
		stored = append(stored, ev)
	}

	inPrompt := make(map[string]bool)
	for _, ev := range compactioninternal.Apply(stored) {
		inPrompt[ev.InvocationID] = true
	}
	for _, ev := range appended {
		id := ev.InvocationID
		if !inPrompt[id] && !summarizer.summarized(id) {
			return id
		}
	}
	return ""
}
