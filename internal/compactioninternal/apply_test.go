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
	"time"

	"github.com/google/go-cmp/cmp"

	"google.golang.org/adk/v2/internal/utils"
	"google.golang.org/adk/v2/session"
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

	if HasUsableSummary(contentless) {
		t.Error("HasUsableSummary() = true for a contentless compaction, want false (nothing to show a model)")
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

// TestApplyEqualRangeSummariesKeepCoverage checks that discarding one of two
// summaries with identical ranges does not also lose what they covered. The
// survivor spans the same events, so its content stands in for them.
//
// Equal ranges are not reachable from a single invocation, since each window
// starts after the previous compaction. They were a second-order consequence of
// two invocations compacting the same session concurrently, which the runner
// now prevents by re-reading and discarding a summary whose range was raced.
func TestApplyEqualRangeSummariesKeepCoverage(t *testing.T) {
	t.Parallel()

	events := []*session.Event{
		textEvent("a", "inv1", 1, "TURN-ONE"),
		modelTextEvent("b", "inv1", 2, "TURN-TWO"),
		compactionEvent("s1", 3, 1, 2, "SUM-1"),
		compactionEvent("s2", 4, 1, 2, "SUM-2"),
		textEvent("c", "inv2", 5, "TURN-FIVE"),
	}

	got := Apply(events)
	if diff := cmp.Diff([]string{"s2", "c"}, ids(got)); diff != "" {
		t.Errorf("Apply() mismatch (-want +got):\n%s", diff)
	}
	// The covered span must still be represented by the surviving summary
	// rather than vanishing along with the discarded one.
	if texts := utils.TextParts(utils.Content(got[0])); len(texts) != 1 || texts[0] != "SUM-2" {
		t.Errorf("surviving summary content = %v, want SUM-2 standing in for the covered turns", texts)
	}
}

// TestApplyPreservesStreamOrder pins that Apply does not reorder by timestamp.
//
// Clock skew between writers, or the microsecond truncation the SQL backend
// applies, can leave a response with an earlier timestamp than the call it
// answers. Sorting on timestamp would then emit the response first.
func TestApplyPreservesStreamOrder(t *testing.T) {
	t.Parallel()

	call := callEvent("call", "inv1", 9, "c1")
	resp := responseEvent("resp", "inv1", 8, "c1") // earlier timestamp than its call
	events := []*session.Event{
		textEvent("u", "inv1", 1, "q"),
		compactionEvent("s1", 2, 1, 1, "SUM"),
		call,
		resp,
	}

	got := ids(Apply(events))
	if diff := cmp.Diff([]string{"s1", "call", "resp"}, got); diff != "" {
		t.Errorf("Apply() mismatch (-want +got):\n%s\nthe response must not precede its call", diff)
	}
}

// TestApplySummaryPrecedesUncoveredTail pins where a summary lands when the
// event declaring it was appended some way after the range it covers.
//
// A compaction event follows the range it summarizes, but not necessarily
// immediately: raw turns can sit in between. Materializing the summary at the
// declaring event's own position would show the model a summary of older
// history after the newer turns it precedes.
func TestApplySummaryPrecedesUncoveredTail(t *testing.T) {
	t.Parallel()

	events := []*session.Event{
		textEvent("a", "inv1", 1, "q1"),
		modelTextEvent("b", "inv1", 2, "a1"),
		textEvent("c", "inv2", 3, "q2"),
		modelTextEvent("d", "inv2", 4, "a2"),
		// Covers only the first exchange, but is appended after the second.
		compactionEvent("s1", 5, 1, 2, "SUM"),
	}

	got := ids(Apply(events))
	if diff := cmp.Diff([]string{"s1", "c", "d"}, got); diff != "" {
		t.Errorf("Apply() mismatch (-want +got):\n%s\nthe summary must precede the turns it does not cover", diff)
	}
}

// TestApplyContentlessRecordDoesNotEvictASummary checks that a compaction
// record carrying no content cannot subsume a real summary.
//
// Subsumption used to key on the weaker "declares a compaction" predicate while
// substitution kept only records with content, so a contentless record could
// evict a usable summary and leave nothing representing the range. A record like
// that reaches a session from a third-party Summarizer or a backend that
// round-trips the field lossily.
func TestApplyContentlessRecordDoesNotEvictASummary(t *testing.T) {
	t.Parallel()

	real := compactionEvent("s1", 3, 1, 2, "SUM")
	// A wider, contentless record recorded afterwards.
	blank := compactionEvent("s2", 4, 1, 2, "")
	blank.Actions.Compaction.CompactedContent = nil

	events := []*session.Event{
		textEvent("a", "inv1", 1, "q1"),
		modelTextEvent("b", "inv1", 2, "a1"),
		real,
		blank,
		textEvent("c", "inv2", 5, "q2"),
	}

	got := ids(Apply(events))
	if diff := cmp.Diff([]string{"s1", "c"}, got); diff != "" {
		t.Errorf("Apply() mismatch (-want +got):\n%s\na contentless record must not destroy a summary already paid for", diff)
	}
}

// TestApplyToleratesNilEvents checks that Apply does not panic on a nil entry.
// Apply is reachable from an exported entry point, so a malformed list must be
// an input it survives rather than a crash.
func TestApplyToleratesNilEvents(t *testing.T) {
	t.Parallel()

	events := []*session.Event{
		textEvent("a", "inv1", 1, "q1"),
		nil,
		modelTextEvent("b", "inv1", 2, "a1"),
		compactionEvent("s1", 3, 1, 1, "SUM"),
		nil,
		textEvent("c", "inv2", 4, "q2"),
	}

	got := ids(Apply(events))
	if diff := cmp.Diff([]string{"s1", "b", "c"}, got); diff != "" {
		t.Errorf("Apply() mismatch (-want +got):\n%s", diff)
	}
}

// TestApplyKeepsAnEventTheSummaryDidNotCover is the property the covered-ID set
// exists for.
//
// Choosing a window filters events out of the middle of its own span, by
// branch, by isolation scope and by what the retained tail holds back. A
// timestamp range covering the ends therefore covers those gaps too, and an
// event in a gap was dropped from every later prompt having been summarized by
// nothing. Its content was simply lost, with no summary standing in for it.
func TestApplyKeepsAnEventTheSummaryDidNotCover(t *testing.T) {
	t.Parallel()

	// The summary spans a..d but stands in only for a and d. Whatever kept b
	// and c out of the window, they were handed to no summarizer.
	summary := compactionEvent("s1", 9, 1, 4, "summary of a and d", excl("inv1", 2), excl("inv1", 3))
	events := []*session.Event{
		textEvent("a", "inv1", 1, "q1"),
		textEvent("b", "inv1", 2, "sibling branch"),
		textEvent("c", "inv1", 3, "retained tail"),
		modelTextEvent("d", "inv1", 4, "a1"),
		summary,
	}

	got := ids(Apply(events))
	if diff := cmp.Diff([]string{"s1", "b", "c"}, got); diff != "" {
		t.Errorf("prompt events mismatch (-want +got):\n%s", diff)
	}
}

// TestApplyKeepsAnEventTiedToTheWindowHead pins the boundary case that a
// timestamp range cannot express.
//
// With events x@1, a@1 and b@3 and a window of [a b], the recorded range is
// [1..3] and x sits inside it while having been summarized by nothing. Ties are
// not hypothetical: the SQL backend truncates timestamps to microseconds, and
// the platform time provider makes replay deterministic.
func TestApplyKeepsAnEventTiedToTheWindowHead(t *testing.T) {
	t.Parallel()

	events := []*session.Event{
		textEvent("x", "inv1", 1, "tied to the head, never summarized"),
		textEvent("a", "inv1", 1, "q1"),
		modelTextEvent("b", "inv1", 3, "a1"),
		compactionEvent("s1", 9, 1, 3, "summary of a and b", excl("inv1", 1)),
	}

	// x keeps its place ahead of the summary, which is emitted where the first
	// event it does cover used to sit.
	//
	// "a" is kept too, and that is the cost of referring to an excluded event
	// by invocation and timestamp rather than by ID: the reference names both
	// events of the tied pair. Over-excluding leaves an event raw beside a
	// summary of it, which is visible and recoverable, and it buys a key that
	// survives a backend that reassigns event IDs. Under-excluding would delete
	// x, which is the failure this whole model exists to remove.
	got := ids(Apply(events))
	if diff := cmp.Diff([]string{"x", "a", "s1"}, got); diff != "" {
		t.Errorf("prompt events mismatch (-want +got):\n%s", diff)
	}
}

// selfWrappingSession returns itself from Unwrap, the shape a third-party
// session decorator can take by accident.
type selfWrappingSession struct{ staticSession }

func (s *selfWrappingSession) Unwrap() session.Session { return s }

// TestUnwrapSessionStopsOnACycle pins that unwrapping terminates.
//
// Unwrap is matched structurally, so any session with the method satisfies it,
// including one written outside this repository. A decorator that returns
// itself would spin the unwrap loop for ever and hang the invocation rather
// than fail it, so the loop gives up instead.
func TestUnwrapSessionStopsOnACycle(t *testing.T) {
	t.Parallel()

	s := &selfWrappingSession{}
	done := make(chan session.Session, 1)
	go func() { done <- UnwrapSession(s) }()

	select {
	case got := <-done:
		if got != session.Session(s) {
			t.Errorf("UnwrapSession returned %T, want the session it gave up on", got)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("UnwrapSession did not return: the unwrap loop has no cycle guard")
	}
}

// TestExcludesSurvivesABackendThatDropsPrecision pins that a hole still matches
// after a round trip through a store that keeps fewer digits than the clock.
//
// A reference is written from an event held in memory and compared against the
// same event read back. The SQL backend truncates event timestamps to
// microseconds while the compaction record travels beside them as JSON at full
// nanosecond precision, and the Vertex AI service takes the event timestamp
// from the server envelope while the reference comes from the client-written
// payload. Comparing exactly answered no for an event the reference names, and
// coverage is the range minus the exclusions, so answering no does not leave
// the event alone: it hands it to a summary that never saw it.
//
// On SQL this is currently masked, and only by accident. AppendEvent truncates
// the caller's event struct in place, so a reference built later from either
// copy agrees. That is an undocumented mutation of an argument the caller still
// owns, and tidying it away would silently start deleting conversation.
func TestExcludesSurvivesABackendThatDropsPrecision(t *testing.T) {
	t.Parallel()

	ns := time.Date(2026, 3, 4, 5, 6, 7, 123456789, time.UTC)
	rng := &session.EventCompaction{
		StartTimestamp: ns.Add(-time.Hour),
		EndTimestamp:   ns.Add(time.Hour),
		// Written at full precision, the way a record reaches storage.
		ExcludedEvents: []session.EventRef{{InvocationID: "inv1", Timestamp: ns}},
	}
	// Read back from a store that keeps microseconds.
	ev := &session.Event{ID: "a", InvocationID: "inv1", Timestamp: ns.Truncate(time.Microsecond)}

	if ev.Timestamp.Before(rng.StartTimestamp) || ev.Timestamp.After(rng.EndTimestamp) {
		t.Fatal("the event is outside the interval, so this test proves nothing")
	}
	if !excludes(rng, ev) {
		t.Error("the hole stopped matching after a round trip, so a summary that never saw this event now covers it")
	}
	// inRange is coverage: inside the interval and not named as a hole. The
	// hole matching is what keeps this event out of the summary's reach.
	if inRange(ev, rng) {
		t.Error("an event the record names as a hole is being treated as covered")
	}

	// The range test is deliberately not normalised. An event just past the end
	// was summarized by nothing, and widening the range to reach it is the
	// deletion this whole mechanism exists to prevent.
	past := &session.Event{ID: "b", InvocationID: "inv1", Timestamp: rng.EndTimestamp.Add(time.Nanosecond)}
	if inRange(past, rng) {
		t.Error("an event after the range end is being treated as covered")
	}
}

// mutableSession is a session whose event list can grow, the way every backend
// mutates its session object in place.
type mutableSession struct {
	staticSession
}

func (m *mutableSession) append(ev *session.Event) { m.events = append(m.events, ev) }

// TestRangeRacedSinceSeesAnAppendThroughALiveHandle pins that the race check
// can still detect a straggler when the caller only has one session object.
//
// RangeRaced compares two session handles. On the mid-turn path both arguments
// were the same live object, and every backend mutates that object in place, so
// the "before" state already contained whatever arrived during the model call
// and the comparison always found nothing. Capturing the identities up front is
// what makes the question answerable.
func TestRangeRacedSinceSeesAnAppendThroughALiveHandle(t *testing.T) {
	t.Parallel()

	sess := &mutableSession{staticSession{events: []*session.Event{
		textEvent("a", "inv1", 1, "q1"),
		modelTextEvent("b", "inv1", 2, "a1"),
	}}}
	summary := compactionEvent("s1", 9, 1, 3, "summary of the turn")

	before := KnownEventIDs(sess)

	// A sibling appends inside the range while the summarizer is working.
	sess.append(textEvent("straggler", "inv2", 3, "please do not lose me"))

	if RangeRaced(sess, sess, summary) {
		t.Error("RangeRaced() = true through one live handle, which it cannot actually detect")
	}
	if !RangeRacedSince(sess, before, summary) {
		t.Error("RangeRacedSince() = false, want true: an event landed inside the range during the call")
	}
}

// TestRangeRacedSinceSurvivesABackendThatRewritesEventIDs pins that the race
// guard identifies an event the way the rest of the package does.
//
// Keying on Event.ID looks natural and is the one field documented as not
// surviving storage: the Vertex AI service replaces it with a server resource
// name on read. The snapshot then holds client-side IDs and the re-read holds
// server ones, nothing matches, every event reads as a racer, and every summary
// is discarded after the model call that produced it was paid for. Tail
// retention, which is the only strategy that bounds prompt growth, could never
// store anything on that backend, and nothing would have said so.
func TestRangeRacedSinceSurvivesABackendThatRewritesEventIDs(t *testing.T) {
	t.Parallel()

	before := []*session.Event{
		textEvent("client-a", "inv1", 1, "q1"),
		modelTextEvent("client-b", "inv1", 2, "a1"),
	}
	known := KnownEventIDs(&staticSession{events: before})

	// The same two events read back from a store that assigned its own IDs.
	after := []*session.Event{
		textEvent("server-a", "inv1", 1, "q1"),
		modelTextEvent("server-b", "inv1", 2, "a1"),
	}
	summary := compactionEvent("s1", 9, 1, 2, "summary of the turn")

	if RangeRacedSince(&staticSession{events: after}, known, summary) {
		t.Error("RangeRacedSince() = true: the backend renamed the events and they were mistaken for racers")
	}

	// A genuine racer is still caught.
	withRacer := append(after, textEvent("server-c", "inv2", 2, "landed during the call"))
	if !RangeRacedSince(&staticSession{events: withRacer}, known, summary) {
		t.Error("RangeRacedSince() = false, want true: an event really did land inside the range")
	}
}
