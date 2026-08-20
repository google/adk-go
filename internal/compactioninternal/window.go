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

	"google.golang.org/genai"

	"google.golang.org/adk/v2/internal/utils"
	"google.golang.org/adk/v2/session"
)

// longestSelfContainedPrefix returns the longest prefix of events that is safe
// to summarize.
//
// A single left-to-right pass tracks "open" obligations keyed by call ID: a
// function call, or a tool-confirmation request, opens one; a function response
// with the same ID closes it. Responses are applied before calls within one
// event, so a response only ever closes an obligation opened by an earlier
// event. Summarizing is safe exactly at the points where nothing is open, so
// the prefix ending at the last such point is returned.
//
// The result is empty when the window never reaches a balanced point, which
// tells the caller to skip this compaction rather than strand a half-finished
// tool interaction. Without this, a summary could swallow a function call while
// leaving its response behind, which downstream prompt assembly rejects.
//
// The prefix is additionally pulled back off a timestamp tie. Compaction
// coverage is an inclusive timestamp range, so if the first excluded event
// shares a timestamp with the last included one, it would fall inside the
// summarized range without having been summarized, and disappear from the
// prompt. Cutting before the whole tied group keeps "summarized" and "covered"
// the same set.
func longestSelfContainedPrefix(events []*session.Event) []*session.Event {
	openIDs := make(map[string]struct{})
	safeLength := 0
	for i, ev := range events {
		for _, resp := range utils.FunctionResponses(utils.Content(ev)) {
			delete(openIDs, resp.ID)
		}
		for _, call := range utils.FunctionCalls(utils.Content(ev)) {
			openIDs[callObligationKey(call, i)] = struct{}{}
		}
		for id := range ev.Actions.RequestedToolConfirmations {
			openIDs[id] = struct{}{}
		}
		// TODO: track outstanding authentication requests here too once
		// adk-go models them on EventActions.
		if len(openIDs) == 0 {
			safeLength = i + 1
		}
	}
	return events[:trimToTimestampBoundary(events, safeLength)]
}

// callObligationKey returns the key a function call is tracked under while
// waiting for its response.
//
// An ID-less call gets a synthetic key that no response can match, so it stays
// open forever and the prefix is cut before it. FunctionCall.ID is optional and
// some providers omit it, and keying such a call on "" would let a single
// unrelated ID-less response close it, or -- worse, and what the earlier
// implementation did -- skip it entirely so the trim that protects every other
// call silently never fires. Refusing to summarize is the safe direction.
func callObligationKey(call *genai.FunctionCall, eventIndex int) string {
	if call.ID != "" {
		return call.ID
	}
	return fmt.Sprintf("\x00no-id\x00%d\x00%s", eventIndex, call.Name)
}

// trimToTimestampBoundary pulls length back so the cut does not fall inside a
// group of events sharing a timestamp.
func trimToTimestampBoundary(events []*session.Event, length int) int {
	if length <= 0 || length >= len(events) {
		return length
	}
	boundary := events[length].Timestamp
	for length > 0 && !events[length-1].Timestamp.Before(boundary) {
		length--
	}
	return length
}

// LatestCompactionEvent returns the newest compaction event in events that no
// other compaction subsumes, or nil when events holds no compaction at all.
//
// A compaction is subsumed when another compaction fully contains its range: a
// strictly wider range, or an identical range appearing later in the stream.
//
// Ties are broken by stream position rather than by greatest end timestamp,
// because the summary written later saw more history and supersedes the earlier
// one even when both cover the same range.
func LatestCompactionEvent(events []*session.Event) *session.Event {
	var latest *session.Event
	for i, ev := range events {
		// hasCompaction, not HasUsableSummary, deliberately. A record with no
		// usable content still marks how far compaction reached, so the next
		// window must start after it. Requiring content here would make the
		// next window re-summarize everything the broken record covered.
		// Substitution keys off the stronger predicate, which is what stops a
		// contentless record from standing in as conversation.
		if !hasCompaction(ev) {
			continue
		}
		if isCompactionSubsumed(i, ev.Actions.Compaction, events) {
			continue
		}
		latest = ev
	}
	return latest
}

// isCompactionSubsumed reports whether the compaction at index i is fully
// covered by another compaction in events. Identical coverage is broken by
// stream position: the earlier event is subsumed by the later one.
func isCompactionSubsumed(i int, rng *session.EventCompaction, events []*session.Event) bool {
	for j, other := range events {
		// HasUsableSummary rather than hasCompaction: only a record carrying
		// usable content may evict another. Keying on the weaker predicate let
		// a contentless record subsume a real summary, destroying one already
		// paid for. Nothing then represented the range: the covered events fell
		// back to raw and the boundary calculation went on pointing at the
		// useless record.
		if j == i || !HasUsableSummary(other) {
			continue
		}
		o := other.Actions.Compaction
		// Subsuming means standing in for everything the other one stood in
		// for. Discarding a record whose events the survivor does not cover
		// would leave those events represented by nothing at all, which is the
		// failure this whole model exists to remove.
		if !coversAllOf(o, rng) {
			continue
		}
		if o.StartTimestamp.Before(rng.StartTimestamp) || o.EndTimestamp.After(rng.EndTimestamp) ||
			len(o.ExcludedEvents) < len(rng.ExcludedEvents) || j > i {
			return true
		}
	}
	return false
}

// selectSlidingWindow returns the events a sliding-window compaction should
// summarize, or nil when there is nothing to compact yet.
//
// The window is a *contiguous slice* of the event list, from the first event of
// the oldest invocation being compacted through the last event of the newest.
// Contiguity is the point: compaction coverage is recorded as an inclusive
// timestamp range, and the prompt builder drops every event inside that range.
// Building the window by filtering instead would let an event be skipped by the
// filter yet still fall inside the range, so it would be dropped from the
// prompt without ever having been summarized. Slicing makes that unexpressible.
//
// Which invocations to cover is decided first, then the slice is taken. An
// invocation counts as new when it has any event after the most recent
// compaction boundary. Once interval new invocations exist, the window reaches
// back overlap further invocations so consecutive summaries share context.
//
// nil comes back when fewer than interval new invocations exist, or when the
// slice has no self-contained prefix left after trimming.
func selectSlidingWindow(events []*session.Event, interval, overlap int) []*session.Event {
	if interval <= 0 {
		return nil
	}

	// Invocations in first-seen order, and whether each still holds anything no
	// summary stands in for. hasCompaction rather than HasUsableSummary: an
	// event declaring a compaction is bookkeeping even when its content is
	// unusable, and must never be counted as a conversational invocation.
	//
	// Asking what is covered, rather than comparing against the newest
	// compaction's end timestamp, is what stops a stall. A window is trimmed to
	// one branch and one isolation scope, and when the branch changes inside an
	// invocation the recorded end stops short of that invocation's last event.
	// Every later turn then saw the same invocation as new, recomputed a
	// byte-identical window, and paid for a model call that changed nothing.
	// Forking a child branch inside one invocation is the ordinary multi-agent
	// shape, so this was not an edge case. Coverage moves forward on each pass
	// even when the cut does not reach the end of a turn.
	var order []string
	isNew := make(map[string]bool)
	for i, ev := range events {
		if hasCompaction(ev) || ev.InvocationID == "" {
			continue
		}
		if _, ok := isNew[ev.InvocationID]; !ok {
			order = append(order, ev.InvocationID)
			isNew[ev.InvocationID] = false
		}
		if !coveredByAny(i, ev, events) {
			isNew[ev.InvocationID] = true
		}
	}

	firstNew := -1
	newCount := 0
	for i, id := range order {
		if isNew[id] {
			if firstNew < 0 {
				firstNew = i
			}
			newCount++
		}
	}
	if firstNew < 0 || newCount < interval {
		return nil
	}

	// Cover at most interval new invocations, rather than running to the end of
	// the session.
	//
	// Uncapped, the window is O(session) instead of O(interval): the first
	// compaction after enabling the feature on an existing deployment would
	// hand a whole live conversation to one model call, which can exceed the
	// summarizer's own context limit. It also compounds, because a summarizer
	// error records nothing, so the next turn recomputes from the same start
	// over a strictly larger window and is more likely to fail again. Capping
	// makes a retry the same size as the attempt that failed, and drains any
	// backlog one bounded window per turn.
	// The end is the interval-th invocation that still needs summarizing, not
	// the interval-th invocation outright.
	//
	// Counting covered ones lets a single invocation that can never be
	// compacted, a call awaiting approval being the ordinary case, pin the
	// start and hold the end one step behind it for ever. The interval means
	// "this many turns of new conversation", so covered turns should not spend
	// it.
	newPositions := make([]int, 0, newCount)
	for i, id := range order {
		if isNew[id] {
			newPositions = append(newPositions, i)
		}
	}
	startID := order[max(0, firstNew-overlap)]
	endID := order[newPositions[min(len(newPositions)-1, interval-1)]]

	// Where each invocation sits in the sequence, so an already-summarized
	// event can be told apart from one deliberately pulled back by overlap.
	// Overlap re-summarizes whole earlier invocations on purpose, and those all
	// sit before firstNew.
	position := make(map[string]int, len(order))
	for i, id := range order {
		position[id] = i
	}
	staleAt := func(idx int, ev *session.Event) bool {
		return position[ev.InvocationID] >= firstNew && coveredByAny(idx, ev, events)
	}

	// Slice from the first uncovered event of startID through the last of
	// endID. Events in between are included whatever they are, including ones
	// with no invocation ID.
	//
	// Skipping what is already summarized within the new invocations is the
	// other half of the stall: an invocation left partly compacted by a scope
	// cut would be re-sliced from the same place on the next pass, the same cut
	// would fall in the same spot, and the window would never move. Events an
	// overlap deliberately pulls back are not skipped, since re-summarizing
	// them is the whole point of overlap.
	// Bounded by where the chosen invocations sit in the sequence, not by the
	// last live event of endID specifically.
	//
	// Anchoring on endID could put first past last and return nil for ever.
	// endID resolves from firstNew, which does not move while nothing is
	// compacted, so once that invocation is fully covered, or its last live
	// event precedes startID's first, every later turn recomputed the same
	// empty answer. Silently: an empty window is indistinguishable from
	// "nothing to do yet". Reached by an ordinary pending tool confirmation,
	// where a paused run reuses its invocation ID, as well as by a
	// late-resuming invocation, and it did not recover when the tool answered.
	endPos := position[endID]
	first, last := -1, -1
	for i, ev := range events {
		if hasCompaction(ev) || staleAt(i, ev) {
			continue
		}
		pos, known := position[ev.InvocationID]
		if !known {
			continue
		}
		if first < 0 && pos >= position[startID] {
			first = i
		}
		if pos <= endPos {
			last = i
		}
	}
	if first < 0 || last < first {
		return nil
	}

	window := make([]*session.Event, 0, last-first+1)
	for off, ev := range events[first : last+1] {
		// Prior summaries are bookkeeping rather than conversation, and an
		// event a summary already stands in for is not re-summarized unless
		// overlap asked for it. Summaries themselves are never re-summarized,
		// so a sliding-window compaction is a constant-factor reduction rather
		// than a bound; tail retention is what bounds prompt growth.
		if hasCompaction(ev) || staleAt(first+off, ev) {
			continue
		}
		window = append(window, ev)
	}

	// A summary inherits the branch and isolation scope of what it covers, so
	// the window has to be homogeneous in both. A contiguous slice of a
	// multi-agent session routinely spans branches, and summarizing across one
	// would fold a sub-agent's content into a summary visible to the parent,
	// defeating the filters that keep those separate.
	window = trimToOneScope(window)

	if trimmed := longestSelfContainedPrefix(window); len(trimmed) > 0 {
		return trimmed
	}
	return skipBlockedHead(window)
}

// trimToOneScope cuts the window at the first event whose branch or isolation
// scope differs from the first event's.
func trimToOneScope(window []*session.Event) []*session.Event {
	if len(window) == 0 {
		return window
	}
	branch, scope := window[0].Branch, window[0].IsolationScope
	for i, ev := range window {
		if ev.Branch != branch || ev.IsolationScope != scope {
			return window[:i]
		}
	}
	return window
}

// skipBlockedHead handles a window whose very first events hold a function call
// that never got a response, which leaves no self-contained prefix at all.
//
// A tool awaiting human approval, or one whose backend died, blocks the head of
// the window permanently. Because the window is anchored to the last compaction
// boundary, that call stays at the head on every later attempt, so compaction
// would stop for the rest of the session and, since "no prefix" and "not enough
// invocations yet" both come back as nil, do so silently. Long tool-using
// sessions are exactly the ones compaction exists for.
//
// So instead of giving up, step past the blocked head and summarize the longest
// self-contained run that follows. The blocked call and everything before it
// stay raw and visible, which is what a pending call needs anyway. The summary
// is a contiguous later range, so the coverage invariant still holds.
//
// A run that answers a call left behind in the skipped head is refused.
// longestSelfContainedPrefix only tracks obligations opened inside the slice it
// is given, so a response whose call sits earlier looks unremarkable to it: the
// response would be summarized while its call stayed raw, and the model would
// be shown a call it had already answered with the answer gone. Refusing every
// unmatched response instead would be too strong, and would stall the ordinary
// long-running-tool resume, where the call is behind the compaction boundary
// and legitimately already summarized. The distinction is whether the call is
// in the head this function chose to skip.
//
// nil still comes back when nothing after the blockage is self-contained
// either.
func skipBlockedHead(window []*session.Event) []*session.Event {
	for start := 1; start < len(window); start++ {
		// Resume just after an event that changed the set of open obligations,
		// so the scan is over boundaries that matter rather than every offset.
		//
		// A response counts, not only a call. One model turn emitting a call to
		// an ordinary tool alongside one to a long-running tool is the standard
		// long-running shape, and the resume point that works there is the one
		// just after the ordinary tool's response: the head then holds that call
		// and its answer, only the long-running call is still open, and the tail
		// answers nothing. Resuming only after an event that opened an
		// obligation could never reach it, so every candidate had the response
		// in the tail with its call open in the head, all of them were refused,
		// and nothing after the blockage was compacted again.
		prev := window[start-1]
		if len(utils.FunctionCalls(utils.Content(prev))) == 0 &&
			len(utils.FunctionResponses(utils.Content(prev))) == 0 &&
			len(prev.Actions.RequestedToolConfirmations) == 0 {
			continue
		}
		tail := longestSelfContainedPrefix(window[start:])
		if len(tail) == 0 {
			continue
		}
		if answersAnyOf(tail, openCallIDs(window[:start])) {
			continue
		}
		return tail
	}
	return nil
}

// openCallIDs returns the call IDs opened by events and not answered by them.
func openCallIDs(events []*session.Event) map[string]struct{} {
	open := make(map[string]struct{})
	for i, ev := range events {
		for _, resp := range utils.FunctionResponses(utils.Content(ev)) {
			delete(open, resp.ID)
		}
		for _, call := range utils.FunctionCalls(utils.Content(ev)) {
			open[callObligationKey(call, i)] = struct{}{}
		}
		for id := range ev.Actions.RequestedToolConfirmations {
			open[id] = struct{}{}
		}
	}
	return open
}

// answersAnyOf reports whether events answer any of the given call IDs.
func answersAnyOf(events []*session.Event, ids map[string]struct{}) bool {
	if len(ids) == 0 {
		return false
	}
	for _, ev := range events {
		for _, resp := range utils.FunctionResponses(utils.Content(ev)) {
			if _, ok := ids[resp.ID]; ok {
				return true
			}
		}
	}
	return false
}

// coversAllOf reports whether a stands in for every event b does.
//
// a's range has to contain b's, and a must not exclude anything b covers.
// Discarding a record whose events the survivor does not cover would leave
// those events represented by nothing at all, which is the failure this whole
// model exists to remove.
func coversAllOf(a, b *session.EventCompaction) bool {
	if a == nil || b == nil {
		return false
	}
	if a.StartTimestamp.After(b.StartTimestamp) || a.EndTimestamp.Before(b.EndTimestamp) {
		return false
	}
	for _, ref := range a.ExcludedEvents {
		// Only a hole inside b's own range is a question. b never spanned the
		// events outside it, so it had nothing to say about them and could not
		// have named them however correct it was. Asking anyway made the test
		// impossible to satisfy for a strictly wider record, which is the case
		// subsumption exists for: nothing was ever absorbed, both summaries
		// materialized, and the prompt came out larger than with no compaction.
		if ref.Timestamp.Before(b.StartTimestamp) || ref.Timestamp.After(b.EndTimestamp) {
			continue
		}
		// An event a leaves out is fine only if b leaves it out too.
		if !namesHole(b, ref) {
			return false
		}
	}
	return true
}

// namesHole reports whether rng excludes the event ref names.
//
// The same comparison excludes uses, and for the same reason. slices.Contains
// compares EventRef with ==, which on the time.Time inside it compares the wall
// clock, the monotonic reading and the *Location pointer rather than the
// instant. Two references to one event stop matching when either has been
// through a store: a round trip drops the monotonic reading, and a backend that
// returns a different timezone changes the pointer. Both name the same instant
// and neither is wrong.
func namesHole(rng *session.EventCompaction, ref session.EventRef) bool {
	at := ref.Timestamp.Truncate(refResolution)
	for _, other := range rng.ExcludedEvents {
		if other.InvocationID == ref.InvocationID && other.Timestamp.Truncate(refResolution).Equal(at) {
			return true
		}
	}
	return false
}
