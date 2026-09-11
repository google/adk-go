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
	"fmt"
	"slices"
	"strings"
	"time"

	"google.golang.org/adk/v2/internal/utils"
	"google.golang.org/adk/v2/session"
)

// Apply rewrites an event list so compaction summaries stand in for the events
// they cover. It is what turns a stored compaction into a smaller prompt.
//
// Each surviving compaction event is replaced by a model-authored event holding
// its summary content, positioned at the compaction's end timestamp. Raw events
// falling inside a surviving range are dropped. A compaction whose range
// another compaction fully contains is discarded along with its summary, so
// re-summarized ranges do not appear twice.
//
// Finally, function calls that a summary swallowed but whose responses arrived
// later are restored, so call and response stay paired.
//
// events is not modified, and is returned unchanged when it holds no
// compactions.
func Apply(events []*session.Event) []*session.Event {
	if !slices.ContainsFunc(events, hasCompaction) {
		return events
	}
	return recoverCompactedFunctionCalls(substituteSummaries(events), events)
}

// hasCompaction reports whether ev declares a compaction at all, usable or not.
// Apply keys off this rather than [HasUsableSummary] so that a malformed
// compaction is still stripped from the prompt instead of leaking through as a
// contentless raw event.
func hasCompaction(ev *session.Event) bool {
	return ev != nil && ev.Actions.Compaction != nil
}

// repairTimeout bounds the corrective write, which no longer stops when the
// caller does, so a wedged backend cannot hold a goroutine open for ever.
var repairTimeout = 30 * time.Second

// RepairContext returns the context the corrective write runs on: the caller's
// values, detached from its cancellation, bounded by repairTimeout.
//
// Storing a summary and correcting it is the one genuinely two-phase write in
// compaction, and between the two the stored record claims a range wider than
// what was actually summarized. On the caller's context, a cancellation
// arriving in that gap -- a user hanging up mid-turn, which is ordinary rather
// than exotic -- leaves the claim standing for good, and prompt assembly then
// drops those events from every later prompt with nothing standing in for them.
// Detaching turns a systematic loss back into the rare one the repair already
// tolerates and logs.
func RepairContext(ctx context.Context) (context.Context, context.CancelFunc) {
	return context.WithTimeout(context.WithoutCancel(ctx), repairTimeout)
}

// keptRange is a compaction range that survived subsumption, along with the
// stream positions that decide how far it reaches and what it materializes as.
//
// The two are not the same once a range has been corrected. index is the
// earliest position in the correction group, because that is how far back the
// group's authority genuinely extends; src is the surviving record, because
// that is the one whose identity and metadata the emitted summary carries.
type keptRange struct {
	// index is the coverage guard: only events before it can be covered, and
	// the summary materializes here when it covers nothing left in the stream.
	index int
	// src is the position of the record supplying the emitted summary event.
	src int
	rng *session.EventCompaction
}

// substituteSummaries drops raw events covered by a surviving compaction and
// materializes each surviving summary in their place, preserving chronological
// order.
func substituteSummaries(events []*session.Event) []*session.Event {
	var recAt []int
	var recs []*session.Event
	for j, ev := range events {
		if hasCompaction(ev) {
			recAt = append(recAt, j)
			recs = append(recs, ev)
		}
	}

	var kept []keptRange
	for k, ev := range recs {
		i := recAt[k]
		if !HasUsableSummary(ev) {
			continue
		}
		if ev.Actions.Compaction.EndTimestamp.Before(ev.Actions.Compaction.StartTimestamp) {
			// An inverted range covers nothing; materializing its summary would
			// duplicate content the raw events still supply. NewSummaryEvent
			// rejects these, but session.EventCompaction is a plain struct that
			// callers can also build directly.
			continue
		}
		if isSubsumedAmong(i, ev.Actions.Compaction, recAt, recs) {
			continue
		}
		pos, folded := foldCorrections(i, ev.Actions.Compaction, recAt, recs)
		kept = append(kept, keptRange{index: pos, src: i, rng: folded})
	}

	// Each surviving summary is emitted where the first event it covers sat,
	// and the events it covers are dropped.
	//
	// Stream position rather than timestamp: sorting the result on timestamp
	// could reorder raw events whose timestamps disagree with their arrival
	// order -- clock skew between writers, or the microsecond truncation the
	// SQL backend applies -- and so put a function response ahead of the call
	// it answers.
	//
	// The first covered event rather than the compaction event's own position:
	// a compaction event is appended after the range it covers, but not
	// necessarily right after it. Tail retention leaves a raw tail in between,
	// and emitting the summary where the compaction event sits would show the
	// model a summary of older history after the recent turns that follow it.
	summariesAt := make(map[int][]keptRange, len(kept))
	for _, k := range kept {
		at := summaryIndex(events, k)
		summariesAt[at] = append(summariesAt[at], k)
	}

	out := make([]*session.Event, 0, len(events))
	for i, ev := range events {
		for _, k := range summariesAt[i] {
			summary := *events[k.src]
			summary.Author = "model"
			summary.Timestamp = k.rng.EndTimestamp
			summary.LLMResponse.Content = k.rng.CompactedContent
			out = append(out, &summary)
		}
		if ev == nil {
			// A nil entry is not conversation and nothing can cover it.
			// Dropping it keeps Apply total over its input: it is reachable
			// from an exported entry point, so a malformed event list should
			// not panic deep inside coverage arithmetic.
			continue
		}
		if hasCompaction(ev) {
			// An event declaring a compaction is bookkeeping, never
			// conversation: its content slot holds nothing to show the model,
			// and its summary was emitted above at the position of the range it
			// covers.
			continue
		}
		if isCovered(i, ev, kept) {
			continue
		}
		out = append(out, ev)
	}
	return out
}

// summaryIndex is the stream position at which k's summary materializes: where
// the first event it covers sat, or the compaction event itself when it covers
// nothing left in the stream.
func summaryIndex(events []*session.Event, k keptRange) int {
	for i, ev := range events {
		if ev == nil || hasCompaction(ev) {
			continue
		}
		if coveredBy(i, ev, k) {
			return i
		}
	}
	return k.index
}

// isCovered reports whether the raw event at index i falls inside a surviving
// compaction range.
func isCovered(i int, ev *session.Event, kept []keptRange) bool {
	if ev == nil {
		return false
	}
	for _, k := range kept {
		if coveredBy(i, ev, k) {
			return true
		}
	}
	return false
}

// coveredBy reports whether the raw event at index i is covered by k.
// Only a compaction appearing later in the stream can cover an event: a summary
// never covers events recorded after it was written.
func coveredBy(i int, ev *session.Event, k keptRange) bool {
	if i >= k.index {
		return false
	}
	return inRange(ev, k.rng)
}

// recoverCompactedFunctionCalls re-injects function-call events that compaction
// removed but whose responses survived.
//
// The case this exists for is a paused long-running tool call: the call and its
// placeholder response are compacted together, then the real result arrives on
// resume as a later event that no summary covers. That surviving response would
// be orphaned, which breaks the call/response pairing prompt assembly requires.
//
// For each orphaned response the original call event is restored from
// sourceEvents (the pre-substitution list) and inserted just before the first
// surviving response referencing it. The whole call event comes back so
// parallel calls stay intact, and for every sibling call in it whose response
// was also compacted away, the freshest response is re-injected too, so the
// sibling does not surface as a phantom pending call.
//
// Only long-running calls are recovered, and that is the only shape this can
// legitimately arise in. longestSelfContainedPrefix guarantees the summarized
// window is balanced, so every call inside it had its response inside it too.
// The one way a response outlives its call is a second response for the same
// call ID arriving after the window, which is exactly the long-running pattern:
// a placeholder response closes the pair, the pair is compacted, and the real
// result lands later.
//
// An unmatched response with no long-running call is a genuine inconsistency,
// and it is left alone rather than guessed at. Recovering it would invent a call
// that never happened, hiding the underlying bug instead of exposing it.
//
// Be aware of where such a response ends up:
// rearrangeEventsForLatestFunctionResponse errors on it only when it is the
// final event, while rearrangeEventsForFunctionResponsesInHistory drops any
// response it cannot pair with a call. So a mid-history orphan disappears from
// the prompt silently rather than loudly. If that ever needs to be made loud,
// the fix belongs in those two functions, not here.
func recoverCompactedFunctionCalls(events, sourceEvents []*session.Event) []*session.Event {
	presentCalls := make(map[string]struct{})
	presentResponses := make(map[string]struct{})
	for _, ev := range events {
		for _, call := range utils.FunctionCalls(utils.Content(ev)) {
			presentCalls[call.ID] = struct{}{}
		}
		for _, resp := range utils.FunctionResponses(utils.Content(ev)) {
			presentResponses[resp.ID] = struct{}{}
		}
	}

	orphaned := make(map[string]struct{})
	for id := range presentResponses {
		if _, ok := presentCalls[id]; !ok && id != "" {
			orphaned[id] = struct{}{}
		}
	}
	if len(orphaned) == 0 {
		return events
	}

	// The long-running call events matching the orphaned responses.
	callEventByID := make(map[string]*session.Event)
	for _, ev := range sourceEvents {
		for _, call := range utils.FunctionCalls(utils.Content(ev)) {
			if _, ok := orphaned[call.ID]; !ok {
				continue
			}
			if _, ok := callEventByID[call.ID]; ok {
				continue
			}
			if slices.Contains(ev.LongRunningToolIDs, call.ID) {
				callEventByID[call.ID] = ev
			}
		}
	}
	if len(callEventByID) == 0 {
		return events
	}

	// Freshest response event per call ID, so a re-injected sibling carries its
	// final result rather than an intermediate placeholder.
	finalResponseByID := make(map[string]*session.Event)
	for _, ev := range sourceEvents {
		for _, resp := range utils.FunctionResponses(utils.Content(ev)) {
			if prev, ok := finalResponseByID[resp.ID]; !ok || !ev.Timestamp.Before(prev.Timestamp) {
				finalResponseByID[resp.ID] = ev
			}
		}
	}

	result := make([]*session.Event, 0, len(events)+len(callEventByID))
	reinjected := make(map[string]struct{})
	for _, ev := range events {
		for _, resp := range utils.FunctionResponses(utils.Content(ev)) {
			callEvent, ok := callEventByID[resp.ID]
			if !ok {
				continue
			}
			if _, done := reinjected[resp.ID]; done {
				continue
			}

			result = append(result, callEvent)

			// Every call in the recovered event is now present, including the
			// parallel siblings that came along for the ride.
			var siblings []*session.Event
			for _, call := range utils.FunctionCalls(utils.Content(callEvent)) {
				reinjected[call.ID] = struct{}{}
				if _, present := presentResponses[call.ID]; present {
					continue
				}
				if sibling, ok := finalResponseByID[call.ID]; ok && !slices.Contains(siblings, sibling) {
					siblings = append(siblings, sibling)
				}
			}
			result = append(result, siblings...)
		}
		result = append(result, ev)
	}
	return result
}

// RangeRaced reports whether the session gained an event inside summary's range
// while the summary was being produced.
//
// Both a competing compaction and an ordinary turn count.
//
// A summary records the holes inside its range, and that list is computed from
// what the framework could see when the summary was built. An event a
// concurrent invocation appends afterwards lands inside the range and is named
// by nothing, so it reads as covered and prompt assembly drops it, having been
// summarized by nothing. Naming the members instead of the holes would have
// made this case safe by omission, at the price of a list that grows with the
// conversation and a key every backend has to preserve.
//
// A second compaction counts for a different reason: two summaries whose
// ranges meet would each stand in for the same turns, so the same content is
// materialized twice into one prompt.
//
// selectedFrom is the session state the window was chosen from, and latest is a
// fresh read taken after summarizing. A compaction present in latest but absent
// from selectedFrom arrived while this one was being produced. Comparing the
// two states makes this exact rather than a guess about timestamps.
//
// Callers discard the summary when this returns true.
func RangeRaced(latest, selectedFrom session.Session, summary *session.Event) bool {
	if selectedFrom == nil {
		return false
	}
	return RangeRacedSince(latest, KnownEventIDs(selectedFrom), summary)
}

// KnownEventIDs snapshots which events a session holds at this moment.
//
// Taking the identities rather than keeping the session is the point. Every
// backend mutates its session object in place, so a handle held across a model
// call is not a record of what was there before the call: by the time it is
// compared it already contains whatever arrived during it, and the comparison
// finds no difference. A caller that cannot re-read a separate snapshot must
// capture this before it starts.
func KnownEventIDs(sess session.Session) map[string]struct{} {
	known := make(map[string]struct{})
	if sess == nil {
		return known
	}
	for _, ev := range collect(sess) {
		known[refKey(ev)] = struct{}{}
	}
	return known
}

// RangeRacedSince is [RangeRaced] against identities captured earlier rather
// than against a second session handle.
func RangeRacedSince(latest session.Session, known map[string]struct{}, summary *session.Event) bool {
	if latest == nil || summary == nil {
		return false
	}
	rng := summary.Actions.Compaction
	if rng == nil {
		return false
	}

	for _, ev := range collect(latest) {
		// Anything already present when the window was selected is what this
		// summary was built from, rather than a racer.
		//
		// Identified the way every other part of this package identifies an
		// event, by invocation and timestamp. Event.ID cannot be used: the
		// Vertex AI service replaces it with a server resource name on read, so
		// the snapshot holds client-side IDs and the re-read holds server ones,
		// nothing matches, and this turn's own events all read as racers. Every
		// summary was then discarded on that backend after being paid for, and
		// tail retention, the only strategy that bounds growth, could never
		// store anything at all. Silently.
		if _, seen := known[refKey(ev)]; seen {
			continue
		}
		if hasCompaction(ev) {
			if overlaps(rng, ev.Actions.Compaction) {
				return true
			}
			continue
		}
		if !ev.Timestamp.Before(rng.StartTimestamp) && !ev.Timestamp.After(rng.EndTimestamp) {
			return true
		}
	}
	return false
}

// refResolution is the granularity a hole reference is compared at.
//
// A reference is written from an event held in memory, at whatever precision
// the clock gave, and compared against the same event read back from a store
// that may keep fewer digits. The SQL backend truncates event timestamps to
// microseconds while the record travels beside them as JSON at full nanosecond
// precision, and the Vertex AI service takes the event timestamp from the
// server envelope while the reference comes from the client-written payload.
// Comparing exactly then answers no for an event the reference names, and
// because coverage is the range minus the exclusions, answering no deletes the
// event rather than leaving it alone.
//
// Microsecond is the coarsest precision any backend here keeps, so truncating
// both sides to it makes the comparison independent of who stored what.
const refResolution = time.Microsecond

// excludes reports whether rng names ev as a hole.
//
// Only the exclusion test is normalised, never inRange. Widening a hole leaves
// an extra event raw beside a summary of it, which is recoverable. Widening the
// range would pull in an event that sits just outside it and was summarized by
// nothing, which is the deletion this is here to prevent.
func excludes(rng *session.EventCompaction, ev *session.Event) bool {
	evAt := ev.Timestamp.Truncate(refResolution)
	for _, ref := range rng.ExcludedEvents {
		if ref.InvocationID == ev.InvocationID && ref.Timestamp.Truncate(refResolution).Equal(evAt) {
			return true
		}
	}
	return false
}

// coverIndex is the compaction records in a session, in stream order, so a
// caller asking the same question of many events walks the records rather than
// the whole session each time.
//
// Selection asks it once per event. Rescanning every event to find the records
// made that quadratic in session length, on a path that runs before every model
// call in a session that only ever grows: 8,000 events took 214ms and 16,000
// took 1.18s, of which 78% was this scan. Records are a small fraction of a
// session, so indexing them once makes it linear in events and negligible.
type coverIndex struct {
	at   []int
	rngs []*session.EventCompaction
}

func newCoverIndex(events []*session.Event) coverIndex {
	// The records are collected once and every later question is asked against
	// that list rather than against the whole session. Subsumption and the hole
	// union both need to compare a record with its peers, and doing either by
	// rescanning the events made index construction cost records times events:
	// 8,000 events with 400 records took 17ms, on a path that runs before every
	// model call.
	var recAt []int
	var recs []*session.Event
	for j, ev := range events {
		if hasCompaction(ev) {
			recAt = append(recAt, j)
			recs = append(recs, ev)
		}
	}

	var idx coverIndex
	for k, ev := range recs {
		// Superseded records are skipped, because prompt assembly skips them
		// too and the two have to agree about what is covered.
		//
		// Indexing every record made selection read one that Apply discards.
		// After a repair the store holds both the flawed record and its
		// correction: assembly subsumes the flawed one and leaves the straggler
		// raw, correctly, while selection found the flawed one first, saw no
		// hole for the straggler and called it covered. The event was then
		// never offered to another window, so it could not be lost and could
		// not be summarized either, which puts it outside the bound tail
		// retention exists to enforce, permanently, one event per lost append
		// race.
		if isSubsumedAmong(recAt[k], ev.Actions.Compaction, recAt, recs) {
			continue
		}
		// Contentless records are skipped for the same reason, and this is the
		// other half of the same agreement. substituteSummaries drops a record
		// with no usable summary, so the events in its range stay raw in the
		// prompt; indexing it here called them covered, and selection then
		// refused to offer them to any window. They were permanently raw and
		// permanently uncompactable at once, which is the unbounded growth the
		// model exists to remove.
		//
		// Deliberately not the rule LatestCompactionEvent uses: that answers
		// how far compaction has reached, this answers what is covered.
		if !HasUsableSummary(ev) {
			continue
		}
		pos, folded := foldCorrections(recAt[k], ev.Actions.Compaction, recAt, recs)
		idx.at = append(idx.at, pos)
		idx.rngs = append(idx.rngs, folded)
	}
	return idx
}

// foldCorrections merges a record with the corrections of it that cover exactly
// the same interval, returning the stream position the group occupies and the
// range it covers.
//
// It returns both because both have to be folded over the same group. Folding
// only the holes is what this used to do, and that moved the two orderings
// apart: a correction is appended after the record it corrects, so the
// survivor's position is later, and position is what stops a summary covering
// events recorded after it was written. An event appended between the original
// and its correction was out of the original's reach and inside the survivor's,
// so the repair covered it with a summary written before it existed -- the loss
// the repair pass exists to prevent, arriving through the repair itself.
//
// The earliest position in the group is the safe one. Nothing before the
// original moved, so everything the group genuinely summarized stays covered,
// while anything appended after it falls back to raw beside the summary, which
// is visible and recoverable.
//
// Records sharing an interval are corrections of one another rather than
// independent summaries: the repair pass writes the same content over the same
// range with a straggler added as a hole. Only one of them materializes, and
// picking a winner loses the holes named by the losers. With two corrections
// naming different stragglers, neither hole set contains the other, so whichever
// loses takes its straggler with it and that event is covered by a summary that
// never described it.
//
// Unioning is the safe direction. A hole is the claim "this event was not
// summarized". Honouring a hole that was not needed leaves an event raw beside a
// summary of it, which is visible and recoverable. Ignoring one that was needed
// deletes conversation. When records disagree, the claim of absence wins.
//
// Scoped to records that share an interval *and* the same summary text, which
// is what a correction is: the repair pass copies the content verbatim and only
// adds a hole. Two independent summaries can also span one interval, when
// several events share a timestamp, and those describe different events. Merging
// their holes and keeping one would discard the other's content while claiming
// its range, which is the loss this function exists to prevent, arriving by a
// different route. Folding across overlapping but different ranges would be
// worse still: a stale record would keep events raw for ever.
//
// Takes a pre-collected record list rather than the whole session: asking this
// per record while rescanning the events cost records times events on the
// selection path. at[k] is the position of recs[k] in that event slice, and i
// is the position of rng's own record.
func foldCorrections(i int, rng *session.EventCompaction, at []int, recs []*session.Event) (int, *session.EventCompaction) {
	earliest := i
	var extra []session.EventRef
	for k, other := range recs {
		o := other.Actions.Compaction
		if o == rng || !sameRange(o, rng) || !sameSummaryContent(o, rng) {
			continue
		}
		earliest = min(earliest, at[k])
		for _, ref := range o.ExcludedEvents {
			if !namesHole(rng, ref) {
				extra = append(extra, ref)
			}
		}
	}
	if len(extra) == 0 {
		return earliest, rng
	}
	merged := *rng
	merged.ExcludedEvents = append(slices.Clone(rng.ExcludedEvents), extra...)
	return earliest, &merged
}

// coversAfter reports whether a record later in the stream than i stands in for
// the event at index i.
//
// Only a record appearing later counts, matching coveredBy and therefore
// matching what prompt assembly actually drops. A summary never stands in for
// an event recorded after it was written, and an event tied to the previous
// range's end but appended afterwards is the case that makes the difference:
// prompt assembly keeps it, so selection has to offer it, or it ends up covered
// by the next range without ever having been summarized.
func (c coverIndex) coversAfter(i int, ev *session.Event) bool {
	for k, at := range c.at {
		if at <= i {
			continue
		}
		if inRange(ev, c.rngs[k]) {
			return true
		}
	}
	return false
}

// overlaps reports whether two compactions could stand in for any of the same
// events.
//
// Intersecting intervals is the answer, and deliberately the conservative one:
// two records whose spans meet may or may not share an event once exclusions
// are applied, and treating a maybe as an overlap costs one discarded summary
// where the opposite costs the same content materialized into a prompt twice.
func overlaps(a, b *session.EventCompaction) bool {
	if a == nil || b == nil {
		return false
	}
	return !a.StartTimestamp.After(b.EndTimestamp) && !b.StartTimestamp.After(a.EndTimestamp)
}

// HasUsableSummary reports whether a record carries content that can stand in
// for the events it covers.
//
// Deliberately only a nil check, matching adk-python, which tests
// compacted_content is not None and nothing more.
//
// The stricter check is tempting and was written and then reverted. A record
// whose content is empty, or holds only a thought or only an attachment,
// materializes as nothing while still suppressing every raw event in its range,
// so the covered turns are deleted and an empty line stands where they were.
// Requiring at least one prose part would keep them raw instead, which is
// visible rather than silent.
//
// Parity wins anyway. Nothing this package writes can be such a record, because
// newSummaryEvent refuses to build one, so the case needs a record from
// somewhere else: an older version, a plugin rewrite, a backend that dropped
// the parts. Diverging on prompt assembly to harden against that would put the
// two implementations in different places on ordinary content too, and this
// stack is meant to track the reference. If it is worth fixing it is worth
// fixing in both, so it belongs upstream first.
func HasUsableSummary(ev *session.Event) bool {
	return ev != nil && ev.Actions.Compaction != nil && ev.Actions.Compaction.CompactedContent != nil
}

// ReloadSession re-reads s from svc and returns the stored session.
//
// Compaction must not run against the session handle it was handed. That handle
// is a snapshot taken before the work started, so a concurrent invocation on the
// same session may have appended events it cannot see, and summarizing against
// it records a range covering those events without having summarized them. It
// may also be a wrapper that an agent installed over the real session, and every
// session service type-asserts on its own concrete type, so appending to a
// wrapper fails.
//
// Re-reading solves both: the result is current, and it is whatever concrete
// type the service issues.
func ReloadSession(ctx context.Context, svc session.Service, s session.Session) (session.Session, error) {
	if svc == nil || s == nil {
		return nil, fmt.Errorf("cannot re-read the session: no session service")
	}
	resp, err := svc.Get(ctx, &session.GetRequest{
		AppName:   s.AppName(),
		UserID:    s.UserID(),
		SessionID: s.ID(),
	})
	if err != nil {
		return nil, fmt.Errorf("failed to re-read the session: %w", err)
	}
	if resp == nil || resp.Session == nil {
		return nil, fmt.Errorf("session %q disappeared while compacting", s.ID())
	}
	return resp.Session, nil
}

// sessionUnwrapper is implemented by a [session.Session] that decorates another
// one. Nothing in the public API exposes it: the decorators are unexported types
// that happen to carry the method.
type sessionUnwrapper interface {
	Unwrap() session.Session
}

// maxUnwrapDepth caps how far [UnwrapSession] will follow a chain of decorators.
const maxUnwrapDepth = 32

// UnwrapSession returns the innermost session s decorates, or s itself.
//
// An agent may wrap the session it hands to a sub-agent so the sub-agent's
// prompt sees a synthetic first-turn seed. That wrapper is fine to read through
// but must not be compacted against: every session service type-asserts on its
// own concrete type, so appending to a wrapper fails outright, and the seed is
// not durable, so recording a range over it would cover an event no store holds.
//
// Unwrapping rather than re-reading is deliberate. It preserves object identity
// with the session the wrapper delegates to, so an event appended here is
// visible through the wrapper immediately. A freshly read session would be a
// different object, and the summary would not reach the prompt being assembled.
func UnwrapSession(s session.Session) session.Session {
	// The depth limit is not a real bound on nesting, which is one or two in
	// practice. It is there because Unwrap is reachable by any session with the
	// right method, including one outside this repository, and a wrapper that
	// returns itself would otherwise spin here for ever. Giving up returns the
	// last session seen, which is a session the caller can still use.
	for range maxUnwrapDepth {
		w, ok := s.(sessionUnwrapper)
		if !ok {
			return s
		}
		inner := w.Unwrap()
		if inner == nil {
			return s
		}
		s = inner
	}
	return s
}

// inRange reports whether rng covers ev.
//
// This is the only place that answers the question. Compaction has already
// grown three predicates for "is this a compaction", the weakest of which
// authorised deletion, and coverage is the one where a disagreement deletes
// conversation, so it gets exactly one definition.
//
// The range says what a summary stands in for and the exclusion list says which
// events inside it were left out, because window selection filters events out
// of the middle of its own span. A reference that names nothing excludes
// nothing, and that is the unsafe direction rather than the safe one: coverage
// is the range minus the exclusions, so an event whose hole stops matching
// becomes covered by a summary that never described it, and is dropped. A
// reference that names too much only leaves an extra event raw. Over-naming is
// the direction to prefer, and the producer errs that way deliberately.
func inRange(ev *session.Event, rng *session.EventCompaction) bool {
	if ev == nil || rng == nil {
		return false
	}
	if ev.Timestamp.Before(rng.StartTimestamp) || ev.Timestamp.After(rng.EndTimestamp) {
		return false
	}
	return !excludes(rng, ev)
}

// RepairAfterAppend returns a corrected record when the session gained an event
// inside a stored record's range after that record was built, or nil when it
// did not.
//
// The race guard runs immediately before the append and cannot cover the append
// itself: the session.Service interface has no conditional write, so between
// the last check and the store there is a window in which another writer can
// land an event. It does not take a hostile plugin or even a concurrent
// invocation to hit. An event carries the timestamp it was created at rather
// than stored at, so parallel tool responses and sub-agent events funnelled
// through a channel are routinely created before a range ends and stored after
// the check, and one that wins the race to append sits before the summary in
// the stream and inside its range, where the positional guard cannot help it.
//
// Prevention needs a primitive that does not exist, so this repairs instead.
// The corrected record carries the same content over the same range and names
// the straggler as a hole, and the identical-range rule in isCompactionSubsumed
// lets it evict the record it corrects. The straggler falls back to raw, which
// is what should have happened.
//
// known is the set of event identities present when the window was chosen, the
// same snapshot the race guard compares against.
//
// One round. A straggler landing during the repair would need another, but each
// round strictly shrinks what the record covers, and the window being closed is
// already the narrow one.
//
// Run the corrective write on the context from RepairContext.
func RepairAfterAppend(stored *session.Event, known map[string]struct{}, latest session.Session) *session.Event {
	if stored == nil || latest == nil {
		return nil
	}
	rng := stored.Actions.Compaction
	if rng == nil {
		return nil
	}

	var stragglers []session.EventRef
	seen := make(map[string]struct{})
	for _, ev := range collect(latest) {
		if ev == nil || hasCompaction(ev) {
			continue
		}
		if !inRange(ev, rng) || excludes(rng, ev) {
			continue
		}
		k := refKey(ev)
		if _, ok := known[k]; ok {
			// Present when the window was chosen, so it is what the summary was
			// built from rather than a straggler.
			continue
		}
		if _, ok := seen[k]; ok {
			continue
		}
		seen[k] = struct{}{}
		stragglers = append(stragglers, session.EventRef{InvocationID: ev.InvocationID, Timestamp: ev.Timestamp})
	}
	if len(stragglers) == 0 {
		return nil
	}

	corrected := *stored
	corrected.ID = ""
	// Strictly after the record it corrects, at the resolution storage keeps.
	//
	// Inheriting the timestamp made the two tie, and session/database breaks a
	// tie on id, which is a comparison of two freshly generated UUIDs: measured
	// against SQLite, the correction was read back first in ten of twenty
	// trials. Prompt assembly does not care, because holes are unioned across
	// records over one range, but the write side does. LatestCompactionEvent
	// picks exactly one, and when it picks the hole-less original the next
	// window treats the straggler as already summarized and leaves it out of
	// the next record's exclusions, which spans it with nothing describing it
	// and no same-range peer left to rescue it.
	corrected.Timestamp = stored.Timestamp.Add(refResolution)
	fixed := *rng
	fixed.ExcludedEvents = append(slices.Clone(rng.ExcludedEvents), stragglers...)
	corrected.Actions = stored.Actions
	corrected.Actions.Compaction = &fixed
	return &corrected
}

// sameSummaryContent reports whether two records carry the same summary text.
func sameSummaryContent(a, b *session.EventCompaction) bool {
	at := strings.Join(utils.TextParts(a.CompactedContent), "\n")
	bt := strings.Join(utils.TextParts(b.CompactedContent), "\n")
	return at != "" && at == bt
}
