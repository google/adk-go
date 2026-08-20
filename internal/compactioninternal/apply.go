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

// keptRange is a compaction range that survived subsumption, along with the
// stream position of the event that declared it.
type keptRange struct {
	index int
	rng   *session.EventCompaction
}

// substituteSummaries drops raw events covered by a surviving compaction and
// materializes each surviving summary in their place, preserving chronological
// order.
func substituteSummaries(events []*session.Event) []*session.Event {
	var kept []keptRange
	for i, ev := range events {
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
		if isCompactionSubsumed(i, ev.Actions.Compaction, events) {
			continue
		}
		kept = append(kept, keptRange{index: i, rng: ev.Actions.Compaction})
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
			summary := *events[k.index]
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
	rng := summary.Actions.Compaction
	if latest == nil || rng == nil {
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

// coveredByAny reports whether any compaction in events stands in for the event
// at index i.
//
// Only a compaction appearing later in the stream counts, matching coveredBy
// and therefore matching what prompt assembly actually drops. A summary never
// stands in for an event recorded after it was written, and an event tied to
// the previous range's end but appended afterwards is the case that makes the
// difference: prompt assembly keeps it, so selection has to offer it, or it is
// covered by the next range without ever having been summarized.
func coveredByAny(i int, ev *session.Event, events []*session.Event) bool {
	for j, other := range events {
		if j <= i || !hasCompaction(other) {
			continue
		}
		if inRange(ev, other.Actions.Compaction) {
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

// HasUsableSummary reports whether ev carries a compaction summary that can
// actually be shown to a model: it declares a compaction, and that compaction
// has content.
//
// Distinct from hasCompaction, which asks whether the event is bookkeeping at
// all. An event declaring a compaction with no content is still bookkeeping and
// must never be treated as conversation, but it has no summary to materialize.
// Conflating the two let a contentless record evict a real summary and, worse,
// authorise deleting the events it claimed to cover.
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
