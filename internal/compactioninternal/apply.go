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

	"google.golang.org/adk/v2/internal/utils"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/session/compaction"
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
// Apply keys off this rather than [IsCompactionEvent] so that a malformed
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
		if !compaction.IsCompactionEvent(ev) {
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

	// Position each event by (timestamp, original index) so summaries slot in
	// among the raw events they neighbour rather than all bunching at the end.
	type positioned struct {
		index int
		event *session.Event
	}
	out := make([]positioned, 0, len(events))

	for _, k := range kept {
		summary := *events[k.index]
		summary.Author = "model"
		summary.Timestamp = k.rng.EndTimestamp
		summary.LLMResponse.Content = k.rng.CompactedContent
		out = append(out, positioned{index: k.index, event: &summary})
	}

	for i, ev := range events {
		// Any event declaring a compaction is handled above, or was dropped as
		// subsumed or unusable. Never re-emit it as a raw event: its content
		// slot holds no conversation, only bookkeeping.
		if hasCompaction(ev) {
			continue
		}
		if isCovered(i, ev, kept) {
			continue
		}
		out = append(out, positioned{index: i, event: ev})
	}

	slices.SortStableFunc(out, func(a, b positioned) int {
		if c := a.event.Timestamp.Compare(b.event.Timestamp); c != 0 {
			return c
		}
		return a.index - b.index
	})

	result := make([]*session.Event, len(out))
	for i, p := range out {
		result[i] = p.event
	}
	return result
}

// isCovered reports whether the raw event at index i falls inside a surviving
// compaction range. Only a compaction appearing later in the stream can cover
// an event: a summary never covers events recorded after it was written.
func isCovered(i int, ev *session.Event, kept []keptRange) bool {
	for _, k := range kept {
		if i >= k.index {
			continue
		}
		if !ev.Timestamp.Before(k.rng.StartTimestamp) && !ev.Timestamp.After(k.rng.EndTimestamp) {
			return true
		}
	}
	return false
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
