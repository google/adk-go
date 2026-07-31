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
	"time"

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
func longestSelfContainedPrefix(events []*session.Event) []*session.Event {
	openIDs := make(map[string]struct{})
	safeLength := 0
	for i, ev := range events {
		for _, resp := range utils.FunctionResponses(utils.Content(ev)) {
			delete(openIDs, resp.ID)
		}
		for _, call := range utils.FunctionCalls(utils.Content(ev)) {
			if call.ID != "" {
				openIDs[call.ID] = struct{}{}
			}
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
	return events[:safeLength]
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
// contained by another compaction in events. Identical ranges are broken by
// stream position: the earlier event is subsumed by the later one.
func isCompactionSubsumed(i int, rng *session.EventCompaction, events []*session.Event) bool {
	for j, other := range events {
		if j == i || !hasCompaction(other) {
			continue
		}
		o := other.Actions.Compaction
		if o.StartTimestamp.After(rng.StartTimestamp) || o.EndTimestamp.Before(rng.EndTimestamp) {
			continue
		}
		if o.StartTimestamp.Before(rng.StartTimestamp) || o.EndTimestamp.After(rng.EndTimestamp) || j > i {
			return true
		}
	}
	return false
}

// selectSlidingWindow returns the events a sliding-window compaction should
// summarize, or nil when there is nothing to compact yet.
//
// It walks events newest to oldest, accumulating distinct invocation IDs of
// non-compaction events. Once it crosses the most recent compaction boundary
// with at least interval new invocations behind it, it keeps going for up to
// overlap further invocations so consecutive summaries share context, then
// stops. The window is returned in chronological order, trimmed by
// longestSelfContainedPrefix.
//
// nil comes back when fewer than interval new invocations exist, or when the
// selected window has no self-contained prefix.
func selectSlidingWindow(events []*session.Event, interval, overlap int) []*session.Event {
	if interval <= 0 {
		return nil
	}

	var window []*session.Event
	seen := make(map[string]struct{})
	var lastCompactEnd time.Time
	targetSize := -1

	for i := len(events) - 1; i >= 0; i-- {
		ev := events[i]

		// hasCompaction, not IsCompactionEvent: an event that declares a
		// compaction but carries no usable content is still bookkeeping, never
		// conversation. Counting it as a real invocation would skew the window.
		if hasCompaction(ev) {
			if end := ev.Actions.Compaction.EndTimestamp; end.After(lastCompactEnd) {
				lastCompactEnd = end
			}
			continue
		}
		if ev.InvocationID == "" {
			continue
		}
		if _, ok := seen[ev.InvocationID]; ok {
			window = append(window, ev)
			continue
		}

		// Crossing the most recent compaction boundary. Either enough new
		// invocations have accumulated and we keep going for `overlap` more, or
		// they have not and there is nothing to do.
		if !ev.Timestamp.After(lastCompactEnd) {
			if len(seen) < interval {
				break
			}
			if targetSize < 0 {
				targetSize = len(seen) + overlap
			}
		}
		if targetSize >= 0 && len(seen) >= targetSize {
			break
		}
		window = append(window, ev)
		seen[ev.InvocationID] = struct{}{}
	}

	if len(seen) < interval {
		return nil
	}
	slices.Reverse(window)
	window = longestSelfContainedPrefix(window)
	if len(window) == 0 {
		return nil
	}
	return window
}
