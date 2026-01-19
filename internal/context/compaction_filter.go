// Copyright 2025 Google LLC
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

package context

import (
	"iter"

	"google.golang.org/adk/session"
)

// FilterEventsForLLM returns an iterator that filters session events for LLM context.
// It implements the critical filtering logic from ADR-010:
//
// 1. Includes all compaction events (they contain summaries)
// 2. Excludes original events that fall within compaction ranges
// 3. Includes all non-compacted events
//
// The result is that LLM context contains only:
// - Compaction summaries (60-80% token reduction)
// - Events after the most recent compaction
//
// This maintains full conversation context while dramatically reducing token usage.
func FilterEventsForLLM(events session.Events) iter.Seq[*session.Event] {
	return func(yield func(*session.Event) bool) {
		if events == nil || events.Len() == 0 {
			return
		}

		// First pass: collect all events and identify compaction ranges
		allEvents := make([]*session.Event, 0, events.Len())
		compactionRanges := make([]*CompactionRange, 0)

		for i := 0; i < events.Len(); i++ {
			event := events.At(i)
			allEvents = append(allEvents, event)

			// Track compaction events and their ranges
			if event.Actions.Compaction != nil {
				compactionRanges = append(compactionRanges, &CompactionRange{
					StartTimestamp: event.Actions.Compaction.StartTimestamp,
					EndTimestamp:   event.Actions.Compaction.EndTimestamp,
				})
			}
		}

		// Second pass: filter events
		for _, event := range allEvents {
			// Always include compaction events themselves
			if event.Actions.Compaction != nil {
				if !yield(event) {
					return
				}
				continue
			}

			// Check if this event is within any compaction range
			if isEventWithinCompactionRange(event, compactionRanges) {
				// Skip this event - it's been compacted
				continue
			}

			// Include non-compacted events
			if !yield(event) {
				return
			}
		}
	}
}

// CompactionRange represents the time window of a compaction event.
type CompactionRange struct {
	StartTimestamp float64
	EndTimestamp   float64
}

// isEventWithinCompactionRange checks if an event's timestamp falls within
// any of the compaction ranges.
func isEventWithinCompactionRange(event *session.Event, ranges []*CompactionRange) bool {
	eventTime := float64(event.Timestamp.Unix())

	for _, r := range ranges {
		if eventTime >= r.StartTimestamp && eventTime <= r.EndTimestamp {
			return true
		}
	}

	return false
}
