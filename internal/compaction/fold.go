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

package compaction

import (
	"google.golang.org/adk/session"
)

// Fold replaces compacted older events with their summary content, leaving
// newer raw events intact.
//
// Behavior:
//  1. If no valid compaction event exists, returns events unchanged.
//  2. Otherwise, finds the latest non-subsumed compaction.
//  3. Returns a slice that contains: a synthetic event carrying the
//     compaction's summary content (Author "model", Timestamp =
//     compaction.StartTimestamp), followed by every event whose Timestamp
//     is strictly after compaction.EndTimestamp.
//
// Events with Actions.Compaction set are dropped from the output (the
// summary supersedes them).
//
// The contents-builder calls this before assembling LLM contents from
// session events, so the model sees a single compacted turn instead of the
// raw older history.
func Fold(events []*session.Event) []*session.Event {
	latest := latestCompactionEvent(events)
	if latest == nil || latest.Actions.Compaction == nil {
		// No compaction; return original slice.
		return events
	}
	end := latest.Actions.Compaction.EndTimestamp
	summary := latest.Actions.Compaction.CompactedContent
	if summary == nil {
		return events
	}

	out := make([]*session.Event, 0, len(events))
	// Synthetic summary event takes the place of all subsumed events.
	seed := session.NewEvent(latest.InvocationID + ":compacted")
	seed.Author = "model"
	seed.Branch = latest.Branch
	seed.Timestamp = latest.Actions.Compaction.StartTimestamp
	seed.Content = summary
	out = append(out, seed)

	// Pass through events strictly newer than the compaction's end timestamp,
	// dropping any compaction-marker events themselves.
	for _, ev := range events {
		if ev.Actions.Compaction != nil {
			continue
		}
		if ev.Timestamp.After(end) {
			out = append(out, ev)
		}
	}
	return out
}
