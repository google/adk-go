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

package utils

import (
	"slices"

	"google.golang.org/adk/v2/session"
)

// ApplyRewinds returns the live events of a session: events, in their
// original order, without the events that rewinds removed. Every code path
// that reads a session's history to decide what happens next, such as LLM
// contents assembly, must read it through this function, so that they all
// agree on what a rewind removed.
//
// events must be the session's events in chronological order. A rewind event
// is one whose Actions.RewindBeforeInvocationID is non-empty. The events are
// walked from the last to the first; a rewind event is dropped together with
// every event from the first event of its target invocation up to it, and the
// walk resumes before that first event, so a rewind inside a rewound span is
// never applied. A rewind whose target invocation has no event before it
// (unknown, only later, or the rewind's own invocation) drops only itself.
//
// ApplyRewinds does not modify events or the events it points to; the
// returned slice is new and shares the event pointers. It ports adk-python's
// _apply_rewinds (events/_rewind_events.py).
func ApplyRewinds(events []*session.Event) []*session.Event {
	var firstIndex map[string]int
	var live []*session.Event
	for i := len(events) - 1; i >= 0; i-- {
		target := events[i].Actions.RewindBeforeInvocationID
		if target == "" {
			live = append(live, events[i])
			continue
		}
		if firstIndex == nil {
			firstIndex = firstIndexByInvocation(events)
		}
		if start, ok := firstIndex[target]; ok && start < i {
			i = start
		}
	}
	slices.Reverse(live)
	return live
}

// firstIndexByInvocation maps each invocation ID to the index of its first
// event in events.
func firstIndexByInvocation(events []*session.Event) map[string]int {
	first := map[string]int{}
	for i, ev := range events {
		if _, seen := first[ev.InvocationID]; !seen {
			first[ev.InvocationID] = i
		}
	}
	return first
}
