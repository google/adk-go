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
	"testing"

	"google.golang.org/adk/v2/session"
)

// These measure the two functions that run against the whole session, which is
// the shape that matters: events are never deleted, so a long-lived session
// only grows, and tail retention runs before every model call rather than once
// a turn.
//
// Window selection used to ask "is this event already covered" by rescanning
// the session for each event, which is quadratic in session length. Measured
// before indexing the records once: 1,000 events 3.3ms, 4,000 42.8ms, 8,000
// 214ms, 16,000 1.18s, with 78% of the time in that scan. After: 78µs, 333µs,
// 972µs, 2.9ms. Keep an eye on the shape of the curve rather than the absolute
// numbers, which depend on the machine.
func benchEvents(n int) []*session.Event {
	evs := make([]*session.Event, 0, n)
	for i := range n {
		evs = append(evs, textEvent(fmt.Sprintf("e%d", i), fmt.Sprintf("inv%d", i/2), i+1, "text"))
	}
	return evs
}

func BenchmarkSelectTailRetentionWindow(b *testing.B) {
	for _, n := range []int{1000, 4000, 8000, 16000} {
		evs := benchEvents(n)
		b.Run(fmt.Sprint(n), func(b *testing.B) {
			for b.Loop() {
				_ = selectTailRetentionWindow(evs, 10, TurnScope{})
			}
		})
	}
}

func BenchmarkApply(b *testing.B) {
	for _, n := range []int{1000, 4000, 8000, 16000} {
		evs := benchEvents(n)
		b.Run(fmt.Sprint(n), func(b *testing.B) {
			for b.Loop() {
				_ = Apply(evs)
			}
		})
	}
}
