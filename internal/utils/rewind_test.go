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

package utils_test

import (
	"testing"

	"github.com/google/go-cmp/cmp"

	"google.golang.org/adk/v2/internal/utils"
	"google.golang.org/adk/v2/session"
)

// ev returns an event with the given ID and invocation ID. A non-empty
// rewindTarget makes it a rewind event.
func ev(id, invocationID, rewindTarget string) *session.Event {
	return &session.Event{
		ID:           id,
		InvocationID: invocationID,
		Actions:      session.EventActions{RewindBeforeInvocationID: rewindTarget},
	}
}

func TestApplyRewinds(t *testing.T) {
	tests := []struct {
		name   string
		events []*session.Event
		want   []string
	}{
		{
			name:   "nil",
			events: nil,
			want:   []string{},
		},
		{
			name:   "no rewind",
			events: []*session.Event{ev("e1", "inv1", ""), ev("e2", "inv2", ""), ev("e3", "inv3", "")},
			want:   []string{"e1", "e2", "e3"},
		},
		{
			name: "drops from the first event of the target invocation",
			events: []*session.Event{
				ev("e1", "inv1", ""), ev("e2a", "inv2", ""), ev("e2b", "inv2", ""), ev("e3", "inv3", ""),
				ev("r1", "r1", "inv2"), ev("e4", "inv4", ""),
			},
			want: []string{"e1", "e4"},
		},
		{
			name:   "rewind before the first invocation",
			events: []*session.Event{ev("e1", "inv1", ""), ev("e2", "inv2", ""), ev("r1", "r1", "inv1"), ev("e3", "inv3", "")},
			want:   []string{"e3"},
		},
		{
			name:   "unknown target drops only the rewind",
			events: []*session.Event{ev("e1", "inv1", ""), ev("r1", "r1", "missing"), ev("e2", "inv2", "")},
			want:   []string{"e1", "e2"},
		},
		{
			name:   "later target drops only the rewind",
			events: []*session.Event{ev("e1", "inv1", ""), ev("r1", "r1", "inv2"), ev("e2", "inv2", "")},
			want:   []string{"e1", "e2"},
		},
		{
			name:   "self target drops only the rewind",
			events: []*session.Event{ev("e1", "inv1", ""), ev("r1", "r1", "r1"), ev("e2", "inv2", "")},
			want:   []string{"e1", "e2"},
		},
		{
			name: "sequential rewinds",
			events: []*session.Event{
				ev("e1", "inv1", ""), ev("e2", "inv2", ""), ev("r1", "r1", "inv2"),
				ev("e3", "inv3", ""), ev("e4", "inv4", ""), ev("r2", "r2", "inv4"),
			},
			want: []string{"e1", "e3"},
		},
		{
			name: "repeated rewind to the same target",
			events: []*session.Event{
				ev("e1", "inv1", ""), ev("e2", "inv2", ""), ev("r1", "r1", "inv2"),
				ev("e3", "inv3", ""), ev("r2", "r2", "inv2"), ev("e4", "inv4", ""),
			},
			want: []string{"e1", "e4"},
		},
		{
			name: "later rewind to an earlier target covers an earlier rewind",
			events: []*session.Event{
				ev("e1", "inv1", ""), ev("e2", "inv2", ""), ev("r1", "r1", "inv2"),
				ev("e3", "inv3", ""), ev("r2", "r2", "inv1"), ev("e4", "inv4", ""),
			},
			want: []string{"e4"},
		},
		{
			// The walk resumes before inv2, so r1, which would have dropped
			// inv1, is itself dropped and never applied.
			name: "rewound rewind is not applied",
			events: []*session.Event{
				ev("e1", "inv1", ""), ev("e2", "inv2", ""), ev("r1", "r1", "inv1"),
				ev("r2", "r2", "inv2"), ev("e5", "inv5", ""),
			},
			want: []string{"e1", "e5"},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := []string{}
			for _, e := range utils.ApplyRewinds(tc.events) {
				got = append(got, e.ID)
			}
			if diff := cmp.Diff(tc.want, got); diff != "" {
				t.Errorf("ApplyRewinds() event IDs mismatch (-want +got):\n%s", diff)
			}
		})
	}
}
