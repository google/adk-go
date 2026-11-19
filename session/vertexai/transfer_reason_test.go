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

package vertexai

import (
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"

	"google.golang.org/adk/v2/session"
)

// The aiplatform EventActions message has no field for a transfer reason, so
// an event with one must be written with raw_event or the reason is lost on
// reload. An event without one keeps the legacy wire format.
func TestEventNeedsRawEvent_TransferReason(t *testing.T) {
	tests := []struct {
		name   string
		reason string
		want   bool
	}{
		{name: "reason set", reason: "r", want: true},
		{name: "reason empty", reason: "", want: false},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			event := &session.Event{Actions: session.EventActions{TransferReason: tc.reason}}
			if got := eventNeedsRawEvent(event); got != tc.want {
				t.Errorf("eventNeedsRawEvent() = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestVertexAiClient_ReloadedTransferEventKeepsReason(t *testing.T) {
	ctx := t.Context()
	client := newFakeVertexAiClient(t, &fakeVertexAiSessionService{})

	ev := session.NewEvent(ctx, "e-transfer")
	ev.Author = "agent"
	ev.Actions.TransferToAgent = "b"
	ev.Actions.TransferReason = "r"
	ev.Actions.StateDelta = map[string]any{"k": "v"}
	if err := client.appendEvent(ctx, "test-app", "session-1", ev); err != nil {
		t.Fatalf("appendEvent() failed: %v", err)
	}

	got, err := client.listSessionEvents(ctx, "test-app", "session-1", time.Time{}, 0)
	if err != nil {
		t.Fatalf("listSessionEvents() failed: %v", err)
	}
	if len(got) != 1 {
		t.Fatalf("len(got) = %d, want 1", len(got))
	}
	actions := got[0].Actions
	if actions.TransferToAgent != "b" || actions.TransferReason != "r" {
		t.Errorf("reloaded (TransferToAgent, TransferReason) = (%q, %q), want (\"b\", \"r\")", actions.TransferToAgent, actions.TransferReason)
	}
	if diff := cmp.Diff(map[string]any{"k": "v"}, actions.StateDelta); diff != "" {
		t.Errorf("reloaded StateDelta mismatch (-want +got):\n%s", diff)
	}
}
