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

	"google.golang.org/adk/v2/session"
)

// No SessionEvent column holds the live session id, so an event whose only
// other field is that id must be written with raw_event or the id reads back
// empty. adk-python keeps it in raw_event too
// (sessions/vertex_ai_session_service.py:491-498). The test fails if
// eventNeedsRawEvent does not count the id.
func TestVertexAiClientLiveSessionIDRoundTrip(t *testing.T) {
	tests := []struct {
		name          string
		liveSessionID string
	}{
		{name: "with live session id", liveSessionID: "live-1"},
		{name: "without live session id", liveSessionID: ""},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ctx := t.Context()
			service := &fakeVertexAiSessionService{}
			client := newFakeVertexAiClient(t, service)

			event := session.NewEvent(ctx, "invocation-1")
			event.Author = "agent"
			event.LiveSessionID = tc.liveSessionID

			if err := client.appendEvent(ctx, "test-app", "session-1", event); err != nil {
				t.Fatalf("appendEvent() failed: %v", err)
			}
			got, err := client.listSessionEvents(ctx, "test-app", "session-1", time.Time{}, 0)
			if err != nil {
				t.Fatalf("listSessionEvents() failed: %v", err)
			}
			if gotLen := len(got); gotLen != 1 {
				t.Fatalf("len(got) = %d, want 1", gotLen)
			}
			if got[0].LiveSessionID != tc.liveSessionID {
				t.Errorf("LiveSessionID = %q, want %q", got[0].LiveSessionID, tc.liveSessionID)
			}
		})
	}
}
