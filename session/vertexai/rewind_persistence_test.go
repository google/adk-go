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

// No SessionEvent column holds a rewind's target, so it survives a reload only
// through raw_event. A rewind event has no content and often empty deltas, so
// nothing else would mark it as a rewind after the reload.
func TestVertexAiClient_ReloadedRewindEventKeepsTarget(t *testing.T) {
	ctx := t.Context()
	client := newFakeVertexAiClient(t, &fakeVertexAiSessionService{})

	ev := session.NewEvent(ctx, "e-rewind")
	ev.Author = "user"
	ev.Actions.RewindBeforeInvocationID = "inv2"
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
	if target := got[0].Actions.RewindBeforeInvocationID; target != "inv2" {
		t.Errorf("reloaded RewindBeforeInvocationID = %q, want \"inv2\"", target)
	}
}
