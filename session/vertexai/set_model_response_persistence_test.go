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

// No SessionEvent column holds Actions.SetModelResponse, so an event without
// NodeInfo (such as a RunLive set_model_response function response) keeps it
// across a reload only when Actions.SetModelResponse triggers raw_event.
func TestVertexAiClient_ReloadedSetModelResponseWithoutNodeInfo(t *testing.T) {
	ctx := t.Context()
	client := newFakeVertexAiClient(t, &fakeVertexAiSessionService{})

	ev := session.NewEvent(ctx, "e-smr")
	ev.Author = "agent"
	ev.Actions.SetModelResponse = map[string]any{"answer": "ok"}
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
	want := map[string]any{"answer": "ok"}
	if diff := cmp.Diff(want, got[0].Actions.SetModelResponse); diff != "" {
		t.Errorf("reloaded Actions.SetModelResponse mismatch (-want +got):\n%s", diff)
	}
}
