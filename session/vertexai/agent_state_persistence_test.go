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

// No SessionEvent column holds an end-of-agent marker or an agent state, so
// they survive a reload only through raw_event. The marker events a resumable
// agent writes have no content and no deltas, so nothing else would carry
// them. An empty agent state is a value of its own: it marks an agent that
// started.
func TestVertexAiClient_ReloadedEventKeepsEndOfAgentAndAgentState(t *testing.T) {
	ended := true
	tests := []struct {
		name    string
		actions session.EventActions
	}{
		{name: "end of agent", actions: session.EventActions{EndOfAgent: &ended}},
		{name: "agent state", actions: session.EventActions{AgentState: map[string]any{"a": "b"}}},
		{name: "empty agent state", actions: session.EventActions{AgentState: map[string]any{}}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := t.Context()
			client := newFakeVertexAiClient(t, &fakeVertexAiSessionService{})

			ev := session.NewEvent(ctx, "inv-1")
			ev.Author = "root_agent"
			ev.Actions.EndOfAgent = tt.actions.EndOfAgent
			ev.Actions.AgentState = tt.actions.AgentState
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
			if diff := cmp.Diff(tt.actions.EndOfAgent, got[0].Actions.EndOfAgent); diff != "" {
				t.Errorf("reloaded EndOfAgent mismatch (-want +got):\n%s", diff)
			}
			if diff := cmp.Diff(tt.actions.AgentState, got[0].Actions.AgentState); diff != "" {
				t.Errorf("reloaded AgentState mismatch (-want +got):\n%s", diff)
			}
		})
	}
}
