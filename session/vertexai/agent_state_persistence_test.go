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

	"github.com/google/go-cmp/cmp"

	"google.golang.org/adk/v2/session"
)

// SessionEvent has no column for agent_state or end_of_agent, so raw_event is
// the only place they can be stored, as adk-python does. These run offline
// because the replay-based suite needs a recording from a live Agent Engine.
func TestAgentStateRoundTripsThroughRawEvent(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		actions session.EventActions
	}{
		{name: "empty agent state", actions: session.EventActions{AgentState: map[string]any{}}},
		{name: "agent state", actions: session.EventActions{AgentState: map[string]any{"current_sub_agent": "b"}}},
		{name: "end of agent", actions: session.EventActions{EndOfAgent: true}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			event := &session.Event{ID: "e", Author: "seq", InvocationID: "inv1", Actions: tc.actions}
			if !eventNeedsRawEvent(event) {
				t.Fatal("eventNeedsRawEvent() = false, so the checkpoint would be written nowhere and lost on reload")
			}

			raw, err := eventToRawEvent(event)
			if err != nil {
				t.Fatalf("eventToRawEvent() error = %v", err)
			}
			got, err := eventFromRawEvent(raw)
			if err != nil {
				t.Fatalf("eventFromRawEvent() error = %v", err)
			}
			if (got.Actions.AgentState == nil) != (tc.actions.AgentState == nil) {
				t.Errorf("AgentState nil = %v, want %v", got.Actions.AgentState == nil, tc.actions.AgentState == nil)
			}
			if diff := cmp.Diff(tc.actions.AgentState, got.Actions.AgentState); diff != "" {
				t.Errorf("AgentState mismatch (-want +got):\n%s", diff)
			}
			if got.Actions.EndOfAgent != tc.actions.EndOfAgent {
				t.Errorf("EndOfAgent = %v, want %v", got.Actions.EndOfAgent, tc.actions.EndOfAgent)
			}
		})
	}
}
