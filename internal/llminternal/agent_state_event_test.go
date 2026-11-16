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

package llminternal_test

import (
	"testing"

	"github.com/google/go-cmp/cmp"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/internal/agent/runconfig"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/internal/llminternal"
)

// stateEventContext returns an invocation context with invocation ID "inv-1"
// and branch "b1" whose run config is rc.
func stateEventContext(t *testing.T, rc *runconfig.RunConfig) agent.InvocationContext {
	t.Helper()
	return icontext.NewInvocationContext(runconfig.ToContext(t.Context(), rc), icontext.InvocationContextParams{
		InvocationID: "inv-1",
		Branch:       "b1",
	})
}

func TestNewAgentStateEvent_And_NewEndOfAgentEventRecordsEnd(t *testing.T) {
	t.Run("state event with a nil state", func(t *testing.T) {
		rc := &runconfig.RunConfig{Resumable: true}
		ev := llminternal.NewAgentStateEvent(stateEventContext(t, rc), "seq", nil)

		if ev.Author != "seq" || ev.InvocationID != "inv-1" || ev.Branch != "b1" {
			t.Errorf("event author, invocation ID, branch = %q, %q, %q; want %q, %q, %q",
				ev.Author, ev.InvocationID, ev.Branch, "seq", "inv-1", "b1")
		}
		if ev.Content != nil {
			t.Errorf("event Content = %v; want nil", ev.Content)
		}
		if ev.Actions.AgentState == nil || len(ev.Actions.AgentState) != 0 {
			t.Errorf("event Actions.AgentState = %#v; want a non-nil empty map", ev.Actions.AgentState)
		}
		if ev.Actions.EndOfAgent != nil {
			t.Errorf("event Actions.EndOfAgent = %v; want nil", *ev.Actions.EndOfAgent)
		}
		if rc.AgentEnded("seq") {
			t.Error("AgentEnded(seq) = true; want false")
		}
	})

	t.Run("state event keeps the state and records nothing", func(t *testing.T) {
		rc := &runconfig.RunConfig{Resumable: true}
		rc.SeedAgentRuns(map[string]map[string]any{"seq": {"current_sub_agent": "a1"}}, nil)
		ev := llminternal.NewAgentStateEvent(stateEventContext(t, rc), "seq", map[string]any{"current_sub_agent": "a2"})

		if diff := cmp.Diff(map[string]any{"current_sub_agent": "a2"}, ev.Actions.AgentState); diff != "" {
			t.Errorf("event Actions.AgentState mismatch (-want +got):\n%s", diff)
		}
		st, ok := rc.AgentState("seq")
		if diff := cmp.Diff(map[string]any{"current_sub_agent": "a1"}, st); diff != "" || !ok {
			t.Errorf("AgentState(seq) ok = %v, mismatch (-want +got):\n%s", ok, diff)
		}
	})

	t.Run("end event records the end", func(t *testing.T) {
		rc := &runconfig.RunConfig{Resumable: true}
		rc.SeedAgentRuns(map[string]map[string]any{"seq": {"current_sub_agent": "a1"}}, nil)
		ev := llminternal.NewEndOfAgentEvent(stateEventContext(t, rc), "seq")

		if ev.Actions.EndOfAgent == nil || !*ev.Actions.EndOfAgent {
			t.Errorf("event Actions.EndOfAgent = %v; want true", ev.Actions.EndOfAgent)
		}
		if ev.Actions.AgentState != nil {
			t.Errorf("event Actions.AgentState = %#v; want nil", ev.Actions.AgentState)
		}
		if !rc.AgentEnded("seq") {
			t.Error("AgentEnded(seq) = false; want true")
		}
		if st, ok := rc.AgentState("seq"); st != nil || ok {
			t.Errorf("AgentState(seq) = %v, %v; want nil, false", st, ok)
		}
	})
}
