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

package sequentialagent_test

import (
	"testing"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/workflowagents/sequentialagent"
	"google.golang.org/adk/v2/runner"
)

// TestSequentialAgent_RunLive_NoSubAgentsEndsWithNoEvent checks that the live
// run of a SequentialAgent with no sub-agents ends with no event and no error,
// and that its session closes, as adk-python's _run_live_impl returns when
// self.sub_agents is empty (agents/sequential_agent.py).
func TestSequentialAgent_RunLive_NoSubAgentsEndsWithNoEvent(t *testing.T) {
	seq, err := sequentialagent.New(sequentialagent.Config{AgentConfig: agent.Config{Name: "seq"}})
	if err != nil {
		t.Fatal(err)
	}
	r, err := runner.NewInMemory("app", seq)
	if err != nil {
		t.Fatal(err)
	}

	sess, stream, err := r.RunLive(t.Context(), "u", "s", agent.LiveRunConfig{})
	if err != nil {
		t.Fatalf("RunLive() error = %v, want nil", err)
	}
	var authors []string
	for ev, err := range stream {
		if err != nil {
			t.Fatalf("stream error = %v, want none", err)
		}
		authors = append(authors, ev.Author)
	}
	if len(authors) != 0 {
		t.Errorf("stream yielded events from %q, want none", authors)
	}
	if err := sess.Close(); err != nil {
		t.Errorf("Close() error = %v, want nil", err)
	}
}
