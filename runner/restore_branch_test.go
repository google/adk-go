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

package runner

import (
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
)

// TestRunner_RestoreBranchFromHistory_ToolBranchAndRewinds pins the two rules
// of adk-python's restore_branch_from_history (agents/_agent_router.py) that
// decide which of the agent's events gives its branch: a tool's message
// branch is skipped, and rewound events, with the function calls they hold,
// do not count.
func TestRunner_RestoreBranchFromHistory_ToolBranchAndRewinds(t *testing.T) {
	t.Parallel()

	worker := must(agent.New(agent.Config{Name: "worker"}))
	root := must(agent.New(agent.Config{Name: "coordinator", SubAgents: []agent.Agent{worker}}))
	r, err := New(Config{AppName: "app", Agent: root, SessionService: session.InMemoryService()})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	ev := func(author, branch, invocationID string, part *genai.Part) *session.Event {
		return &session.Event{
			Author:       author,
			Branch:       branch,
			InvocationID: invocationID,
			LLMResponse:  model.LLMResponse{Content: &genai.Content{Role: "model", Parts: []*genai.Part{part}}},
		}
	}
	call := func(author, branch, invocationID, id string) *session.Event {
		return ev(author, branch, invocationID, &genai.Part{FunctionCall: &genai.FunctionCall{ID: id, Name: "do"}})
	}
	text := func(author, branch, invocationID string) *session.Event {
		return ev(author, branch, invocationID, &genai.Part{Text: "t"})
	}
	rewind := func(invocationID, before string) *session.Event {
		e := &session.Event{Author: "user", InvocationID: invocationID}
		e.Actions.RewindBeforeInvocationID = before
		return e
	}

	tests := []struct {
		name         string
		events       []*session.Event
		invocationID string
		want         string
	}{
		{
			name: "tool_branch_skipped_for_older_agent_branch",
			events: []*session.Event{
				call("worker", "worker@1", "inv_1", "fc-1"),
				text("worker", "do@fc-1", "inv_1"),
			},
			invocationID: "inv_1",
			want:         "worker@1",
		},
		{
			name: "only_tool_branches",
			events: []*session.Event{
				call("coordinator", "", "inv_1", "fc-1"),
				text("worker", "do@fc-1", "inv_1"),
			},
			want: "",
		},
		{
			name: "rewound_branch_skipped",
			events: []*session.Event{
				text("worker", "root@1.worker@1", "inv_1"),
				text("worker", "root@1.worker@2", "inv_2"),
				rewind("inv_3", "inv_2"),
			},
			want: "root@1.worker@1",
		},
		{
			name: "rewound_call_makes_no_tool_branch",
			events: []*session.Event{
				text("worker", "do@fc-rewound", "inv_1"),
				call("coordinator", "", "inv_2", "fc-rewound"),
				rewind("inv_3", "inv_2"),
			},
			want: "do@fc-rewound",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			sess := buildSessionWithEvents(t, tc.events)
			if got := r.restoreBranchFromHistory(sess, worker, tc.invocationID); got != tc.want {
				t.Errorf("restoreBranchFromHistory(worker, %q) = %q, want %q", tc.invocationID, got, tc.want)
			}
		})
	}
}
