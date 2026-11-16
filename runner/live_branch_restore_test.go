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
	"iter"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/internal/llminternal"
	"google.golang.org/adk/v2/session"
)

// llmState names the embedded llminternal.Agent of liveLLMAgent apart from
// its embedded agent.Agent.
type llmState = llminternal.Agent

// liveLLMAgent is an LLM agent, so the router treats it as one, whose live
// run records the branch of its invocation and yields one event on it.
type liveLLMAgent struct {
	agent.Agent
	llmState
	gotBranch *string
}

// FindAgent returns a itself for its own name, so the router runs a rather
// than the agent it wraps.
func (a *liveLLMAgent) FindAgent(name string) agent.Agent {
	if name == a.Name() {
		return a
	}
	return a.Agent.FindAgent(name)
}

func (a *liveLLMAgent) RunLive(ctx agent.InvocationContext) (agent.LiveSession, iter.Seq2[*session.Event, error], error) {
	*a.gotBranch = ctx.Branch()
	return &dummyLiveSession{}, func(yield func(*session.Event, error) bool) {
		ev := session.NewEvent(ctx, ctx.InvocationID())
		ev.Author = a.Name()
		ev.Branch = ctx.Branch()
		ev.Content = genai.NewContentFromText("live reply", genai.RoleModel)
		yield(ev, nil)
	}, nil
}

// TestRunner_RunLive_RestoresSubAgentBranchFromHistory pins that a live run
// routed to a sub-agent continues on the branch of the sub-agent's newest
// live event that has one, passing over a tool's message branch and rewound
// invocations, and on the root branch when every such event was rewound
// (adk-python live/_runner_utils.py:254-260 with restore_branch_from_history,
// agents/_agent_router.py; test_agent_router.py:650).
func TestRunner_RunLive_RestoresSubAgentBranchFromHistory(t *testing.T) {
	t.Parallel()
	rewind := func(invocationID, before string) *session.Event {
		ev := &session.Event{Author: "user", InvocationID: invocationID}
		ev.Actions.RewindBeforeInvocationID = before
		return ev
	}
	tests := []struct {
		name   string
		events []*session.Event
		want   string
	}{
		{
			name: "restores_latest_live_non_tool_branch",
			events: []*session.Event{
				routingNamedCall("sub_agent1", "root@1.sub_agent1@1", "inv_1", "fc_1", "my_tool"),
				routingText("sub_agent1", "my_tool@fc_1", "inv_1", "tool message"),
				routingText("sub_agent1", "root@1.sub_agent1@2", "inv_2", "rewound"),
				rewind("inv_3", "inv_2"),
			},
			want: "root@1.sub_agent1@1",
		},
		{
			// sub_agent1's event of inv_3 routes the run to it, but has no
			// branch.
			name: "all_rewound_leaves_branch_empty",
			events: []*session.Event{
				routingText("sub_agent1", "root@1.sub_agent1@1", "inv_1", "rewound"),
				rewind("inv_2", "inv_1"),
				routingText("sub_agent1", "", "inv_3", "no branch"),
			},
			want: "",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			inner := must(llmagent.New(llmagent.Config{Name: "sub_agent1", Model: routingModel{reply: "unused"}}))
			gotBranch := "unset"
			sub := &liveLLMAgent{Agent: inner, llmState: inner.(llminternal.Agent), gotBranch: &gotBranch}
			root := must(llmagent.New(llmagent.Config{Name: "root", Model: routingModel{reply: "unused"}, SubAgents: []agent.Agent{sub}}))
			svc := session.InMemoryService()
			r, err := New(Config{AppName: "app", Agent: root, SessionService: svc})
			if err != nil {
				t.Fatalf("New: %v", err)
			}
			createdSession(t, svc, "s1", tc.events)

			_, events, err := r.RunLive(t.Context(), "u1", "s1", agent.LiveRunConfig{})
			if err != nil {
				t.Fatalf("RunLive: %v", err)
			}
			var yielded []string
			for ev, err := range events {
				if err != nil {
					t.Fatalf("live event: %v", err)
				}
				yielded = append(yielded, ev.Author+"|"+ev.Branch)
			}

			if gotBranch != tc.want {
				t.Errorf("sub_agent1 ran live on branch %q, want %q", gotBranch, tc.want)
			}
			if want := []string{"sub_agent1|" + tc.want}; len(yielded) != 1 || yielded[0] != want[0] {
				t.Errorf("yielded %q, want %q", yielded, want)
			}
		})
	}
}
