// Copyright 2025 Google LLC
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
	"context"
	"testing"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/session"
	"google.golang.org/adk/sessionservice"
)

func TestRunner_findAgentToRun(t *testing.T) {
	t.Parallel()

	sessionID := session.ID{
		AppName:   "test",
		UserID:    "userID",
		SessionID: "sessionID",
	}

	agentTree := agentTree(t)

	tests := []struct {
		name      string
		rootAgent agent.Agent
		session   sessionservice.StoredSession
		wantAgent agent.Agent
		wantErr   bool
	}{
		{
			name: "last event from agent allowing transfer",
			session: createSession(t, t.Context(), sessionID, []*session.Event{
				{
					Author: "allows_transfer_agent",
				},
				{
					Author: "user",
				},
			}),
			rootAgent: agentTree.root,
			wantAgent: agentTree.allowsTransferAgent,
		},
		{
			name: "last event from agent not allowing transfer",
			session: createSession(t, t.Context(), sessionID, []*session.Event{
				{
					Author: "no_transfer_agent",
				},
				{
					Author: "user",
				},
			}),
			rootAgent: agentTree.root,
			wantAgent: agentTree.root,
		},
		{
			name: "no events from agents, call root",
			session: createSession(t, t.Context(), sessionID, []*session.Event{
				{
					Author: "user",
				},
			}),
			rootAgent: agentTree.root,
			wantAgent: agentTree.root,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := &Runner{
				RootAgent: tt.rootAgent,
			}
			gotAgent, err := r.findAgentToRun(tt.session)
			if (err != nil) != tt.wantErr {
				t.Errorf("Runner.findAgentToRun() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if tt.wantAgent != gotAgent {
				t.Errorf("Runner.findAgentToRun() = %+v, want %+v", gotAgent.Name(), tt.wantAgent.Name())
			}
		})
	}
}

func Test_findAgent(t *testing.T) {
	agentTree := agentTree(t)

	oneAgent := must(llmagent.New(llmagent.Config{
		Name: "test",
	}))

	tests := []struct {
		name      string
		root      agent.Agent
		target    string
		wantAgent agent.Agent
	}{
		{
			name:      "ok",
			root:      agentTree.root,
			target:    agentTree.allowsTransferAgent.Name(),
			wantAgent: agentTree.allowsTransferAgent,
		},
		{
			name:      "finds in one node tree",
			root:      oneAgent,
			target:    oneAgent.Name(),
			wantAgent: oneAgent,
		},
		{
			name:      "doesn't fail if agent is missing in the tree",
			root:      agentTree.root,
			target:    "random",
			wantAgent: nil,
		},
		{
			name:      "doesn't fail on the empty tree",
			root:      nil,
			target:    "random",
			wantAgent: nil,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if gotAgent := findAgent(tt.root, tt.target); gotAgent != tt.wantAgent {
				t.Errorf("Runner.findAgent() = %+v, want %+v", gotAgent.Name(), tt.wantAgent.Name())
			}
		})
	}
}

func Test_isTransferrableAcrossAgentTree(t *testing.T) {
	tests := []struct {
		name  string
		agent agent.Agent
		want  bool
	}{
		{
			name: "disallow for agent with DisallowTransferToParent",
			agent: must(llmagent.New(llmagent.Config{
				Name:                     "test",
				DisallowTransferToParent: true,
			})),
			want: false,
		},
		{
			name: "disallow for non-LLM agent",
			agent: must(agent.New(agent.Config{
				Name: "test",
			})),
			want: false,
		},
		{
			name: "allow for the default LLM agent",
			agent: must(llmagent.New(llmagent.Config{
				Name: "test",
			})),
			want: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			runner, err := New("testApp", tt.agent, sessionservice.Mem())
			if err != nil {
				t.Fatal(err)
			}
			if got := runner.isTransferableAcrossAgentTree(tt.agent); got != tt.want {
				t.Errorf("isTransferrableAcrossAgentTree() = %v, want %v", got, tt.want)
			}
		})
	}
}

// creates agentTree for tests and returns references to the agents
func agentTree(t *testing.T) agentTreeStruct {
	t.Helper()

	sub1 := must(llmagent.New(llmagent.Config{
		Name:                     "no_transfer_agent",
		DisallowTransferToParent: true,
	}))
	sub2 := must(llmagent.New(llmagent.Config{
		Name: "allows_transfer_agent",
	}))
	parent := must(llmagent.New(llmagent.Config{
		Name:      "root",
		SubAgents: []agent.Agent{sub1, sub2},
	}))

	return agentTreeStruct{
		root:                parent,
		noTransferAgent:     sub1,
		allowsTransferAgent: sub2,
	}
}

type agentTreeStruct struct {
	root, noTransferAgent, allowsTransferAgent agent.Agent
}

func must[T agent.Agent](a T, err error) T {
	if err != nil {
		panic(err)
	}
	return a
}

func createSession(t *testing.T, ctx context.Context, id session.ID, events []*session.Event) sessionservice.StoredSession {
	t.Helper()

	service := sessionservice.Mem()

	storedSession, err := service.Create(ctx, &sessionservice.CreateRequest{
		AppName:   id.AppName,
		UserID:    id.UserID,
		SessionID: id.SessionID,
	})
	if err != nil {
		t.Fatal(err)
	}

	for _, event := range events {
		if err := service.AppendEvent(ctx, storedSession, event); err != nil {
			t.Fatal(err)
		}
	}

	return storedSession
}
