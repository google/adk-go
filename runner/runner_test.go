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
	"iter"
	"testing"

	"google.golang.org/adk"
	"google.golang.org/adk/agent"
	"google.golang.org/genai"
)

func TestRunner_findAgentToRun(t *testing.T) {
	t.Parallel()

	agentTree := agentTree(t)

	tests := []struct {
		name      string
		rootAgent adk.Agent
		session   *adk.Session
		wantAgent adk.Agent
		wantErr   bool
	}{
		{
			name: "last event from agent allowing transfer",
			session: &adk.Session{
				Events: []*adk.Event{
					{
						Author: "allows_transfer_agent",
					},
					{
						Author: "user",
					},
				},
			},
			rootAgent: agentTree.root,
			wantAgent: agentTree.allowsTransferAgent,
		},
		{
			name: "last event from agent not allowing transfer",
			session: &adk.Session{
				Events: []*adk.Event{
					{
						Author: "no_transfer_agent",
					},
					{
						Author: "user",
					},
				},
			},
			rootAgent: agentTree.root,
			wantAgent: agentTree.root,
		},
		{
			name: "no events from agents, call root",
			session: &adk.Session{
				Events: []*adk.Event{
					{
						Author: "user",
					},
				},
			},
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
				t.Errorf("Runner.findAgentToRun() = %+v, want %+v", gotAgent.Spec(), tt.wantAgent.Spec())
			}
		})
	}
}

func Test_findAgent(t *testing.T) {
	agentTree := agentTree(t)

	oneAgent := must(agent.NewLLMAgent("test", nil))

	tests := []struct {
		name      string
		root      adk.Agent
		target    string
		wantAgent adk.Agent
	}{
		{
			name:      "ok",
			root:      agentTree.root,
			target:    agentTree.allowsTransferAgent.Spec().Name,
			wantAgent: agentTree.allowsTransferAgent,
		},
		{
			name:      "finds in one node tree",
			root:      oneAgent,
			target:    oneAgent.Spec().Name,
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
				t.Errorf("Runner.findAgent() = %+v, want %+v", gotAgent.Spec(), tt.wantAgent.Spec())
			}
		})
	}
}

func Test_isTransferrableAcrossAgentTree(t *testing.T) {
	tests := []struct {
		name  string
		agent adk.Agent
		want  bool
	}{
		{
			name:  "disallow for agent with DisallowTransferToParent",
			agent: must(agent.NewLLMAgent("test", nil, agent.WithDisallowTransferToParent())),
			want:  false,
		},
		{
			name:  "disallow for non-LLM agent",
			agent: &customAgent{},
			want:  false,
		},
		{
			name:  "allow for the default LLM agent",
			agent: must(agent.NewLLMAgent("test", nil)),
			want:  true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := isTransferrableAcrossAgentTree(tt.agent); got != tt.want {
				t.Errorf("isTransferrableAcrossAgentTree() = %v, want %v", got, tt.want)
			}
		})
	}
}

// creates agentTree for tests and returns references to the agents
func agentTree(t *testing.T) agentTreeStruct {
	t.Helper()

	sub1 := must(agent.NewLLMAgent("no_transfer_agent", nil, agent.WithDisallowTransferToParent()))
	sub2 := must(agent.NewLLMAgent("allows_transfer_agent", nil))

	parent, err := agent.NewLLMAgent("root", nil, agent.WithSubAgents(sub1, sub2))
	if err != nil {
		t.Fatal(err)
	}

	return agentTreeStruct{
		root:                parent,
		noTransferAgent:     sub1,
		allowsTransferAgent: sub2,
	}
}

type agentTreeStruct struct {
	root, noTransferAgent, allowsTransferAgent adk.Agent
}

func must[T adk.Agent](a T, err error) T {
	if err != nil {
		panic(err)
	}
	return a
}

type customAgent struct {
	spec *adk.AgentSpec

	callCounter int
}

func (a *customAgent) Spec() *adk.AgentSpec { return a.spec }

func (a *customAgent) Run(context.Context, *adk.InvocationContext) iter.Seq2[*adk.Event, error] {
	return func(yield func(*adk.Event, error) bool) {
		a.callCounter++

		yield(&adk.Event{
			LLMResponse: &adk.LLMResponse{
				Content: genai.NewContentFromText("hello", genai.RoleModel),
			},
		}, nil)
	}
}
