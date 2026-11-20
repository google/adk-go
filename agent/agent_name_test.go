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

package agent_test

import (
	"errors"
	"strings"
	"testing"

	legacya2a "github.com/a2aproject/a2a-go/a2a"
	"github.com/a2aproject/a2a-go/v2/a2a"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	legacyremoteagent "google.golang.org/adk/v2/agent/remoteagent"
	remoteagent "google.golang.org/adk/v2/agent/remoteagent/v2"
	"google.golang.org/adk/v2/agent/workflowagent"
	"google.golang.org/adk/v2/agent/workflowagents/loopagent"
	"google.golang.org/adk/v2/agent/workflowagents/parallelagent"
	"google.golang.org/adk/v2/agent/workflowagents/sequentialagent"
	"google.golang.org/adk/v2/workflow"
)

type constructor struct {
	name  string
	build func(name string) (agent.Agent, error)
}

// constructors returns every public agent constructor, each building an
// otherwise valid agent with the given name.
func constructors() []constructor {
	return []constructor{
		{
			name: "agent.New",
			build: func(name string) (agent.Agent, error) {
				return agent.New(agent.Config{Name: name})
			},
		},
		{
			name: "llmagent.New",
			build: func(name string) (agent.Agent, error) {
				return llmagent.New(llmagent.Config{Name: name})
			},
		},
		{
			name: "remoteagent.NewA2A",
			build: func(name string) (agent.Agent, error) {
				return remoteagent.NewA2A(remoteagent.A2AConfig{Name: name, AgentCard: &a2a.AgentCard{}})
			},
		},
		{
			name: "legacy remoteagent.NewA2A",
			build: func(name string) (agent.Agent, error) {
				return legacyremoteagent.NewA2A(legacyremoteagent.A2AConfig{Name: name, AgentCard: &legacya2a.AgentCard{}})
			},
		},
		{
			name: "loopagent.New",
			build: func(name string) (agent.Agent, error) {
				return loopagent.New(loopagent.Config{AgentConfig: agent.Config{Name: name}})
			},
		},
		{
			name: "parallelagent.New",
			build: func(name string) (agent.Agent, error) {
				return parallelagent.New(parallelagent.Config{AgentConfig: agent.Config{Name: name}})
			},
		},
		{
			name: "sequentialagent.New",
			build: func(name string) (agent.Agent, error) {
				return sequentialagent.New(sequentialagent.Config{AgentConfig: agent.Config{Name: name}})
			},
		},
		{
			name: "workflowagent.New",
			build: func(name string) (agent.Agent, error) {
				node := workflow.NewFunctionNode("echo", func(_ agent.Context, in string) (string, error) {
					return in, nil
				}, workflow.NodeConfig{})
				return workflowagent.New(workflowagent.Config{Name: name, Edges: workflow.Chain(workflow.Start, node)})
			},
		},
	}
}

// Every constructor builds its agent through agent.New, so each one rejects
// an empty name. A nameless agent wrapped by agenttool would otherwise send
// the model a function declaration with an empty name, which Vertex AI
// rejects for the whole request.
func TestConstructors_RejectEmptyName(t *testing.T) {
	for _, tc := range constructors() {
		t.Run(tc.name, func(t *testing.T) {
			// The named build passing shows the config is otherwise valid, so
			// the error below comes from the empty name.
			if _, err := tc.build("named"); err != nil {
				t.Fatalf("build(%q) error = %v, want nil", "named", err)
			}
			a, err := tc.build("")
			if err == nil {
				t.Fatalf("build(%q) error = nil, want an error", "")
			}
			if a != nil {
				t.Errorf("build(%q) agent = %v, want nil", "", a)
			}
		})
	}
}

// The session records the end user's messages under the author "user", so an
// agent with that name could not be told apart from the user, as in
// adk-python's BaseAgent.validate_name. The check is case-sensitive.
func TestConstructors_RejectReservedUserName(t *testing.T) {
	for _, tc := range constructors() {
		t.Run(tc.name, func(t *testing.T) {
			a, err := tc.build("user")
			if !errors.Is(err, agent.ErrInvalidAgentConfig) {
				t.Fatalf("build(%q) error = %v, want an error wrapping agent.ErrInvalidAgentConfig", "user", err)
			}
			if want := "agent name cannot be `user`, it is reserved for the end user's input"; !strings.Contains(err.Error(), want) {
				t.Errorf("build(%q) error = %q, want it to contain %q", "user", err, want)
			}
			if a != nil {
				t.Errorf("build(%q) agent = %v, want nil", "user", a)
			}
		})
		for _, name := range []string{"User", "USER"} {
			t.Run(tc.name+"/accepts "+name, func(t *testing.T) {
				a, err := tc.build(name)
				if err != nil {
					t.Fatalf("build(%q) error = %v, want nil", name, err)
				}
				if got := a.Name(); got != name {
					t.Errorf("build(%q).Name() = %q, want %q", name, got, name)
				}
			})
		}
	}
	// The name is refused whatever else the config holds, sub-agents included.
	t.Run("agent.New with a sub-agent", func(t *testing.T) {
		sub, err := agent.New(agent.Config{Name: "sub"})
		if err != nil {
			t.Fatalf("agent.New(%q) error = %v, want nil", "sub", err)
		}
		a, err := agent.New(agent.Config{Name: "user", SubAgents: []agent.Agent{sub}})
		if !errors.Is(err, agent.ErrInvalidAgentConfig) {
			t.Fatalf("agent.New(%q) with a sub-agent error = %v, want an error wrapping agent.ErrInvalidAgentConfig", "user", err)
		}
		if a != nil {
			t.Errorf("agent.New(%q) with a sub-agent agent = %v, want nil", "user", a)
		}
	})
}
