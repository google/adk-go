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
	"testing"

	"github.com/a2aproject/a2a-go/v2/a2a"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	remoteagent "google.golang.org/adk/v2/agent/remoteagent/v2"
	"google.golang.org/adk/v2/agent/workflowagent"
	"google.golang.org/adk/v2/agent/workflowagents/loopagent"
	"google.golang.org/adk/v2/agent/workflowagents/parallelagent"
	"google.golang.org/adk/v2/agent/workflowagents/sequentialagent"
	"google.golang.org/adk/v2/workflow"
)

// Every constructor builds its agent through agent.New, so each one rejects
// an empty name. A nameless agent wrapped by agenttool would otherwise send
// the model a function declaration with an empty name, which Vertex AI
// rejects for the whole request.
func TestConstructors_RejectEmptyName(t *testing.T) {
	tests := []struct {
		name  string
		build func(name string) (agent.Agent, error)
	}{
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
	for _, tc := range tests {
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
