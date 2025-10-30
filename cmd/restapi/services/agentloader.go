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

package services

import (
	"fmt"

	"google.golang.org/adk/agent"
)

type AgentLoader interface {
	ListAgents() []string
	LoadAgent(string) (agent.Agent, error)
	RootAgent() agent.Agent
}

type MultiAgentLoader struct {
	agents map[string]agent.Agent
	root   agent.Agent
}

func NewAgentLoader(root agent.Agent, agents ...agent.Agent) (*MultiAgentLoader, error) {
	m := make(map[string]agent.Agent)
	m[root.Name()] = root
	for _, a := range agents {
		if _, ok := m[a.Name()]; ok {
			// duplicate name
			return nil, fmt.Errorf("duplicate agent name: %s", a.Name())
		}
		m[a.Name()] = a
	}
	return &MultiAgentLoader{
		agents: m,
		root:   root,
	}, nil
}

func (m *MultiAgentLoader) RootAgent() agent.Agent {
	return m.root
}

func (m *MultiAgentLoader) ListAgents() []string {
	agents := make([]string, 0, len(m.agents))
	for name := range m.agents {
		agents = append(agents, name)
	}
	return agents
}

func (m *MultiAgentLoader) LoadAgent(name string) (agent.Agent, error) {
	agent, ok := m.agents[name]
	if !ok {
		return nil, fmt.Errorf("agent %s not found. Please specify one of those: %v", name, m.ListAgents())
	}
	return agent, nil
}
