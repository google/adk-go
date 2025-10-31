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

// AgentLoader allows to load a particular agent by name and get the root agent
type AgentLoader interface {
	ListAgents() []string
	LoadAgent(string) (agent.Agent, error)
	RootAgent() agent.Agent
}

// multiAgentLoader should be used when you have multiple agents
type multiAgentLoader struct {
	agentMap map[string]agent.Agent
	root     agent.Agent
}

// singleAgentLoader should be used when you have only one agent
type singleAgentLoader struct {
	root agent.Agent
}

func NewSingleAgentLoader(a agent.Agent) AgentLoader {
	return &singleAgentLoader{root: a}
}

func (s *singleAgentLoader) ListAgents() []string {
	return []string{s.root.Name()}
}

func (s *singleAgentLoader) LoadAgent(name string) (agent.Agent, error) {
	if name == "" {
		return s.root, nil
	}
	if name == s.root.Name() {
		return s.root, nil
	}
	return nil, fmt.Errorf("cannot load agent '%s' - provide an empty string or use '%s'", name, s.root.Name())
}

func (s *singleAgentLoader) RootAgent() agent.Agent {
	return s.root
}

func NewMultiAgentLoader(root agent.Agent, agents ...agent.Agent) (AgentLoader, error) {
	m := make(map[string]agent.Agent)
	m[root.Name()] = root
	for _, a := range agents {
		if _, ok := m[a.Name()]; ok {
			// duplicate name
			return nil, fmt.Errorf("duplicate agent name: %s", a.Name())
		}
		m[a.Name()] = a
	}
	return &multiAgentLoader{
		agentMap: m,
		root:     root,
	}, nil
}

func (m *multiAgentLoader) ListAgents() []string {
	agents := make([]string, 0, len(m.agentMap))
	for name := range m.agentMap {
		agents = append(agents, name)
	}
	return agents
}

func (m *multiAgentLoader) LoadAgent(name string) (agent.Agent, error) {
	agent, ok := m.agentMap[name]
	if !ok {
		return nil, fmt.Errorf("agent %s not found. Please specify one of those: %v", name, m.ListAgents())
	}
	return agent, nil
}

func (m *multiAgentLoader) RootAgent() agent.Agent {
	return m.root
}
