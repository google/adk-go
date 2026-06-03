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

package agent

import (
	"fmt"
)

// Loader allows to load a particular agent by name and get the root agent
type Loader interface {
	// ListAgents returns a list of names of all agents
	ListAgents() []string
	// LoadAgent returns an agent by its name. Returns error if there is no agent with such a name.
	LoadAgent(name string) (Agent, error)
	// RootAgent returns the root agent
	RootAgent() Agent
}

// multiLoader should be used when you have multiple agents
type multiLoader struct {
	agentMap map[string]Agent
	root     Agent
}

// singleLoader should be used when you have only one agent
type singleLoader struct {
	root Agent
}

// NewSingleLoader returns a loader with only one agent, which becomes the root agent
func NewSingleLoader(a Agent) Loader {
	return &singleLoader{root: a}
}

// singleAgentLoader implements AgentLoader. Returns root agent's name
func (s *singleLoader) ListAgents() []string {
	return []string{s.root.Name()}
}

// singleAgentLoader implements AgentLoader. Returns root for empty name and for root.Name(), error otherwise.
func (s *singleLoader) LoadAgent(name string) (Agent, error) {
	if name == "" {
		return s.root, nil
	}
	if name == s.root.Name() {
		return s.root, nil
	}
	return nil, fmt.Errorf("cannot load agent '%s' - provide an empty string or use '%s'", name, s.root.Name())
}

// singleAgentLoader implements AgentLoader. Returns the root agent.
func (s *singleLoader) RootAgent() Agent {
	return s.root
}

// NewMultiLoader returns a new AgentLoader with the given root Agent and other agents.
// Returns an error if more than one agent (including root) shares the same name
func NewMultiLoader(root Agent, agents ...Agent) (Loader, error) {
	m := make(map[string]Agent)
	m[root.Name()] = root
	for _, a := range agents {
		if _, ok := m[a.Name()]; ok {
			// duplicate name
			return nil, fmt.Errorf("duplicate agent name: %s", a.Name())
		}
		m[a.Name()] = a
	}
	return &multiLoader{
		agentMap: m,
		root:     root,
	}, nil
}

// multiAgentLoader implements AgentLoader. Returns the list of all agents' names (including root agent)
func (m *multiLoader) ListAgents() []string {
	agents := make([]string, 0, len(m.agentMap))
	for name := range m.agentMap {
		agents = append(agents, name)
	}
	return agents
}

// multiAgentLoader implements LoadAgent. Returns an agent with given name or error if no such an agent is found
func (m *multiLoader) LoadAgent(name string) (Agent, error) {
	agent, ok := m.agentMap[name]
	if !ok {
		return nil, fmt.Errorf("agent %s not found. Please specify one of those: %v", name, m.ListAgents())
	}
	return agent, nil
}

// multiAgentLoader implements LoadAgent.
func (m *multiLoader) RootAgent() Agent {
	return m.root
}
