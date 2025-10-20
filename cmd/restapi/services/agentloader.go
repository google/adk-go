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
	Root() agent.Agent
	ListAgents() []string
	LoadAgent(string) (agent.Agent, error)
}

// SingleAgentLoader should be used when you have only one agent
type SingleAgentLoader struct {
	main agent.Agent
}

func NewSingleAgentLoader(a agent.Agent) *SingleAgentLoader {
	return &SingleAgentLoader{main: a}
}

func (s *SingleAgentLoader) ListAgents() []string {
	return []string{s.main.Name()}
}

func (s *SingleAgentLoader) LoadAgent(name string) (agent.Agent, error) {
	if name == "" {
		return s.main, nil
	}
	if name == s.main.Name() {
		return s.main, nil
	}
	return nil, fmt.Errorf("cannot load agent '%s' - provide empty string or use '%s'", name, s.main.Name())
}

// MultiAgentLoader should be used when you have more than one agent
type MultiAgentLoader struct {
	agents map[string]agent.Agent
}

func NewStaticAgentLoader(agents map[string]agent.Agent) *MultiAgentLoader {
	return &MultiAgentLoader{
		agents: agents,
	}
}

func (s *MultiAgentLoader) ListAgents() []string {
	agents := make([]string, 0, len(s.agents))
	for name := range s.agents {
		agents = append(agents, name)
	}
	return agents
}

func (s *MultiAgentLoader) LoadAgent(name string) (agent.Agent, error) {
	agent, ok := s.agents[name]
	if !ok {
		return nil, fmt.Errorf("agent %s not found", name)
	}
	return agent, nil
}
