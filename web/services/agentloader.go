package services

import (
	"fmt"

	"google.golang.org/adk/agent"
)

type AgentLoader interface {
	ListAgents() []string
	LoadAgent(string) (agent.Agent, error)
}

type StaticAgentLoader struct {
	agents map[string]agent.Agent
}

func NewStaticAgentLoader(agents map[string]agent.Agent) *StaticAgentLoader {
	return &StaticAgentLoader{
		agents: agents,
	}
}

func (s *StaticAgentLoader) ListAgents() []string {
	agents := make([]string, 0, len(s.agents))
	for name := range s.agents {
		agents = append(agents, name)
	}
	return agents
}

func (s *StaticAgentLoader) LoadAgent(name string) (agent.Agent, error) {
	agent, ok := s.agents[name]
	if !ok {
		return nil, fmt.Errorf("agent %s not found", name)
	}
	return agent, nil
}
