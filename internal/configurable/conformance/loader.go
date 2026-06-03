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

package conformance

import (
	"fmt"
	"sort"

	"google.golang.org/adk/agent"
)

type conformanceAgentLoader struct {
	agentMap    map[string]agent.Agent
	agentsNames []string
}

// NewConformanceAgentLoader returns a new AgentLoader with the given root Agent and other agents.
// Returns an error if more than one agent (including root) shares the same name
func NewConformanceAgentLoader(agentMap map[string]agent.Agent) (agent.Loader, error) {
	agentsNames := make([]string, 0, len(agentMap))
	for name := range agentMap {
		agentsNames = append(agentsNames, name)
	}
	sort.Strings(agentsNames)
	return &conformanceAgentLoader{
		agentMap:    agentMap,
		agentsNames: agentsNames,
	}, nil
}

// conformanceAgentLoader implements AgentLoader. Returns the list of all agents' names (including root agent)
func (m *conformanceAgentLoader) ListAgents() []string {
	return m.agentsNames
}

// conformanceAgentLoader implements LoadAgent. Returns an agent with given name or error if no such an agent is found
func (m *conformanceAgentLoader) LoadAgent(name string) (agent.Agent, error) {
	agent, ok := m.agentMap[name]
	if !ok {
		return nil, fmt.Errorf("agent %s not found. Please specify one of those: %v", name, m.ListAgents())
	}
	return agent, nil
}

// conformanceAgentLoader implements LoadAgent.
func (m *conformanceAgentLoader) RootAgent() agent.Agent {
	if len(m.agentsNames) == 0 {
		return nil
	}
	return m.agentMap[m.agentsNames[0]]
}
