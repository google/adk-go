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

// Package sequentialagent provides an agent that runs its sub-agents in a sequence.
package sequentialagent

import (
	"fmt"
	"iter"

	"google.golang.org/adk/agent"
	agentinternal "google.golang.org/adk/internal/agent"
	"google.golang.org/adk/session"
)

// New creates a SequentialAgent.
//
// SequentialAgent executes its sub-agents once, in the order they are listed.
//
// Use the SequentialAgent when you want the execution to occur in a fixed,
// strict order.
func New(cfg Config) (agent.Agent, error) {
	if cfg.AgentConfig.Run != nil {
		return nil, fmt.Errorf("LoopAgent doesn't allow custom Run implementations")
	}

	sequentialAgentImpl := &sequentialAgent{}
	cfg.AgentConfig.Run = sequentialAgentImpl.Run

	sequentialAgent, err := agent.New(cfg.AgentConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create base agent: %w", err)
	}

	internalAgent, ok := sequentialAgent.(agentinternal.Agent)
	if !ok {
		return nil, fmt.Errorf("internal error: failed to convert to internal agent")
	}
	state := agentinternal.Reveal(internalAgent)
	state.AgentType = agentinternal.TypeSequentialAgent
	state.Config = cfg

	return sequentialAgent, nil
}

// Config defines the configuration for a SequentialAgent.
type Config struct {
	// Basic agent setup.
	AgentConfig agent.Config
}

type sequentialAgent struct{}

func (a *sequentialAgent) Run(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		for _, subAgent := range ctx.Agent().SubAgents() {
			for event, err := range subAgent.Run(ctx) {
				// TODO: ensure consistency -- if there's an error, return and close iterator, verify everywhere in ADK.
				if !yield(event, err) {
					return
				}
			}
		}
	}
}
