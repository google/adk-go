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

// Package loopagent provides an agent that repeatedly runs its sub-agents for a
// specified number of iterations or until termination condition is met.
package loopagent

import (
	"fmt"
	"iter"

	"google.golang.org/adk/agent"
	agentinternal "google.golang.org/adk/internal/agent"
	"google.golang.org/adk/session"
)

// Config defines the configuration for a LoopAgent.
type Config struct {
	// Basic agent setup.
	AgentConfig agent.Config

	// If MaxIterations == 0, then LoopAgent runs indefinitely or until any
	// sub-agent escalates.
	MaxIterations uint
}

// New creates a LoopAgent.
//
// LoopAgent repeatedly runs its sub-agents in sequence for a specified number
// of iterations or until a termination condition is met.
//
// Use the LoopAgent when your workflow involves repetition or iterative
// refinement, such as like revising code.
func New(cfg Config) (agent.Agent, error) {
	if cfg.AgentConfig.Run != nil {
		return nil, fmt.Errorf("LoopAgent doesn't allow custom Run implementations")
	}

	loopAgentImpl := &loopAgent{
		maxIterations: cfg.MaxIterations,
	}
	cfg.AgentConfig.Run = loopAgentImpl.Run

	loopAgent, err := agent.New(cfg.AgentConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create base agent: %w", err)
	}

	internalAgent, ok := loopAgent.(agentinternal.Agent)
	if !ok {
		return nil, fmt.Errorf("internal error: failed to convert to internal agent")
	}
	state := agentinternal.Reveal(internalAgent)
	state.AgentType = agentinternal.TypeLoopAgent
	state.Config = cfg

	return loopAgent, nil
}

type loopAgent struct {
	maxIterations uint
}

func (a *loopAgent) Run(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
	count := a.maxIterations

	return func(yield func(*session.Event, error) bool) {
		for {
			shouldExit := false
			for _, subAgent := range ctx.Agent().SubAgents() {
				for event, err := range subAgent.Run(ctx) {
					// TODO: ensure consistency -- if there's an error, return and close iterator, verify everywhere in ADK.
					if !yield(event, err) {
						return
					}

					if event.Actions.Escalate {
						shouldExit = true
					}
				}
				if shouldExit {
					return
				}
			}

			if count > 0 {
				count--
				if count == 0 {
					return
				}
			}
		}
	}
}
