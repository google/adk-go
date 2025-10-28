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

package sequentialagent

import (
	"fmt"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/workflowagents/loopagent"
	agentinternal "google.golang.org/adk/internal/agent"
)

// New creates a SequentialAgent.
//
// SequentialAgent executes its sub-agents once, in the order they are listed.
//
// Use the SequentialAgent when you want the execution to occur in a fixed,
// strict order.
func New(cfg Config) (agent.Agent, error) {
	sequentialAgent, err := loopagent.New(loopagent.Config{
		AgentConfig:   cfg.AgentConfig,
		MaxIterations: 1,
	})
	if err != nil {
		return nil, err
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

type Config struct {
	// Basic agent setup.
	AgentConfig agent.Config
}
