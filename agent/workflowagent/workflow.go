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

package workflowagent

import (
	"google.golang.org/adk/agent"
	"google.golang.org/adk/workflow"
)

// Config is the configuration for creating a new Workflow agent.
type Config struct {
	AgentConfig agent.Config
	Edges       []workflow.Edge
}

// New creates a new Workflow agent.
func New(cfg Config) (agent.Agent, error) {
	w := workflow.New(cfg.Edges)

	agentCfg := cfg.AgentConfig
	agentCfg.Run = w.Run

	return agent.New(agentCfg)
}
