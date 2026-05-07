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

// package vertexai provides utilities for Agent Engine deployments
package vertexai

import "fmt"

const (
	agentEngineTemplate         = "projects/%s/locations/%s/reasoningEngines/%s"
	agentEngineSessionbTemplate = agentEngineTemplate + "/sessions/%s"
)

type AgentEngineData struct {
	Location        string
	ProjectID       string
	ReasoningEngine string
}

// AgentEngineResource returns a formatted string indicating agent engine instance
// (template `projects/%s/locations/%s/reasoningEngines/%s`)
func AgentEngineResource(data *AgentEngineData) string {
	return fmt.Sprintf(agentEngineTemplate, data.ProjectID, data.Location, data.ReasoningEngine)
}

// AgentEngineResource returns a formatted string indicating specific session for an agent engine instance
// (template `projects/%s/locations/%s/reasoningEngines/%s/sessions/%s`)
func SessionResource(data *AgentEngineData, sessionID string) string {
	return fmt.Sprintf(agentEngineSessionbTemplate, data.ProjectID, data.Location, data.ReasoningEngine, sessionID)
}
