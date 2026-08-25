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

package models

import "google.golang.org/genai"

// LanguageGo is the runtime discriminator reported by this server in
// [AppInfo.Language].
const LanguageGo = "go"

// AppInfo describes an ADK app and the agents it contains. It is the response
// body of GET /apps/{app_name}/app-info.
type AppInfo struct {
	// Name of the app, as registered with the agent loader.
	Name string `json:"name"`
	// RootAgentName is the name of the agent that starts the execution. It is
	// frequently different from Name.
	RootAgentName string `json:"rootAgentName"`
	// Description of the root agent.
	Description string `json:"description"`
	// Language identifies the runtime serving the app. Always [LanguageGo].
	Language string `json:"language"`
	// Agents holds every agent reachable from the root, keyed by agent name.
	// The map is flat: nesting is recovered by following [AgentInfo.SubAgents]
	// from RootAgentName.
	Agents map[string]*AgentInfo `json:"agents,omitempty"`
}

// AgentInfo describes a single agent within an app.
//
// Agents that are not LLM agents (workflow, loop, sequential, parallel, custom
// and remote agents) report an empty Instruction and no Tools, because they
// have neither.
type AgentInfo struct {
	// Name of the agent, unique within the app.
	Name string `json:"name"`
	// Description of the agent's capability.
	Description string `json:"description"`
	// Instruction given to the model. Empty for agents that have none, and for
	// agents whose instruction is produced dynamically by a provider, which
	// cannot be resolved outside of an invocation.
	Instruction string `json:"instruction"`
	// Tools the agent exposes to the model. Function-callable tools are
	// reported as function declarations; native Gemini tools such as Google
	// Search are reported in their own [genai.Tool] form.
	//
	// Never nil: the wire contract requires the field, and a nil slice would
	// marshal to null.
	Tools []*genai.Tool `json:"tools"`
	// SubAgents holds the names of this agent's direct children. Look them up
	// in [AppInfo.Agents].
	SubAgents []string `json:"subAgents,omitempty"`
}
