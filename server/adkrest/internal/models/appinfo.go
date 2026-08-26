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
//
// Agents is flat: nesting is recovered by following [AgentInfo.SubAgents] from
// RootAgentName.
type AppInfo struct {
	Name          string                `json:"name"`
	RootAgentName string                `json:"rootAgentName"`
	Description   string                `json:"description"`
	Language      string                `json:"language"`
	Agents        map[string]*AgentInfo `json:"agents,omitempty"`
}

// AgentInfo describes a single agent within an app.
//
// Agents that are not LLM agents report an empty Instruction and no Tools,
// because they have neither.
type AgentInfo struct {
	Name        string        `json:"name"`
	Description string        `json:"description"`
	Instruction string        `json:"instruction"`
	Tools       []*genai.Tool `json:"tools"`
	SubAgents   []string      `json:"subAgents,omitempty"`
}
