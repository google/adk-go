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

// Package transfertoagenttool provides the transfer_to_agent tool over a list
// of agent names, the tool an LLM agent's model calls to hand the conversation
// to another agent of the same agent tree.
//
// The flow already gives an agent with transfer targets (sub-agents, or a
// parent or peers it may transfer to) a transfer_to_agent tool of its own,
// built from those targets. This package is for an agent that should offer a
// transfer the flow does not compute, or a transfer reason: list the tool New
// returns in the agent's tools. When the model calls it, the tool sets the
// event's session.EventActions.TransferToAgent, and the flow then runs that
// agent, as it does for the flow's own tool.
//
// It is the Go counterpart of adk-python's TransferToAgentTool
// (google.adk.tools.TransferToAgentTool).
package transfertoagenttool

import (
	"google.golang.org/adk/v2/internal/llminternal"
	"google.golang.org/adk/v2/tool"
)

// Config configures the tool New builds.
type Config struct {
	// AgentNames is the list of agent names the model may pass as agent_name, in the
	// order the declaration offers them. Nil and empty are the same: the
	// declaration offers no enum, in both declaration forms.
	AgentNames []string
	// IncludeTransferReason makes the declaration offer the optional transfer_reason
	// parameter and the tool record the value the model sends. Zero value: false.
	IncludeTransferReason bool
}

// New returns the transfer_to_agent tool for cfg. It never fails, and its
// Name is "transfer_to_agent".
//
// New copies cfg.AgentNames, so the caller may change its slice afterwards.
// The tool is immutable: it is safe for concurrent use and may be listed in
// several agents.
//
// The tool declares one required string parameter, agent_name, whose enum is
// cfg.AgentNames in order; with an empty or nil list it declares no enum
// (adk-python declares "enum": [], which no value satisfies). With
// cfg.IncludeTransferReason it also declares the optional string parameter
// transfer_reason, default "", and its description has a line for it. The
// declaration is built for each request: as a JSON Schema in
// ParametersJsonSchema by default, or as a genai.Schema in Parameters when the
// request's context carries feature.LegacyFunctionDeclarationSchema.
//
// When the model calls the tool, it sets the call's
// session.EventActions.TransferToAgent to agent_name and returns nil, which
// the flow answers with {"result": nil}. With cfg.IncludeTransferReason it also sets
// session.EventActions.TransferReason to the transfer_reason the model sent,
// when that is a string; without it, a transfer_reason the model sends is
// ignored. The call fails, and sets nothing, when agent_name is missing or
// not a non-empty string, and when it is not in cfg.AgentNames. A listed
// name no agent of the tree has is accepted by the tool and then fails the
// run in the flow.
//
// An agent with transfer targets already carries the flow's transfer_to_agent
// tool. Listing this tool in such an agent fails each model request with a
// duplicate tool error.
func New(cfg Config) tool.Tool {
	return llminternal.NewNamedTransferToAgentTool(cfg.AgentNames, cfg.IncludeTransferReason)
}
