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
// of agent names. An LLM agent's model calls it to name the agent of the same
// agent tree that the conversation should go to.
//
// The tool records the name in the event's
// session.EventActions.TransferToAgent, and the flow handles that transfer as
// it handles one made through the transfer_to_agent tool it builds itself:
//
//   - For the calling agent's own name, it runs the caller again. On the node
//     path (an LLM agent root, or a root set in runner.Config.Node) it fails
//     the run instead.
//   - For one of the caller's transfer targets, it runs that agent. The
//     targets are the caller's sub-agents and, when its parent is an LLM
//     agent, the parent and the parent's other sub-agents, less those that
//     llmagent.Config.DisallowTransferToParent and DisallowTransferToPeers
//     exclude. A sub-agent or peer whose llmagent.Config.Mode is
//     llmagent.ModeTask or llmagent.ModeSingleTurn is not a target.
//   - For any other name, it fails the run. When the name is that of an agent
//     of the tree, the error matches agent.ErrInvalidTransferTarget.
//
// The flow adds its own transfer_to_agent tool to the model requests of an
// agent with transfer targets, unless it leaves the tool out because the agent
// has a built-in search tool its model cannot combine with function calling.
// Listing this tool in an agent whose requests carry the flow's tool fails the
// run with a duplicate tool error when the request is built. An agent without
// transfer targets can list it, and the only transfer the flow runs from such
// an agent is to the agent itself, off the node path.
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
// request's context carries feature.LegacyFunctionDeclarationSchema; for a
// Vertex AI model it adds a null response schema in the same form.
//
// When the model calls the tool, it sets the call's
// session.EventActions.TransferToAgent to agent_name and returns nil, which
// the flow answers with {"result": nil}. With cfg.IncludeTransferReason it also sets
// session.EventActions.TransferReason to the transfer_reason the model sent,
// when that is a string; without it, a transfer_reason the model sends is
// ignored. The call fails, and sets nothing, when agent_name is missing or
// not a non-empty string, and when it is neither in cfg.AgentNames nor the
// name of an agent of the calling agent's tree. A name the tool accepts goes
// to the flow, which runs only the calling agent itself or one of its transfer
// targets and fails the run for any other name, a listed name no agent of the
// tree has included (see the package documentation).
//
// Listing this tool in an agent whose model requests carry the flow's own
// transfer_to_agent tool fails the run with a duplicate tool error when the
// request is built (see the package documentation for when the flow adds it).
func New(cfg Config) tool.Tool {
	return llminternal.NewNamedTransferToAgentTool(cfg.AgentNames, cfg.IncludeTransferReason)
}
