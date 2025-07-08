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

package agent

import (
	"bytes"
	"context"
	"fmt"
	"slices"
	"text/template"

	"github.com/google/adk-go"
	"google.golang.org/genai"
)

// * Single-flow
//
// SingleFlow is the LLM flows that handles tools calls.
//
//  A single flow only consider an agent itself and tools.
//  No sub-agents are allowed for single flow, I.e.,
//      DisallowTransferToParent == true &&
//      DisallowTransferToPeers == true &&
//      len(SubAgents) == 0
//
// * Auto-flow
//
// Agent transfer is allowed in the following direction:
//
//  1. from parent to sub-agent;
//
//  2. from sub-agent to parent;
//
//  3. from sub-agent to its peer agents;
//
//     For peer-agent transfers, it's only enabled when all below conditions are met:
//
//     - The parent agent is also of AutoFlow;
//     - `disallow_transfer_to_peer` option of this agent is False (default).
//
// Depending on the target agent flow type, the transfer may be automatically
// reversed. The condition is as below:
//
//   - If the flow type of the tranferee agent is also auto, transfee agent will
//     remain as the active agent. The transfee agent will respond to the user's
//     next message directly.
//   - If the flow type of the transfere agent is not auto, the active agent will
//     be reversed back to previous agent.

func agentTransferRequestProcessor(ctx context.Context, parentCtx *adk.InvocationContext, req *adk.LLMRequest) error {
	agent := asLLMAgent(parentCtx.Agent)
	if agent == nil {
		return nil // TODO: support agent types other than LLMAgent, that have parent/subagents?
	}
	if !agent.useAutoFlow() {
		return nil
	}

	targets := transferTarget(agent)
	if len(targets) == 0 {
		return nil
	}

	// TODO(hyangah): why do we set this up in request processor
	// instead of registering this as a normal function tool of the Agent?
	transferToAgentTool := &transferToAgentTool{}
	si, err := instructionsForTransferToAgent(agent, targets, transferToAgentTool)
	if err != nil {
		return err
	}
	req.AppendInstructions(si)
	tc := &adk.ToolContext{
		InvocationContext: parentCtx,
	}
	return transferToAgentTool.ProcessRequest(ctx, tc, req)
}

type transferToAgentTool struct{}

// Description implements adk.Tool.
func (t *transferToAgentTool) Description() string {
	return `Transfer the question to another agent.
This tool hands off control to another agent when it's more suitable to answer the user's question according to the agent's description.`
}

// Name implements adk.Tool.
func (t *transferToAgentTool) Name() string {
	return "transfer_to_agent"
}

func (t *transferToAgentTool) FunctionDeclaration() *genai.FunctionDeclaration {
	return &genai.FunctionDeclaration{
		Name:        t.Name(),
		Description: t.Description(),
		Parameters: &genai.Schema{
			Description: "the agent name to transfer to",
			Title:       "agent_name",
			Type:        "string",
			// TODO: set allowed values for agent_name.
		},
	}
}

// ProcessRequest implements adk.Tool.
func (t *transferToAgentTool) ProcessRequest(ctx context.Context, tc *adk.ToolContext, req *adk.LLMRequest) error {
	return req.AppendTools(t)
}

// Run implements adk.Tool.
func (t *transferToAgentTool) Run(ctx context.Context, tc *adk.ToolContext, args map[string]any) (map[string]any, error) {
	if args == nil {
		return nil, fmt.Errorf("invalid arguments: %v", args)
	}
	agent, ok := args["agent_name"].(string)
	if !ok || agent == "" {
		return nil, fmt.Errorf("invalid agent name: %v", args)
	}
	tc.EventActions.TransferToAgent = agent
	return map[string]any{}, nil
}

var _ adk.Tool = (*transferToAgentTool)(nil)

func transferTarget(current *LLMAgent) []adk.Agent {
	targets := slices.Clone(current.SubAgents)

	if !current.DisallowTransferToParent && current.ParentAgent != nil {
		targets = append(targets, current.ParentAgent)
	}
	// For peer-agent transfers, it's only enabled when all below conditions are met:
	// - the parent agent is also of AutoFlow.
	// - DisallowTransferToPeers is false.
	if !current.DisallowTransferToPeers {
		parent := asLLMAgent(current.ParentAgent)
		if parent != nil && parent.useAutoFlow() {
			targets = append(targets, parent.SubAgents...)
		}
	}
	return targets
}

func instructionsForTransferToAgent(agent *LLMAgent, targets []adk.Agent, transferTool adk.Tool) (string, error) {
	tmpl, err := template.New("transfer_to_agent_prompt").Parse(agentTransferInstructionTemplate)
	if err != nil {
		return "", err
	}
	parent := agent.ParentAgent
	if agent.DisallowTransferToParent {
		parent = nil
	}

	var buf bytes.Buffer
	if err := tmpl.Execute(&buf, struct {
		AgentName string
		Parent    adk.Agent
		Targets   []adk.Agent
		ToolName  string
	}{
		AgentName: agent.Name(),
		Parent:    parent,
		Targets:   targets,
		ToolName:  transferTool.Name(),
	}); err != nil {
		return "", err
	}
	return buf.String(), nil
}

// Prompt source:
//  flows/llm_flows/agent_transfer.py _build_target_agents_instructions.

const agentTransferInstructionTemplate = `You have a list of other agents to transfer to:
{{range .Targets}}
Agent name: {{.Name}}
Agent description: {{.Description}}
{{end}}
If you are the best to answer the question according to your description, you
can answer it.
If another agent is better for answering the question according to its
description, call '{{.ToolName}}' function to transfer the
question to that agent. When transfering, do not generate any text other than
the function call.
{{if .Parent}}
Your parent agent is {{.Parent.Name}}. If neither the other agents nor
you are best for answering the question according to the descriptions, transfer
to your parent agent. If you don't have parent agent, try answer by yourself.
{{end}}
`
