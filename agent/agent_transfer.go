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

	"google.golang.org/adk/types"
	"google.golang.org/genai"
)

// From src/google/adk/flows/llm_flows/auto_flow.py
//
// * SingleFlow
//
// SingleFlow is the LLM flow that handles tool calls.
//
//  A single flow only considers the agent itself and its tools.
//  No sub-agents are allowed for a single flow, i.e.,
//      DisallowTransferToParent == true &&
//      DisallowTransferToPeers == true &&
//      len(SubAgents) == 0
//
// * AutoFlow
//
// Agent transfers are allowed in the following directions:
//
//  1. From parent to sub-agent.
//  2. From sub-agent to parent.
//  3. From sub-agent to its peer agent.
//
// Peer-agent transfers are only enabled when all the following conditions are met:
//
//  - The parent agent is also an LLMAgent.
//  - This agent has DisallowTransferToPeers set to false (default).
//
// Depending on the target agent type, the transfer may be automatically
// reversed. See python's Runner._find_agent_to_run method for which
// agent will remain active to handle the next user message.
// (src/google/adk/runners.py)
//
// TODO: implement it in the runners package and update this doc.

func agentTransferRequestProcessor(ctx context.Context, parentCtx *types.InvocationContext, req *types.LLMRequest) error {
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
	tc := &types.ToolContext{
		InvocationContext: parentCtx,
	}
	return transferToAgentTool.ProcessRequest(ctx, tc, req)
}

type transferToAgentTool struct{}

// Description implements types.Tool.
func (t *transferToAgentTool) Description() string {
	return `Transfer the question to another agent.
This tool hands off control to another agent when it's more suitable to answer the user's question according to the agent's description.`
}

// Name implements types.Tool.
func (t *transferToAgentTool) Name() string {
	return "transfer_to_agent"
}

func (t *transferToAgentTool) FunctionDeclaration() *genai.FunctionDeclaration {
	return &genai.FunctionDeclaration{
		Name:        t.Name(),
		Description: t.Description(),
		Parameters: &genai.Schema{
			Type: "object",
			Properties: map[string]*genai.Schema{
				"agent_name": {
					Type:        "string",
					Description: "the agent name to transfer to",
				},
			},
			Required: []string{"agent_name"},
		},
	}
}

// ProcessRequest implements types.Tool.
func (t *transferToAgentTool) ProcessRequest(ctx context.Context, tc *types.ToolContext, req *types.LLMRequest) error {
	return req.AppendTools(t)
}

// Run implements types.Tool.
func (t *transferToAgentTool) Run(ctx context.Context, tc *types.ToolContext, args map[string]any) (map[string]any, error) {
	if args == nil {
		return nil, fmt.Errorf("missing argument")
	}
	agent, ok := args["agent_name"].(string)
	if !ok || agent == "" {
		return nil, fmt.Errorf("empty agent_name: %v", args)
	}
	tc.EventActions.TransferToAgent = agent
	return map[string]any{}, nil
}

var _ types.Tool = (*transferToAgentTool)(nil)

func transferTarget(current *LLMAgent) []types.Agent {
	targets := slices.Clone(current.Spec().SubAgents)

	if !current.DisallowTransferToParent && current.Spec().Parent() != nil {
		targets = append(targets, current.Spec().Parent())
	}
	// For peer-agent transfers, it's only enabled when all below conditions are met:
	// - the parent agent is also of AutoFlow.
	// - DisallowTransferToPeers is false.
	if !current.DisallowTransferToPeers {
		parent := asLLMAgent(current.Spec().Parent())
		if parent != nil && parent.useAutoFlow() {
			for _, peer := range parent.Spec().SubAgents {
				if peer.Spec().Name != current.Spec().Name {
					targets = append(targets, peer)
				}
			}
		}
	}
	return targets
}

var transferToAgentPromptTmpl = template.Must(
	template.New("transfer_to_agent_prompt").Parse(agentTransferInstructionTemplate))

func instructionsForTransferToAgent(agent *LLMAgent, targets []types.Agent, transferTool types.Tool) (string, error) {
	parent := agent.Spec().Parent()
	if agent.DisallowTransferToParent {
		parent = nil
	}

	var buf bytes.Buffer
	if err := transferToAgentPromptTmpl.Execute(&buf, struct {
		AgentName string
		Parent    types.Agent
		Targets   []types.Agent
		ToolName  string
	}{
		AgentName: agent.Spec().Name,
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
