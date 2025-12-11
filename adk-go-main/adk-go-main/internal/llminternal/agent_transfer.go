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

package llminternal

import (
	"bytes"
	"fmt"
	"slices"

	"github.com/google/safehtml/template"
	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/internal/agent/parentmap"
	"google.golang.org/adk/internal/toolinternal"
	"google.golang.org/adk/internal/utils"
	"google.golang.org/adk/model"
	"google.golang.org/adk/tool"
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

func AgentTransferRequestProcessor(ctx agent.InvocationContext, req *model.LLMRequest) error {
	// TODO: support agent types other than LLMAgent, that have parent/subagents?
	agent := ctx.Agent()
	if !shouldUseAutoFlow(agent) {
		return nil
	}

	parents := parentmap.FromContext(ctx)

	targets := transferTargets(agent, parents[agent.Name()])
	if len(targets) == 0 {
		return nil
	}

	// TODO(hyangah): why do we set this up in request processor
	// instead of registering this as a normal function tool of the Agent?
	transferToAgentTool := &TransferToAgentTool{}
	si, err := instructionsForTransferToAgent(agent, parents[agent.Name()], targets, transferToAgentTool)
	if err != nil {
		return err
	}
	utils.AppendInstructions(req, si)
	return appendTools(req, transferToAgentTool)
}

type TransferToAgentTool struct{}

// Description implements tool.Tool.
func (t *TransferToAgentTool) Description() string {
	return `Transfer the question to another agent.
This tool hands off control to another agent when it's more suitable to answer the user's question according to the agent's description.`
}

// Name implements tool.Tool.
func (t *TransferToAgentTool) Name() string {
	return "transfer_to_agent"
}

// IsLongRunning implements tool.Tool.
func (t *TransferToAgentTool) IsLongRunning() bool {
	return false
}

func (t *TransferToAgentTool) Declaration() *genai.FunctionDeclaration {
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
func (t *TransferToAgentTool) ProcessRequest(ctx tool.Context, req *model.LLMRequest) error {
	return appendTools(req, t)
}

// Run implements types.Tool.
func (t *TransferToAgentTool) Run(ctx tool.Context, args any) (map[string]any, error) {
	if args == nil {
		return nil, fmt.Errorf("missing argument")
	}
	m, ok := args.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("unexpected args type: %T", args)
	}
	agent, ok := m["agent_name"].(string)
	if !ok || agent == "" {
		return nil, fmt.Errorf("empty agent_name: %v", args)
	}
	ctx.Actions().TransferToAgent = agent
	return map[string]any{}, nil
}

var _ tool.Tool = (*TransferToAgentTool)(nil)

func transferTargets(agent, parent agent.Agent) []agent.Agent {
	targets := slices.Clone(agent.SubAgents())

	llmAgent := asLLMAgent(agent)
	llmParent := asLLMAgent(parent)

	if llmParent == nil {
		return targets
	}

	if !llmAgent.internal().DisallowTransferToParent {
		targets = append(targets, parent)
	}
	// For peer-agent transfers, it's only enabled when all below conditions are met:
	// - the parent agent is also of AutoFlow.
	// - DisallowTransferToPeers is false.
	if !llmAgent.internal().DisallowTransferToPeers {
		if shouldUseAutoFlow(parent) {
			for _, peer := range parent.SubAgents() {
				if peer.Name() != agent.Name() {
					targets = append(targets, peer)
				}
			}
		}
	}
	return targets
}

func asLLMAgent(agent agent.Agent) Agent {
	if agent == nil {
		return nil
	}
	if llmAgent, ok := agent.(Agent); ok {
		return llmAgent
	}
	return nil
}

func shouldUseAutoFlow(agent agent.Agent) bool {
	a := asLLMAgent(agent)
	if a == nil {
		return false
	}
	return len(agent.SubAgents()) != 0 || !a.internal().DisallowTransferToParent || !a.internal().DisallowTransferToPeers
}

// AppendTools appends the tools to the request.
// Appending duplicate tools or nameless tools is an error.
func appendTools(r *model.LLMRequest, tools ...tool.Tool) error {
	if r.Tools == nil {
		r.Tools = make(map[string]any)
	}

	var declarations []*genai.FunctionDeclaration

	for i, tool := range tools {
		if tool == nil || tool.Name() == "" {
			return fmt.Errorf("tools[%d] tool without name: %v", i, tool)
		}
		name := tool.Name()
		if _, ok := r.Tools[name]; ok {
			return fmt.Errorf("tools[%d] duplicate tool: %q", i, name)
		}
		r.Tools[name] = tool

		if fnTool, ok := tool.(toolinternal.FunctionTool); ok {
			if decl := fnTool.Declaration(); decl != nil {
				// TODO: verify for duplicates.
				declarations = append(declarations, decl)
			}
		}
	}
	if len(declarations) == 0 {
		return nil
	}
	if r.Config == nil {
		r.Config = &genai.GenerateContentConfig{}
	}
	// Find an existing genai.Tool with FunctionDeclarations
	var funcTool *genai.Tool
	for _, gt := range r.Config.Tools {
		if gt.FunctionDeclarations != nil {
			funcTool = gt
			break
		}
	}
	if funcTool != nil {
		funcTool.FunctionDeclarations = append(funcTool.FunctionDeclarations, declarations...)
	} else {
		r.Config.Tools = append(r.Config.Tools, &genai.Tool{
			FunctionDeclarations: declarations,
		})
	}
	return nil
}

var transferToAgentPromptTmpl = template.Must(
	template.New("transfer_to_agent_prompt").Parse(agentTransferInstructionTemplate))

func instructionsForTransferToAgent(curAgent, parent agent.Agent, targets []agent.Agent, transferTool tool.Tool) (string, error) {
	if asLLMAgent(curAgent).internal().DisallowTransferToParent {
		parent = nil
	}

	var buf bytes.Buffer
	if err := transferToAgentPromptTmpl.Execute(&buf, struct {
		AgentName string
		Parent    agent.Agent
		Targets   []agent.Agent
		ToolName  string
	}{
		AgentName: curAgent.Name(),
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
