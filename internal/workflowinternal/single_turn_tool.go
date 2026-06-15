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

package workflowinternal

import (
	"fmt"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/internal/llminternal"
	"google.golang.org/adk/internal/toolinternal/toolutils"
	"google.golang.org/adk/internal/utils"
	"google.golang.org/adk/model"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/workflow"
)

type SingleTurnTool struct {
	agent           agent.Agent
	funcDeclaration *genai.FunctionDeclaration
}

func NewSingleTurnTool(a agent.Agent) (tool.Tool, error) {
	s := &SingleTurnTool{
		agent:           a,
		funcDeclaration: MakeFunctionDeclaration(a),
	}

	return s, nil
}

func (t *SingleTurnTool) Name() string {
	return t.agent.Name()
}

func (t *SingleTurnTool) Description() string {
	return t.agent.Description()
}

func (t *SingleTurnTool) IsLongRunning() bool {
	return false
}

func (t *SingleTurnTool) Run(toolCtx agent.Context, args any) (map[string]any, error) {
	margs, ok := args.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("single turn tool expects map[string]any arguments, got %T", args)
	}

	var nodeInput any

	if t.funcDeclaration.Parameters != nil {
		if err := utils.ValidateMapOnSchema(margs, t.funcDeclaration.Parameters, true); err != nil {
			return nil, fmt.Errorf("argument validation failed for agent %s: %w", t.agent.Name(), err)
		}
		nodeInput = margs
	} else {
		nodeInput = margs["request"]
	}

	// nc, ok := workflow.NodeContextFromGoContext(toolCtx)
	// if !ok {
	// 	return nil, fmt.Errorf("failed to infer node context")
	// }

	node, err := workflow.NewAgentNode(t.agent, workflow.NodeConfig{})
	if err != nil {
		return nil, fmt.Errorf("failed to create agent node: %w", err)
	}

	result, err := workflow.RunNode[any](toolCtx, node, nodeInput, workflow.WithUseSubBranch())
	if err != nil {
		return nil, fmt.Errorf("failed to run agent node: %w", err)
	}

	return map[string]any{"result": result}, nil
}

// Declaration returns the function declaration for the wrapped agent.
// It generates a function declaration based on the agent's input schema.
// If the agent does not have an input schema, a default schema with a
// "request" string parameter is used.
func (t *SingleTurnTool) Declaration() *genai.FunctionDeclaration {
	return t.funcDeclaration
}

func (t *SingleTurnTool) ProcessRequest(ctx agent.Context, req *model.LLMRequest) error {
	return toolutils.PackTool(req, t)
}

func getInputSchema(cur agent.Agent) *genai.Schema {
	llmAgent, ok := cur.(llminternal.Agent)
	if ok && llmAgent != nil {
		return llminternal.Reveal(llmAgent).InputSchema
	}

	if len(cur.SubAgents()) > 0 {
		return getInputSchema(cur.SubAgents()[0])
	}

	return nil
}

func getOutputSchema(cur agent.Agent) *genai.Schema {
	llmAgent, ok := cur.(llminternal.Agent)
	if ok && llmAgent != nil {
		return llminternal.Reveal(llmAgent).OutputSchema
	}

	if len(cur.SubAgents()) > 0 {
		return getOutputSchema(cur.SubAgents()[len(cur.SubAgents())-1])
	}

	return nil
}

// MakeFunctionDeclaration creates a function declaration for a given agent.
func MakeFunctionDeclaration(curAgent agent.Agent) *genai.FunctionDeclaration {
	decl := &genai.FunctionDeclaration{
		Name:        curAgent.Name(),
		Description: curAgent.Description(),
	}

	agentInputSchema := getInputSchema(curAgent)

	if agentInputSchema != nil {
		decl.Parameters = agentInputSchema
	} else {
		decl.Parameters = &genai.Schema{
			Type: "OBJECT",
			Properties: map[string]*genai.Schema{
				"request": {Type: "STRING"},
			},
			Required: []string{"request"},
		}
	}
	return decl
}
