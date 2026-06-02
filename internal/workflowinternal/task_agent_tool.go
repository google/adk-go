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
	"strings"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/internal/llminternal"
	"google.golang.org/adk/internal/llminternal/googlellm"
	"google.golang.org/adk/internal/toolinternal/toolutils"
	"google.golang.org/adk/model"
	"google.golang.org/adk/tool"
)

type TaskAgentTool struct {
	agent           agent.Agent
	funcDeclaration *genai.FunctionDeclaration
}

func NewTaskAgentTool(curAgent agent.Agent) (tool.Tool, error) {
	if curAgent == nil {
		return nil, fmt.Errorf("NewTaskAgentTool: agent is nil")
	}

	return &TaskAgentTool{
		agent:           curAgent,
		funcDeclaration: createTaskAgentFuncDeclaration(curAgent),
	}, nil
}

func (t *TaskAgentTool) Declaration() *genai.FunctionDeclaration {
	return t.funcDeclaration
}

func (t *TaskAgentTool) Run(toolCtx tool.Context, args any) (map[string]any, error) {
	// Framework handles task delegation dispatch directly via the wrapper.
	// TODO: add _defer_response logic.
	return nil, nil
}

func (t *TaskAgentTool) Name() string {
	return t.agent.Name()
}

func (t *TaskAgentTool) Description() string {
	return t.agent.Description()
}

func (t *TaskAgentTool) IsLongRunning() bool {
	return false
}

func (t *TaskAgentTool) ProcessRequest(ctx tool.Context, req *model.LLMRequest) error {
	return toolutils.PackTool(req, t)
}

func createTaskAgentFuncDeclaration(curAgent agent.Agent) *genai.FunctionDeclaration {
	decl := &genai.FunctionDeclaration{
		Name: curAgent.Name(),
		Description: strings.TrimSpace(
			fmt.Sprintf(
				"%s\nIMPORTANT: This tool delegates execution to a specialized agent. Do NOT call this tool in parallel with any other tools.",
				curAgent.Description())),
	}

	agentInputSchema := getInputSchema(curAgent)
	if agentInputSchema != nil {
		decl.Parameters = agentInputSchema
	} else {
		decl.Parameters = &genai.Schema{
			Type: "OBJECT",
			Properties: map[string]*genai.Schema{
				"request": {
					Type:        "STRING",
					Description: "Detailed instructions or context for the task sub-agent.",
				},
			},
			Required: []string{"request"},
		}
	}

	if llmAgent, ok := curAgent.(llminternal.Agent); ok && llmAgent != nil {
		agentState := llminternal.Reveal(llmAgent)

		if !googlellm.IsGeminiAPIVariant(agentState.Model) {
			outputSchema := getOutputSchema(curAgent)
			if outputSchema != nil {
				decl.ResponseJsonSchema = &genai.Schema{Type: "OBJECT"}
			} else {
				decl.ResponseJsonSchema = &genai.Schema{Type: "STRING"}
			}
		}
	}

	return decl
}
