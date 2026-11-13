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
	"context"
	"fmt"
	"strings"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/internal/llminternal"
	"google.golang.org/adk/v2/internal/llminternal/googlellm"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/toolutils"
)

type TaskAgentTool struct {
	agent agent.Agent
}

func NewTaskAgentTool(curAgent agent.Agent) (tool.Tool, error) {
	if curAgent == nil {
		return nil, fmt.Errorf("NewTaskAgentTool: agent is nil")
	}

	return &TaskAgentTool{agent: curAgent}, nil
}

// Declaration builds the declaration on each call: the sub-agent's model can
// come from an ancestor constructed after the tool.
func (t *TaskAgentTool) Declaration() *genai.FunctionDeclaration {
	return createTaskAgentFuncDeclaration(t.agent)
}

func (t *TaskAgentTool) Run(toolCtx agent.Context, args any) (map[string]any, error) {
	// Framework handles task delegation dispatch directly via the
	// LlmAgent chat wrapper.
	return nil, nil
}

// DefersResponse implements toolinternal.ResponseDeferrer: the
// FunctionResponse for a task delegation is produced by the chat
// wrapper (synthesizeTaskFREvent), not by the standard tool-execution
// pipeline.
func (t *TaskAgentTool) DefersResponse() bool {
	return true
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

func (t *TaskAgentTool) ProcessRequest(ctx agent.Context, req *model.LLMRequest) error {
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

	agentInputSchema := InputSchema(curAgent)
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
		// A model that does not resolve is treated as one that is not on the
		// Gemini API; the step that calls the model reports the error.
		resolved, err := llminternal.ResolveModel(context.Background(), curAgent)
		if err != nil {
			resolved = nil
		}

		if !googlellm.IsGeminiAPIVariant(resolved) {
			if hasOutputSchema(curAgent) {
				decl.ResponseJsonSchema = &genai.Schema{Type: "OBJECT"}
			} else {
				decl.ResponseJsonSchema = &genai.Schema{Type: "STRING"}
			}
		}
	}

	return decl
}
