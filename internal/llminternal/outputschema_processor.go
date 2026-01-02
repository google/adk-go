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
	"encoding/json"
	"fmt"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/internal/llminternal/googlellm"
	"google.golang.org/adk/internal/toolinternal/toolutils"
	"google.golang.org/adk/internal/utils"
	"google.golang.org/adk/model"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
)

const (
	instructionForProcessor = "IMPORTANT: You have access to other tools, but you must provide " +
		"your final response using the set_model_response tool with the " +
		"required structured format. After using any other tools needed " +
		"to complete the task, always call set_model_response with your " +
		"final answer in the specified schema format."
)

// outputSchemaRequestProcessor adds the set_model_response tool to handle structured output.
func outputSchemaRequestProcessor(ctx agent.InvocationContext, req *model.LLMRequest) error {
	llmAgent := asLLMAgent(ctx.Agent())
	if llmAgent == nil {
		return nil
	}

	state := llmAgent.internal()
	// Check if we need the processor in the first place.
	if state.OutputSchema == nil || !needOutputSchemaProcessor(state) {
		return nil
	}

	// Add the set_model_response tool to handle structured output
	setResponseTool := &setModelResponseTool{schema: state.OutputSchema}
	if err := toolutils.PackTool(req, setResponseTool); err != nil {
		return fmt.Errorf("failed to pack set_model_response tool: %w", err)
	}

	// Add instruction about using the set_model_response tool
	utils.AppendInstructions(req, instructionForProcessor)

	return nil
}

// createFinalModelResponseEvent creates a final model response event from set_model_response JSON.
func createFinalModelResponseEvent(invocationContext agent.InvocationContext, response string) *session.Event {
	// Create a proper model response event
	finalEvent := session.NewEvent(invocationContext.InvocationID())
	finalEvent.Author = invocationContext.Agent().Name()
	finalEvent.Branch = invocationContext.Branch()
	finalEvent.Content = &genai.Content{
		Role:  "model",
		Parts: []*genai.Part{{Text: response}},
	}
	return finalEvent
}

// retrieveStructuredModelResponse checks if function response contains set_model_response tool and extract JSON.
func retrieveStructuredModelResponse(ev *session.Event) (string, error) {
	if ev == nil || ev.LLMResponse.Content == nil {
		return "", fmt.Errorf("event or LLMResponse.Content is nil")
	}

	for _, part := range ev.LLMResponse.Content.Parts {
		if part.FunctionResponse != nil && part.FunctionResponse.Name == "set_model_response" {
			bytes, err := json.Marshal(part.FunctionResponse.Response)
			if err != nil {
				return "", fmt.Errorf("failed to marshal set_model_response: %w", err)
			}
			return string(bytes), nil
		}
	}

	return "", fmt.Errorf("set_model_response tool not found")
}

func needOutputSchemaProcessor(state *State) bool {
	if state == nil || state.Model == nil {
		return false
	}
	hasTools := len(state.Tools) > 0 || len(state.Toolsets) > 0
	canUseOutputSchemaWithTools := googlellm.CanGeminiModelUseOutputSchemaWithTools(state.Model.Name())
	return hasTools && !canUseOutputSchemaWithTools
}

// setModelResponseTool implements tool.Tool and toolinternal.FunctionTool.
type setModelResponseTool struct {
	schema *genai.Schema
}

func (t *setModelResponseTool) Name() string {
	return "set_model_response"
}

func (t *setModelResponseTool) Description() string {
	return "Set your final response using the required output schema. Use this tool to provide your final structured answer instead of outputting text directly."
}

func (t *setModelResponseTool) IsLongRunning() bool {
	return false
}

func (t *setModelResponseTool) Declaration() *genai.FunctionDeclaration {
	return &genai.FunctionDeclaration{
		Name:                 t.Name(),
		Description:          t.Description(),
		ParametersJsonSchema: t.schema,
	}
}

func (t *setModelResponseTool) Run(ctx tool.Context, args any) (map[string]any, error) {
	m, ok := args.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("unexpected args type for set_model_response: %T", args)
	}
	if err := utils.ValidateMapOnSchema(m, t.schema, false); err != nil {
		return nil, fmt.Errorf("invalid output schema: %w", err)
	}
	return m, nil
}
