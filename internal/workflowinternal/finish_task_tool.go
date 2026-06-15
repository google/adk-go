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
	"google.golang.org/adk/internal/toolinternal/toolutils"
	"google.golang.org/adk/internal/utils"
	"google.golang.org/adk/model"
	"google.golang.org/adk/tool"
)

const (
	FinishTaskToolName      = "finish_task"
	FinishTaskSuccessResult = "Task completed."
)

// NewFinishTaskTool allows the model to signal that the agent has completed its
// task. On success it sets `tool_context.actions.finish_task` with a
// serialized `TaskResult` dict.
//
// Args:
//
//	taskAgent: The task agent this tool belongs to. The agent's
//	  `output_schema` is used for validation. If None, the default
//	  schema (a single `result` string) is used.
func NewFinishTaskTool(taskAgent agent.Agent) (tool.Tool, error) {
	llmAgentInternal, ok := taskAgent.(llminternal.Agent)
	if !ok {
		return nil, fmt.Errorf("NewFinishTaskTool: %q is not an LLMAgent", taskAgent.Name())
	}

	llmAgentState := llminternal.Reveal(llmAgentInternal)

	newTool := &FinishTaskTool{
		taskAgentName: taskAgent.Name(),
	}

	if llmAgentState.OutputSchema != nil {
		newTool.userSchema = llmAgentState.OutputSchema
	} else {
		newTool.userSchema = &genai.Schema{
			Type: genai.TypeObject,
			Properties: map[string]*genai.Schema{
				defaultWrapperKey: {
					Type:        genai.TypeString,
					Description: "A brief summary of what the agent accomplished.",
				},
			},
			Required: []string{defaultWrapperKey},
		}
	}

	// Function declaration parameters must be an OBJECT schema.
	if isObjectSchema(newTool.userSchema) {
		newTool.wrappedSchema = newTool.userSchema
	} else {
		newTool.wrapperKey = defaultWrapperKey
		newTool.wrappedSchema = &genai.Schema{
			Type: genai.TypeObject,
			Properties: map[string]*genai.Schema{
				newTool.wrapperKey: newTool.userSchema,
			},
			Required: []string{newTool.wrapperKey},
		}
	}

	newTool.description = "Signal that this agent has completed its delegated task. Call this when you have finished your delegated task."
	if llmAgentState.OutputSchema != nil {
		newTool.description += " Pass the required output data in the parameters."
	}

	return newTool, nil
}

type FinishTaskTool struct {
	taskAgentName string
	description   string
	// userSchema it's either LLMAgent.OutputSchema or the default schema if not set by the user.
	userSchema *genai.Schema
	// if userSchema is not an object, we wrap it into object using wrapperKey
	// As function call parameters must be an OBJECT schema, so non-object output schemas are
	// wrapped inside `{type: OBJECT, properties: {result: <schema>}, required:
	// [result]}` and the model-provided value is unwrapped before validation.
	wrappedSchema *genai.Schema
	wrapperKey    string
}

const defaultWrapperKey = "result"

func isObjectSchema(s *genai.Schema) bool {
	if s == nil {
		return false
	}
	return genai.Type(strings.ToUpper(string(s.Type))) == genai.TypeObject
}

func (t *FinishTaskTool) Name() string {
	return FinishTaskToolName
}

func (t *FinishTaskTool) Description() string {
	return t.description
}

func (t *FinishTaskTool) IsLongRunning() bool {
	return false
}

func (t *FinishTaskTool) WrapperKey() string {
	return t.wrapperKey
}

func (t *FinishTaskTool) Declaration() *genai.FunctionDeclaration {
	return &genai.FunctionDeclaration{
		Name:        t.Name(),
		Description: t.Description(),
		Parameters:  t.wrappedSchema,
	}
}

func (t *FinishTaskTool) ProcessRequest(ctx agent.Context, req *model.LLMRequest) error {
	instructions := `
Do NOT call 'finish_task' prematurely. Use your available tools to
fully complete every aspect of the delegated task first. If the
task is unclear, ask the user for clarification before proceeding.
Once the task is fully complete, call 'finish_task' by itself with
no accompanying text output.
`

	utils.AppendInstructions(req, instructions)

	return toolutils.PackTool(req, t)
}

func (t *FinishTaskTool) Run(ctx agent.Context, args any) (map[string]any, error) {
	if args == nil {
		return nil, fmt.Errorf("missing argument")
	}

	m, ok := args.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("unexpected args type: %T, expected map[string]any", args)
	}

	if err := t.validateArgs(m); err != nil {
		return map[string]any{
			"error": fmt.Sprintf(
				"Invoking `%s()` failed due to validation errors:\n%s\n"+
					"You could retry calling this tool, but it is IMPORTANT for you"+
					" to provide all the mandatory parameters with correct types.",
				t.Name(), err,
			),
		}, nil
	}

	return map[string]any{"result": FinishTaskSuccessResult}, nil
}

func (t *FinishTaskTool) validateArgs(args map[string]any) error {
	if t.wrapperKey == "" {
		return utils.ValidateMapOnSchema(args, t.userSchema, false)
	}

	return utils.ValidateMapOnSchema(
		args,
		t.wrappedSchema,
		false,
	)
}
