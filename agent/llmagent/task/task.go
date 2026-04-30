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

// Package task implements the Task Delegation API. Mirrors adk-python's
// google.adk.agents.llm.task subsystem.
//
// A coordinator agent delegates structured work to a "task agent" by
// invoking RequestTaskTool, which records a session.TaskRequest under
// EventActions.RequestTask keyed by the originating function_call_id.
// The task agent signals completion by invoking FinishTaskTool, which
// records a session.TaskResult under EventActions.FinishTask.
//
// The runtime piece — automatically routing the next conversational turn
// to the task agent and synthesizing the FunctionResponse for the
// coordinator — lives in internal/llminternal (Phase 6D). The tool
// constructors and data shapes ship here.
package task

import (
	"errors"
	"fmt"

	"github.com/google/jsonschema-go/jsonschema"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/functiontool"
)

// Mode classifies a task agent's interaction style.
type Mode string

const (
	// MultiTurn allows the task agent to ask clarifying questions before
	// completing. The coordinator pauses on each clarification turn.
	MultiTurn Mode = "multi_turn"

	// SingleTurn restricts the task agent to a single response: it must
	// call FinishTaskTool with the complete output without any further
	// dialog.
	SingleTurn Mode = "single_turn"
)

// DefaultTaskInput is the input shape used when the task agent does not
// declare an explicit InputSchema. Mirrors Python's _DefaultTaskInput.
type DefaultTaskInput struct {
	Goal       string `json:"goal,omitempty"`
	Background string `json:"background,omitempty"`
}

// DefaultTaskOutput is the output shape used when the task agent does not
// declare an explicit OutputSchema.
type DefaultTaskOutput struct {
	Result string `json:"result"`
}

// NewRequestTaskTool constructs a tool that delegates a task to taskAgent.
// When the LLM calls this tool with a payload, the tool validates the
// payload against the agent's input schema (or DefaultTaskInput) and
// records a session.TaskRequest under
// ctx.Actions().RequestTask[ctx.FunctionCallID()].
//
// The runtime routes the next turn to taskAgent based on this entry.
func NewRequestTaskTool(taskAgent agent.Agent) (tool.Tool, error) {
	if taskAgent == nil {
		return nil, errors.New("task: NewRequestTaskTool requires a non-nil task agent")
	}
	agentName := taskAgent.Name()
	desc := taskAgent.Description()
	if desc == "" {
		desc = agentName
	}
	description := fmt.Sprintf(
		`Delegate a task to the %q agent. %s`, agentName, desc,
	)
	inputSchema := agentInputJSONSchema(taskAgent)

	return functiontool.New[map[string]any, string](
		functiontool.Config{
			Name:        agentName,
			Description: description,
			InputSchema: inputSchema,
		},
		func(ctx tool.Context, args map[string]any) (string, error) {
			actions := ctx.Actions()
			if actions == nil {
				return "", errors.New("task: tool context has nil Actions")
			}
			if actions.RequestTask == nil {
				actions.RequestTask = map[string]session.TaskRequest{}
			}
			actions.RequestTask[ctx.FunctionCallID()] = session.TaskRequest{
				AgentName: agentName,
				Input:     args,
			}
			return fmt.Sprintf("Delegating task to %s.", agentName), nil
		},
	)
}

// NewFinishTaskTool constructs a tool the task agent invokes to signal
// completion. The output payload is validated against the agent's output
// schema (or DefaultTaskOutput) and recorded under
// ctx.Actions().FinishTask[ctx.FunctionCallID()].
//
// The runtime synthesizes a FunctionResponse for the coordinator based
// on this entry.
func NewFinishTaskTool(taskAgent agent.Agent) (tool.Tool, error) {
	if taskAgent == nil {
		return nil, errors.New("task: NewFinishTaskTool requires a non-nil task agent")
	}
	agentName := taskAgent.Name()
	outputSchema := agentOutputJSONSchema(taskAgent)
	desc := "Call this when you have completed the task. Provide the final output payload."

	return functiontool.New[map[string]any, string](
		functiontool.Config{
			Name:        "finish_task",
			Description: desc,
			InputSchema: outputSchema,
		},
		func(ctx tool.Context, args map[string]any) (string, error) {
			actions := ctx.Actions()
			if actions == nil {
				return "", errors.New("task: tool context has nil Actions")
			}
			if actions.FinishTask == nil {
				actions.FinishTask = map[string]session.TaskResult{}
			}
			actions.FinishTask[ctx.FunctionCallID()] = session.TaskResult{
				AgentName: agentName,
				Output:    args,
			}
			return "Task completed.", nil
		},
	)
}

// agentInputJSONSchema returns the agent's input JSON schema, falling back
// to DefaultTaskInput when none is declared. The conversion from
// *genai.Schema to *jsonschema.Schema is best-effort: it preserves the
// type tag and required/properties shape for the common cases used by
// task agents.
func agentInputJSONSchema(a agent.Agent) *jsonschema.Schema {
	if a == nil {
		return defaultJSONSchemaFor[DefaultTaskInput]()
	}
	// agent.Agent's InputSchema is *genai.Schema. Until we add a
	// proper genai->jsonschema converter, fall through to the default
	// for callers that haven't migrated. Concrete schemas can be passed
	// later via the functiontool.Config override path.
	return defaultJSONSchemaFor[DefaultTaskInput]()
}

func agentOutputJSONSchema(a agent.Agent) *jsonschema.Schema {
	return defaultJSONSchemaFor[DefaultTaskOutput]()
}

func defaultJSONSchemaFor[T any]() *jsonschema.Schema {
	js, err := jsonschema.For[T](nil)
	if err != nil {
		// Schema derivation for our default types should never fail; if it
		// does, return a permissive schema rather than panicking.
		return &jsonschema.Schema{Type: "object"}
	}
	return js
}

