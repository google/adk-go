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
	"errors"
	"fmt"
	"iter"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/internal/llminternal/googlellm"
	"google.golang.org/adk/v2/internal/toolinternal"
	"google.golang.org/adk/v2/internal/utils"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool/setmodelresponsetool"
	"google.golang.org/adk/v2/tool/toolutils"
)

const (
	instructionForProcessor = "IMPORTANT: You have access to other tools, but you must provide " +
		"your final response using the set_model_response tool with the " +
		"required structured format. After using any other tools needed " +
		"to complete the task, always call set_model_response with your " +
		"final answer in the specified schema format."
)

// outputSchemaRequestProcessor adds the set_model_response tool to handle structured output.
func outputSchemaRequestProcessor(ctx agent.InvocationContext, req *model.LLMRequest, f *Flow) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		llmAgent := asLLMAgent(ctx.Agent())
		if llmAgent == nil {
			return
		}

		state := llmAgent.internal()
		// Check if we need the processor in the first place.
		if !needOutputSchemaProcessor(state) {
			return
		}
		schema := outputSchemaFor(ctx, state)
		if schema == nil {
			return
		}

		// Add the set_model_response tool to handle structured output
		t, err := setmodelresponsetool.New(schema)
		if err != nil {
			yield(nil, fmt.Errorf("build set_model_response tool: %w", err))
			return
		}
		setResponseTool, ok := t.(toolutils.Tool)
		if !ok {
			yield(nil, errors.New("build set_model_response tool: the tool has no declaration"))
			return
		}
		if err := toolutils.PackTool(req, setResponseTool); err != nil {
			yield(nil, fmt.Errorf("failed to pack set_model_response tool: %w", err))
			return
		}

		// Add instruction about using the set_model_response tool
		utils.AppendInstructions(req, instructionForProcessor)
	}
}

// createFinalModelResponseEvent creates a final model response event from set_model_response JSON.
func createFinalModelResponseEvent(invocationContext agent.InvocationContext, response string) *session.Event {
	// Create a proper model response event
	finalEvent := session.NewEvent(invocationContext, invocationContext.InvocationID())
	finalEvent.Author = invocationContext.Agent().Name()
	finalEvent.Branch = invocationContext.Branch()
	finalEvent.Content = &genai.Content{
		Role:  "model",
		Parts: []*genai.Part{{Text: response}},
	}
	return finalEvent
}

// structuredFinalEvent returns the final model-response event for a function
// response event whose set_model_response call stored an answer, or nil when
// the event holds none.
func structuredFinalEvent(ctx agent.InvocationContext, respEv *session.Event) (*session.Event, error) {
	// A set_model_response response without a stored answer is the feedback
	// for an invalid call, so the model gets another turn.
	if respEv == nil || respEv.Actions.SetModelResponse == nil {
		return nil, nil
	}
	text, err := retrieveStructuredModelResponse(respEv)
	if err != nil || text == "" {
		return nil, err
	}
	return createFinalModelResponseEvent(ctx, text), nil
}

// retrieveStructuredModelResponse returns the JSON of the answer in an event
// that holds a set_model_response function response: the stored answer when
// there is one, else the function response itself. It returns "" when the
// event holds no such response.
func retrieveStructuredModelResponse(ev *session.Event) (string, error) {
	if ev == nil || ev.LLMResponse.Content == nil {
		return "", nil
	}

	for _, part := range ev.LLMResponse.Content.Parts {
		if part.FunctionResponse != nil && part.FunctionResponse.Name == setmodelresponsetool.Name {
			var answer any = part.FunctionResponse.Response
			if ev.Actions.SetModelResponse != nil {
				answer = ev.Actions.SetModelResponse
			}
			bytes, err := json.Marshal(answer)
			if err != nil {
				return "", fmt.Errorf("failed to marshal set_model_response: %w", err)
			}
			return string(bytes), nil
		}
	}

	return "", nil
}

func needOutputSchemaProcessor(state *State) bool {
	if state == nil || state.Model == nil {
		return false
	}
	// Task-mode agents already have a structured-output mechanism:
	// the auto-injected finish_task tool whose declaration mirrors
	// the agent's OutputSchema.
	if state.Mode == ModeTask {
		return false
	}
	hasTools := len(state.Tools) > 0 || len(state.Toolsets) > 0
	return hasTools && !canUseOutputSchemaWithTools(state.Model)
}

// canUseOutputSchemaWithTools reports whether llm accepts a native output
// schema and tools in one request. A model that implements
// model.CapabilitiesProvider decides for itself; any other model gets
// adk-python's fallback (models/base_llm.py:92-124), which allows it only for
// a Gemini model name on Vertex AI.
func canUseOutputSchemaWithTools(llm model.LLM) bool {
	if llm == nil {
		return false
	}
	if cp, ok := llm.(model.CapabilitiesProvider); ok {
		return cp.Capabilities().OutputSchemaAndTools
	}
	return googlellm.GeminiOutputSchemaAndTools(llm)
}

// setModelResponseTool builds the public set_model_response tool for its
// schema on each call and delegates to it. The flow packs the public tool
// directly; this type keeps the package's own tests of the tool running
// against the one implementation.
type setModelResponseTool struct {
	schema *genai.Schema
}

func (t *setModelResponseTool) build() (toolinternal.FunctionTool, error) {
	pt, err := setmodelresponsetool.New(t.schema)
	if err != nil {
		return nil, err
	}
	ft, ok := pt.(toolinternal.FunctionTool)
	if !ok {
		return nil, fmt.Errorf("set_model_response tool %T is not a function tool", pt)
	}
	return ft, nil
}

func (t *setModelResponseTool) Name() string {
	return setmodelresponsetool.Name
}

// Description returns "" when the tool cannot be built (a nil schema).
func (t *setModelResponseTool) Description() string {
	ft, err := t.build()
	if err != nil {
		return ""
	}
	return ft.Description()
}

func (t *setModelResponseTool) IsLongRunning() bool {
	return false
}

// Declaration returns nil when the tool cannot be built (a nil schema).
func (t *setModelResponseTool) Declaration() *genai.FunctionDeclaration {
	ft, err := t.build()
	if err != nil {
		return nil
	}
	return ft.Declaration()
}

func (t *setModelResponseTool) Run(ctx agent.Context, args any) (map[string]any, error) {
	ft, err := t.build()
	if err != nil {
		return nil, err
	}
	return ft.Run(ctx, args)
}
