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
	"fmt"
	"iter"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
)

// basicRequestProcessor populates the LLMRequest
// with the agent's LLM generation configs.
func basicRequestProcessor(ctx agent.InvocationContext, req *model.LLMRequest, f *Flow) iter.Seq2[*session.Event, error] {
	// reference: adk-python src/google/adk/flows/llm_flows/basic.py
	return func(yield func(*session.Event, error) bool) {
		llmAgent := asLLMAgent(ctx.Agent())
		if llmAgent == nil {
			return // do nothing.
		}

		state := llmAgent.internal()

		config, err := clone(state.GenerateContentConfig)
		if err != nil {
			yield(nil, fmt.Errorf("clone generation config: %w", err))
			return
		}
		req.Config = config
		if req.Config == nil {
			req.Config = &genai.GenerateContentConfig{}
		}

		// Set OutputSchema directly if no tools are present or native combo support exists.
		// Otherwise, OutputSchemaRequestProcessor will be used to provide a tool-based workaround.
		//
		// Task-mode agents skip OutputSchema configuration entirely:
		// structured output for tasks is collected via the FinishTaskTool's
		// declaration (the model emits the value inside the finish_task
		// FC args, not as a structured text response).
		if state.Mode != ModeTask && state.OutputSchema != nil && !needOutputSchemaProcessor(state) {
			// Unlike Python, isolate schema mutations made by model callbacks.
			schema, err := clone(state.OutputSchema)
			if err != nil {
				yield(nil, fmt.Errorf("clone output schema: %w", err))
				return
			}
			req.Config.ResponseSchema = schema
			req.Config.ResponseMIMEType = "application/json"
		}

		// TODO: missing features
		//  populate LLMRequest LiveConnectConfig setting
	}
}
