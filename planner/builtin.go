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

package planner

import (
	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/model"
)

// BuiltInPlanner is the built-in planner that uses model's built-in thinking features.
type BuiltInPlanner struct {
	// ThinkingConfig is the config for model built-in thinking features. An error
	// will be returned if this field is set for models that don't support
	// thinking.
	ThinkingConfig *genai.ThinkingConfig
}

// NewBuiltInPlanner initializes the built-in planner.
//
// Args:
//
//	thinkingConfig: Config for model built-in thinking features. An error
//	  will be returned if this field is set for models that don't support
//	  thinking.
func NewBuiltInPlanner(thinkingConfig *genai.ThinkingConfig) *BuiltInPlanner {
	return &BuiltInPlanner{
		ThinkingConfig: thinkingConfig,
	}
}

// ApplyThinkingConfig applies the thinking config to the LLM request.
//
// Args:
//
//	llmRequest: The LLM request to apply the thinking config to.
func (b *BuiltInPlanner) ApplyThinkingConfig(llmRequest *model.LLMRequest) {
	if b.ThinkingConfig != nil {
		if llmRequest.Config == nil {
			llmRequest.Config = &genai.GenerateContentConfig{}
		}
		llmRequest.Config.ThinkingConfig = b.ThinkingConfig
	}
}

// BuildPlanningInstruction implements BasePlanner.
func (b *BuiltInPlanner) BuildPlanningInstruction(readonlyContext agent.ReadonlyContext, llmRequest *model.LLMRequest) string {
	// Built-in planner doesn't add planning instructions
	return ""
}

// ProcessPlanningResponse implements BasePlanner.
func (b *BuiltInPlanner) ProcessPlanningResponse(callbackContext agent.CallbackContext, responseParts []*genai.Part) []*genai.Part {
	// Built-in planner doesn't process planning response
	return nil
}
