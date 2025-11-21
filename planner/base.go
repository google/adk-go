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

// Package planner provides interfaces and implementations for AI agent planning capabilities.
package planner

import (
	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/model"
)

// BasePlanner is the abstract base interface for all planners.
//
// The planner allows the agent to generate plans for the queries to guide its
// action.
type BasePlanner interface {
	// BuildPlanningInstruction builds the system instruction to be appended to the LLM request for planning.
	//
	// Args:
	//   readonlyContext: The readonly context of the invocation.
	//   llmRequest: The LLM request. Readonly.
	//
	// Returns:
	//   The planning system instruction, or empty string if no instruction is needed.
	BuildPlanningInstruction(readonlyContext agent.ReadonlyContext, llmRequest *model.LLMRequest) string

	// ProcessPlanningResponse processes the LLM response for planning.
	//
	// Args:
	//   callbackContext: The callback context of the invocation.
	//   responseParts: The LLM response parts. Readonly.
	//
	// Returns:
	//   The processed response parts, or nil if no processing is needed.
	ProcessPlanningResponse(callbackContext agent.CallbackContext, responseParts []*genai.Part) []*genai.Part
}
