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

// Package planner defines the interface and built-in implementations for agent planning.
package planner

import (
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/model"
)

// Planner allows an agent to generate plans for queries to guide its action.
type Planner interface {
	// BuildPlanningInstruction builds the system instruction to be appended to
	// the LLM request for planning, or returns an empty string if no instruction
	// is needed.
	BuildPlanningInstruction(ctx agent.ReadonlyContext, req *model.LLMRequest) string
	// ProcessPlanningResponse processes the LLM response parts for planning, or
	// returns nil if no processing is needed.
	ProcessPlanningResponse(ctx agent.Context, responseParts []*genai.Part) []*genai.Part
}
