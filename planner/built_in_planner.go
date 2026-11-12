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

package planner

import (
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/model"
)

// BuiltInPlanner configures an LLM agent to use the model's built-in thinking
// feature via a genai.ThinkingConfig.
//
// Unlike PlanReActPlanner and custom Planner implementations, BuiltInPlanner
// does not append a planning instruction and does not clear Thought flags on
// request history parts. It implements Planner but not RewritingPlanner, so the
// flow does not rewrite model response parts for it. A struct that embeds
// *BuiltInPlanner and defines its own ProcessPlanningResponse implements
// RewritingPlanner, and the flow calls that method.
//
// If both llmagent.Config.GenerateContentConfig.ThinkingConfig and
// BuiltInPlanner.ThinkingConfig are non-nil, BuiltInPlanner.ThinkingConfig
// takes precedence and overwrites the request's ThinkingConfig on every model
// call.
//
// The zero value (ThinkingConfig == nil) is valid and acts as a no-op.
// BuiltInPlanner is safe for concurrent use as long as ThinkingConfig is not
// mutated concurrently.
type BuiltInPlanner struct {
	// ThinkingConfig is applied to each outbound model.LLMRequest.Config.
	// If nil, ApplyThinkingConfig leaves the request unchanged.
	ThinkingConfig *genai.ThinkingConfig
}

// NewBuiltInPlanner returns a non-nil *BuiltInPlanner whose ThinkingConfig is
// thinkingConfig. A nil thinkingConfig is allowed and gives a planner that
// changes no request.
func NewBuiltInPlanner(thinkingConfig *genai.ThinkingConfig) *BuiltInPlanner {
	return &BuiltInPlanner{ThinkingConfig: thinkingConfig}
}

// ApplyThinkingConfig sets req.Config.ThinkingConfig to p.ThinkingConfig,
// allocating req.Config if it is nil and overwriting any thinking config the
// request already has. Afterwards req.Config is non-nil and
// req.Config.ThinkingConfig is the same pointer as p.ThinkingConfig.
//
// p and req may be nil. When p, req or p.ThinkingConfig is nil,
// ApplyThinkingConfig leaves req unmodified.
func (p *BuiltInPlanner) ApplyThinkingConfig(req *model.LLMRequest) {
	if p == nil || req == nil || p.ThinkingConfig == nil {
		return
	}
	if req.Config == nil {
		req.Config = &genai.GenerateContentConfig{}
	}
	req.Config.ThinkingConfig = p.ThinkingConfig
}

// BuildPlanningInstruction implements Planner and always returns ("", nil).
func (p *BuiltInPlanner) BuildPlanningInstruction(ctx agent.ReadonlyContext, req *model.LLMRequest) (string, error) {
	return "", nil
}
