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

// NewBuiltInPlanner creates a [Planner] that uses the model's built-in thinking
// features by applying cfg to each model request. An error will be returned by
// the model if thinking is configured for a model that does not support it.
//
// When cfg is non-nil, it overrides any ThinkingConfig already present on the
// request's GenerateContentConfig.
func NewBuiltInPlanner(cfg *genai.ThinkingConfig) Planner {
	return &builtInPlanner{thinkingConfig: cfg}
}

type builtInPlanner struct {
	thinkingConfig *genai.ThinkingConfig
}

// ApplyThinkingConfig applies the planner's ThinkingConfig to req, overriding
// any ThinkingConfig already set on req.Config.
func (p *builtInPlanner) ApplyThinkingConfig(req *model.LLMRequest) {
	if req == nil || p.thinkingConfig == nil {
		return
	}
	if req.Config == nil {
		req.Config = &genai.GenerateContentConfig{}
	}
	// Deliberately omits adk-python's debug log when overwriting an existing
	// ThinkingConfig because ADK Go library code does not write to a process-wide logger.
	cfgCopy := *p.thinkingConfig
	if p.thinkingConfig.ThinkingBudget != nil {
		budget := *p.thinkingConfig.ThinkingBudget
		cfgCopy.ThinkingBudget = &budget
	}
	req.Config.ThinkingConfig = &cfgCopy
}

func (p *builtInPlanner) BuildPlanningInstruction(agent.ReadonlyContext, *model.LLMRequest) string {
	return ""
}

func (p *builtInPlanner) ProcessPlanningResponse(agent.Context, []*genai.Part) []*genai.Part {
	return nil
}
