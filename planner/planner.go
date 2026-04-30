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

// Package planner provides chain-of-thought planning hooks for LLM
// agents. Mirrors adk-python's google.adk.planners subsystem.
//
// A planner injects a planning system instruction into the LLM request
// (BuildPlanningInstruction) and post-processes the LLM's response parts
// (ProcessPlanningResponse). Two implementations ship:
//
//   - BuiltIn forwards the planning concern to the model's native
//     thinking config (Gemini's thinking budget). It does not modify the
//     prompt.
//   - PlanReAct adds an explicit Plan/Reasoning/Final-Answer template to
//     the system instruction and tags response parts so the runtime can
//     distinguish thoughts from final output.
package planner

import (
	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/model"
)

// Planner is the contract shared by all planners.
//
// BuildPlanningInstruction returns the system-instruction text to append
// to llmRequest, or "" when the planner has no instruction to add.
//
// ProcessPlanningResponse rewrites the model's response parts. It returns
// the new slice (possibly the same length, possibly different) or nil
// when no rewrite is needed.
type Planner interface {
	Name() string
	BuildPlanningInstruction(ctx agent.CallbackContext, req *model.LLMRequest) string
	ProcessPlanningResponse(ctx agent.CallbackContext, parts []*genai.Part) []*genai.Part
}

// BuiltIn is the no-op planner that defers planning to the model's
// thinking config. Use this when targeting Gemini models that natively
// support a thinking budget — the model handles the chain-of-thought
// internally and ADK doesn't need to scaffold one.
type BuiltIn struct {
	// ThinkingBudget, when non-nil, is propagated to llmRequest.Config
	// to enable / cap the model's native thinking.
	ThinkingBudget *int32
}

// Name implements Planner.
func (b *BuiltIn) Name() string { return "builtin" }

// BuildPlanningInstruction implements Planner. BuiltIn injects no
// instruction; the model is expected to plan natively.
func (b *BuiltIn) BuildPlanningInstruction(_ agent.CallbackContext, req *model.LLMRequest) string {
	if b.ThinkingBudget != nil {
		if req.Config == nil {
			req.Config = &genai.GenerateContentConfig{}
		}
		if req.Config.ThinkingConfig == nil {
			req.Config.ThinkingConfig = &genai.ThinkingConfig{}
		}
		req.Config.ThinkingConfig.ThinkingBudget = b.ThinkingBudget
	}
	return ""
}

// ProcessPlanningResponse implements Planner. BuiltIn doesn't rewrite
// parts; the model's output is final as-is.
func (b *BuiltIn) ProcessPlanningResponse(_ agent.CallbackContext, _ []*genai.Part) []*genai.Part {
	return nil
}

// PlanReAct is the explicit-prompting planner. It instructs the model to
// emit /*PLANNING*/ , /*REASONING*/ , /*ACTION*/ , and /*FINAL_ANSWER*/
// sections, then post-processes the response so callers can distinguish
// thoughts (planning + reasoning) from the final answer.
//
// Mirrors adk-python's PlanReActPlanner.
type PlanReAct struct {
	// Instruction overrides the default planning prompt template. When
	// empty, DefaultPlanReActInstruction is used.
	Instruction string
}

// DefaultPlanReActInstruction is the default planning prompt for
// PlanReAct. The text matches adk-python's plan_re_act_planner.py.
const DefaultPlanReActInstruction = `When you are given a task, follow these phases in order, separated by the markers shown:

/*PLANNING*/
List the steps you intend to take, in order, before doing any of them.

/*REASONING*/
Explain your reasoning at each step. Cite tool results, sources, or assumptions you are relying on.

/*ACTION*/
Make any tool calls necessary to advance the plan.

/*FINAL_ANSWER*/
Provide the final answer to the user. Only output what the user should see in this section.`

// Name implements Planner.
func (p *PlanReAct) Name() string { return "plan_re_act" }

// BuildPlanningInstruction implements Planner.
func (p *PlanReAct) BuildPlanningInstruction(_ agent.CallbackContext, _ *model.LLMRequest) string {
	if p.Instruction != "" {
		return p.Instruction
	}
	return DefaultPlanReActInstruction
}

// ProcessPlanningResponse marks any text part appearing before the
// /*FINAL_ANSWER*/ marker as a thought (genai.Part.Thought = true) so
// downstream consumers can hide planning text from end users by default.
//
// Returns nil when no /*FINAL_ANSWER*/ marker is present (the model's
// output is treated as already-final in that case).
func (p *PlanReAct) ProcessPlanningResponse(_ agent.CallbackContext, parts []*genai.Part) []*genai.Part {
	finalIdx := indexOfFinalAnswer(parts)
	if finalIdx < 0 {
		return nil
	}
	out := make([]*genai.Part, len(parts))
	for i, part := range parts {
		if part == nil {
			out[i] = nil
			continue
		}
		clone := *part
		if i < finalIdx {
			clone.Thought = true
		}
		out[i] = &clone
	}
	return out
}

// indexOfFinalAnswer returns the index of the first part whose Text
// contains "/*FINAL_ANSWER*/", or -1 when no such part exists.
func indexOfFinalAnswer(parts []*genai.Part) int {
	for i, p := range parts {
		if p == nil {
			continue
		}
		if containsFinalAnswerMarker(p.Text) {
			return i
		}
	}
	return -1
}

func containsFinalAnswerMarker(s string) bool {
	const marker = "/*FINAL_ANSWER*/"
	for i := 0; i+len(marker) <= len(s); i++ {
		if s[i:i+len(marker)] == marker {
			return true
		}
	}
	return false
}
