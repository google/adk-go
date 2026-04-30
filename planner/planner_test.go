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

package planner_test

import (
	"strings"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/model"
	"google.golang.org/adk/planner"
)

func TestBuiltIn_NoInstruction(t *testing.T) {
	p := &planner.BuiltIn{}
	if got := p.BuildPlanningInstruction(nil, &model.LLMRequest{}); got != "" {
		t.Errorf("BuiltIn.BuildPlanningInstruction = %q, want empty", got)
	}
}

func TestBuiltIn_AppliesThinkingBudget(t *testing.T) {
	budget := int32(2048)
	p := &planner.BuiltIn{ThinkingBudget: &budget}
	req := &model.LLMRequest{}
	p.BuildPlanningInstruction(nil, req)
	if req.Config == nil || req.Config.ThinkingConfig == nil {
		t.Fatal("ThinkingConfig not set")
	}
	if req.Config.ThinkingConfig.ThinkingBudget == nil ||
		*req.Config.ThinkingConfig.ThinkingBudget != 2048 {
		t.Errorf("ThinkingBudget = %v, want 2048", req.Config.ThinkingConfig.ThinkingBudget)
	}
}

func TestPlanReAct_DefaultInstruction(t *testing.T) {
	p := &planner.PlanReAct{}
	got := p.BuildPlanningInstruction(nil, &model.LLMRequest{})
	for _, marker := range []string{"/*PLANNING*/", "/*REASONING*/", "/*ACTION*/", "/*FINAL_ANSWER*/"} {
		if !strings.Contains(got, marker) {
			t.Errorf("instruction missing marker %q", marker)
		}
	}
}

func TestPlanReAct_OverrideInstruction(t *testing.T) {
	p := &planner.PlanReAct{Instruction: "custom plan template"}
	got := p.BuildPlanningInstruction(nil, &model.LLMRequest{})
	if got != "custom plan template" {
		t.Errorf("instruction = %q", got)
	}
}

func TestPlanReAct_MarksThoughts(t *testing.T) {
	parts := []*genai.Part{
		{Text: "/*PLANNING*/ steps"},
		{Text: "/*REASONING*/ rationale"},
		{Text: "/*FINAL_ANSWER*/ the answer"},
	}
	p := &planner.PlanReAct{}
	out := p.ProcessPlanningResponse(nil, parts)
	if out == nil {
		t.Fatal("ProcessPlanningResponse returned nil")
	}
	if !out[0].Thought || !out[1].Thought {
		t.Errorf("planning + reasoning should be marked Thought")
	}
	if out[2].Thought {
		t.Errorf("final answer should not be marked Thought")
	}
}

func TestPlanReAct_NoFinalAnswerReturnsNil(t *testing.T) {
	parts := []*genai.Part{{Text: "no markers here"}}
	p := &planner.PlanReAct{}
	out := p.ProcessPlanningResponse(nil, parts)
	if out != nil {
		t.Errorf("expected nil, got %v", out)
	}
}
