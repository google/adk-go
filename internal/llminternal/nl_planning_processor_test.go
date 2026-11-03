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

package llminternal

import (
	"slices"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/planner"
)

type customTestPlanner struct {
	instruction          string
	processCalled        bool
	receivedResponsePart []*genai.Part
	replacementParts     []*genai.Part
}

func (p *customTestPlanner) BuildPlanningInstruction(agent.ReadonlyContext, *model.LLMRequest) string {
	return p.instruction
}

func (p *customTestPlanner) ProcessPlanningResponse(_ agent.Context, responseParts []*genai.Part) []*genai.Part {
	p.processCalled = true
	p.receivedResponsePart = responseParts
	if p.replacementParts != nil {
		return p.replacementParts
	}
	return responseParts
}

func newTestPlannerInvocationContext(t *testing.T, p planner.Planner) agent.InvocationContext {
	t.Helper()
	baseAgent, err := agent.New(agent.Config{Name: "test_agent"})
	if err != nil {
		t.Fatalf("agent.New: %v", err)
	}
	mock := &mockLLMAgent{
		Agent: baseAgent,
		s: &State{
			Planner: p,
		},
	}
	return icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{
		Agent: mock,
	})
}

func TestNLPlanningRequestProcessor(t *testing.T) {
	t.Parallel()

	// Corresponds to: test_built_in_planner_content_list_unchanged
	t.Run("built-in planner leaves content list unchanged", func(t *testing.T) {
		t.Parallel()

		p := planner.NewBuiltInPlanner(&genai.ThinkingConfig{IncludeThoughts: true})
		ctx := newTestPlannerInvocationContext(t, p)
		req := &model.LLMRequest{
			Contents: []*genai.Content{
				genai.NewContentFromText("Hello", genai.RoleUser),
				{
					Role: genai.RoleModel,
					Parts: []*genai.Part{
						{Text: "thinking...", Thought: true},
						{Text: "Here is my response"},
					},
				},
				genai.NewContentFromText("Follow up", genai.RoleUser),
			},
		}
		wantContents := clone(req.Contents)

		for ev, err := range nlPlanningRequestProcessor(ctx, req, &Flow{}) {
			if ev != nil || err != nil {
				t.Fatalf("nlPlanningRequestProcessor yielded (%v, %v), want none", ev, err)
			}
		}

		if diff := cmp.Diff(wantContents, req.Contents); diff != "" {
			t.Errorf("req.Contents modified (-want +got):\n%s", diff)
		}
	})

	// Corresponds to: test_built_in_planner_apply_thinking_config_called
	t.Run("built-in planner applies thinking config and overrides existing config", func(t *testing.T) {
		t.Parallel()

		plannerBudget := int32(2048)
		existingBudget := int32(512)
		wantCfg := &genai.ThinkingConfig{IncludeThoughts: true, ThinkingBudget: &plannerBudget}
		p := planner.NewBuiltInPlanner(wantCfg)
		ctx := newTestPlannerInvocationContext(t, p)
		req := &model.LLMRequest{
			Config: &genai.GenerateContentConfig{
				ThinkingConfig: &genai.ThinkingConfig{ThinkingBudget: &existingBudget},
			},
		}

		for ev, err := range nlPlanningRequestProcessor(ctx, req, &Flow{}) {
			if ev != nil || err != nil {
				t.Fatalf("nlPlanningRequestProcessor yielded (%v, %v), want none", ev, err)
			}
		}

		if diff := cmp.Diff(wantCfg, req.Config.ThinkingConfig); diff != "" {
			t.Errorf("req.Config.ThinkingConfig mismatch (-want +got):\n%s", diff)
		}
		if req.Config.SystemInstruction != nil {
			t.Errorf("req.Config.SystemInstruction = %v, want nil", req.Config.SystemInstruction)
		}
	})

	// Corresponds to: test_plan_react_planner_instruction_appended
	t.Run("plan-react planner appends planning instruction", func(t *testing.T) {
		t.Parallel()

		p := planner.NewPlanReActPlanner()
		ctx := newTestPlannerInvocationContext(t, p)
		req := &model.LLMRequest{
			Config: &genai.GenerateContentConfig{
				SystemInstruction: genai.NewContentFromText("Original instruction", genai.RoleUser),
			},
		}

		for ev, err := range nlPlanningRequestProcessor(ctx, req, &Flow{}) {
			if ev != nil || err != nil {
				t.Fatalf("nlPlanningRequestProcessor yielded (%v, %v), want none", ev, err)
			}
		}

		if req.Config == nil || req.Config.SystemInstruction == nil || len(req.Config.SystemInstruction.Parts) == 0 {
			t.Fatal("req.Config.SystemInstruction is empty")
		}
		got := req.Config.SystemInstruction.Parts[0].Text
		if !strings.HasPrefix(got, "Original instruction\n\n") {
			t.Errorf("SystemInstruction = %q, want prefix %q", got, "Original instruction\n\n")
		}
		if !strings.Contains(got, "/*PLANNING*/") {
			t.Errorf("SystemInstruction = %q, want planning instruction appended", got)
		}
	})

	// Corresponds to: test_remove_thought_from_request_with_thoughts
	t.Run("plan-react planner removes thought flags from content parts", func(t *testing.T) {
		t.Parallel()

		p := planner.NewPlanReActPlanner()
		ctx := newTestPlannerInvocationContext(t, p)
		req := &model.LLMRequest{
			Contents: []*genai.Content{
				genai.NewContentFromText("initial query", genai.RoleUser),
				{
					Role: genai.RoleModel,
					Parts: []*genai.Part{
						{Text: "Text with thought", Thought: true},
						{Text: "Regular text"},
					},
				},
				genai.NewContentFromText("follow up", genai.RoleUser),
			},
		}

		for ev, err := range nlPlanningRequestProcessor(ctx, req, &Flow{}) {
			if ev != nil || err != nil {
				t.Fatalf("nlPlanningRequestProcessor yielded (%v, %v), want none", ev, err)
			}
		}

		for ci, content := range req.Contents {
			for pi, part := range content.Parts {
				if part.Thought {
					t.Errorf("req.Contents[%d].Parts[%d].Thought = true, want false", ci, pi)
				}
			}
		}
	})

	// Corresponds to: test_custom_planner_instruction_appended
	t.Run("custom planner instruction is appended", func(t *testing.T) {
		t.Parallel()

		p := &customTestPlanner{instruction: "Custom instruction"}
		ctx := newTestPlannerInvocationContext(t, p)
		req := &model.LLMRequest{}

		for ev, err := range nlPlanningRequestProcessor(ctx, req, &Flow{}) {
			if ev != nil || err != nil {
				t.Fatalf("nlPlanningRequestProcessor yielded (%v, %v), want none", ev, err)
			}
		}

		want := genai.NewContentFromText("Custom instruction", genai.RoleUser)
		if diff := cmp.Diff(want, req.Config.SystemInstruction); diff != "" {
			t.Errorf("req.Config.SystemInstruction mismatch (-want +got):\n%s", diff)
		}
	})

	// Corresponds to: test_custom_planner_removes_thought_from_request
	t.Run("custom planner removes thought from request", func(t *testing.T) {
		t.Parallel()

		p := &customTestPlanner{instruction: "Custom instruction"}
		ctx := newTestPlannerInvocationContext(t, p)
		req := &model.LLMRequest{
			Contents: []*genai.Content{
				genai.NewContentFromText("initial query", genai.RoleUser),
				{
					Role: genai.RoleModel,
					Parts: []*genai.Part{
						{Text: "Text with thought", Thought: true},
						{Text: "Regular text"},
					},
				},
			},
		}

		for ev, err := range nlPlanningRequestProcessor(ctx, req, &Flow{}) {
			if ev != nil || err != nil {
				t.Fatalf("nlPlanningRequestProcessor yielded (%v, %v), want none", ev, err)
			}
		}

		for ci, content := range req.Contents {
			for pi, part := range content.Parts {
				if part.Thought {
					t.Errorf("req.Contents[%d].Parts[%d].Thought = true, want false", ci, pi)
				}
			}
		}
	})
}

func TestNLPlanningResponseProcessor(t *testing.T) {
	t.Parallel()

	// Corresponds to: test_process_planning_response_not_called_without_override
	t.Run("built-in planner leaves response parts unchanged", func(t *testing.T) {
		t.Parallel()

		p := planner.NewBuiltInPlanner(&genai.ThinkingConfig{IncludeThoughts: true})
		ctx := newTestPlannerInvocationContext(t, p)
		originalParts := []*genai.Part{
			{Text: "/*PLANNING*/plan", Thought: true},
			{Text: "Here is my response"},
		}
		resp := &model.LLMResponse{
			Content: &genai.Content{
				Role:  genai.RoleModel,
				Parts: slices.Clone(originalParts),
			},
		}

		if err := nlPlanningResponseProcessor(ctx, &model.LLMRequest{}, resp); err != nil {
			t.Fatalf("nlPlanningResponseProcessor failed: %v", err)
		}
		if diff := cmp.Diff(originalParts, resp.Content.Parts); diff != "" {
			t.Errorf("resp.Content.Parts modified (-want +got):\n%s", diff)
		}
	})

	t.Run("plan-react planner rewrites response parts", func(t *testing.T) {
		t.Parallel()

		p := planner.NewPlanReActPlanner()
		ctx := newTestPlannerInvocationContext(t, p)
		resp := &model.LLMResponse{
			Content: &genai.Content{
				Role: genai.RoleModel,
				Parts: []*genai.Part{
					{Text: "/*PLANNING*/Step 1/*FINAL_ANSWER*/Done."},
				},
			},
		}

		if err := nlPlanningResponseProcessor(ctx, &model.LLMRequest{}, resp); err != nil {
			t.Fatalf("nlPlanningResponseProcessor failed: %v", err)
		}
		want := []*genai.Part{
			{Text: "Step 1", Thought: true},
			{Text: "Done."},
		}
		if diff := cmp.Diff(want, resp.Content.Parts); diff != "" {
			t.Errorf("resp.Content.Parts mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("custom planner response processor is called", func(t *testing.T) {
		t.Parallel()

		wantParts := []*genai.Part{{Text: "rewritten"}}
		p := &customTestPlanner{replacementParts: wantParts}
		ctx := newTestPlannerInvocationContext(t, p)
		inputParts := []*genai.Part{{Text: "original"}}
		resp := &model.LLMResponse{
			Content: &genai.Content{
				Role:  genai.RoleModel,
				Parts: inputParts,
			},
		}

		if err := nlPlanningResponseProcessor(ctx, &model.LLMRequest{}, resp); err != nil {
			t.Fatalf("nlPlanningResponseProcessor failed: %v", err)
		}
		if !p.processCalled {
			t.Fatal("customTestPlanner.ProcessPlanningResponse was not called")
		}
		if diff := cmp.Diff(inputParts, p.receivedResponsePart); diff != "" {
			t.Errorf("receivedResponsePart mismatch (-want +got):\n%s", diff)
		}
		if diff := cmp.Diff(wantParts, resp.Content.Parts); diff != "" {
			t.Errorf("resp.Content.Parts mismatch (-want +got):\n%s", diff)
		}
	})
}
