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
	"bytes"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/planner"
)

func functionCallNames(parts []*genai.Part) []string {
	var names []string
	for _, p := range parts {
		if p != nil && p.FunctionCall != nil {
			names = append(names, p.FunctionCall.Name)
		}
	}
	return names
}

func TestPlanReActPlanner_ProcessPlanningResponse(t *testing.T) {
	t.Parallel()

	// Corresponds to: test_strips_planning_tag_from_thought_part
	t.Run("strips planning tag from thought part", func(t *testing.T) {
		t.Parallel()

		p := planner.NewPlanReActPlanner()
		firstInput := &genai.Part{
			Text:             "/*PLANNING*/Step 1: look it up.",
			ThoughtSignature: []byte("sig1"),
		}
		responseParts := []*genai.Part{
			firstInput,
			{
				Text:             "/*REASONING*/I need to call the tool.",
				ThoughtSignature: []byte("sig2"),
			},
			genai.NewPartFromFunctionCall("lookup", map[string]any{"q": "test"}),
		}

		result := p.ProcessPlanningResponse(nil, responseParts)

		var textParts []*genai.Part
		for _, part := range result {
			if part.Text != "" {
				textParts = append(textParts, part)
			}
		}
		if len(textParts) != 2 {
			t.Fatalf("len(textParts) = %d, want 2", len(textParts))
		}
		for _, part := range textParts {
			if strings.Contains(part.Text, "/*PLANNING*/") {
				t.Errorf("part.Text = %q, want /*PLANNING*/ stripped", part.Text)
			}
			if strings.Contains(part.Text, "/*REASONING*/") {
				t.Errorf("part.Text = %q, want /*REASONING*/ stripped", part.Text)
			}
			if !part.Thought {
				t.Errorf("part.Thought = false for %q, want true", part.Text)
			}
		}
		if !bytes.Equal(textParts[0].ThoughtSignature, []byte("sig1")) {
			t.Errorf("textParts[0].ThoughtSignature = %q, want %q", textParts[0].ThoughtSignature, "sig1")
		}
		if !bytes.Equal(textParts[1].ThoughtSignature, []byte("sig2")) {
			t.Errorf("textParts[1].ThoughtSignature = %q, want %q", textParts[1].ThoughtSignature, "sig2")
		}
		if diff := cmp.Diff([]string{"lookup"}, functionCallNames(result)); diff != "" {
			t.Errorf("functionCallNames mismatch (-want +got):\n%s", diff)
		}
		if firstInput.Text != "/*PLANNING*/Step 1: look it up." || firstInput.Thought {
			t.Errorf("caller input part was mutated in place: %+v", firstInput)
		}
	})

	// Corresponds to: test_strips_final_answer_tag_boundary
	t.Run("strips final answer tag boundary", func(t *testing.T) {
		t.Parallel()

		p := planner.NewPlanReActPlanner()
		responseParts := []*genai.Part{
			{Text: "/*REASONING*/Some reasoning./*FINAL_ANSWER*/The answer is 42."},
		}

		result := p.ProcessPlanningResponse(nil, responseParts)

		var texts []string
		for _, part := range result {
			if part.Text != "" {
				texts = append(texts, part.Text)
			}
		}
		combined := strings.Join(texts, " ")
		if strings.Contains(combined, "/*FINAL_ANSWER*/") {
			t.Errorf("combined = %q, want /*FINAL_ANSWER*/ stripped", combined)
		}
		if strings.Contains(combined, "/*REASONING*/") {
			t.Errorf("combined = %q, want /*REASONING*/ stripped", combined)
		}
		if !strings.Contains(combined, "The answer is 42.") {
			t.Errorf("combined = %q, want it to contain %q", combined, "The answer is 42.")
		}
		if len(result) != 2 {
			t.Fatalf("len(result) = %d, want 2", len(result))
		}
		if !result[0].Thought || result[0].Text != "Some reasoning." {
			t.Errorf("result[0] = {Text: %q, Thought: %v}, want {Text: %q, Thought: true}", result[0].Text, result[0].Thought, "Some reasoning.")
		}
		if result[1].Thought || result[1].Text != "The answer is 42." {
			t.Errorf("result[1] = {Text: %q, Thought: %v}, want {Text: %q, Thought: false}", result[1].Text, result[1].Thought, "The answer is 42.")
		}
	})

	// Corresponds to: test_strips_multiple_planning_tags
	t.Run("strips multiple planning tags", func(t *testing.T) {
		t.Parallel()

		p := planner.NewPlanReActPlanner()
		responseParts := []*genai.Part{
			{
				Text: "/*PLANNING*/Initial plan.\n" +
					"/*REASONING*/Some reasoning.\n" +
					"/*FINAL_ANSWER*/The answer is 42.",
			},
		}

		result := p.ProcessPlanningResponse(nil, responseParts)

		var texts []string
		for _, part := range result {
			if part.Text != "" {
				texts = append(texts, part.Text)
			}
		}
		combined := strings.Join(texts, " ")
		for _, tag := range []string{"/*PLANNING*/", "/*REASONING*/", "/*FINAL_ANSWER*/"} {
			if strings.Contains(combined, tag) {
				t.Errorf("combined = %q, want %s stripped", combined, tag)
			}
		}
		for _, want := range []string{"Initial plan.", "Some reasoning.", "The answer is 42."} {
			if !strings.Contains(combined, want) {
				t.Errorf("combined = %q, want it to contain %q", combined, want)
			}
		}
	})

	// Corresponds to: test_part_without_leading_tag_not_marked_as_thought
	t.Run("part without leading tag not marked as thought", func(t *testing.T) {
		t.Parallel()

		p := planner.NewPlanReActPlanner()
		responseParts := []*genai.Part{
			{Text: "Here is the answer /*PLANNING*/ with stray tag."},
		}

		result := p.ProcessPlanningResponse(nil, responseParts)

		if len(result) != 1 {
			t.Fatalf("len(result) = %d, want 1", len(result))
		}
		if result[0].Thought {
			t.Errorf("result[0].Thought = true, want false")
		}
		if got, want := result[0].Text, "Here is the answer /*PLANNING*/ with stray tag."; got != want {
			t.Errorf("result[0].Text = %q, want %q", got, want)
		}
	})

	// Corresponds to: test_bare_tag_part_is_marked_as_thought
	t.Run("bare tag part is marked as thought", func(t *testing.T) {
		t.Parallel()

		p := planner.NewPlanReActPlanner()
		responseParts := []*genai.Part{
			{Text: "/*ACTION*/"},
			genai.NewPartFromFunctionCall("lookup", map[string]any{"q": "test"}),
		}

		result := p.ProcessPlanningResponse(nil, responseParts)

		if len(result) != 2 {
			t.Fatalf("len(result) = %d, want 2", len(result))
		}
		if result[0].Text != "" {
			t.Errorf("result[0].Text = %q, want empty string", result[0].Text)
		}
		if !result[0].Thought {
			t.Errorf("result[0].Thought = false, want true")
		}
		if diff := cmp.Diff([]string{"lookup"}, functionCallNames(result)); diff != "" {
			t.Errorf("functionCallNames mismatch (-want +got):\n%s", diff)
		}
	})

	// Corresponds to: test_sole_bare_tag_part_is_marked_as_thought
	t.Run("sole bare tag part is marked as thought", func(t *testing.T) {
		t.Parallel()

		p := planner.NewPlanReActPlanner()
		responseParts := []*genai.Part{
			{Text: "/*ACTION*/"},
		}

		result := p.ProcessPlanningResponse(nil, responseParts)

		if len(result) != 1 {
			t.Fatalf("len(result) = %d, want 1", len(result))
		}
		if result[0].Text != "" {
			t.Errorf("result[0].Text = %q, want empty string", result[0].Text)
		}
		if !result[0].Thought {
			t.Errorf("result[0].Thought = false, want true")
		}
	})

	// Corresponds to: test_preserves_all_leading_parallel_function_calls
	t.Run("preserves all leading parallel function calls", func(t *testing.T) {
		t.Parallel()

		p := planner.NewPlanReActPlanner()
		responseParts := []*genai.Part{
			genai.NewPartFromFunctionCall("get_weather", map[string]any{"city": "SF"}),
			genai.NewPartFromFunctionCall("get_time", map[string]any{"city": "SF"}),
		}

		result := p.ProcessPlanningResponse(nil, responseParts)

		if diff := cmp.Diff([]string{"get_weather", "get_time"}, functionCallNames(result)); diff != "" {
			t.Errorf("functionCallNames mismatch (-want +got):\n%s", diff)
		}
	})

	// Corresponds to: test_preserves_parallel_function_calls_after_leading_text
	t.Run("preserves parallel function calls after leading text", func(t *testing.T) {
		t.Parallel()

		p := planner.NewPlanReActPlanner()
		responseParts := []*genai.Part{
			{Text: "Let me look that up."},
			genai.NewPartFromFunctionCall("get_weather", map[string]any{"city": "SF"}),
			genai.NewPartFromFunctionCall("get_time", map[string]any{"city": "SF"}),
		}

		result := p.ProcessPlanningResponse(nil, responseParts)

		if diff := cmp.Diff([]string{"get_weather", "get_time"}, functionCallNames(result)); diff != "" {
			t.Errorf("functionCallNames mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("filters empty function call names and stops after first function call group", func(t *testing.T) {
		t.Parallel()

		p := planner.NewPlanReActPlanner()
		responseParts := []*genai.Part{
			{FunctionCall: &genai.FunctionCall{Name: ""}},
			{Text: "/*REPLANNING*/Revised plan."},
			genai.NewPartFromFunctionCall("first_call", map[string]any{}),
			genai.NewPartFromFunctionCall("second_call", map[string]any{}),
			{Text: "Trailing text after function calls"},
			genai.NewPartFromFunctionCall("later_call", map[string]any{}),
		}

		result := p.ProcessPlanningResponse(nil, responseParts)

		if len(result) != 3 {
			t.Fatalf("len(result) = %d, want 3", len(result))
		}
		if !result[0].Thought || result[0].Text != "Revised plan." {
			t.Errorf("result[0] = {Text: %q, Thought: %v}, want {Text: %q, Thought: true}", result[0].Text, result[0].Thought, "Revised plan.")
		}
		if diff := cmp.Diff([]string{"first_call", "second_call"}, functionCallNames(result)); diff != "" {
			t.Errorf("functionCallNames mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("returns nil for empty response parts", func(t *testing.T) {
		t.Parallel()

		p := planner.NewPlanReActPlanner()
		if got := p.ProcessPlanningResponse(nil, nil); got != nil {
			t.Errorf("ProcessPlanningResponse(nil, nil) = %v, want nil", got)
		}
		if got := p.ProcessPlanningResponse(nil, []*genai.Part{}); got != nil {
			t.Errorf("ProcessPlanningResponse(nil, []) = %v, want nil", got)
		}
	})
}

func TestPlanReActPlanner_BuildPlanningInstruction(t *testing.T) {
	t.Parallel()

	p := planner.NewPlanReActPlanner()
	inst := p.BuildPlanningInstruction(nil, &model.LLMRequest{})
	for _, want := range []string{
		"/*PLANNING*/",
		"/*REPLANNING*/",
		"/*REASONING*/",
		"/*ACTION*/",
		"/*FINAL_ANSWER*/",
		"The revised plan should be under /*REPLANNING*/.",
	} {
		if !strings.Contains(inst, want) {
			t.Errorf("BuildPlanningInstruction() missing %q", want)
		}
	}
}

func TestBuiltInPlanner(t *testing.T) {
	t.Parallel()

	thinkingCfg := &genai.ThinkingConfig{IncludeThoughts: true}
	p := planner.NewBuiltInPlanner(thinkingCfg)

	if got := p.BuildPlanningInstruction(nil, &model.LLMRequest{}); got != "" {
		t.Errorf("BuildPlanningInstruction() = %q, want empty string", got)
	}
	if got := p.ProcessPlanningResponse(nil, []*genai.Part{{Text: "hello"}}); got != nil {
		t.Errorf("ProcessPlanningResponse() = %v, want nil", got)
	}
}
