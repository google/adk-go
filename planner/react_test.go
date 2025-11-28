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
	"strings"
	"testing"

	"google.golang.org/genai"
)

func TestPlanReActPlanner_New(t *testing.T) {
	planner := NewReActPlanner()
	if planner == nil {
		t.Fatal("NewReActPlanner returned nil")
	}
}

func TestPlanReActPlanner_BuildPlanningInstruction(t *testing.T) {
	planner := NewReActPlanner()

	instruction := planner.BuildPlanningInstruction(nil, nil)

	if instruction == "" {
		t.Error("BuildPlanningInstruction should return non-empty instruction")
	}

	expectedTags := []string{
		PlanningTag,
		ReasoningTag,
		ActionTag,
		FinalAnswerTag,
	}

	for _, tag := range expectedTags {
		if !strings.Contains(instruction, tag) {
			t.Errorf("Instruction should contain tag %q", tag)
		}
	}

	expectedContent := []string{
		"planning part should be under",
		"reasoning parts should be under",
		"final answer part should be under",
		"tool code snippets",
		"available tools",
	}

	instructionLower := strings.ToLower(instruction)
	for _, content := range expectedContent {
		if !strings.Contains(instructionLower, content) {
			t.Errorf("Instruction should contain content about %q", content)
		}
	}
}

func TestPlanReActPlanner_ProcessPlanningResponse_EmptyParts(t *testing.T) {
	planner := NewReActPlanner()

	result := planner.ProcessPlanningResponse(nil, nil)
	if result != nil {
		t.Errorf("Expected nil result for nil parts, got %+v", result)
	}

	result = planner.ProcessPlanningResponse(nil, []*genai.Part{})
	if result != nil {
		t.Errorf("Expected nil result for empty parts, got %+v", result)
	}
}

func TestPlanReActPlanner_ProcessPlanningResponse_TextParts(t *testing.T) {
	planner := NewReActPlanner()

	tests := []struct {
		name          string
		inputParts    []*genai.Part
		expectThought bool
		description   string
	}{
		{
			name: "regular text",
			inputParts: []*genai.Part{
				{Text: "This is regular text"},
			},
			expectThought: false,
			description:   "Regular text should not be marked as thought",
		},
		{
			name: "planning tag",
			inputParts: []*genai.Part{
				{Text: PlanningTag + " This is planning content"},
			},
			expectThought: true,
			description:   "Planning tag content should be marked as thought",
		},
		{
			name: "reasoning tag",
			inputParts: []*genai.Part{
				{Text: ReasoningTag + " This is reasoning content"},
			},
			expectThought: true,
			description:   "Reasoning tag content should be marked as thought",
		},
		{
			name: "action tag",
			inputParts: []*genai.Part{
				{Text: ActionTag + " This is action content"},
			},
			expectThought: true,
			description:   "Action tag content should be marked as thought",
		},
		{
			name: "replanning tag",
			inputParts: []*genai.Part{
				{Text: ReplanningTag + " This is replanning content"},
			},
			expectThought: true,
			description:   "Replanning tag content should be marked as thought",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			result := planner.ProcessPlanningResponse(nil, test.inputParts)

			if result == nil {
				t.Fatal("Expected non-nil result")
			}

			if len(result) != 1 {
				t.Fatalf("Expected 1 result part, got %d", len(result))
			}

			if result[0].Thought != test.expectThought {
				t.Errorf("%s: expected Thought=%v, got %v",
					test.description, test.expectThought, result[0].Thought)
			}
		})
	}
}

func TestPlanReActPlanner_ProcessPlanningResponse_FinalAnswerSplit(t *testing.T) {
	planner := NewReActPlanner()

	tests := []struct {
		name          string
		inputText     string
		expectedParts int
		expectThought []bool
		expectedTexts []string
	}{
		{
			name:          "final answer split",
			inputText:     "This is reasoning " + FinalAnswerTag + " This is the final answer",
			expectedParts: 2,
			expectThought: []bool{true, false},
			expectedTexts: []string{
				"This is reasoning " + FinalAnswerTag,
				" This is the final answer",
			},
		},
		{
			name:          "multiple final answer tags - last one wins",
			inputText:     "Start " + FinalAnswerTag + " middle " + FinalAnswerTag + " final",
			expectedParts: 2,
			expectThought: []bool{true, false},
			expectedTexts: []string{
				"Start " + FinalAnswerTag + " middle " + FinalAnswerTag,
				" final",
			},
		},
		{
			name:          "only final answer tag",
			inputText:     FinalAnswerTag + " Just the answer",
			expectedParts: 2,
			expectThought: []bool{true, false},
			expectedTexts: []string{
				FinalAnswerTag,
				" Just the answer",
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			inputParts := []*genai.Part{{Text: test.inputText}}
			result := planner.ProcessPlanningResponse(nil, inputParts)

			if result == nil {
				t.Fatal("Expected non-nil result")
			}

			if len(result) != test.expectedParts {
				t.Fatalf("Expected %d result parts, got %d", test.expectedParts, len(result))
			}

			for i, part := range result {
				if part.Thought != test.expectThought[i] {
					t.Errorf("Part %d: expected Thought=%v, got %v",
						i, test.expectThought[i], part.Thought)
				}

				if part.Text != test.expectedTexts[i] {
					t.Errorf("Part %d: expected text %q, got %q",
						i, test.expectedTexts[i], part.Text)
				}
			}
		})
	}
}

func TestPlanReActPlanner_ProcessPlanningResponse_FunctionCalls(t *testing.T) {
	planner := NewReActPlanner()

	tests := []struct {
		name          string
		inputParts    []*genai.Part
		expectedParts int
	}{
		{
			name: "function call preserved",
			inputParts: []*genai.Part{
				{Text: "Some text"},
				{FunctionCall: &genai.FunctionCall{Name: "test_function"}},
				{Text: "More text after function"},
			},
			expectedParts: 2, // Text + function call (text after function call should be dropped)
		},
		{
			name: "empty function call filtered",
			inputParts: []*genai.Part{
				{Text: "Some text"},
				{FunctionCall: &genai.FunctionCall{Name: ""}},
				{FunctionCall: &genai.FunctionCall{Name: "valid_function"}},
			},
			expectedParts: 2, // Text + valid function call
		},
		{
			name: "multiple function calls",
			inputParts: []*genai.Part{
				{Text: "Some text"},
				{FunctionCall: &genai.FunctionCall{Name: "function1"}},
				{FunctionCall: &genai.FunctionCall{Name: "function2"}},
				{Text: "Text after functions"},
			},
			expectedParts: 3, // Text + function1 + function2
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			result := planner.ProcessPlanningResponse(nil, test.inputParts)

			if result == nil {
				t.Fatal("Expected non-nil result")
			}

			if len(result) != test.expectedParts {
				t.Fatalf("mismatched parts, expected %d result parts, got %d",
					test.expectedParts, len(result))
			}

			foundFunctionCall := false
			for _, part := range result {
				if part.FunctionCall != nil && part.FunctionCall.Name != "" {
					foundFunctionCall = true
				}
			}

			if !foundFunctionCall && test.expectedParts > 1 {
				t.Error("Expected to find at least one valid function call in result")
			}
		})
	}
}

func TestPlanReActPlanner_SplitByLastPattern(t *testing.T) {
	planner := NewReActPlanner()

	tests := []struct {
		name      string
		text      string
		separator string
		expect    []string
	}{
		{"simple split", "hello|world", "|", []string{"hello|", "world"}},
		{"no separator", "hello world", "|", []string{"hello world", ""}},
		{"multiple separators", "a|b|c|d", "|", []string{"a|b|c|", "d"}},
		{"separator at end", "hello|", "|", []string{"hello|", ""}},
		{"separator at start", "|hello", "|", []string{"|", "hello"}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got1, got2 := planner.splitByLastPattern(tt.text, tt.separator)

			if got1 != tt.expect[0] || got2 != tt.expect[1] {
				t.Errorf("splitByLastPattern(%q, %q) = (%q, %q), want (%q, %q)",
					tt.text, tt.separator, got1, got2, tt.expect[0], tt.expect[1])
			}
		})
	}
}

func TestPlanReActPlanner_MarkAsThought(t *testing.T) {
	planner := NewReActPlanner()

	tests := []struct {
		name string
		part *genai.Part
	}{
		{
			name: "text part",
			part: &genai.Part{Text: "Some text"},
		},
		{
			name: "empty text part",
			part: &genai.Part{Text: ""},
		},
		{
			name: "function call part",
			part: &genai.Part{FunctionCall: &genai.FunctionCall{Name: "test"}},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			expected := test.part.Text != ""
			planner.markAsThought(test.part)

			if test.part.Thought != expected {
				t.Errorf("Expected Thought=%v, got %v", expected, test.part.Thought)
			}
		})
	}
}
