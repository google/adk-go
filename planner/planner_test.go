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
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/planner"
)

func TestPlanReActPlanner_ProcessPlanningResponse_FunctionCallGroupAndTrailingDrop(t *testing.T) {
	pLead := &genai.Part{Text: "/*PLANNING*/Plan step"}
	pEmptyBefore := &genai.Part{FunctionCall: &genai.FunctionCall{ID: "id-0", Name: ""}}
	pCall1 := &genai.Part{
		FunctionCall:     &genai.FunctionCall{ID: "id-1", Name: "get_weather", Args: map[string]any{"city": "SF"}},
		ThoughtSignature: []byte("sig-fc1"),
	}
	pCall2 := &genai.Part{FunctionCall: &genai.FunctionCall{ID: "id-2", Name: "get_time", Args: map[string]any{"city": "SF"}}}
	pTrailingText := &genai.Part{Text: "/*REASONING*/Drop me"}
	pLaterCall := &genai.Part{FunctionCall: &genai.FunctionCall{ID: "id-3", Name: "get_news"}}

	got, err := planner.NewPlanReActPlanner().ProcessPlanningResponse(nil, []*genai.Part{pLead, pEmptyBefore, pCall1, pCall2, pTrailingText, pLaterCall})
	if err != nil {
		t.Fatalf("ProcessPlanningResponse() error = %v, want nil", err)
	}

	want := []*genai.Part{
		{Text: "Plan step", Thought: true},
		{
			FunctionCall:     &genai.FunctionCall{ID: "id-1", Name: "get_weather", Args: map[string]any{"city": "SF"}},
			ThoughtSignature: []byte("sig-fc1"),
		},
		{FunctionCall: &genai.FunctionCall{ID: "id-2", Name: "get_time", Args: map[string]any{"city": "SF"}}},
	}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Fatalf("ProcessPlanningResponse() mismatch (-want +got):\n%s", diff)
	}
	if got[0] != pLead || got[1] != pCall1 || got[2] != pCall2 {
		t.Errorf("ProcessPlanningResponse() returned copies, want the original pointers [pLead, pCall1, pCall2]")
	}
}

func TestPlanReActPlanner_ProcessPlanningResponse_LeadingReplanningAndEmbeddedTagsInPlace(t *testing.T) {
	orig := &genai.Part{
		Text:             "/*REPLANNING*/Revise /*PLANNING*/old /*REASONING*/why /*ACTION*/run",
		ThoughtSignature: []byte("sig-replan"),
	}

	got, err := planner.NewPlanReActPlanner().ProcessPlanningResponse(nil, []*genai.Part{orig})
	if err != nil {
		t.Fatalf("ProcessPlanningResponse() error = %v, want nil", err)
	}

	want := []*genai.Part{{Text: "Revise old why run", Thought: true, ThoughtSignature: []byte("sig-replan")}}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Fatalf("ProcessPlanningResponse() mismatch (-want +got):\n%s", diff)
	}
	if got[0] != orig {
		t.Errorf("ProcessPlanningResponse() returned a copy, want the original part rewritten in place")
	}
}

func TestPlanReActPlanner_ProcessPlanningResponse_PreservesNonLeadingThoughtAndNonTextParts(t *testing.T) {
	pThought := &genai.Part{
		Text:             "Native model thought /*ACTION*/ kept",
		Thought:          true,
		ThoughtSignature: []byte("sig-native"),
	}
	pBlob := &genai.Part{InlineData: &genai.Blob{MIMEType: "image/png", Data: []byte{0x89, 0x50}}}

	got, err := planner.NewPlanReActPlanner().ProcessPlanningResponse(nil, []*genai.Part{pThought, pBlob})
	if err != nil {
		t.Fatalf("ProcessPlanningResponse() error = %v, want nil", err)
	}

	want := []*genai.Part{
		{Text: "Native model thought /*ACTION*/ kept", Thought: true, ThoughtSignature: []byte("sig-native")},
		{InlineData: &genai.Blob{MIMEType: "image/png", Data: []byte{0x89, 0x50}}},
	}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Fatalf("ProcessPlanningResponse() mismatch (-want +got):\n%s", diff)
	}
	if got[0] != pThought || got[1] != pBlob {
		t.Errorf("ProcessPlanningResponse() returned copies, want the original pointers [pThought, pBlob]")
	}

	t.Run("nil_part_kept_in_place", func(t *testing.T) {
		input := []*genai.Part{
			nil,
			{Text: "Native model thought", Thought: true, ThoughtSignature: []byte("sig-native")},
			{InlineData: &genai.Blob{MIMEType: "image/png", Data: []byte{0x89, 0x50}}},
		}

		got, err := planner.NewPlanReActPlanner().ProcessPlanningResponse(nil, input)
		if err != nil {
			t.Fatalf("ProcessPlanningResponse() error = %v, want nil", err)
		}

		want := []*genai.Part{
			nil,
			{Text: "Native model thought", Thought: true, ThoughtSignature: []byte("sig-native")},
			{InlineData: &genai.Blob{MIMEType: "image/png", Data: []byte{0x89, 0x50}}},
		}
		if diff := cmp.Diff(want, got); diff != "" {
			t.Errorf("ProcessPlanningResponse() mismatch (-want +got):\n%s", diff)
		}
	})
}

func TestPlanReActPlanner_ProcessPlanningResponse_FinalAnswerSplitBoundaries(t *testing.T) {
	tests := []struct {
		name  string
		input []*genai.Part
		want  []*genai.Part
	}{
		{
			name:  "multiple_final_answer_and_tags_in_answer",
			input: []*genai.Part{{Text: "/*PLANNING*/P1/*FINAL_ANSWER*/draft/*REASONING*/R2/*FINAL_ANSWER*/Final /*PLANNING*/ literal"}},
			want: []*genai.Part{
				{Text: "P1/*FINAL_ANSWER*/draftR2", Thought: true},
				{Text: "Final /*PLANNING*/ literal", Thought: false},
			},
		},
		{
			name:  "only_tags_before_final_answer",
			input: []*genai.Part{{Text: "/*PLANNING*//*REASONING*//*FINAL_ANSWER*/Only answer"}},
			want:  []*genai.Part{{Text: "Only answer", Thought: false}},
		},
		{
			name:  "empty_after_final_answer",
			input: []*genai.Part{{Text: "/*REASONING*/Only reasoning/*FINAL_ANSWER*/"}},
			want:  []*genai.Part{{Text: "Only reasoning", Thought: true}},
		},
		{
			// cmp.Diff tells a nil slice from an empty one, so this also checks
			// that the result is non-nil.
			name:  "sole_final_answer_tag",
			input: []*genai.Part{{Text: "/*FINAL_ANSWER*/"}},
			want:  []*genai.Part{},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := planner.NewPlanReActPlanner().ProcessPlanningResponse(nil, tc.input)
			if err != nil {
				t.Fatalf("ProcessPlanningResponse() error = %v, want nil", err)
			}
			if diff := cmp.Diff(tc.want, got); diff != "" {
				t.Errorf("ProcessPlanningResponse() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}
