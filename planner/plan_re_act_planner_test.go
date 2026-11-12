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
	"sync"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/planner"
)

// wantPlanReActInstruction is the text adk-python v2.12.0 builds in
// PlanReActPlanner._build_nl_planner_instruction
// (src/google/adk/planners/plan_re_act_planner.py:172-229): six preambles,
// each starting and ending with a newline, joined by "\n\n".
const wantPlanReActInstruction = `
When answering the question, try to leverage the available tools to gather the information instead of your memorized knowledge.

Follow this process when answering the question: (1) first come up with a plan in natural language text format; (2) Then use tools to execute the plan and provide reasoning between tool code snippets to make a summary of current state and next step. Tool code snippets and reasoning should be interleaved with each other. (3) In the end, return one final answer.

Follow this format when answering the question: (1) The planning part should be under /*PLANNING*/. (2) The tool code snippets should be under /*ACTION*/, and the reasoning parts should be under /*REASONING*/. (3) The final answer part should be under /*FINAL_ANSWER*/.



Below are the requirements for the planning:
The plan is made to answer the user query if following the plan. The plan is coherent and covers all aspects of information from user query, and only involves the tools that are accessible by the agent. The plan contains the decomposed steps as a numbered list where each step should use one or multiple available tools. By reading the plan, you can intuitively know which tools to trigger or what actions to take.
If the initial plan cannot be successfully executed, you should learn from previous execution results and revise your plan. The revised plan should be under /*REPLANNING*/. Then use tools to follow the new plan.



Below are the requirements for the reasoning:
The reasoning makes a summary of the current trajectory based on the user query and tool outputs. Based on the tool outputs and plan, the reasoning also comes up with instructions to the next steps, making the trajectory closer to the final answer.



Below are the requirements for the final answer:
The final answer should be precise and follow query formatting requirements. Some queries may not be answerable with the available tools and information. In those cases, inform the user why you cannot process their query and ask for more information.



Below are the requirements for the tool code:

**Custom Tools:** The available tools are described in the context and can be directly used.
- Code must be valid self-contained Python snippets with no imports and no references to tools or Python libraries that are not in the context.
- You cannot use any parameters or fields that are not explicitly defined in the APIs in the context.
- The code snippets should be readable, efficient, and directly relevant to the user query and reasoning steps.
- When using the tools, you should use the library name together with the function name, e.g., vertex_search.search().
- If Python libraries are not provided in the context, NEVER write your own code other than the function calls using the provided tools.



VERY IMPORTANT instruction that you MUST follow in addition to the above instructions:

You should ask for clarification if you need more information to answer the question.
You should prefer using the information available in the context instead of repeated tool use.
`

func TestPlanReActPlanner_DropsTrailingEmptyNameFunctionCall(t *testing.T) {
	pA := &genai.Part{FunctionCall: &genai.FunctionCall{Name: "a"}}
	pEmpty := &genai.Part{FunctionCall: &genai.FunctionCall{Name: ""}}
	pB := &genai.Part{FunctionCall: &genai.FunctionCall{Name: "b"}}

	got, err := planner.NewPlanReActPlanner().ProcessPlanningResponse(nil, []*genai.Part{pA, pEmpty, pB})
	if err != nil {
		t.Fatalf("ProcessPlanningResponse() error = %v, want nil", err)
	}

	want := []*genai.Part{
		{FunctionCall: &genai.FunctionCall{Name: "a"}},
		{FunctionCall: &genai.FunctionCall{Name: "b"}},
	}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Fatalf("ProcessPlanningResponse() mismatch (-want +got):\n%s", diff)
	}
	if got[0] != pA || got[1] != pB {
		t.Errorf("ProcessPlanningResponse() returned copies, want the original pointers [pA, pB]")
	}
}

func TestPlanReActPlanner_FinalAnswerPreservesThoughtSignature(t *testing.T) {
	tests := []struct {
		name  string
		input []*genai.Part
		want  []*genai.Part
	}{
		{
			name:  "reasoning_and_answer",
			input: []*genai.Part{{Text: "/*REASONING*/r/*FINAL_ANSWER*/a", ThoughtSignature: []byte("sig-f2")}},
			want: []*genai.Part{
				{Text: "r", Thought: true, ThoughtSignature: nil},
				{Text: "a", Thought: false, ThoughtSignature: []byte("sig-f2")},
			},
		},
		{
			name:  "thought_flag_not_copied_to_answer",
			input: []*genai.Part{{Text: "/*REASONING*/r/*FINAL_ANSWER*/a", Thought: true, ThoughtSignature: []byte("sig-f2")}},
			want: []*genai.Part{
				{Text: "r", Thought: true, ThoughtSignature: nil},
				{Text: "a", Thought: false, ThoughtSignature: []byte("sig-f2")},
			},
		},
		{
			name:  "reasoning_only",
			input: []*genai.Part{{Text: "/*REASONING*/r/*FINAL_ANSWER*/", ThoughtSignature: []byte("sig-f2")}},
			want:  []*genai.Part{{Text: "r", Thought: true, ThoughtSignature: []byte("sig-f2")}},
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

func TestPlanReActPlanner_BuildPlanningInstruction(t *testing.T) {
	got, err := planner.NewPlanReActPlanner().BuildPlanningInstruction(nil, nil)
	if err != nil {
		t.Fatalf("BuildPlanningInstruction() error = %v, want nil", err)
	}
	if diff := cmp.Diff(wantPlanReActInstruction, got); diff != "" {
		t.Errorf("BuildPlanningInstruction() mismatch (-want +got):\n%s", diff)
	}
}

func TestPlanReActPlanner_EmptyOrNilPartsReturnsNil(t *testing.T) {
	tests := []struct {
		name  string
		input []*genai.Part
	}{
		{name: "nil", input: nil},
		{name: "empty", input: []*genai.Part{}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := planner.NewPlanReActPlanner().ProcessPlanningResponse(nil, tc.input)
			if err != nil {
				t.Fatalf("ProcessPlanningResponse() error = %v, want nil", err)
			}
			if got != nil {
				t.Errorf("ProcessPlanningResponse() = %#v, want nil", got)
			}
		})
	}
}

// TestPlanReActPlanner_ConcurrentUseIsSafe shares one planner between
// goroutines, as an agent run concurrently does. It is meant for -race: a
// planner that kept or cached state between calls would be reported there.
func TestPlanReActPlanner_ConcurrentUseIsSafe(t *testing.T) {
	p := planner.NewPlanReActPlanner()
	want := []*genai.Part{
		{Text: "plan", Thought: true},
		{Text: "why", Thought: true},
		{Text: "answer", ThoughtSignature: []byte("sig")},
	}

	var wg sync.WaitGroup
	for range 8 {
		wg.Go(func() {
			inst, err := p.BuildPlanningInstruction(nil, nil)
			if err != nil || inst != wantPlanReActInstruction {
				t.Errorf("BuildPlanningInstruction() = (%d bytes, %v), want (%d bytes, nil)", len(inst), err, len(wantPlanReActInstruction))
			}
			parts := []*genai.Part{
				{Text: "/*PLANNING*/plan"},
				{Text: "/*REASONING*/why/*FINAL_ANSWER*/answer", ThoughtSignature: []byte("sig")},
			}
			got, err := p.ProcessPlanningResponse(nil, parts)
			if err != nil {
				t.Errorf("ProcessPlanningResponse() error = %v, want nil", err)
			}
			if diff := cmp.Diff(want, got); diff != "" {
				t.Errorf("ProcessPlanningResponse() mismatch (-want +got):\n%s", diff)
			}
		})
	}
	wg.Wait()
}
