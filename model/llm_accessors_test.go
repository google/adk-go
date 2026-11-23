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

package model_test

import (
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"
)

// A response with nothing to report gives nil slices, not empty ones, and a
// nil *LLMResponse is valid input.
func TestLLMResponseAccessorsNoResult(t *testing.T) {
	tests := []struct {
		name string
		resp *model.LLMResponse
	}{
		{"nil receiver", nil},
		{"zero value", &model.LLMResponse{}},
		{"empty non-nil part list", &model.LLMResponse{Content: &genai.Content{Parts: []*genai.Part{}}}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := tc.resp.FunctionCalls(); got != nil {
				t.Errorf("FunctionCalls() = %#v, want nil", got)
			}
			if got := tc.resp.FunctionResponses(); got != nil {
				t.Errorf("FunctionResponses() = %#v, want nil", got)
			}
			if got := tc.resp.HasTrailingCodeExecutionResult(); got {
				t.Errorf("HasTrailingCodeExecutionResult() = true, want false")
			}
		})
	}
}

// The accessors return the pointers stored in the content, in part order,
// including unnamed ones, skip text and nil parts, and leave the content
// unchanged.
func TestLLMResponseAccessorsReturnStoredValues(t *testing.T) {
	callA := &genai.FunctionCall{Name: "a"}
	callB := &genai.FunctionCall{}
	respA := &genai.FunctionResponse{Name: "ra"}
	respB := &genai.FunctionResponse{}
	resp := &model.LLMResponse{Content: &genai.Content{Parts: []*genai.Part{
		{Text: "x"},
		nil,
		{FunctionCall: callA},
		{FunctionResponse: respA},
		{FunctionCall: callB},
		{FunctionResponse: respB},
	}}}

	calls := resp.FunctionCalls()
	if len(calls) != 2 || calls[0] != callA || calls[1] != callB {
		t.Errorf("FunctionCalls() = %v, want the stored pointers [a, unnamed]", calls)
	}
	responses := resp.FunctionResponses()
	if len(responses) != 2 || responses[0] != respA || responses[1] != respB {
		t.Errorf("FunctionResponses() = %v, want the stored pointers [ra, unnamed]", responses)
	}
	if resp.HasTrailingCodeExecutionResult() {
		t.Errorf("HasTrailingCodeExecutionResult() = true, want false")
	}
	if len(resp.Content.Parts) != 6 || resp.Content.Parts[1] != nil {
		t.Errorf("Content.Parts changed: len %d, Parts[1] nil = %v; want len 6 with Parts[1] nil", len(resp.Content.Parts), len(resp.Content.Parts) > 1 && resp.Content.Parts[1] == nil)
	}
}

// Only the last non-nil part decides HasTrailingCodeExecutionResult; a code
// execution result earlier in the content does not count.
func TestLLMResponseAccessorsTrailingCodeExecutionResult(t *testing.T) {
	tests := []struct {
		name  string
		parts []*genai.Part
		want  bool
	}{
		{"code result before text and nil", []*genai.Part{{CodeExecutionResult: &genai.CodeExecutionResult{}}, {Text: "done"}, nil}, false},
		{"code result after text, before nil", []*genai.Part{{Text: "running"}, {CodeExecutionResult: &genai.CodeExecutionResult{}}, nil}, true},
		{"executable code after a code result", []*genai.Part{{CodeExecutionResult: &genai.CodeExecutionResult{}}, {ExecutableCode: &genai.ExecutableCode{Code: "print(1)"}}}, false},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			resp := &model.LLMResponse{Content: &genai.Content{Parts: tc.parts}}
			if got := resp.HasTrailingCodeExecutionResult(); got != tc.want {
				t.Errorf("HasTrailingCodeExecutionResult() = %v, want %v", got, tc.want)
			}
		})
	}
}
