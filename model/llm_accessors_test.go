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
	"slices"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"
)

// A response with nothing to report yields nothing, and a nil *LLMResponse is
// valid input.
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
			if got := slices.Collect(tc.resp.FunctionCalls()); len(got) != 0 {
				t.Errorf("FunctionCalls() yielded %v, want nothing", got)
			}
			if got := slices.Collect(tc.resp.FunctionResponses()); len(got) != 0 {
				t.Errorf("FunctionResponses() yielded %v, want nothing", got)
			}
		})
	}
}

// The accessors yield the pointers stored in the content, in part order,
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

	calls := slices.Collect(resp.FunctionCalls())
	if len(calls) != 2 || calls[0] != callA || calls[1] != callB {
		t.Errorf("FunctionCalls() = %v, want the stored pointers [a, unnamed]", calls)
	}
	responses := slices.Collect(resp.FunctionResponses())
	if len(responses) != 2 || responses[0] != respA || responses[1] != respB {
		t.Errorf("FunctionResponses() = %v, want the stored pointers [ra, unnamed]", responses)
	}
	if len(resp.Content.Parts) != 6 || resp.Content.Parts[1] != nil {
		t.Errorf("Content.Parts changed: len %d, Parts[1] nil = %v; want len 6 with Parts[1] nil", len(resp.Content.Parts), len(resp.Content.Parts) > 1 && resp.Content.Parts[1] == nil)
	}
}

// Breaking out of a range over FunctionCalls or FunctionResponses stops the
// iterator. An iterator that kept yielding would make the range loop panic.
func TestLLMResponseAccessorsStopWhenTheLoopBreaks(t *testing.T) {
	callA := &genai.FunctionCall{Name: "a"}
	respA := &genai.FunctionResponse{Name: "ra"}
	resp := &model.LLMResponse{Content: &genai.Content{Parts: []*genai.Part{
		{FunctionCall: callA},
		{FunctionCall: &genai.FunctionCall{Name: "b"}},
		{FunctionResponse: respA},
		{FunctionResponse: &genai.FunctionResponse{Name: "rb"}},
	}}}

	var firstCall *genai.FunctionCall
	for fc := range resp.FunctionCalls() {
		firstCall = fc
		break
	}
	var firstResponse *genai.FunctionResponse
	for fr := range resp.FunctionResponses() {
		firstResponse = fr
		break
	}

	if firstCall != callA {
		t.Errorf("first FunctionCalls() value = %v, want the stored call a", firstCall)
	}
	if firstResponse != respA {
		t.Errorf("first FunctionResponses() value = %v, want the stored response ra", firstResponse)
	}
}
