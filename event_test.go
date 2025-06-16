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

package adk

import (
	"testing"

	"google.golang.org/genai"
)

func TestIsFinalResponse(t *testing.T) {
	testCases := []struct {
		name  string
		event *Event
		want  bool
	}{
		{
			name:  "actions is nil",
			event: &Event{},
			want:  true,
		},
		{
			name: "skip summarization",
			event: &Event{
				Actions: &EventActions{
					SkipSummarization: true,
				},
			},
			want: true,
		},
		{
			name: "long running tool",
			event: &Event{
				Actions:            &EventActions{},
				LongRunningToolIDs: []string{"tool-123"},
			},
			want: true,
		},
		{
			name: "llm response with function call",
			event: &Event{
				Actions: &EventActions{},
				LLMResponse: &LLMResponse{
					Content: genai.NewContentFromFunctionCall("foo", nil, "model"),
				},
			},
			want: false,
		},
		{
			name: "llm response with function response",
			event: &Event{
				Actions: &EventActions{},
				LLMResponse: &LLMResponse{
					Content: genai.NewContentFromFunctionResponse("foo", nil, "model"),
				},
			},
			want: false,
		},
		{
			name: "llm response with partial response",
			event: &Event{
				Actions: &EventActions{},
				LLMResponse: &LLMResponse{
					Partial: true,
				},
			},
			want: false,
		},
		{
			name: "llm response with trailing code execution result",
			event: &Event{
				Actions: &EventActions{},
				LLMResponse: &LLMResponse{
					Content: genai.NewContentFromCodeExecutionResult(genai.OutcomeOK, "", "model"),
				},
			},
			want: false,
		},
		{
			name: "final response",
			event: &Event{
				Actions: &EventActions{},
				LLMResponse: &LLMResponse{
					Content: genai.NewContentFromText("this is the final response", "model"),
				},
			},
			want: true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			if got := tc.event.IsFinalResponse(); got != tc.want {
				t.Errorf("IsFinalResponse() = %v, want %v", got, tc.want)
			}
		})
	}
}
