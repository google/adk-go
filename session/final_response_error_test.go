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

package session_test

import (
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
)

// An event with an error code that is not partial and holds no function call
// is final; ErrorMessage alone does not make an error event (adk-python
// events/event.py:297-302). The clauses apply in order: compaction, then
// SkipSummarization or long-running tools, then the error clause.
func TestIsFinalResponseErrorClause(t *testing.T) {
	tests := []struct {
		name  string
		event *session.Event
		want  bool
	}{
		{
			name: "error code with a trailing code execution result",
			event: &session.Event{LLMResponse: model.LLMResponse{
				ErrorCode: "MAX_TOKENS",
				Content: &genai.Content{Parts: []*genai.Part{
					{Text: "x"},
					{CodeExecutionResult: &genai.CodeExecutionResult{}},
				}},
			}},
			want: true,
		},
		{
			name: "error code and message with a function response",
			event: &session.Event{LLMResponse: model.LLMResponse{
				ErrorCode:    "SOME_ERROR",
				ErrorMessage: "Something went wrong",
				Content: &genai.Content{Parts: []*genai.Part{
					{FunctionResponse: &genai.FunctionResponse{Name: "f"}},
				}},
			}},
			want: true,
		},
		{
			name: "error code with a function call and a function response",
			event: &session.Event{LLMResponse: model.LLMResponse{
				ErrorCode: "SOME_ERROR",
				Content: &genai.Content{Parts: []*genai.Part{
					{FunctionCall: &genai.FunctionCall{Name: "f"}},
					{FunctionResponse: &genai.FunctionResponse{Name: "f"}},
				}},
			}},
			want: false,
		},
		{
			name: "error message without an error code",
			event: &session.Event{LLMResponse: model.LLMResponse{
				ErrorMessage: "Something went wrong",
				Content: &genai.Content{Parts: []*genai.Part{
					{FunctionResponse: &genai.FunctionResponse{Name: "f"}},
				}},
			}},
			want: false,
		},
		{
			name: "error message without an error code on a text event",
			event: &session.Event{LLMResponse: model.LLMResponse{
				ErrorMessage: "Something went wrong",
				Content:      &genai.Content{Parts: []*genai.Part{{Text: "x"}}},
			}},
			want: true,
		},
		{
			name: "error code on a compaction event",
			event: &session.Event{
				LLMResponse: model.LLMResponse{
					ErrorCode: "SOME_ERROR",
					Content:   &genai.Content{Parts: []*genai.Part{{Text: "x"}}},
				},
				Actions: session.EventActions{Compaction: &session.EventCompaction{}},
			},
			want: false,
		},
		{
			name: "partial event with an error code",
			event: &session.Event{LLMResponse: model.LLMResponse{
				ErrorCode: "SOME_ERROR",
				Partial:   true,
				Content:   &genai.Content{Parts: []*genai.Part{{Text: "x"}}},
			}},
			want: false,
		},
		{
			name: "skip summarization overrides error code with function call",
			event: &session.Event{
				LLMResponse: model.LLMResponse{
					ErrorCode: "SOME_ERROR",
					Content: &genai.Content{Parts: []*genai.Part{
						{FunctionCall: &genai.FunctionCall{Name: "f"}},
					}},
				},
				Actions: session.EventActions{SkipSummarization: true},
			},
			want: true,
		},
		{
			name: "compaction overrides skip summarization",
			event: &session.Event{
				Actions: session.EventActions{
					Compaction:        &session.EventCompaction{},
					SkipSummarization: true,
				},
			},
			want: false,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := tc.event.IsFinalResponse(); got != tc.want {
				t.Errorf("IsFinalResponse() = %v, want %v", got, tc.want)
			}
		})
	}
}
