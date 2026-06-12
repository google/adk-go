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

package session

import (
	"testing"

  	"google.golang.org/adk/model"
  	"google.golang.org/genai"
)

func TestHasFunctionCalls(t *testing.T) {
	tests := []struct {
		name string
		resp *model.LLMResponse
		want bool
	}{
		{
			name: "NilResponse",
			resp: nil,
			want: false,
		},
		{
			name: "NilContent",
			resp: &model.LLMResponse{Content: nil},
			want: false,
		},
		{
			name: "HasFunctionCall",
			resp: &model.LLMResponse{Content: &genai.Content{Parts: []*genai.Part{{
				FunctionCall: &genai.FunctionCall{Name: "test", Args: map[string]any{}},
			}}}},
			want: true,
		},
		{
			name: "HasToolCall",
			resp: &model.LLMResponse{Content: &genai.Content{Parts: []*genai.Part{{
				ToolCall: &genai.ToolCall{ID: "123", ToolType: "function_call", Args: map[string]any{"name": "test", "args": map[string]any{}}},
			}}}},
			want: true,
		},
		{
			name: "Neither",
			resp: &model.LLMResponse{Content: &genai.Content{Parts: []*genai.Part{{
				Text: "hello world",
			}}}},
			want: false,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := hasFunctionCalls(tc.resp); got != tc.want {
				t.Errorf("hasFunctionCalls() = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestHasFunctionResponses(t *testing.T) {
	tests := []struct {
		name string
		resp *model.LLMResponse
		want bool
	}{
		{
			name: "NilResponse",
			resp: nil,
			want: false,
		},
		{
			name: "NilContent",
			resp: &model.LLMResponse{Content: nil},
			want: false,
		},
		{
			name: "HasFunctionResponse",
			resp: &model.LLMResponse{Content: &genai.Content{Parts: []*genai.Part{{
				FunctionResponse: &genai.FunctionResponse{Name: "test", Response: map[string]any{}},
			}}}},
			want: true,
		},
		{
			name: "HasToolResponse",
			resp: &model.LLMResponse{Content: &genai.Content{Parts: []*genai.Part{{
				ToolResponse: &genai.ToolResponse{ID: "123", ToolType: "function_call", Response: map[string]any{"name": "test", "response": map[string]any{}}},
			}}}},
			want: true,
		},
		{
			name: "Neither",
			resp: &model.LLMResponse{Content: &genai.Content{Parts: []*genai.Part{{
				Text: "hello world",
			}}}},
			want: false,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := hasFunctionResponses(tc.resp); got != tc.want {
				t.Errorf("hasFunctionResponses() = %v, want %v", got, tc.want)
			}
		})
	}
}
