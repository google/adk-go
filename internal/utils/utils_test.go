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

package utils

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"
)

func TestFunctionCalls(t *testing.T) {
	tests := []struct {
		name    string
		content *genai.Content
		want    []*genai.FunctionCall
	}{
		{
			name:    "NilContent",
			content: nil,
			want:    nil,
		},
		{
			name:    "NoFunctionCalls",
			content: &genai.Content{Parts: []*genai.Part{{Text: "hello"}}},
			want:    nil,
		},
		{
			name: "TraditionalFunctionCall",
			content: &genai.Content{Parts: []*genai.Part{{
				FunctionCall: &genai.FunctionCall{ID: "1", Name: "my_func", Args: map[string]any{"arg1": "val1"}},
			}}},
			want: []*genai.FunctionCall{{ID: "1", Name: "my_func", Args: map[string]any{"arg1": "val1"}}},
		},
		{
			name: "ToolCallFunctionCall",
			content: &genai.Content{Parts: []*genai.Part{{
				ToolCall: &genai.ToolCall{
					ID:       "2",
					ToolType: genai.ToolType("function_call"),
					Args: map[string]any{
						"name": "my_func2",
						"args": map[string]any{"arg2": "val2"},
					},
				},
			}}},
			want: []*genai.FunctionCall{{ID: "2", Name: "my_func2", Args: map[string]any{"arg2": "val2"}}},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := FunctionCalls(tc.content)
			if diff := cmp.Diff(tc.want, got); diff != "" {
				t.Errorf("FunctionCalls() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestFunctionResponses(t *testing.T) {
	tests := []struct {
		name    string
		content *genai.Content
		want    []*genai.FunctionResponse
	}{
		{
			name:    "NilContent",
			content: nil,
			want:    nil,
		},
		{
			name: "TraditionalFunctionResponse",
			content: &genai.Content{Parts: []*genai.Part{{
				FunctionResponse: &genai.FunctionResponse{ID: "1", Name: "my_func", Response: map[string]any{"result": "success"}},
			}}},
			want: []*genai.FunctionResponse{{ID: "1", Name: "my_func", Response: map[string]any{"result": "success"}}},
		},
		{
			name: "ToolResponseFunctionResponse",
			content: &genai.Content{Parts: []*genai.Part{{
				ToolResponse: &genai.ToolResponse{
					ID:       "2",
					ToolType: genai.ToolType("function_call"),
					Response: map[string]any{
						"name":     "my_func2",
						"response": map[string]any{"result": "success2"},
					},
				},
			}}},
			want: []*genai.FunctionResponse{{ID: "2", Name: "my_func2", Response: map[string]any{"result": "success2"}}},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := FunctionResponses(tc.content)
			if diff := cmp.Diff(tc.want, got); diff != "" {
				t.Errorf("FunctionResponses() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}
