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

package shared

import (
	"testing"

	"google.golang.org/genai"
)

func TestCompletedContentSupersedes(t *testing.T) {
	call := func(id, city string) *genai.Part {
		return &genai.Part{FunctionCall: &genai.FunctionCall{
			ID: id, Name: "get_weather", Args: map[string]any{"city": city},
		}}
	}
	tests := []struct {
		name      string
		aggregate []*genai.Part
		completed []*genai.Part
		want      bool
	}{
		{
			name:      "reasoning containing the answer is not visible output",
			aggregate: []*genai.Part{{Text: "hello"}},
			completed: []*genai.Part{{Text: "hello is the answer", Thought: true}},
		},
		{
			name:      "reasoning alone does not justify replacement",
			aggregate: []*genai.Part{{Text: "Checking", Thought: true}},
			completed: []*genai.Part{{Text: "Checked", Thought: true}},
		},
		{
			name:      "completed text can precede and follow streamed refusal",
			aggregate: []*genai.Part{{Text: "I cannot help."}},
			completed: []*genai.Part{{Text: "Here is "}, {Text: "I cannot help."}, {Text: " Sorry."}},
			want:      true,
		},
		{
			name:      "shorter text does not replace the answer",
			aggregate: []*genai.Part{{Text: "hello"}},
			completed: []*genai.Part{{Text: "hel"}},
		},
		{
			name:      "same name with different arguments is a different call",
			aggregate: []*genai.Part{call("call_1", "SF")},
			completed: []*genai.Part{call("call_1", "NY")},
		},
		{
			name:      "different call IDs remain distinct",
			aggregate: []*genai.Part{call("call_1", "SF")},
			completed: []*genai.Part{call("call_2", "SF")},
		},
		{
			name:      "one completed call cannot replace two streamed calls",
			aggregate: []*genai.Part{call("call_1", "SF"), call("call_1", "SF")},
			completed: []*genai.Part{call("call_1", "SF")},
		},
		{
			name:      "matching calls allow recovery of completed text",
			aggregate: []*genai.Part{call("call_1", "SF")},
			completed: []*genai.Part{call("call_1", "SF"), {Text: "Checking the weather."}},
			want:      true,
		},
		{
			name:      "reordered calls can match one to one",
			aggregate: []*genai.Part{call("call_1", "SF"), call("call_2", "NY")},
			completed: []*genai.Part{call("call_2", "NY"), call("call_1", "SF")},
			want:      true,
		},
		{
			name:      "completed function call replaces reasoning alone",
			aggregate: []*genai.Part{{Text: "Checking", Thought: true}},
			completed: []*genai.Part{call("call_1", "SF")},
			want:      true,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			aggregate := &genai.Content{Role: genai.RoleModel, Parts: tc.aggregate}
			completed := &genai.Content{Role: genai.RoleModel, Parts: tc.completed}
			if got := CompletedContentSupersedes(aggregate, completed); got != tc.want {
				t.Errorf("CompletedContentSupersedes() = %v, want %v", got, tc.want)
			}
		})
	}
}
