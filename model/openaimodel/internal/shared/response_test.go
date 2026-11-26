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
	"fmt"
	"math"
	"testing"

	"google.golang.org/genai"
)

func TestTokenCount(t *testing.T) {
	tests := []struct {
		in     int64
		want   int32
		wantOK bool
	}{
		{in: 0, want: 0, wantOK: true},
		{in: 1_000_000, want: 1_000_000, wantOK: true},
		{in: math.MaxInt32, want: math.MaxInt32, wantOK: true},
		// Out of range either way: no request can have such a count, and a
		// blind int32 conversion would turn these into plausible ones.
		{in: -1},
		{in: math.MinInt32},
		{in: math.MinInt32 - 1}, // int32() wraps to MaxInt32
		{in: -(1 << 32) + 1},    // int32() wraps to 1
		{in: math.MaxInt32 + 1}, // int32() wraps to MinInt32
		{in: math.MaxInt64},
	}
	for _, tc := range tests {
		t.Run(fmt.Sprint(tc.in), func(t *testing.T) {
			got, ok := TokenCount(tc.in)
			if got != tc.want || ok != tc.wantOK {
				t.Errorf("TokenCount(%d) = (%d, %t), want (%d, %t)", tc.in, got, ok, tc.want, tc.wantOK)
			}
		})
	}
}

func TestInvalidUsage(t *testing.T) {
	tests := []struct {
		name       string
		counts     []UsageCount
		wantName   string
		wantReason string
	}{
		{name: "none"},
		{name: "all valid", counts: []UsageCount{{"a", 0}, {"b", math.MaxInt32}}},
		{name: "negative", counts: []UsageCount{{"a", 1}, {"b", -1}}, wantName: "b", wantReason: "negative"},
		{name: "too large", counts: []UsageCount{{"a", math.MaxInt32 + 1}}, wantName: "a", wantReason: "exceeds_int32"},
		{name: "first wins", counts: []UsageCount{{"a", -5}, {"b", math.MaxInt64}}, wantName: "a", wantReason: "negative"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			name, reason := InvalidUsage(tc.counts...)
			if name != tc.wantName || reason != tc.wantReason {
				t.Errorf("InvalidUsage() = (%q, %q), want (%q, %q)", name, reason, tc.wantName, tc.wantReason)
			}
		})
	}
}

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
