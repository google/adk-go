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

package agent

import (
	"testing"
	"time"

	"github.com/google/adk-go"
	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"
)

func TestGetContents(t *testing.T) {
	t.Parallel()
	testCases := []struct {
		name      string
		agentName string
		branch    string
		events    []*adk.Event
		want      []*genai.Content
	}{
		{
			name:      "NilEvent",
			agentName: "testAgent",
			events:    nil,
			want:      nil,
		},
		{
			name:      "EmptyEvents",
			agentName: "testAgent",
			events:    []*adk.Event{},
			want:      nil,
		},
		{
			name:      "UserAndAgentEvents",
			agentName: "testAgent",
			events: []*adk.Event{
				{
					Author: "user",
					LLMResponse: &adk.LLMResponse{
						Content: genai.NewContentFromText("Hello", "user"),
					},
				},
				{
					Author: "testAgent",
					LLMResponse: &adk.LLMResponse{
						Content: genai.NewContentFromText("Hi there", "model"),
					},
				},
			},
			want: []*genai.Content{
				genai.NewContentFromText("Hello", "user"),
				genai.NewContentFromText("Hi there", "model"),
			},
		},
		{
			name:      "anotherAgentEvent",
			agentName: "testAgent",
			events: []*adk.Event{
				{
					Author: "anotherAgent",
					LLMResponse: &adk.LLMResponse{
						Content: genai.NewContentFromText("Foreign message", "model"),
					},
				},
			},
			want: []*genai.Content{
				{
					Role: "user",
					Parts: []*genai.Part{
						{Text: "For context:"},
						{Text: "[anotherAgent] said: Foreign message"},
					},
				},
			},
		},
		{
			name:      "FilterByBranch",
			agentName: "testAgent",
			branch:    "branch1",
			events: []*adk.Event{
				{
					Author: "user",
					Branch: "branch1",
					LLMResponse: &adk.LLMResponse{
						Content: genai.NewContentFromText("In branch 1", "user"),
					},
				},
				{
					Author: "user",
					Branch: "branch1.task1",
					LLMResponse: &adk.LLMResponse{
						Content: genai.NewContentFromText("In branch 1 and task 1", "user"),
					},
				},
				{
					Author: "user",
					Branch: "branch12",
					LLMResponse: &adk.LLMResponse{
						Content: genai.NewContentFromText("In branch 12", "user"),
					},
				},
				{
					Author: "user",
					Branch: "branch2",
					LLMResponse: &adk.LLMResponse{
						Content: genai.NewContentFromText("In branch 2", "user"),
					},
				},
			},
			want: []*genai.Content{
				genai.NewContentFromText("In branch 1", "user"),
				genai.NewContentFromText("In branch 1 and task 1", "user"),
			},
		},
		{
			name:      "AuthEvent",
			agentName: "testAgent",
			events: []*adk.Event{
				{
					Author: "testAgent",
					LLMResponse: &adk.LLMResponse{
						Content: &genai.Content{
							Role: "model",
							Parts: []*genai.Part{
								{FunctionCall: &genai.FunctionCall{Name: "adk_request_credential"}},
							},
						},
					},
				},
			},
			want: nil,
		},
		{
			name:      "EventWithoutContent",
			agentName: "testAgent",
			events: []*adk.Event{
				{Author: "user"},
			},
			want: nil,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			got := getContents(tc.agentName, tc.branch, tc.events)
			if diff := cmp.Diff(tc.want, got, cmp.AllowUnexported(genai.FunctionCall{}, genai.FunctionResponse{})); diff != "" {
				t.Errorf("getContents() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestConvertForeignEvent(t *testing.T) {
	t.Parallel()
	now := time.Now()
	testCases := []struct {
		name  string
		event *adk.Event
		want  *adk.Event
	}{
		{
			name: "Text",
			event: &adk.Event{
				Time:   now,
				Author: "foreign",
				LLMResponse: &adk.LLMResponse{
					Content: genai.NewContentFromText("hello", "model"),
				},
				Branch: "b",
			},
			want: &adk.Event{
				Time:   now,
				Author: "user",
				LLMResponse: &adk.LLMResponse{
					Content: &genai.Content{
						Role: "user",
						Parts: []*genai.Part{
							{Text: "For context:"},
							{Text: "[foreign] said: hello"},
						},
					},
				},
				Branch: "b",
			},
		},
		{
			name: "FunctionCall",
			event: &adk.Event{
				Time:   now,
				Author: "foreign",
				LLMResponse: &adk.LLMResponse{
					Content: &genai.Content{
						Role: "model",
						Parts: []*genai.Part{
							{FunctionCall: &genai.FunctionCall{Name: "test", Args: map[string]any{"a": "b"}}},
						},
					},
				},
				Branch: "b",
			},
			want: &adk.Event{
				Time:   now,
				Author: "user",
				LLMResponse: &adk.LLMResponse{
					Content: &genai.Content{
						Role: "user",
						Parts: []*genai.Part{
							{Text: "For context:"},
							{Text: `[foreign] called tool "test" with parameters: map[a:b]`},
						},
					},
				},
				Branch: "b",
			},
		},
		{
			name: "FunctionResponse",
			event: &adk.Event{
				Time:   now,
				Author: "foreign",
				LLMResponse: &adk.LLMResponse{
					Content: &genai.Content{
						Role: "model",
						Parts: []*genai.Part{
							{FunctionResponse: &genai.FunctionResponse{Name: "test", Response: map[string]any{"c": "d"}}},
						},
					},
				},
				Branch: "b",
			},
			want: &adk.Event{
				Time:   now,
				Author: "user",
				LLMResponse: &adk.LLMResponse{
					Content: &genai.Content{
						Role: "user",
						Parts: []*genai.Part{
							{Text: "For context:"},
							{Text: `[foreign] "test" tool returned result: map[c:d]`},
						},
					},
				},
				Branch: "b",
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			got := convertForeignEvent(tc.event)
			if diff := cmp.Diff(tc.want, got, cmp.AllowUnexported(genai.FunctionCall{}, genai.FunctionResponse{})); diff != "" {
				t.Errorf("convertForeignEvent() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}
