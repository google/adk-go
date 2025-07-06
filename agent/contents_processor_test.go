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

type model struct {
	adk.Model
}

// Test behavior around Agent's IncludeContents.
func TestContentsRequestProcessor_IncludeContents(t *testing.T) {
	const agentName = "testAgent"
	model := &model{}

	emptyEvent := []*adk.Event{}
	helloAndGoodBye := []*adk.Event{
		{
			Author: "user", // Not in the current turn in multi-agent scenario. See buildContentsCurrentTurnContextOnly.
			LLMResponse: &adk.LLMResponse{
				Content: genai.NewContentFromText("hello", "user"),
			},
		},
		{
			Author: "user",
			LLMResponse: &adk.LLMResponse{
				Content: genai.NewContentFromText("good bye", "user"),
			},
		},
	}
	agentTransfer := []*adk.Event{
		{
			Author: "anotherAgent", // History.
			LLMResponse: &adk.LLMResponse{
				Content: genai.NewContentFromFunctionCall("func1", nil, "model"),
			},
		},
		{
			Author: "anotherAgent",
			LLMResponse: &adk.LLMResponse{
				Content: genai.NewContentFromFunctionResponse("func1", nil, "user"),
			},
		},
		{
			Author: "anotherAgent", // Beginning of the current turn started by another agent.
			LLMResponse: &adk.LLMResponse{
				Content: genai.NewContentFromText("transfer to testAgent", "model"),
			},
		},
		{
			Author: agentName, // See python flows/llm_flows/base_llm_flow.py BaseLlmFlow._run_one_step_async.
			LLMResponse: &adk.LLMResponse{
				Content: genai.NewContentFromFunctionCall("func1", nil, "model"),
			},
		},
	}
	robot := []*adk.Event{
		{
			Author: agentName,
			LLMResponse: &adk.LLMResponse{
				Content: genai.NewContentFromText("do func1", "user"),
			},
		},
		{
			Author: agentName,
			LLMResponse: &adk.LLMResponse{
				Content: genai.NewContentFromFunctionCall("func1", nil, "model"),
			},
		},
		{
			Author: agentName,
			LLMResponse: &adk.LLMResponse{
				Content: genai.NewContentFromFunctionResponse("func1", nil, "user"),
			},
		},
	}

	t.Parallel()
	testCases := []struct {
		name            string
		includeContents string
		events          []*adk.Event
		want            []*genai.Content
	}{
		{
			name:            "empty",
			includeContents: "default",
			events:          emptyEvent,
		},
		{
			name:            "empty",
			includeContents: "none",
			events:          emptyEvent,
		},
		{
			name:            "helloAndGoodBye",
			includeContents: "",
			events:          helloAndGoodBye,
			want: []*genai.Content{
				genai.NewContentFromText("hello", "user"),
				genai.NewContentFromText("good bye", "user"),
			},
		},
		{
			name:            "helloAndGoodBye",
			includeContents: "default", // default == ""
			events:          helloAndGoodBye,
			want: []*genai.Content{
				genai.NewContentFromText("hello", "user"),
				genai.NewContentFromText("good bye", "user"),
			},
		},
		{
			name:            "helloAndGoodBye",
			includeContents: "none",
			events:          helloAndGoodBye,
			want: []*genai.Content{
				genai.NewContentFromText("good bye", "user"),
			},
		},
		{
			name:            "agentTransfer",
			includeContents: "",
			events:          agentTransfer,
			want: []*genai.Content{
				// events from other agents are converted by convertForeignEvent.
				{
					Parts: []*genai.Part{
						{Text: "For context:"},
						{Text: `[anotherAgent] called tool "func1" with parameters: null`},
					},
					Role: "user",
				},
				{
					Parts: []*genai.Part{
						{Text: "For context:"},
						{Text: `[anotherAgent] "func1" tool returned result: null`},
					},
					Role: "user",
				},
				{
					Parts: []*genai.Part{
						{Text: "For context:"},
						{Text: "[anotherAgent] said: transfer to testAgent"},
					},
					Role: "user",
				},
				genai.NewContentFromFunctionCall("func1", nil, "model"),
			},
		},
		{
			name:            "agentTransfer",
			includeContents: "none",
			events:          agentTransfer,
			want: []*genai.Content{
				{
					Parts: []*genai.Part{
						{Text: "For context:"},
						{Text: "[anotherAgent] said: transfer to testAgent"},
					},
					Role: "user",
				},
				genai.NewContentFromFunctionCall("func1", nil, "model"),
			},
		},
		{
			name:            "robot",
			includeContents: "default",
			events:          robot,
			want: []*genai.Content{
				genai.NewContentFromText("do func1", "user"),
				genai.NewContentFromFunctionCall("func1", nil, "model"),
				genai.NewContentFromFunctionResponse("func1", nil, "user"),
			},
		},
		{
			name:            "robot",
			includeContents: "none",
			events:          robot,
			want: []*genai.Content{
				genai.NewContentFromText("do func1", "user"),
				genai.NewContentFromFunctionCall("func1", nil, "model"),
				genai.NewContentFromFunctionResponse("func1", nil, "user"),
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name+"/include_contents="+tc.includeContents, func(t *testing.T) {
			agent := &LLMAgent{
				AgentName:       agentName,
				Model:           model,
				IncludeContents: tc.includeContents,
			}
			invCtx := &adk.InvocationContext{
				InvocationID: "12345",
				Agent:        agent,
				Session:      &adk.Session{Events: tc.events},
			}

			req := &adk.LLMRequest{Model: model}
			if err := contentsRequestProcessor(t.Context(), invCtx, req); err != nil {
				t.Fatalf("contentsRequestProcessor failed: %v", err)
			}
			got := req.Contents
			if diff := cmp.Diff(tc.want, got); diff != "" {
				t.Errorf("LLMRequest after contentsRequestProcessor mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestContentsRequestProcessor(t *testing.T) {
	const agentName = "testAgent"
	model := &model{}

	t.Parallel()
	testCases := []struct {
		name   string
		branch string
		events []*adk.Event
		want   []*genai.Content
	}{
		{
			name:   "NilEvent",
			events: nil,
			want:   nil,
		},
		{
			name:   "EmptyEvents",
			events: []*adk.Event{},
			want:   nil,
		},
		{
			name: "UserAndAgentEvents",
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
			name: "anotherAgentEvent",
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
			name:   "FilterByBranch",
			branch: "branch1",
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
			name: "AuthEvent",
			events: []*adk.Event{
				{
					Author: agentName,
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
			name: "EventWithoutContent",
			events: []*adk.Event{
				{Author: "user"},
			},
			want: nil,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			agent := &LLMAgent{
				AgentName: agentName,
				Model:     model,
			}
			invCtx := &adk.InvocationContext{
				InvocationID: "12345",
				Agent:        agent,
				Branch:       tc.branch,
				Session:      &adk.Session{Events: tc.events},
			}

			req := &adk.LLMRequest{Model: model}
			if err := contentsRequestProcessor(t.Context(), invCtx, req); err != nil {
				t.Fatalf("contentRequestProcessor failed: %v", err)
			}
			got := req.Contents
			if diff := cmp.Diff(tc.want, got); diff != "" {
				t.Errorf("LLMRequest after contentRequestProcessor mismatch (-want +got):\n%s", diff)
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
							{Text: `[foreign] called tool "test" with parameters: {"a":"b"}`},
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
							{Text: `[foreign] "test" tool returned result: {"c":"d"}`},
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
