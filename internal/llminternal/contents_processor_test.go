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

package llminternal_test

import (
	"iter"
	"slices"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/internal/llminternal"
	"google.golang.org/adk/internal/utils"
	"google.golang.org/adk/llm"
	"google.golang.org/adk/session"
	"google.golang.org/genai"
)

type model struct {
	llm.Model
}

// Test behavior around Agent's IncludeContents.
func TestContentsRequestProcessor_IncludeContents(t *testing.T) {
	const agentName = "testAgent"
	model := &model{}

	emptyEvent := []*session.Event{}
	helloAndGoodBye := []*session.Event{
		{
			Author: "user", // Not in the current turn in multi-agent scenario. See buildContentsCurrentTurnContextOnly.
			LLMResponse: &llm.Response{
				Content: genai.NewContentFromText("hello", "user"),
			},
		},
		{
			Author: "user",
			LLMResponse: &llm.Response{
				Content: genai.NewContentFromText("good bye", "user"),
			},
		},
	}
	agentTransfer := []*session.Event{
		{
			Author: "anotherAgent", // History.
			LLMResponse: &llm.Response{
				Content: genai.NewContentFromFunctionCall("func1", nil, "model"),
			},
		},
		{
			Author: "anotherAgent",
			LLMResponse: &llm.Response{
				Content: genai.NewContentFromFunctionResponse("func1", nil, "user"),
			},
		},
		{
			Author: "anotherAgent", // Beginning of the current turn started by another agent.
			LLMResponse: &llm.Response{
				Content: genai.NewContentFromText("transfer to testAgent", "model"),
			},
		},
		{
			Author: agentName, // See python flows/llm_flows/base_llm_flow.py BaseLlmFlow._run_one_step_async.
			LLMResponse: &llm.Response{
				Content: genai.NewContentFromFunctionCall("func1", nil, "model"),
			},
		},
	}
	robot := []*session.Event{
		{
			Author: agentName,
			LLMResponse: &llm.Response{
				Content: genai.NewContentFromText("do func1", "user"),
			},
		},
		{
			Author: agentName,
			LLMResponse: &llm.Response{
				Content: genai.NewContentFromFunctionCall("func1", nil, "model"),
			},
		},
		{
			Author: agentName,
			LLMResponse: &llm.Response{
				Content: genai.NewContentFromFunctionResponse("func1", nil, "user"),
			},
		},
	}

	t.Parallel()
	testCases := []struct {
		name            string
		includeContents string
		events          []*session.Event
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
			testAgent := utils.Must(llmagent.New(llmagent.Config{
				Name:            agentName,
				Model:           model,
				IncludeContents: tc.includeContents,
			}))

			ctx := agent.NewContext(t.Context(), testAgent, nil, &fakeSession{
				events: tc.events,
			}, "")

			req := &llm.Request{}
			if err := llminternal.ContentsRequestProcessor(ctx, req); err != nil {
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
		events []*session.Event
		want   []*genai.Content
	}{
		{
			name:   "NilEvent",
			events: nil,
			want:   nil,
		},
		{
			name:   "EmptyEvents",
			events: []*session.Event{},
			want:   nil,
		},
		{
			name: "UserAndAgentEvents",
			events: []*session.Event{
				{
					Author: "user",
					LLMResponse: &llm.Response{
						Content: genai.NewContentFromText("Hello", "user"),
					},
				},
				{
					Author: "testAgent",
					LLMResponse: &llm.Response{
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
			events: []*session.Event{
				{
					Author: "anotherAgent",
					LLMResponse: &llm.Response{
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
			events: []*session.Event{
				{
					Author: "user",
					Branch: "branch1",
					LLMResponse: &llm.Response{
						Content: genai.NewContentFromText("In branch 1", "user"),
					},
				},
				{
					Author: "user",
					Branch: "branch1.task1",
					LLMResponse: &llm.Response{
						Content: genai.NewContentFromText("In branch 1 and task 1", "user"),
					},
				},
				{
					Author: "user",
					Branch: "branch12",
					LLMResponse: &llm.Response{
						Content: genai.NewContentFromText("In branch 12", "user"),
					},
				},
				{
					Author: "user",
					Branch: "branch2",
					LLMResponse: &llm.Response{
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
			events: []*session.Event{
				{
					Author: agentName,
					LLMResponse: &llm.Response{
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
			events: []*session.Event{
				{Author: "user"},
			},
			want: nil,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			testAgent := utils.Must(llmagent.New(llmagent.Config{
				Name:  "testAgent",
				Model: model,
			}))

			ctx := agent.NewContext(t.Context(), testAgent, nil, &fakeSession{
				events: tc.events,
			}, tc.branch)

			req := &llm.Request{}
			if err := llminternal.ContentsRequestProcessor(ctx, req); err != nil {
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
		event *session.Event
		want  *session.Event
	}{
		{
			name: "Text",
			event: &session.Event{
				Time:   now,
				Author: "foreign",
				LLMResponse: &llm.Response{
					Content: genai.NewContentFromText("hello", "model"),
				},
				Branch: "b",
			},
			want: &session.Event{
				Time:   now,
				Author: "user",
				LLMResponse: &llm.Response{
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
			event: &session.Event{
				Time:   now,
				Author: "foreign",
				LLMResponse: &llm.Response{
					Content: &genai.Content{
						Role: "model",
						Parts: []*genai.Part{
							{FunctionCall: &genai.FunctionCall{Name: "test", Args: map[string]any{"a": "b"}}},
						},
					},
				},
				Branch: "b",
			},
			want: &session.Event{
				Time:   now,
				Author: "user",
				LLMResponse: &llm.Response{
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
			event: &session.Event{
				Time:   now,
				Author: "foreign",
				LLMResponse: &llm.Response{
					Content: &genai.Content{
						Role: "model",
						Parts: []*genai.Part{
							{FunctionResponse: &genai.FunctionResponse{Name: "test", Response: map[string]any{"c": "d"}}},
						},
					},
				},
				Branch: "b",
			},
			want: &session.Event{
				Time:   now,
				Author: "user",
				LLMResponse: &llm.Response{
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
			got := llminternal.ConvertForeignEvent(tc.event)
			if diff := cmp.Diff(tc.want, got, cmp.AllowUnexported(genai.FunctionCall{}, genai.FunctionResponse{})); diff != "" {
				t.Errorf("convertForeignEvent() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestContentsRequestProcessor_NonLLMAgent(t *testing.T) {
	testAgent := utils.Must(agent.New(agent.Config{
		Name: "test_agent",
	}))

	ctx := agent.NewContext(t.Context(), testAgent, nil, nil, "")

	req := &llm.Request{}
	if err := llminternal.ContentsRequestProcessor(ctx, req); err != nil {
		t.Fatalf("contentRequestProcessor failed: %v", err)
	}
	got := req
	want := &llm.Request{}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("LLMRequest after contentRequestProcessor mismatch (-want +got):\n%s", diff)
	}
}

type fakeSession struct {
	events []*session.Event
}

func (s *fakeSession) State() session.State {
	return nil
}

func (s *fakeSession) Events() session.Events {
	return s
}

func (s *fakeSession) ID() session.ID {
	return session.ID{}
}

func (s *fakeSession) Updated() time.Time {
	return time.Time{}
}

func (s *fakeSession) All() iter.Seq[*session.Event] {
	return slices.Values(s.events)
}

func (s *fakeSession) Len() int {
	return len(s.events)
}

func (s *fakeSession) At(i int) *session.Event {
	return s.events[i]
}

var _ session.Session = (*fakeSession)(nil)
var _ session.Events = (*fakeSession)(nil)
