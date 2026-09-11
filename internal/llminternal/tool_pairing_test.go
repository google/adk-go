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

package llminternal_test

import (
	"slices"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent/llmagent"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/internal/llminternal"
	"google.golang.org/adk/v2/internal/utils"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
)

// The placeholders the request processor supplies. Repeated here so a change to
// the text of either one has to be made deliberately.
const (
	missingResult = "No response available for this function call."
	pendingResult = "This call is awaiting a response and has not completed yet."
)

func functionCallContent(calls ...*genai.FunctionCall) *genai.Content {
	parts := make([]*genai.Part, 0, len(calls))
	for _, call := range calls {
		parts = append(parts, &genai.Part{FunctionCall: call})
	}
	return &genai.Content{Role: genai.RoleModel, Parts: parts}
}

func functionResponseContent(responses ...*genai.FunctionResponse) *genai.Content {
	parts := make([]*genai.Part, 0, len(responses))
	for _, response := range responses {
		parts = append(parts, &genai.Part{FunctionResponse: response})
	}
	return &genai.Content{Role: genai.RoleUser, Parts: parts}
}

// TestContentsRequestProcessor_PairUnansweredFunctionCalls covers the history a
// turn that ended between a function call and its result leaves behind. Every
// call must reach the model with a result in the immediately following content,
// because that is the pairing a provider such as Anthropic enforces.
func TestContentsRequestProcessor_PairUnansweredFunctionCalls(t *testing.T) {
	const agentName = "test_agent"

	fcSearch := &genai.FunctionCall{ID: "call_1", Name: "search_tool", Args: map[string]any{"query": "test"}}
	fcFetch := &genai.FunctionCall{ID: "call_2", Name: "fetch_tool", Args: map[string]any{"url": "http://example.com"}}
	frSearch := &genai.FunctionResponse{ID: "call_1", Name: "search_tool", Response: map[string]any{"results": "item1"}}

	fcAsk := &genai.FunctionCall{ID: "lro_call_1", Name: "ask_user", Args: map[string]any{"question": "proceed?"}}
	frAsk := &genai.FunctionResponse{ID: "lro_call_1", Name: "ask_user", Response: map[string]any{"answer": "yes"}}

	fcNoIDFirst := &genai.FunctionCall{Name: "search_tool", Args: map[string]any{"query": "a"}}
	fcNoIDSecond := &genai.FunctionCall{Name: "fetch_tool", Args: map[string]any{"url": "http://example.com"}}
	frNoID := &genai.FunctionResponse{Name: "search_tool", Response: map[string]any{"results": "item1"}}

	// Ids ADK generates for a provider that omits them. They are stripped from
	// the request, so the contents reaching the model carry an empty id.
	fcGenSearch := &genai.FunctionCall{ID: "adk-AAAA", Name: "search_tool", Args: map[string]any{"query": "test"}}
	fcGenFetch := &genai.FunctionCall{ID: "adk-BBBB", Name: "fetch_tool", Args: map[string]any{"url": "http://example.com"}}
	frGenFetch := &genai.FunctionResponse{ID: "adk-BBBB", Name: "fetch_tool", Response: map[string]any{"body": "<html/>"}}
	stripped := func(call *genai.FunctionCall) *genai.FunctionCall {
		return &genai.FunctionCall{Name: call.Name, Args: call.Args}
	}

	userEvent := func(text string) *session.Event {
		return &session.Event{
			Author:      "user",
			LLMResponse: model.LLMResponse{Content: genai.NewContentFromText(text, genai.RoleUser)},
		}
	}
	callEvent := func(longRunningIDs []string, calls ...*genai.FunctionCall) *session.Event {
		parts := make([]*genai.Part, 0, len(calls))
		for _, call := range calls {
			parts = append(parts, &genai.Part{FunctionCall: call})
		}
		return &session.Event{
			Author:             agentName,
			LongRunningToolIDs: longRunningIDs,
			LLMResponse:        model.LLMResponse{Content: &genai.Content{Role: genai.RoleModel, Parts: parts}},
		}
	}
	responseEvent := func(responses ...*genai.FunctionResponse) *session.Event {
		return &session.Event{
			Author:      "user",
			LLMResponse: model.LLMResponse{Content: functionResponseContent(responses...)},
		}
	}

	testCases := []struct {
		name   string
		events []*session.Event
		want   []*genai.Content
	}{
		{
			name: "interrupted call at the end of the history",
			events: []*session.Event{
				userEvent("search for test"),
				callEvent(nil, fcSearch),
			},
			want: []*genai.Content{
				genai.NewContentFromText("search for test", genai.RoleUser),
				functionCallContent(fcSearch),
				functionResponseContent(&genai.FunctionResponse{
					ID:       "call_1",
					Name:     "search_tool",
					Response: map[string]any{"result": missingResult},
				}),
			},
		},
		{
			name: "interrupted call in the middle of the history",
			events: []*session.Event{
				userEvent("search for test"),
				callEvent(nil, fcSearch),
				userEvent("are you still there?"),
			},
			want: []*genai.Content{
				genai.NewContentFromText("search for test", genai.RoleUser),
				functionCallContent(fcSearch),
				functionResponseContent(&genai.FunctionResponse{
					ID:       "call_1",
					Name:     "search_tool",
					Response: map[string]any{"result": missingResult},
				}),
				genai.NewContentFromText("are you still there?", genai.RoleUser),
			},
		},
		{
			name: "partially answered turn keeps the results in one content",
			events: []*session.Event{
				userEvent("search and fetch"),
				callEvent(nil, fcSearch, fcFetch),
				responseEvent(frSearch),
			},
			want: []*genai.Content{
				genai.NewContentFromText("search and fetch", genai.RoleUser),
				functionCallContent(fcSearch, fcFetch),
				functionResponseContent(
					frSearch,
					&genai.FunctionResponse{
						ID:       "call_2",
						Name:     "fetch_tool",
						Response: map[string]any{"result": missingResult},
					},
				),
			},
		},
		{
			name: "placeholder stays ahead of a trailing text part",
			events: []*session.Event{
				userEvent("search and fetch"),
				callEvent(nil, fcSearch, fcFetch),
				{
					Author: "user",
					LLMResponse: model.LLMResponse{Content: &genai.Content{
						Role: genai.RoleUser,
						Parts: []*genai.Part{
							{FunctionResponse: frSearch},
							{Text: "and please hurry"},
						},
					}},
				},
			},
			want: []*genai.Content{
				genai.NewContentFromText("search and fetch", genai.RoleUser),
				functionCallContent(fcSearch, fcFetch),
				{Role: genai.RoleUser, Parts: []*genai.Part{
					{FunctionResponse: frSearch},
					{FunctionResponse: &genai.FunctionResponse{
						ID:       "call_2",
						Name:     "fetch_tool",
						Response: map[string]any{"result": missingResult},
					}},
					{Text: "and please hurry"},
				}},
			},
		},
		{
			name: "call held open reports that it awaits a response",
			events: []*session.Event{
				userEvent("do the thing"),
				callEvent([]string{"lro_call_1"}, fcAsk),
			},
			want: []*genai.Content{
				genai.NewContentFromText("do the thing", genai.RoleUser),
				functionCallContent(fcAsk),
				functionResponseContent(&genai.FunctionResponse{
					ID:       "lro_call_1",
					Name:     "ask_user",
					Response: map[string]any{"result": pendingResult},
				}),
			},
		},
		{
			name: "answered call held open keeps its own result",
			events: []*session.Event{
				userEvent("do the thing"),
				callEvent([]string{"lro_call_1"}, fcAsk),
				responseEvent(frAsk),
			},
			want: []*genai.Content{
				genai.NewContentFromText("do the thing", genai.RoleUser),
				functionCallContent(fcAsk),
				functionResponseContent(frAsk),
			},
		},
		{
			name: "one result does not answer two calls without an id",
			events: []*session.Event{
				userEvent("search twice"),
				callEvent(nil, fcNoIDFirst, fcNoIDSecond),
				responseEvent(frNoID),
			},
			want: []*genai.Content{
				genai.NewContentFromText("search twice", genai.RoleUser),
				functionCallContent(fcNoIDFirst, fcNoIDSecond),
				functionResponseContent(
					frNoID,
					&genai.FunctionResponse{
						Name:     "fetch_tool",
						Response: map[string]any{"result": missingResult},
					},
				),
			},
		},
		{
			name: "call held open under a generated id reports that it awaits a response",
			events: []*session.Event{
				userEvent("search for test"),
				callEvent([]string{"adk-AAAA"}, fcGenSearch),
			},
			want: []*genai.Content{
				genai.NewContentFromText("search for test", genai.RoleUser),
				functionCallContent(stripped(fcGenSearch)),
				functionResponseContent(&genai.FunctionResponse{
					Name:     "search_tool",
					Response: map[string]any{"result": pendingResult},
				}),
			},
		},
		{
			name: "call held open under a generated id next to an answered call",
			events: []*session.Event{
				userEvent("do both"),
				callEvent([]string{"adk-AAAA"}, fcGenSearch, fcGenFetch),
				responseEvent(frGenFetch),
			},
			want: []*genai.Content{
				genai.NewContentFromText("do both", genai.RoleUser),
				functionCallContent(stripped(fcGenSearch), stripped(fcGenFetch)),
				functionResponseContent(
					&genai.FunctionResponse{Name: "fetch_tool", Response: frGenFetch.Response},
					&genai.FunctionResponse{
						Name:     "search_tool",
						Response: map[string]any{"result": pendingResult},
					},
				),
			},
		},
		{
			name: "results are hoisted ahead of a leading text part",
			events: []*session.Event{
				userEvent("search and fetch"),
				callEvent(nil, fcSearch, fcFetch),
				{
					Author: "user",
					LLMResponse: model.LLMResponse{Content: &genai.Content{
						Role: genai.RoleUser,
						Parts: []*genai.Part{
							{Text: "here is what I found"},
							{FunctionResponse: frSearch},
						},
					}},
				},
			},
			want: []*genai.Content{
				genai.NewContentFromText("search and fetch", genai.RoleUser),
				functionCallContent(fcSearch, fcFetch),
				{Role: genai.RoleUser, Parts: []*genai.Part{
					{FunctionResponse: frSearch},
					{FunctionResponse: &genai.FunctionResponse{
						ID:       "call_2",
						Name:     "fetch_tool",
						Response: map[string]any{"result": missingResult},
					}},
					{Text: "here is what I found"},
				}},
			},
		},
		{
			name: "history whose calls are all answered is left alone",
			events: []*session.Event{
				userEvent("search for test"),
				callEvent(nil, fcSearch),
				responseEvent(frSearch),
			},
			want: []*genai.Content{
				genai.NewContentFromText("search for test", genai.RoleUser),
				functionCallContent(fcSearch),
				functionResponseContent(frSearch),
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			testAgent := utils.Must(llmagent.New(llmagent.Config{
				Name:  agentName,
				Model: &testModel{},
			}))
			ctx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{
				Agent:   testAgent,
				Session: &fakeSession{events: tc.events},
			})

			req := &model.LLMRequest{}
			for _, err := range llminternal.ContentsRequestProcessor(ctx, req, &llminternal.Flow{}) {
				if err != nil {
					t.Fatalf("ContentsRequestProcessor failed: %v", err)
				}
			}

			if diff := cmp.Diff(tc.want, req.Contents); diff != "" {
				t.Errorf("Contents mismatch (-want +got):\n%s", diff)
			}
			assertCallsAreAnswered(t, req.Contents)
		})
	}
}

// assertCallsAreAnswered checks the invariant the provider enforces: the
// content that follows a call answers every call it carries, once. A call is
// matched to a response by id, or by name when the id is empty, and each
// response answers a single call.
func assertCallsAreAnswered(t *testing.T, contents []*genai.Content) {
	t.Helper()
	for i, content := range contents {
		calls := utils.FunctionCalls(content)
		if len(calls) == 0 {
			continue
		}
		var following *genai.Content
		if i+1 < len(contents) {
			following = contents[i+1]
		}
		remaining := utils.FunctionResponses(following)
		for _, call := range calls {
			matched := slices.IndexFunc(remaining, func(response *genai.FunctionResponse) bool {
				if call.ID != "" {
					return response.ID == call.ID
				}
				return response.ID == "" && response.Name == call.Name
			})
			if matched < 0 {
				t.Errorf("content %d: call %s (id %q) has no result in the following content", i, call.Name, call.ID)
				continue
			}
			remaining = slices.Delete(remaining, matched, matched+1)
		}
		for _, response := range remaining {
			t.Errorf("content %d: result for %s (id %q) answers no call", i+1, response.Name, response.ID)
		}
	}
}

// TestContentsRequestProcessor_UnpairedHistoryFailsTheCheck guards
// assertCallsAreAnswered: the same check must reject a history that was never
// repaired, otherwise it would pass on any input.
func TestContentsRequestProcessor_UnpairedHistoryFailsTheCheck(t *testing.T) {
	unpaired := []*genai.Content{
		genai.NewContentFromText("search for test", genai.RoleUser),
		genai.NewContentFromFunctionCall("search_tool", map[string]any{"query": "test"}, genai.RoleModel),
		genai.NewContentFromText("are you still there?", genai.RoleUser),
	}

	recorder := &testing.T{}
	assertCallsAreAnswered(recorder, unpaired)
	if !recorder.Failed() {
		t.Error("assertCallsAreAnswered accepted a history with an unanswered call")
	}

	fetch := &genai.FunctionResponse{Name: "fetch_tool", Response: map[string]any{"body": "<html/>"}}
	doubled := []*genai.Content{
		genai.NewContentFromText("do both", genai.RoleUser),
		functionCallContent(
			&genai.FunctionCall{Name: "search_tool", Args: map[string]any{"query": "test"}},
			&genai.FunctionCall{Name: "fetch_tool", Args: map[string]any{"url": "http://example.com"}},
		),
		functionResponseContent(fetch, fetch),
	}

	recorder = &testing.T{}
	assertCallsAreAnswered(recorder, doubled)
	if !recorder.Failed() {
		t.Error("assertCallsAreAnswered accepted two results for one call and none for the other")
	}
}
