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

package llminternal

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
)

// TestBuildContents_UserResponseToOtherAgentCallIsContext checks that a user's
// function response to a call another agent made reaches the current agent as
// context, like the call itself, as in adk-python's _get_contents. Left as a
// raw response, its call is missing from the history: it is dropped mid-history
// and fails the request when it is the last event.
func TestBuildContents_UserResponseToOtherAgentCallIsContext(t *testing.T) {
	event := func(author string, content *genai.Content) *session.Event {
		return &session.Event{Author: author, LLMResponse: model.LLMResponse{Content: content}}
	}
	onBranch := func(branch string, ev *session.Event) *session.Event {
		ev.Branch = branch
		return ev
	}
	call := func(id string) *genai.Content {
		return &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{ID: id, Name: "adk_request_input", Args: map[string]any{"message": "approve"}}}}}
	}
	response := func(id string) *genai.Content {
		return &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{ID: id, Name: "adk_request_input", Response: map[string]any{"result": "ok"}}}}}
	}
	context := func(parts ...string) *genai.Content {
		c := &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{{Text: "For context:"}}}
		for _, p := range parts {
			c.Parts = append(c.Parts, &genai.Part{Text: p})
		}
		return c
	}
	calledB := context(`[b] called tool ` + "`adk_request_input`" + ` with parameters: {"message":"approve"}`)
	answered := context(`[user] ` + "`adk_request_input`" + ` tool returned result: {"result":"ok"}`)

	done := context("[d] said: done")
	tests := []struct {
		name   string
		branch string
		events []*session.Event
		want   []*genai.Content
	}{
		{
			name: "response is the last event",
			events: []*session.Event{
				event("user", genai.NewContentFromText("start", genai.RoleUser)),
				event("b", call("b1")),
				event("user", response("b1")),
			},
			want: []*genai.Content{genai.NewContentFromText("start", genai.RoleUser), calledB, answered},
		},
		{
			name: "response mid-history",
			events: []*session.Event{
				event("user", genai.NewContentFromText("start", genai.RoleUser)),
				event("b", call("b1")),
				event("user", response("b1")),
				event("d", genai.NewContentFromText("done", genai.RoleModel)),
			},
			want: []*genai.Content{genai.NewContentFromText("start", genai.RoleUser), calledB, answered, done},
		},
		{
			name: "response to the agent's own call stays a response",
			events: []*session.Event{
				event("user", genai.NewContentFromText("start", genai.RoleUser)),
				event("after", call("a1")),
				event("user", response("a1")),
			},
			want: []*genai.Content{genai.NewContentFromText("start", genai.RoleUser), call("a1"), response("a1")},
		},
		{
			// Some providers reuse call ids. A response pairs with the latest
			// call before it, so a later call with the same id does not turn
			// the agent's own exchange into context.
			name: "call id reused by another agent later",
			events: []*session.Event{
				event("user", genai.NewContentFromText("start", genai.RoleUser)),
				event("after", call("c0")),
				event("after", response("c0")),
				event("b", call("c0")),
				event("user", response("c0")),
			},
			want: []*genai.Content{
				genai.NewContentFromText("start", genai.RoleUser),
				call("c0"),
				response("c0"),
				context(`[b] called tool ` + "`adk_request_input`" + ` with parameters: {"message":"approve"}`),
				context(`[user] ` + "`adk_request_input`" + ` tool returned result: {"result":"ok"}`),
			},
		},
		{
			name: "response to a call the user made stays a response",
			events: []*session.Event{
				event("user", genai.NewContentFromText("start", genai.RoleUser)),
				event("user", call("u1")),
				event("user", response("u1")),
			},
			want: []*genai.Content{genai.NewContentFromText("start", genai.RoleUser), call("u1"), response("u1")},
		},
		{
			// The call reaches the agent as text, so nothing is left for the
			// response to pair with and history rearrangement drops it.
			name: "response without an id is not matched",
			events: []*session.Event{
				event("user", genai.NewContentFromText("start", genai.RoleUser)),
				event("b", call("")),
				event("user", response("")),
				event("d", genai.NewContentFromText("done", genai.RoleModel)),
			},
			want: []*genai.Content{
				genai.NewContentFromText("start", genai.RoleUser),
				context(`[b] called tool ` + "`adk_request_input`" + ` with parameters: {"message":"approve"}`),
				done,
			},
		},
		{
			// The call is on a branch this agent does not see, so its
			// response is dropped as an orphan, as in adk-python.
			name:   "response to a call outside the agent's branch is not shown",
			branch: "root.after",
			events: []*session.Event{
				event("user", genai.NewContentFromText("start", genai.RoleUser)),
				onBranch("root.wf", event("b", call("b1"))),
				event("user", response("b1")),
				event("d", genai.NewContentFromText("done", genai.RoleModel)),
			},
			want: []*genai.Content{genai.NewContentFromText("start", genai.RoleUser), done},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := buildContentsDefault("after", tc.branch, "", tc.events, false, nil)
			if err != nil {
				t.Fatal(err)
			}
			if diff := cmp.Diff(tc.want, got); diff != "" {
				t.Errorf("contents mismatch (-want +got):\n%s", diff)
			}
		})
	}
}
