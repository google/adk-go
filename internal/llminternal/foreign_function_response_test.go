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
			// The call is on a branch this agent does not see, so the response
			// stays a raw response, and history rearrangement drops it because
			// no call before it matches. adk-python drops it too, as an orphan.
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

// TestBuildContents_UserTextWithResponseToOtherAgentCall checks that text sent
// in the same message as the answer to another agent's call arrives as context
// with it, whether the agent gets its history or only the current turn.
func TestBuildContents_UserTextWithResponseToOtherAgentCall(t *testing.T) {
	call := &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{ID: "b1", Name: "adk_request_input", Args: map[string]any{"message": "approve"}}}}}
	answer := &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{
		{Text: "note"},
		{FunctionResponse: &genai.FunctionResponse{ID: "b1", Name: "adk_request_input", Response: map[string]any{"result": "ok"}}},
	}}
	events := []*session.Event{
		{Author: "user", LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("start", genai.RoleUser)}},
		{Author: "b", LLMResponse: model.LLMResponse{Content: call}},
		{Author: "user", LLMResponse: model.LLMResponse{Content: answer}},
	}
	calledB := &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{
		{Text: "For context:"},
		{Text: `[b] called tool ` + "`adk_request_input`" + ` with parameters: {"message":"approve"}`},
	}}
	answered := &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{
		{Text: "For context:"},
		{Text: "[user] said: note"},
		{Text: `[user] ` + "`adk_request_input`" + ` tool returned result: {"result":"ok"}`},
	}}

	tests := []struct {
		name  string
		build func() ([]*genai.Content, error)
		want  []*genai.Content
	}{
		{
			name:  "history",
			build: func() ([]*genai.Content, error) { return buildContentsDefault("after", "", "", events, false, nil) },
			want:  []*genai.Content{genai.NewContentFromText("start", genai.RoleUser), calledB, answered},
		},
		{
			name: "current turn only",
			build: func() ([]*genai.Content, error) {
				return buildContentsCurrentTurnContextOnly("after", "", "", events, false, nil)
			},
			want: []*genai.Content{calledB, answered},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := tc.build()
			if err != nil {
				t.Fatal(err)
			}
			if diff := cmp.Diff(tc.want, got); diff != "" {
				t.Errorf("contents mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// TestBuildContents_ResponseInSuffixToOtherAgentCallBeforeIt checks that when
// only a suffix of the session is built, as the compaction token estimate does,
// a response in the suffix pairs with the call before it the same way as when
// the whole session is built.
func TestBuildContents_ResponseInSuffixToOtherAgentCallBeforeIt(t *testing.T) {
	event := func(author, branch string, content *genai.Content) *session.Event {
		return &session.Event{Author: author, Branch: branch, LLMResponse: model.LLMResponse{Content: content}}
	}
	callWithRole := func(role, id string) *genai.Content {
		return &genai.Content{Role: role, Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{ID: id, Name: "adk_request_input", Args: map[string]any{"message": "approve"}}}}}
	}
	call := func(id string) *genai.Content { return callWithRole(genai.RoleModel, id) }
	response := func(id string) *genai.Content {
		return &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{ID: id, Name: "adk_request_input", Response: map[string]any{"result": "ok"}}}}}
	}
	start := event("user", "", genai.NewContentFromText("start", genai.RoleUser))
	d := event("d", "", genai.NewContentFromText("done", genai.RoleModel))
	done := &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{{Text: "For context:"}, {Text: "[d] said: done"}}}
	answered := &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{{Text: "For context:"}, {Text: `[user] ` + "`adk_request_input`" + ` tool returned result: {"result":"ok"}`}}}
	calledB := &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{{Text: "For context:"}, {Text: `[b] called tool ` + "`adk_request_input`" + ` with parameters: {"message":"approve"}`}}}

	tests := []struct {
		name   string
		branch string
		events []*session.Event
		// from is the index in events where the built suffix starts.
		from int
		want []*genai.Content
	}{
		{
			name:   "call visible to the agent",
			events: []*session.Event{start, event("b", "", call("b1")), event("user", "", response("b1")), d},
			from:   2,
			want:   []*genai.Content{answered, done},
		},
		{
			// As when the whole session is built, a call the agent does not see
			// leaves its response unmatched, and it is dropped.
			name:   "call outside the agent's branch",
			branch: "root.after",
			events: []*session.Event{start, event("b", "root.wf", call("b1")), event("user", "", response("b1")), d},
			from:   2,
			want:   []*genai.Content{done},
		},
		{
			// The whole build skips an event whose content has no role, so its
			// call does not pair with the response there either.
			name:   "call event without a role",
			events: []*session.Event{start, event("b", "", callWithRole("", "b1")), event("user", "", response("b1")), d},
			from:   2,
			want:   []*genai.Content{done},
		},
		{
			// Only calls before the suffix are recorded: b's later call with the
			// same id must not turn the agent's own response into context.
			name:   "id reused by a call after the suffix starts",
			events: []*session.Event{start, event("after", "", call("c0")), event("after", "", response("c0")), event("b", "", call("c0"))},
			from:   2,
			want:   []*genai.Content{calledB},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := buildContentsDefaultWithCallSource("after", tc.branch, "", tc.events[tc.from:], tc.events, false, nil)
			if err != nil {
				t.Fatal(err)
			}
			if diff := cmp.Diff(tc.want, got); diff != "" {
				t.Errorf("contents mismatch (-want +got):\n%s", diff)
			}
		})
	}
}
