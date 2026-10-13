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
	"encoding/json"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	icontext "google.golang.org/adk/internal/context"
	"google.golang.org/adk/internal/llminternal"
	"google.golang.org/adk/internal/utils"
	"google.golang.org/adk/model"
	"google.golang.org/adk/session"
)

func TestContentsRequestProcessor_ForeignThoughts(t *testing.T) {
	t.Parallel()

	publicPart := &genai.Part{
		InlineData:       &genai.Blob{MIMEType: "image/png", Data: []byte{1, 2, 3}},
		ThoughtSignature: []byte{4, 5},
	}
	for _, tc := range []struct {
		name, author string
		parts        []*genai.Part
		want         []*genai.Content
	}{
		{
			name: "thought-only foreign event is omitted", author: "otherAgent",
			parts: []*genai.Part{{Text: "private reasoning", Thought: true}},
		},
		{
			name: "mixed foreign event retains public parts", author: "otherAgent",
			parts: []*genai.Part{
				{Text: "private reasoning", Thought: true},
				{Text: "public answer"},
				publicPart,
			},
			want: []*genai.Content{{Role: "user", Parts: []*genai.Part{
				{Text: "For context:"},
				{Text: "[otherAgent] said: public answer"},
				publicPart,
			}}},
		},
		{
			name: "own-agent thoughts are retained", author: "testAgent",
			parts: []*genai.Part{{Text: "own reasoning", Thought: true, ThoughtSignature: []byte{6, 7}}},
			want: []*genai.Content{{Role: "model", Parts: []*genai.Part{
				{Text: "own reasoning", Thought: true, ThoughtSignature: []byte{6, 7}},
			}}},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			user := foreignThoughtTextEvent("user", 1, "request")
			event := &session.Event{
				ID: "response-event", InvocationID: "response-invocation",
				Author: tc.author, Timestamp: foreignThoughtAt(2),
				LLMResponse: model.LLMResponse{Content: &genai.Content{Role: "model", Parts: tc.parts}},
				Actions:     session.EventActions{StateDelta: map[string]any{"state": "unchanged"}},
			}
			events := []*session.Event{user, event}
			ctx := foreignThoughtInvocationCtx(t, events, "")
			got := foreignThoughtRequestContents(t, events, ctx)
			want := append([]*genai.Content{user.Content}, tc.want...)
			if !cmp.Equal(want, got) {
				t.Error("request contents did not preserve the expected public and own-agent parts")
			}
		})
	}
}

func TestContentsRequestProcessor_ForeignThoughtCurrentTurn(t *testing.T) {
	t.Parallel()

	for _, tc := range []struct {
		name, author      string
		parts             []*genai.Part
		successiveThought bool
		keepCurrentUser   bool
		wantTail          []*genai.Content
	}{
		{
			name: "foreign text thought does not start a turn", author: "otherAgent",
			parts: []*genai.Part{{Text: "private reasoning", Thought: true}}, keepCurrentUser: true,
		},
		{
			name: "foreign empty thought does not start a turn", author: "otherAgent",
			parts: []*genai.Part{{Thought: true}}, keepCurrentUser: true,
		},
		{
			name: "multiple foreign thought parts do not start a turn", author: "otherAgent",
			parts: []*genai.Part{{Text: "private reasoning", Thought: true}, {Thought: true}}, keepCurrentUser: true,
		},
		{
			name: "successive foreign thoughts do not start a turn", author: "otherAgent",
			parts:             []*genai.Part{{Text: "private reasoning", Thought: true}},
			successiveThought: true, keepCurrentUser: true,
		},
		{
			name: "mixed foreign response still starts a turn", author: "otherAgent",
			parts: []*genai.Part{{Text: "private reasoning", Thought: true}, {Text: "public answer"}},
			wantTail: []*genai.Content{{Role: "user", Parts: []*genai.Part{
				{Text: "For context:"}, {Text: "[otherAgent] said: public answer"},
			}}},
		},
		{
			name: "own thoughts remain in the current user turn", author: "testAgent",
			parts: []*genai.Part{{Text: "own reasoning", Thought: true}}, keepCurrentUser: true,
			wantTail: []*genai.Content{{Role: "model", Parts: []*genai.Part{
				{Text: "own reasoning", Thought: true},
			}}},
		},
		{
			name: "user thought content starts its own turn", author: "user",
			parts: []*genai.Part{{Text: "user input", Thought: true}},
			wantTail: []*genai.Content{{Role: "user", Parts: []*genai.Part{
				{Text: "user input", Thought: true},
			}}},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			current := foreignThoughtTextEvent("user", 3, "current request")
			event := &session.Event{
				Author: tc.author, Timestamp: foreignThoughtAt(4),
				LLMResponse: model.LLMResponse{Content: &genai.Content{Role: "model", Parts: tc.parts}},
			}
			if tc.author == "user" {
				event.Content.Role = "user"
			}
			events := []*session.Event{
				foreignThoughtTextEvent("user", 1, "older request"),
				foreignThoughtTextEvent("testAgent", 2, "older answer"),
				current, event,
			}
			if tc.successiveThought {
				events = append(events, &session.Event{
					Author: "anotherAgent", Timestamp: foreignThoughtAt(5),
					LLMResponse: model.LLMResponse{Content: &genai.Content{
						Role: "model", Parts: []*genai.Part{{Thought: true}},
					}},
				})
			}
			ctx := foreignThoughtInvocationCtx(t, events, "none")
			got := foreignThoughtRequestContents(t, events, ctx)
			var want []*genai.Content
			if tc.keepCurrentUser {
				want = append(want, current.Content)
			}
			want = append(want, tc.wantTail...)
			if !cmp.Equal(want, got) {
				t.Error("request contents selected the wrong current-turn boundary")
			}
		})
	}
}

var foreignThoughtEpoch = time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)

func foreignThoughtAt(n int) time.Time {
	return foreignThoughtEpoch.Add(time.Duration(n) * time.Second)
}

func foreignThoughtTextEvent(author string, ts int, text string) *session.Event {
	role := "model"
	if author == "user" {
		role = "user"
	}
	return &session.Event{
		Author:      author,
		Timestamp:   foreignThoughtAt(ts),
		LLMResponse: model.LLMResponse{Content: genai.NewContentFromText(text, genai.Role(role))},
	}
}

func foreignThoughtInvocationCtx(t *testing.T, events []*session.Event, includeContents llmagent.IncludeContents) agent.InvocationContext {
	t.Helper()
	testAgent := utils.Must(llmagent.New(llmagent.Config{
		Name: "testAgent", Model: &testModel{}, IncludeContents: includeContents,
	}))
	return icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{
		Agent:   testAgent,
		Session: &fakeSession{events: events},
	})
}

func foreignThoughtRequestContents(t *testing.T, events []*session.Event, ctx agent.InvocationContext) []*genai.Content {
	t.Helper()
	before, err := json.Marshal(events)
	if err != nil {
		t.Fatal("could not snapshot session events")
	}
	req := &model.LLMRequest{}
	for event, err := range llminternal.ContentsRequestProcessor(ctx, req, &llminternal.Flow{}) {
		if event != nil || err != nil {
			t.Fatal("contents processing yielded an unexpected event or error")
		}
	}
	after, err := json.Marshal(events)
	if err != nil {
		t.Fatal("could not snapshot session events after processing")
	}
	if !cmp.Equal(before, after) {
		t.Error("contents processing mutated the original session events")
	}
	return req.Contents
}
