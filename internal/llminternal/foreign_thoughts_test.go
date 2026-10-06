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

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/internal/llminternal"
	"google.golang.org/adk/v2/internal/utils"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
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
			user := compactionTextEvent("user", 1, "request")
			event := &session.Event{
				ID: "response-event", InvocationID: "response-invocation",
				Author: tc.author, Timestamp: compactionAt(2),
				LLMResponse: model.LLMResponse{Content: &genai.Content{Role: "model", Parts: tc.parts}},
				Actions:     session.EventActions{StateDelta: map[string]any{"state": "unchanged"}},
			}
			events := []*session.Event{user, event}
			ctx := compactionInvocationCtx(t, "testAgent", events, false)
			got := foreignThoughtRequestContents(t, events, ctx)
			want := append([]*genai.Content{user.Content}, tc.want...)
			if !cmp.Equal(wantWithContinuation(want), got) {
				t.Error("request contents did not preserve the expected public and own-agent parts")
			}
		})
	}
}

func TestContentsRequestProcessor_ForeignThoughtCompactionSummary(t *testing.T) {
	t.Parallel()

	for _, usable := range []bool{true, false} {
		name := "usable summary bypasses foreign thought conversion"
		if !usable {
			name = "unusable summary does not expose foreign thoughts"
		}
		t.Run(name, func(t *testing.T) {
			old := compactionTextEvent("user", 1, "old request")
			recent := compactionTextEvent("user", 4, "recent request")
			summary := compactionSummaryEvent(3, 1, 2, "public summary")
			summary.Author = "otherAgent"
			summary.Content = &genai.Content{Role: "model", Parts: []*genai.Part{
				{Text: "private summary reasoning", Thought: true},
			}}
			want := []*genai.Content{summary.Actions.Compaction.CompactedContent, recent.Content}
			if !usable {
				summary.Actions.Compaction.CompactedContent = nil
				want = []*genai.Content{old.Content, recent.Content}
			}
			events := []*session.Event{old, summary, recent}
			ctx := compactionInvocationCtx(t, "testAgent", events, true)
			got := foreignThoughtRequestContents(t, events, ctx)
			if !cmp.Equal(want, got) {
				t.Error("foreign thought filtering changed compaction summary handling")
			}
		})
	}
}

func TestContentsRequestProcessor_ForeignThoughtCompactionHole(t *testing.T) {
	t.Parallel()

	foreign := compactionTextEvent("otherAgent", 2, "public answer")
	foreign.InvocationID = "foreign-invocation"
	foreign.Content.Parts = append([]*genai.Part{{Text: "private reasoning", Thought: true}}, foreign.Content.Parts...)
	summary := compactionSummaryEvent(3, 1, 2, "public summary")
	summary.Actions.Compaction.ExcludedEvents = []session.EventRef{{
		InvocationID: foreign.InvocationID, Timestamp: foreign.Timestamp,
	}}
	recent := compactionTextEvent("user", 4, "recent request")
	events := []*session.Event{compactionTextEvent("user", 1, "old request"), foreign, summary, recent}
	ctx := compactionInvocationCtx(t, "testAgent", events, true)
	got := foreignThoughtRequestContents(t, events, ctx)
	want := []*genai.Content{
		summary.Actions.Compaction.CompactedContent,
		{Role: "user", Parts: []*genai.Part{
			{Text: "For context:"},
			{Text: "[otherAgent] said: public answer"},
		}},
		recent.Content,
	}
	if !cmp.Equal(want, got) {
		t.Error("filtering foreign thoughts lost the event protected by the compaction hole")
	}
}

func TestContentsRequestProcessor_ForeignThoughtCurrentTurn(t *testing.T) {
	t.Parallel()

	for _, tc := range []struct {
		name, author      string
		parts             []*genai.Part
		successiveThought bool
		summary           bool
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
		{
			name: "usable foreign summary still starts a turn", author: "otherAgent",
			parts: []*genai.Part{{Text: "private summary reasoning", Thought: true}}, summary: true,
			wantTail: []*genai.Content{genai.NewContentFromText("public summary", "model")},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			current := compactionTextEvent("user", 3, "current request")
			event := &session.Event{
				Author: tc.author, Timestamp: compactionAt(4),
				LLMResponse: model.LLMResponse{Content: &genai.Content{Role: "model", Parts: tc.parts}},
			}
			if tc.author == "user" {
				event.Content.Role = "user"
			}
			if tc.summary {
				// The current user is outside this range, so skipping the summary
				// pivot would incorrectly include that user alongside the summary.
				event.Actions = compactionSummaryEvent(4, 1, 2, "public summary").Actions
			}
			events := []*session.Event{
				compactionTextEvent("user", 1, "older request"),
				compactionTextEvent("testAgent", 2, "older answer"),
				current, event,
			}
			if tc.successiveThought {
				events = append(events, &session.Event{
					Author: "anotherAgent", Timestamp: compactionAt(5),
					LLMResponse: model.LLMResponse{Content: &genai.Content{
						Role: "model", Parts: []*genai.Part{{Thought: true}},
					}},
				})
			}
			base := compactionInvocationCtx(t, "testAgent", events, tc.summary)
			testAgent := utils.Must(llmagent.New(llmagent.Config{
				Name: "testAgent", Model: &testModel{}, IncludeContents: "none",
			}))
			ctx := icontext.NewInvocationContext(base, icontext.InvocationContextParams{
				Agent: testAgent, Session: base.Session(),
			})
			got := foreignThoughtRequestContents(t, events, ctx)
			var want []*genai.Content
			if tc.keepCurrentUser {
				want = append(want, current.Content)
			}
			want = append(want, tc.wantTail...)
			if !cmp.Equal(wantWithContinuation(want), got) {
				t.Error("request contents selected the wrong current-turn boundary")
			}
		})
	}
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
