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

package runner

import (
	"iter"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/session"
)

// stubEvents is a session.Events backed by a plain slice.
type stubEvents []*session.Event

func (e stubEvents) All() iter.Seq[*session.Event] {
	return func(yield func(*session.Event) bool) {
		for _, ev := range e {
			if !yield(ev) {
				return
			}
		}
	}
}

func (e stubEvents) Len() int                { return len(e) }
func (e stubEvents) At(i int) *session.Event { return e[i] }

// stubSession satisfies session.Session but only serves Events; the
// embedded nil interface panics on any other method, which keeps these
// tests honest about what answeredInterrupts is allowed to read.
type stubSession struct {
	session.Session
	events stubEvents
}

func (s stubSession) Events() session.Events { return s.events }

func sessionOf(events ...*session.Event) session.Session {
	return stubSession{events: events}
}

// interruptEvent is a model turn that raised long-running tool calls.
func interruptEvent(ids ...string) *session.Event {
	ev := &session.Event{}
	ev.LongRunningToolIDs = ids
	ev.Content = &genai.Content{Role: "model", Parts: callParts(ids...)}
	return ev
}

// answerEvent is a past user turn that already answered ids.
func answerEvent(ids ...string) *session.Event {
	ev := &session.Event{}
	ev.Content = userTurn(responseParts(ids...)...)
	return ev
}

func callParts(ids ...string) []*genai.Part {
	parts := make([]*genai.Part, 0, len(ids))
	for _, id := range ids {
		parts = append(parts, &genai.Part{
			FunctionCall: &genai.FunctionCall{ID: id, Name: "ask"},
		})
	}
	return parts
}

func responseParts(ids ...string) []*genai.Part {
	parts := make([]*genai.Part, 0, len(ids))
	for _, id := range ids {
		parts = append(parts, responsePart(id))
	}
	return parts
}

func responsePart(id string) *genai.Part {
	return &genai.Part{
		FunctionResponse: &genai.FunctionResponse{
			ID:       id,
			Name:     "ask",
			Response: map[string]any{"payload": "approve"},
		},
	}
}

func userTurn(parts ...*genai.Part) *genai.Content {
	return &genai.Content{Role: genai.RoleUser, Parts: parts}
}

func TestAnsweredInterrupts(t *testing.T) {
	tests := []struct {
		name        string
		sess        session.Session
		userContent *genai.Content
		want        map[string]bool
	}{
		{
			name:        "nil session",
			sess:        nil,
			userContent: userTurn(responsePart("call-1")),
			want:        nil,
		},
		{
			name:        "nil user content",
			sess:        sessionOf(interruptEvent("call-1")),
			userContent: nil,
			want:        nil,
		},
		{
			name:        "answers the open interrupt",
			sess:        sessionOf(interruptEvent("call-1")),
			userContent: userTurn(responsePart("call-1")),
			want:        map[string]bool{"call-1": true},
		},
		{
			// A plain follow-up message must stay a fresh turn even
			// though history holds an answered long-running call.
			// Deciding resume from history instead of this turn is the
			// bug this helper exists to prevent.
			name:        "fresh text turn after a completed resume",
			sess:        sessionOf(interruptEvent("call-1"), answerEvent("call-1")),
			userContent: userTurn(&genai.Part{Text: "and now do something else"}),
			want:        nil,
		},
		{
			// Same, but the fresh turn carries a FunctionResponse for a
			// call that was never long-running: still not a resume.
			name:        "response to a call that was never long-running",
			sess:        sessionOf(interruptEvent("call-1")),
			userContent: userTurn(responsePart("other-call")),
			want:        map[string]bool{},
		},
		{
			name:        "no long-running call in history at all",
			sess:        sessionOf(answerEvent("call-1")),
			userContent: userTurn(responsePart("call-1")),
			want:        map[string]bool{},
		},
		{
			name:        "empty response ID",
			sess:        sessionOf(interruptEvent("call-1")),
			userContent: userTurn(responsePart("")),
			want:        map[string]bool{},
		},
		{
			// An earlier turn already answered call-1; answering it
			// again still counts, so the sub-agent continues from
			// history rather than restarting.
			name:        "already-answered ID answered again",
			sess:        sessionOf(interruptEvent("call-1"), answerEvent("call-1")),
			userContent: userTurn(responsePart("call-1")),
			want:        map[string]bool{"call-1": true},
		},
		{
			name:        "duplicate IDs collapse",
			sess:        sessionOf(interruptEvent("call-1")),
			userContent: userTurn(responsePart("call-1"), responsePart("call-1")),
			want:        map[string]bool{"call-1": true},
		},
		{
			name:        "only the long-running subset is reported",
			sess:        sessionOf(interruptEvent("call-1"), interruptEvent("call-2")),
			userContent: userTurn(responsePart("call-1"), responsePart("call-3")),
			want:        map[string]bool{"call-1": true},
		},
		{
			name:        "parallel interrupts answered together",
			sess:        sessionOf(interruptEvent("call-1", "call-2")),
			userContent: userTurn(responsePart("call-1"), responsePart("call-2")),
			want:        map[string]bool{"call-1": true, "call-2": true},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := answeredInterrupts(tc.sess, tc.userContent)
			if diff := cmp.Diff(tc.want, got); diff != "" {
				t.Errorf("answeredInterrupts() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestResumeUserContent(t *testing.T) {
	tests := []struct {
		name        string
		userContent *genai.Content
		resolved    map[string]bool
		want        *genai.Content
	}{
		{
			// The original question is still on the turn; forwarding it
			// would make the sub-agent re-run from START.
			name:        "drops text parts",
			userContent: userTurn(&genai.Part{Text: "ask the subagent"}, responsePart("call-1")),
			resolved:    map[string]bool{"call-1": true},
			want:        userTurn(responsePart("call-1")),
		},
		{
			name:        "drops FunctionResponses outside resolved",
			userContent: userTurn(responsePart("call-1"), responsePart("call-2")),
			resolved:    map[string]bool{"call-1": true},
			want:        userTurn(responsePart("call-1")),
		},
		{
			name:        "drops FunctionCall parts",
			userContent: userTurn(callParts("call-1")[0], responsePart("call-1")),
			resolved:    map[string]bool{"call-1": true},
			want:        userTurn(responsePart("call-1")),
		},
		{
			name:        "keeps every resolved response",
			userContent: userTurn(responsePart("call-1"), responsePart("call-2")),
			resolved:    map[string]bool{"call-1": true, "call-2": true},
			want:        userTurn(responsePart("call-1"), responsePart("call-2")),
		},
		{
			name:        "preserves role",
			userContent: &genai.Content{Role: "custom", Parts: []*genai.Part{responsePart("call-1")}},
			resolved:    map[string]bool{"call-1": true},
			want:        &genai.Content{Role: "custom", Parts: []*genai.Part{responsePart("call-1")}},
		},
		{
			name:        "nil user content",
			userContent: nil,
			resolved:    map[string]bool{"call-1": true},
			want:        nil,
		},
		{
			// Production callers always pass a non-empty resolved; an
			// empty one must still filter everything out rather than
			// waving the whole turn through.
			name:        "empty resolved forwards nothing",
			userContent: userTurn(responsePart("call-1")),
			resolved:    nil,
			want:        nil,
		},
		{
			name:        "nothing matches",
			userContent: userTurn(&genai.Part{Text: "hello"}, responsePart("call-2")),
			resolved:    map[string]bool{"call-1": true},
			want:        nil,
		},
		{
			name:        "skips nil parts",
			userContent: userTurn(nil, responsePart("call-1")),
			resolved:    map[string]bool{"call-1": true},
			want:        userTurn(responsePart("call-1")),
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := resumeUserContent(tc.userContent, tc.resolved)
			if diff := cmp.Diff(tc.want, got); diff != "" {
				t.Errorf("resumeUserContent() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}
