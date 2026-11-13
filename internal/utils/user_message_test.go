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

package utils_test

import (
	"iter"
	"slices"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/internal/utils"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
)

type eventList []*session.Event

func (l eventList) All() iter.Seq[*session.Event] { return slices.Values(l) }
func (l eventList) Len() int                      { return len(l) }
func (l eventList) At(i int) *session.Event       { return l[i] }

func userEvent(invocationID, author string, parts ...*genai.Part) *session.Event {
	return &session.Event{
		InvocationID: invocationID,
		Author:       author,
		LLMResponse:  model.LLMResponse{Content: &genai.Content{Role: genai.RoleUser, Parts: parts}},
	}
}

func TestFindUserMessageForInvocation(t *testing.T) {
	text := func(s string) *genai.Part { return &genai.Part{Text: s} }
	response := &genai.Part{FunctionResponse: &genai.FunctionResponse{ID: "fc-1", Name: "tool"}}
	image := &genai.Part{InlineData: &genai.Blob{MIMEType: "image/png", Data: []byte{1}}}
	rewind := &session.Event{InvocationID: "inv-2", Author: "user", Actions: session.EventActions{RewindBeforeInvocationID: "inv-1"}}

	tests := []struct {
		name         string
		events       eventList
		invocationID string
		want         *genai.Content
	}{
		{
			name:         "skips a function-response-only event",
			events:       eventList{userEvent("inv-1", "user", response), userEvent("inv-1", "user", text("original"))},
			invocationID: "inv-1",
			want:         &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{{Text: "original"}}},
		},
		{
			name:         "skips an event with text and a function response",
			events:       eventList{userEvent("inv-1", "user", text("ok"), response), userEvent("inv-1", "user", text("original"))},
			invocationID: "inv-1",
			want:         &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{{Text: "original"}}},
		},
		{
			name:         "accepts an image-only message",
			events:       eventList{userEvent("inv-1", "user", image)},
			invocationID: "inv-1",
			want:         &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{{InlineData: &genai.Blob{MIMEType: "image/png", Data: []byte{1}}}}},
		},
		{
			name:         "takes the first message",
			events:       eventList{userEvent("inv-1", "user", text("first")), userEvent("inv-1", "user", text("second"))},
			invocationID: "inv-1",
			want:         &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{{Text: "first"}}},
		},
		{
			name:         "skips other invocations",
			events:       eventList{userEvent("inv-2", "user", text("x"))},
			invocationID: "inv-1",
		},
		{
			name:         "skips events not authored by the user",
			events:       eventList{userEvent("inv-1", "root_agent", text("x"))},
			invocationID: "inv-1",
		},
		{
			name:         "skips an event with no parts",
			events:       eventList{userEvent("inv-1", "user"), userEvent("inv-1", "user", text("original"))},
			invocationID: "inv-1",
			want:         &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{{Text: "original"}}},
		},
		{
			name:         "skips rewound events",
			events:       eventList{userEvent("inv-1", "user", text("rewound")), rewind},
			invocationID: "inv-1",
		},
		{
			name:   "empty invocation ID matches any invocation",
			events: eventList{userEvent("inv-2", "user", text("x"))},
			want:   &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{{Text: "x"}}},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := utils.FindUserMessageForInvocation(tt.events, tt.invocationID)
			if diff := cmp.Diff(tt.want, got); diff != "" {
				t.Errorf("FindUserMessageForInvocation() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}
