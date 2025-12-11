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

package remoteagent

import (
	"fmt"
	"testing"

	"github.com/a2aproject/a2a-go/a2a"
	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	icontext "google.golang.org/adk/internal/context"
	"google.golang.org/adk/model"
	"google.golang.org/adk/server/adka2a"
	"google.golang.org/adk/session"
)

func newTestInvocationContext(t *testing.T, agentName string, events ...*session.Event) agent.InvocationContext {
	t.Helper()
	ctx := t.Context()
	store := session.InMemoryService()
	resp, err := store.Create(ctx, &session.CreateRequest{AppName: "test", UserID: "test-user"})
	if err != nil {
		t.Errorf("store.Create() error = %v", err)
		return nil
	}
	for _, event := range events {
		if err := store.AppendEvent(ctx, resp.Session, event); err != nil {
			t.Errorf("store.AppendEvent() error = %v", err)
			return nil
		}
	}
	agent, err := agent.New(agent.Config{Name: agentName})
	if err != nil {
		t.Errorf("agent.New() error = %v", err)
		return nil
	}
	return icontext.NewInvocationContext(ctx, icontext.InvocationContextParams{
		Agent:   agent,
		Session: resp.Session,
	})
}

func newEventFromParts(author string, parts ...*genai.Part) *session.Event {
	var role genai.Role = genai.RoleModel
	if author == "user" {
		role = genai.RoleUser
	}
	event := &session.Event{Author: author}
	if len(parts) > 0 {
		event.Content = genai.NewContentFromParts(parts, role)
	}
	return event
}

func TestGetUserFunctionCallAt(t *testing.T) {
	testCases := []struct {
		name    string
		events  []*session.Event
		atIndex int
		success bool
	}{
		{
			name: "success",
			events: []*session.Event{
				newEventFromParts(genai.RoleModel, &genai.Part{FunctionCall: &genai.FunctionCall{ID: "id-1"}}),
				newEventFromParts(genai.RoleUser, &genai.Part{FunctionResponse: &genai.FunctionResponse{ID: "id-1"}}),
			},
			atIndex: 1,
			success: true,
		},
		{
			name: "success with event in-between",
			events: []*session.Event{
				newEventFromParts(genai.RoleModel, &genai.Part{FunctionCall: &genai.FunctionCall{ID: "id-1"}}),
				newEventFromParts(genai.RoleModel, &genai.Part{Text: "another event"}),
				newEventFromParts(genai.RoleUser, &genai.Part{FunctionResponse: &genai.FunctionResponse{ID: "id-1"}}),
			},
			atIndex: 2,
			success: true,
		},
		{
			name: "success with multiple parts in-between",
			events: []*session.Event{
				newEventFromParts(genai.RoleModel,
					&genai.Part{Text: "calling"},
					&genai.Part{FunctionCall: &genai.FunctionCall{ID: "id-1"}},
					&genai.Part{Text: "called"},
				),
				newEventFromParts(genai.RoleUser,
					&genai.Part{Text: "responding"},
					&genai.Part{FunctionResponse: &genai.FunctionResponse{ID: "id-1"}},
					&genai.Part{Text: "responded"},
				),
			},
			atIndex: 1,
			success: true,
		},
		{
			name: "failf if not response index",
			events: []*session.Event{
				newEventFromParts(genai.RoleModel, &genai.Part{FunctionCall: &genai.FunctionCall{ID: "id-1"}}),
				newEventFromParts(genai.RoleModel, &genai.Part{Text: "another event"}),
				newEventFromParts(genai.RoleUser, &genai.Part{FunctionResponse: &genai.FunctionResponse{ID: "id-1"}}),
			},
			atIndex: 1,
			success: false,
		},
		{
			name: "fail if not user author",
			events: []*session.Event{
				newEventFromParts(genai.RoleModel, &genai.Part{FunctionCall: &genai.FunctionCall{ID: "id-1"}}),
				newEventFromParts(genai.RoleModel, &genai.Part{FunctionResponse: &genai.FunctionResponse{ID: "id-1"}}),
			},
			success: false,
		},
		{
			name: "fail if no matching function call",
			events: []*session.Event{
				newEventFromParts(genai.RoleModel, &genai.Part{FunctionCall: &genai.FunctionCall{ID: "id-2"}}),
				newEventFromParts(genai.RoleUser, &genai.Part{FunctionResponse: &genai.FunctionResponse{ID: "id-1"}}),
			},
			success: false,
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			ictx := newTestInvocationContext(t, "test-agent", tc.events...)
			got := getUserFunctionCallAt(ictx.Session().Events(), tc.atIndex)
			if !tc.success && got != nil {
				t.Errorf("getUserFunctionCallAt() = %v, want nil", got)
			}
			if tc.success && got == nil {
				t.Error("getUserFunctionCallAt() = nil, want non-nil")
			}
		})
	}
}

func TestToMissingRemoteSessionParts(t *testing.T) {
	remoteName := "remote-agent"
	testCases := []struct {
		name          string
		events        []*session.Event
		wantParts     []a2a.Part
		wantContextID string
	}{
		{
			name: "all message parts collected",
			events: []*session.Event{
				newEventFromParts("user", &genai.Part{Text: "hello"}),
				newEventFromParts("user", &genai.Part{Text: "foo"}, &genai.Part{Text: "bar"}),
			},
			wantParts: []a2a.Part{
				a2a.TextPart{Text: "hello"},
				a2a.TextPart{Text: "foo"},
				a2a.TextPart{Text: "bar"},
			},
		},
		{
			name: "other agent messages are rephrased",
			events: []*session.Event{
				newEventFromParts("another-agent", &genai.Part{Text: "foo"}),
				newEventFromParts("user", &genai.Part{Text: "bar"}),
			},
			wantParts: []a2a.Part{
				a2a.TextPart{Text: "For context:"},
				a2a.TextPart{Text: "[another-agent] said: foo"},
				a2a.TextPart{Text: "bar"},
			},
		},
		{
			name: "other agent thoughts are skipped",
			events: []*session.Event{
				newEventFromParts("another-agent", &genai.Part{Text: "foo", Thought: true}),
				newEventFromParts("user", &genai.Part{Text: "bar"}),
			},
			wantParts: []a2a.Part{
				a2a.TextPart{Text: "bar"},
			},
		},
		{
			name: "events before the last remote response excluded",
			events: []*session.Event{
				newEventFromParts("user", &genai.Part{Text: "hello"}),
				newEventFromParts(remoteName, &genai.Part{Text: "hi"}),
				newEventFromParts("user", &genai.Part{Text: "foo"}),
				newEventFromParts("user", &genai.Part{Text: "bar"}),
			},
			wantParts: []a2a.Part{
				a2a.TextPart{Text: "foo"},
				a2a.TextPart{Text: "bar"},
			},
		},
		{
			name: "contextID of the last remote agent response returned",
			events: []*session.Event{
				{
					Author: remoteName,
					LLMResponse: model.LLMResponse{
						Content:        genai.NewContentFromParts([]*genai.Part{{Text: "hi"}}, genai.RoleModel),
						CustomMetadata: adka2a.ToCustomMetadata(a2a.NewTaskID(), "ctxID-123"),
					},
				},
			},
			wantParts:     []a2a.Part{},
			wantContextID: "ctxID-123",
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			ictx := newTestInvocationContext(t, remoteName, tc.events...)
			gotParts, gotContextID := toMissingRemoteSessionParts(ictx, ictx.Session().Events())
			if tc.wantContextID != gotContextID {
				t.Errorf("toMissingRemoteSessionParts() contextID = %s, want %s", gotContextID, tc.wantContextID)
			}
			if diff := cmp.Diff(tc.wantParts, gotParts); diff != "" {
				t.Errorf("toMissingRemoteSessionParts() wrong result (+got,-want):\ngot = %v\nwant = %v\ndiff = %v", gotParts, tc.wantParts, diff)
			}
		})
	}
}

func TestPresentAsUserMessage(t *testing.T) {
	testCases := []struct {
		name  string
		input *session.Event
		want  *session.Event
	}{
		{
			name:  "text presented",
			input: newEventFromParts("some agent", genai.NewPartFromText("hello")),
			want: newEventFromParts(
				"user",
				genai.NewPartFromText("For context:"),
				genai.NewPartFromText("[some agent] said: hello"),
			),
		},
		{
			name:  "function call presented",
			input: newEventFromParts("some agent", genai.NewPartFromFunctionCall("get_weather", map[string]any{"city": "Warsaw"})),
			want: newEventFromParts(
				"user",
				genai.NewPartFromText("For context:"),
				genai.NewPartFromText(fmt.Sprintf("[some agent] called tool get_weather with parameters: %v", map[string]any{"city": "Warsaw"})),
			),
		},
		{
			name:  "function call result presented",
			input: newEventFromParts("some agent", genai.NewPartFromFunctionResponse("get_weather", map[string]any{"temp": "1C"})),
			want: newEventFromParts(
				"user",
				genai.NewPartFromText("For context:"),
				genai.NewPartFromText(fmt.Sprintf("[some agent] get_weather tool returned result: %v", map[string]any{"temp": "1C"})),
			),
		},
		{
			name: "other part types unmodified",
			input: newEventFromParts(
				"some agent",
				genai.NewPartFromFile(genai.File{Name: "cat.png"}),
				genai.NewPartFromExecutableCode("print('hello, world!')", genai.LanguagePython),
				genai.NewPartFromCodeExecutionResult(genai.OutcomeOK, "hello, world!"),
			),
			want: newEventFromParts(
				"user",
				genai.NewPartFromText("For context:"),
				genai.NewPartFromFile(genai.File{Name: "cat.png"}),
				genai.NewPartFromExecutableCode("print('hello, world!')", genai.LanguagePython),
				genai.NewPartFromCodeExecutionResult(genai.OutcomeOK, "hello, world!"),
			),
		},
		{
			name:  "thought skipped",
			input: newEventFromParts("some agent", &genai.Part{Text: "hello", Thought: true}),
			want:  newEventFromParts("user"),
		},
		{
			name:  "thought with other parts",
			input: newEventFromParts("some agent", &genai.Part{Text: "thinking...", Thought: true}, genai.NewPartFromText("done")),
			want: newEventFromParts(
				"user",
				genai.NewPartFromText("For context:"),
				genai.NewPartFromText("[some agent] said: done"),
			),
		},
	}
	ignoreFields := []cmp.Option{
		cmpopts.IgnoreFields(session.Event{}, "ID"),
		cmpopts.IgnoreFields(session.Event{}, "InvocationID"),
		cmpopts.IgnoreFields(session.Event{}, "Timestamp"),
		cmpopts.IgnoreFields(session.EventActions{}, "StateDelta"),
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			ictx := newTestInvocationContext(t, "test")
			got := presentAsUserMessage(ictx, tc.input)
			if diff := cmp.Diff(tc.want, got, ignoreFields...); diff != "" {
				t.Errorf("presentAsUserMessage() wrong result (+got,-want):\ngot = %+v\nwant = %+v\ndiff = %v", got, tc.want, diff)
			}
		})
	}
}
