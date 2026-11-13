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

package abortevents_test

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/internal/abortevents"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/platform"
	"google.golang.org/adk/v2/session"
)

var fixedTime = time.Date(2026, 11, 13, 6, 0, 0, 0, time.UTC)

// deterministicCtx makes the IDs "ev-1", "ev-2", ... and the timestamp
// fixedTime, so expected events can be written as literals.
func deterministicCtx(t *testing.T) context.Context {
	t.Helper()
	n := 0
	ctx := platform.WithUUIDProvider(t.Context(), func() string {
		n++
		return fmt.Sprintf("ev-%d", n)
	})
	return platform.WithTimeProvider(ctx, func() time.Time { return fixedTime })
}

func callEvent(inv, author, branch, scope string, calls ...*genai.FunctionCall) *session.Event {
	parts := make([]*genai.Part, 0, len(calls))
	for _, fc := range calls {
		parts = append(parts, &genai.Part{FunctionCall: fc})
	}
	return &session.Event{
		InvocationID:   inv,
		Author:         author,
		Branch:         branch,
		IsolationScope: scope,
		LLMResponse:    model.LLMResponse{Content: &genai.Content{Role: genai.RoleModel, Parts: parts}},
	}
}

func responseEvent(inv, author string, frs ...*genai.FunctionResponse) *session.Event {
	parts := make([]*genai.Part, 0, len(frs))
	for _, fr := range frs {
		parts = append(parts, &genai.Part{FunctionResponse: fr})
	}
	return &session.Event{
		InvocationID: inv,
		Author:       author,
		LLMResponse:  model.LLMResponse{Content: &genai.Content{Role: genai.RoleUser, Parts: parts}},
	}
}

// sealed is the event BuildAbortEvents is expected to return for the given
// group: its content answers each (id, name) pair in order.
func sealed(id, author, branch, scope string, calls ...[2]string) *session.Event {
	ev := &session.Event{
		ID:             id,
		Timestamp:      fixedTime,
		InvocationID:   "inv_1",
		Author:         author,
		Branch:         branch,
		IsolationScope: scope,
		Actions:        session.EventActions{StateDelta: map[string]any{}, ArtifactDelta: map[string]int64{}},
		LLMResponse: model.LLMResponse{
			ErrorCode:    "INVOCATION_ABORTED",
			ErrorMessage: "Invocation was aborted by client.",
		},
	}
	if len(calls) == 0 {
		return ev
	}
	content := &genai.Content{Role: "user"}
	for _, c := range calls {
		content.Parts = append(content.Parts, &genai.Part{FunctionResponse: &genai.FunctionResponse{
			ID:       c[0],
			Name:     c[1],
			Response: map[string]any{"error": "Invocation was aborted by client."},
		}})
	}
	ev.Content = content
	return ev
}

func TestBuildAbortEvents_DanglingAnsweredIdLessAndOtherInvocation(t *testing.T) {
	lr := callEvent("inv_1", "agent", "", "",
		&genai.FunctionCall{ID: "call_1", Name: "tool_1"},
		&genai.FunctionCall{ID: "call_2", Name: "tool_2"},
		&genai.FunctionCall{Name: "tool_no_id"},
		&genai.FunctionCall{ID: "call_lr", Name: "tool_lr"},
	)
	lr.LongRunningToolIDs = []string{"call_lr"}
	events := []*session.Event{
		callEvent("inv_0", "agent", "", "", &genai.FunctionCall{ID: "old_call", Name: "tool_old"}),
		lr,
		responseEvent("inv_1", "agent", &genai.FunctionResponse{ID: "call_1", Name: "tool_1", Response: map[string]any{"ok": true}}),
	}

	got := abortevents.BuildAbortEvents(deterministicCtx(t), events, "inv_1", "root_agent", "")

	want := []*session.Event{
		sealed("ev-1", "agent", "", "", [2]string{"call_2", "tool_2"}, [2]string{"call_lr", "tool_lr"}),
	}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("BuildAbortEvents() mismatch (-want +got):\n%s", diff)
	}
}

func TestBuildAbortEvents_GroupingAndBranchFallback(t *testing.T) {
	events := []*session.Event{
		callEvent("inv_1", "researcher", "root.researcher", "", &genai.FunctionCall{ID: "call_a", Name: "t_a"}),
		callEvent("inv_1", "researcher", "root.researcher", "", &genai.FunctionCall{ID: "call_b", Name: "t_b"}),
		callEvent("inv_1", "coder", "", "", &genai.FunctionCall{ID: "call_c", Name: "t_c"}),
		callEvent("inv_1", "coder", "root.coder", "task_1", &genai.FunctionCall{ID: "call_d", Name: "t_d"}),
	}

	got := abortevents.BuildAbortEvents(deterministicCtx(t), events, "inv_1", "root_agent", "inv_branch")

	want := []*session.Event{
		sealed("ev-1", "researcher", "root.researcher", "", [2]string{"call_a", "t_a"}, [2]string{"call_b", "t_b"}),
		sealed("ev-2", "coder", "inv_branch", "", [2]string{"call_c", "t_c"}),
		sealed("ev-3", "coder", "root.coder", "task_1", [2]string{"call_d", "t_d"}),
	}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("BuildAbortEvents() mismatch (-want +got):\n%s", diff)
	}
}

func TestBuildAbortEvents_NothingDanglingAndPausedTaskReply(t *testing.T) {
	coordinatorCall := callEvent("inv_1", "coordinator", "", "", &genai.FunctionCall{ID: "call_1", Name: "task_worker"})
	tests := []struct {
		name          string
		events        []*session.Event
		defaultBranch string
		want          []*session.Event
	}{
		{
			name: "nothing_dangling",
			events: []*session.Event{
				callEvent("inv_1", "agent", "", "", &genai.FunctionCall{ID: "call_1", Name: "tool_1"}),
				responseEvent("inv_1", "agent", &genai.FunctionResponse{ID: "call_1", Name: "tool_1"}),
			},
			defaultBranch: "sub_branch",
			want:          []*session.Event{sealed("ev-1", "root_agent", "sub_branch", "")},
		},
		{
			name:          "no_events",
			events:        nil,
			defaultBranch: "",
			want:          []*session.Event{sealed("ev-1", "root_agent", "", "")},
		},
		{
			name: "paused_task_text_reply",
			events: []*session.Event{
				coordinatorCall,
				{
					InvocationID:   "inv_1",
					Author:         "task_worker",
					IsolationScope: "call_1",
					LLMResponse:    model.LLMResponse{Content: genai.NewContentFromText("Which city?", genai.RoleModel)},
				},
			},
			want: nil,
		},
		{
			name: "user_reply_in_task_scope_is_not_a_paused_reply",
			events: []*session.Event{
				coordinatorCall,
				{
					InvocationID:   "inv_1",
					Author:         "user",
					IsolationScope: "call_1",
					LLMResponse:    model.LLMResponse{Content: genai.NewContentFromText("Paris", genai.RoleUser)},
				},
			},
			want: []*session.Event{sealed("ev-1", "coordinator", "", "", [2]string{"call_1", "task_worker"})},
		},
		{
			name: "final_reply_outside_a_task_scope_is_not_a_paused_reply",
			events: []*session.Event{
				coordinatorCall,
				{
					InvocationID: "inv_1",
					Author:       "coordinator",
					LLMResponse:  model.LLMResponse{Content: genai.NewContentFromText("Working on it", genai.RoleModel)},
				},
			},
			want: []*session.Event{sealed("ev-1", "coordinator", "", "", [2]string{"call_1", "task_worker"})},
		},
		{
			name: "task_worker_dangling_call_not_final",
			events: []*session.Event{
				coordinatorCall,
				callEvent("inv_1", "task_worker", "", "call_1", &genai.FunctionCall{ID: "tool_1", Name: "search"}),
			},
			want: []*session.Event{
				sealed("ev-1", "coordinator", "", "", [2]string{"call_1", "task_worker"}),
				sealed("ev-2", "task_worker", "", "call_1", [2]string{"tool_1", "search"}),
			},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := abortevents.BuildAbortEvents(deterministicCtx(t), tc.events, "inv_1", "root_agent", tc.defaultBranch)
			if diff := cmp.Diff(tc.want, got); diff != "" {
				t.Errorf("BuildAbortEvents() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestIsAbortEvent(t *testing.T) {
	tests := []struct {
		name string
		ev   *session.Event
		want bool
	}{
		{name: "nil", ev: nil, want: false},
		{name: "abort_code", ev: &session.Event{LLMResponse: model.LLMResponse{ErrorCode: "INVOCATION_ABORTED"}}, want: true},
		{name: "abort_message_other_code", ev: &session.Event{LLMResponse: model.LLMResponse{ErrorCode: "QUOTA", ErrorMessage: "Invocation was aborted by client."}}, want: false},
		{name: "no_error", ev: &session.Event{}, want: false},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := abortevents.IsAbortEvent(tc.ev); got != tc.want {
				t.Errorf("IsAbortEvent() = %v, want %v", got, tc.want)
			}
		})
	}
}
