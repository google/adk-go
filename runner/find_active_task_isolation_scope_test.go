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
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/internal/abortevents"
	"google.golang.org/adk/v2/internal/workflowinternal"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
)

// TestFindActiveTaskIsolationScope walks session events backwards and
// returns the most-recent non-empty IsolationScope not closed by a
// successful finish_task FR.
func TestFindActiveTaskIsolationScope(t *testing.T) {
	t.Parallel()

	const (
		scopeA = "adk-task-A"
		scopeB = "adk-task-B"
	)

	// modelEventWithFC opens a task scope: a model-role event in
	// `scope` carrying a single FC.
	modelEventWithFC := func(scope, name, id string) *session.Event {
		return &session.Event{
			Author:         "task_agent",
			IsolationScope: scope,
			LLMResponse: model.LLMResponse{
				Content: &genai.Content{
					Role:  "model",
					Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{Name: name, ID: id}}},
				},
			},
		}
	}

	userEventWithFR := func(scope, name, id string, resp map[string]any) *session.Event {
		return &session.Event{
			Author:         "user",
			IsolationScope: scope,
			LLMResponse: model.LLMResponse{
				Content: &genai.Content{
					Role: "user",
					Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{
						Name:     name,
						ID:       id,
						Response: resp,
					}}},
				},
			},
		}
	}
	finishSuccessFR := func(scope, id string) *session.Event {
		return userEventWithFR(scope, workflowinternal.FinishTaskToolName, id,
			map[string]any{"result": workflowinternal.FinishTaskSuccessResult})
	}
	finishErrorFR := func(scope, id string) *session.Event {
		return userEventWithFR(scope, workflowinternal.FinishTaskToolName, id,
			map[string]any{"error": "validation failed: ..."})
	}

	tests := []struct {
		name   string
		events []*session.Event
		want   string
	}{
		{
			name:   "empty_session",
			events: nil,
			want:   "",
		},
		{
			// Pre-delegation: only unscoped events.
			name: "no_scoped_events",
			events: []*session.Event{
				{Author: "user"},
				{Author: "coordinator"},
			},
			want: "",
		},
		{
			// Happy path: one open scope, not yet closed.
			name: "single_open_scope",
			events: []*session.Event{
				{Author: "user"},
				modelEventWithFC(scopeA, "confirmation", "fc-A"),
			},
			want: scopeA,
		},
		{
			// Successful finish_task FR closes the scope.
			name: "scope_closed_by_finish_task_success",
			events: []*session.Event{
				modelEventWithFC(scopeA, "finish_task", "ft-A"),
				finishSuccessFR(scopeA, "ft-A"),
			},
			want: "",
		},
		{
			// Error FR does NOT close the scope (task retries).
			name: "error_finish_task_fr_keeps_scope_open",
			events: []*session.Event{
				modelEventWithFC(scopeA, "finish_task", "ft-A"),
				finishErrorFR(scopeA, "ft-A"),
			},
			want: scopeA,
		},
		{
			// Newer open, older closed: backward walk hits the
			// newer one first and returns it.
			name: "newer_open_older_closed",
			events: []*session.Event{
				modelEventWithFC(scopeA, "finish_task", "ft-A"),
				finishSuccessFR(scopeA, "ft-A"),
				modelEventWithFC(scopeB, "confirmation", "fc-B"),
			},
			want: scopeB,
		},
		{
			// Newer closed, older open: helper must skip the
			// closed newer one — proves it's "most-recent
			// unclosed", not just "most-recent".
			name: "newer_closed_older_open",
			events: []*session.Event{
				modelEventWithFC(scopeA, "confirmation", "fc-A"),
				modelEventWithFC(scopeB, "finish_task", "ft-B"),
				finishSuccessFR(scopeB, "ft-B"),
			},
			want: scopeA,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			sess := buildSessionWithEvents(t, tc.events)
			if got, _ := findActiveTaskIsolationScope(sess); got != tc.want {
				t.Errorf("findActiveTaskIsolationScope = %q, want %q", got, tc.want)
			}
		})
	}
}

func TestFindActiveTaskIsolationScope_NilSession(t *testing.T) {
	t.Parallel()
	if got, gotInv := findActiveTaskIsolationScope(nil); got != "" || gotInv != "" {
		t.Errorf("findActiveTaskIsolationScope(nil) = (%q, %q), want (\"\", \"\")", got, gotInv)
	}
}

// TestFindActiveTaskIsolationScope_TwoPassErrorResultInvocationIDAndRewinds
// pins the rules of adk-python's _find_active_task_scope (runners.py) beyond
// the scope itself: a finished scope stays closed whatever follows it, a
// failed task closes its scope, the invocation of the scoped event is
// returned, and rewound events do not count.
func TestFindActiveTaskIsolationScope_TwoPassErrorResultInvocationIDAndRewinds(t *testing.T) {
	t.Parallel()

	scoped := func(author, scope, invocationID string, part *genai.Part) *session.Event {
		return &session.Event{
			Author:         author,
			InvocationID:   invocationID,
			IsolationScope: scope,
			LLMResponse:    model.LLMResponse{Content: &genai.Content{Role: "model", Parts: []*genai.Part{part}}},
		}
	}
	text := func(author, scope, invocationID, s string) *session.Event {
		return scoped(author, scope, invocationID, &genai.Part{Text: s})
	}
	finish := func(scope, invocationID, result string) *session.Event {
		return scoped("user", scope, invocationID, &genai.Part{FunctionResponse: &genai.FunctionResponse{
			Name:     workflowinternal.FinishTaskToolName,
			ID:       "ft-1",
			Response: map[string]any{"result": result},
		}})
	}
	rewind := func(invocationID, before string) *session.Event {
		ev := &session.Event{Author: "user", InvocationID: invocationID}
		ev.Actions.RewindBeforeInvocationID = before
		return ev
	}

	tests := []struct {
		name      string
		events    []*session.Event
		wantScope string
		wantInv   string
	}{
		{
			name: "event_after_finish_does_not_reopen_scope",
			events: []*session.Event{
				text("task_agent", "sA", "inv-1", "question"),
				finish("sA", "inv-1", "Task completed."),
				text("task_agent", "sA", "inv-1", "post-finish status"),
			},
		},
		{
			name: "failed_task_closes_scope",
			events: []*session.Event{
				text("task_agent", "sA", "inv-1", "question"),
				finish("sA", "inv-1", "Task failed."),
			},
		},
		{
			name: "invocation_of_newest_scoped_event",
			events: []*session.Event{
				text("task_agent", "sA", "inv-1", "q1"),
				text("user", "sA", "inv-2", "a1"),
			},
			wantScope: "sA",
			wantInv:   "inv-2",
		},
		{
			name: "rewound_finish_reopens_scope",
			events: []*session.Event{
				text("task_agent", "sA", "inv-1", "q1"),
				finish("sA", "inv-2", "Task completed."),
				rewind("inv-3", "inv-2"),
			},
			wantScope: "sA",
			wantInv:   "inv-1",
		},
		{
			// A model that calls finish_task twice in one turn gets one
			// event with both responses. As in adk-python, the first
			// decides, and an error leaves the scope open.
			name: "first_finish_task_response_decides",
			events: []*session.Event{
				text("task_agent", "sA", "inv-1", "q1"),
				{
					Author:         "user",
					InvocationID:   "inv-1",
					IsolationScope: "sA",
					LLMResponse: model.LLMResponse{Content: &genai.Content{Role: "user", Parts: []*genai.Part{
						{FunctionResponse: &genai.FunctionResponse{
							Name: workflowinternal.FinishTaskToolName, ID: "ft-1", Response: map[string]any{"error": "missing result"},
						}},
						{FunctionResponse: &genai.FunctionResponse{
							Name: workflowinternal.FinishTaskToolName, ID: "ft-2", Response: map[string]any{"result": "Task completed."},
						}},
					}}},
				},
			},
			wantScope: "sA",
			wantInv:   "inv-1",
		},
		{
			name: "rewound_scope_is_not_open",
			events: []*session.Event{
				text("task_agent", "sA", "inv-1", "q1"),
				rewind("inv-2", "inv-1"),
			},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			sess := buildSessionWithEvents(t, tc.events)
			gotScope, gotInv := findActiveTaskIsolationScope(sess)
			if gotScope != tc.wantScope || gotInv != tc.wantInv {
				t.Errorf("findActiveTaskIsolationScope = (%q, %q), want (%q, %q)", gotScope, gotInv, tc.wantScope, tc.wantInv)
			}
		})
	}
}

// TestFindActiveTaskIsolationScope_AbortedInvocations pins the abort rule of
// adk-python's _find_active_task_scope (runners.py): a scoped event whose
// invocation has an INVOCATION_ABORTED event anywhere in the live events does
// not keep its scope open, while scopes of other invocations, and finish_task
// responses inside an aborted invocation, count as before.
func TestFindActiveTaskIsolationScope_AbortedInvocations(t *testing.T) {
	t.Parallel()

	event := func(author, scope, invocationID string, part *genai.Part) *session.Event {
		ev := &session.Event{Author: author, InvocationID: invocationID, IsolationScope: scope}
		if part != nil {
			ev.Content = &genai.Content{Role: "model", Parts: []*genai.Part{part}}
		}
		return ev
	}
	text := func(author, scope, invocationID, s string) *session.Event {
		return event(author, scope, invocationID, &genai.Part{Text: s})
	}
	call := func(author, scope, invocationID, name, id string) *session.Event {
		return event(author, scope, invocationID, &genai.Part{FunctionCall: &genai.FunctionCall{Name: name, ID: id}})
	}
	finishSuccess := func(scope, invocationID string) *session.Event {
		return event("user", scope, invocationID, &genai.Part{FunctionResponse: &genai.FunctionResponse{
			Name:     workflowinternal.FinishTaskToolName,
			ID:       "ft-A",
			Response: map[string]any{"result": workflowinternal.FinishTaskSuccessResult},
		}})
	}
	// seal is an abort event answering the call name/id, or with no content
	// when name is empty.
	seal := func(author, scope, invocationID, name, id string) *session.Event {
		var part *genai.Part
		if name != "" {
			part = &genai.Part{FunctionResponse: &genai.FunctionResponse{
				Name:     name,
				ID:       id,
				Response: map[string]any{"error": abortevents.Message},
			}}
		}
		ev := event(author, scope, invocationID, part)
		ev.ErrorCode = abortevents.ErrorCode
		ev.ErrorMessage = abortevents.Message
		return ev
	}

	tests := []struct {
		name      string
		events    []*session.Event
		wantScope string
		wantInv   string
	}{
		{
			name: "aborted_single_turn_task_returns_empty",
			events: []*session.Event{
				call("coordinator", "", "inv-1", "task_worker", "adk-task-A"),
				call("task_worker", "adk-task-A", "inv-1", "_slow_tool", "fc-slow"),
				seal("coordinator", "", "inv-1", "task_worker", "adk-task-A"),
				seal("task_worker", "adk-task-A", "inv-1", "_slow_tool", "fc-slow"),
			},
		},
		{
			// A turn that joined the paused task under its invocation ID
			// and was aborted closes the earlier paused reply too.
			name: "aborted_joined_turn_borrowing_invocation_id_closes_earlier_paused_reply",
			events: []*session.Event{
				call("coordinator", "", "inv-1", "task_worker", "adk-task-A"),
				text("task_worker", "adk-task-A", "inv-1", "Which city?"),
				text("user", "adk-task-A", "inv-1", "Paris"),
				call("task_worker", "adk-task-A", "inv-1", "_slow_tool", "fc-slow"),
				seal("coordinator", "", "inv-1", "task_worker", "adk-task-A"),
				seal("task_worker", "adk-task-A", "inv-1", "_slow_tool", "fc-slow"),
			},
		},
		{
			// The rule is per invocation: an aborted turn under a new
			// invocation ID leaves the earlier paused reply open.
			name: "aborted_second_turn_with_distinct_invocation_id_keeps_turn1_scope_open",
			events: []*session.Event{
				call("coordinator", "", "inv-1", "task_worker", "adk-task-A"),
				text("task_worker", "adk-task-A", "inv-1", "Which city?"),
				text("user", "adk-task-A", "inv-2", "Paris"),
				call("task_worker", "adk-task-A", "inv-2", "_slow_tool", "fc-slow"),
				seal("task_worker", "adk-task-A", "inv-2", "_slow_tool", "fc-slow"),
			},
			wantScope: "adk-task-A",
			wantInv:   "inv-1",
		},
		{
			name: "newer_aborted_scope_falls_back_to_older_open_scope",
			events: []*session.Event{
				text("task_worker", "adk-task-A", "inv-1", "Paused A"),
				call("task_worker", "adk-task-B", "inv-2", "_slow_tool", "fc-B"),
				seal("task_worker", "adk-task-B", "inv-2", "_slow_tool", "fc-B"),
			},
			wantScope: "adk-task-A",
			wantInv:   "inv-1",
		},
		{
			// The abort event itself has no scope.
			name: "newer_aborted_and_older_finished_returns_empty",
			events: []*session.Event{
				call("task_worker", "adk-task-A", "inv-1", workflowinternal.FinishTaskToolName, "ft-A"),
				finishSuccess("adk-task-A", "inv-1"),
				call("task_worker", "adk-task-B", "inv-2", "_slow_tool", "fc-B"),
				seal("coordinator", "", "inv-2", "", ""),
			},
		},
		{
			// A resumed invocation appends events after its seal.
			name: "abort_event_preceding_later_scoped_event_of_same_invocation_still_closes_scope",
			events: []*session.Event{
				seal("coordinator", "", "inv-1", "", ""),
				call("task_worker", "adk-task-A", "inv-1", "tool_after_resume", "fc-2"),
			},
		},
		{
			name: "finish_task_in_aborted_invocation_still_marks_scope_finished_against_earlier_clean_invocation",
			events: []*session.Event{
				text("task_worker", "adk-task-A", "inv-1", "Paused A"),
				finishSuccess("adk-task-A", "inv-2"),
				seal("coordinator", "", "inv-2", "", ""),
			},
		},
		{
			name: "other_error_code_does_not_close_scope",
			events: []*session.Event{
				text("task_worker", "adk-task-A", "inv-1", "Which city?"),
				{InvocationID: "inv-1", Author: "coordinator", LLMResponse: model.LLMResponse{ErrorCode: "OTHER_ERROR", ErrorMessage: "boom"}},
			},
			wantScope: "adk-task-A",
			wantInv:   "inv-1",
		},
		{
			name: "two_aborted_invocations_close_both_scopes",
			events: []*session.Event{
				text("task_worker", "adk-task-A", "inv-1", "Paused A"),
				seal("coordinator", "", "inv-1", "", ""),
				text("task_worker", "adk-task-B", "inv-2", "Paused B"),
				seal("coordinator", "", "inv-2", "", ""),
			},
		},
		{
			// inv-1 was resumed after inv-2 and aborted; rewinding before
			// inv-2 drops that seal but keeps inv-1's paused reply.
			name: "rewound_abort_event_does_not_close_scope",
			events: []*session.Event{
				text("task_worker", "adk-task-A", "inv-1", "Paused A"),
				text("user", "", "inv-2", "Hello"),
				seal("coordinator", "", "inv-1", "", ""),
				{InvocationID: "inv-3", Author: "user", Actions: session.EventActions{RewindBeforeInvocationID: "inv-2"}},
			},
			wantScope: "adk-task-A",
			wantInv:   "inv-1",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			sess := buildSessionWithEvents(t, tc.events)
			gotScope, gotInv := findActiveTaskIsolationScope(sess)
			if gotScope != tc.wantScope || gotInv != tc.wantInv {
				t.Errorf("findActiveTaskIsolationScope = (%q, %q), want (%q, %q)", gotScope, gotInv, tc.wantScope, tc.wantInv)
			}
		})
	}
}

// buildSessionWithEvents creates an in-memory session and appends
// each event in order, preserving IsolationScope.
func buildSessionWithEvents(t *testing.T, events []*session.Event) session.Session {
	t.Helper()
	ctx := t.Context()
	svc := session.InMemoryService()
	resp, err := svc.Create(ctx, &session.CreateRequest{
		AppName: "test-app",
		UserID:  "test-user",
	})
	if err != nil {
		t.Fatalf("create session: %v", err)
	}
	for i, ev := range events {
		if err := svc.AppendEvent(ctx, resp.Session, ev); err != nil {
			t.Fatalf("AppendEvent[%d]: %v", i, err)
		}
	}
	return resp.Session
}
