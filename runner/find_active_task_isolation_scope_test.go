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
