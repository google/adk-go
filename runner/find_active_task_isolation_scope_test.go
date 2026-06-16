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
	"context"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/internal/workflowinternal"
	"google.golang.org/adk/model"
	"google.golang.org/adk/session"
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
			if got := findActiveTaskIsolationScope(sess); got != tc.want {
				t.Errorf("findActiveTaskIsolationScope = %q, want %q", got, tc.want)
			}
		})
	}
}

func TestFindActiveTaskIsolationScope_NilSession(t *testing.T) {
	t.Parallel()
	if got := findActiveTaskIsolationScope(nil); got != "" {
		t.Errorf("findActiveTaskIsolationScope(nil) = %q, want \"\"", got)
	}
}

// buildSessionWithEvents creates an in-memory session and appends
// each event in order, preserving IsolationScope.
func buildSessionWithEvents(t *testing.T, events []*session.Event) session.Session {
	t.Helper()
	ctx := context.Background()
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
