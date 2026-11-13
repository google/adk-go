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
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/internal/llminternal"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
)

func resumeCallEvent(id, author, branch string, longRunning ...string) *session.Event {
	return &session.Event{
		ID:                 id,
		Author:             author,
		Branch:             branch,
		InvocationID:       "inv-1",
		LongRunningToolIDs: longRunning,
		LLMResponse: model.LLMResponse{Content: &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{
			{FunctionCall: &genai.FunctionCall{ID: "fc-1", Name: "tool"}},
		}}},
	}
}

func resumeAnswerEvent(id, author, branch, callID string) *session.Event {
	return &session.Event{
		ID:           id,
		Author:       author,
		Branch:       branch,
		InvocationID: "inv-1",
		LLMResponse: model.LLMResponse{Content: &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{
			{FunctionResponse: &genai.FunctionResponse{ID: callID, Name: "tool"}},
		}}},
	}
}

func TestDecideResumeAction(t *testing.T) {
	call := resumeCallEvent("call", "worker", "")
	tests := []struct {
		name      string
		answer    *session.Event
		resumable bool
		want      llminternal.ResumeRoute
	}{
		{"no answer", nil, true, llminternal.ResumeContinue},
		{"user answer, not resumable", resumeAnswerEvent("a", "user", "", "fc-1"), false, llminternal.ResumeRouteToAuthor},
		{"agent answer, resumable", resumeAnswerEvent("a", "worker", "", "fc-1"), true, llminternal.ResumeRouteToAuthor},
		{"agent answer, not resumable", resumeAnswerEvent("a", "worker", "", "fc-1"), false, llminternal.ResumeContinue},
		{"user answer on a branch the call opened", resumeAnswerEvent("a", "user", "worker@1.tool@fc-1", "fc-1"), false, llminternal.ResumeReplayCalls},
		{"agent answer on a branch the call opened", resumeAnswerEvent("a", "worker", "worker@1.tool@fc-1", "fc-1"), false, llminternal.ResumeContinue},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := llminternal.DecideResumeAction(call, tt.answer, tt.resumable); got != tt.want {
				t.Errorf("DecideResumeAction() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestBranchCarriesCall(t *testing.T) {
	calls := []*genai.FunctionCall{{ID: "fc-1"}, {ID: ""}}
	tests := []struct {
		branch string
		want   bool
	}{
		{"", false},
		{"agent@fc-1", true},
		{"parent@1.tool@fc-1.node", true},
		{"agent@fc-10", false},
		{"agent@xfc-1", false},
		{"fc-1", false},
		{"a@b@fc-1", true},
		{"agent@", false},
	}
	for _, tt := range tests {
		if got := llminternal.BranchCarriesCall(tt.branch, calls); got != tt.want {
			t.Errorf("BranchCarriesCall(%q) = %v, want %v", tt.branch, got, tt.want)
		}
	}
}

func TestInvocationEvents(t *testing.T) {
	fnResponse := func(id, branch, callID string) *session.Event {
		return resumeAnswerEvent(id, "user", branch, callID)
	}
	other := &session.Event{ID: "other-inv", Author: "a", InvocationID: "inv-0", Branch: "p.c"}
	ownCall := resumeCallEvent("own-call", "c", "p.c")
	events := []*session.Event{
		other,
		{ID: "user-root", Author: "user", InvocationID: "inv-1"},
		{ID: "agent-root", Author: "a", InvocationID: "inv-1"},
		ownCall,
		{ID: "agent-child-branch", Author: "c", InvocationID: "inv-1", Branch: "p.c.d"},
		{ID: "user-desc", Author: "user", InvocationID: "inv-1", Branch: "p.c.d"},
		{ID: "user-sibling", Author: "user", InvocationID: "inv-1", Branch: "p.cx"},
		fnResponse("fr-own", "p.c", "fc-1"),
		fnResponse("fr-foreign", "p.c", "fc-other"),
	}
	ids := func(evs []*session.Event) []string {
		var out []string
		for _, ev := range evs {
			out = append(out, ev.ID)
		}
		return out
	}
	tests := []struct {
		name             string
		branch           string
		invocation, onBr bool
		want             []string
	}{
		{"all", "p.c", false, false, []string{"other-inv", "user-root", "agent-root", "own-call", "agent-child-branch", "user-desc", "user-sibling", "fr-own", "fr-foreign"}},
		{"current invocation", "p.c", true, false, []string{"user-root", "agent-root", "own-call", "agent-child-branch", "user-desc", "user-sibling", "fr-own", "fr-foreign"}},
		{"current invocation and branch", "p.c", true, true, []string{"user-root", "own-call", "user-desc", "fr-own"}},
		{"empty context branch keeps every user event", "", true, true, []string{"user-root", "agent-root", "user-desc", "user-sibling", "fr-own", "fr-foreign"}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{
				Session:      &fakeSession{events: events},
				Branch:       tt.branch,
				InvocationID: "inv-1",
			})
			got := ids(llminternal.InvocationEvents(ctx, tt.invocation, tt.onBr))
			if diff := cmp.Diff(tt.want, got); diff != "" {
				t.Errorf("InvocationEvents() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestShouldPauseInvocation(t *testing.T) {
	longRunning := resumeCallEvent("call", "worker", "", "fc-1")
	tests := []struct {
		name   string
		ev     *session.Event
		events []*session.Event
		want   bool
	}{
		{"long-running call", longRunning, []*session.Event{longRunning}, true},
		{"long-running call not yet stored", longRunning, nil, true},
		{"ordinary call", resumeCallEvent("call", "worker", ""), nil, false},
		{"long-running ID of another call", resumeCallEvent("call", "worker", "", "fc-2"), nil, false},
		{
			"later user event on the call's sub-branch",
			longRunning,
			[]*session.Event{longRunning, {ID: "u", Author: "user", Branch: "worker@1.tool@fc-1"}},
			false,
		},
		{
			"later agent event on the call's sub-branch",
			longRunning,
			[]*session.Event{longRunning, {ID: "u", Author: "worker", Branch: "worker@1.tool@fc-1"}},
			true,
		},
		{
			"user event on the call's sub-branch before the call",
			longRunning,
			[]*session.Event{{ID: "u", Author: "user", Branch: "worker@1.tool@fc-1"}, longRunning},
			true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{
				Session:      &fakeSession{events: tt.events},
				InvocationID: "inv-1",
			})
			if got := llminternal.ShouldPauseInvocation(ctx, tt.ev); got != tt.want {
				t.Errorf("ShouldPauseInvocation() = %v, want %v", got, tt.want)
			}
		})
	}
}
