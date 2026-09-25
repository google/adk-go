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

package main

import (
	"context"
	"io"
	"net/http"
	"testing"
)

func TestAuthorizeIssue(t *testing.T) {
	// No issue bound to the context (e.g. a tool call outside an audit session).
	if _, ok := authorizeIssue(context.Background(), 7); ok {
		t.Error("authorizeIssue with no bound issue = ok, want rejected")
	}

	ctx := withAuditedIssue(context.Background(), 7)
	if _, ok := authorizeIssue(ctx, 7); !ok {
		t.Error("authorizeIssue(7) on a session scoped to 7 = rejected, want ok")
	}
	// The defense against prompt injection: a different issue must be refused.
	if msg, ok := authorizeIssue(ctx, 8); ok {
		t.Errorf("authorizeIssue(8) on a session scoped to 7 = ok, want rejected (msg=%q)", msg)
	}
}

func TestDoGetIssueStateRejectsUnauthorizedRead(t *testing.T) {
	// The read tool is authorized like the mutating tools: untrusted content must
	// not make the model pull an out-of-scope issue's data into context.
	var calls int
	c := testClient(t, baseCfg(), http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		calls++
		_, _ = io.WriteString(w, `{}`)
	}))
	ctx := withAuditedIssue(context.Background(), 7)
	st, err := c.doGetIssueState(ctx, 8)
	if err != nil {
		t.Fatalf("doGetIssueState(8) unexpected Go error = %v", err)
	}
	if st.Status == "success" {
		t.Error("doGetIssueState(8) on a session scoped to 7 = success, want error status")
	}
	if calls != 0 {
		t.Errorf("doGetIssueState(8) made %d HTTP calls for an unauthorized read, want 0", calls)
	}
}

func TestDoAddLabelRecordsToolError(t *testing.T) {
	// An infrastructure failure must be recorded so the run can exit non-zero
	// (fail loud), even though the error is also handed back to the model.
	c := testClient(t, baseCfg(), http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = io.WriteString(w, `{"message":"boom"}`)
	}))
	ctx := withAuditedIssue(context.Background(), 7)
	// Adding the clarification label is gated on the same precondition as marking
	// stale, so the tool needs an observation to get as far as the API call.
	c.recordObservation(7, IssueState{
		Status: "success", LastActionRole: string(roleMaintainer),
		LastActionType: string(eventCommented), LastCommentText: "repro please?",
		DaysSinceActivity: 30, DaysSinceLastActorAction: 30, DaysSinceAuthorAction: 35,
		StaleThresholdDays: 14, CloseThresholdDays: 7,
	})
	if c.hadToolError() {
		t.Fatal("hadToolError() should start false")
	}
	if _, err := c.doAddLabel(ctx, 7, "request clarification"); err == nil {
		t.Fatal("doAddLabel(7) on HTTP 500 = nil error, want error")
	}
	if !c.hadToolError() {
		t.Error("doAddLabel did not record the infrastructure error (run would not fail loud)")
	}
}
