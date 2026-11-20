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

package session_test

import (
	"testing"

	"github.com/google/go-cmp/cmp"

	"google.golang.org/adk/v2/session"
)

// A resumable runner rebuilds agents from the stored EndOfAgent and
// AgentState, so a caller that keeps and changes its event after appending it
// must not change what the session holds.
func TestInMemoryService_StoresEndOfAgentAndAgentStateWithoutAlias(t *testing.T) {
	ctx := t.Context()
	svc := session.InMemoryService()
	created, err := svc.Create(ctx, &session.CreateRequest{AppName: "app", UserID: "u1", SessionID: "s1"})
	if err != nil {
		t.Fatalf("Create() failed: %v", err)
	}
	ended := true
	ev := session.NewEvent(ctx, "inv-1")
	ev.Author = "root_agent"
	ev.Actions.EndOfAgent = &ended
	ev.Actions.AgentState = map[string]any{"step": 1}
	if err := svc.AppendEvent(ctx, created.Session, ev); err != nil {
		t.Fatalf("AppendEvent() failed: %v", err)
	}

	*ev.Actions.EndOfAgent = false
	ev.Actions.AgentState["step"] = 2

	resp, err := svc.Get(ctx, &session.GetRequest{AppName: "app", UserID: "u1", SessionID: "s1"})
	if err != nil {
		t.Fatalf("Get() failed: %v", err)
	}
	if n := resp.Session.Events().Len(); n != 1 {
		t.Fatalf("stored session has %d events, want 1", n)
	}
	got := resp.Session.Events().At(0).Actions
	if got.EndOfAgent == nil || !*got.EndOfAgent {
		t.Errorf("stored EndOfAgent = %v, want a pointer to true", got.EndOfAgent)
	}
	if diff := cmp.Diff(map[string]any{"step": 1}, got.AgentState); diff != "" {
		t.Errorf("stored AgentState mismatch (-want +got):\n%s", diff)
	}
}
