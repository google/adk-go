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
	"encoding/json"
	"testing"

	"google.golang.org/adk/v2/session"
)

// The JSON form is what the database session service stores for an event's
// actions, and what adk-python writes as rewind_before_invocation_id in
// camelCase.
func TestEventActionsJSON_RewindBeforeInvocationID(t *testing.T) {
	tests := []struct {
		name    string
		actions session.EventActions
		wantKey bool
	}{
		{name: "set", actions: session.EventActions{RewindBeforeInvocationID: "inv2"}, wantKey: true},
		{name: "empty", actions: session.EventActions{}, wantKey: false},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			data, err := json.Marshal(tc.actions)
			if err != nil {
				t.Fatalf("json.Marshal: %v", err)
			}
			var wire map[string]any
			if err := json.Unmarshal(data, &wire); err != nil {
				t.Fatalf("json.Unmarshal to map: %v", err)
			}
			value, ok := wire["rewindBeforeInvocationId"]
			if ok != tc.wantKey {
				t.Fatalf("key rewindBeforeInvocationId present = %v in %s, want %v", ok, data, tc.wantKey)
			}
			if tc.wantKey && value != "inv2" {
				t.Errorf("rewindBeforeInvocationId = %v, want \"inv2\"", value)
			}

			var back session.EventActions
			if err := json.Unmarshal(data, &back); err != nil {
				t.Fatalf("json.Unmarshal: %v", err)
			}
			if back.RewindBeforeInvocationID != tc.actions.RewindBeforeInvocationID {
				t.Errorf("round trip RewindBeforeInvocationID = %q, want %q", back.RewindBeforeInvocationID, tc.actions.RewindBeforeInvocationID)
			}
		})
	}
}

func TestInMemoryService_ReloadedRewindEventKeepsTarget(t *testing.T) {
	ctx := t.Context()
	svc := session.InMemoryService()
	created, err := svc.Create(ctx, &session.CreateRequest{AppName: "app", UserID: "u1", SessionID: "s1"})
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	ev := session.NewEvent(ctx, "e-rewind")
	ev.Author = "user"
	ev.Actions.RewindBeforeInvocationID = "inv2"
	if err := svc.AppendEvent(ctx, created.Session, ev); err != nil {
		t.Fatalf("AppendEvent: %v", err)
	}

	resp, err := svc.Get(ctx, &session.GetRequest{AppName: "app", UserID: "u1", SessionID: "s1"})
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if n := resp.Session.Events().Len(); n != 1 {
		t.Fatalf("reloaded session has %d events, want 1", n)
	}
	if got := resp.Session.Events().At(0).Actions.RewindBeforeInvocationID; got != "inv2" {
		t.Errorf("reloaded RewindBeforeInvocationID = %q, want \"inv2\"", got)
	}
}
