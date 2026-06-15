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

// FINDING H4 — in-memory session service drops Event.IsolationScope on append.
//
// Bug: inMemoryService.AppendEvent builds a manual deep copy of the incoming
// Event (the `eventCopy := &Event{...}` literal in session/inmemory.go). That
// literal copies Routes/RequestedInput/Output/NodeInfo/Branch but OMITS the
// IsolationScope field, so the stored copy always has IsolationScope == "".
//
// Expected: after AppendEvent + Get round-trip, the persisted event preserves
// the IsolationScope value that was set on the appended event.
//
// This test currently FAILS, demonstrating the bug.

package session_test

import (
	"testing"

	"google.golang.org/adk/session"
)

func TestVbugH4_IsolationScopeRoundTrip(t *testing.T) {
	ctx := t.Context()
	service := session.InMemoryService()

	createResp, err := service.Create(ctx, &session.CreateRequest{AppName: "app", UserID: "user"})
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	sess := createResp.Session

	event := &session.Event{
		ID:             "scoped_event",
		Author:         "agent",
		IsolationScope: "scope-1",
	}
	if err := service.AppendEvent(ctx, sess, event); err != nil {
		t.Fatalf("AppendEvent: %v", err)
	}

	got, err := service.Get(ctx, &session.GetRequest{AppName: "app", UserID: "user", SessionID: sess.ID()})
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	evs := got.Session.Events()
	if evs.Len() != 1 {
		t.Fatalf("got %d events, want 1", evs.Len())
	}
	ev := evs.At(0)
	if ev.IsolationScope != "scope-1" {
		t.Errorf("IsolationScope not persisted: got %q, want %q", ev.IsolationScope, "scope-1")
	}
}
