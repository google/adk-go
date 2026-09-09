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

package session

import (
	"testing"
	"time"
)

// TestSessionAppendEvent_ReturnsUnsharedDelta pins that appendEvent hands back
// a delta map no one else holds.
//
// AppendEvent stores the returned map on the canonical record while holding
// only the service lock, never s.mu. The map appendEvent appends to s.events is
// reachable by anyone with the live session handle the moment s.mu is released,
// so returning that map — or a clone taken after the unlock — lets a caller
// walking session history write into a map the service is reading. A map read
// racing a map write is a runtime throw, not a recoverable panic, so this is
// worth a guard even though it cannot be observed sequentially.
//
// This test cannot prove the clone happens under the lock; that needs a
// concurrent writer, which reproduces the throw only probabilistically and
// takes the whole test binary down with it when it fires. What it does prove is
// that the returned map is not the published one, which is the shape the bug
// took.
func TestSessionAppendEvent_ReturnsUnsharedDelta(t *testing.T) {
	sess := &session{id: id{appName: "app", userID: "user", sessionID: "s1"}}

	event := &Event{
		ID:        "e1",
		Timestamp: time.Now(),
		Actions:   EventActions{StateDelta: map[string]any{"temp:scratch": "x", "keep": "y"}},
	}

	got, err := sess.appendEvent(event)
	if err != nil {
		t.Fatalf("appendEvent: %v", err)
	}
	if len(sess.events) != 1 {
		t.Fatalf("len(sess.events) = %d, want 1", len(sess.events))
	}

	// Write through the returned map. Nothing the session published may follow.
	got["injected"] = true

	published := sess.events[0].Actions.StateDelta
	if _, ok := published["injected"]; ok {
		t.Error("the published event shares its StateDelta map with the value appendEvent returned; " +
			"a caller holding the live handle can then write into a map AppendEvent reads without s.mu")
	}
	// The caller's own event must not follow either.
	if _, ok := event.Actions.StateDelta["injected"]; ok {
		t.Error("the caller's event shares its StateDelta map with the value appendEvent returned")
	}
}
