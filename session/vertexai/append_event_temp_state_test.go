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

package vertexai

import (
	"strings"
	"testing"

	"google.golang.org/adk/v2/session"
)

// TestAppendEvent_stripsTempKeysBeforePersist pins the #1410 contract offline:
// temp: keys must be absent from the AppendEvent RPC payload, while the
// caller-provided event and the in-process local session still see them for
// the rest of the invocation. The shared suite case of the same name needs a
// re-recorded Agent Engine replay after this change (see casesWithoutReplay).
func TestAppendEvent_stripsTempKeysBeforePersist(t *testing.T) {
	fake := &fakeVertexAiSessionService{}
	client := newFakeVertexAiClient(t, fake)
	svc := &vertexAiService{client: client}

	sess := &localSession{
		appName:   "123",
		userID:    "user1",
		sessionID: "owned",
		state:     map[string]any{},
	}
	event := &session.Event{
		ID:           "event1",
		Author:       "user",
		InvocationID: "inv1",
		Actions: session.EventActions{
			StateDelta: map[string]any{
				"temp:k1": "v1",
				"sk":      "v2",
			},
		},
	}

	if err := svc.AppendEvent(t.Context(), sess, event); err != nil {
		t.Fatalf("AppendEvent() error = %v", err)
	}

	if _, ok := event.Actions.StateDelta["temp:k1"]; !ok {
		t.Errorf("caller event lost temp:k1; AppendEvent must not mutate the input: %v", event.Actions.StateDelta)
	}
	if event.Actions.StateDelta["sk"] != "v2" {
		t.Errorf("caller event lost sk: %v", event.Actions.StateDelta)
	}

	if got := len(fake.events); got != 1 {
		t.Fatalf("AppendEvent RPCs stored %d events, want 1", got)
	}
	actions := fake.events[0].GetActions()
	if actions == nil || actions.StateDelta == nil {
		t.Fatalf("persisted event Actions.StateDelta = nil, want non-temp keys retained")
	}
	assertPersistedDeltaHasNoTemp(t, actions.StateDelta.AsMap(), "AppendEvent RPC payload")

	// Local session still receives the untrimmed event so temp: remains readable
	// for the rest of the invocation via session state / local event list trim.
	if _, err := sess.State().Get("temp:k1"); err != nil {
		t.Errorf("local session state missing temp:k1 after AppendEvent: %v", err)
	}
	if got, err := sess.State().Get("sk"); err != nil || got != "v2" {
		t.Errorf("local session state sk = %v, %v; want v2", got, err)
	}
}

// TestAppendEvent_stripsTempKeysFromRawEvent pins the other stored copy of the
// delta. AppendEvent writes StateDelta twice: the Actions column and, when
// eventNeedsRawEvent is true, the raw_event blob. Get prefers raw_event, so a
// trim that only cleaned createAiplatformpbEventActions would leak temp: keys
// on read-back while TestAppendEvent_stripsTempKeysBeforePersist stayed green.
func TestAppendEvent_stripsTempKeysFromRawEvent(t *testing.T) {
	fake := &fakeVertexAiSessionService{}
	client := newFakeVertexAiClient(t, fake)
	svc := &vertexAiService{client: client}

	sess := &localSession{
		appName:   "123",
		userID:    "user1",
		sessionID: "owned",
		state:     map[string]any{},
	}
	event := &session.Event{
		ID:           "event2",
		Author:       "user",
		InvocationID: "inv1",
		Output:       map[string]any{"node": "out"}, // forces raw_event
		Actions: session.EventActions{
			StateDelta: map[string]any{
				"temp:k1": "v1",
				"sk":      "v2",
			},
		},
	}

	if err := svc.AppendEvent(t.Context(), sess, event); err != nil {
		t.Fatalf("AppendEvent() error = %v", err)
	}

	if _, ok := event.Actions.StateDelta["temp:k1"]; !ok {
		t.Errorf("caller event lost temp:k1; AppendEvent must not mutate the input: %v", event.Actions.StateDelta)
	}
	if event.Actions.StateDelta["sk"] != "v2" {
		t.Errorf("caller event lost sk: %v", event.Actions.StateDelta)
	}

	if got := len(fake.events); got != 1 {
		t.Fatalf("AppendEvent RPCs stored %d events, want 1", got)
	}

	actions := fake.events[0].GetActions()
	if actions == nil || actions.StateDelta == nil {
		t.Fatalf("persisted event Actions.StateDelta = nil, want non-temp keys retained")
	}
	col := actions.StateDelta.AsMap()
	assertPersistedDeltaHasNoTemp(t, col, "Actions.StateDelta")

	raw := fake.events[0].GetRawEvent()
	if raw == nil {
		t.Fatal("raw_event is nil; Output must populate it so this test can see the blob copy of StateDelta")
	}
	blob := raw.AsMap()
	rawActions, ok := blob["actions"].(map[string]any)
	if !ok {
		t.Fatalf("raw_event actions = %T (%v), want map", blob["actions"], blob["actions"])
	}
	rawDelta, ok := rawActions["stateDelta"].(map[string]any)
	if !ok {
		t.Fatalf("raw_event actions.stateDelta = %T (%v), want map", rawActions["stateDelta"], rawActions["stateDelta"])
	}
	assertPersistedDeltaHasNoTemp(t, rawDelta, "raw_event")

	if _, err := sess.State().Get("temp:k1"); err != nil {
		t.Errorf("local session state missing temp:k1 after AppendEvent: %v", err)
	}
	if got, err := sess.State().Get("sk"); err != nil || got != "v2" {
		t.Errorf("local session state sk = %v, %v; want v2", got, err)
	}
}

func assertPersistedDeltaHasNoTemp(t *testing.T, delta map[string]any, where string) {
	t.Helper()
	for k := range delta {
		if strings.HasPrefix(k, session.KeyPrefixTemp) {
			t.Errorf("temp key reached %s: key %q, delta %v", where, k, delta)
		}
	}
	if got := delta["sk"]; got != "v2" {
		t.Errorf("non-temp key missing from %s: got %v, want sk=v2", where, delta)
	}
}
