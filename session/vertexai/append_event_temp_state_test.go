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
	"context"
	"errors"
	"strings"
	"sync"
	"testing"

	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/types/known/timestamppb"

	"google.golang.org/adk/v2/session"

	aiplatformpb "cloud.google.com/go/aiplatform/apiv1beta1/aiplatformpb"
	longrunningpb "cloud.google.com/go/longrunning/autogen/longrunningpb"
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

// createSessionFake stores the session a CreateSession RPC carries and serves
// it back from GetSession, as Agent Engine does once the operation finishes.
type createSessionFake struct {
	aiplatformpb.UnimplementedSessionServiceServer

	mu      sync.Mutex
	created *aiplatformpb.Session
}

func (f *createSessionFake) CreateSession(_ context.Context, req *aiplatformpb.CreateSessionRequest) (*longrunningpb.Operation, error) {
	sess, ok := proto.Clone(req.GetSession()).(*aiplatformpb.Session)
	if !ok {
		return nil, errors.New("unexpected session type")
	}
	sess.Name = req.GetParent() + "/sessions/" + req.GetSessionId()
	sess.UpdateTime = timestamppb.Now()
	f.mu.Lock()
	f.created = sess
	f.mu.Unlock()
	return &longrunningpb.Operation{Name: sess.Name + "/operations/1"}, nil
}

func (f *createSessionFake) GetSession(context.Context, *aiplatformpb.GetSessionRequest) (*aiplatformpb.Session, error) {
	sess, ok := proto.Clone(f.createdSession()).(*aiplatformpb.Session)
	if !ok {
		return nil, errors.New("unexpected session type")
	}
	return sess, nil
}

func (f *createSessionFake) createdSession() *aiplatformpb.Session {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.created
}

// TestCreate_dropsTempKeysFromInitialState pins that temp: keys in
// CreateRequest.State never reach Agent Engine, so they are absent from the
// created session and every later read of it. AppendEvent already trims them
// from state deltas; adk-python drops them in create_session too.
func TestCreate_dropsTempKeysFromInitialState(t *testing.T) {
	tests := []struct {
		name      string
		state     map[string]any
		wantState map[string]any // nil: no session_state sent at all
	}{
		{
			name:      "temp and session keys",
			state:     map[string]any{"temp:k1": "v1", "sk": "v2"},
			wantState: map[string]any{"sk": "v2"},
		},
		{
			name:  "only temp keys",
			state: map[string]any{"temp:k1": "v1"},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			fake := &createSessionFake{}
			svc := &vertexAiService{client: newFakeVertexAiClient(t, fake)}

			resp, err := svc.Create(t.Context(), &session.CreateRequest{
				AppName:   "123",
				UserID:    "user1",
				SessionID: "s1",
				State:     tc.state,
			})
			if err != nil {
				t.Fatalf("Create() error = %v", err)
			}

			if _, ok := tc.state["temp:k1"]; !ok {
				t.Errorf("Create removed temp:k1 from the caller's map; it must not mutate the request")
			}

			sent := fake.createdSession().GetSessionState()
			if tc.wantState == nil {
				if sent != nil {
					t.Errorf("CreateSession RPC session_state = %v, want none", sent.AsMap())
				}
			} else {
				got := sent.AsMap()
				for k := range got {
					if strings.HasPrefix(k, session.KeyPrefixTemp) {
						t.Errorf("temp key reached the CreateSession RPC payload: key %q, state %v", k, got)
					}
				}
				for k, want := range tc.wantState {
					if got[k] != want {
						t.Errorf("CreateSession RPC session_state[%q] = %v, want %v", k, got[k], want)
					}
				}
			}

			if _, err := resp.Session.State().Get("temp:k1"); !errors.Is(err, session.ErrStateKeyNotExist) {
				t.Errorf("created session State().Get(%q) error = %v, want %v", "temp:k1", err, session.ErrStateKeyNotExist)
			}
		})
	}
}
