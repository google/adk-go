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
	"errors"
	"iter"
	"maps"
	"testing"
	"time"

	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
)

// builtSession is a session.Session that a caller implements, with a State
// whose Set records each key or fails with setErr. A nil state makes State
// return nil.
type builtSession struct {
	state *recordingState
}

func (s *builtSession) ID() string      { return "owned" }
func (s *builtSession) AppName() string { return "123" }
func (s *builtSession) UserID() string  { return "user1" }
func (s *builtSession) State() session.State {
	if s.state == nil {
		return nil
	}
	return s.state
}
func (s *builtSession) Events() session.Events    { return nil }
func (s *builtSession) LastUpdateTime() time.Time { return time.Time{} }

type recordingState struct {
	set    map[string]any
	setErr error
}

func (st *recordingState) Get(key string) (any, error) {
	v, ok := st.set[key]
	if !ok {
		return nil, session.ErrStateKeyNotExist
	}
	return v, nil
}

func (st *recordingState) All() iter.Seq2[string, any] { return maps.All(st.set) }

func (st *recordingState) Set(key string, value any) error {
	if st.setErr != nil {
		return st.setErr
	}
	st.set[key] = value
	return nil
}

// AppendEvent takes the session value Get returned and a copy of it made
// with session.CloneSession alike: the event is sent once, and the value
// receives it without its temp: keys, the whole state delta, temp: keys
// included, and its timestamp, as adk-python's append_event updates any
// Session object (vertex_ai_session_service.py:413-421,
// base_session_service.py:182-186).
func TestAppendEvent_sessionValueReceivesEvent(t *testing.T) {
	tests := []struct {
		name string
		val  func(session.Session) session.Session
	}{
		{name: "value from Get", val: func(s session.Session) session.Session { return s }},
		{name: "CloneSession copy", val: session.CloneSession},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ctx := t.Context()
			s, fake := newFakeService(t)
			got, err := s.Get(ctx, &session.GetRequest{AppName: "123", UserID: "user1", SessionID: "owned"})
			if err != nil {
				t.Fatalf("Get() error = %v", err)
			}
			val := tc.val(got.Session)
			ts := time.Date(2026, 11, 13, 1, 2, 3, 0, time.UTC)
			ev := &session.Event{ID: "e1", Author: "user", Timestamp: ts, Actions: session.EventActions{StateDelta: map[string]any{"k": "v", "temp:t": "x"}}}

			if err := s.AppendEvent(ctx, val, ev); err != nil {
				t.Fatalf("AppendEvent() error = %v, want nil", err)
			}

			if n := fake.appends.Load(); n != 1 {
				t.Errorf("AppendEvent RPCs = %d, want 1", n)
			}
			if evs := val.Events(); evs.Len() != 1 || evs.At(0).ID != "e1" {
				t.Fatalf("session events = %d, want exactly the event %q", evs.Len(), "e1")
			}
			if got := val.Events().At(0).Actions.StateDelta; !maps.Equal(got, map[string]any{"k": "v"}) {
				t.Errorf("session event state delta = %v, want %v", got, map[string]any{"k": "v"})
			}
			for k, want := range map[string]any{"k": "v", "temp:t": "x"} {
				if v, err := val.State().Get(k); err != nil || v != want {
					t.Errorf("session state[%q] = %v (err %v), want %q", k, v, err, want)
				}
			}
			if got := val.LastUpdateTime(); !got.Equal(ts) {
				t.Errorf("session LastUpdateTime() = %v, want %v", got, ts)
			}
		})
	}
}

// A partial event is neither sent nor applied to a session copy.
func TestAppendEvent_partialEventLeavesCloneUnchanged(t *testing.T) {
	ctx := t.Context()
	s, fake := newFakeService(t)
	got, err := s.Get(ctx, &session.GetRequest{AppName: "123", UserID: "user1", SessionID: "owned"})
	if err != nil {
		t.Fatalf("Get() error = %v", err)
	}
	clone := session.CloneSession(got.Session)
	ev := &session.Event{ID: "e1", Author: "agent", LLMResponse: model.LLMResponse{Partial: true}, Actions: session.EventActions{StateDelta: map[string]any{"k": "v"}}}

	if err := s.AppendEvent(ctx, clone, ev); err != nil {
		t.Fatalf("AppendEvent(partial) error = %v, want nil", err)
	}

	if n := fake.appends.Load(); n != 0 {
		t.Errorf("AppendEvent RPCs = %d, want 0", n)
	}
	if n := clone.Events().Len(); n != 0 {
		t.Errorf("clone events = %d, want 0", n)
	}
	if _, err := clone.State().Get("k"); !errors.Is(err, session.ErrStateKeyNotExist) {
		t.Errorf("clone state[%q] error = %v, want session.ErrStateKeyNotExist", "k", err)
	}
}

// A session value the caller built receives the state delta through its
// State, and one without a State receives nothing; when its Set fails, the
// event stays stored and AppendEvent returns the Set error wrapped.
func TestAppendEvent_callerBuiltSession(t *testing.T) {
	setErr := errors.New("state is read-only")
	tests := []struct {
		name    string
		state   *recordingState
		wantErr error
		wantSet map[string]any
	}{
		{name: "state receives delta", state: &recordingState{set: map[string]any{}}, wantSet: map[string]any{"k": "v", "temp:t": "x"}},
		{name: "set fails", state: &recordingState{set: map[string]any{}, setErr: setErr}, wantErr: setErr, wantSet: map[string]any{}},
		{name: "nil state"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			s, fake := newFakeService(t)
			built := &builtSession{state: tc.state}
			ev := &session.Event{ID: "e1", Author: "user", Actions: session.EventActions{StateDelta: map[string]any{"k": "v", "temp:t": "x"}}}

			err := s.AppendEvent(t.Context(), built, ev)

			if !errors.Is(err, tc.wantErr) || (tc.wantErr == nil) != (err == nil) {
				t.Errorf("AppendEvent() error = %v, want %v", err, tc.wantErr)
			}
			if n := fake.appends.Load(); n != 1 {
				t.Errorf("AppendEvent RPCs = %d, want 1", n)
			}
			if tc.state != nil && !maps.Equal(tc.state.set, tc.wantSet) {
				t.Errorf("state after AppendEvent = %v, want %v", tc.state.set, tc.wantSet)
			}
		})
	}
}

// An append whose request fails leaves the session value unchanged, temp:
// keys included. adk-python applies temp: keys to the session before the
// request (vertex_ai_session_service.py:414-421), so they stay after a
// failed one.
func TestAppendEvent_failedRequestLeavesSessionUnchanged(t *testing.T) {
	s, fake := newFakeService(t)
	sess := &localSession{appName: "123", userID: "user1", sessionID: "denied", state: map[string]any{}}
	ev := &session.Event{ID: "e1", Author: "user", Timestamp: time.Date(2026, 11, 13, 1, 2, 3, 0, time.UTC), Actions: session.EventActions{StateDelta: map[string]any{"k": "v", "temp:t": "x"}}}

	if err := s.AppendEvent(t.Context(), sess, ev); err == nil {
		t.Fatal("AppendEvent(unwritable session) error = nil, want the backend's error")
	}

	if n := fake.appends.Load(); n != 1 {
		t.Errorf("AppendEvent RPCs = %d, want 1", n)
	}
	if n := sess.Events().Len(); n != 0 {
		t.Errorf("session events = %d, want 0", n)
	}
	if len(sess.state) != 0 {
		t.Errorf("session state = %v, want empty", sess.state)
	}
	if got := sess.LastUpdateTime(); !got.IsZero() {
		t.Errorf("session LastUpdateTime() = %v, want the zero time", got)
	}
}
