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

package firestore

import (
	"errors"
	"maps"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"

	"google.golang.org/adk/v2/session"
)

func TestLocalSession_StateAndEvents(t *testing.T) {
	ts := time.Date(2026, time.May, 1, 12, 0, 0, 0, time.UTC)
	s := &localSession{
		appName:   "app1",
		userID:    "user1",
		sessionID: "sess1",
		state:     map[string]any{"k1": "v1"},
		updatedAt: ts,
		revision:  0,
	}

	if got := s.AppName(); got != "app1" {
		t.Errorf("AppName() = %q, want %q", got, "app1")
	}
	if got := s.UserID(); got != "user1" {
		t.Errorf("UserID() = %q, want %q", got, "user1")
	}
	if got := s.ID(); got != "sess1" {
		t.Errorf("ID() = %q, want %q", got, "sess1")
	}
	if got := s.LastUpdateTime(); !got.Equal(ts) {
		t.Errorf("LastUpdateTime() = %v, want %v", got, ts)
	}

	val, err := s.State().Get("k1")
	if err != nil || val != "v1" {
		t.Errorf("State().Get(k1) = (%v, %v), want (v1, nil)", val, err)
	}
	if _, err := s.State().Get("missing"); !errors.Is(err, session.ErrStateKeyNotExist) {
		t.Errorf("State().Get(missing) error = %v, want %v", err, session.ErrStateKeyNotExist)
	}
	if err := s.State().Set("k2", "v2"); err != nil {
		t.Fatalf("State().Set(k2) error = %v", err)
	}

	collected := maps.Collect(s.State().All())
	if diff := cmp.Diff(map[string]any{"k1": "v1", "k2": "v2"}, collected); diff != "" {
		t.Errorf("State().All() mismatch (-want +got):\n%s", diff)
	}

	// Early break in State().All() iterator must not panic or hang.
	count := 0
	for range s.State().All() {
		count++
		break
	}
	if count != 1 {
		t.Errorf("State().All() early break count = %d, want 1", count)
	}

	ev1 := &session.Event{ID: "e1", Author: "user", Timestamp: ts.Add(time.Second)}
	ev2 := &session.Event{ID: "e2", Author: "agent", Timestamp: ts.Add(2 * time.Second)}
	s.commitAppend(nil, ev1, 1)
	s.commitAppend(nil, ev2, 2)

	if got := s.Events().Len(); got != 2 {
		t.Fatalf("Events().Len() = %d, want 2", got)
	}
	if got := s.Events().At(0); got == nil || got.ID != "e1" {
		t.Errorf("Events().At(0) = %v, want e1", got)
	}
	if got := s.Events().At(-1); got != nil {
		t.Errorf("Events().At(-1) = %v, want nil", got)
	}
	if got := s.Events().At(2); got != nil {
		t.Errorf("Events().At(2) = %v, want nil", got)
	}

	// Early break in Events().All() iterator must not panic or hang.
	evCount := 0
	for range s.Events().All() {
		evCount++
		break
	}
	if evCount != 1 {
		t.Errorf("Events().All() early break count = %d, want 1", evCount)
	}
}

func TestState_SetOnNilMapReturnsError(t *testing.T) {
	s := &localSession{appName: "app", userID: "user", sessionID: "sess"}
	if err := s.State().Set("k", "v"); err == nil {
		t.Fatal("State().Set on nil map: got nil error, want non-nil")
	}
}

func TestExtractAndMergeStates(t *testing.T) {
	delta := map[string]any{
		"app:theme":  "dark",
		"user:lang":  "pt",
		"temp:nonce": "123",
		"step":       "done",
	}
	appD, userD, sessD := extractStateDeltas(delta)
	if diff := cmp.Diff(map[string]any{"theme": "dark"}, appD); diff != "" {
		t.Errorf("appDelta mismatch (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff(map[string]any{"lang": "pt"}, userD); diff != "" {
		t.Errorf("userDelta mismatch (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff(map[string]any{"step": "done"}, sessD); diff != "" {
		t.Errorf("sessionDelta mismatch (-want +got):\n%s", diff)
	}

	merged := mergeStates(appD, userD, sessD)
	wantMerged := map[string]any{
		"app:theme": "dark",
		"user:lang": "pt",
		"step":      "done",
	}
	if diff := cmp.Diff(wantMerged, merged); diff != "" {
		t.Errorf("mergeStates mismatch (-want +got):\n%s", diff)
	}
}

func TestSerializationAndParsingHelpers(t *testing.T) {
	t.Run("marshalSessionState", func(t *testing.T) {
		got, err := marshalSessionState(nil)
		if err != nil || got != "{}" {
			t.Errorf("marshalSessionState(nil) = (%q, %v), want ({}, nil)", got, err)
		}

		oversized := map[string]any{"big": string(make([]byte, maxFirestorePayloadBytes+10))}
		if _, err := marshalSessionState(oversized); err == nil {
			t.Error("marshalSessionState(oversized) = nil error, want error")
		}
	})

	t.Run("decodeSessionState", func(t *testing.T) {
		for _, tc := range []struct {
			name    string
			in      any
			want    map[string]any
			wantErr bool
		}{
			{"nil", nil, map[string]any{}, false},
			{"empty string", "", map[string]any{}, false},
			{"json null string", "null", map[string]any{}, false},
			{"json object string", `{"k":"v"}`, map[string]any{"k": "v"}, false},
			{"map value", map[string]any{"k": "v"}, map[string]any{"k": "v"}, false},
			{"invalid json", `{bad`, nil, true},
			{"unexpected type", 123, nil, true},
		} {
			t.Run(tc.name, func(t *testing.T) {
				got, err := decodeSessionState(tc.in)
				if (err != nil) != tc.wantErr {
					t.Fatalf("decodeSessionState(%v) error = %v, wantErr %v", tc.in, err, tc.wantErr)
				}
				if !tc.wantErr {
					if diff := cmp.Diff(tc.want, got); diff != "" {
						t.Errorf("decodeSessionState(%v) mismatch (-want +got):\n%s", tc.in, diff)
					}
				}
			})
		}
	})

	t.Run("decodeEventData", func(t *testing.T) {
		ev, err := decodeEventData(`{"id":"e1","author":"user","invocationId":"inv1"}`)
		if err != nil {
			t.Fatalf("decodeEventData(string) error = %v", err)
		}
		if ev.ID != "e1" || ev.Author != "user" {
			t.Errorf("decodeEventData(string) = %+v", ev)
		}

		if _, err := decodeEventData(`{invalid`); err == nil {
			t.Error("decodeEventData(invalid json) = nil error, want error")
		}
		if _, err := decodeEventData(42); err == nil {
			t.Error("decodeEventData(int) = nil error, want error")
		}
	})

	t.Run("parseUpdateTime", func(t *testing.T) {
		want := time.Date(2026, 6, 15, 10, 30, 0, 0, time.UTC)
		if got := parseUpdateTime(want); !got.Equal(want) {
			t.Errorf("parseUpdateTime(time.Time) = %v, want %v", got, want)
		}
		if got := parseUpdateTime(float64(want.Unix())); !got.Equal(want) {
			t.Errorf("parseUpdateTime(float64) = %v, want %v", got, want)
		}
		if got := parseUpdateTime(want.Unix()); !got.Equal(want) {
			t.Errorf("parseUpdateTime(int64) = %v, want %v", got, want)
		}
		if got := parseUpdateTime(want.Format(time.RFC3339Nano)); !got.Equal(want) {
			t.Errorf("parseUpdateTime(RFC3339Nano) = %v, want %v", got, want)
		}
		if got := parseUpdateTime("not-a-time"); !got.IsZero() {
			t.Errorf("parseUpdateTime(invalid string) = %v, want zero", got)
		}
		if got := parseUpdateTime(nil); !got.IsZero() {
			t.Errorf("parseUpdateTime(nil) = %v, want zero", got)
		}
	})

	t.Run("parseRevision", func(t *testing.T) {
		if got := parseRevision(int64(7)); got != 7 {
			t.Errorf("parseRevision(int64) = %d, want 7", got)
		}
		if got := parseRevision(int(8)); got != 8 {
			t.Errorf("parseRevision(int) = %d, want 8", got)
		}
		if got := parseRevision(float64(9)); got != 9 {
			t.Errorf("parseRevision(float64) = %d, want 9", got)
		}
		if got := parseRevision(nil); got != 0 {
			t.Errorf("parseRevision(nil) = %d, want 0", got)
		}
	})
}
