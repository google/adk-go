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

package redis_test

import (
	"testing"
	"time"

	"google.golang.org/adk/session/redis"

	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
)

// An event appended with an explicit Partial: false reads back with
// PartialSet true, as adk-python's Redis service keeps partial=False
// (test_append_event_complete[redis]), and an event that left Partial unset
// reads back with PartialSet false.
func TestAppendEvent_ExplicitPartialFalseSurvivesGet(t *testing.T) {
	ctx := t.Context()
	svc, _, _ := newTestService(t, redis.Config{})
	sess := mustCreate(t, ctx, svc, "a", "u", "s1", nil)
	ts := time.Date(2026, 11, 13, 1, 2, 3, 0, time.UTC)
	explicit := &session.Event{ID: "explicit", Author: "agent", Timestamp: ts, LLMResponse: model.LLMResponse{Partial: false, PartialSet: true}}
	unset := &session.Event{ID: "unset", Author: "agent", Timestamp: ts.Add(time.Second)}
	for _, ev := range []*session.Event{explicit, unset} {
		if err := svc.AppendEvent(ctx, sess, ev); err != nil {
			t.Fatalf("AppendEvent(%q) error = %v", ev.ID, err)
		}
	}

	evs := mustGet(t, ctx, svc, "a", "u", "s1").Events()

	if evs.Len() != 2 {
		t.Fatalf("Get() returned %d events, want 2", evs.Len())
	}
	if got := evs.At(0); got.ID != "explicit" || got.Partial || !got.PartialSet {
		t.Errorf("event 0 = {ID: %q, Partial: %t, PartialSet: %t}, want {ID: %q, Partial: false, PartialSet: true}", got.ID, got.Partial, got.PartialSet, "explicit")
	}
	if got := evs.At(1); got.ID != "unset" || got.Partial || got.PartialSet {
		t.Errorf("event 1 = {ID: %q, Partial: %t, PartialSet: %t}, want {ID: %q, Partial: false, PartialSet: false}", got.ID, got.Partial, got.PartialSet, "unset")
	}
}

// A session document whose events carry no "partial" key, as the service
// wrote before it recorded an explicit Partial: false, still loads, and its
// events read back with PartialSet false; an event with "partial": false
// reads back with PartialSet true.
func TestGet_StoredEventsWithAndWithoutPartialKey(t *testing.T) {
	ctx := t.Context()
	svc, mr, _ := newTestService(t, redis.Config{})
	mustSet(t, mr, "adk:session:a:u:s1", `{"id":"s1","app_name":"a","user_id":"u","state":{},`+
		`"events":[{"id":"old","author":"agent","timestamp":"2026-11-13T01:02:03Z"},`+
		`{"id":"new","author":"agent","timestamp":"2026-11-13T01:02:04Z","partial":false}],`+
		`"last_update_time":"2026-11-13T01:02:04Z"}`)

	evs := mustGet(t, ctx, svc, "a", "u", "s1").Events()

	if evs.Len() != 2 {
		t.Fatalf("Get() returned %d events, want 2", evs.Len())
	}
	if got := evs.At(0); got.ID != "old" || got.PartialSet {
		t.Errorf("event 0 = {ID: %q, PartialSet: %t}, want {ID: %q, PartialSet: false}", got.ID, got.PartialSet, "old")
	}
	if got := evs.At(1); got.ID != "new" || got.Partial || !got.PartialSet {
		t.Errorf("event 1 = {ID: %q, Partial: %t, PartialSet: %t}, want {ID: %q, Partial: false, PartialSet: true}", got.ID, got.Partial, got.PartialSet, "new")
	}
}

// A session document whose event list holds null, which no service writes,
// still loads, and an append to it still succeeds.
func TestAppendEvent_StoredNullEventDoesNotPanic(t *testing.T) {
	ctx := t.Context()
	svc, mr, _ := newTestService(t, redis.Config{})
	mustSet(t, mr, "adk:session:a:u:s1", `{"id":"s1","app_name":"a","user_id":"u","state":{},`+
		`"events":[null],"last_update_time":"2026-11-13T01:02:03Z"}`)
	sess := mustGet(t, ctx, svc, "a", "u", "s1")

	err := svc.AppendEvent(ctx, sess, &session.Event{ID: "e1", Author: "agent", Timestamp: time.Date(2026, 11, 13, 1, 2, 4, 0, time.UTC)})
	if err != nil {
		t.Fatalf("AppendEvent() error = %v, want nil", err)
	}
	if n := mustGet(t, ctx, svc, "a", "u", "s1").Events().Len(); n != 2 {
		t.Errorf("Get() returned %d events, want 2", n)
	}
}

// A session document with an event that does not decode is an error, not a
// session with the event dropped.
func TestGet_UndecodableStoredEventIsAnError(t *testing.T) {
	ctx := t.Context()
	svc, mr, _ := newTestService(t, redis.Config{})
	mustSet(t, mr, "adk:session:a:u:s1", `{"id":"s1","app_name":"a","user_id":"u","state":{},`+
		`"events":[{"id":"e1","timestamp":"not a time"}],"last_update_time":"2026-11-13T01:02:03Z"}`)

	_, err := svc.Get(ctx, &session.GetRequest{AppName: "a", UserID: "u", SessionID: "s1"})

	if err == nil {
		t.Error("Get() error = nil, want an error for the undecodable event")
	}
}
