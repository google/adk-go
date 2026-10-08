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
	"testing"

	"google.golang.org/adk/v2/session"
)

// The shared suite covers Get and AppendEvent on a missing session for every
// session.Service, but this package can only run a suite case that has recorded
// Agent Engine traffic to replay, and recording needs a live backend. These
// tests pin the same behavior against the in-process fake in
// delete_ownership_test.go, which needs no network and no recording. They are
// what service_test.go's casesWithoutReplay points at.
//
// They also pin the other half of the contract: an error that is not a missing
// session must not come back as session.ErrNotFound, or the REST layer answers
// 404 for a backend failure and the client retries something that will never
// work.

func TestGet_missingSession_wrapsErrNotFound(t *testing.T) {
	s, _ := newFakeService(t)

	_, err := s.Get(t.Context(), &session.GetRequest{AppName: "123", UserID: "user1", SessionID: "other"})
	if err == nil {
		t.Fatalf("Get(missing session) error = nil, want an error wrapping session.ErrNotFound")
	}
	if !errors.Is(err, session.ErrNotFound) {
		t.Errorf("Get(missing session) error = %v, want an error wrapping session.ErrNotFound", err)
	}
}

func TestGet_permissionDenied_isNotErrNotFound(t *testing.T) {
	s, _ := newFakeService(t)

	_, err := s.Get(t.Context(), &session.GetRequest{AppName: "123", UserID: "user1", SessionID: "denied"})
	if err == nil {
		t.Fatalf("Get(unreadable session) error = nil, want a permission error")
	}
	if errors.Is(err, session.ErrNotFound) {
		t.Errorf("Get(unreadable session) error = %v, want an error that is NOT session.ErrNotFound", err)
	}
}

// A session that exists but belongs to someone else is not a missing session:
// reporting it as one would let a caller distinguish "no such session" from
// "not yours" by probing, and would have the REST layer answer 404 for what is
// really a refusal.
func TestGet_wrongUser_isNotErrNotFound(t *testing.T) {
	s, _ := newFakeService(t)

	_, err := s.Get(t.Context(), &session.GetRequest{AppName: "123", UserID: "user2", SessionID: "owned"})
	if err == nil {
		t.Fatalf("Get(other user's session) error = nil, want an ownership error")
	}
	if errors.Is(err, session.ErrNotFound) {
		t.Errorf("Get(other user's session) error = %v, want an error that is NOT session.ErrNotFound", err)
	}
}

func TestAppendEvent_missingSession_wrapsErrNotFound(t *testing.T) {
	s, fake := newFakeService(t)

	sess := &localSession{appName: "123", userID: "user1", sessionID: "other"}
	err := s.AppendEvent(t.Context(), sess, &session.Event{ID: "e1", Author: "user", InvocationID: "inv1"})
	if err == nil {
		t.Fatalf("AppendEvent(missing session) error = nil, want an error wrapping session.ErrNotFound")
	}
	if !errors.Is(err, session.ErrNotFound) {
		t.Errorf("AppendEvent(missing session) error = %v, want an error wrapping session.ErrNotFound", err)
	}
	if got := fake.appends.Load(); got != 1 {
		t.Errorf("AppendEvent RPCs = %d, want 1: the error must come from the backend, not from a local short-circuit", got)
	}
}

func TestAppendEvent_permissionDenied_isNotErrNotFound(t *testing.T) {
	s, _ := newFakeService(t)

	sess := &localSession{appName: "123", userID: "user1", sessionID: "denied"}
	err := s.AppendEvent(t.Context(), sess, &session.Event{ID: "e1", Author: "user", InvocationID: "inv1"})
	if err == nil {
		t.Fatalf("AppendEvent(unwritable session) error = nil, want a permission error")
	}
	if errors.Is(err, session.ErrNotFound) {
		t.Errorf("AppendEvent(unwritable session) error = %v, want an error that is NOT session.ErrNotFound", err)
	}
}

// The negative control: an append to a session that is there still succeeds, so
// a wrap that fired on every error could not pass.
func TestAppendEvent_existingSession_succeeds(t *testing.T) {
	s, fake := newFakeService(t)

	sess := &localSession{appName: "123", userID: "user1", sessionID: "owned"}
	if err := s.AppendEvent(t.Context(), sess, &session.Event{ID: "e1", Author: "user", InvocationID: "inv1"}); err != nil {
		t.Fatalf("AppendEvent(existing session) error = %v, want nil", err)
	}
	if got := fake.appends.Load(); got != 1 {
		t.Errorf("AppendEvent RPCs = %d, want 1", got)
	}
}
