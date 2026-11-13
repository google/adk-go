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

package database

import (
	"errors"
	"testing"

	"google.golang.org/adk/v2/session"
)

// A second Create of a taken session ID returns an error that wraps
// session.ErrAlreadyExists and leaves the stored session and the app state
// as they were, as adk-python raises AlreadyExistsError
// (database_session_service.py:661-665, :713-723).
func TestDatabaseService_Create_DuplicateWrapsErrAlreadyExists(t *testing.T) {
	ctx := t.Context()
	svc := newFromURL(t, "sqlite:///:memory:")
	if _, err := svc.Create(ctx, &session.CreateRequest{AppName: "app", UserID: "u1", SessionID: "s1", State: map[string]any{"app:k": "v1", "k": "v1"}}); err != nil {
		t.Fatalf("Create() error = %v", err)
	}

	_, err := svc.Create(ctx, &session.CreateRequest{AppName: "app", UserID: "u1", SessionID: "s1", State: map[string]any{"app:k": "v2", "k": "v2"}})

	if !errors.Is(err, session.ErrAlreadyExists) {
		t.Fatalf("Create(duplicate) error = %v, want an error wrapping session.ErrAlreadyExists", err)
	}
	got, err := svc.Get(ctx, &session.GetRequest{AppName: "app", UserID: "u1", SessionID: "s1"})
	if err != nil {
		t.Fatalf("Get() error = %v", err)
	}
	for _, key := range []string{"app:k", "k"} {
		if v, _ := got.Session.State().Get(key); v != "v1" {
			t.Errorf("state[%q] after Create(duplicate) = %v, want %q", key, v, "v1")
		}
	}
}

// A Create whose insert fails for another reason than a taken ID does not
// report ErrAlreadyExists, whether the lookup that follows finds no session
// with that ID (the user has another one) or fails itself.
func TestDatabaseService_Create_OtherInsertFailureIsNotErrAlreadyExists(t *testing.T) {
	tests := []struct {
		name  string
		setup string
	}{
		{name: "insert refused", setup: "CREATE TRIGGER refuse BEFORE INSERT ON sessions BEGIN SELECT RAISE(ABORT, 'refused'); END"},
		{name: "no sessions table", setup: "DROP TABLE sessions"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ctx := t.Context()
			svc := newFromURL(t, "sqlite:///:memory:")
			if _, err := svc.Create(ctx, &session.CreateRequest{AppName: "app", UserID: "u1", SessionID: "s0"}); err != nil {
				t.Fatalf("Create(s0) error = %v", err)
			}
			if err := svc.(*databaseService).db.Exec(tc.setup).Error; err != nil {
				t.Fatalf("Exec(%q) error = %v", tc.setup, err)
			}

			_, err := svc.Create(ctx, &session.CreateRequest{AppName: "app", UserID: "u1", SessionID: "s1"})

			if err == nil {
				t.Fatal("Create() error = nil, want an error")
			}
			if errors.Is(err, session.ErrAlreadyExists) {
				t.Errorf("Create() error = %v, want an error that does not wrap session.ErrAlreadyExists", err)
			}
		})
	}
}
