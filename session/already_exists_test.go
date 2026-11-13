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
	"errors"
	"testing"

	"google.golang.org/adk/v2/session"
)

// A second Create of a taken session ID returns an error that wraps
// ErrAlreadyExists and keeps the service's message, as adk-python raises
// AlreadyExistsError (in_memory_session_service.py:134).
func TestInMemoryService_Create_DuplicateWrapsErrAlreadyExists(t *testing.T) {
	ctx := t.Context()
	svc := session.InMemoryService()
	if _, err := svc.Create(ctx, &session.CreateRequest{AppName: "app", UserID: "u1", SessionID: "s1"}); err != nil {
		t.Fatalf("Create() error = %v", err)
	}

	_, err := svc.Create(ctx, &session.CreateRequest{AppName: "app", UserID: "u1", SessionID: "s1"})

	if !errors.Is(err, session.ErrAlreadyExists) {
		t.Errorf("Create(duplicate) error = %v, want an error wrapping session.ErrAlreadyExists", err)
	}
	if err == nil || err.Error() != "session s1 already exists" {
		t.Errorf("Create(duplicate) error = %v, want %q", err, "session s1 already exists")
	}
}
