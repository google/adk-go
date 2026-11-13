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

package peragent_test

import (
	"errors"
	"testing"

	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/session/peragent"
)

// A second Create of a taken session ID returns an error that wraps
// session.ErrAlreadyExists, as adk-python's SqliteSessionService, which the
// per-agent service wraps, raises AlreadyExistsError
// (sqlite_session_service.py:229-237).
func TestPerAgentService_Create_DuplicateWrapsErrAlreadyExists(t *testing.T) {
	ctx := t.Context()
	svc := newService(t, peragent.Config{AgentsRoot: t.TempDir()})
	if _, err := svc.Create(ctx, &session.CreateRequest{AppName: "app", UserID: "u1", SessionID: "s1"}); err != nil {
		t.Fatalf("Create() error = %v", err)
	}

	_, err := svc.Create(ctx, &session.CreateRequest{AppName: "app", UserID: "u1", SessionID: "s1"})

	if !errors.Is(err, session.ErrAlreadyExists) {
		t.Errorf("Create(duplicate) error = %v, want an error wrapping session.ErrAlreadyExists", err)
	}
}
