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

package method

import (
	"net/http/httptest"
	"testing"

	"google.golang.org/adk/v2/session"
)

// TestGetSessionHandler_MissingUserID checks that a request without user_id
// fails even when the session service holds a session of the user "" with
// the requested ID.
func TestGetSessionHandler_MissingUserID(t *testing.T) {
	ctx := t.Context()
	svc := session.InMemoryService()
	if _, err := svc.Create(ctx, &session.CreateRequest{AppName: "123", UserID: "", SessionID: "s1"}); err != nil {
		t.Fatalf("Create() failed: %v", err)
	}
	h := NewGetSessionHandler(svc, "123", "get_session", "")

	rw := httptest.NewRecorder()
	err := h.Handle(ctx, rw, []byte(`{"class_method":"get_session","input":{"session_id":"s1"}}`))
	if err == nil {
		t.Fatal("Handle() error = nil, want an error for a missing user_id")
	}
	if got := rw.Body.Len(); got != 0 {
		t.Errorf("Handle() wrote %d bytes, want 0", got)
	}
}
