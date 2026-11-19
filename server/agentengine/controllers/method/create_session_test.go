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

// TestCreateSessionHandler_MissingUserID checks that a request without
// user_id fails and creates no session, because the session service accepts
// an empty user ID.
func TestCreateSessionHandler_MissingUserID(t *testing.T) {
	ctx := t.Context()
	svc := session.InMemoryService()
	h := NewCreateSessionHandler(svc, "123", "create_session", "")

	err := h.Handle(ctx, httptest.NewRecorder(), []byte(`{"class_method":"create_session","input":{"state":{"k":"v"}}}`))
	if err == nil {
		t.Fatal("Handle() error = nil, want an error for a missing user_id")
	}

	resp, err := svc.List(ctx, &session.ListRequest{AppName: "123"})
	if err != nil {
		t.Fatalf("List() failed: %v", err)
	}
	if got := len(resp.Sessions); got != 0 {
		t.Errorf("List() returned %d sessions, want 0", got)
	}
}
