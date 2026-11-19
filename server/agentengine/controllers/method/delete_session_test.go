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

// TestDeleteSessionHandler_MissingUserID checks that a request without
// user_id fails and leaves the session of the user "" with the requested ID
// in place.
func TestDeleteSessionHandler_MissingUserID(t *testing.T) {
	ctx := t.Context()
	svc := session.InMemoryService()
	if _, err := svc.Create(ctx, &session.CreateRequest{AppName: "123", UserID: "", SessionID: "s1"}); err != nil {
		t.Fatalf("Create() failed: %v", err)
	}
	h := NewDeleteSessionHandler(svc, "123", "delete_session", "")

	err := h.Handle(ctx, httptest.NewRecorder(), []byte(`{"class_method":"delete_session","input":{"session_id":"s1"}}`))
	if err == nil {
		t.Fatal("Handle() error = nil, want an error for a missing user_id")
	}

	if _, err := svc.Get(ctx, &session.GetRequest{AppName: "123", UserID: "", SessionID: "s1"}); err != nil {
		t.Errorf("Get() after Handle() failed: %v, want the session still stored", err)
	}
}
