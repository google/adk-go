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
	"encoding/json"
	"net/http/httptest"
	"slices"
	"testing"

	"google.golang.org/adk/v2/server/agentengine/internal/models"
	"google.golang.org/adk/v2/session"
)

// TestListSessionHandler_FiltersByUserID checks that list_sessions returns
// only the sessions of the requested user_id.
func TestListSessionHandler_FiltersByUserID(t *testing.T) {
	ctx := t.Context()
	svc := session.InMemoryService()
	for _, req := range []*session.CreateRequest{
		{AppName: "123", UserID: "u1", SessionID: "s1"},
		{AppName: "123", UserID: "u1", SessionID: "s2"},
		{AppName: "123", UserID: "u2", SessionID: "s3"},
	} {
		if _, err := svc.Create(ctx, req); err != nil {
			t.Fatalf("Create(%q) failed: %v", req.SessionID, err)
		}
	}
	h := NewListSessionHandler(svc, "123", "list_sessions", "")
	rw := httptest.NewRecorder()

	if err := h.Handle(ctx, rw, []byte(`{"class_method":"list_sessions","input":{"user_id":"u1"}}`)); err != nil {
		t.Fatalf("Handle() failed: %v", err)
	}

	var resp models.ListSessionResponse
	if err := json.Unmarshal(rw.Body.Bytes(), &resp); err != nil {
		t.Fatalf("json.Unmarshal() failed: %v", err)
	}
	var got []string
	for _, s := range resp.Output.Sessions {
		got = append(got, s.UserID+"/"+s.ID)
	}
	slices.Sort(got)
	if want := []string{"u1/s1", "u1/s2"}; !slices.Equal(got, want) {
		t.Errorf("Handle() listed sessions %v, want %v", got, want)
	}
}
