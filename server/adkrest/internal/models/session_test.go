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

package models_test

import (
	"testing"
	"time"

	"google.golang.org/adk/v2/server/adkrest/internal/models"
	"google.golang.org/adk/v2/session"
)

// TestFromSessionAcceptsLastUpdateInFirstSecondOf1970 pins that a session whose
// last update falls in [0s, 1s) of the epoch converts. Its lastUpdateTime is 0
// in whole seconds, which FromSession used to report as "updated_at is empty",
// failing every GET and list that included the session.
func TestFromSessionAcceptsLastUpdateInFirstSecondOf1970(t *testing.T) {
	ctx := t.Context()
	service := session.InMemoryService()
	created, err := service.Create(ctx, &session.CreateRequest{AppName: "app", UserID: "u", SessionID: "s"})
	if err != nil {
		t.Fatalf("Create() failed: %v", err)
	}
	event := session.NewEvent(ctx, "inv")
	event.Author = "user"
	event.Timestamp = time.Unix(0, 5e8).UTC()
	if err := service.AppendEvent(ctx, created.Session, event); err != nil {
		t.Fatalf("AppendEvent() failed: %v", err)
	}
	got, err := service.Get(ctx, &session.GetRequest{AppName: "app", UserID: "u", SessionID: "s"})
	if err != nil {
		t.Fatalf("Get() failed: %v", err)
	}
	if !got.Session.LastUpdateTime().Equal(event.Timestamp) {
		t.Fatalf("LastUpdateTime() = %v, want %v; the test no longer sets up the case it pins", got.Session.LastUpdateTime(), event.Timestamp)
	}
	mapped, err := models.FromSession(got.Session)
	if err != nil {
		t.Fatalf("FromSession() failed: %v", err)
	}
	if mapped.UpdatedAt != 0 {
		t.Errorf("FromSession().UpdatedAt = %d, want 0", mapped.UpdatedAt)
	}
}
