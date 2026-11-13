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
	"encoding/json"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"

	"google.golang.org/adk/v2/server/adkrest/internal/models"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool/toolauth"
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

// TestSessionStateCredentialsGoOutWithoutSecrets pins that a credential an app
// keeps in session state reaches the client without its secrets, while other
// state keeps every key.
func TestSessionStateCredentialsGoOutWithoutSecrets(t *testing.T) {
	ctx := t.Context()
	service := session.InMemoryService()
	created, err := service.Create(ctx, &session.CreateRequest{AppName: "app", UserID: "u", SessionID: "s", State: map[string]any{
		"drive": toolauth.Credential{AuthType: toolauth.CredentialTypeOAuth2, OAuth2: &toolauth.OAuth2Auth{
			ClientID: "app", AccessToken: "at", RefreshToken: "rt", TokenEndpointAuthMethod: "client_secret_post",
		}},
		"login": map[string]any{"user": "ann", "password": "pw"},
	}})
	if err != nil {
		t.Fatalf("Create() failed: %v", err)
	}
	mapped, err := models.FromSession(created.Session)
	if err != nil {
		t.Fatalf("FromSession() failed: %v", err)
	}

	b, err := json.Marshal(mapped)
	if err != nil {
		t.Fatalf("json.Marshal(session) failed: %v", err)
	}

	var got struct {
		State map[string]any `json:"state"`
	}
	if err := json.Unmarshal(b, &got); err != nil {
		t.Fatalf("json.Unmarshal(session) failed: %v", err)
	}
	want := map[string]any{
		"drive": map[string]any{"authType": "oauth2", "oauth2": map[string]any{
			"clientId": "app", "tokenEndpointAuthMethod": "client_secret_post",
		}},
		"login": map[string]any{"user": "ann", "password": "pw"},
	}
	if diff := cmp.Diff(want, got.State); diff != "" {
		t.Errorf("state mismatch (-want +got):\n%s", diff)
	}
}
