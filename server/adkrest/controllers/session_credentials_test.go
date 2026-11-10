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

package controllers_test

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gorilla/mux"

	"google.golang.org/adk/v2/server/adkrest/controllers"
	"google.golang.org/adk/v2/server/authn"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool/toolauth"
)

// TestEditingASiblingFieldKeepsTheCredentialTokens is the regression from
// #2245: GET strips the credential secrets, the client edits the name next to
// the credential, and PATCH writes the redacted credential back. The update
// must keep the stored tokens.
func TestEditingASiblingFieldKeepsTheCredentialTokens(t *testing.T) {
	cred := toolauth.Credential{
		AuthType: toolauth.CredentialTypeOAuth2,
		OAuth2: &toolauth.OAuth2Auth{
			ClientID:     "app",
			AccessToken:  "at",
			RefreshToken: "rt",
		},
	}
	service := session.InMemoryService()
	created, err := service.Create(t.Context(), &session.CreateRequest{
		AppName: "app",
		UserID:  "user",
		State:   map[string]any{"profile": map[string]any{"name": "ann", "cred": cred}},
	})
	if err != nil {
		t.Fatalf("Create() failed: %v", err)
	}

	controller := controllers.NewSessionsAPIController(service)
	ctx := authn.WithCaller(t.Context(), &authn.Caller{UserID: "user"})
	vars := map[string]string{
		"app_name":   "app",
		"user_id":    "user",
		"session_id": created.Session.ID(),
	}

	get := httptest.NewRecorder()
	controller.GetSessionHandler(get, mux.SetURLVars(httptest.NewRequestWithContext(ctx, http.MethodGet, "/", nil), vars))
	if get.Code != http.StatusOK {
		t.Fatalf("GetSessionHandler() status = %d, want %d", get.Code, http.StatusOK)
	}
	var read struct {
		State map[string]any `json:"state"`
	}
	if err := json.Unmarshal(get.Body.Bytes(), &read); err != nil {
		t.Fatalf("Unmarshal(GET) failed: %v", err)
	}
	read.State["profile"].(map[string]any)["name"] = "bob"

	body, err := json.Marshal(map[string]any{"stateDelta": map[string]any{"profile": read.State["profile"]}})
	if err != nil {
		t.Fatalf("Marshal(PATCH) failed: %v", err)
	}
	patch := httptest.NewRecorder()
	controller.UpdateSessionHandler(patch, mux.SetURLVars(httptest.NewRequestWithContext(ctx, http.MethodPatch, "/", bytes.NewReader(body)), vars))
	if patch.Code != http.StatusOK {
		t.Fatalf("UpdateSessionHandler() status = %d, want %d; body: %s", patch.Code, http.StatusOK, patch.Body.String())
	}

	reread, err := service.Get(t.Context(), &session.GetRequest{AppName: "app", UserID: "user", SessionID: created.Session.ID()})
	if err != nil {
		t.Fatalf("Get() failed: %v", err)
	}
	profile, err := reread.Session.State().Get("profile")
	if err != nil {
		t.Fatalf("State().Get(profile) failed: %v", err)
	}
	encoded, err := json.Marshal(profile)
	if err != nil {
		t.Fatalf("Marshal(profile) failed: %v", err)
	}
	if !bytes.Contains(encoded, []byte(`"accessToken":"at"`)) {
		t.Errorf("editing name erased the stored access token: %s", encoded)
	}
	if !bytes.Contains(encoded, []byte(`"refreshToken":"rt"`)) {
		t.Errorf("editing name erased the stored refresh token: %s", encoded)
	}
	if !bytes.Contains(encoded, []byte(`"name":"bob"`)) {
		t.Errorf("the edited name was not stored: %s", encoded)
	}
}
