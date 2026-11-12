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
		t.Error("editing name erased the stored access token")
	}
	if !bytes.Contains(encoded, []byte(`"refreshToken":"rt"`)) {
		t.Error("editing name erased the stored refresh token")
	}
	if !bytes.Contains(encoded, []byte(`"name":"bob"`)) {
		t.Error("the edited name was not stored")
	}
}

// TestClearingAuthTypeDoesNotRestoreSecrets covers the first review finding:
// restoring into an object whose authType no longer names a credential would
// leave the secrets visible, because outbound redaction no longer strips them.
func TestClearingAuthTypeDoesNotRestoreSecrets(t *testing.T) {
	cred := toolauth.Credential{
		AuthType: toolauth.CredentialTypeOAuth2,
		OAuth2:   &toolauth.OAuth2Auth{ClientID: "app", AccessToken: "at", RefreshToken: "rt"},
	}
	service := session.InMemoryService()
	created, err := service.Create(t.Context(), &session.CreateRequest{
		AppName: "app",
		UserID:  "user",
		State:   map[string]any{"drive": cred},
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
	read.State["drive"].(map[string]any)["authType"] = "none"

	body, err := json.Marshal(map[string]any{"stateDelta": map[string]any{"drive": read.State["drive"]}})
	if err != nil {
		t.Fatalf("Marshal(PATCH) failed: %v", err)
	}
	patch := httptest.NewRecorder()
	controller.UpdateSessionHandler(patch, mux.SetURLVars(httptest.NewRequestWithContext(ctx, http.MethodPatch, "/", bytes.NewReader(body)), vars))
	if patch.Code != http.StatusOK {
		t.Fatalf("UpdateSessionHandler() status = %d, want %d; body: %s", patch.Code, http.StatusOK, patch.Body.String())
	}
	if bytes.Contains(patch.Body.Bytes(), []byte(`"accessToken":"at"`)) {
		t.Error("PATCH response exposed the stored access token after authType was cleared")
	}
	if bytes.Contains(patch.Body.Bytes(), []byte(`"refreshToken":"rt"`)) {
		t.Error("PATCH response exposed the stored refresh token after authType was cleared")
	}

	later := httptest.NewRecorder()
	controller.GetSessionHandler(later, mux.SetURLVars(httptest.NewRequestWithContext(ctx, http.MethodGet, "/", nil), vars))
	if later.Code != http.StatusOK {
		t.Fatalf("second GetSessionHandler() status = %d, want %d", later.Code, http.StatusOK)
	}
	if bytes.Contains(later.Body.Bytes(), []byte(`"accessToken":"at"`)) {
		t.Error("a later GET exposed the stored access token after authType was cleared")
	}
}

// TestReorderingArrayCredentialsKeepsTheirOwnTokens covers the second review
// finding: matching array entries by index would hand a reordered credential
// the other entry's tokens.
func TestReorderingArrayCredentialsKeepsTheirOwnTokens(t *testing.T) {
	credA := toolauth.Credential{
		AuthType: toolauth.CredentialTypeOAuth2,
		OAuth2:   &toolauth.OAuth2Auth{ClientID: "a", AccessToken: "at-a", RefreshToken: "rt-a"},
	}
	credB := toolauth.Credential{
		AuthType: toolauth.CredentialTypeOAuth2,
		OAuth2:   &toolauth.OAuth2Auth{ClientID: "b", AccessToken: "at-b", RefreshToken: "rt-b"},
	}
	service := session.InMemoryService()
	created, err := service.Create(t.Context(), &session.CreateRequest{
		AppName: "app",
		UserID:  "user",
		State:   map[string]any{"creds": []any{credA, credB}},
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
	creds := read.State["creds"].([]any)
	if len(creds) != 2 {
		t.Fatalf("GET returned %d credentials, want 2", len(creds))
	}
	creds[0], creds[1] = creds[1], creds[0]

	body, err := json.Marshal(map[string]any{"stateDelta": map[string]any{"creds": creds}})
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
	stored, err := reread.Session.State().Get("creds")
	if err != nil {
		t.Fatalf("State().Get(creds) failed: %v", err)
	}
	want := map[string]string{"a": "at-a", "b": "at-b"}
	for _, item := range stored.([]any) {
		cred := item.(map[string]any)
		oauth := cred["oauth2"].(map[string]any)
		clientID := oauth["clientId"].(string)
		if got := oauth["accessToken"]; got != want[clientID] {
			t.Errorf("credential %q got access token %v, want %q", clientID, got, want[clientID])
		}
	}
}
