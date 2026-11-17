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
	"context"
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
		t.Fatalf("UpdateSessionHandler() status = %d, want %d", patch.Code, http.StatusOK)
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

// TestChangedAuthTypeDoesNotRestoreSecrets covers the first review finding:
// restoring into an object whose authType no longer names a credential would
// leave the secrets visible, because outbound redaction no longer strips them.
// "oauth_2" normalizes onto the stored authType, so only the "still a
// credential" check can refuse it.
func TestChangedAuthTypeDoesNotRestoreSecrets(t *testing.T) {
	for _, authType := range []string{"none", "oauth_2"} {
		t.Run(authType, func(t *testing.T) {
			f := newSessionFixture(t, map[string]any{
				"drive": oauth2Credential("app", "AT-TYPE-CHANGE", "RT-TYPE-CHANGE"),
			})
			state := f.readState(t)
			drive := state["drive"].(map[string]any)
			drive["authType"] = authType
			patch := f.patch(t, map[string]any{"drive": drive})

			assertNoSecrets(t, "PATCH response", patch.Body.Bytes(), "AT-TYPE-CHANGE", "RT-TYPE-CHANGE")
			stored := f.stored(t, "drive").(map[string]any)
			if stored["authType"] != authType {
				t.Error("the changed authType was not stored")
			}
			for _, key := range []string{"accessToken", "refreshToken"} {
				if _, ok := stored[key]; ok {
					t.Errorf("the stored credential kept %q after the authType changed", key)
				}
			}
		})
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
		t.Fatalf("UpdateSessionHandler() status = %d, want %d", patch.Code, http.StatusOK)
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
	for i, item := range stored.([]any) {
		cred := item.(map[string]any)
		oauth := cred["oauth2"].(map[string]any)
		clientID := oauth["clientId"].(string)
		if got := oauth["accessToken"]; got != want[clientID] {
			t.Errorf("credential %d got the wrong access token", i)
		}
	}
}

// TestChangingClientIDDoesNotRestoreSecrets checks that a credential the
// client re-points at another app is treated as a replacement, not as the
// stored credential with a field missing.
func TestChangingClientIDDoesNotRestoreSecrets(t *testing.T) {
	f := newSessionFixture(t, map[string]any{"drive": oauth2Credential("app", "at", "rt")})
	state := f.readState(t)
	drive := state["drive"].(map[string]any)
	drive["oauth2"].(map[string]any)["clientId"] = "other"
	f.patch(t, map[string]any{"drive": drive})

	stored := f.stored(t, "drive").(map[string]any)
	oauth := stored["oauth2"].(map[string]any)
	if oauth["clientId"] != "other" {
		t.Error("the changed clientId was not stored")
	}
	if _, ok := oauth["accessToken"]; ok {
		t.Error("changing clientId restored the old access token")
	}
	if _, ok := oauth["refreshToken"]; ok {
		t.Error("changing clientId restored the old refresh token")
	}
}

// TestChangingAuthTypeDoesNotCarrySecretsAcross checks that an API key is not
// restored into a credential whose authType now names another scheme.
func TestChangingAuthTypeDoesNotCarrySecretsAcross(t *testing.T) {
	f := newSessionFixture(t, map[string]any{"drive": toolauth.Credential{
		AuthType: toolauth.CredentialTypeAPIKey,
		APIKey:   "key",
	}})
	state := f.readState(t)
	drive := state["drive"].(map[string]any)
	drive["authType"] = "http"
	patch := f.patch(t, map[string]any{"drive": drive})
	if bytes.Contains(patch.Body.Bytes(), []byte(`"apiKey"`)) {
		t.Error("PATCH response exposed the api key after the authType changed")
	}

	stored := f.stored(t, "drive").(map[string]any)
	if stored["authType"] != "http" {
		t.Error("the changed authType was not stored")
	}
	if _, ok := stored["apiKey"]; ok {
		t.Error("changing authType carried the api key into the http credential")
	}
}

// TestChangingRedirectURIKeepsTokens checks that a non-identifying field can
// change without losing the credential's secrets.
func TestChangingRedirectURIKeepsTokens(t *testing.T) {
	cred := toolauth.Credential{
		AuthType: toolauth.CredentialTypeOAuth2,
		OAuth2: &toolauth.OAuth2Auth{
			ClientID:     "app",
			RedirectURI:  "https://old.example/cb",
			AccessToken:  "at",
			RefreshToken: "rt",
		},
	}
	f := newSessionFixture(t, map[string]any{"drive": cred})
	state := f.readState(t)
	oauth := state["drive"].(map[string]any)["oauth2"].(map[string]any)
	oauth["redirectUri"] = "https://new.example/cb"
	f.patch(t, map[string]any{"drive": state["drive"]})

	stored := f.stored(t, "drive").(map[string]any)["oauth2"].(map[string]any)
	if stored["redirectUri"] != "https://new.example/cb" {
		t.Error("the edited redirectUri was not stored")
	}
	if stored["accessToken"] != "at" {
		t.Error("editing redirectUri erased the stored access token")
	}
	if stored["refreshToken"] != "rt" {
		t.Error("editing redirectUri erased the stored refresh token")
	}
}

// TestSwappingObjectCredentialsDoesNotRestoreSecrets checks that credentials
// moved between object keys do not receive the secrets of the key they landed
// on.
func TestSwappingObjectCredentialsDoesNotRestoreSecrets(t *testing.T) {
	f := newSessionFixture(t, map[string]any{"creds": map[string]any{
		"a": oauth2Credential("a", "at-a", "rt-a"),
		"b": oauth2Credential("b", "at-b", "rt-b"),
	}})
	state := f.readState(t)
	creds := state["creds"].(map[string]any)
	creds["a"], creds["b"] = creds["b"], creds["a"]
	f.patch(t, map[string]any{"creds": creds})

	stored := f.stored(t, "creds").(map[string]any)
	for key, wantClientID := range map[string]string{"a": "b", "b": "a"} {
		oauth := stored[key].(map[string]any)["oauth2"].(map[string]any)
		if oauth["clientId"] != wantClientID {
			t.Error("a swapped credential has the wrong clientId")
		}
		if _, ok := oauth["accessToken"]; ok {
			t.Error("swapping credentials restored a token")
		}
	}
}

// TestAppendingArrayCredentialKeepsExistingTokens checks that a longer array
// still matches the entries that were already there.
func TestAppendingArrayCredentialKeepsExistingTokens(t *testing.T) {
	f := newSessionFixture(t, map[string]any{"creds": []any{oauth2Credential("a", "at-a", "rt-a")}})
	state := f.readState(t)
	creds := state["creds"].([]any)
	creds = append(creds, map[string]any{
		"authType": "oauth2",
		"oauth2":   map[string]any{"clientId": "b"},
	})
	f.patch(t, map[string]any{"creds": creds})

	stored := f.stored(t, "creds").([]any)
	if len(stored) != 2 {
		t.Fatalf("stored %d credentials, want 2", len(stored))
	}
	byClient := map[string]map[string]any{}
	for _, item := range stored {
		oauth := item.(map[string]any)["oauth2"].(map[string]any)
		byClient[oauth["clientId"].(string)] = oauth
	}
	if byClient["a"]["accessToken"] != "at-a" {
		t.Error("appending a credential erased the existing access token")
	}
	if _, ok := byClient["b"]["accessToken"]; ok {
		t.Error("appending a credential restored a token for the new entry")
	}
}

// TestRemovingArrayCredentialKeepsRemainingTokens checks that a shorter array
// still matches the entries that remain.
func TestRemovingArrayCredentialKeepsRemainingTokens(t *testing.T) {
	f := newSessionFixture(t, map[string]any{"creds": []any{
		oauth2Credential("a", "at-a", "rt-a"),
		oauth2Credential("b", "at-b", "rt-b"),
	}})
	state := f.readState(t)
	creds := state["creds"].([]any)
	creds = creds[1:]
	f.patch(t, map[string]any{"creds": creds})

	stored := f.stored(t, "creds").([]any)
	if len(stored) != 1 {
		t.Fatalf("stored %d credentials, want 1", len(stored))
	}
	oauth := stored[0].(map[string]any)["oauth2"].(map[string]any)
	if oauth["clientId"] != "b" {
		t.Error("the wrong credential survived the removal")
	}
	if oauth["accessToken"] != "at-b" {
		t.Error("removing a credential erased the remaining access token")
	}
	if oauth["refreshToken"] != "rt-b" {
		t.Error("removing a credential erased the remaining refresh token")
	}
}

// TestDuplicatedArrayCredentialRestoresNothing checks that an identity that
// appears more than once in the submitted array cannot be matched, so no entry
// receives a secret.
func TestDuplicatedArrayCredentialRestoresNothing(t *testing.T) {
	f := newSessionFixture(t, map[string]any{"creds": []any{oauth2Credential("a", "at-a", "rt-a")}})
	state := f.readState(t)
	creds := state["creds"].([]any)
	creds = append(creds, map[string]any{
		"authType": "oauth2",
		"oauth2":   map[string]any{"clientId": "a"},
	})
	f.patch(t, map[string]any{"creds": creds})

	stored := f.stored(t, "creds").([]any)
	if len(stored) != 2 {
		t.Fatalf("stored %d credentials, want 2", len(stored))
	}
	for i, item := range stored {
		oauth := item.(map[string]any)["oauth2"].(map[string]any)
		if _, ok := oauth["accessToken"]; ok {
			t.Errorf("entry %d received an access token, want none", i)
		}
		if _, ok := oauth["refreshToken"]; ok {
			t.Errorf("entry %d received a refresh token, want none", i)
		}
	}
}

// TestUnknownAuthTypeSpellingDoesNotRestoreSecrets checks that the submitted
// object must still be recognized as a credential, even when its authType
// normalizes to the stored one.
func TestUnknownAuthTypeSpellingDoesNotRestoreSecrets(t *testing.T) {
	f := newSessionFixture(t, map[string]any{"drive": toolauth.Credential{
		AuthType: toolauth.CredentialTypeOpenIDConnect,
		OAuth2:   &toolauth.OAuth2Auth{ClientID: "app", AccessToken: "at", RefreshToken: "rt"},
	}})
	state := f.readState(t)
	drive := state["drive"].(map[string]any)
	drive["authType"] = "OpenIDConnect"
	patch := f.patch(t, map[string]any{"drive": drive})
	for _, secret := range []string{`"accessToken":"at"`, `"refreshToken":"rt"`} {
		if bytes.Contains(patch.Body.Bytes(), []byte(secret)) {
			t.Error("PATCH response exposed a credential secret after an unrecognized authType spelling")
		}
	}

	oauth := f.stored(t, "drive").(map[string]any)["oauth2"].(map[string]any)
	for _, key := range []string{"accessToken", "refreshToken"} {
		if _, ok := oauth[key]; ok {
			t.Error("the stored credential kept a secret after an unrecognized authType spelling")
		}
	}
}

// TestReplacingOneDuplicateIdentityRestoresNothing checks that replacing one of
// two credentials that share an identity leaves the other one as submitted:
// the identity is ambiguous, so nothing is restored, the same as on main.
func TestReplacingOneDuplicateIdentityRestoresNothing(t *testing.T) {
	f := newSessionFixture(t, map[string]any{"creds": []any{
		oauth2Credential("a", "at-1", "rt-1"),
		oauth2Credential("a", "at-2", "rt-2"),
	}})
	state := f.readState(t)
	creds := state["creds"].([]any)
	creds[0] = map[string]any{
		"authType": "oauth2",
		"oauth2":   map[string]any{"clientId": "b"},
	}
	f.patch(t, map[string]any{"creds": creds})

	stored := f.stored(t, "creds").([]any)
	if len(stored) != 2 {
		t.Fatalf("stored %d credentials, want 2", len(stored))
	}
	replacement := stored[0].(map[string]any)["oauth2"].(map[string]any)
	if replacement["clientId"] != "b" {
		t.Error("the replacement credential has the wrong clientId")
	}
	if _, ok := replacement["accessToken"]; ok {
		t.Error("the replacement credential received a token")
	}
	kept := stored[1].(map[string]any)["oauth2"].(map[string]any)
	if kept["clientId"] != "a" {
		t.Error("the untouched credential has the wrong clientId")
	}
	if _, ok := kept["accessToken"]; ok {
		t.Error("replacing a duplicate identity restored the other entry's access token")
	}
	if _, ok := kept["refreshToken"]; ok {
		t.Error("replacing a duplicate identity restored the other entry's refresh token")
	}
}

// TestRemovingFirstOfTwoLookalikeCredentialsRestoresNothing covers the case
// where the client deletes one of two entries that look the same once their
// secrets are hidden. Which stored entry the survivor is cannot be told from
// the payload, so nothing is restored into it.
func TestRemovingFirstOfTwoLookalikeCredentialsRestoresNothing(t *testing.T) {
	f := newSessionFixture(t, map[string]any{"creds": []any{
		oauth2CredentialWithRedirect("app", "https://one.example/cb", "at-1", "rt-1"),
		oauth2CredentialWithRedirect("app", "https://two.example/cb", "at-2", "rt-2"),
	}})
	state := f.readState(t)
	creds := state["creds"].([]any)
	if len(creds) != 2 {
		t.Fatalf("GET returned %d credentials, want 2", len(creds))
	}
	patch := f.patch(t, map[string]any{"creds": creds[1:]})

	stored := f.stored(t, "creds").([]any)
	if len(stored) != 1 {
		t.Fatalf("stored %d credentials, want 1", len(stored))
	}
	oauth := stored[0].(map[string]any)["oauth2"].(map[string]any)
	if oauth["redirectUri"] != "https://two.example/cb" {
		t.Error("the wrong credential survived the removal")
	}
	if _, ok := oauth["accessToken"]; ok {
		t.Error("the survivor received the removed entry's access token")
	}
	if _, ok := oauth["refreshToken"]; ok {
		t.Error("the survivor received the removed entry's refresh token")
	}
	encoded, err := json.Marshal(stored)
	if err != nil {
		t.Fatalf("Marshal(stored) failed: %v", err)
	}
	assertNoSecrets(t, "stored credentials", encoded, "at-1", "rt-1", "at-2", "rt-2")
	assertNoSecrets(t, "PATCH response", patch.Body.Bytes(), "at-1", "rt-1", "at-2", "rt-2")
}

// TestReorderingLookalikeCredentialsRestoresNothing covers two entries that
// share an identity: a reorder must not move either entry's secrets to the
// other one.
func TestReorderingLookalikeCredentialsRestoresNothing(t *testing.T) {
	f := newSessionFixture(t, map[string]any{"creds": []any{
		oauth2CredentialWithRedirect("app", "https://one.example/cb", "at-1", "rt-1"),
		oauth2CredentialWithRedirect("app", "https://two.example/cb", "at-2", "rt-2"),
	}})
	state := f.readState(t)
	creds := state["creds"].([]any)
	if len(creds) != 2 {
		t.Fatalf("GET returned %d credentials, want 2", len(creds))
	}
	creds[0], creds[1] = creds[1], creds[0]
	patch := f.patch(t, map[string]any{"creds": creds})

	stored := f.stored(t, "creds").([]any)
	if len(stored) != 2 {
		t.Fatalf("stored %d credentials, want 2", len(stored))
	}
	want := []string{"https://two.example/cb", "https://one.example/cb"}
	for i, item := range stored {
		oauth := item.(map[string]any)["oauth2"].(map[string]any)
		if oauth["redirectUri"] != want[i] {
			t.Errorf("credential %d has the wrong redirectUri after a reorder", i)
		}
		if _, ok := oauth["accessToken"]; ok {
			t.Errorf("credential %d received an access token after a reorder", i)
		}
	}
	encoded, err := json.Marshal(stored)
	if err != nil {
		t.Fatalf("Marshal(stored) failed: %v", err)
	}
	assertNoSecrets(t, "stored credentials", encoded, "at-1", "rt-1", "at-2", "rt-2")
	assertNoSecrets(t, "PATCH response", patch.Body.Bytes(), "at-1", "rt-1", "at-2", "rt-2")
}

// TestPrependingLookalikeCredentialRestoresNothing covers an entry inserted in
// front of two entries that share an identity: the shift must not hand either
// entry's secrets to its neighbor.
func TestPrependingLookalikeCredentialRestoresNothing(t *testing.T) {
	f := newSessionFixture(t, map[string]any{"creds": []any{
		oauth2CredentialWithRedirect("app", "https://one.example/cb", "at-1", "rt-1"),
		oauth2CredentialWithRedirect("app", "https://two.example/cb", "at-2", "rt-2"),
	}})
	state := f.readState(t)
	creds := state["creds"].([]any)
	if len(creds) != 2 {
		t.Fatalf("GET returned %d credentials, want 2", len(creds))
	}
	prepended := append([]any{map[string]any{
		"authType": "oauth2",
		"oauth2":   map[string]any{"clientId": "app", "redirectUri": "https://new.example/cb"},
	}}, creds...)
	patch := f.patch(t, map[string]any{"creds": prepended})

	stored := f.stored(t, "creds").([]any)
	if len(stored) != 3 {
		t.Fatalf("stored %d credentials, want 3", len(stored))
	}
	for i, item := range stored {
		oauth := item.(map[string]any)["oauth2"].(map[string]any)
		if _, ok := oauth["accessToken"]; ok {
			t.Errorf("credential %d received an access token after a prepend", i)
		}
	}
	encoded, err := json.Marshal(stored)
	if err != nil {
		t.Fatalf("Marshal(stored) failed: %v", err)
	}
	assertNoSecrets(t, "stored credentials", encoded, "at-1", "rt-1", "at-2", "rt-2")
	assertNoSecrets(t, "PATCH response", patch.Body.Bytes(), "at-1", "rt-1", "at-2", "rt-2")
}

// TestHTTPUsernameChangeDoesNotRestoreSecrets checks that changing the visible
// username of an HTTP credential is a replacement: the password of the previous
// user must not move to the new one.
func TestHTTPUsernameChangeDoesNotRestoreSecrets(t *testing.T) {
	f := newSessionFixture(t, map[string]any{"http": toolauth.Credential{
		AuthType: toolauth.CredentialTypeHTTP,
		HTTP: &toolauth.HTTPAuth{
			Scheme:      "basic",
			Credentials: toolauth.HTTPCredentials{Username: "alice", Password: "alice-password"},
		},
	}})
	state := f.readState(t)
	http := state["http"].(map[string]any)
	credentials := http["http"].(map[string]any)["credentials"].(map[string]any)
	credentials["username"] = "bob"
	patch := f.patch(t, map[string]any{"http": http})

	stored := f.stored(t, "http").(map[string]any)["http"].(map[string]any)["credentials"].(map[string]any)
	if stored["username"] != "bob" {
		t.Error("the changed username was not stored")
	}
	if _, ok := stored["password"]; ok {
		t.Error("bob received alice's password")
	}
	assertNoSecrets(t, "PATCH response", patch.Body.Bytes(), "alice-password")
}

// TestHTTPSchemeChangeDoesNotRestoreSecrets checks that changing the scheme of
// an HTTP credential is a replacement: a bearer token must not move into a
// basic credential.
func TestHTTPSchemeChangeDoesNotRestoreSecrets(t *testing.T) {
	f := newSessionFixture(t, map[string]any{"http": toolauth.Credential{
		AuthType: toolauth.CredentialTypeHTTP,
		HTTP: &toolauth.HTTPAuth{
			Scheme:      "bearer",
			Credentials: toolauth.HTTPCredentials{Username: "alice", Token: "bearer-token"},
		},
	}})
	state := f.readState(t)
	http := state["http"].(map[string]any)
	http["http"].(map[string]any)["scheme"] = "basic"
	patch := f.patch(t, map[string]any{"http": http})

	stored := f.stored(t, "http").(map[string]any)["http"].(map[string]any)
	if stored["scheme"] != "basic" {
		t.Error("the changed scheme was not stored")
	}
	credentials := stored["credentials"].(map[string]any)
	if _, ok := credentials["token"]; ok {
		t.Error("the basic credential received the bearer token")
	}
	assertNoSecrets(t, "PATCH response", patch.Body.Bytes(), "bearer-token")
}

// TestServiceAccountEmailChangeDoesNotRestoreSecrets checks that changing the
// visible client email of a service account is a replacement: the private key
// must not move to the new email.
func TestServiceAccountEmailChangeDoesNotRestoreSecrets(t *testing.T) {
	f := newSessionFixture(t, map[string]any{"sa": toolauth.Credential{
		AuthType: toolauth.CredentialTypeServiceAccount,
		ServiceAccount: &toolauth.ServiceAccount{
			ServiceAccountCredential: &toolauth.ServiceAccountCredential{
				ClientEmail: "alice@proj.iam.gserviceaccount.com",
				ProjectID:   "proj",
				PrivateKey:  "SA-PRIVATE-KEY",
			},
		},
	}})
	state := f.readState(t)
	sa := state["sa"].(map[string]any)
	credential := sa["serviceAccount"].(map[string]any)["serviceAccountCredential"].(map[string]any)
	credential["clientEmail"] = "bob@proj.iam.gserviceaccount.com"
	patch := f.patch(t, map[string]any{"sa": sa})

	stored := f.stored(t, "sa").(map[string]any)
	credential = stored["serviceAccount"].(map[string]any)["serviceAccountCredential"].(map[string]any)
	if credential["clientEmail"] != "bob@proj.iam.gserviceaccount.com" {
		t.Error("the changed client email was not stored")
	}
	if _, ok := credential["privateKey"]; ok {
		t.Error("bob received alice's private key")
	}
	assertNoSecrets(t, "PATCH response", patch.Body.Bytes(), "SA-PRIVATE-KEY")
}

// TestSwappingDuplicateIdentitiesUnderObjectKeysRestoresNothing checks that two
// credentials with the same identity under different object keys do not receive
// each other's secrets when they are swapped.
func TestSwappingDuplicateIdentitiesUnderObjectKeysRestoresNothing(t *testing.T) {
	f := newSessionFixture(t, map[string]any{"creds": map[string]any{
		"one": oauth2CredentialWithRedirect("app", "https://one.example/cb", "at-1", "rt-1"),
		"two": oauth2CredentialWithRedirect("app", "https://two.example/cb", "at-2", "rt-2"),
	}})
	state := f.readState(t)
	creds := state["creds"].(map[string]any)
	creds["one"], creds["two"] = creds["two"], creds["one"]
	patch := f.patch(t, map[string]any{"creds": creds})

	stored := f.stored(t, "creds").(map[string]any)
	for key, wantRedirect := range map[string]string{
		"one": "https://two.example/cb",
		"two": "https://one.example/cb",
	} {
		oauth := stored[key].(map[string]any)["oauth2"].(map[string]any)
		if oauth["redirectUri"] != wantRedirect {
			t.Error("a swapped credential has the wrong redirectUri")
		}
		if _, ok := oauth["accessToken"]; ok {
			t.Error("a swapped credential received an access token")
		}
	}
	assertNoSecrets(t, "PATCH response", patch.Body.Bytes(), "at-1", "rt-1", "at-2", "rt-2")
}

// assertNoSecrets checks that encoded, a JSON document, does not contain any of
// secrets.
func assertNoSecrets(t *testing.T, what string, encoded []byte, secrets ...string) {
	t.Helper()
	for _, secret := range secrets {
		if bytes.Contains(encoded, []byte(secret)) {
			t.Errorf("%s carries a secret, want it hidden", what)
		}
	}
}

// oauth2CredentialWithRedirect returns a credential that another one can share
// an identity with while a visible field tells them apart.
func oauth2CredentialWithRedirect(clientID, redirectURI, accessToken, refreshToken string) toolauth.Credential {
	return toolauth.Credential{
		AuthType: toolauth.CredentialTypeOAuth2,
		OAuth2: &toolauth.OAuth2Auth{
			ClientID:     clientID,
			RedirectURI:  redirectURI,
			AccessToken:  accessToken,
			RefreshToken: refreshToken,
		},
	}
}

func oauth2Credential(clientID, accessToken, refreshToken string) toolauth.Credential {
	return toolauth.Credential{
		AuthType: toolauth.CredentialTypeOAuth2,
		OAuth2: &toolauth.OAuth2Auth{
			ClientID:     clientID,
			AccessToken:  accessToken,
			RefreshToken: refreshToken,
		},
	}
}

type sessionFixture struct {
	service    session.Service
	controller *controllers.SessionsAPIController
	ctx        context.Context
	vars       map[string]string
}

func newSessionFixture(t *testing.T, state map[string]any) *sessionFixture {
	t.Helper()
	service := session.InMemoryService()
	created, err := service.Create(t.Context(), &session.CreateRequest{
		AppName: "app",
		UserID:  "user",
		State:   state,
	})
	if err != nil {
		t.Fatalf("Create() failed: %v", err)
	}
	return &sessionFixture{
		service:    service,
		controller: controllers.NewSessionsAPIController(service),
		ctx:        authn.WithCaller(t.Context(), &authn.Caller{UserID: "user"}),
		vars: map[string]string{
			"app_name":   "app",
			"user_id":    "user",
			"session_id": created.Session.ID(),
		},
	}
}

func (f *sessionFixture) readState(t *testing.T) map[string]any {
	t.Helper()
	get := httptest.NewRecorder()
	f.controller.GetSessionHandler(get, mux.SetURLVars(httptest.NewRequestWithContext(f.ctx, http.MethodGet, "/", nil), f.vars))
	if get.Code != http.StatusOK {
		t.Fatalf("GetSessionHandler() status = %d, want %d", get.Code, http.StatusOK)
	}
	var read struct {
		State map[string]any `json:"state"`
	}
	if err := json.Unmarshal(get.Body.Bytes(), &read); err != nil {
		t.Fatalf("Unmarshal(GET) failed: %v", err)
	}
	return read.State
}

func (f *sessionFixture) patch(t *testing.T, delta map[string]any) *httptest.ResponseRecorder {
	t.Helper()
	body, err := json.Marshal(map[string]any{"stateDelta": delta})
	if err != nil {
		t.Fatalf("Marshal(PATCH) failed: %v", err)
	}
	patch := httptest.NewRecorder()
	f.controller.UpdateSessionHandler(patch, mux.SetURLVars(httptest.NewRequestWithContext(f.ctx, http.MethodPatch, "/", bytes.NewReader(body)), f.vars))
	if patch.Code != http.StatusOK {
		t.Fatalf("UpdateSessionHandler() status = %d, want %d", patch.Code, http.StatusOK)
	}
	return patch
}

func (f *sessionFixture) stored(t *testing.T, key string) any {
	t.Helper()
	reread, err := f.service.Get(t.Context(), &session.GetRequest{
		AppName:   "app",
		UserID:    "user",
		SessionID: f.vars["session_id"],
	})
	if err != nil {
		t.Fatalf("Get() failed: %v", err)
	}
	value, err := reread.Session.State().Get(key)
	if err != nil {
		t.Fatalf("State().Get(%s) failed: %v", key, err)
	}
	return value
}
