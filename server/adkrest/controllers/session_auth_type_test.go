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
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gorilla/mux"

	"google.golang.org/adk/v2/server/adkrest/controllers"
	"google.golang.org/adk/v2/server/authn"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool/toolauth"
)

type authTypeFixture struct {
	service    session.Service
	controller *controllers.SessionsAPIController
	ctx        context.Context
	vars       map[string]string
}

func newAuthTypeFixture(t *testing.T, state map[string]any) *authTypeFixture {
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
	return &authTypeFixture{
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

func (f *authTypeFixture) get(t *testing.T) *httptest.ResponseRecorder {
	t.Helper()
	rec := httptest.NewRecorder()
	f.controller.GetSessionHandler(rec, mux.SetURLVars(httptest.NewRequestWithContext(f.ctx, http.MethodGet, "/", nil), f.vars))
	if rec.Code != http.StatusOK {
		t.Fatalf("GetSessionHandler() status = %d, want %d", rec.Code, http.StatusOK)
	}
	return rec
}

func (f *authTypeFixture) patch(t *testing.T, body string) *httptest.ResponseRecorder {
	t.Helper()
	rec := httptest.NewRecorder()
	f.controller.UpdateSessionHandler(rec, mux.SetURLVars(httptest.NewRequestWithContext(
		f.ctx, http.MethodPatch, "/", strings.NewReader(body)), f.vars))
	return rec
}

func (f *authTypeFixture) stored(t *testing.T, key string) any {
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

func TestChangingTopLevelAuthTypeToNonCredentialIsRejected(t *testing.T) {
	f := newAuthTypeFixture(t, map[string]any{
		"authType": "oauth2",
		"oauth2": map[string]any{
			"clientId":     "app",
			"accessToken":  "at",
			"refreshToken": "rt",
		},
		"profile": map[string]any{"name": "ann", "token": "page-2"},
	})
	if body := f.get(t).Body.String(); strings.Contains(body, `"accessToken":"at"`) || strings.Contains(body, `"token":"page-2"`) {
		t.Fatal("GET exposed a credential secret")
	}

	patch := f.patch(t, `{"stateDelta":{"authType":"none"}}`)
	if patch.Code != http.StatusBadRequest {
		t.Fatalf("UpdateSessionHandler() status = %d, want %d", patch.Code, http.StatusBadRequest)
	}
	if body := patch.Body.String(); strings.Contains(body, `"accessToken":"at"`) || strings.Contains(body, `"token":"page-2"`) {
		t.Fatal("rejected PATCH exposed a credential secret")
	}
	if body := f.get(t).Body.String(); strings.Contains(body, `"accessToken":"at"`) || strings.Contains(body, `"token":"page-2"`) {
		t.Fatal("GET after rejected PATCH exposed a credential secret")
	}

	storedOAuth := f.stored(t, "oauth2")
	oauth, ok := storedOAuth.(map[string]any)
	if !ok {
		t.Fatalf("stored oauth2 has type %T, want map[string]any", storedOAuth)
	}
	if oauth["accessToken"] != "at" {
		t.Error("the rejected PATCH erased the stored access token")
	}
}

func TestChangingSnakeCaseAuthTypeToNonCredentialIsRejected(t *testing.T) {
	f := newAuthTypeFixture(t, map[string]any{
		"auth_type": "oauth2",
		"oauth2": map[string]any{
			"client_id":     "app",
			"access_token":  "at",
			"refresh_token": "rt",
		},
	})

	patch := f.patch(t, `{"stateDelta":{"auth_type":"none"}}`)
	if patch.Code != http.StatusBadRequest {
		t.Fatalf("UpdateSessionHandler() status = %d, want %d", patch.Code, http.StatusBadRequest)
	}
	storedOAuth := f.stored(t, "oauth2")
	oauth, ok := storedOAuth.(map[string]any)
	if !ok {
		t.Fatalf("stored oauth2 has type %T, want map[string]any", storedOAuth)
	}
	if oauth["access_token"] != "at" {
		t.Error("the rejected PATCH erased the stored access token")
	}
}

func TestChangingTypedTopLevelAuthTypeToNonCredentialIsRejected(t *testing.T) {
	f := newAuthTypeFixture(t, map[string]any{
		"authType": toolauth.CredentialTypeOAuth2,
		"oauth2": &toolauth.OAuth2Auth{
			ClientID:     "app",
			AccessToken:  "at",
			RefreshToken: "rt",
		},
	})

	patch := f.patch(t, `{"stateDelta":{"authType":"none"}}`)
	if patch.Code != http.StatusBadRequest {
		t.Fatalf("UpdateSessionHandler() status = %d, want %d", patch.Code, http.StatusBadRequest)
	}
}

func TestChangingTopLevelAuthTypeToAnotherCredentialIsAllowed(t *testing.T) {
	f := newAuthTypeFixture(t, map[string]any{
		"authType": "oauth2",
		"oauth2": map[string]any{
			"clientId":     "app",
			"accessToken":  "at",
			"refreshToken": "rt",
		},
		"profile": map[string]any{"name": "ann", "token": "page-2"},
	})

	patch := f.patch(t, `{"stateDelta":{"authType":"http"}}`)
	if patch.Code != http.StatusOK {
		t.Fatalf("UpdateSessionHandler() status = %d, want %d", patch.Code, http.StatusOK)
	}
	if body := patch.Body.String(); strings.Contains(body, `"accessToken":"at"`) || strings.Contains(body, `"token":"page-2"`) {
		t.Fatal("PATCH response exposed a credential secret after a recognized authType change")
	}
	if got := f.stored(t, "authType"); got != "http" {
		t.Errorf("stored authType = %v, want http", got)
	}
}
