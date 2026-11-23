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

package authhandler_test

import (
	"errors"
	"testing"

	"github.com/google/go-cmp/cmp"

	"google.golang.org/adk/v2/internal/authhandler"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool/toolauth"
)

func newState(t *testing.T) session.State {
	t.Helper()
	resp, err := session.InMemoryService().Create(t.Context(), &session.CreateRequest{AppName: "app", UserID: "user"})
	if err != nil {
		t.Fatalf("Create() error = %v", err)
	}
	return resp.Session.State()
}

// oauthRequest is the request ADK kept in the session for one sign-in.
func oauthRequest() *toolauth.Config {
	return &toolauth.Config{
		AuthScheme: toolauth.Scheme{
			Type: toolauth.SchemeTypeOAuth2,
			Flows: &toolauth.OAuthFlows{AuthorizationCode: &toolauth.OAuthFlow{
				AuthorizationURL: "https://idp.example/authorize",
				TokenURL:         "https://idp.example/token",
				Scopes:           map[string]string{},
			}},
		},
		RawAuthCredential: &toolauth.Credential{
			AuthType: toolauth.CredentialTypeOAuth2,
			OAuth2:   &toolauth.OAuth2Auth{ClientID: "app-123"},
		},
		ExchangedAuthCredential: &toolauth.Credential{
			AuthType: toolauth.CredentialTypeOAuth2,
			OAuth2: &toolauth.OAuth2Auth{
				ClientID:            "app-123",
				State:               "Xy7",
				CodeVerifier:        "v-secret",
				CodeChallengeMethod: "S256",
			},
		},
		CredentialKey: "adk_oauth",
	}
}

// oauthAnswer answers oauthRequest naming another client and sending a secret.
func oauthAnswer() map[string]any {
	return map[string]any{
		"authScheme": map[string]any{"type": "apiKey", "in": "header", "name": "ignored"},
		"exchangedAuthCredential": map[string]any{
			"authType": "oauth2",
			"oauth2": map[string]any{
				"clientId":        "other-app",
				"clientSecret":    "client-sent",
				"state":           "Xy7",
				"authResponseUri": "https://adk.example/cb?state=Xy7&code=4/0Ab",
				"redirectUri":     "https://adk.example/cb",
			},
		},
	}
}

func TestStoreAuthResponse(t *testing.T) {
	tests := []struct {
		name      string
		requested *toolauth.Config
		resp      map[string]any
		wantKey   string
		want      toolauth.Credential
	}{
		{
			name:      "oauth2 answer is pinned to the configured client and loses its secret",
			requested: oauthRequest(),
			resp:      oauthAnswer(),
			wantKey:   "adk_oauth",
			want: toolauth.Credential{
				AuthType: toolauth.CredentialTypeOAuth2,
				OAuth2: &toolauth.OAuth2Auth{
					ClientID:            "app-123",
					State:               "Xy7",
					AuthResponseURI:     "https://adk.example/cb?state=Xy7&code=4/0Ab",
					RedirectURI:         "https://adk.example/cb",
					CodeVerifier:        "v-secret",
					CodeChallengeMethod: "S256",
				},
			},
		},
		{
			name: "api key answer is stored as it came",
			requested: &toolauth.Config{
				AuthScheme:        toolauth.Scheme{Type: toolauth.SchemeTypeAPIKey, In: "header", Name: "X-Api-Key"},
				RawAuthCredential: &toolauth.Credential{AuthType: toolauth.CredentialTypeAPIKey},
				CredentialKey:     "adk_api",
			},
			resp:    map[string]any{"authType": "apiKey", "apiKey": "user-key"},
			wantKey: "adk_api",
			want:    toolauth.Credential{AuthType: toolauth.CredentialTypeAPIKey, APIKey: "user-key"},
		},
		{
			name: "oauth2 answer to a request without configured oauth2 keeps its client ID",
			requested: &toolauth.Config{
				AuthScheme:        toolauth.Scheme{Type: toolauth.SchemeTypeAPIKey, In: "header", Name: "X-Api-Key"},
				RawAuthCredential: &toolauth.Credential{AuthType: toolauth.CredentialTypeAPIKey},
				CredentialKey:     "adk_api",
			},
			resp: map[string]any{
				"authType": "oauth2",
				"oauth2":   map[string]any{"clientId": "other-app", "clientSecret": "client-sent", "authCode": "4/0Ab"},
			},
			wantKey: "adk_api",
			want: toolauth.Credential{
				AuthType: toolauth.CredentialTypeOAuth2,
				OAuth2:   &toolauth.OAuth2Auth{ClientID: "other-app", AuthCode: "4/0Ab"},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			state := newState(t)
			key, err := authhandler.StoreAuthResponse(state, tt.requested, tt.resp)
			if err != nil {
				t.Fatalf("StoreAuthResponse() error = %v", err)
			}
			if key != tt.wantKey {
				t.Errorf("StoreAuthResponse() key = %q, want %q", key, tt.wantKey)
			}
			got, err := state.Get("temp:" + tt.wantKey)
			if err != nil {
				t.Fatalf("state.Get(temp key) error = %v", err)
			}
			if diff := cmp.Diff(tt.want, got); diff != "" {
				t.Errorf("stored credential mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// adk-python's parse_and_store_auth_response writes only the temp key, so a
// credential an app keeps under the bare key survives a sign-in.
func TestStoreAuthResponseLeavesTheBareKeyAlone(t *testing.T) {
	state := newState(t)
	if err := state.Set("adk_oauth", "old-token"); err != nil {
		t.Fatalf("state.Set() error = %v", err)
	}
	if _, err := authhandler.StoreAuthResponse(state, oauthRequest(), oauthAnswer()); err != nil {
		t.Fatalf("StoreAuthResponse() error = %v", err)
	}
	if got, err := state.Get("adk_oauth"); err != nil || got != "old-token" {
		t.Errorf("state.Get(bare key) = %v, %v; want the old value kept", got, err)
	}
}

func TestStoreAuthResponseUnreadableAnswerStoresNothing(t *testing.T) {
	state := newState(t)
	if _, err := authhandler.StoreAuthResponse(state, oauthRequest(), map[string]any{"result": "ok"}); err == nil {
		t.Fatalf("StoreAuthResponse() error = nil, want an error")
	}
	if _, err := state.Get("temp:adk_oauth"); !errors.Is(err, session.ErrStateKeyNotExist) {
		t.Errorf("state.Get(temp key) error = %v, want ErrStateKeyNotExist", err)
	}
}
