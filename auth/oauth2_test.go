// Copyright 2025 Google LLC
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

package auth

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestOAuth2Exchanger_ClientCredentialsFlow(t *testing.T) {
	t.Parallel()

	var gotAuthHeader string
	var gotAudience string

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := r.ParseForm(); err != nil {
			t.Fatalf("ParseForm() error = %v", err)
		}
		if grant := r.FormValue("grant_type"); grant != "client_credentials" {
			t.Fatalf("grant_type = %s, want client_credentials", grant)
		}
		gotAudience = r.FormValue("audience")
		gotAuthHeader = r.Header.Get("Authorization")
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"access_token":"token123","token_type":"bearer","expires_in":60}`)
	}))
	defer server.Close()

	scheme := &OAuth2Scheme{
		Flows: &OAuthFlows{
			ClientCredentials: &OAuthFlowClientCredentials{
				TokenURL: server.URL,
				Scopes:   map[string]string{"read": "Read access"},
			},
		},
	}
	cred := &AuthCredential{
		AuthType: AuthCredentialTypeOAuth2,
		OAuth2: &OAuth2Auth{
			ClientID:     "client-id",
			ClientSecret: "client-secret",
			Audience:     "https://api.example.com",
		},
	}

	ex := NewOAuth2Exchanger()
	res, err := ex.exchangeClientCredentials(context.Background(), cred, scheme)
	if err != nil {
		t.Fatalf("exchangeClientCredentials() error = %v", err)
	}
	if res.Credential.OAuth2.AccessToken != "token123" {
		t.Fatalf("access token = %s, want token123", res.Credential.OAuth2.AccessToken)
	}
	if gotAudience != "https://api.example.com" {
		t.Fatalf("audience = %s, want https://api.example.com", gotAudience)
	}
	if gotAuthHeader == "" || !strings.HasPrefix(gotAuthHeader, "Basic ") {
		t.Fatalf("authorization header = %s, want HTTP Basic credentials", gotAuthHeader)
	}
}

func TestOAuth2Exchanger_ClientCredentials_ClientSecretPost(t *testing.T) {
	t.Parallel()

	var gotAuthHeader string
	var clientID string
	var clientSecret string

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := r.ParseForm(); err != nil {
			t.Fatalf("ParseForm() error = %v", err)
		}
		clientID = r.FormValue("client_id")
		clientSecret = r.FormValue("client_secret")
		gotAuthHeader = r.Header.Get("Authorization")
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"access_token":"token456","token_type":"bearer","expires_in":60}`)
	}))
	defer server.Close()

	scheme := &OAuth2Scheme{
		Flows: &OAuthFlows{
			ClientCredentials: &OAuthFlowClientCredentials{
				TokenURL: server.URL,
			},
		},
	}
	cred := &AuthCredential{
		AuthType: AuthCredentialTypeOAuth2,
		OAuth2: &OAuth2Auth{
			ClientID:                "client-id",
			ClientSecret:            "client-secret",
			TokenEndpointAuthMethod: "client_secret_post",
		},
	}

	ex := NewOAuth2Exchanger()
	res, err := ex.exchangeClientCredentials(context.Background(), cred, scheme)
	if err != nil {
		t.Fatalf("exchangeClientCredentials() error = %v", err)
	}
	if res.Credential.OAuth2.AccessToken != "token456" {
		t.Fatalf("access token = %s, want token456", res.Credential.OAuth2.AccessToken)
	}
	if gotAuthHeader != "" {
		t.Fatalf("authorization header = %s, want empty for client_secret_post", gotAuthHeader)
	}
	if clientID != "client-id" || clientSecret != "client-secret" {
		t.Fatalf("client credentials were not sent in body")
	}
}

func TestOAuth2Exchanger_AuthorizationCode_OpenID(t *testing.T) {
	t.Parallel()

	var receivedCode string

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := r.ParseForm(); err != nil {
			t.Fatalf("ParseForm() error = %v", err)
		}
		receivedCode = r.FormValue("code")
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"access_token":"oidc-access","refresh_token":"oidc-refresh","token_type":"bearer","expires_in":120}`)
	}))
	defer server.Close()

	scheme := &OpenIDConnectScheme{
		AuthorizationEndpoint: "https://example.com/oauth2/authorize",
		TokenEndpoint:         server.URL,
		Scopes:                []string{"openid"},
	}
	cred := &AuthCredential{
		AuthType: AuthCredentialTypeOpenIDConnect,
		OAuth2: &OAuth2Auth{
			ClientID:     "client-id",
			ClientSecret: "client-secret",
			RedirectURI:  "https://localhost/callback",
			AuthCode:     "auth-code",
		},
	}

	ex := NewOAuth2Exchanger()
	res, err := ex.exchangeAuthorizationCode(context.Background(), cred, scheme)
	if err != nil {
		t.Fatalf("exchangeAuthorizationCode() error = %v", err)
	}
	if receivedCode != "auth-code" {
		t.Fatalf("code = %s, want auth-code", receivedCode)
	}
	if res.Credential.OAuth2.AccessToken != "oidc-access" {
		t.Fatalf("access token = %s, want oidc-access", res.Credential.OAuth2.AccessToken)
	}
	if res.Credential.OAuth2.RefreshToken != "oidc-refresh" {
		t.Fatalf("refresh token = %s, want oidc-refresh", res.Credential.OAuth2.RefreshToken)
	}
	if res.Credential.OAuth2.AuthCode != "" {
		t.Fatalf("auth code was not cleared")
	}
}

func TestOAuth2Refresher_OpenIDConnect(t *testing.T) {
	t.Parallel()

	var grantType string

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := r.ParseForm(); err != nil {
			t.Fatalf("ParseForm() error = %v", err)
		}
		grantType = r.FormValue("grant_type")
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"access_token":"new-token","refresh_token":"new-refresh","token_type":"bearer","expires_in":60}`)
	}))
	defer server.Close()

	scheme := &OpenIDConnectScheme{
		TokenEndpoint: server.URL,
	}
	cred := &AuthCredential{
		AuthType: AuthCredentialTypeOpenIDConnect,
		OAuth2: &OAuth2Auth{
			ClientID:     "client-id",
			ClientSecret: "client-secret",
			AccessToken:  "old-token",
			RefreshToken: "old-refresh",
			ExpiresAt:    time.Now().Add(-time.Minute).Unix(),
		},
	}

	refresher := NewOAuth2Refresher()
	newCred, err := refresher.Refresh(context.Background(), cred, scheme)
	if err != nil {
		t.Fatalf("Refresh() error = %v", err)
	}
	if grantType != "refresh_token" {
		t.Fatalf("grant_type = %s, want refresh_token", grantType)
	}
	if newCred.OAuth2.AccessToken != "new-token" {
		t.Fatalf("access token = %s, want new-token", newCred.OAuth2.AccessToken)
	}
	if newCred.OAuth2.RefreshToken != "new-refresh" {
		t.Fatalf("refresh token = %s, want new-refresh", newCred.OAuth2.RefreshToken)
	}
}
