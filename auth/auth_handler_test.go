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
	"crypto/rand"
	"errors"
	"net/url"
	"strings"
	"testing"
)

func TestNewAuthHandler(t *testing.T) {
	cfg := &AuthConfig{
		AuthScheme: &OAuth2Scheme{},
	}
	handler := NewAuthHandler(cfg)
	if handler == nil {
		t.Error("NewAuthHandler returned nil")
	}
}

func TestAuthHandler_GenerateAuthRequest_NonOAuth(t *testing.T) {
	// For non-OAuth schemes, should return a copy as-is
	cfg := &AuthConfig{
		AuthScheme: &APIKeyScheme{
			In:   APIKeyInHeader,
			Name: "X-API-Key",
		},
		RawAuthCredential: &AuthCredential{
			AuthType: AuthCredentialTypeAPIKey,
			APIKey:   "my-key",
		},
	}
	handler := NewAuthHandler(cfg)
	result := handler.GenerateAuthRequest()

	if result == cfg {
		t.Error("GenerateAuthRequest should return a copy, not same pointer")
	}
	if result.RawAuthCredential.APIKey != "my-key" {
		t.Error("API key should be preserved")
	}
}

func TestAuthHandler_GenerateAuthRequest_OAuth2_NoCredential(t *testing.T) {
	cfg := &AuthConfig{
		AuthScheme: &OAuth2Scheme{
			Flows: &OAuthFlows{
				AuthorizationCode: &OAuthFlowAuthorizationCode{
					AuthorizationURL: "https://example.com/auth",
					TokenURL:         "https://example.com/token",
				},
			},
		},
		// No RawAuthCredential
	}
	handler := NewAuthHandler(cfg)
	result := handler.GenerateAuthRequest()

	// Should return copy without generating auth URI
	if result.ExchangedAuthCredential != nil && result.ExchangedAuthCredential.OAuth2 != nil &&
		result.ExchangedAuthCredential.OAuth2.AuthURI != "" {
		t.Error("Should not generate auth URI without raw credential")
	}
}

func TestAuthHandler_GenerateAuthRequest_OAuth2_WithCredential(t *testing.T) {
	cfg := &AuthConfig{
		AuthScheme: &OAuth2Scheme{
			Flows: &OAuthFlows{
				AuthorizationCode: &OAuthFlowAuthorizationCode{
					AuthorizationURL: "https://example.com/auth",
					TokenURL:         "https://example.com/token",
					Scopes:           map[string]string{"read": "Read access"},
				},
			},
		},
		RawAuthCredential: &AuthCredential{
			AuthType: AuthCredentialTypeOAuth2,
			OAuth2: &OAuth2Auth{
				ClientID:     "client-id",
				ClientSecret: "client-secret",
				RedirectURI:  "https://localhost/callback",
			},
		},
	}
	handler := NewAuthHandler(cfg)
	result := handler.GenerateAuthRequest()

	if result.ExchangedAuthCredential == nil {
		t.Fatal("ExchangedAuthCredential should be set")
	}
	if result.ExchangedAuthCredential.OAuth2 == nil {
		t.Fatal("ExchangedAuthCredential.OAuth2 should be set")
	}
	if result.ExchangedAuthCredential.OAuth2.AuthURI == "" {
		t.Error("AuthURI should be generated")
	}
	if !strings.Contains(result.ExchangedAuthCredential.OAuth2.AuthURI, "https://example.com/auth") {
		t.Errorf("AuthURI = %q, should contain authorization URL", result.ExchangedAuthCredential.OAuth2.AuthURI)
	}
	if result.ExchangedAuthCredential.OAuth2.State == "" {
		t.Error("State should be generated for CSRF protection")
	}
}

func TestAuthHandler_GenerateAuthRequest_OAuth2_ExistingAuthURI(t *testing.T) {
	// If auth_uri already exists in exchanged credential, return as-is
	existingAuthURI := "https://example.com/existing-auth"
	cfg := &AuthConfig{
		AuthScheme: &OAuth2Scheme{
			Flows: &OAuthFlows{
				AuthorizationCode: &OAuthFlowAuthorizationCode{
					AuthorizationURL: "https://example.com/auth",
					TokenURL:         "https://example.com/token",
				},
			},
		},
		RawAuthCredential: &AuthCredential{
			AuthType: AuthCredentialTypeOAuth2,
			OAuth2: &OAuth2Auth{
				ClientID: "client-id",
			},
		},
		ExchangedAuthCredential: &AuthCredential{
			AuthType: AuthCredentialTypeOAuth2,
			OAuth2: &OAuth2Auth{
				AuthURI: existingAuthURI,
			},
		},
	}
	handler := NewAuthHandler(cfg)
	result := handler.GenerateAuthRequest()

	if result.ExchangedAuthCredential.OAuth2.AuthURI != existingAuthURI {
		t.Errorf("AuthURI = %q, want %q (should preserve existing)", result.ExchangedAuthCredential.OAuth2.AuthURI, existingAuthURI)
	}
}

func TestAuthHandler_GenerateAuthURI_OpenIDConnect(t *testing.T) {
	t.Cleanup(func() { randRead = rand.Read })
	randRead = func(b []byte) (int, error) {
		for i := range b {
			b[i] = byte(i)
		}
		return len(b), nil
	}

	cfg := &AuthConfig{
		AuthScheme: &OpenIDConnectScheme{
			AuthorizationEndpoint: "https://example.com/oauth2/authorize",
			TokenEndpoint:         "https://example.com/oauth2/token",
			Scopes:                []string{"openid", "profile"},
		},
		RawAuthCredential: &AuthCredential{
			AuthType: AuthCredentialTypeOpenIDConnect,
			OAuth2: &OAuth2Auth{
				ClientID:     "client-id",
				ClientSecret: "client-secret",
				RedirectURI:  "https://localhost/callback",
			},
		},
	}

	handler := NewAuthHandler(cfg)
	cred := handler.generateAuthURI()
	if cred == nil || cred.OAuth2 == nil {
		t.Fatal("generateAuthURI() returned nil")
	}

	parsed, err := url.Parse(cred.OAuth2.AuthURI)
	if err != nil {
		t.Fatalf("parse auth URI: %v", err)
	}
	q := parsed.Query()

	if got := q.Get("client_id"); got != "client-id" {
		t.Fatalf("client_id = %s, want client-id", got)
	}
	if got := q.Get("scope"); got != "openid profile" {
		t.Fatalf("scope = %s, want 'openid profile'", got)
	}
	if got := q.Get("access_type"); got != "offline" {
		t.Fatalf("access_type = %s, want offline", got)
	}
	if got := q.Get("state"); got != "000102030405060708090a0b0c0d0e0f" {
		t.Fatalf("state = %s, want deterministic hex value", got)
	}
}

func TestAuthHandler_GenerateAuthURI_IncludesAudience(t *testing.T) {
	t.Cleanup(func() { randRead = rand.Read })
	randRead = func(b []byte) (int, error) {
		for i := range b {
			b[i] = 0xAB
		}
		return len(b), nil
	}

	cfg := &AuthConfig{
		AuthScheme: &OAuth2Scheme{
			Flows: &OAuthFlows{
				AuthorizationCode: &OAuthFlowAuthorizationCode{
					AuthorizationURL: "https://example.com/oauth2/authorize",
					TokenURL:         "https://example.com/oauth2/token",
					Scopes:           map[string]string{"read": "Read access"},
				},
			},
		},
		RawAuthCredential: &AuthCredential{
			AuthType: AuthCredentialTypeOAuth2,
			OAuth2: &OAuth2Auth{
				ClientID:     "client-id",
				ClientSecret: "client-secret",
				RedirectURI:  "https://localhost/callback",
				Audience:     "https://example.com/audience",
			},
		},
	}

	handler := NewAuthHandler(cfg)
	cred := handler.generateAuthURI()
	if cred == nil {
		t.Fatal("generateAuthURI() returned nil")
	}

	parsed, err := url.Parse(cred.OAuth2.AuthURI)
	if err != nil {
		t.Fatalf("parse auth URI: %v", err)
	}
	if got := parsed.Query().Get("audience"); got != "https://example.com/audience" {
		t.Fatalf("audience = %s, want https://example.com/audience", got)
	}
}

func TestGenerateRandomState_PanicsOnError(t *testing.T) {
	t.Cleanup(func() { randRead = rand.Read })
	randRead = func([]byte) (int, error) {
		return 0, errors.New("entropy exhausted")
	}

	defer func() {
		if r := recover(); r == nil {
			t.Fatal("generateRandomState() did not panic on rand failure")
		}
	}()
	_ = generateRandomState()
}

func TestAuthHandler_GetAuthResponse(t *testing.T) {
	cfg := &AuthConfig{
		CredentialKey: "test-key",
	}
	handler := NewAuthHandler(cfg)

	expectedCred := &AuthCredential{
		AuthType: AuthCredentialTypeOAuth2,
		OAuth2: &OAuth2Auth{
			AccessToken: "token123",
		},
	}

	stateGetter := func(key string) interface{} {
		if key == "temp:test-key" {
			return expectedCred
		}
		return nil
	}

	result := handler.GetAuthResponse(stateGetter)
	if result != expectedCred {
		t.Errorf("GetAuthResponse() = %v, want %v", result, expectedCred)
	}
}

func TestAuthHandler_GetAuthResponse_NotFound(t *testing.T) {
	cfg := &AuthConfig{
		CredentialKey: "test-key",
	}
	handler := NewAuthHandler(cfg)

	stateGetter := func(key string) interface{} {
		return nil
	}

	result := handler.GetAuthResponse(stateGetter)
	if result != nil {
		t.Errorf("GetAuthResponse() = %v, want nil", result)
	}
}
