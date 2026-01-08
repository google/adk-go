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

package llminternal

import (
	"testing"

	"google.golang.org/adk/auth"
	contextinternal "google.golang.org/adk/internal/context"
	"google.golang.org/adk/session"
)

func TestGenerateAuthEvent_Nil(t *testing.T) {
	inv := contextinternal.NewInvocationContext(t.Context(), contextinternal.InvocationContextParams{})

	// Nil event
	result := GenerateAuthEvent(inv, nil)
	if result != nil {
		t.Error("GenerateAuthEvent(nil) should return nil")
	}

	// Empty RequestedAuthConfigs
	event := &session.Event{
		Actions: session.EventActions{
			RequestedAuthConfigs: make(map[string]*auth.AuthConfig),
		},
	}
	result = GenerateAuthEvent(inv, event)
	if result != nil {
		t.Error("GenerateAuthEvent with empty RequestedAuthConfigs should return nil")
	}
}

// Note: TestGenerateAuthEvent_CreatesEvent and TestGenerateAuthEvent_MultipleCalls
// are skipped as they require full invocation context with agent setup.
// The GenerateAuthEvent function is tested indirectly through integration tests.

func TestGenerateFunctionCallID(t *testing.T) {
	id1 := generateFunctionCallID()
	id2 := generateFunctionCallID()

	if id1 == "" {
		t.Error("generateFunctionCallID() returned empty string")
	}
	if id1 == id2 {
		t.Error("generateFunctionCallID() should return unique IDs")
	}
	if len(id1) < 4 || id1[:4] != "adk-" {
		t.Errorf("generateFunctionCallID() = %q, should start with 'adk-'", id1)
	}
}

func TestParseAuthConfigFromMap(t *testing.T) {
	data := map[string]any{
		"credential_key": "test-key",
		"exchanged_auth_credential": map[string]any{
			"auth_type": "oauth2",
			"oauth2": map[string]any{
				"access_token":  "token123",
				"refresh_token": "refresh456",
				"expires_at":    float64(1234567890),
			},
		},
	}

	config, err := parseAuthConfigFromMap(data)
	if err != nil {
		t.Fatalf("parseAuthConfigFromMap() error = %v", err)
	}
	if config.CredentialKey != "test-key" {
		t.Errorf("CredentialKey = %q, want %q", config.CredentialKey, "test-key")
	}
	if config.ExchangedAuthCredential == nil {
		t.Fatal("ExchangedAuthCredential should not be nil")
	}
	if config.ExchangedAuthCredential.OAuth2 == nil {
		t.Fatal("OAuth2 should not be nil")
	}
	if config.ExchangedAuthCredential.OAuth2.AccessToken != "token123" {
		t.Errorf("AccessToken = %q, want %q", config.ExchangedAuthCredential.OAuth2.AccessToken, "token123")
	}
}

func TestParseAuthConfigFromMap_CamelCase(t *testing.T) {
	data := map[string]any{
		"credentialKey": "camel-key",
		"exchangedAuthCredential": map[string]any{
			"authType": "oauth2",
			"oauth2": map[string]any{
				"accessToken": "camel_token",
			},
		},
	}

	config, err := parseAuthConfigFromMap(data)
	if err != nil {
		t.Fatalf("parseAuthConfigFromMap() error = %v", err)
	}
	if config.CredentialKey != "camel-key" {
		t.Errorf("CredentialKey = %q, want %q", config.CredentialKey, "camel-key")
	}
	if got := config.ExchangedAuthCredential.OAuth2.AccessToken; got != "camel_token" {
		t.Errorf("AccessToken = %q, want %q", got, "camel_token")
	}
}

func TestParseAuthCredentialFromMap(t *testing.T) {
	data := map[string]any{
		"auth_type": "oauth2",
		"oauth2": map[string]any{
			"access_token":  "access",
			"refresh_token": "refresh",
			"expires_at":    float64(9999999999),
		},
	}

	cred, err := parseAuthCredentialFromMap(data)
	if err != nil {
		t.Fatalf("parseAuthCredentialFromMap() error = %v", err)
	}
	if cred.AuthType != auth.AuthCredentialTypeOAuth2 {
		t.Errorf("AuthType = %v, want %v", cred.AuthType, auth.AuthCredentialTypeOAuth2)
	}
	if cred.OAuth2.AccessToken != "access" {
		t.Errorf("AccessToken = %q, want %q", cred.OAuth2.AccessToken, "access")
	}
	if cred.OAuth2.RefreshToken != "refresh" {
		t.Errorf("RefreshToken = %q, want %q", cred.OAuth2.RefreshToken, "refresh")
	}
}

func TestParseAuthCredentialFromMap_WithHyphenKeys(t *testing.T) {
	data := map[string]any{
		"auth-type": "oauth2",
		"oauth2": map[string]any{
			"access-token":  "hy-access",
			"refresh-token": "hy-refresh",
		},
	}

	cred, err := parseAuthCredentialFromMap(data)
	if err != nil {
		t.Fatalf("parseAuthCredentialFromMap() error = %v", err)
	}
	if cred.AuthType != auth.AuthCredentialTypeOAuth2 {
		t.Errorf("AuthType = %v, want %v", cred.AuthType, auth.AuthCredentialTypeOAuth2)
	}
	if cred.OAuth2.AccessToken != "hy-access" {
		t.Errorf("AccessToken = %q, want %q", cred.OAuth2.AccessToken, "hy-access")
	}
	if cred.OAuth2.RefreshToken != "hy-refresh" {
		t.Errorf("RefreshToken = %q, want %q", cred.OAuth2.RefreshToken, "hy-refresh")
	}
}

func TestParseAuthCredentialFromMap_NotAMap(t *testing.T) {
	_, err := parseAuthCredentialFromMap("not a map")
	if err == nil {
		t.Error("parseAuthCredentialFromMap() should error for non-map input")
	}
}
