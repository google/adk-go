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
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
)

func TestNewAuthConfig(t *testing.T) {
	scheme := &OAuth2Scheme{
		Flows: &OAuthFlows{
			AuthorizationCode: &OAuthFlowAuthorizationCode{
				AuthorizationURL: "https://example.com/auth",
				TokenURL:         "https://example.com/token",
			},
		},
	}
	cred := &AuthCredential{
		AuthType: AuthCredentialTypeOAuth2,
		OAuth2: &OAuth2Auth{
			ClientID:     "client-id",
			ClientSecret: "client-secret",
		},
	}

	cfg := NewAuthConfig(scheme, cred)

	if cfg.AuthScheme != scheme {
		t.Error("AuthScheme not set correctly")
	}
	if cfg.RawAuthCredential != cred {
		t.Error("RawAuthCredential not set correctly")
	}
	if cfg.CredentialKey == "" {
		t.Error("CredentialKey should be auto-generated")
	}
	if !strings.HasPrefix(cfg.CredentialKey, "adk_") {
		t.Errorf("CredentialKey = %q, want prefix 'adk_'", cfg.CredentialKey)
	}
}

func TestAuthConfig_generateCredentialKey_Deterministic(t *testing.T) {
	scheme := &APIKeyScheme{
		In:   APIKeyInHeader,
		Name: "X-API-Key",
	}
	cred := &AuthCredential{
		AuthType: AuthCredentialTypeAPIKey,
		APIKey:   "test-key",
	}

	cfg1 := NewAuthConfig(scheme, cred)
	cfg2 := NewAuthConfig(scheme, cred)

	if cfg1.CredentialKey != cfg2.CredentialKey {
		t.Errorf("generateCredentialKey not deterministic: %q != %q", cfg1.CredentialKey, cfg2.CredentialKey)
	}
}

func TestAuthConfig_generateCredentialKey_Different(t *testing.T) {
	scheme := &APIKeyScheme{
		In:   APIKeyInHeader,
		Name: "X-API-Key",
	}
	cred1 := &AuthCredential{
		AuthType: AuthCredentialTypeAPIKey,
		APIKey:   "key-1",
	}
	cred2 := &AuthCredential{
		AuthType: AuthCredentialTypeAPIKey,
		APIKey:   "key-2",
	}

	cfg1 := NewAuthConfig(scheme, cred1)
	cfg2 := NewAuthConfig(scheme, cred2)

	if cfg1.CredentialKey == cfg2.CredentialKey {
		t.Error("Different credentials should produce different keys")
	}
}

func TestAuthConfig_Copy_Nil(t *testing.T) {
	var cfg *AuthConfig
	got := cfg.Copy()
	if got != nil {
		t.Errorf("Copy() of nil = %v, want nil", got)
	}
}

func TestAuthConfig_Copy(t *testing.T) {
	scheme := &HTTPScheme{
		Scheme:       "bearer",
		BearerFormat: "JWT",
	}
	cfg := &AuthConfig{
		AuthScheme: scheme,
		RawAuthCredential: &AuthCredential{
			AuthType: AuthCredentialTypeHTTP,
			HTTP: &HTTPAuth{
				Scheme: "bearer",
				Credentials: &HTTPCredentials{
					Token: "raw-token",
				},
			},
		},
		ExchangedAuthCredential: &AuthCredential{
			AuthType: AuthCredentialTypeHTTP,
			HTTP: &HTTPAuth{
				Scheme: "bearer",
				Credentials: &HTTPCredentials{
					Token: "exchanged-token",
				},
			},
		},
		CredentialKey: "adk_test_key",
	}

	got := cfg.Copy()

	if got == cfg {
		t.Error("Copy() returned same pointer")
	}
	if got.RawAuthCredential == cfg.RawAuthCredential {
		t.Error("Copy() returned same RawAuthCredential pointer")
	}
	if got.ExchangedAuthCredential == cfg.ExchangedAuthCredential {
		t.Error("Copy() returned same ExchangedAuthCredential pointer")
	}
	if diff := cmp.Diff(cfg, got); diff != "" {
		t.Errorf("Copy() mismatch (-want +got):\n%s", diff)
	}
}
