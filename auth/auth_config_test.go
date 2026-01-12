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
	"errors"
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

	cfg, err := NewAuthConfig(scheme, cred)
	if err != nil {
		t.Fatalf("NewAuthConfig() error = %v", err)
	}

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

	cfg1, err := NewAuthConfig(scheme, cred)
	if err != nil {
		t.Fatalf("NewAuthConfig() error = %v", err)
	}
	cfg2, err := NewAuthConfig(scheme, cred)
	if err != nil {
		t.Fatalf("NewAuthConfig() error = %v", err)
	}

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

	cfg1, err := NewAuthConfig(scheme, cred1)
	if err != nil {
		t.Fatalf("NewAuthConfig() error = %v", err)
	}
	cfg2, err := NewAuthConfig(scheme, cred2)
	if err != nil {
		t.Fatalf("NewAuthConfig() error = %v", err)
	}

	if cfg1.CredentialKey == cfg2.CredentialKey {
		t.Error("Different credentials should produce different keys")
	}
}

func TestAuthConfig_generateCredentialKey_ScopeOrder(t *testing.T) {
	makeCfg := func(scopes map[string]string) *AuthConfig {
		cfg, err := NewAuthConfig(&OAuth2Scheme{
			Flows: &OAuthFlows{
				AuthorizationCode: &OAuthFlowAuthorizationCode{
					AuthorizationURL: "https://example.com/auth",
					TokenURL:         "https://example.com/token",
					Scopes:           scopes,
				},
			},
		}, &AuthCredential{
			AuthType: AuthCredentialTypeOAuth2,
			OAuth2: &OAuth2Auth{
				ClientID: "client",
			},
		})
		if err != nil {
			t.Fatalf("NewAuthConfig() error = %v", err)
		}
		return cfg
	}

	cfg1 := makeCfg(map[string]string{
		"read":  "Read",
		"write": "Write",
	})
	cfg2 := makeCfg(map[string]string{
		"write": "Write",
		"read":  "Read",
	})

	if cfg1.CredentialKey != cfg2.CredentialKey {
		t.Fatalf("credential keys differ for same scopes: %q vs %q", cfg1.CredentialKey, cfg2.CredentialKey)
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

func TestAuthConfig_CopyDeepCopiesAuthScheme(t *testing.T) {
	cfg := &AuthConfig{
		AuthScheme: &OAuth2Scheme{
			Description: "orig",
			Flows: &OAuthFlows{
				ClientCredentials: &OAuthFlowClientCredentials{
					TokenURL: "https://example.com/token",
					Scopes: map[string]string{
						"repo": "read",
					},
				},
			},
		},
	}

	got := cfg.Copy()

	orig := cfg.AuthScheme.(*OAuth2Scheme)
	orig.Description = "mutated"
	orig.Flows.ClientCredentials.Scopes["repo"] = "write"

	copied := got.AuthScheme.(*OAuth2Scheme)
	if copied.Description != "orig" {
		t.Fatalf("copied.Description = %q, want %q", copied.Description, "orig")
	}
	if copied.Flows.ClientCredentials.Scopes["repo"] != "read" {
		t.Fatalf("copied scope = %q, want %q", copied.Flows.ClientCredentials.Scopes["repo"], "read")
	}
}

func TestNewAuthConfig_MarshalError(t *testing.T) {
	scheme := &badScheme{}
	if _, err := NewAuthConfig(scheme, nil); err == nil {
		t.Fatal("NewAuthConfig() did not return error for unmarshalable scheme")
	}
}

type badScheme struct{}

func (b *badScheme) GetType() SecuritySchemeType {
	return SecuritySchemeType("bad")
}

func (b *badScheme) MarshalJSON() ([]byte, error) {
	return nil, errors.New("cannot marshal")
}
