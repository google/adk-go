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
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"

	"google.golang.org/adk/v2/internal/authhandler"
	"google.golang.org/adk/v2/tool/toolauth"
)

func apiKeyConfig() *toolauth.Config {
	return &toolauth.Config{
		AuthScheme:        toolauth.Scheme{Type: toolauth.SchemeTypeAPIKey, In: "header", Name: "X-Api-Key"},
		RawAuthCredential: &toolauth.Credential{AuthType: toolauth.CredentialTypeAPIKey, APIKey: "k3y"},
	}
}

func rawHasAuthURI(c *toolauth.Config) {
	c.RawAuthCredential.OAuth2.AuthURI = "https://idp.example/authorize?ready=raw"
}

func exchangedHasAuthURI(c *toolauth.Config) {
	c.ExchangedAuthCredential = &toolauth.Credential{
		AuthType: toolauth.CredentialTypeOAuth2,
		OAuth2:   &toolauth.OAuth2Auth{ClientID: "app-123", AuthURI: "https://idp.example/authorize?ready=exchanged"},
	}
}

// configuredSecrets names the agent-configured secrets left in cfg.
func configuredSecrets(cfg *toolauth.Config) []string {
	var found []string
	for name, c := range map[string]*toolauth.Credential{"raw": cfg.RawAuthCredential, "exchanged": cfg.ExchangedAuthCredential} {
		if c == nil {
			continue
		}
		if c.APIKey != "" {
			found = append(found, name+" API key")
		}
		if c.OAuth2 != nil && c.OAuth2.ClientSecret != "" {
			found = append(found, name+" client secret")
		}
	}
	return found
}

func exchangedAuthURI(cfg *toolauth.Config) string {
	if cfg.ExchangedAuthCredential == nil || cfg.ExchangedAuthCredential.OAuth2 == nil {
		return ""
	}
	return cfg.ExchangedAuthCredential.OAuth2.AuthURI
}

func TestGenerateAuthRequestKeepsKeyAndDropsSecrets(t *testing.T) {
	tests := []struct {
		name string
		cfg  *toolauth.Config
	}{
		{"api key", apiKeyConfig()},
		{"oauth2 that needs an authorization URI", authCodeConfig(nil)},
		{"oauth2 with an explicit credential key", authCodeConfig(func(c *toolauth.Config) {
			c.CredentialKey = "my_tool_tokens"
		})},
		{"oauth2 whose raw credential has an authorization URI", authCodeConfig(rawHasAuthURI)},
		{"oauth2 whose exchanged credential has an authorization URI", authCodeConfig(exchangedHasAuthURI)},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			wantKey, err := tt.cfg.Key()
			if err != nil {
				t.Fatalf("Key() error = %v", err)
			}
			got, err := authhandler.GenerateAuthRequest(tt.cfg)
			if err != nil {
				t.Fatalf("GenerateAuthRequest() error = %v", err)
			}
			if got.CredentialKey != wantKey {
				t.Errorf("CredentialKey = %q, want the tool config's key %q", got.CredentialKey, wantKey)
			}
			if secrets := configuredSecrets(got); len(secrets) > 0 {
				t.Errorf("request carries configured secrets: %v", secrets)
			}
		})
	}
}

func TestGenerateAuthRequestKeepsAnExistingAuthURI(t *testing.T) {
	tests := []struct {
		name string
		cfg  *toolauth.Config
		want string
	}{
		{"copied from the raw credential", authCodeConfig(rawHasAuthURI), "https://idp.example/authorize?ready=raw"},
		{"kept in the exchanged credential", authCodeConfig(exchangedHasAuthURI), "https://idp.example/authorize?ready=exchanged"},
		{"none for a scheme without a sign-in page", apiKeyConfig(), ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := authhandler.GenerateAuthRequest(tt.cfg)
			if err != nil {
				t.Fatalf("GenerateAuthRequest() error = %v", err)
			}
			if uri := exchangedAuthURI(got); uri != tt.want {
				t.Errorf("exchanged AuthURI = %q, want %q", uri, tt.want)
			}
		})
	}
}

func TestGenerateAuthRequestGeneratesAMissingAuthURI(t *testing.T) {
	tests := []struct {
		name       string
		cfg        *toolauth.Config
		wantPrefix string
	}{
		{"oauth2 flow", authCodeConfig(nil), "https://idp.example/authorize?"},
		{"openid connect with inline endpoints", authCodeConfig(func(c *toolauth.Config) {
			c.AuthScheme = toolauth.Scheme{
				Type:                  toolauth.SchemeTypeOpenIDConnect,
				AuthorizationEndpoint: "https://oidc.example/auth",
				TokenEndpoint:         "https://oidc.example/token",
			}
		}), "https://oidc.example/auth?"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := authhandler.GenerateAuthRequest(tt.cfg)
			if err != nil {
				t.Fatalf("GenerateAuthRequest() error = %v", err)
			}
			if uri := exchangedAuthURI(got); !strings.HasPrefix(uri, tt.wantPrefix) {
				t.Errorf("exchanged AuthURI = %q, want one starting %q", uri, tt.wantPrefix)
			}
		})
	}
}

func TestGenerateAuthRequestRejects(t *testing.T) {
	tests := []struct {
		name string
		mod  func(*toolauth.Config)
	}{
		{"no raw credential", func(c *toolauth.Config) { c.RawAuthCredential = nil }},
		{"raw credential without oauth2", func(c *toolauth.Config) { c.RawAuthCredential.OAuth2 = nil }},
		{"no client ID", func(c *toolauth.Config) { c.RawAuthCredential.OAuth2.ClientID = "" }},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := authhandler.GenerateAuthRequest(authCodeConfig(tt.mod)); err == nil {
				t.Errorf("GenerateAuthRequest() error = nil, want an error")
			}
		})
	}
}

func TestGenerateAuthRequestLeavesConfigUnchanged(t *testing.T) {
	tests := []struct {
		name string
		mod  func(*toolauth.Config)
	}{
		{"oauth2 that needs an authorization URI", nil},
		{"oauth2 whose raw credential has an authorization URI", rawHasAuthURI},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := authCodeConfig(tt.mod)
			if _, err := authhandler.GenerateAuthRequest(cfg); err != nil {
				t.Fatalf("GenerateAuthRequest() error = %v", err)
			}
			if diff := cmp.Diff(authCodeConfig(tt.mod), cfg); diff != "" {
				t.Errorf("config was modified (-want +got):\n%s", diff)
			}
		})
	}
}
