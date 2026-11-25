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
	"maps"
	"net/url"
	"testing"

	"github.com/google/go-cmp/cmp"

	"google.golang.org/adk/v2/internal/authhandler"
	"google.golang.org/adk/v2/tool/toolauth"
)

// authCodeConfig returns an authorization code config for a client with a
// secret, with mod applied on top.
func authCodeConfig(mod func(*toolauth.Config)) *toolauth.Config {
	cfg := &toolauth.Config{
		AuthScheme: toolauth.Scheme{
			Type: toolauth.SchemeTypeOAuth2,
			Flows: &toolauth.OAuthFlows{AuthorizationCode: &toolauth.OAuthFlow{
				AuthorizationURL: "https://idp.example/authorize",
				TokenURL:         "https://idp.example/token",
				Scopes:           map[string]string{"b.scope": "B", "a.scope": "A"},
			}},
		},
		RawAuthCredential: &toolauth.Credential{
			AuthType: toolauth.CredentialTypeOAuth2,
			OAuth2: &toolauth.OAuth2Auth{
				ClientID:     "app-123",
				ClientSecret: "s3cret",
				RedirectURI:  "https://adk.example/cb",
			},
		},
	}
	if mod != nil {
		mod(cfg)
	}
	return cfg
}

func publicClient(c *toolauth.Config) { c.RawAuthCredential.OAuth2.ClientSecret = "" }

func parseAuthURI(t *testing.T, cred *toolauth.Credential) *url.URL {
	t.Helper()
	u, err := url.Parse(cred.OAuth2.AuthURI)
	if err != nil {
		t.Fatalf("AuthURI does not parse: %v", err)
	}
	return u
}

func TestGenerateAuthURI(t *testing.T) {
	base := func(overrides url.Values) url.Values {
		q := url.Values{
			"response_type": {"code"},
			"client_id":     {"app-123"},
			"redirect_uri":  {"https://adk.example/cb"},
			"scope":         {"a.scope b.scope"},
			"access_type":   {"offline"},
			"prompt":        {"consent"},
		}
		maps.Copy(q, overrides)
		return q
	}
	tests := []struct {
		name         string
		cfg          *toolauth.Config
		wantEndpoint string
		wantQuery    url.Values
	}{
		{
			name:         "client with a secret gets a plain authorization code request",
			cfg:          authCodeConfig(nil),
			wantEndpoint: "https://idp.example/authorize",
			wantQuery:    base(nil),
		},
		{
			name: "prompt, audience and nonce are sent when set",
			cfg: authCodeConfig(func(c *toolauth.Config) {
				o := c.RawAuthCredential.OAuth2
				o.Prompt, o.Audience, o.Nonce = "select_account", "api://x", "n-1"
			}),
			wantEndpoint: "https://idp.example/authorize",
			wantQuery: base(url.Values{
				"prompt":   {"select_account"},
				"audience": {"api://x"},
				"nonce":    {"n-1"},
			}),
		},
		{
			name: "openid connect scheme with inline endpoints uses them",
			cfg: authCodeConfig(func(c *toolauth.Config) {
				c.AuthScheme = toolauth.Scheme{
					Type:                  toolauth.SchemeTypeOpenIDConnect,
					AuthorizationEndpoint: "https://oidc.example/auth",
					TokenEndpoint:         "https://oidc.example/token",
					Scopes:                []string{"openid", "email"},
				}
			}),
			wantEndpoint: "https://oidc.example/auth",
			wantQuery:    base(url.Values{"scope": {"openid email"}}),
		},
		{
			name: "implicit flow wins over authorization code",
			cfg: authCodeConfig(func(c *toolauth.Config) {
				c.AuthScheme.Flows.Implicit = &toolauth.OAuthFlow{
					AuthorizationURL: "https://idp.example/implicit",
					Scopes:           map[string]string{"i.scope": "I"},
				}
			}),
			wantEndpoint: "https://idp.example/implicit",
			wantQuery:    base(url.Values{"scope": {"i.scope"}}),
		},
		{
			name: "client credentials flow falls back to its token URL",
			cfg: authCodeConfig(func(c *toolauth.Config) {
				c.AuthScheme.Flows = &toolauth.OAuthFlows{ClientCredentials: &toolauth.OAuthFlow{
					TokenURL: "https://idp.example/token",
					Scopes:   map[string]string{"c.scope": "C"},
				}}
			}),
			wantEndpoint: "https://idp.example/token",
			wantQuery:    base(url.Values{"scope": {"c.scope"}}),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := authhandler.GenerateAuthURI(tt.cfg)
			if err != nil {
				t.Fatalf("GenerateAuthURI() error = %v", err)
			}
			u := parseAuthURI(t, got)
			q := u.Query()
			if q.Get("state") == "" || q.Get("state") != got.OAuth2.State {
				t.Errorf("state in the URI and in the credential differ, or are empty")
			}
			q.Del("state")
			if endpoint := u.Scheme + "://" + u.Host + u.Path; endpoint != tt.wantEndpoint {
				t.Errorf("endpoint = %q, want %q", endpoint, tt.wantEndpoint)
			}
			if diff := cmp.Diff(tt.wantQuery, q); diff != "" {
				t.Errorf("query mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestGenerateAuthURIPublicClientUsesS256(t *testing.T) {
	got, err := authhandler.GenerateAuthURI(authCodeConfig(publicClient))
	if err != nil {
		t.Fatalf("GenerateAuthURI() error = %v", err)
	}
	q := parseAuthURI(t, got).Query()
	if got.OAuth2.CodeChallengeMethod != "S256" || q.Get("code_challenge_method") != "S256" {
		t.Errorf("challenge method = %q in the credential and %q in the URI, want S256 in both",
			got.OAuth2.CodeChallengeMethod, q.Get("code_challenge_method"))
	}
	if len(got.OAuth2.CodeVerifier) != 48 {
		t.Errorf("len(CodeVerifier) = %d, want 48", len(got.OAuth2.CodeVerifier))
	}
	if q.Get("code_challenge") == "" {
		t.Errorf("URI has no code_challenge")
	}
}

// The verifier and challenge are the pair in RFC 7636, appendix B.
func TestGenerateAuthURIKeepsSuppliedVerifier(t *testing.T) {
	const verifier = "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"
	cfg := authCodeConfig(func(c *toolauth.Config) {
		c.RawAuthCredential.OAuth2.CodeChallengeMethod = "S256"
		c.RawAuthCredential.OAuth2.CodeVerifier = verifier
	})
	got, err := authhandler.GenerateAuthURI(cfg)
	if err != nil {
		t.Fatalf("GenerateAuthURI() error = %v", err)
	}
	if got.OAuth2.CodeVerifier != verifier {
		t.Errorf("CodeVerifier was replaced")
	}
	if challenge := parseAuthURI(t, got).Query().Get("code_challenge"); challenge != "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM" {
		t.Errorf("code_challenge = %q, want the RFC 7636 value", challenge)
	}
}

func TestGenerateAuthURIRejects(t *testing.T) {
	tests := []struct {
		name string
		mod  func(*toolauth.Config)
	}{
		{"no raw credential", func(c *toolauth.Config) { c.RawAuthCredential = nil }},
		{"raw credential without oauth2", func(c *toolauth.Config) { c.RawAuthCredential.OAuth2 = nil }},
		{"challenge method other than S256", func(c *toolauth.Config) { c.RawAuthCredential.OAuth2.CodeChallengeMethod = "plain" }},
		{"scheme without flows", func(c *toolauth.Config) { c.AuthScheme = toolauth.Scheme{Type: toolauth.SchemeTypeOAuth2} }},
		{"flow without an authorization endpoint", func(c *toolauth.Config) {
			c.AuthScheme.Flows.AuthorizationCode = &toolauth.OAuthFlow{}
		}},
		{"openid connect with only a discovery URL", func(c *toolauth.Config) {
			c.AuthScheme = toolauth.Scheme{
				Type:             toolauth.SchemeTypeOpenIDConnect,
				OpenIDConnectURL: "https://oidc.example/.well-known/openid-configuration",
			}
		}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := authhandler.GenerateAuthURI(authCodeConfig(tt.mod)); err == nil {
				t.Errorf("GenerateAuthURI() error = nil, want an error")
			}
		})
	}
}

func TestGenerateAuthURILeavesConfigUnchanged(t *testing.T) {
	cfg := authCodeConfig(publicClient)
	if _, err := authhandler.GenerateAuthURI(cfg); err != nil {
		t.Fatalf("GenerateAuthURI() error = %v", err)
	}
	if diff := cmp.Diff(authCodeConfig(publicClient), cfg); diff != "" {
		t.Errorf("config was modified (-want +got):\n%s", diff)
	}
}
