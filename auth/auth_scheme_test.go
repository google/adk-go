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

import "testing"

func TestAPIKeyScheme_GetType(t *testing.T) {
	scheme := &APIKeyScheme{
		In:   APIKeyInHeader,
		Name: "X-API-Key",
	}
	if got := scheme.GetType(); got != SecuritySchemeTypeAPIKey {
		t.Errorf("GetType() = %v, want %v", got, SecuritySchemeTypeAPIKey)
	}
}

func TestHTTPScheme_GetType(t *testing.T) {
	scheme := &HTTPScheme{
		Scheme:       "bearer",
		BearerFormat: "JWT",
	}
	if got := scheme.GetType(); got != SecuritySchemeTypeHTTP {
		t.Errorf("GetType() = %v, want %v", got, SecuritySchemeTypeHTTP)
	}
}

func TestOAuth2Scheme_GetType(t *testing.T) {
	scheme := &OAuth2Scheme{
		Flows: &OAuthFlows{
			AuthorizationCode: &OAuthFlowAuthorizationCode{
				AuthorizationURL: "https://example.com/auth",
				TokenURL:         "https://example.com/token",
			},
		},
	}
	if got := scheme.GetType(); got != SecuritySchemeTypeOAuth2 {
		t.Errorf("GetType() = %v, want %v", got, SecuritySchemeTypeOAuth2)
	}
}

func TestOpenIDConnectScheme_GetType(t *testing.T) {
	scheme := &OpenIDConnectScheme{
		OpenIDConnectURL: "https://example.com/.well-known/openid-configuration",
	}
	if got := scheme.GetType(); got != SecuritySchemeTypeOpenIDConnect {
		t.Errorf("GetType() = %v, want %v", got, SecuritySchemeTypeOpenIDConnect)
	}
}

func TestAuthScheme_Interface(t *testing.T) {
	// Verify all scheme types implement AuthScheme interface
	var _ AuthScheme = (*APIKeyScheme)(nil)
	var _ AuthScheme = (*HTTPScheme)(nil)
	var _ AuthScheme = (*OAuth2Scheme)(nil)
	var _ AuthScheme = (*OpenIDConnectScheme)(nil)
}
