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

package toolauth_test

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"

	"google.golang.org/adk/v2/tool/toolauth"
)

const (
	testTokenURL   = "https://example.com/token"
	testAuthURL    = "https://example.com/authorize"
	testRefreshURL = "https://example.com/refresh"
)

// pythonFlowCases pairs OAuth flows with the JSON adk-python 2.12.0 writes for
// the same flows, through fastapi's OAuthFlows and
// model_dump(by_alias=True, exclude_none=True).
var pythonFlowCases = []struct {
	name  string
	flows toolauth.OAuthFlows
	json  string
}{
	{
		name:  "authorization code",
		flows: toolauth.OAuthFlows{AuthorizationCode: &toolauth.OAuthFlow{AuthorizationURL: testAuthURL, TokenURL: testTokenURL}},
		json:  `{"authorizationCode":{"scopes":{},"authorizationUrl":"https://example.com/authorize","tokenUrl":"https://example.com/token"}}`,
	},
	{
		name: "client credentials with a scope",
		flows: toolauth.OAuthFlows{ClientCredentials: &toolauth.OAuthFlow{
			TokenURL: testTokenURL,
			Scopes:   map[string]string{"read": "Read access"},
		}},
		json: `{"clientCredentials":{"scopes":{"read":"Read access"},"tokenUrl":"https://example.com/token"}}`,
	},
	{
		name:  "implicit with a refresh URL",
		flows: toolauth.OAuthFlows{Implicit: &toolauth.OAuthFlow{AuthorizationURL: testAuthURL, RefreshURL: testRefreshURL}},
		json:  `{"implicit":{"refreshUrl":"https://example.com/refresh","scopes":{},"authorizationUrl":"https://example.com/authorize"}}`,
	},
	{
		name:  "password",
		flows: toolauth.OAuthFlows{Password: &toolauth.OAuthFlow{TokenURL: testTokenURL}},
		json:  `{"password":{"scopes":{},"tokenUrl":"https://example.com/token"}}`,
	},
	{
		name: "all four flows",
		flows: toolauth.OAuthFlows{
			Implicit:          &toolauth.OAuthFlow{AuthorizationURL: testAuthURL},
			Password:          &toolauth.OAuthFlow{TokenURL: testTokenURL},
			ClientCredentials: &toolauth.OAuthFlow{TokenURL: testTokenURL},
			AuthorizationCode: &toolauth.OAuthFlow{AuthorizationURL: testAuthURL, TokenURL: testTokenURL},
		},
		json: `{"implicit":{"scopes":{},"authorizationUrl":"https://example.com/authorize"},"password":{"scopes":{},"tokenUrl":"https://example.com/token"},"clientCredentials":{"scopes":{},"tokenUrl":"https://example.com/token"},"authorizationCode":{"scopes":{},"authorizationUrl":"https://example.com/authorize","tokenUrl":"https://example.com/token"}}`,
	},
	{
		// fastapi's models take no snake_case names, so authorization_url is an
		// unknown key next to authorizationUrl, kept and written back.
		name: "unknown keys",
		flows: toolauth.OAuthFlows{
			AuthorizationCode: &toolauth.OAuthFlow{
				AuthorizationURL: testAuthURL,
				TokenURL:         testTokenURL,
				Extra:            map[string]any{"authorization_url": "x", "x-vendor": json.Number("1")},
			},
			Extra: map[string]any{"x-flows": true},
		},
		json: `{"authorizationCode":{"scopes":{},"authorizationUrl":"https://example.com/authorize","tokenUrl":"https://example.com/token","authorization_url":"x","x-vendor":1},"x-flows":true}`,
	},
	{
		name:  "no flows",
		flows: toolauth.OAuthFlows{},
		json:  `{}`,
	},
}

func TestOAuthFlowsMarshalMatchesPython(t *testing.T) {
	for _, tc := range pythonFlowCases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := json.Marshal(tc.flows)
			if err != nil {
				t.Fatalf("json.Marshal() error = %v", err)
			}
			if diff := cmp.Diff(decodeJSON(t, []byte(tc.json)), decodeJSON(t, got)); diff != "" {
				t.Errorf("json.Marshal() differs from adk-python (-want +got):\n%s", diff)
			}
		})
	}
}

func TestOAuthFlowsUnmarshalPythonJSON(t *testing.T) {
	for _, tc := range pythonFlowCases {
		t.Run(tc.name, func(t *testing.T) {
			var got toolauth.OAuthFlows
			if err := json.Unmarshal([]byte(tc.json), &got); err != nil {
				t.Fatalf("json.Unmarshal() error = %v", err)
			}
			// Python writes empty scopes as {}, which decodes to an empty map
			// where the case has nil.
			if diff := cmp.Diff(tc.flows, got, cmpopts.EquateEmpty()); diff != "" {
				t.Errorf("json.Unmarshal() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestOAuthFlowsUnmarshalRejects(t *testing.T) {
	// adk-python 2.12.0 rejects every one of these.
	tests := []struct {
		name    string
		data    string
		wantErr string
	}{
		{name: "implicit without authorizationUrl", data: `{"implicit":{}}`, wantErr: "implicit flow needs authorizationUrl"},
		{name: "password without tokenUrl", data: `{"password":{}}`, wantErr: "password flow needs tokenUrl"},
		{name: "client credentials without tokenUrl", data: `{"clientCredentials":{}}`, wantErr: "clientCredentials flow needs tokenUrl"},
		{
			name:    "authorization code without tokenUrl",
			data:    `{"authorizationCode":{"authorizationUrl":"https://example.com/authorize"}}`,
			wantErr: "authorizationCode flow needs tokenUrl",
		},
		{
			name:    "authorization code without authorizationUrl",
			data:    `{"authorizationCode":{"tokenUrl":"https://example.com/token"}}`,
			wantErr: "authorizationCode flow needs authorizationUrl",
		},
		{
			name:    "snake_case name only",
			data:    `{"password":{"token_url":"https://example.com/token"}}`,
			wantErr: "password flow needs tokenUrl",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var flows toolauth.OAuthFlows
			err := json.Unmarshal([]byte(tc.data), &flows)
			if err == nil {
				t.Fatal("json.Unmarshal() succeeded, want an error")
			}
			if !strings.Contains(err.Error(), tc.wantErr) {
				t.Errorf("json.Unmarshal() error = %q, want it to contain %q", err, tc.wantErr)
			}
		})
	}
}

func TestOAuthFlowsGrantType(t *testing.T) {
	tests := []struct {
		name  string
		flows toolauth.OAuthFlows
		want  toolauth.OAuthGrantType
	}{
		{
			name:  "client credentials",
			flows: toolauth.OAuthFlows{ClientCredentials: &toolauth.OAuthFlow{TokenURL: testTokenURL}},
			want:  toolauth.OAuthGrantTypeClientCredentials,
		},
		{
			name:  "authorization code",
			flows: toolauth.OAuthFlows{AuthorizationCode: &toolauth.OAuthFlow{AuthorizationURL: testAuthURL, TokenURL: testTokenURL}},
			want:  toolauth.OAuthGrantTypeAuthorizationCode,
		},
		{
			name:  "implicit",
			flows: toolauth.OAuthFlows{Implicit: &toolauth.OAuthFlow{AuthorizationURL: testAuthURL}},
			want:  toolauth.OAuthGrantTypeImplicit,
		},
		{
			name:  "password",
			flows: toolauth.OAuthFlows{Password: &toolauth.OAuthFlow{TokenURL: testTokenURL}},
			want:  toolauth.OAuthGrantTypePassword,
		},
		{
			// An OAuth2 scheme declaring no flow has no grant type to exchange with.
			name:  "no flow",
			flows: toolauth.OAuthFlows{},
			want:  "",
		},
		{
			// adk-python's from_flow checks client credentials first.
			name: "client credentials before authorization code",
			flows: toolauth.OAuthFlows{
				ClientCredentials: &toolauth.OAuthFlow{TokenURL: testTokenURL},
				AuthorizationCode: &toolauth.OAuthFlow{AuthorizationURL: testAuthURL, TokenURL: testTokenURL},
			},
			want: toolauth.OAuthGrantTypeClientCredentials,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := tc.flows.GrantType(); got != tc.want {
				t.Errorf("GrantType() = %q, want %q", got, tc.want)
			}
		})
	}
}

func TestOAuthGrantTypeValuesAreWireNames(t *testing.T) {
	// These strings go on the wire as the OAuth2 grant_type parameter, so they
	// must stay exactly as RFC 6749 names them.
	tests := map[toolauth.OAuthGrantType]string{
		toolauth.OAuthGrantTypeClientCredentials: "client_credentials",
		toolauth.OAuthGrantTypeAuthorizationCode: "authorization_code",
		toolauth.OAuthGrantTypeImplicit:          "implicit",
		toolauth.OAuthGrantTypePassword:          "password",
	}
	for got, want := range tests {
		if string(got) != want {
			t.Errorf("grant type = %q, want %q", got, want)
		}
	}
}

const (
	testOIDCDiscoveryURL = "https://example.com/.well-known/openid-configuration"
	testUserinfoURL      = "https://example.com/userinfo"
	testRevokeURL        = "https://example.com/revoke"
)

// pythonSchemeCases pairs auth schemes with the JSON adk-python 2.12.0 writes
// for the same scheme after reading it into an AuthConfig, which resolves the
// scheme union, through model_dump(by_alias=True, exclude_none=True).
var pythonSchemeCases = []struct {
	name   string
	scheme toolauth.Scheme
	json   string
}{
	{
		name:   "API key",
		scheme: toolauth.Scheme{Type: toolauth.SchemeTypeAPIKey, In: "header", Name: "X-Api-Key"},
		json:   `{"type":"apiKey","in":"header","name":"X-Api-Key"}`,
	},
	{
		name:   "HTTP basic",
		scheme: toolauth.Scheme{Type: toolauth.SchemeTypeHTTP, Description: "Basic auth", Scheme: "basic"},
		json:   `{"type":"http","description":"Basic auth","scheme":"basic"}`,
	},
	{
		name:   "HTTP bearer with a format",
		scheme: toolauth.Scheme{Type: toolauth.SchemeTypeHTTP, Scheme: "bearer", BearerFormat: "JWT"},
		json:   `{"type":"http","scheme":"bearer","bearerFormat":"JWT"}`,
	},
	{
		name: "OAuth2",
		scheme: toolauth.Scheme{
			Type: toolauth.SchemeTypeOAuth2,
			Flows: &toolauth.OAuthFlows{AuthorizationCode: &toolauth.OAuthFlow{
				AuthorizationURL: testAuthURL,
				TokenURL:         testTokenURL,
				Scopes:           map[string]string{"read": "Read access"},
			}},
		},
		json: `{"type":"oauth2","flows":{"authorizationCode":{"scopes":{"read":"Read access"},"authorizationUrl":"https://example.com/authorize","tokenUrl":"https://example.com/token"}}}`,
	},
	{
		name:   "OAuth2 with an issuer",
		scheme: toolauth.Scheme{Type: toolauth.SchemeTypeOAuth2, Flows: &toolauth.OAuthFlows{}, IssuerURL: "https://issuer.example.com"},
		json:   `{"type":"oauth2","flows":{},"issuer_url":"https://issuer.example.com"}`,
	},
	{
		name:   "OpenID Connect discovery URL",
		scheme: toolauth.Scheme{Type: toolauth.SchemeTypeOpenIDConnect, OpenIDConnectURL: testOIDCDiscoveryURL},
		json:   `{"type":"openIdConnect","openIdConnectUrl":"https://example.com/.well-known/openid-configuration"}`,
	},
	{
		name: "OpenID Connect endpoints",
		scheme: toolauth.Scheme{
			Type:                              toolauth.SchemeTypeOpenIDConnect,
			AuthorizationEndpoint:             testAuthURL,
			TokenEndpoint:                     testTokenURL,
			UserinfoEndpoint:                  testUserinfoURL,
			RevocationEndpoint:                testRevokeURL,
			TokenEndpointAuthMethodsSupported: []string{"client_secret_basic"},
			GrantTypesSupported:               []string{"authorization_code"},
			Scopes:                            []string{"openid", "email"},
		},
		json: `{"type":"openIdConnect","authorization_endpoint":"https://example.com/authorize","token_endpoint":"https://example.com/token","userinfo_endpoint":"https://example.com/userinfo","revocation_endpoint":"https://example.com/revoke","token_endpoint_auth_methods_supported":["client_secret_basic"],"grant_types_supported":["authorization_code"],"scopes":["openid","email"]}`,
	},
	{
		name: "OpenID Connect endpoints with empty lists",
		scheme: toolauth.Scheme{
			Type:                              toolauth.SchemeTypeOpenIDConnect,
			AuthorizationEndpoint:             testAuthURL,
			TokenEndpoint:                     testTokenURL,
			TokenEndpointAuthMethodsSupported: []string{},
			GrantTypesSupported:               []string{},
			Scopes:                            []string{},
		},
		json: `{"type":"openIdConnect","authorization_endpoint":"https://example.com/authorize","token_endpoint":"https://example.com/token","token_endpoint_auth_methods_supported":[],"grant_types_supported":[],"scopes":[]}`,
	},
	{
		name:   "custom scheme",
		scheme: toolauth.Scheme{Type: "mock_custom_type", Extra: map[string]any{"x-provider": "acme"}},
		json:   `{"type":"mock_custom_type","x-provider":"acme"}`,
	},
	{
		// adk-python reads this as a custom scheme, since "body" is not an
		// OpenAPI key location, and writes it back unchanged.
		name:   "API key in an unknown location",
		scheme: toolauth.Scheme{Type: toolauth.SchemeTypeAPIKey, In: "body", Name: "X"},
		json:   `{"type":"apiKey","in":"body","name":"X"}`,
	},
	{
		// The scheme models take no other spelling of type, so type_ is an
		// unknown key.
		name:   "custom scheme with type_",
		scheme: toolauth.Scheme{Type: "custom", Extra: map[string]any{"type_": "other"}},
		json:   `{"type":"custom","type_":"other"}`,
	},
}

func TestSchemeMarshalMatchesPython(t *testing.T) {
	for _, tc := range pythonSchemeCases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := json.Marshal(tc.scheme)
			if err != nil {
				t.Fatalf("json.Marshal() error = %v", err)
			}
			if diff := cmp.Diff(decodeJSON(t, []byte(tc.json)), decodeJSON(t, got)); diff != "" {
				t.Errorf("json.Marshal() differs from adk-python (-want +got):\n%s", diff)
			}
		})
	}
}

func TestSchemeUnmarshalPythonJSON(t *testing.T) {
	for _, tc := range pythonSchemeCases {
		t.Run(tc.name, func(t *testing.T) {
			var got toolauth.Scheme
			if err := json.Unmarshal([]byte(tc.json), &got); err != nil {
				t.Fatalf("json.Unmarshal() error = %v", err)
			}
			if diff := cmp.Diff(tc.scheme, got, cmpopts.EquateEmpty()); diff != "" {
				t.Errorf("json.Unmarshal() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestSchemeInfersMissingType(t *testing.T) {
	// Each input has no type. want is what adk-python 2.12.0 writes after
	// reading it, with the type its models inferred.
	tests := []struct {
		name string
		in   string
		want string
	}{
		{
			name: "API key",
			in:   `{"in":"header","name":"X-Api-Key"}`,
			want: `{"type":"apiKey","in":"header","name":"X-Api-Key"}`,
		},
		{
			name: "HTTP",
			in:   `{"scheme":"basic"}`,
			want: `{"type":"http","scheme":"basic"}`,
		},
		{
			name: "OAuth2",
			in:   `{"flows":{"password":{"tokenUrl":"https://example.com/token"}}}`,
			want: `{"type":"oauth2","flows":{"password":{"scopes":{},"tokenUrl":"https://example.com/token"}}}`,
		},
		{
			name: "OpenID Connect discovery URL",
			in:   `{"openIdConnectUrl":"https://example.com/.well-known/openid-configuration"}`,
			want: `{"type":"openIdConnect","openIdConnectUrl":"https://example.com/.well-known/openid-configuration"}`,
		},
		{
			name: "OpenID Connect endpoints",
			in:   `{"authorization_endpoint":"https://example.com/authorize","token_endpoint":"https://example.com/token"}`,
			want: `{"type":"openIdConnect","authorization_endpoint":"https://example.com/authorize","token_endpoint":"https://example.com/token"}`,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var s toolauth.Scheme
			if err := json.Unmarshal([]byte(tc.in), &s); err != nil {
				t.Fatalf("json.Unmarshal() error = %v", err)
			}
			got, err := json.Marshal(s)
			if err != nil {
				t.Fatalf("json.Marshal() error = %v", err)
			}
			if diff := cmp.Diff(decodeJSON(t, []byte(tc.want)), decodeJSON(t, got)); diff != "" {
				t.Errorf("round trip differs from adk-python (-want +got):\n%s", diff)
			}
		})
	}
}

func TestSchemeRejectsSchemeWithNothingToInferTypeFrom(t *testing.T) {
	// adk-python 2.12.0 reads both as {"type":"http","scheme":"bearer"}, only
	// because fastapi's HTTPBearer has a default for every field.
	for _, data := range []string{`{}`, `{"description":"d"}`} {
		var s toolauth.Scheme
		err := json.Unmarshal([]byte(data), &s)
		if err == nil || !strings.Contains(err.Error(), "missing type") {
			t.Errorf("json.Unmarshal(%s) error = %v, want a missing type error", data, err)
		}
	}
}
