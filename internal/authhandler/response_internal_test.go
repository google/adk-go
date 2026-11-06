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

package authhandler

import (
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"

	"google.golang.org/adk/v2/tool/toolauth"
)

func TestParseAuthResponse(t *testing.T) {
	answered := &toolauth.Credential{
		AuthType: toolauth.CredentialTypeOAuth2,
		OAuth2: &toolauth.OAuth2Auth{
			ClientID:        "app-123",
			State:           "Xy7",
			AuthResponseURI: "https://adk.example/cb?state=Xy7&code=4/0Ab",
			RedirectURI:     "https://adk.example/cb",
		},
	}
	scheme := map[string]any{
		"type": "oauth2",
		"flows": map[string]any{"authorizationCode": map[string]any{
			"authorizationUrl": "https://idp.example/authorize",
			"tokenUrl":         "https://idp.example/token",
			"scopes":           map[string]any{},
		}},
	}
	tests := []struct {
		name string
		resp map[string]any
		want *toolauth.Credential
	}{
		{
			name: "config as ADK Web sends it",
			resp: map[string]any{
				"authScheme": scheme,
				"exchangedAuthCredential": map[string]any{
					"authType": "oauth2",
					"oauth2": map[string]any{
						"clientId":        "app-123",
						"state":           "Xy7",
						"authResponseUri": "https://adk.example/cb?state=Xy7&code=4/0Ab",
						"redirectUri":     "https://adk.example/cb",
					},
				},
			},
			want: answered,
		},
		{
			name: "config in snake_case",
			resp: map[string]any{
				"auth_scheme": scheme,
				"exchanged_auth_credential": map[string]any{
					"auth_type": "oauth2",
					"oauth2": map[string]any{
						"client_id":         "app-123",
						"state":             "Xy7",
						"auth_response_uri": "https://adk.example/cb?state=Xy7&code=4/0Ab",
						"redirect_uri":      "https://adk.example/cb",
					},
				},
			},
			want: answered,
		},
		{
			name: "bare credential",
			resp: map[string]any{"authType": "apiKey", "apiKey": "k3y"},
			want: &toolauth.Credential{AuthType: toolauth.CredentialTypeAPIKey, APIKey: "k3y"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := parseAuthResponse(tt.resp)
			if err != nil {
				t.Fatalf("parseAuthResponse() error = %v", err)
			}
			if diff := cmp.Diff(tt.want, got); diff != "" {
				t.Errorf("parseAuthResponse() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestParseAuthResponseRejects(t *testing.T) {
	const planted = "PLANTED-SECRET"
	tests := []struct {
		name string
		resp map[string]any
	}{
		{
			name: "config without an exchanged credential",
			resp: map[string]any{"authScheme": map[string]any{"type": "apiKey", "in": "header", "name": planted}},
		},
		{
			name: "neither a config nor a credential",
			resp: map[string]any{"authType": planted, "code": planted},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := parseAuthResponse(tt.resp)
			if err == nil {
				t.Fatalf("parseAuthResponse() error = nil, want an error")
			}
			if strings.Contains(err.Error(), planted) {
				t.Errorf("error quotes a value from the answer: %v", err)
			}
		})
	}
}

// storedRequest is the request ADK keeps in the session for one sign-in: no
// client secret, the PKCE verifier still in.
func storedRequest() *toolauth.Config {
	return &toolauth.Config{
		AuthScheme: toolauth.Scheme{
			Type: toolauth.SchemeTypeOAuth2,
			Flows: &toolauth.OAuthFlows{AuthorizationCode: &toolauth.OAuthFlow{
				AuthorizationURL: "https://idp.example/authorize",
				TokenURL:         "https://idp.example/token",
				Scopes:           map[string]string{"a.scope": "A"},
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
				AuthURI:             "https://idp.example/authorize?state=Xy7",
				State:               "Xy7",
				RedirectURI:         "https://tool.example/cb",
				CodeVerifier:        "v-secret",
				CodeChallengeMethod: "S256",
			},
		},
		CredentialKey: "adk_k",
	}
}

// webAnswer is an answer like the one ADK Web sends back for storedRequest.
func webAnswer() *toolauth.Credential {
	return &toolauth.Credential{
		AuthType: toolauth.CredentialTypeOAuth2,
		OAuth2: &toolauth.OAuth2Auth{
			ClientID:        "app-123",
			State:           "Xy7",
			AuthResponseURI: "https://adk.example/cb?state=Xy7&code=4/0Ab",
			RedirectURI:     "https://adk.example/cb",
		},
	}
}

func TestPinAuthResponse(t *testing.T) {
	tests := []struct {
		name      string
		requested *toolauth.Config
		answer    *toolauth.Credential
		want      *toolauth.Credential
	}{
		{
			name:      "verifier and challenge method come back, the answer's redirect URI is kept",
			requested: storedRequest(),
			answer:    webAnswer(),
			want: &toolauth.Credential{
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
			name: "raw credential fills in when the request has no exchanged one",
			requested: &toolauth.Config{
				AuthScheme: storedRequest().AuthScheme,
				RawAuthCredential: &toolauth.Credential{
					AuthType: toolauth.CredentialTypeOAuth2,
					OAuth2: &toolauth.OAuth2Auth{
						ClientID:                "app-123",
						ClientSecret:            "s3cret",
						TokenEndpointAuthMethod: "client_secret_post",
					},
				},
			},
			answer: &toolauth.Credential{
				AuthType: toolauth.CredentialTypeOAuth2,
				OAuth2:   &toolauth.OAuth2Auth{AuthResponseURI: "https://adk.example/cb?code=4/0Ab"},
			},
			want: &toolauth.Credential{
				AuthType: toolauth.CredentialTypeOAuth2,
				OAuth2: &toolauth.OAuth2Auth{
					ClientID:                "app-123",
					ClientSecret:            "s3cret",
					AuthResponseURI:         "https://adk.example/cb?code=4/0Ab",
					TokenEndpointAuthMethod: "client_secret_post",
				},
			},
		},
		{
			name:      "an answer without oauth2 takes the request's whole section",
			requested: storedRequest(),
			answer:    &toolauth.Credential{AuthType: toolauth.CredentialTypeOAuth2},
			want: &toolauth.Credential{
				AuthType: toolauth.CredentialTypeOAuth2,
				OAuth2: &toolauth.OAuth2Auth{
					ClientID:            "app-123",
					AuthURI:             "https://idp.example/authorize?state=Xy7",
					State:               "Xy7",
					RedirectURI:         "https://tool.example/cb",
					CodeVerifier:        "v-secret",
					CodeChallengeMethod: "S256",
				},
			},
		},
		{
			name:      "the request's state fills in a blank one",
			requested: storedRequest(),
			answer: &toolauth.Credential{
				AuthType: toolauth.CredentialTypeOAuth2,
				OAuth2: &toolauth.OAuth2Auth{
					ClientID:        "app-123",
					AuthResponseURI: "https://adk.example/cb?code=4/0Ab",
					RedirectURI:     "https://adk.example/cb",
				},
			},
			want: &toolauth.Credential{
				AuthType: toolauth.CredentialTypeOAuth2,
				OAuth2: &toolauth.OAuth2Auth{
					ClientID:            "app-123",
					State:               "Xy7",
					AuthResponseURI:     "https://adk.example/cb?code=4/0Ab",
					RedirectURI:         "https://adk.example/cb",
					CodeVerifier:        "v-secret",
					CodeChallengeMethod: "S256",
				},
			},
		},
		{
			name:      "the request's state replaces the answer's",
			requested: storedRequest(),
			answer: &toolauth.Credential{
				AuthType: toolauth.CredentialTypeOAuth2,
				OAuth2: &toolauth.OAuth2Auth{
					ClientID:        "app-123",
					State:           "attacker",
					AuthResponseURI: "https://adk.example/cb?state=attacker&code=4/0Ab",
					RedirectURI:     "https://adk.example/cb",
				},
			},
			want: &toolauth.Credential{
				AuthType: toolauth.CredentialTypeOAuth2,
				OAuth2: &toolauth.OAuth2Auth{
					ClientID:            "app-123",
					State:               "Xy7",
					AuthResponseURI:     "https://adk.example/cb?state=attacker&code=4/0Ab",
					RedirectURI:         "https://adk.example/cb",
					CodeVerifier:        "v-secret",
					CodeChallengeMethod: "S256",
				},
			},
		},
		{
			// adk-python pins this in test_auth_resume.py: the token request has to
			// repeat the redirect URI the sign-in link used.
			name:      "the request's redirect URI fills in when the answer has none",
			requested: storedRequest(),
			answer: &toolauth.Credential{
				AuthType: toolauth.CredentialTypeOAuth2,
				OAuth2: &toolauth.OAuth2Auth{
					ClientID:        "app-123",
					State:           "Xy7",
					AuthResponseURI: "https://tool.example/cb?state=Xy7&code=4/0Ab",
				},
			},
			want: &toolauth.Credential{
				AuthType: toolauth.CredentialTypeOAuth2,
				OAuth2: &toolauth.OAuth2Auth{
					ClientID:            "app-123",
					State:               "Xy7",
					AuthResponseURI:     "https://tool.example/cb?state=Xy7&code=4/0Ab",
					RedirectURI:         "https://tool.example/cb",
					CodeVerifier:        "v-secret",
					CodeChallengeMethod: "S256",
				},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := pinAuthResponse(tt.requested, tt.answer)
			if diff := cmp.Diff(tt.want, got.ExchangedAuthCredential); diff != "" {
				t.Errorf("exchanged credential mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestPinAuthResponseKeepsTheRequest(t *testing.T) {
	got := pinAuthResponse(storedRequest(), webAnswer())
	ignoreExchanged := cmpopts.IgnoreFields(toolauth.Config{}, "ExchangedAuthCredential")
	if diff := cmp.Diff(storedRequest(), got, ignoreExchanged); diff != "" {
		t.Errorf("result differs from the request outside the exchanged credential (-want +got):\n%s", diff)
	}
}

// Callers fill the configured client into the result, so writing to it must
// not reach the request or the answer either.
func TestPinAuthResponseLeavesInputsUnchanged(t *testing.T) {
	tests := []struct {
		name   string
		answer func() *toolauth.Credential
	}{
		{"answer with oauth2", webAnswer},
		{"answer without oauth2", func() *toolauth.Credential {
			return &toolauth.Credential{AuthType: toolauth.CredentialTypeOAuth2}
		}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			requested, answer := storedRequest(), tt.answer()
			got := pinAuthResponse(requested, answer)
			got.ExchangedAuthCredential.OAuth2.ClientSecret = "written later"
			if diff := cmp.Diff(storedRequest(), requested); diff != "" {
				t.Errorf("request was modified (-want +got):\n%s", diff)
			}
			if diff := cmp.Diff(tt.answer(), answer); diff != "" {
				t.Errorf("answer was modified (-want +got):\n%s", diff)
			}
		})
	}
}
