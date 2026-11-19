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
	"bytes"
	"encoding/hex"
	"fmt"
	"log/slog"
	"strings"
	"testing"

	"google.golang.org/adk/v2/tool/toolauth"
)

func TestModelsKeepSecretsOutOfPrinting(t *testing.T) {
	// The first six cases port the repr and str tests in adk-python's
	// test_auth_credential.py. Each output is checked for every secret, and for
	// %x for its hex form too.
	tests := []struct {
		name    string
		value   any
		secrets []string
		// visible must still show in the %+v output.
		visible []string
	}{
		{
			name:    "API key",
			value:   toolauth.Credential{AuthType: toolauth.CredentialTypeAPIKey, APIKey: "sk-live-secret-api-key-12345"},
			secrets: []string{"sk-live-secret-api-key-12345"},
			visible: []string{"apiKey"},
		},
		{
			name: "HTTP credentials",
			value: toolauth.Credential{
				AuthType: toolauth.CredentialTypeHTTP,
				HTTP: &toolauth.HTTPAuth{
					Scheme:            "basic",
					Credentials:       toolauth.HTTPCredentials{Username: "my_user", Password: "secret_password_999", Token: "secret_token_abc"},
					AdditionalHeaders: map[string]string{"Authorization": "Bearer secret_bearer_token"},
				},
			},
			secrets: []string{"secret_password_999", "secret_token_abc", "secret_bearer_token"},
			visible: []string{"my_user", "Authorization"},
		},
		{
			name: "OAuth2 credentials",
			value: toolauth.Credential{
				AuthType: toolauth.CredentialTypeOAuth2,
				OAuth2: &toolauth.OAuth2Auth{
					ClientID:        "my_client_id",
					ClientSecret:    "top_secret_client_secret",
					AccessToken:     "secret_access_token",
					RefreshToken:    "secret_refresh_token",
					IDToken:         "secret_id_token",
					AuthCode:        "secret_auth_code",
					AuthResponseURI: "https://example.com/callback?code=secret_response_code",
					CodeVerifier:    "secret_code_verifier",
				},
			},
			secrets: []string{
				"top_secret_client_secret", "secret_access_token", "secret_refresh_token", "secret_id_token",
				"secret_auth_code", "secret_response_code", "secret_code_verifier",
			},
			visible: []string{"my_client_id"},
		},
		{
			name: "service account key",
			value: toolauth.ServiceAccountCredential{
				Type:                    "service_account",
				ProjectID:               "test_project",
				PrivateKeyID:            "secret_private_key_id",
				PrivateKey:              "-----BEGIN PRIVATE KEY-----\nsecret_key_data\n-----END PRIVATE KEY-----",
				ClientEmail:             "test@iam.gserviceaccount.com",
				ClientID:                "12345",
				AuthURI:                 "https://example.com/o/oauth2/auth",
				TokenURI:                "https://example.com/token",
				AuthProviderX509CertURL: "https://example.com/oauth2/v1/certs",
				ClientX509CertURL:       "https://example.com/robot/v1/metadata/x509/test",
				UniverseDomain:          "example.com",
			},
			secrets: []string{"secret_key_data", "secret_private_key_id"},
			visible: []string{"test@iam.gserviceaccount.com"},
		},
		{
			name: "unknown key",
			value: toolauth.Credential{
				AuthType: toolauth.CredentialTypeAPIKey,
				Extra:    map[string]any{"undeclared_secret": "secret_extra_value"},
			},
			secrets: []string{"secret_extra_value"},
			visible: []string{"undeclared_secret"},
		},
		{
			name: "nested unknown key",
			value: toolauth.Credential{
				AuthType: toolauth.CredentialTypeOAuth2,
				OAuth2: &toolauth.OAuth2Auth{
					ClientID: "my_client_id",
					Extra:    map[string]any{"unexpected_token": "secret_unexpected_token"},
				},
			},
			secrets: []string{"secret_unexpected_token"},
			visible: []string{"my_client_id", "unexpected_token"},
		},
		{
			name: "credential request",
			value: toolauth.ToolArguments{
				FunctionCallID: "call-1",
				AuthConfig: toolauth.Config{
					AuthScheme: toolauth.Scheme{Type: toolauth.SchemeTypeOAuth2, Flows: &toolauth.OAuthFlows{}},
					RawAuthCredential: &toolauth.Credential{
						AuthType: toolauth.CredentialTypeOAuth2,
						OAuth2:   &toolauth.OAuth2Auth{ClientID: "my_client_id", ClientSecret: "top_secret_client_secret"},
					},
					ExchangedAuthCredential: &toolauth.Credential{
						AuthType: toolauth.CredentialTypeServiceAccount,
						ServiceAccount: &toolauth.ServiceAccount{
							ServiceAccountCredential: &toolauth.ServiceAccountCredential{PrivateKey: "secret_key_data"},
							Extra:                    map[string]any{"x-sa": "secret_sa_extra"},
						},
					},
					Extra: map[string]any{"x-config": "secret_config_extra"},
				},
				Extra: map[string]any{"x-args": "secret_args_extra"},
			},
			secrets: []string{"top_secret_client_secret", "secret_key_data", "secret_sa_extra", "secret_config_extra", "secret_args_extra"},
			visible: []string{"call-1", "my_client_id", "x-config", "x-args"},
		},
		// The cases below log each model type on its own, where slog's JSON
		// handler would call MarshalJSON if the type had no LogValue.
		{
			name: "config",
			value: toolauth.Config{
				AuthScheme:        toolauth.Scheme{Type: toolauth.SchemeTypeAPIKey, In: "header", Name: "X-Api-Key"},
				RawAuthCredential: &toolauth.Credential{AuthType: toolauth.CredentialTypeAPIKey, APIKey: "sk-live-secret-api-key-12345"},
				Extra:             map[string]any{"x-config": "secret_config_extra"},
			},
			secrets: []string{"sk-live-secret-api-key-12345", "secret_config_extra"},
			visible: []string{"X-Api-Key", "x-config"},
		},
		{
			name: "OAuth2 credential",
			value: toolauth.OAuth2Auth{
				ClientID:     "my_client_id",
				ClientSecret: "top_secret_client_secret",
				Extra:        map[string]any{"x-oauth2": "secret_oauth2_extra"},
			},
			secrets: []string{"top_secret_client_secret", "secret_oauth2_extra"},
			visible: []string{"my_client_id", "x-oauth2"},
		},
		{
			name: "HTTP auth",
			value: toolauth.HTTPAuth{
				Scheme:            "basic",
				Credentials:       toolauth.HTTPCredentials{Username: "my_user", Password: "secret_password_999"},
				AdditionalHeaders: map[string]string{"X-Api-Key": "secret_header_value"},
				Extra:             map[string]any{"x-http": "secret_http_extra"},
			},
			secrets: []string{"secret_password_999", "secret_header_value", "secret_http_extra"},
			visible: []string{"my_user", "X-Api-Key", "x-http"},
		},
		{
			name: "HTTP credentials",
			value: toolauth.HTTPCredentials{
				Username: "my_user",
				Token:    "secret_token_abc",
				Extra:    map[string]any{"x-cred": "secret_cred_extra"},
			},
			secrets: []string{"secret_token_abc", "secret_cred_extra"},
			visible: []string{"my_user", "x-cred"},
		},
		{
			name: "service account",
			value: toolauth.ServiceAccount{
				ServiceAccountCredential: &toolauth.ServiceAccountCredential{
					ClientEmail: "test@iam.gserviceaccount.com",
					PrivateKey:  "secret_key_data",
					Extra:       map[string]any{"x-key": "secret_key_extra"},
				},
				Extra: map[string]any{"x-sa": "secret_sa_extra"},
			},
			secrets: []string{"secret_key_data", "secret_key_extra", "secret_sa_extra"},
			visible: []string{"test@iam.gserviceaccount.com", "x-key", "x-sa"},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var logged bytes.Buffer
			slog.New(slog.NewJSONHandler(&logged, nil)).Info("credential", "value", tc.value)
			outputs := map[string]string{"slog JSON": logged.String()}
			for _, verb := range []string{"%v", "%+v", "%#v", "%s", "%q", "%x"} {
				outputs[verb] = fmt.Sprintf(verb, tc.value)
			}
			for name, out := range outputs {
				for _, secret := range tc.secrets {
					if strings.Contains(out, secret) || strings.Contains(out, hex.EncodeToString([]byte(secret))) {
						t.Errorf("%s output contains the secret %q", name, secret)
					}
				}
			}
			for _, v := range tc.visible {
				if !strings.Contains(outputs["%+v"], v) {
					t.Errorf("%%+v output = %s, want it to show %q", outputs["%+v"], v)
				}
			}
		})
	}
}

func TestGoSyntaxPrintingNamesModelTypes(t *testing.T) {
	config := toolauth.Config{
		AuthScheme:        toolauth.Scheme{Type: toolauth.SchemeTypeAPIKey, In: "header", Name: "X"},
		RawAuthCredential: &toolauth.Credential{AuthType: toolauth.CredentialTypeAPIKey, APIKey: "k"},
	}
	got := fmt.Sprintf("%#v", config)
	// fmt prints a nil pointer to a type with a Format method as <nil>.
	want := `toolauth.Config{AuthScheme:toolauth.Scheme{Type:"apiKey", Description:"", In:"header", Name:"X", ` +
		`Scheme:"", BearerFormat:"", Flows:(*toolauth.OAuthFlows)(nil), IssuerURL:"", OpenIDConnectURL:"", ` +
		`AuthorizationEndpoint:"", TokenEndpoint:"", UserinfoEndpoint:"", RevocationEndpoint:"", ` +
		`TokenEndpointAuthMethodsSupported:[]string(nil), GrantTypesSupported:[]string(nil), Scopes:[]string(nil), ` +
		`Extra:map[string]interface {}(nil)}, ` +
		`RawAuthCredential:toolauth.Credential{AuthType:"apiKey", ResourceRef:"", APIKey:"<redacted>", ` +
		`HTTP:<nil>, ServiceAccount:<nil>, OAuth2:<nil>, Extra:map[string]interface {}(nil)}, ` +
		`ExchangedAuthCredential:<nil>, CredentialKey:"", Extra:map[string]interface {}(nil)}`
	if got != want {
		t.Errorf("%%#v output = %s, want %s", got, want)
	}
}

func TestPrintingLeavesSecretsInPlace(t *testing.T) {
	// Only the printed text is redacted; the values stay readable, as in
	// adk-python.
	cred := toolauth.Credential{
		AuthType: toolauth.CredentialTypeHTTP,
		HTTP: &toolauth.HTTPAuth{
			Scheme:            "basic",
			Credentials:       toolauth.HTTPCredentials{Password: "p"},
			AdditionalHeaders: map[string]string{"X-Secret": "h"},
		},
		Extra: map[string]any{"undeclared_secret": "v"},
	}
	_ = fmt.Sprintf("%+v", cred)

	if got := cred.HTTP.Credentials.Password; got != "p" {
		t.Errorf("Password = %q after printing, want %q", got, "p")
	}
	if got := cred.HTTP.AdditionalHeaders["X-Secret"]; got != "h" {
		t.Errorf(`AdditionalHeaders["X-Secret"] = %q after printing, want %q`, got, "h")
	}
	if got := cred.Extra["undeclared_secret"]; got != "v" {
		t.Errorf(`Extra["undeclared_secret"] = %v after printing, want %q`, got, "v")
	}
}
