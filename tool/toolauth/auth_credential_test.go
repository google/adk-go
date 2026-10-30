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
	"encoding/json"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"

	"google.golang.org/adk/v2/tool/toolauth"
)

func saKey(keyType string) *toolauth.ServiceAccountCredential {
	return &toolauth.ServiceAccountCredential{
		Type:                    keyType,
		ProjectID:               "proj",
		PrivateKeyID:            "kid",
		PrivateKey:              "-----BEGIN PRIVATE KEY-----\nabc\n-----END PRIVATE KEY-----\n",
		ClientEmail:             "sa@proj.iam.gserviceaccount.com",
		ClientID:                "123",
		AuthURI:                 "https://accounts.google.com/o/oauth2/auth",
		TokenURI:                "https://oauth2.googleapis.com/token",
		AuthProviderX509CertURL: "https://www.googleapis.com/oauth2/v1/certs",
		ClientX509CertURL:       "https://www.googleapis.com/robot/v1/metadata/x509/sa",
		UniverseDomain:          "googleapis.com",
	}
}

// pythonCases pairs credentials with the JSON adk-python 2.12.0 writes for the
// same credential, from model_dump(exclude_none=True): camel by alias, snake by
// field name.
var pythonCases = []struct {
	name  string
	cred  toolauth.Credential
	camel string
	snake string
}{
	{
		name:  "API key",
		cred:  toolauth.Credential{AuthType: toolauth.CredentialTypeAPIKey, APIKey: "test-api-key"},
		camel: `{"authType":"apiKey","apiKey":"test-api-key"}`,
		snake: `{"auth_type":"apiKey","api_key":"test-api-key"}`,
	},
	{
		name: "resource ref",
		cred: toolauth.Credential{
			AuthType:    toolauth.CredentialTypeAPIKey,
			ResourceRef: "projects/1/locations/us/resources/r",
		},
		camel: `{"authType":"apiKey","resourceRef":"projects/1/locations/us/resources/r"}`,
		snake: `{"auth_type":"apiKey","resource_ref":"projects/1/locations/us/resources/r"}`,
	},
	{
		name:  "HTTP bearer with no credentials",
		cred:  toolauth.Credential{AuthType: toolauth.CredentialTypeHTTP, HTTP: &toolauth.HTTPAuth{Scheme: "bearer"}},
		camel: `{"authType":"http","http":{"scheme":"bearer","credentials":{}}}`,
		snake: `{"auth_type":"http","http":{"scheme":"bearer","credentials":{}}}`,
	},
	{
		name: "HTTP basic with a header",
		cred: toolauth.Credential{
			AuthType: toolauth.CredentialTypeHTTP,
			HTTP: &toolauth.HTTPAuth{
				Scheme:            "basic",
				Credentials:       toolauth.HTTPCredentials{Username: "user", Password: "pass"},
				AdditionalHeaders: map[string]string{"X-Goog-User-Project": "proj"},
			},
		},
		camel: `{"authType":"http","http":{"scheme":"basic","credentials":{"username":"user","password":"pass"},"additionalHeaders":{"X-Goog-User-Project":"proj"}}}`,
		snake: `{"auth_type":"http","http":{"scheme":"basic","credentials":{"username":"user","password":"pass"},"additional_headers":{"X-Goog-User-Project":"proj"}}}`,
	},
	{
		// An empty map is written, where a nil one is not.
		name: "HTTP basic with an empty header map",
		cred: toolauth.Credential{
			AuthType: toolauth.CredentialTypeHTTP,
			HTTP:     &toolauth.HTTPAuth{Scheme: "basic", AdditionalHeaders: map[string]string{}},
		},
		camel: `{"authType":"http","http":{"scheme":"basic","credentials":{},"additionalHeaders":{}}}`,
		snake: `{"auth_type":"http","http":{"scheme":"basic","credentials":{},"additional_headers":{}}}`,
	},
	{
		name: "OAuth2 client",
		cred: toolauth.Credential{
			AuthType: toolauth.CredentialTypeOAuth2,
			OAuth2:   &toolauth.OAuth2Auth{ClientID: "client", TokenEndpointAuthMethod: "client_secret_basic"},
		},
		camel: `{"authType":"oauth2","oauth2":{"clientId":"client","tokenEndpointAuthMethod":"client_secret_basic"}}`,
		snake: `{"auth_type":"oauth2","oauth2":{"client_id":"client","token_endpoint_auth_method":"client_secret_basic"}}`,
	},
	{
		name: "OAuth2 with every field",
		cred: toolauth.Credential{
			AuthType: toolauth.CredentialTypeOAuth2,
			OAuth2: &toolauth.OAuth2Auth{
				ClientID:                "client",
				ClientSecret:            "secret",
				AuthURI:                 "https://example.com/auth",
				Nonce:                   "n",
				State:                   "s",
				RedirectURI:             "https://example.com/cb",
				AuthResponseURI:         "https://example.com/cb?code=c",
				AuthCode:                "c",
				AccessToken:             "at",
				RefreshToken:            "rt",
				IDToken:                 "it",
				ExpiresAt:               1767225600,
				ExpiresIn:               3600,
				Audience:                "aud",
				Prompt:                  "consent",
				CodeVerifier:            "cv",
				CodeChallengeMethod:     "S256",
				TokenEndpointAuthMethod: "client_secret_post",
			},
		},
		camel: `{"authType":"oauth2","oauth2":{"clientId":"client","clientSecret":"secret","authUri":"https://example.com/auth","nonce":"n","state":"s","redirectUri":"https://example.com/cb","authResponseUri":"https://example.com/cb?code=c","authCode":"c","accessToken":"at","refreshToken":"rt","idToken":"it","expiresAt":1767225600,"expiresIn":3600,"audience":"aud","prompt":"consent","codeVerifier":"cv","codeChallengeMethod":"S256","tokenEndpointAuthMethod":"client_secret_post"}}`,
		snake: `{"auth_type":"oauth2","oauth2":{"client_id":"client","client_secret":"secret","auth_uri":"https://example.com/auth","nonce":"n","state":"s","redirect_uri":"https://example.com/cb","auth_response_uri":"https://example.com/cb?code=c","auth_code":"c","access_token":"at","refresh_token":"rt","id_token":"it","expires_at":1767225600,"expires_in":3600,"audience":"aud","prompt":"consent","code_verifier":"cv","code_challenge_method":"S256","token_endpoint_auth_method":"client_secret_post"}}`,
	},
	{
		name: "OpenID Connect client",
		cred: toolauth.Credential{
			AuthType: toolauth.CredentialTypeOpenIDConnect,
			OAuth2: &toolauth.OAuth2Auth{
				ClientID:                "client",
				ClientSecret:            "secret",
				RedirectURI:             "https://example.com/cb",
				TokenEndpointAuthMethod: "client_secret_basic",
			},
		},
		camel: `{"authType":"openIdConnect","oauth2":{"clientId":"client","clientSecret":"secret","redirectUri":"https://example.com/cb","tokenEndpointAuthMethod":"client_secret_basic"}}`,
		snake: `{"auth_type":"openIdConnect","oauth2":{"client_id":"client","client_secret":"secret","redirect_uri":"https://example.com/cb","token_endpoint_auth_method":"client_secret_basic"}}`,
	},
	{
		name: "service account through ADC",
		cred: toolauth.Credential{
			AuthType: toolauth.CredentialTypeServiceAccount,
			ServiceAccount: &toolauth.ServiceAccount{
				Scopes:               []string{"https://www.googleapis.com/auth/cloud-platform"},
				UseDefaultCredential: true,
			},
		},
		camel: `{"authType":"serviceAccount","serviceAccount":{"scopes":["https://www.googleapis.com/auth/cloud-platform"],"useDefaultCredential":true,"useIdToken":false}}`,
		snake: `{"auth_type":"serviceAccount","service_account":{"scopes":["https://www.googleapis.com/auth/cloud-platform"],"use_default_credential":true,"use_id_token":false}}`,
	},
	{
		name: "service account through ADC with an empty scope list",
		cred: toolauth.Credential{
			AuthType:       toolauth.CredentialTypeServiceAccount,
			ServiceAccount: &toolauth.ServiceAccount{Scopes: []string{}, UseDefaultCredential: true},
		},
		camel: `{"authType":"serviceAccount","serviceAccount":{"scopes":[],"useDefaultCredential":true,"useIdToken":false}}`,
		snake: `{"auth_type":"serviceAccount","service_account":{"scopes":[],"use_default_credential":true,"use_id_token":false}}`,
	},
	{
		name: "service account key for an ID token",
		cred: toolauth.Credential{
			AuthType: toolauth.CredentialTypeServiceAccount,
			ServiceAccount: &toolauth.ServiceAccount{
				ServiceAccountCredential: saKey("service_account"),
				UseIDToken:               true,
				Audience:                 "https://svc.run.app",
			},
		},
		camel: `{"authType":"serviceAccount","serviceAccount":{"serviceAccountCredential":{"type":"service_account","projectId":"proj","privateKeyId":"kid","privateKey":"-----BEGIN PRIVATE KEY-----\nabc\n-----END PRIVATE KEY-----\n","clientEmail":"sa@proj.iam.gserviceaccount.com","clientId":"123","authUri":"https://accounts.google.com/o/oauth2/auth","tokenUri":"https://oauth2.googleapis.com/token","authProviderX509CertUrl":"https://www.googleapis.com/oauth2/v1/certs","clientX509CertUrl":"https://www.googleapis.com/robot/v1/metadata/x509/sa","universeDomain":"googleapis.com"},"useDefaultCredential":false,"useIdToken":true,"audience":"https://svc.run.app"}}`,
		snake: `{"auth_type":"serviceAccount","service_account":{"service_account_credential":{"type_":"service_account","project_id":"proj","private_key_id":"kid","private_key":"-----BEGIN PRIVATE KEY-----\nabc\n-----END PRIVATE KEY-----\n","client_email":"sa@proj.iam.gserviceaccount.com","client_id":"123","auth_uri":"https://accounts.google.com/o/oauth2/auth","token_uri":"https://oauth2.googleapis.com/token","auth_provider_x509_cert_url":"https://www.googleapis.com/oauth2/v1/certs","client_x509_cert_url":"https://www.googleapis.com/robot/v1/metadata/x509/sa","universe_domain":"googleapis.com"},"use_default_credential":false,"use_id_token":true,"audience":"https://svc.run.app"}}`,
	},
	{
		name: "service account key with no type",
		cred: toolauth.Credential{
			AuthType:       toolauth.CredentialTypeServiceAccount,
			ServiceAccount: &toolauth.ServiceAccount{ServiceAccountCredential: saKey("")},
		},
		camel: `{"authType":"serviceAccount","serviceAccount":{"serviceAccountCredential":{"type":"","projectId":"proj","privateKeyId":"kid","privateKey":"-----BEGIN PRIVATE KEY-----\nabc\n-----END PRIVATE KEY-----\n","clientEmail":"sa@proj.iam.gserviceaccount.com","clientId":"123","authUri":"https://accounts.google.com/o/oauth2/auth","tokenUri":"https://oauth2.googleapis.com/token","authProviderX509CertUrl":"https://www.googleapis.com/oauth2/v1/certs","clientX509CertUrl":"https://www.googleapis.com/robot/v1/metadata/x509/sa","universeDomain":"googleapis.com"},"useDefaultCredential":false,"useIdToken":false}}`,
		snake: `{"auth_type":"serviceAccount","service_account":{"service_account_credential":{"type_":"","project_id":"proj","private_key_id":"kid","private_key":"-----BEGIN PRIVATE KEY-----\nabc\n-----END PRIVATE KEY-----\n","client_email":"sa@proj.iam.gserviceaccount.com","client_id":"123","auth_uri":"https://accounts.google.com/o/oauth2/auth","token_uri":"https://oauth2.googleapis.com/token","auth_provider_x509_cert_url":"https://www.googleapis.com/oauth2/v1/certs","client_x509_cert_url":"https://www.googleapis.com/robot/v1/metadata/x509/sa","universe_domain":"googleapis.com"},"use_default_credential":false,"use_id_token":false}}`,
	},
}

// decodeJSON decodes data for a comparison that ignores key order and spacing.
// Numbers stay as written.
func decodeJSON(t *testing.T, data []byte) any {
	t.Helper()
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.UseNumber()
	var v any
	if err := dec.Decode(&v); err != nil {
		t.Fatalf("decoding JSON: %v", err)
	}
	return v
}

func TestCredentialMarshalMatchesPython(t *testing.T) {
	for _, tc := range pythonCases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := json.Marshal(tc.cred)
			if err != nil {
				t.Fatalf("json.Marshal() error = %v", err)
			}
			if diff := cmp.Diff(decodeJSON(t, []byte(tc.camel)), decodeJSON(t, got)); diff != "" {
				t.Errorf("json.Marshal() differs from adk-python (-want +got):\n%s", diff)
			}
		})
	}
}

func TestCredentialUnmarshalPythonJSON(t *testing.T) {
	for _, tc := range pythonCases {
		for _, form := range []struct{ name, data string }{{"camel", tc.camel}, {"snake", tc.snake}} {
			t.Run(tc.name+"/"+form.name, func(t *testing.T) {
				var got toolauth.Credential
				if err := json.Unmarshal([]byte(form.data), &got); err != nil {
					t.Fatalf("json.Unmarshal() error = %v", err)
				}
				if diff := cmp.Diff(tc.cred, got); diff != "" {
					t.Errorf("json.Unmarshal() mismatch (-want +got):\n%s", diff)
				}
			})
		}
	}
}

func TestOAuth2AuthMarshalWritesDefaultTokenEndpointAuthMethod(t *testing.T) {
	got, err := json.Marshal(toolauth.OAuth2Auth{ClientID: "client"})
	if err != nil {
		t.Fatalf("json.Marshal() error = %v", err)
	}
	// What adk-python 2.12.0 writes for OAuth2Auth(client_id="client").
	want := `{"clientId":"client","tokenEndpointAuthMethod":"client_secret_basic"}`
	if diff := cmp.Diff(decodeJSON(t, []byte(want)), decodeJSON(t, got)); diff != "" {
		t.Errorf("json.Marshal() differs from adk-python (-want +got):\n%s", diff)
	}
}

func TestCredentialUnmarshalPrefersCamelCase(t *testing.T) {
	// adk-python 2.12.0 reads the alias when an object has both names. Every
	// field here comes under both, because which name a decoder without that
	// preference picks depends on map iteration order, and one field alone
	// would catch it only on some runs.
	data := `{"authType":"oauth2","auth_type":"apiKey","oauth2":{` +
		`"clientId":"camel","client_id":"snake","clientSecret":"camel","client_secret":"snake",` +
		`"authUri":"camel","auth_uri":"snake","redirectUri":"camel","redirect_uri":"snake",` +
		`"authResponseUri":"camel","auth_response_uri":"snake","authCode":"camel","auth_code":"snake",` +
		`"accessToken":"camel","access_token":"snake","refreshToken":"camel","refresh_token":"snake",` +
		`"idToken":"camel","id_token":"snake","codeVerifier":"camel","code_verifier":"snake",` +
		`"codeChallengeMethod":"camel","code_challenge_method":"snake",` +
		`"expiresAt":1,"expires_at":2,"expiresIn":1,"expires_in":2,` +
		`"tokenEndpointAuthMethod":"client_secret_post","token_endpoint_auth_method":"client_secret_jwt"}}`
	var got toolauth.Credential
	if err := json.Unmarshal([]byte(data), &got); err != nil {
		t.Fatalf("json.Unmarshal() error = %v", err)
	}
	// Extra stays nil: the snake_case keys name fields, so they are dropped.
	want := toolauth.Credential{
		AuthType: toolauth.CredentialTypeOAuth2,
		OAuth2: &toolauth.OAuth2Auth{
			ClientID:                "camel",
			ClientSecret:            "camel",
			AuthURI:                 "camel",
			RedirectURI:             "camel",
			AuthResponseURI:         "camel",
			AuthCode:                "camel",
			AccessToken:             "camel",
			RefreshToken:            "camel",
			IDToken:                 "camel",
			CodeVerifier:            "camel",
			CodeChallengeMethod:     "camel",
			ExpiresAt:               1,
			ExpiresIn:               1,
			TokenEndpointAuthMethod: "client_secret_post",
		},
	}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("json.Unmarshal() mismatch (-want +got):\n%s", diff)
	}
}

func TestServiceAccountCredentialReadsKeyFile(t *testing.T) {
	keyFile := `{
  "type": "service_account",
  "project_id": "proj",
  "private_key_id": "kid",
  "private_key": "-----BEGIN PRIVATE KEY-----\nabc\n-----END PRIVATE KEY-----\n",
  "client_email": "sa@proj.iam.gserviceaccount.com",
  "client_id": "123",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/sa",
  "universe_domain": "googleapis.com"
}`
	var got toolauth.ServiceAccountCredential
	if err := json.Unmarshal([]byte(keyFile), &got); err != nil {
		t.Fatalf("json.Unmarshal() error = %v", err)
	}
	if diff := cmp.Diff(*saKey("service_account"), got); diff != "" {
		t.Errorf("json.Unmarshal() mismatch (-want +got):\n%s", diff)
	}
}

func TestCredentialKeepsUnknownKeys(t *testing.T) {
	// The input, and the output adk-python 2.12.0 writes for it.
	const in = `{"authType":"oauth2","oauth2":{"clientId":"c","scopes":["a"],"rapt_token":"r"},"tenantId":"t","count":12345678901234567890}`
	const want = `{"authType":"oauth2","oauth2":{"clientId":"c","tokenEndpointAuthMethod":"client_secret_basic","scopes":["a"],"rapt_token":"r"},"tenantId":"t","count":12345678901234567890}`
	var cred toolauth.Credential
	if err := json.Unmarshal([]byte(in), &cred); err != nil {
		t.Fatalf("json.Unmarshal() error = %v", err)
	}
	wantExtra := map[string]any{"tenantId": "t", "count": json.Number("12345678901234567890")}
	if diff := cmp.Diff(wantExtra, cred.Extra); diff != "" {
		t.Errorf("Extra mismatch (-want +got):\n%s", diff)
	}
	if cred.OAuth2 == nil {
		t.Fatal("OAuth2 = nil, want the decoded credential")
	}
	wantOAuth2Extra := map[string]any{"scopes": []any{"a"}, "rapt_token": "r"}
	if diff := cmp.Diff(wantOAuth2Extra, cred.OAuth2.Extra); diff != "" {
		t.Errorf("OAuth2.Extra mismatch (-want +got):\n%s", diff)
	}
	got, err := json.Marshal(cred)
	if err != nil {
		t.Fatalf("json.Marshal() error = %v", err)
	}
	if diff := cmp.Diff(decodeJSON(t, []byte(want)), decodeJSON(t, got)); diff != "" {
		t.Errorf("json.Marshal() differs from adk-python (-want +got):\n%s", diff)
	}
}

func TestCredentialMarshalSkipsExtraKeysThatNameFields(t *testing.T) {
	cred := toolauth.Credential{
		AuthType: toolauth.CredentialTypeAPIKey,
		APIKey:   "k",
		Extra:    map[string]any{"apiKey": "x", "api_key": "y", "tenantId": "t"},
	}
	got, err := json.Marshal(cred)
	if err != nil {
		t.Fatalf("json.Marshal() error = %v", err)
	}
	// Compared as bytes, because decoding would hide a key written twice.
	want := `{"authType":"apiKey","apiKey":"k","tenantId":"t"}`
	if string(got) != want {
		t.Errorf("json.Marshal() = %s, want %s", got, want)
	}
}

func TestCredentialUnmarshalRejects(t *testing.T) {
	// adk-python 2.12.0 rejects every one of these.
	tests := []struct {
		name    string
		data    string
		wantErr string
		// secret must not appear in the error.
		secret string
	}{
		{name: "unknown authType", data: `{"authType":"api_key"}`, wantErr: "unknown authType"},
		{name: "no authType", data: `{}`, wantErr: "missing required field authType"},
		{name: "null authType", data: `{"authType":null}`, wantErr: "missing required field authType"},
		{
			name:    "HTTP without credentials",
			data:    `{"authType":"http","http":{"scheme":"basic"}}`,
			wantErr: "missing required field credentials",
		},
		{
			name:    "HTTP with null credentials",
			data:    `{"authType":"http","http":{"scheme":"basic","credentials":null}}`,
			wantErr: "missing required field credentials",
		},
		{
			name:    "HTTP without scheme",
			data:    `{"authType":"http","http":{"credentials":{}}}`,
			wantErr: "missing required field scheme",
		},
		{
			name:    "unsupported tokenEndpointAuthMethod",
			data:    `{"authType":"oauth2","oauth2":{"tokenEndpointAuthMethod":"tls_client_auth"}}`,
			wantErr: "unsupported tokenEndpointAuthMethod",
		},
		{
			// adk-python uses none only in the token request it builds for a
			// client without a secret, never in the model.
			name:    "none as tokenEndpointAuthMethod",
			data:    `{"authType":"oauth2","oauth2":{"clientId":"c","tokenEndpointAuthMethod":"none"}}`,
			wantErr: "unsupported tokenEndpointAuthMethod",
		},
		{
			name:    "service account with neither a key nor ADC",
			data:    `{"authType":"serviceAccount","serviceAccount":{}}`,
			wantErr: "serviceAccountCredential is required",
		},
		{
			name:    "ID token without an audience",
			data:    `{"authType":"serviceAccount","serviceAccount":{"useDefaultCredential":true,"useIdToken":true}}`,
			wantErr: "audience is required",
		},
		{
			name:    "key with missing fields",
			data:    `{"authType":"serviceAccount","serviceAccount":{"serviceAccountCredential":{"projectId":"p"}}}`,
			wantErr: "missing required field privateKeyId",
		},
		{
			name:    "API key as a list",
			data:    `{"authType":"apiKey","apiKey":["sk-live-secret-api-key-12345"]}`,
			wantErr: "field apiKey",
			secret:  "sk-live-secret-api-key-12345",
		},
		{
			name:    "API key as a number",
			data:    `{"authType":"apiKey","apiKey":12345678901}`,
			wantErr: "field apiKey",
			secret:  "12345678901",
		},
		{
			// encoding/json quotes a number in full only when it overflows an
			// integer field.
			name:    "number too large for expiresAt",
			data:    `{"authType":"oauth2","oauth2":{"expiresAt":98765432109876543210}}`,
			wantErr: "field expiresAt",
			secret:  "98765432109876543210",
		},
		{
			name:    "credential that is not an object",
			data:    `{"authType":"oauth2","oauth2":"secret-string"}`,
			wantErr: "want a JSON object",
			secret:  "secret-string",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var cred toolauth.Credential
			err := json.Unmarshal([]byte(tc.data), &cred)
			if err == nil {
				t.Fatal("json.Unmarshal() succeeded, want an error")
			}
			if tc.secret != "" && strings.Contains(err.Error(), tc.secret) {
				t.Fatal("json.Unmarshal() error quotes the rejected value")
			}
			if !strings.Contains(err.Error(), tc.wantErr) {
				t.Errorf("json.Unmarshal() error = %q, want it to contain %q", err, tc.wantErr)
			}
		})
	}
}
