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

func apiKeyConfig() toolauth.Config {
	return toolauth.Config{
		AuthScheme:        toolauth.Scheme{Type: toolauth.SchemeTypeAPIKey, In: "header", Name: "X-Api-Key"},
		RawAuthCredential: &toolauth.Credential{AuthType: toolauth.CredentialTypeAPIKey, APIKey: "k"},
		CredentialKey:     "my_api_key",
	}
}

const apiKeyConfigJSON = `{"authScheme":{"type":"apiKey","in":"header","name":"X-Api-Key"},"rawAuthCredential":{"authType":"apiKey","apiKey":"k"},"credentialKey":"my_api_key"}`

// pythonConfigCases pairs auth configs with the JSON adk-python 2.12.0 writes
// for the same config, from model_dump(by_alias=True, exclude_none=True). Each
// has an explicit credential key, which Python then writes as given.
var pythonConfigCases = []struct {
	name   string
	config toolauth.Config
	json   string
}{
	{
		name:   "API key",
		config: apiKeyConfig(),
		json:   apiKeyConfigJSON,
	},
	{
		name: "OAuth2 waiting for consent",
		config: toolauth.Config{
			AuthScheme: toolauth.Scheme{
				Type:  toolauth.SchemeTypeOAuth2,
				Flows: &toolauth.OAuthFlows{AuthorizationCode: &toolauth.OAuthFlow{AuthorizationURL: testAuthURL, TokenURL: testTokenURL}},
			},
			RawAuthCredential: &toolauth.Credential{
				AuthType: toolauth.CredentialTypeOAuth2,
				OAuth2:   &toolauth.OAuth2Auth{ClientID: "cid", ClientSecret: "sec", TokenEndpointAuthMethod: "client_secret_basic"},
			},
			ExchangedAuthCredential: &toolauth.Credential{
				AuthType: toolauth.CredentialTypeOAuth2,
				OAuth2: &toolauth.OAuth2Auth{
					ClientID:                "cid",
					AuthURI:                 testAuthURL + "?state=s1",
					State:                   "s1",
					TokenEndpointAuthMethod: "client_secret_basic",
				},
			},
			CredentialKey: "my_oauth_key",
		},
		json: `{"authScheme":{"type":"oauth2","flows":{"authorizationCode":{"scopes":{},"authorizationUrl":"https://example.com/authorize","tokenUrl":"https://example.com/token"}}},"rawAuthCredential":{"authType":"oauth2","oauth2":{"clientId":"cid","clientSecret":"sec","tokenEndpointAuthMethod":"client_secret_basic"}},"exchangedAuthCredential":{"authType":"oauth2","oauth2":{"clientId":"cid","authUri":"https://example.com/authorize?state=s1","state":"s1","tokenEndpointAuthMethod":"client_secret_basic"}},"credentialKey":"my_oauth_key"}`,
	},
	{
		name: "unknown key",
		config: toolauth.Config{
			AuthScheme:    toolauth.Scheme{Type: toolauth.SchemeTypeAPIKey, In: "header", Name: "X-Api-Key"},
			CredentialKey: "k1",
			Extra:         map[string]any{"x-note": "n"},
		},
		json: `{"authScheme":{"type":"apiKey","in":"header","name":"X-Api-Key"},"credentialKey":"k1","x-note":"n"}`,
	},
}

func TestConfigMarshalMatchesPython(t *testing.T) {
	for _, tc := range pythonConfigCases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := json.Marshal(tc.config)
			if err != nil {
				t.Fatalf("json.Marshal() error = %v", err)
			}
			if diff := cmp.Diff(decodeJSON(t, []byte(tc.json)), decodeJSON(t, got)); diff != "" {
				t.Errorf("json.Marshal() differs from adk-python (-want +got):\n%s", diff)
			}
		})
	}
}

func TestConfigUnmarshalPythonJSON(t *testing.T) {
	for _, tc := range pythonConfigCases {
		t.Run(tc.name, func(t *testing.T) {
			var got toolauth.Config
			if err := json.Unmarshal([]byte(tc.json), &got); err != nil {
				t.Fatalf("json.Unmarshal() error = %v", err)
			}
			if diff := cmp.Diff(tc.config, got, cmpopts.EquateEmpty()); diff != "" {
				t.Errorf("json.Unmarshal() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestConfigUnmarshalSnakeCase(t *testing.T) {
	// adk-python 2.12.0 reads this as the same config as apiKeyConfigJSON.
	data := `{"auth_scheme":{"type":"apiKey","in":"header","name":"X-Api-Key"},"raw_auth_credential":{"auth_type":"apiKey","api_key":"k"},"credential_key":"my_api_key"}`
	var got toolauth.Config
	if err := json.Unmarshal([]byte(data), &got); err != nil {
		t.Fatalf("json.Unmarshal() error = %v", err)
	}
	if diff := cmp.Diff(apiKeyConfig(), got, cmpopts.EquateEmpty()); diff != "" {
		t.Errorf("json.Unmarshal() mismatch (-want +got):\n%s", diff)
	}
}

func TestToolArgumentsMatchesPython(t *testing.T) {
	args := toolauth.ToolArguments{FunctionCallID: "call-1", AuthConfig: apiKeyConfig()}
	// What adk-python 2.12.0 writes for the same arguments.
	want := `{"functionCallId":"call-1","authConfig":` + apiKeyConfigJSON + `}`

	got, err := json.Marshal(args)
	if err != nil {
		t.Fatalf("json.Marshal() error = %v", err)
	}
	if diff := cmp.Diff(decodeJSON(t, []byte(want)), decodeJSON(t, got)); diff != "" {
		t.Errorf("json.Marshal() differs from adk-python (-want +got):\n%s", diff)
	}

	for name, data := range map[string]string{
		"camelCase": want,
		"snake_case": `{"function_call_id":"call-1","auth_config":{"auth_scheme":{"type":"apiKey","in":"header","name":"X-Api-Key"},` +
			`"raw_auth_credential":{"auth_type":"apiKey","api_key":"k"},"credential_key":"my_api_key"}}`,
	} {
		var decoded toolauth.ToolArguments
		if err := json.Unmarshal([]byte(data), &decoded); err != nil {
			t.Fatalf("json.Unmarshal(%s) error = %v", name, err)
		}
		if diff := cmp.Diff(args, decoded, cmpopts.EquateEmpty()); diff != "" {
			t.Errorf("json.Unmarshal(%s) mismatch (-want +got):\n%s", name, diff)
		}
	}
}

func TestConfigAndToolArgumentsUnmarshalRejects(t *testing.T) {
	// adk-python 2.12.0 rejects every one of these.
	tests := []struct {
		name    string
		data    string
		into    any
		wantErr string
	}{
		{
			name:    "config without authScheme",
			data:    `{"credentialKey":"k"}`,
			into:    &toolauth.Config{},
			wantErr: "missing required field authScheme",
		},
		{
			name:    "config with null authScheme",
			data:    `{"authScheme":null}`,
			into:    &toolauth.Config{},
			wantErr: "missing required field authScheme",
		},
		{
			name:    "arguments without functionCallId",
			data:    `{"authConfig":{"authScheme":{"type":"apiKey","in":"header","name":"X"}}}`,
			into:    &toolauth.ToolArguments{},
			wantErr: "missing required field functionCallId",
		},
		{
			name:    "arguments without authConfig",
			data:    `{"functionCallId":"c"}`,
			into:    &toolauth.ToolArguments{},
			wantErr: "missing required field authConfig",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := json.Unmarshal([]byte(tc.data), tc.into)
			if err == nil {
				t.Fatal("json.Unmarshal() succeeded, want an error")
			}
			if !strings.Contains(err.Error(), tc.wantErr) {
				t.Errorf("json.Unmarshal() error = %q, want it to contain %q", err, tc.wantErr)
			}
		})
	}
}

func TestRequestCredentialFunctionCallName(t *testing.T) {
	// ADK Web and adk-python recognize the request by this exact name.
	if toolauth.RequestCredentialFunctionCallName != "adk_request_credential" {
		t.Errorf("RequestCredentialFunctionCallName = %q, want %q", toolauth.RequestCredentialFunctionCallName, "adk_request_credential")
	}
}

func TestConfigKeyMatchesPython(t *testing.T) {
	// Each want is the credential_key adk-python 2.12.0 gives the same config
	// after AuthConfig.model_validate. The authorization URL carries & and the
	// non-ASCII case carries é and ’, which Python's canonical JSON writes
	// unescaped.
	tests := []struct {
		name string
		data string
		want string
	}{
		{
			name: "API key scheme, no credential",
			data: `{"authScheme":{"type":"apiKey","in":"header","name":"X-Api-Key"}}`,
			want: "adk_apiKey_12037b6b68d3604d_",
		},
		{
			name: "API key",
			data: `{"authScheme":{"type":"apiKey","in":"header","name":"X-Api-Key"},"rawAuthCredential":{"authType":"apiKey","apiKey":"k"}}`,
			want: "adk_apiKey_12037b6b68d3604d_apiKey_39fdaa2147b3f2d1",
		},
		{
			name: "HTTP bearer",
			data: `{"authScheme":{"type":"http","scheme":"bearer"},"rawAuthCredential":{"authType":"http","http":{"scheme":"bearer","credentials":{"token":"t"}}}}`,
			want: "adk_http_178204f758fb78d5_http_26adec9b11eb83bc",
		},
		{
			// The flow fields in the credential are left out of the key.
			name: "OAuth2 midway through a flow",
			data: `{"authScheme":{"type":"oauth2","flows":{"authorizationCode":{"authorizationUrl":"https://example.com/authorize?audience=api&prompt=consent","tokenUrl":"https://example.com/token","scopes":{"read":"Read"}}}},` +
				`"rawAuthCredential":{"authType":"oauth2","oauth2":{"clientId":"cid","clientSecret":"sec","redirectUri":"https://app/cb","authUri":"https://x","state":"s","accessToken":"at","expiresAt":5}}}`,
			want: "adk_oauth2_8e6640e02b286ea7_oauth2_0833ddacfe0bd302",
		},
		{
			// Ported from test_credential_key_is_stable_across_redirect_uri: the
			// same key as the case above, with another redirect URI and the flow
			// fields that case leaves unset.
			name: "OAuth2 with another redirect URI",
			data: `{"authScheme":{"type":"oauth2","flows":{"authorizationCode":{"authorizationUrl":"https://example.com/authorize?audience=api&prompt=consent","tokenUrl":"https://example.com/token","scopes":{"read":"Read"}}}},` +
				`"rawAuthCredential":{"authType":"oauth2","oauth2":{"clientId":"cid","clientSecret":"sec","redirectUri":"https://other/cb",` +
				`"authResponseUri":"https://other/cb?code=c","authCode":"c","refreshToken":"rt","expiresIn":3600}}}`,
			want: "adk_oauth2_8e6640e02b286ea7_oauth2_0833ddacfe0bd302",
		},
		{
			// These OAuth2 fields are not flow fields, so they count.
			name: "OAuth2 with an ID token, nonce and code verifier",
			data: `{"authScheme":{"type":"oauth2","flows":{"authorizationCode":{"authorizationUrl":"https://example.com/authorize?audience=api&prompt=consent","tokenUrl":"https://example.com/token","scopes":{"read":"Read"}}}},` +
				`"rawAuthCredential":{"authType":"oauth2","oauth2":{"clientId":"cid","clientSecret":"sec","idToken":"it","nonce":"n","codeVerifier":"cv"}}}`,
			want: "adk_oauth2_8e6640e02b286ea7_oauth2_466def9e19464a97",
		},
		{
			// The canonical JSON writes the number as it arrived; as a float64 it
			// would lose digits.
			name: "OAuth2 with a large number in a nested unknown key",
			data: `{"authScheme":{"type":"oauth2","flows":{"authorizationCode":{"authorizationUrl":"https://example.com/authorize?audience=api&prompt=consent","tokenUrl":"https://example.com/token","scopes":{"read":"Read"}}}},` +
				`"rawAuthCredential":{"authType":"oauth2","oauth2":{"clientId":"cid","x-count":12345678901234567890}}}`,
			want: "adk_oauth2_8e6640e02b286ea7_oauth2_1dc65c3746ce454a",
		},
		{
			// A nested unknown key counts towards the key, and a top-level one
			// does not.
			name: "OAuth2 with unknown keys",
			data: `{"authScheme":{"type":"oauth2","flows":{"authorizationCode":{"authorizationUrl":"https://example.com/authorize?audience=api&prompt=consent","tokenUrl":"https://example.com/token"}}},` +
				`"rawAuthCredential":{"authType":"oauth2","oauth2":{"clientId":"cid","scopes":["a"]},"x-top":1}}`,
			want: "adk_oauth2_6a7110d552770734_oauth2_589cf589f818b95c",
		},
		{
			name: "OpenID Connect endpoints",
			data: `{"authScheme":{"type":"openIdConnect","authorization_endpoint":"https://example.com/authorize?audience=api&prompt=consent","token_endpoint":"https://example.com/token","scopes":["openid"]},` +
				`"rawAuthCredential":{"authType":"openIdConnect","oauth2":{"clientId":"cid","clientSecret":"sec"}}}`,
			want: "adk_openIdConnect_4450fdfd7924efbf_openIdConnect_d62d4f4b945e72e7",
		},
		{
			name: "OpenID Connect discovery URL",
			data: `{"authScheme":{"type":"openIdConnect","openIdConnectUrl":"https://example.com/.well-known/openid-configuration"}}`,
			want: "adk_openIdConnect_480372faebed744f_",
		},
		{
			name: "service account",
			data: `{"authScheme":{"type":"http","scheme":"bearer"},"rawAuthCredential":{"authType":"serviceAccount","serviceAccount":{"useDefaultCredential":true,"scopes":["s"]}}}`,
			want: "adk_http_178204f758fb78d5_serviceAccount_739833674b9d134f",
		},
		{
			// An empty list or map counts towards the key, unlike a missing one.
			name: "service account with an empty scope list",
			data: `{"authScheme":{"type":"http","scheme":"bearer"},"rawAuthCredential":{"authType":"serviceAccount","serviceAccount":{"useDefaultCredential":true,"scopes":[]}}}`,
			want: "adk_http_178204f758fb78d5_serviceAccount_50785dccd95b1002",
		},
		{
			name: "HTTP bearer with an empty header map",
			data: `{"authScheme":{"type":"http","scheme":"bearer"},"rawAuthCredential":{"authType":"http","http":{"scheme":"bearer","credentials":{"token":"t"},"additionalHeaders":{}}}}`,
			want: "adk_http_178204f758fb78d5_http_b4147d4fceb4365c",
		},
		{
			name: "OpenID Connect endpoints with an empty scope list",
			data: `{"authScheme":{"type":"openIdConnect","authorization_endpoint":"https://example.com/authorize","token_endpoint":"https://example.com/token","scopes":[]}}`,
			want: "adk_openIdConnect_591dd13c93a8e50b_",
		},
		{
			name: "non-ASCII",
			data: `{"authScheme":{"type":"apiKey","in":"header","name":"X-Clé","description":"Clé d’API"}}`,
			want: "adk_apiKey_95c9f80bb80acb35_",
		},
		{
			// Ported from test_credential_key_with_custom_auth_scheme.
			name: "custom scheme",
			data: `{"authScheme":{"type":"mock_custom_type"}}`,
			want: "adk_mock_custom_type_0be4d0a15c0e9488_",
		},
		{
			// Ported from test_get_credential_key_with_extras: unknown keys on
			// the scheme leave the key unchanged.
			name: "custom scheme with an unknown key",
			data: `{"authScheme":{"type":"mock_custom_type","x-provider":"acme"}}`,
			want: "adk_mock_custom_type_0be4d0a15c0e9488_",
		},
		{
			name: "key among the credential's unknown keys",
			data: `{"authScheme":{"type":"apiKey","in":"header","name":"X","credential_key":"from_scheme"},"rawAuthCredential":{"authType":"apiKey","credentialKey":"from_cred"}}`,
			want: "from_cred",
		},
		{
			name: "empty key among the credential's unknown keys",
			data: `{"authScheme":{"type":"apiKey","in":"header","name":"X-Api-Key"},"rawAuthCredential":{"authType":"apiKey","apiKey":"k","credential_key":""}}`,
			want: "adk_apiKey_12037b6b68d3604d_apiKey_39fdaa2147b3f2d1",
		},
		{
			name: "key among the scheme's unknown keys",
			data: `{"authScheme":{"type":"apiKey","in":"header","name":"X","credential_key":"from_scheme"}}`,
			want: "from_scheme",
		},
		{
			// Ported from test_custom_credential_key.
			name: "explicit key",
			data: `{"authScheme":{"type":"apiKey","in":"header","name":"X","credential_key":"from_scheme"},"credentialKey":"explicit"}`,
			want: "explicit",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var config toolauth.Config
			if err := json.Unmarshal([]byte(tc.data), &config); err != nil {
				t.Fatalf("json.Unmarshal() error = %v", err)
			}
			got, err := config.Key()
			if err != nil {
				t.Fatalf("Key() error = %v", err)
			}
			if got != tc.want {
				t.Errorf("Key() = %q, want %q", got, tc.want)
			}
		})
	}
}

func TestConfigKeyMatchesPythonExtendedOAuth2(t *testing.T) {
	config := toolauth.Config{AuthScheme: toolauth.Scheme{
		Type: toolauth.SchemeTypeOAuth2,
		Flows: &toolauth.OAuthFlows{AuthorizationCode: &toolauth.OAuthFlow{
			AuthorizationURL: "https://example.com/authorize?audience=api&prompt=consent",
			TokenURL:         testTokenURL,
		}},
		IssuerURL: "https://issuer",
	}}
	got, err := config.Key()
	if err != nil {
		t.Fatalf("Key() error = %v", err)
	}
	// The key adk-python 2.12.0 gives an AuthConfig built from the same
	// ExtendedOAuth2 scheme.
	if want := "adk_oauth2_0154a991a741d11d_"; got != want {
		t.Errorf("Key() = %q, want %q", got, want)
	}
}

func TestConfigKeyCountsSchemeFieldsOfCustomScheme(t *testing.T) {
	// Name is a field of Scheme, so it counts for a custom type too.
	first := toolauth.Config{AuthScheme: toolauth.Scheme{Type: "gcp_provider", Name: "projects/p/locations/l/authProviders/a"}}
	second := toolauth.Config{AuthScheme: toolauth.Scheme{Type: "gcp_provider", Name: "projects/p/locations/l/authProviders/b"}}
	firstKey, err := first.Key()
	if err != nil {
		t.Fatalf("Key() error = %v", err)
	}
	secondKey, err := second.Key()
	if err != nil {
		t.Fatalf("Key() error = %v", err)
	}
	if firstKey == secondKey {
		t.Errorf("Key() = %q for both schemes, want different keys", firstKey)
	}
}

func TestConfigKeyFailsOnUnencodableExtra(t *testing.T) {
	config := toolauth.Config{AuthScheme: toolauth.Scheme{
		Type:  toolauth.SchemeTypeOAuth2,
		Flows: &toolauth.OAuthFlows{Extra: map[string]any{"x-callback": func() {}}},
	}}
	if got, err := config.Key(); err == nil {
		t.Errorf("Key() = %q, want an error", got)
	}
}

func TestConfigMarshalWritesKey(t *testing.T) {
	// Neither config sets CredentialKey. Each want is what adk-python 2.12.0
	// writes for the same config: the key it derived, or the one it found among
	// the credential's unknown keys, which also stays where it was.
	tests := []struct {
		name   string
		config toolauth.Config
		want   string
	}{
		{
			name: "derived key",
			config: toolauth.Config{
				AuthScheme:        toolauth.Scheme{Type: toolauth.SchemeTypeAPIKey, In: "header", Name: "X-Api-Key"},
				RawAuthCredential: &toolauth.Credential{AuthType: toolauth.CredentialTypeAPIKey, APIKey: "k"},
			},
			want: `{"authScheme":{"type":"apiKey","in":"header","name":"X-Api-Key"},"rawAuthCredential":{"authType":"apiKey","apiKey":"k"},"credentialKey":"adk_apiKey_12037b6b68d3604d_apiKey_39fdaa2147b3f2d1"}`,
		},
		{
			name: "key among the credential's unknown keys",
			config: toolauth.Config{
				AuthScheme: toolauth.Scheme{Type: toolauth.SchemeTypeAPIKey, In: "header", Name: "X"},
				RawAuthCredential: &toolauth.Credential{
					AuthType: toolauth.CredentialTypeAPIKey,
					Extra:    map[string]any{"credentialKey": "from_cred"},
				},
			},
			want: `{"authScheme":{"type":"apiKey","in":"header","name":"X"},"rawAuthCredential":{"authType":"apiKey","credentialKey":"from_cred"},"credentialKey":"from_cred"}`,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := json.Marshal(tc.config)
			if err != nil {
				t.Fatalf("json.Marshal() error = %v", err)
			}
			if diff := cmp.Diff(decodeJSON(t, []byte(tc.want)), decodeJSON(t, got)); diff != "" {
				t.Errorf("json.Marshal() differs from adk-python (-want +got):\n%s", diff)
			}
		})
	}
}

func TestConfigWithoutSecretsMatchesPython(t *testing.T) {
	// Each want is what adk-python 2.12.0's _without_configured_secrets returns
	// for the same config. Its credentialKey is the key the original derived,
	// with the secrets still in.
	tests := []struct {
		name string
		data string
		want string
	}{
		{
			// The PKCE verifier stays: Python needs it back on resume.
			name: "OAuth2 waiting for consent",
			data: `{"authScheme":{"type":"oauth2","flows":{"authorizationCode":{"authorizationUrl":"https://example.com/authorize","tokenUrl":"https://example.com/token"}}},` +
				`"rawAuthCredential":{"authType":"oauth2","oauth2":{"clientId":"cid","clientSecret":"sec"}},` +
				`"exchangedAuthCredential":{"authType":"oauth2","oauth2":{"clientId":"cid","clientSecret":"sec","authUri":"https://example.com/authorize?state=s1","state":"s1","codeVerifier":"v1","codeChallengeMethod":"S256"}}}`,
			want: `{"authScheme":{"type":"oauth2","flows":{"authorizationCode":{"scopes":{},"authorizationUrl":"https://example.com/authorize","tokenUrl":"https://example.com/token"}}},` +
				`"rawAuthCredential":{"authType":"oauth2","oauth2":{"clientId":"cid","tokenEndpointAuthMethod":"client_secret_basic"}},` +
				`"exchangedAuthCredential":{"authType":"oauth2","oauth2":{"clientId":"cid","authUri":"https://example.com/authorize?state=s1","state":"s1","codeVerifier":"v1","codeChallengeMethod":"S256","tokenEndpointAuthMethod":"client_secret_basic"}},` +
				`"credentialKey":"adk_oauth2_5a0ef268bdc8890b_oauth2_0833ddacfe0bd302"}`,
		},
		{
			name: "HTTP basic",
			data: `{"authScheme":{"type":"http","scheme":"basic"},` +
				`"rawAuthCredential":{"authType":"http","http":{"scheme":"basic","credentials":{"username":"u","password":"p","token":"t"},"additionalHeaders":{"X-Secret":"h"}}}}`,
			want: `{"authScheme":{"type":"http","scheme":"basic"},"rawAuthCredential":{"authType":"http","http":{"scheme":"basic","credentials":{"username":"u"}}},` +
				`"credentialKey":"adk_http_fc8df715f5002444_http_ac2b0193e6f6c376"}`,
		},
		{
			name: "API key",
			data: `{"authScheme":{"type":"apiKey","in":"header","name":"X-Api-Key"},"rawAuthCredential":{"authType":"apiKey","apiKey":"k"}}`,
			want: `{"authScheme":{"type":"apiKey","in":"header","name":"X-Api-Key"},"rawAuthCredential":{"authType":"apiKey"},` +
				`"credentialKey":"adk_apiKey_12037b6b68d3604d_apiKey_39fdaa2147b3f2d1"}`,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var config toolauth.Config
			if err := json.Unmarshal([]byte(tc.data), &config); err != nil {
				t.Fatalf("json.Unmarshal() error = %v", err)
			}
			stripped, err := config.WithoutSecrets()
			if err != nil {
				t.Fatalf("WithoutSecrets() error = %v", err)
			}
			got, err := json.Marshal(stripped)
			if err != nil {
				t.Fatalf("json.Marshal() error = %v", err)
			}
			if diff := cmp.Diff(decodeJSON(t, []byte(tc.want)), decodeJSON(t, got)); diff != "" {
				t.Errorf("WithoutSecrets() differs from adk-python (-want +got):\n%s", diff)
			}
		})
	}
}

func TestConfigWithoutSecretsDropsServiceAccountKey(t *testing.T) {
	// adk-python leaves the service account key in. ADK Go strips it with the
	// other secrets the agent was configured with.
	config := toolauth.Config{
		AuthScheme: toolauth.Scheme{Type: toolauth.SchemeTypeHTTP, Scheme: "bearer"},
		RawAuthCredential: &toolauth.Credential{
			AuthType:       toolauth.CredentialTypeServiceAccount,
			ServiceAccount: &toolauth.ServiceAccount{ServiceAccountCredential: saKey("service_account")},
		},
	}
	stripped, err := config.WithoutSecrets()
	if err != nil {
		t.Fatalf("WithoutSecrets() error = %v", err)
	}
	want := saKey("service_account")
	want.PrivateKey, want.PrivateKeyID = "", ""
	if diff := cmp.Diff(want, stripped.RawAuthCredential.ServiceAccount.ServiceAccountCredential); diff != "" {
		t.Errorf("WithoutSecrets() key mismatch (-want +got):\n%s", diff)
	}
}

func TestConfigWithoutSecretsCopiesConfigsBuiltInGo(t *testing.T) {
	// Neither the scheme, which has no type, nor the credential, which has no
	// authType, would decode from JSON. adk-python's model_copy copies whatever
	// the model holds, and so does WithoutSecrets, leaving every value as it
	// was.
	config := toolauth.Config{
		AuthScheme: toolauth.Scheme{Description: "d"},
		RawAuthCredential: &toolauth.Credential{
			APIKey: "k",
			OAuth2: &toolauth.OAuth2Auth{ClientID: "cid", ClientSecret: "sec"},
		},
		CredentialKey: "my_key",
		Extra:         map[string]any{"n": 5},
	}
	got, err := config.WithoutSecrets()
	if err != nil {
		t.Fatalf("WithoutSecrets() error = %v", err)
	}
	want := &toolauth.Config{
		AuthScheme:        toolauth.Scheme{Description: "d"},
		RawAuthCredential: &toolauth.Credential{OAuth2: &toolauth.OAuth2Auth{ClientID: "cid"}},
		CredentialKey:     "my_key",
		Extra:             map[string]any{"n": 5},
	}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("WithoutSecrets() mismatch (-want +got):\n%s", diff)
	}
}

// configWithEverySharedValue returns a config that sets every map, slice and
// pointer a copy could share with the original. Its credential holds an HTTP,
// an OAuth2 and a service account credential at once, which only Go code can
// build.
func configWithEverySharedValue() toolauth.Config {
	return toolauth.Config{
		AuthScheme: toolauth.Scheme{
			Type: toolauth.SchemeTypeOpenIDConnect,
			Flows: &toolauth.OAuthFlows{
				Implicit:          &toolauth.OAuthFlow{AuthorizationURL: testAuthURL},
				Password:          &toolauth.OAuthFlow{TokenURL: testTokenURL},
				ClientCredentials: &toolauth.OAuthFlow{TokenURL: testTokenURL},
				AuthorizationCode: &toolauth.OAuthFlow{
					AuthorizationURL: testAuthURL,
					TokenURL:         testTokenURL,
					Scopes:           map[string]string{"read": "Read"},
					Extra:            map[string]any{"x-flow": "f"},
				},
				Extra: map[string]any{"x-flows": "f"},
			},
			TokenEndpointAuthMethodsSupported: []string{"client_secret_basic"},
			GrantTypesSupported:               []string{"authorization_code"},
			Scopes:                            []string{"openid"},
			Extra:                             map[string]any{"x-nested": map[string]any{"list": []any{map[string]any{"k": "v"}}}},
		},
		RawAuthCredential: &toolauth.Credential{
			AuthType: toolauth.CredentialTypeHTTP,
			HTTP: &toolauth.HTTPAuth{
				Scheme:            "basic",
				Credentials:       toolauth.HTTPCredentials{Username: "u", Password: "p", Extra: map[string]any{"x-cred": "c"}},
				AdditionalHeaders: map[string]string{"X-Secret": "h"},
				Extra:             map[string]any{"x-http": "h"},
			},
			OAuth2: &toolauth.OAuth2Auth{ClientID: "cid", ClientSecret: "sec", Extra: map[string]any{"x-oauth2": "o"}},
			ServiceAccount: &toolauth.ServiceAccount{
				ServiceAccountCredential: &toolauth.ServiceAccountCredential{PrivateKey: "pk", Extra: map[string]any{"x-key": "k"}},
				Scopes:                   []string{"s"},
				Extra:                    map[string]any{"x-sa": "s"},
			},
			Extra: map[string]any{"x-raw": "r"},
		},
		ExchangedAuthCredential: &toolauth.Credential{
			AuthType: toolauth.CredentialTypeAPIKey,
			APIKey:   "k",
			Extra:    map[string]any{"x-exchanged": "e"},
		},
		Extra: map[string]any{"x-config": "c"},
	}
}

func TestConfigWithoutSecretsLeavesOriginalAlone(t *testing.T) {
	config := configWithEverySharedValue()
	stripped, err := config.WithoutSecrets()
	if err != nil {
		t.Fatalf("WithoutSecrets() error = %v", err)
	}

	scheme := &stripped.AuthScheme
	scheme.Flows.Implicit.RefreshURL = "changed"
	scheme.Flows.Password.RefreshURL = "changed"
	scheme.Flows.ClientCredentials.RefreshURL = "changed"
	scheme.Flows.AuthorizationCode.Scopes["read"] = "changed"
	scheme.Flows.AuthorizationCode.Extra["x-flow"] = "changed"
	scheme.Flows.Extra["x-flows"] = "changed"
	scheme.TokenEndpointAuthMethodsSupported[0] = "changed"
	scheme.GrantTypesSupported[0] = "changed"
	scheme.Scopes[0] = "changed"
	scheme.Extra["x-nested"].(map[string]any)["list"].([]any)[0].(map[string]any)["k"] = "changed"
	raw := stripped.RawAuthCredential
	raw.HTTP.Credentials.Username = "changed"
	raw.HTTP.Credentials.Extra["x-cred"] = "changed"
	raw.HTTP.Extra["x-http"] = "changed"
	raw.OAuth2.ClientID = "changed"
	raw.OAuth2.Extra["x-oauth2"] = "changed"
	raw.ServiceAccount.ServiceAccountCredential.ClientEmail = "changed"
	raw.ServiceAccount.ServiceAccountCredential.Extra["x-key"] = "changed"
	raw.ServiceAccount.Scopes[0] = "changed"
	raw.ServiceAccount.Extra["x-sa"] = "changed"
	raw.Extra["x-raw"] = "changed"
	stripped.ExchangedAuthCredential.Extra["x-exchanged"] = "changed"
	stripped.Extra["x-config"] = "changed"

	if diff := cmp.Diff(configWithEverySharedValue(), config); diff != "" {
		t.Errorf("original changed (-want +got):\n%s", diff)
	}
}
