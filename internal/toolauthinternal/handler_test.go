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

package toolauthinternal

import (
	"net/http"
	"net/url"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"golang.org/x/oauth2"

	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool/toolauth"
)

// newState returns an empty session state.
func newState(t *testing.T) session.State {
	t.Helper()
	svc := session.InMemoryService()
	resp, err := svc.Create(t.Context(), &session.CreateRequest{AppName: "app", UserID: "user"})
	if err != nil {
		t.Fatalf("Create() error = %v", err)
	}
	return resp.Session.State()
}

// stateValue returns the value under key, or nil when there is none.
func stateValue(t *testing.T, state session.State, key string) any {
	t.Helper()
	v, err := state.Get(key)
	if err != nil {
		return nil
	}
	return v
}

func mustKey(t *testing.T, cfg *toolauth.Config) string {
	t.Helper()
	key, err := cfg.Key()
	if err != nil {
		t.Fatalf("Key() error = %v", err)
	}
	return key
}

// parseAuthURI returns the parsed authorization URI of cred.
func parseAuthURI(t *testing.T, cred *toolauth.Credential) *url.URL {
	t.Helper()
	u, err := url.Parse(cred.OAuth2.AuthURI)
	if err != nil {
		t.Fatalf("auth URI does not parse: %v", err)
	}
	return u
}

// assertNoConfiguredSecrets fails when cfg carries a secret a tool is
// configured with.
func assertNoConfiguredSecrets(t *testing.T, cfg *toolauth.Config) {
	t.Helper()
	for name, c := range map[string]*toolauth.Credential{"raw": cfg.RawAuthCredential, "exchanged": cfg.ExchangedAuthCredential} {
		if c == nil {
			continue
		}
		if c.APIKey != "" {
			t.Errorf("%s credential has an API key", name)
		}
		if c.OAuth2 != nil && c.OAuth2.ClientSecret != "" {
			t.Errorf("%s credential has a client secret", name)
		}
		if c.HTTP != nil && (c.HTTP.Credentials.Password != "" || c.HTTP.Credentials.Token != "" || c.HTTP.AdditionalHeaders != nil) {
			t.Errorf("%s credential has an HTTP secret", name)
		}
	}
}

func TestGenerateURI(t *testing.T) {
	// reference: test_generate_auth_uri_oauth2
	t.Run("OAuth2 scheme", func(t *testing.T) {
		raw := oauth2Credential()
		got, err := GenerateURI(raw, oauth2Scheme("https://example.com/oauth2/token"))
		if err != nil {
			t.Fatalf("GenerateURI() error = %v", err)
		}
		u := parseAuthURI(t, got)
		if u.Scheme+"://"+u.Host+u.Path != authorizeURL {
			t.Errorf("auth URI endpoint = %s://%s%s, want %s", u.Scheme, u.Host, u.Path, authorizeURL)
		}
		q := u.Query()
		want := map[string]string{
			"client_id":     "mock_client_id",
			"response_type": "code",
			"scope":         "read write",
			"access_type":   "offline",
			"prompt":        "consent",
			"redirect_uri":  redirectURL,
		}
		for k, v := range want {
			if q.Get(k) != v {
				t.Errorf("auth URI %s = %q, want %q", k, q.Get(k), v)
			}
		}
		if got.OAuth2.State == "" || q.Get("state") != got.OAuth2.State {
			t.Errorf("auth URI state does not match the credential's non-empty State")
		}
		for _, k := range []string{"code_challenge", "client_secret"} {
			if q.Has(k) {
				t.Errorf("auth URI has %s, want none for a confidential client", k)
			}
		}
		if got.OAuth2.ClientSecret != "mock_client_secret" || got.OAuth2.CodeVerifier != "" {
			t.Errorf("GenerateURI() changed the client fields of the copy")
		}
		if raw.OAuth2.AuthURI != "" || raw.OAuth2.State != "" {
			t.Errorf("GenerateURI() changed the raw credential")
		}
	})

	// reference: test_generate_auth_uri_openid
	t.Run("OpenID Connect scheme", func(t *testing.T) {
		got, err := GenerateURI(oauth2Credential(), openIDScheme("https://example.com/oauth2/token"))
		if err != nil {
			t.Fatalf("GenerateURI() error = %v", err)
		}
		u := parseAuthURI(t, got)
		if !strings.HasPrefix(got.OAuth2.AuthURI, authorizeURL) {
			t.Errorf("auth URI does not start with %s", authorizeURL)
		}
		q := u.Query()
		if q.Get("client_id") != "mock_client_id" || q.Get("scope") != "openid profile email" {
			t.Errorf("auth URI client_id, scope = %q, %q; want mock_client_id, openid profile email", q.Get("client_id"), q.Get("scope"))
		}
		if got.OAuth2.State == "" || q.Get("state") != got.OAuth2.State {
			t.Errorf("auth URI state does not match the credential's non-empty State")
		}
	})

	t.Run("state is new each time", func(t *testing.T) {
		a, err := GenerateURI(oauth2Credential(), oauth2Scheme(""))
		if err != nil {
			t.Fatal(err)
		}
		b, err := GenerateURI(oauth2Credential(), oauth2Scheme(""))
		if err != nil {
			t.Fatal(err)
		}
		if a.OAuth2.State == b.OAuth2.State {
			t.Errorf("two calls returned the same state")
		}
	})

	t.Run("public client uses PKCE S256", func(t *testing.T) {
		raw := oauth2Credential()
		raw.OAuth2.ClientSecret = ""
		got, err := GenerateURI(raw, oauth2Scheme(""))
		if err != nil {
			t.Fatalf("GenerateURI() error = %v", err)
		}
		q := parseAuthURI(t, got).Query()
		if got.OAuth2.CodeVerifier == "" || got.OAuth2.CodeChallengeMethod != "S256" {
			t.Fatalf("credential has verifier %t and method %q; want a verifier and S256", got.OAuth2.CodeVerifier != "", got.OAuth2.CodeChallengeMethod)
		}
		if q.Get("code_challenge_method") != "S256" || q.Get("code_challenge") != oauth2.S256ChallengeFromVerifier(got.OAuth2.CodeVerifier) {
			t.Errorf("auth URI does not carry the S256 challenge of the credential's verifier")
		}
	})

	t.Run("S256 with a configured verifier", func(t *testing.T) {
		raw := oauth2Credential()
		raw.OAuth2.CodeChallengeMethod = "S256"
		raw.OAuth2.CodeVerifier = "configured-verifier-0123456789-0123456789-0123"
		got, err := GenerateURI(raw, oauth2Scheme(""))
		if err != nil {
			t.Fatalf("GenerateURI() error = %v", err)
		}
		if got.OAuth2.CodeVerifier != raw.OAuth2.CodeVerifier {
			t.Errorf("credential verifier is not the configured one")
		}
		if parseAuthURI(t, got).Query().Get("code_challenge") != oauth2.S256ChallengeFromVerifier(raw.OAuth2.CodeVerifier) {
			t.Errorf("auth URI does not carry the challenge of the configured verifier")
		}
	})

	t.Run("audience, prompt and nonce", func(t *testing.T) {
		raw := oauth2Credential()
		raw.OAuth2.Audience = "https://api.example.com"
		raw.OAuth2.Prompt = "select_account"
		raw.OAuth2.Nonce = "mock_nonce"
		got, err := GenerateURI(raw, oauth2Scheme(""))
		if err != nil {
			t.Fatalf("GenerateURI() error = %v", err)
		}
		q := parseAuthURI(t, got).Query()
		for k, want := range map[string]string{"audience": raw.OAuth2.Audience, "prompt": "select_account", "nonce": "mock_nonce"} {
			if q.Get(k) != want {
				t.Errorf("auth URI %s = %q, want %q", k, q.Get(k), want)
			}
		}
	})

	t.Run("client credentials flow token URL is the endpoint", func(t *testing.T) {
		scheme := toolauth.Scheme{Type: toolauth.SchemeTypeOAuth2, Flows: &toolauth.OAuthFlows{
			ClientCredentials: &toolauth.OAuthFlow{TokenURL: "https://example.com/token", Scopes: map[string]string{"b": "", "a": ""}},
		}}
		got, err := GenerateURI(oauth2Credential(), scheme)
		if err != nil {
			t.Fatalf("GenerateURI() error = %v", err)
		}
		u := parseAuthURI(t, got)
		if u.Host+u.Path != "example.com/token" || u.Query().Get("scope") != "a b" {
			t.Errorf("auth URI endpoint, scope = %s%s, %q; want example.com/token, a b", u.Host, u.Path, u.Query().Get("scope"))
		}
	})

	errorCases := []struct {
		name   string
		raw    func() *toolauth.Credential
		scheme toolauth.Scheme
	}{
		{"nil raw credential", func() *toolauth.Credential { return nil }, oauth2Scheme("")},
		{"raw credential without oauth2", func() *toolauth.Credential {
			return &toolauth.Credential{AuthType: toolauth.CredentialTypeAPIKey, APIKey: "k"}
		}, oauth2Scheme("")},
		{"scheme without flows", oauth2Credential, toolauth.Scheme{Type: toolauth.SchemeTypeOAuth2}},
		{"flows without an endpoint", oauth2Credential, toolauth.Scheme{Type: toolauth.SchemeTypeOAuth2, Flows: &toolauth.OAuthFlows{}}},
		{"unsupported code challenge method", func() *toolauth.Credential {
			c := oauth2Credential()
			c.OAuth2.CodeChallengeMethod = "plain"
			return c
		}, oauth2Scheme("")},
	}
	for _, tc := range errorCases {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := GenerateURI(tc.raw(), tc.scheme); err == nil {
				t.Errorf("GenerateURI() error = nil, want one")
			}
		})
	}
}

func TestGenerateRequest(t *testing.T) {
	scheme := oauth2Scheme("https://example.com/oauth2/token")

	// reference: test_non_oauth_scheme
	t.Run("non-OAuth scheme", func(t *testing.T) {
		cred := &toolauth.Credential{AuthType: toolauth.CredentialTypeAPIKey, APIKey: "test_api_key"}
		cfg := &toolauth.Config{
			AuthScheme:              toolauth.Scheme{Type: toolauth.SchemeTypeAPIKey, In: "header", Name: "test_api_key"},
			RawAuthCredential:       cred,
			ExchangedAuthCredential: &toolauth.Credential{AuthType: toolauth.CredentialTypeAPIKey, APIKey: "test_api_key"},
		}
		got, err := GenerateRequest(cfg)
		if err != nil {
			t.Fatalf("GenerateRequest() error = %v", err)
		}
		want := &toolauth.Config{
			AuthScheme:              cfg.AuthScheme,
			RawAuthCredential:       &toolauth.Credential{AuthType: toolauth.CredentialTypeAPIKey},
			ExchangedAuthCredential: &toolauth.Credential{AuthType: toolauth.CredentialTypeAPIKey},
			CredentialKey:           mustKey(t, cfg),
		}
		if diff := cmp.Diff(want, got); diff != "" {
			t.Errorf("GenerateRequest() mismatch (-want +got):\n%s", diff)
		}
		if cred.APIKey != "test_api_key" {
			t.Errorf("GenerateRequest() changed the config it was given")
		}
	})

	// reference: test_with_existing_auth_uri
	t.Run("exchanged credential already has an auth URI", func(t *testing.T) {
		cfg := &toolauth.Config{AuthScheme: scheme, RawAuthCredential: oauth2Credential(), ExchangedAuthCredential: oauth2CredentialWithAuthCode()}
		got, err := GenerateRequest(cfg)
		if err != nil {
			t.Fatalf("GenerateRequest() error = %v", err)
		}
		if got.ExchangedAuthCredential.OAuth2.AuthURI != cfg.ExchangedAuthCredential.OAuth2.AuthURI {
			t.Errorf("auth URI = %q, want the existing one", got.ExchangedAuthCredential.OAuth2.AuthURI)
		}
		assertNoConfiguredSecrets(t, got)
	})

	// reference: test_missing_raw_credential
	t.Run("missing raw credential", func(t *testing.T) {
		_, err := GenerateRequest(&toolauth.Config{AuthScheme: scheme})
		if err == nil || !strings.Contains(err.Error(), "requires a raw auth credential") {
			t.Errorf("GenerateRequest() error = %v, want a missing raw credential", err)
		}
	})

	// reference: test_missing_oauth2_in_raw_credential
	t.Run("missing oauth2 in raw credential", func(t *testing.T) {
		cred := &toolauth.Credential{AuthType: toolauth.CredentialTypeAPIKey, APIKey: "test_api_key"}
		_, err := GenerateRequest(&toolauth.Config{AuthScheme: scheme, RawAuthCredential: cred, ExchangedAuthCredential: cred})
		if err == nil || !strings.Contains(err.Error(), "requires oauth2 in the raw auth credential") {
			t.Errorf("GenerateRequest() error = %v, want a missing oauth2 section", err)
		}
	})

	// reference: test_auth_uri_in_raw_credential
	t.Run("auth URI in raw credential", func(t *testing.T) {
		raw := oauth2CredentialWithAuthCode()
		got, err := GenerateRequest(&toolauth.Config{AuthScheme: scheme, RawAuthCredential: raw})
		if err != nil {
			t.Fatalf("GenerateRequest() error = %v", err)
		}
		if got.ExchangedAuthCredential.OAuth2.AuthURI != raw.OAuth2.AuthURI {
			t.Errorf("auth URI = %q, want the raw credential's", got.ExchangedAuthCredential.OAuth2.AuthURI)
		}
		assertNoConfiguredSecrets(t, got)
	})

	// reference: test_missing_client_credentials
	t.Run("missing client ID", func(t *testing.T) {
		cred := &toolauth.Credential{AuthType: toolauth.CredentialTypeOAuth2, OAuth2: &toolauth.OAuth2Auth{RedirectURI: redirectURL}}
		_, err := GenerateRequest(&toolauth.Config{AuthScheme: scheme, RawAuthCredential: cred, ExchangedAuthCredential: cred})
		if err == nil || !strings.Contains(err.Error(), "requires a client ID") {
			t.Errorf("GenerateRequest() error = %v, want a missing client ID", err)
		}
	})

	// The reference also required a client secret; adk-python now accepts a
	// public client and uses PKCE for it.
	t.Run("public client without a secret", func(t *testing.T) {
		raw := oauth2Credential()
		raw.OAuth2.ClientSecret = ""
		got, err := GenerateRequest(&toolauth.Config{AuthScheme: scheme, RawAuthCredential: raw})
		if err != nil {
			t.Fatalf("GenerateRequest() error = %v", err)
		}
		if o := got.ExchangedAuthCredential.OAuth2; o.CodeChallengeMethod != "S256" || o.CodeVerifier == "" {
			t.Errorf("exchanged credential has method %q and verifier %t; want S256 and a verifier", o.CodeChallengeMethod, o.CodeVerifier != "")
		}
	})

	// reference: test_generate_new_auth_uri
	t.Run("generates a new auth URI", func(t *testing.T) {
		cfg := &toolauth.Config{AuthScheme: scheme, RawAuthCredential: oauth2Credential(), ExchangedAuthCredential: oauth2Credential()}
		key := mustKey(t, cfg)
		got, err := GenerateRequest(cfg)
		if err != nil {
			t.Fatalf("GenerateRequest() error = %v", err)
		}
		o := got.ExchangedAuthCredential.OAuth2
		if !strings.HasPrefix(o.AuthURI, authorizeURL+"?") || o.State == "" {
			t.Errorf("exchanged credential has no new auth URI and state")
		}
		if o.ClientID != "mock_client_id" || o.RedirectURI != redirectURL {
			t.Errorf("exchanged credential lost the client ID or redirect URI")
		}
		if got.CredentialKey != key {
			t.Errorf("CredentialKey = %q, want %q", got.CredentialKey, key)
		}
		assertNoConfiguredSecrets(t, got)
		if cfg.ExchangedAuthCredential.OAuth2.AuthURI != "" || cfg.RawAuthCredential.OAuth2.ClientSecret == "" {
			t.Errorf("GenerateRequest() changed the config it was given")
		}
	})

	t.Run("keeps a configured credential key", func(t *testing.T) {
		cfg := &toolauth.Config{AuthScheme: scheme, RawAuthCredential: oauth2Credential(), CredentialKey: "my_key"}
		got, err := GenerateRequest(cfg)
		if err != nil {
			t.Fatalf("GenerateRequest() error = %v", err)
		}
		if got.CredentialKey != "my_key" {
			t.Errorf("CredentialKey = %q, want my_key", got.CredentialKey)
		}
	})

	t.Run("strips HTTP secrets", func(t *testing.T) {
		cred := &toolauth.Credential{AuthType: toolauth.CredentialTypeHTTP, HTTP: &toolauth.HTTPAuth{
			Scheme:            "basic",
			Credentials:       toolauth.HTTPCredentials{Username: "u", Password: "p", Token: "t"},
			AdditionalHeaders: map[string]string{"X-Key": "v"},
		}}
		got, err := GenerateRequest(&toolauth.Config{
			AuthScheme:        toolauth.Scheme{Type: toolauth.SchemeTypeHTTP, Scheme: "basic"},
			RawAuthCredential: cred,
		})
		if err != nil {
			t.Fatalf("GenerateRequest() error = %v", err)
		}
		assertNoConfiguredSecrets(t, got)
		if got.RawAuthCredential.HTTP.Credentials.Username != "u" {
			t.Errorf("username was dropped, want it kept")
		}
	})
}

func TestResponse(t *testing.T) {
	newConfig := func(tokenURL string) *toolauth.Config {
		return &toolauth.Config{AuthScheme: oauth2Scheme(tokenURL), RawAuthCredential: oauth2Credential(), ExchangedAuthCredential: oauth2Credential()}
	}

	// reference: test_get_auth_response_exists
	t.Run("stored credential", func(t *testing.T) {
		srv := newTokenServer(t, nil)
		cfg := newConfig(srv.URL)
		state := newState(t)
		stored := oauth2Credential()
		stored.OAuth2.AuthURI = authorizeURL + "?client_id=mock_client_id"
		stored.OAuth2.State = "mock_state"
		if err := state.Set("temp:"+mustKey(t, cfg), stored); err != nil {
			t.Fatal(err)
		}
		got, err := Response(t.Context(), cfg, state)
		if err != nil {
			t.Fatalf("Response() error = %v", err)
		}
		if diff := cmp.Diff(stored, got); diff != "" {
			t.Errorf("Response() mismatch (-want +got):\n%s", diff)
		}
		if n := len(srv.recorded()); n != 0 {
			t.Errorf("token endpoint got %d requests, want 0", n)
		}
	})

	// reference: test_get_auth_response_not_exists
	t.Run("nothing stored", func(t *testing.T) {
		got, err := Response(t.Context(), newConfig(""), newState(t))
		if err != nil || got != nil {
			t.Errorf("Response() = %v, %v; want nil, nil", got, err)
		}
	})

	apiKeyConfig := &toolauth.Config{AuthScheme: toolauth.Scheme{Type: toolauth.SchemeTypeAPIKey, In: "header", Name: "X-Key"}}
	bearerConfig := &toolauth.Config{AuthScheme: toolauth.Scheme{Type: toolauth.SchemeTypeHTTP, Scheme: "bearer"}}
	basicConfig := &toolauth.Config{AuthScheme: toolauth.Scheme{Type: toolauth.SchemeTypeHTTP, Scheme: "basic"}}
	for _, tc := range []struct {
		name   string
		cfg    *toolauth.Config
		prefix string
		value  any
		want   *toolauth.Credential
	}{
		{
			name:   "API key string under the temp key",
			cfg:    apiKeyConfig,
			prefix: "temp:",
			value:  "k1",
			want:   &toolauth.Credential{AuthType: toolauth.CredentialTypeAPIKey, APIKey: "k1"},
		},
		{
			name:  "bearer token string under the bare key",
			cfg:   bearerConfig,
			value: "t1",
			want: &toolauth.Credential{AuthType: toolauth.CredentialTypeHTTP, HTTP: &toolauth.HTTPAuth{
				Scheme: "bearer", Credentials: toolauth.HTTPCredentials{Token: "t1"},
			}},
		},
		{
			name:  "basic auth string is not a credential",
			cfg:   basicConfig,
			value: "t1",
		},
		{
			name:   "JSON object",
			cfg:    apiKeyConfig,
			prefix: "temp:",
			value:  map[string]any{"authType": "apiKey", "apiKey": "k2"},
			want:   &toolauth.Credential{AuthType: toolauth.CredentialTypeAPIKey, APIKey: "k2"},
		},
		{
			name:   "malformed JSON object is skipped",
			cfg:    apiKeyConfig,
			prefix: "temp:",
			value:  map[string]any{"authType": "nope"},
		},
		{
			name:   "credential value",
			cfg:    apiKeyConfig,
			prefix: "temp:",
			value:  toolauth.Credential{AuthType: toolauth.CredentialTypeAPIKey, APIKey: "k3"},
			want:   &toolauth.Credential{AuthType: toolauth.CredentialTypeAPIKey, APIKey: "k3"},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			state := newState(t)
			if err := state.Set(tc.prefix+mustKey(t, tc.cfg), tc.value); err != nil {
				t.Fatal(err)
			}
			got, err := Response(t.Context(), tc.cfg, state)
			if err != nil {
				t.Fatalf("Response() error = %v", err)
			}
			if diff := cmp.Diff(tc.want, got); diff != "" {
				t.Errorf("Response() mismatch (-want +got):\n%s", diff)
			}
		})
	}

	t.Run("temp key wins over the bare key", func(t *testing.T) {
		state := newState(t)
		key := mustKey(t, apiKeyConfig)
		if err := state.Set(key, "bare"); err != nil {
			t.Fatal(err)
		}
		if err := state.Set("temp:"+key, "temp"); err != nil {
			t.Fatal(err)
		}
		got, err := Response(t.Context(), apiKeyConfig, state)
		if err != nil || got == nil || got.APIKey != "temp" {
			t.Errorf("Response() did not return the credential under the temp key (err %v)", err)
		}
	})

	t.Run("pins the configured client and exchanges a public client", func(t *testing.T) {
		srv := newTokenServer(t, nil)
		cfg := newConfig(srv.URL)
		cfg.RawAuthCredential.OAuth2.ClientID = "configured_client"
		cfg.RawAuthCredential.OAuth2.ClientSecret = ""
		key := "temp:" + mustKey(t, cfg)
		stored := oauth2CredentialWithAuthCode()
		stored.OAuth2.ClientID = "client_chosen_by_the_client"
		stored.OAuth2.ClientSecret = ""
		state := newState(t)
		if err := state.Set(key, stored); err != nil {
			t.Fatal(err)
		}
		got, err := Response(t.Context(), cfg, state)
		if err != nil {
			t.Fatalf("Response() error = %v", err)
		}
		reqs := srv.recorded()
		if len(reqs) != 1 {
			t.Fatalf("token endpoint got %d requests, want 1", len(reqs))
		}
		if reqs[0].form.Get("client_id") != "configured_client" || reqs[0].form.Has("client_secret") {
			t.Errorf("form client_id = %q and has secret %t; want configured_client and no secret", reqs[0].form.Get("client_id"), reqs[0].form.Has("client_secret"))
		}
		if got.OAuth2.AccessToken != "mock_access_token" {
			t.Errorf("Response() did not return the exchanged token")
		}
		written, ok := stateValue(t, state, key).(*toolauth.Credential)
		if !ok || written.OAuth2.AccessToken != "mock_access_token" {
			t.Errorf("state does not hold the exchanged credential")
		}
	})

	t.Run("writes the exchanged credential back without the secret", func(t *testing.T) {
		srv := newTokenServer(t, nil)
		cfg := newConfig(srv.URL)
		key := mustKey(t, cfg)
		state := newState(t)
		stored := oauth2CredentialWithAuthCode()
		stored.OAuth2.ClientSecret = ""
		if err := state.Set(key, stored); err != nil {
			t.Fatal(err)
		}
		got, err := Response(t.Context(), cfg, state)
		if err != nil {
			t.Fatalf("Response() error = %v", err)
		}
		if got.OAuth2.ClientSecret != "mock_client_secret" {
			t.Errorf("returned credential lacks the configured secret")
		}
		req := srv.recorded()[0]
		if req.header.Get("Authorization") == "" {
			t.Errorf("exchange did not authenticate with the configured secret")
		}
		written, ok := stateValue(t, state, key).(*toolauth.Credential)
		if !ok || written.OAuth2.ClientSecret != "" || written.OAuth2.AccessToken == "" {
			t.Errorf("state under the bare key does not hold the exchanged credential without its secret")
		}
	})
}

func TestParseAndStoreResponse(t *testing.T) {
	// reference: test_non_oauth_scheme
	t.Run("non-OAuth scheme stores the credential as it is", func(t *testing.T) {
		srv := newTokenServer(t, nil)
		cfg := &toolauth.Config{
			AuthScheme:              toolauth.Scheme{Type: toolauth.SchemeTypeAPIKey, In: "header", Name: "test_api_key"},
			RawAuthCredential:       oauth2Credential(),
			ExchangedAuthCredential: oauth2CredentialWithAuthCode(),
		}
		state := newState(t)
		if err := ParseAndStoreResponse(t.Context(), cfg, state); err != nil {
			t.Fatalf("ParseAndStoreResponse() error = %v", err)
		}
		want := oauth2CredentialWithAuthCode()
		want.OAuth2.ClientSecret = ""
		if diff := cmp.Diff(want, stateValue(t, state, "temp:"+mustKey(t, cfg))); diff != "" {
			t.Errorf("stored credential mismatch (-want +got):\n%s", diff)
		}
		if n := len(srv.recorded()); n != 0 {
			t.Errorf("token endpoint got %d requests, want 0", n)
		}
	})

	// reference: test_oauth_scheme
	t.Run("OAuth2 scheme exchanges the code", func(t *testing.T) {
		srv := newTokenServer(t, nil)
		cfg := &toolauth.Config{
			AuthScheme:              oauth2Scheme(srv.URL),
			RawAuthCredential:       oauth2Credential(),
			ExchangedAuthCredential: oauth2CredentialWithAuthCode(),
		}
		cfg.ExchangedAuthCredential.OAuth2.ClientID = "client_chosen_by_the_client"
		cfg.ExchangedAuthCredential.OAuth2.ClientSecret = ""
		state := newState(t)
		if err := ParseAndStoreResponse(t.Context(), cfg, state); err != nil {
			t.Fatalf("ParseAndStoreResponse() error = %v", err)
		}
		stored, ok := stateValue(t, state, "temp:"+mustKey(t, cfg)).(*toolauth.Credential)
		if !ok {
			t.Fatalf("state holds no credential")
		}
		if stored.OAuth2.AccessToken != "mock_access_token" || stored.OAuth2.ClientSecret != "" {
			t.Errorf("stored credential has token %t and secret %t; want a token and no secret", stored.OAuth2.AccessToken != "", stored.OAuth2.ClientSecret != "")
		}
		reqs := srv.recorded()
		if len(reqs) != 1 {
			t.Fatalf("token endpoint got %d requests, want 1", len(reqs))
		}
		if user, _, _ := (&http.Request{Header: reqs[0].header}).BasicAuth(); user != "mock_client_id" {
			t.Errorf("exchange used client %q, want the configured mock_client_id", user)
		}
	})

	t.Run("public client is stored without an exchange", func(t *testing.T) {
		srv := newTokenServer(t, nil)
		raw := oauth2Credential()
		raw.OAuth2.ClientSecret = ""
		cfg := &toolauth.Config{AuthScheme: oauth2Scheme(srv.URL), RawAuthCredential: raw, ExchangedAuthCredential: oauth2CredentialWithAuthCode()}
		state := newState(t)
		if err := ParseAndStoreResponse(t.Context(), cfg, state); err != nil {
			t.Fatalf("ParseAndStoreResponse() error = %v", err)
		}
		stored, ok := stateValue(t, state, "temp:"+mustKey(t, cfg)).(*toolauth.Credential)
		if !ok || stored.OAuth2.AuthCode != "mock_auth_code" || stored.OAuth2.AccessToken != "" {
			t.Errorf("state does not hold the unexchanged credential")
		}
		if n := len(srv.recorded()); n != 0 {
			t.Errorf("token endpoint got %d requests, want 0", n)
		}
	})

	t.Run("no raw credential strips client secret from exchanged credential without an exchange", func(t *testing.T) {
		srv := newTokenServer(t, nil)
		cfg := &toolauth.Config{
			AuthScheme:              oauth2Scheme(srv.URL),
			ExchangedAuthCredential: oauth2CredentialWithAuthCode(),
		}
		state := newState(t)
		if err := ParseAndStoreResponse(t.Context(), cfg, state); err != nil {
			t.Fatalf("ParseAndStoreResponse() error = %v", err)
		}
		if n := len(srv.recorded()); n != 0 {
			t.Errorf("token endpoint got %d requests, want 0", n)
		}
		stored, ok := stateValue(t, state, "temp:"+mustKey(t, cfg)).(*toolauth.Credential)
		if !ok || stored.OAuth2.ClientSecret != "" || stored.OAuth2.AccessToken != "" {
			t.Errorf("stored credential has secret %t or token %t; want neither", stored.OAuth2.ClientSecret != "", stored.OAuth2.AccessToken != "")
		}
	})

	t.Run("failed exchange stores nothing", func(t *testing.T) {
		srv := newTokenServer(t, func(w http.ResponseWriter, r *http.Request) {
			http.Error(w, "no", http.StatusUnauthorized)
		})
		cfg := &toolauth.Config{AuthScheme: oauth2Scheme(srv.URL), RawAuthCredential: oauth2Credential(), ExchangedAuthCredential: oauth2CredentialWithAuthCode()}
		state := newState(t)
		if err := ParseAndStoreResponse(t.Context(), cfg, state); err == nil {
			t.Errorf("ParseAndStoreResponse() error = nil, want one")
		}
		if v := stateValue(t, state, "temp:"+mustKey(t, cfg)); v != nil {
			t.Errorf("state holds %T, want nothing", v)
		}
	})
}

func TestCredentialFromString(t *testing.T) {
	if _, err := credentialFromString("", toolauth.Scheme{Type: toolauth.SchemeTypeAPIKey}); err == nil {
		t.Errorf("credentialFromString(\"\") error = nil, want one")
	}
	for _, scheme := range []toolauth.Scheme{{}, {Type: toolauth.SchemeTypeOAuth2}} {
		got, err := credentialFromString("tok", scheme)
		if err != nil {
			t.Fatalf("credentialFromString() error = %v", err)
		}
		want := &toolauth.Credential{AuthType: toolauth.CredentialTypeOAuth2, OAuth2: &toolauth.OAuth2Auth{AccessToken: "tok"}}
		if diff := cmp.Diff(want, got); diff != "" {
			t.Errorf("credentialFromString() for scheme type %q mismatch (-want +got):\n%s", scheme.Type, diff)
		}
	}
	got, err := credentialFromString("tok", toolauth.Scheme{Type: toolauth.SchemeTypeHTTP})
	if err != nil || got.HTTP == nil || got.HTTP.Scheme != "bearer" {
		t.Errorf("credentialFromString() for an http scheme without a scheme name did not default to bearer")
	}
}
