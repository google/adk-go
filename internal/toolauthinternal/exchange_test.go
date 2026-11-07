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
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"

	"google.golang.org/adk/v2/tool/toolauth"
)

const (
	authorizeURL = "https://example.com/oauth2/authorize"
	redirectURL  = "https://example.com/callback"
)

// tokenRequest is what a fake token endpoint received.
type tokenRequest struct {
	form   url.Values
	header http.Header
}

// tokenServer is a fake OAuth2 token endpoint that records its requests.
type tokenServer struct {
	*httptest.Server
	mu       sync.Mutex
	requests []tokenRequest
}

func (s *tokenServer) recorded() []tokenRequest {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]tokenRequest(nil), s.requests...)
}

// newTokenServer starts a token endpoint that records each request and then
// calls handler, or answers with a fixed token when handler is nil.
func newTokenServer(t *testing.T, handler http.HandlerFunc) *tokenServer {
	t.Helper()
	s := &tokenServer{}
	s.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := r.ParseForm(); err != nil {
			http.Error(w, "bad form", http.StatusBadRequest)
			return
		}
		s.mu.Lock()
		s.requests = append(s.requests, tokenRequest{form: r.PostForm, header: r.Header.Clone()})
		s.mu.Unlock()
		if handler != nil {
			handler(w, r)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"access_token":"mock_access_token","token_type":"bearer","expires_in":3600,` +
			`"refresh_token":"mock_refresh_token","id_token":"mock_id_token"}`))
	}))
	t.Cleanup(s.Close)
	return s
}

func oauth2Scheme(tokenURL string) toolauth.Scheme {
	return toolauth.Scheme{
		Type: toolauth.SchemeTypeOAuth2,
		Flows: &toolauth.OAuthFlows{AuthorizationCode: &toolauth.OAuthFlow{
			AuthorizationURL: authorizeURL,
			TokenURL:         tokenURL,
			Scopes:           map[string]string{"write": "Write access", "read": "Read access"},
		}},
	}
}

func openIDScheme(tokenURL string) toolauth.Scheme {
	return toolauth.Scheme{
		Type:                  toolauth.SchemeTypeOpenIDConnect,
		OpenIDConnectURL:      "https://example.com/.well-known/openid-configuration",
		AuthorizationEndpoint: authorizeURL,
		TokenEndpoint:         tokenURL,
		Scopes:                []string{"openid", "profile", "email"},
	}
}

func oauth2Credential() *toolauth.Credential {
	return &toolauth.Credential{
		AuthType: toolauth.CredentialTypeOAuth2,
		OAuth2: &toolauth.OAuth2Auth{
			ClientID:     "mock_client_id",
			ClientSecret: "mock_client_secret",
			RedirectURI:  redirectURL,
		},
	}
}

func oauth2CredentialWithAuthCode() *toolauth.Credential {
	c := oauth2Credential()
	c.OAuth2.AuthURI = authorizeURL + "?client_id=mock_client_id&scope=read,write"
	c.OAuth2.State = "mock_state"
	c.OAuth2.AuthCode = "mock_auth_code"
	c.OAuth2.AuthResponseURI = redirectURL + "?code=mock_auth_code&state=mock_state"
	return c
}

func TestExchangeToken(t *testing.T) {
	// Each case returns the credential it was given, without a request.
	unchanged := []struct {
		name   string
		cred   func() *toolauth.Credential
		scheme func(tokenURL string) toolauth.Scheme
	}{
		// reference: test_openid_missing_token_endpoint
		{
			name:   "OpenID Connect scheme without a token endpoint",
			cred:   oauth2CredentialWithAuthCode,
			scheme: func(string) toolauth.Scheme { return openIDScheme("") },
		},
		// reference: test_oauth2_missing_token_url
		{
			name:   "OAuth2 scheme without a token URL",
			cred:   oauth2CredentialWithAuthCode,
			scheme: func(string) toolauth.Scheme { return oauth2Scheme("") },
		},
		// reference: test_non_oauth_scheme
		{
			name: "API key scheme",
			cred: oauth2CredentialWithAuthCode,
			scheme: func(string) toolauth.Scheme {
				return toolauth.Scheme{Type: toolauth.SchemeTypeAPIKey, In: "header", Name: "test_api_key"}
			},
		},
		// reference: test_missing_credentials
		{
			name:   "credential without an oauth2 section",
			cred:   func() *toolauth.Credential { return &toolauth.Credential{AuthType: toolauth.CredentialTypeOAuth2} },
			scheme: oauth2Scheme,
		},
		// reference: test_credentials_with_token
		{
			name: "credential that already has an access token",
			cred: func() *toolauth.Credential {
				c := oauth2CredentialWithAuthCode()
				c.OAuth2.AccessToken = "mock_access_token"
				return c
			},
			scheme: oauth2Scheme,
		},
		{
			name: "credential without a client ID",
			cred: func() *toolauth.Credential {
				c := oauth2CredentialWithAuthCode()
				c.OAuth2.ClientID = ""
				return c
			},
			scheme: oauth2Scheme,
		},
		{
			name: "credential without an authorization code",
			cred: func() *toolauth.Credential {
				c := oauth2CredentialWithAuthCode()
				c.OAuth2.AuthCode = ""
				c.OAuth2.AuthResponseURI = ""
				return c
			},
			scheme: oauth2Scheme,
		},
		{
			name: "client_secret_jwt client",
			cred: func() *toolauth.Credential {
				c := oauth2CredentialWithAuthCode()
				c.OAuth2.TokenEndpointAuthMethod = "client_secret_jwt"
				return c
			},
			scheme: oauth2Scheme,
		},
		{
			name: "private_key_jwt public client",
			cred: func() *toolauth.Credential {
				c := oauth2CredentialWithAuthCode()
				c.OAuth2.ClientSecret = ""
				c.OAuth2.TokenEndpointAuthMethod = "private_key_jwt"
				return c
			},
			scheme: oauth2Scheme,
		},
		{
			name: "implicit flow",
			cred: oauth2CredentialWithAuthCode,
			scheme: func(tokenURL string) toolauth.Scheme {
				s := oauth2Scheme(tokenURL)
				s.Flows.Implicit = &toolauth.OAuthFlow{AuthorizationURL: authorizeURL}
				s.Flows.AuthorizationCode = nil
				s.Flows.ClientCredentials = nil
				return s
			},
		},
	}
	for _, tc := range unchanged {
		t.Run(tc.name, func(t *testing.T) {
			srv := newTokenServer(t, nil)
			cred := tc.cred()
			got, err := ExchangeToken(t.Context(), cred, tc.scheme(srv.URL))
			if err != nil {
				t.Fatalf("ExchangeToken() error = %v", err)
			}
			if got != cred {
				t.Errorf("ExchangeToken() returned a different credential, want the one it was given")
			}
			if n := len(srv.recorded()); n != 0 {
				t.Errorf("token endpoint got %d requests, want 0", n)
			}
		})
	}

	// reference: test_successful_token_exchange
	t.Run("exchanges an authorization code", func(t *testing.T) {
		srv := newTokenServer(t, nil)
		cred := oauth2CredentialWithAuthCode()
		before := time.Now()
		got, err := ExchangeToken(t.Context(), cred, oauth2Scheme(srv.URL))
		if err != nil {
			t.Fatalf("ExchangeToken() error = %v", err)
		}
		reqs := srv.recorded()
		if len(reqs) != 1 {
			t.Fatalf("token endpoint got %d requests, want 1", len(reqs))
		}
		form := reqs[0].form
		for k, want := range map[string]string{
			"grant_type":   "authorization_code",
			"code":         "mock_auth_code",
			"redirect_uri": redirectURL,
		} {
			if got := form.Get(k); got != want {
				t.Errorf("form %s = %q, want %q", k, got, want)
			}
		}
		for _, k := range []string{"scope", "client_secret", "code_verifier"} {
			if form.Has(k) {
				t.Errorf("form has %s, want none", k)
			}
		}
		req := &http.Request{Header: reqs[0].header}
		user, pass, ok := req.BasicAuth()
		if !ok || user != "mock_client_id" || pass != "mock_client_secret" {
			t.Errorf("basic auth = %q, %t; want the client ID and secret", user, ok)
		}

		o := got.OAuth2
		if o.AccessToken != "mock_access_token" || o.RefreshToken != "mock_refresh_token" || o.IDToken != "mock_id_token" {
			t.Errorf("access, refresh or ID token is not the one the endpoint returned")
		}
		if o.ExpiresIn != 3600 {
			t.Errorf("ExpiresIn = %d, want 3600", o.ExpiresIn)
		}
		if earliest := before.Add(3600 * time.Second).Unix(); o.ExpiresAt < earliest || o.ExpiresAt > earliest+60 {
			t.Errorf("ExpiresAt = %d, want about %d", o.ExpiresAt, earliest)
		}
		if got.AuthType != toolauth.CredentialTypeOAuth2 {
			t.Errorf("AuthType = %q, want oauth2", got.AuthType)
		}
		if cred.OAuth2.AccessToken != "" {
			t.Errorf("ExchangeToken() changed the credential it was given")
		}
	})

	t.Run("sends the PKCE code verifier", func(t *testing.T) {
		srv := newTokenServer(t, nil)
		cred := oauth2CredentialWithAuthCode()
		cred.OAuth2.CodeVerifier = "mock_verifier"
		if _, err := ExchangeToken(t.Context(), cred, oauth2Scheme(srv.URL)); err != nil {
			t.Fatalf("ExchangeToken() error = %v", err)
		}
		if got := srv.recorded()[0].form.Get("code_verifier"); got != "mock_verifier" {
			t.Errorf("form code_verifier = %q, want mock_verifier", got)
		}
	})

	t.Run("public client sends its ID in the form and no secret", func(t *testing.T) {
		srv := newTokenServer(t, nil)
		cred := oauth2CredentialWithAuthCode()
		cred.OAuth2.ClientSecret = ""
		if _, err := ExchangeToken(t.Context(), cred, oauth2Scheme(srv.URL)); err != nil {
			t.Fatalf("ExchangeToken() error = %v", err)
		}
		req := srv.recorded()[0]
		if got := req.form.Get("client_id"); got != "mock_client_id" {
			t.Errorf("form client_id = %q, want mock_client_id", got)
		}
		if req.form.Has("client_secret") {
			t.Errorf("form has client_secret, want none")
		}
		if req.header.Get("Authorization") != "" {
			t.Errorf("request has an Authorization header, want none")
		}
	})

	t.Run("client_secret_post sends the secret in the form", func(t *testing.T) {
		srv := newTokenServer(t, nil)
		cred := oauth2CredentialWithAuthCode()
		cred.OAuth2.TokenEndpointAuthMethod = "client_secret_post"
		if _, err := ExchangeToken(t.Context(), cred, oauth2Scheme(srv.URL)); err != nil {
			t.Fatalf("ExchangeToken() error = %v", err)
		}
		req := srv.recorded()[0]
		if req.form.Get("client_id") != "mock_client_id" || req.form.Get("client_secret") != "mock_client_secret" {
			t.Errorf("form does not carry the client ID and secret")
		}
		if req.header.Get("Authorization") != "" {
			t.Errorf("request has an Authorization header, want none")
		}
	})

	t.Run("client credentials grant", func(t *testing.T) {
		srv := newTokenServer(t, nil)
		scheme := toolauth.Scheme{
			Type: toolauth.SchemeTypeOAuth2,
			Flows: &toolauth.OAuthFlows{ClientCredentials: &toolauth.OAuthFlow{
				TokenURL: srv.URL,
				Scopes:   map[string]string{"read": "Read access"},
			}},
		}
		got, err := ExchangeToken(t.Context(), oauth2Credential(), scheme)
		if err != nil {
			t.Fatalf("ExchangeToken() error = %v", err)
		}
		form := srv.recorded()[0].form
		if form.Get("grant_type") != "client_credentials" {
			t.Errorf("form grant_type = %q, want client_credentials", form.Get("grant_type"))
		}
		if form.Has("scope") {
			t.Errorf("form has scope, want none")
		}
		if got.OAuth2.AccessToken != "mock_access_token" {
			t.Errorf("AccessToken is not the mock token")
		}
	})

	t.Run("OpenID Connect scheme uses client credentials when supported", func(t *testing.T) {
		srv := newTokenServer(t, nil)
		scheme := openIDScheme(srv.URL)
		scheme.GrantTypesSupported = []string{"authorization_code", "client_credentials"}
		if _, err := ExchangeToken(t.Context(), oauth2CredentialWithAuthCode(), scheme); err != nil {
			t.Fatalf("ExchangeToken() error = %v", err)
		}
		if got := srv.recorded()[0].form.Get("grant_type"); got != "client_credentials" {
			t.Errorf("form grant_type = %q, want client_credentials", got)
		}
	})

	t.Run("OpenID Connect scheme uses the authorization code by default", func(t *testing.T) {
		srv := newTokenServer(t, nil)
		if _, err := ExchangeToken(t.Context(), oauth2CredentialWithAuthCode(), openIDScheme(srv.URL)); err != nil {
			t.Fatalf("ExchangeToken() error = %v", err)
		}
		if got := srv.recorded()[0].form.Get("grant_type"); got != "authorization_code" {
			t.Errorf("form grant_type = %q, want authorization_code", got)
		}
	})

	t.Run("takes the code from the auth response URI", func(t *testing.T) {
		srv := newTokenServer(t, nil)
		cred := oauth2CredentialWithAuthCode()
		cred.OAuth2.AuthCode = ""
		cred.OAuth2.AuthResponseURI = redirectURL + "?code=uri_code&state=mock_state#"
		if _, err := ExchangeToken(t.Context(), cred, oauth2Scheme(srv.URL)); err != nil {
			t.Fatalf("ExchangeToken() error = %v", err)
		}
		if got := srv.recorded()[0].form.Get("code"); got != "uri_code" {
			t.Errorf("form code = %q, want uri_code", got)
		}
	})

	for _, tc := range []struct{ name, uri string }{
		{"refuses an auth response URI with another state", redirectURL + "?code=uri_code&state=forged"},
		{"refuses an auth response URI without a state", redirectURL + "?code=uri_code"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			srv := newTokenServer(t, nil)
			cred := oauth2CredentialWithAuthCode()
			cred.OAuth2.AuthResponseURI = tc.uri
			if _, err := ExchangeToken(t.Context(), cred, oauth2Scheme(srv.URL)); err == nil {
				t.Errorf("ExchangeToken() error = nil, want a state mismatch")
			}
			if n := len(srv.recorded()); n != 0 {
				t.Errorf("token endpoint got %d requests, want 0", n)
			}
		})
	}

	t.Run("error does not quote the token response", func(t *testing.T) {
		const marker = "secret-marker-in-body"
		srv := newTokenServer(t, func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusBadRequest)
			_, _ = w.Write([]byte(`{"error":"invalid_grant","error_description":"` + marker + `"}`))
		})
		_, err := ExchangeToken(t.Context(), oauth2CredentialWithAuthCode(), oauth2Scheme(srv.URL))
		if err == nil {
			t.Fatal("ExchangeToken() error = nil, want one")
		}
		if strings.Contains(err.Error(), marker) || strings.Contains(err.Error(), "mock_auth_code") {
			t.Errorf("error quotes the token response or the code")
		}
		if !strings.Contains(err.Error(), "400") {
			t.Errorf("error = %q, want the HTTP status", err)
		}
	})

	t.Run("cancelled while the request is in flight", func(t *testing.T) {
		started := make(chan struct{})
		srv := newTokenServer(t, func(w http.ResponseWriter, r *http.Request) {
			close(started)
			<-r.Context().Done()
		})
		ctx, cancel := context.WithCancel(t.Context())
		go func() {
			<-started
			cancel()
		}()
		_, err := ExchangeToken(ctx, oauth2CredentialWithAuthCode(), oauth2Scheme(srv.URL))
		if !errors.Is(err, context.Canceled) {
			t.Errorf("ExchangeToken() error = %v, want context.Canceled", err)
		}
	})

	t.Run("timed out while the request is in flight", func(t *testing.T) {
		tokenRequestTimeout = 10 * time.Millisecond
		t.Cleanup(func() { tokenRequestTimeout = defaultTokenRequestTimeout })
		srv := newTokenServer(t, func(w http.ResponseWriter, r *http.Request) {
			<-r.Context().Done()
		})
		_, err := ExchangeToken(t.Context(), oauth2CredentialWithAuthCode(), oauth2Scheme(srv.URL))
		if !errors.Is(err, context.DeadlineExceeded) {
			t.Errorf("ExchangeToken() error = %v, want context.DeadlineExceeded", err)
		}
	})

	t.Run("transport error does not quote the URL or credentials", func(t *testing.T) {
		const secretHost = "secret-token-host.invalid"
		_, err := ExchangeToken(t.Context(), oauth2CredentialWithAuthCode(), oauth2Scheme("http://user:secret-pass@"+secretHost+":1/token"))
		if err == nil {
			t.Fatal("ExchangeToken() error = nil, want one")
		}
		for _, secret := range []string{secretHost, "secret-pass", "mock_auth_code", "mock_client_secret"} {
			if strings.Contains(err.Error(), secret) {
				t.Errorf("error quotes a secret or URL component")
			}
		}
	})

	t.Run("malformed auth response URI error does not quote the URI", func(t *testing.T) {
		srv := newTokenServer(t, nil)
		cred := oauth2CredentialWithAuthCode()
		cred.OAuth2.AuthResponseURI = "http://[::1/?code=secret-uri-code&state=mock_state"
		_, err := ExchangeToken(t.Context(), cred, oauth2Scheme(srv.URL))
		if err == nil {
			t.Fatal("ExchangeToken() error = nil, want one")
		}
		if strings.Contains(err.Error(), "secret-uri-code") || strings.Contains(err.Error(), "[::1") {
			t.Errorf("error quotes the auth response URI")
		}
		if n := len(srv.recorded()); n != 0 {
			t.Errorf("token endpoint got %d requests, want 0", n)
		}
	})
}

func TestAuthorizationCode(t *testing.T) {
	o := &toolauth.OAuth2Auth{AuthCode: "plain", AuthResponseURI: "https://example.com/cb?state=s", State: "s"}
	got, err := authorizationCode(o)
	if err != nil {
		t.Fatalf("authorizationCode() error = %v", err)
	}
	if diff := cmp.Diff("plain", got); diff != "" {
		t.Errorf("authorizationCode() mismatch (-want +got):\n%s", diff)
	}
}
