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

package authtool_test

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"net/url"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/exp/authtool"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool/toolauth"
)

// countingExchanger answers every exchange with a copy of result, or of the
// credential it got when result is nil, and records what it was given.
type countingExchanger struct {
	result      *toolauth.Credential
	didExchange bool

	mu    sync.Mutex
	calls int
	cfgs  []*toolauth.Config
	creds []*toolauth.Credential
}

func (e *countingExchanger) Exchange(_ agent.Context, cfg *toolauth.Config, cred *toolauth.Credential) (*toolauth.Credential, bool, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.calls++
	e.cfgs = append(e.cfgs, cfg)
	e.creds = append(e.creds, cred.Clone())
	if e.result == nil {
		return cred, e.didExchange, nil
	}
	return e.result.Clone(), e.didExchange, nil
}

func (e *countingExchanger) count() int {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.calls
}

func newManager(t *testing.T, cfg *toolauth.Config) *authtool.CredentialManager {
	t.Helper()
	m, err := authtool.NewCredentialManager(cfg, nil)
	if err != nil {
		t.Fatalf("NewCredentialManager() error = %v", err)
	}
	return m
}

// putAuthResponse stores cred where ADK keeps the user's answer for cfg.
func putAuthResponse(t *testing.T, ctx *fakeToolContext, cred toolauth.Credential) {
	t.Helper()
	if err := ctx.state.Set(session.KeyPrefixTemp+"key-1", cred); err != nil {
		t.Fatalf("State.Set() error = %v", err)
	}
}

func oauthCred(o toolauth.OAuth2Auth) *toolauth.Credential {
	return &toolauth.Credential{AuthType: toolauth.CredentialTypeOAuth2, OAuth2: &o}
}

// accessToken returns cred's access token, or a marker when there is none.
func accessToken(cred *toolauth.Credential) string {
	if cred == nil {
		return "<nil credential>"
	}
	if cred.OAuth2 == nil {
		return "<no oauth2>"
	}
	return cred.OAuth2.AccessToken
}

// Test 15. Fails if the nil or zero-scheme guard moves, or a valid config is
// refused.
func TestNewCredentialManager_ZeroConfig(t *testing.T) {
	tests := []struct {
		name    string
		cfg     *toolauth.Config
		wantErr bool
	}{
		{name: "nil config", cfg: nil, wantErr: true},
		{name: "zero scheme", cfg: &toolauth.Config{}, wantErr: true},
		{name: "api key scheme", cfg: &toolauth.Config{AuthScheme: toolauth.Scheme{Type: toolauth.SchemeTypeAPIKey}}, wantErr: false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m, err := authtool.NewCredentialManager(tt.cfg, nil)
			if gotErr := err != nil; gotErr != tt.wantErr {
				t.Fatalf("NewCredentialManager() error = %v, want error %v", err, tt.wantErr)
			}
			if gotNil := m == nil; gotNil != tt.wantErr {
				t.Errorf("NewCredentialManager() manager nil = %v, want %v", gotNil, tt.wantErr)
			}
		})
	}
}

// Test 16. adk-python's texts (auth/credential_manager.py:407-438). Fails if
// a check is dropped or reordered: a missing raw credential wins over a
// missing URL.
func TestGetAuthCredential_Validation(t *testing.T) {
	noTokenURL := func(raw *toolauth.Credential) *toolauth.Config {
		cfg := codeConfig("")
		cfg.RawAuthCredential = raw
		return cfg
	}
	oidc := toolauth.Scheme{Type: toolauth.SchemeTypeOpenIDConnect, TokenEndpoint: "https://idp.example/token"}
	tests := []struct {
		name string
		cfg  *toolauth.Config
		want string
	}{
		{
			name: "oauth2 without raw credential",
			cfg:  &toolauth.Config{AuthScheme: codeConfig("https://idp.example/token").AuthScheme},
			want: "raw_auth_credential is required for auth_scheme type oauth2",
		},
		{
			name: "openIdConnect without raw credential",
			cfg:  &toolauth.Config{AuthScheme: oidc},
			want: "raw_auth_credential is required for auth_scheme type openIdConnect",
		},
		{
			name: "oauth2 raw credential without oauth2 section",
			cfg: &toolauth.Config{
				AuthScheme:        codeConfig("https://idp.example/token").AuthScheme,
				RawAuthCredential: &toolauth.Credential{AuthType: toolauth.CredentialTypeOAuth2},
			},
			want: "auth_config.raw_credential.oauth2 required for credential type oauth2",
		},
		{
			name: "openIdConnect raw credential without oauth2 section",
			cfg: &toolauth.Config{
				AuthScheme:        oidc,
				RawAuthCredential: &toolauth.Credential{AuthType: toolauth.CredentialTypeOpenIDConnect},
			},
			want: "auth_config.raw_credential.oauth2 required for credential type openIdConnect",
		},
		{
			name: "missing token URL",
			cfg:  noTokenURL(codeConfig("").RawAuthCredential),
			want: "OAuth scheme info is missing, and auto-discovery has failed to fill them in.",
		},
		{
			name: "missing raw credential wins over missing token URL",
			cfg:  noTokenURL(nil),
			want: "raw_auth_credential is required for auth_scheme type oauth2",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			svc := newFakeService()
			cred, err := newManager(t, tt.cfg).GetAuthCredential(newToolContext(t, svc))
			if err == nil || err.Error() != tt.want {
				t.Fatalf("GetAuthCredential() error = %v, want %q", err, tt.want)
			}
			if cred != nil {
				t.Errorf("GetAuthCredential() credential = %+v, want nil", cred)
			}
			if loads, saves := svc.counts(); loads != 0 || saves != 0 {
				t.Errorf("service calls = %d loads, %d saves, want none", loads, saves)
			}
		})
	}
}

// Test 17. Ruling R2: inputs this package does not support fail with an error
// that names them (texts from the work split, section 4 point 5). Fails if a
// custom scheme falls into another path, discovery is reported as missing
// URLs, or a service account is treated as "no credential".
func TestGetAuthCredential_UnsupportedInputs(t *testing.T) {
	withIssuer := codeConfig("")
	withIssuer.AuthScheme.IssuerURL = "https://idp.example"
	serviceAccount := ccConfig("https://idp.example/token")
	serviceAccount.RawAuthCredential = &toolauth.Credential{
		AuthType:       toolauth.CredentialTypeServiceAccount,
		ServiceAccount: &toolauth.ServiceAccount{UseDefaultCredential: true, Scopes: []string{"read"}},
	}
	tests := []struct {
		name string
		cfg  *toolauth.Config
		want string
	}{
		{
			name: "custom scheme",
			cfg:  &toolauth.Config{AuthScheme: toolauth.Scheme{Type: "custom"}, CredentialKey: "key-1"},
			want: "custom auth scheme is not supported",
		},
		{name: "discovery from an issuer URL", cfg: withIssuer, want: "OAuth2 auto-discovery is not supported"},
		{name: "service account", cfg: serviceAccount, want: "service account is not supported"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			svc := newFakeService()
			cred, err := newManager(t, tt.cfg).GetAuthCredential(newToolContext(t, svc))
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("GetAuthCredential() error = %v, want it to contain %q", err, tt.want)
			}
			if cred != nil {
				t.Errorf("GetAuthCredential() credential = %+v, want nil", cred)
			}
			if loads, saves := svc.counts(); loads != 0 || saves != 0 {
				t.Errorf("service calls = %d loads, %d saves, want none", loads, saves)
			}
		})
	}
}

// Test 18. Fails if the raw pointer is returned, a map is shared, or the ready
// path goes on to load or exchange.
func TestGetAuthCredential_ReadyCredentialIsCopy(t *testing.T) {
	bearer := &toolauth.Config{
		AuthScheme: toolauth.Scheme{Type: toolauth.SchemeTypeHTTP, Scheme: "bearer"},
		RawAuthCredential: &toolauth.Credential{AuthType: toolauth.CredentialTypeHTTP, HTTP: &toolauth.HTTPAuth{
			Scheme:            "bearer",
			Credentials:       toolauth.HTTPCredentials{Token: "test-bearer-1"},
			AdditionalHeaders: map[string]string{"X-A": "1"},
		}},
		CredentialKey: "key-1",
	}
	tests := []struct {
		name   string
		cfg    *toolauth.Config
		mutate func(*toolauth.Credential)
	}{
		{name: "api key", cfg: apiKeyConfig(), mutate: func(c *toolauth.Credential) { c.APIKey = "changed" }},
		{name: "http bearer", cfg: bearer, mutate: func(c *toolauth.Credential) { c.HTTP.AdditionalHeaders["X-A"] = "2" }},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			want := tt.cfg.RawAuthCredential.Clone()
			svc := newFakeService()
			ex := &countingExchanger{}
			m := newManager(t, tt.cfg)
			m.RegisterCredentialExchanger(want.AuthType, ex)

			cred, err := m.GetAuthCredential(newToolContext(t, svc))
			if err != nil {
				t.Fatalf("GetAuthCredential() error = %v", err)
			}
			if diff := cmp.Diff(want, cred); diff != "" {
				t.Fatalf("GetAuthCredential() mismatch (-want +got):\n%s", diff)
			}
			tt.mutate(cred)
			if diff := cmp.Diff(want, tt.cfg.RawAuthCredential); diff != "" {
				t.Errorf("raw credential changed through the result (-want +got):\n%s", diff)
			}
			if loads, saves := svc.counts(); loads != 0 || saves != 0 {
				t.Errorf("service calls = %d loads, %d saves, want none", loads, saves)
			}
			if n := ex.count(); n != 0 {
				t.Errorf("exchanger calls = %d, want 0", n)
			}
		})
	}
}

// Test 19. Fails if the load order changes or the config's own
// ExchangedAuthCredential is read as a cache.
func TestGetAuthCredential_LoadOrder(t *testing.T) {
	ts := newTokenServer(t)
	t.Run("service before auth response", func(t *testing.T) {
		cfg := codeConfig(ts.URL)
		svc := newFakeService()
		svc.put(t, cfg, oauthCred(toolauth.OAuth2Auth{AccessToken: "svc-access"}))
		ctx := newToolContext(t, svc)
		putAuthResponse(t, ctx, *oauthCred(toolauth.OAuth2Auth{AccessToken: "state-access"}))

		cred, err := newManager(t, cfg).GetAuthCredential(ctx)
		if err != nil || accessToken(cred) != "svc-access" {
			t.Fatalf("GetAuthCredential() = %s, %v, want svc-access, nil", accessToken(cred), err)
		}
	})
	t.Run("auth response with the configured client", func(t *testing.T) {
		cfg := codeConfig(ts.URL)
		ctx := newToolContext(t, newFakeService())
		putAuthResponse(t, ctx, *oauthCred(toolauth.OAuth2Auth{AccessToken: "state-access"}))

		cred, err := newManager(t, cfg).GetAuthCredential(ctx)
		if err != nil || accessToken(cred) != "state-access" {
			t.Fatalf("GetAuthCredential() = %s, %v, want state-access, nil", accessToken(cred), err)
		}
		if cred.OAuth2.ClientSecret != "secret-1" {
			t.Errorf("GetAuthCredential() client secret re-attached = %v, want true", cred.OAuth2.ClientSecret == "secret-1")
		}
	})
	t.Run("client credentials from the raw credential", func(t *testing.T) {
		cred, err := newManager(t, ccConfig(ts.URL)).GetAuthCredential(newToolContext(t, newFakeService()))
		if err != nil || accessToken(cred) != "cc-access" {
			t.Fatalf("GetAuthCredential() = %s, %v, want cc-access, nil", accessToken(cred), err)
		}
		reqs := ts.seen()
		if len(reqs) != 1 || reqs[0].form.Get("grant_type") != "client_credentials" {
			t.Errorf("token requests = %d (first grant_type %q), want 1 client_credentials", len(reqs), firstGrant(reqs))
		}
	})
	t.Run("authorization code needs the user", func(t *testing.T) {
		cred, err := newManager(t, codeConfig(ts.URL)).GetAuthCredential(newToolContext(t, newFakeService()))
		if cred != nil || err != nil {
			t.Fatalf("GetAuthCredential() = %s, %v, want nil, nil", accessToken(cred), err)
		}
	})
	t.Run("config cache is not read", func(t *testing.T) {
		cfg := codeConfig(ts.URL)
		cfg.ExchangedAuthCredential = oauthCred(toolauth.OAuth2Auth{AccessToken: "cfg-access"})
		cred, err := newManager(t, cfg).GetAuthCredential(newToolContext(t, newFakeService()))
		if cred != nil || err != nil {
			t.Fatalf("GetAuthCredential() = %s, %v, want nil, nil", accessToken(cred), err)
		}
	})
}

func firstGrant(reqs []tokenRequest) string {
	if len(reqs) == 0 {
		return ""
	}
	return reqs[0].form.Get("grant_type")
}

// Test 20. Fails if a check dereferences a nil raw credential.
func TestGetAuthCredential_NilRawCredential(t *testing.T) {
	cfg := apiKeyConfig()
	cfg.RawAuthCredential = nil
	cred, err := newManager(t, cfg).GetAuthCredential(newToolContext(t, newFakeService()))
	if cred != nil || err != nil {
		t.Fatalf("GetAuthCredential() = %+v, %v, want nil, nil", cred, err)
	}
}

// Test 21. Ruling R4, amendment A4: a refused authorization code fails the
// call. Fails if the manager keeps the code-only credential or saves it.
func TestGetAuthCredential_AuthResponseExchangeFailure(t *testing.T) {
	ts := newTokenServer(t)
	ts.set(func(s *tokenServer) { s.status, s.body = 400, invalidGrant })
	cfg := codeConfig(ts.URL)
	svc := newFakeService()
	ctx := newToolContext(t, svc)
	putAuthResponse(t, ctx, *oauthCred(toolauth.OAuth2Auth{AuthCode: "code-1"}))

	cred, err := newManager(t, cfg).GetAuthCredential(ctx)
	if cred != nil || err == nil || !strings.Contains(err.Error(), "invalid_grant") {
		t.Fatalf("GetAuthCredential() = %s, %v, want nil and an error naming invalid_grant", accessToken(cred), err)
	}
	if n := len(ts.seen()); n != 1 {
		t.Errorf("token requests = %d, want 1", n)
	}
	if got := svc.stored(t, cfg); got != nil {
		t.Errorf("service holds %s, want nothing", accessToken(got))
	}
}

// Test 22. Fails if scopes are sent (adk-python omits them,
// auth/oauth2_credential_util.py:177-178), the grant type is wrong, the
// secret goes in the body, or the expiry is not taken from expires_in.
func TestGetAuthCredential_ClientCredentialsExchange(t *testing.T) {
	ts := newTokenServer(t)
	before := time.Now().Unix()
	cred, err := newManager(t, ccConfig(ts.URL)).GetAuthCredential(newToolContext(t, nil))
	if err != nil || accessToken(cred) != "cc-access" {
		t.Fatalf("GetAuthCredential() = %s, %v, want cc-access, nil", accessToken(cred), err)
	}
	if cred.OAuth2.ExpiresIn != 3600 {
		t.Errorf("ExpiresIn = %d, want 3600", cred.OAuth2.ExpiresIn)
	}
	if at := cred.OAuth2.ExpiresAt; at < before+3590 || at > before+3610 {
		t.Errorf("ExpiresAt = now%+d, want now+3600", at-before)
	}
	reqs := ts.seen()
	if len(reqs) != 1 {
		t.Fatalf("token requests = %d, want 1", len(reqs))
	}
	want := url.Values{"grant_type": {"client_credentials"}}
	if diff := cmp.Diff(want, reqs[0].form); diff != "" {
		t.Errorf("token request form mismatch (-want +got):\n%s", diff)
	}
	wantAuth := "Basic " + base64.StdEncoding.EncodeToString([]byte("client-1:secret-1"))
	if reqs[0].authorization != wantAuth {
		t.Errorf("token request sent the client in the Authorization header = %v, want true", reqs[0].authorization == wantAuth)
	}
}

// Test 23. Fails if the client authentication method is ignored, or a request
// that cannot be built fails the call instead of returning the raw
// credential (amendment A3).
func TestGetAuthCredential_ClientAuthMethod(t *testing.T) {
	tests := []struct {
		name        string
		client      toolauth.OAuth2Auth
		wantForm    url.Values
		wantRequest bool
	}{
		{
			name:        "client_secret_post",
			client:      toolauth.OAuth2Auth{ClientID: "client-1", ClientSecret: "secret-1", TokenEndpointAuthMethod: "client_secret_post"},
			wantForm:    url.Values{"grant_type": {"client_credentials"}, "client_id": {"client-1"}, "client_secret": {"secret-1"}},
			wantRequest: true,
		},
		{
			name:        "public client",
			client:      toolauth.OAuth2Auth{ClientID: "client-1"},
			wantForm:    url.Values{"grant_type": {"client_credentials"}, "client_id": {"client-1"}},
			wantRequest: true,
		},
		{name: "private_key_jwt", client: toolauth.OAuth2Auth{ClientID: "client-1", TokenEndpointAuthMethod: "private_key_jwt"}},
		{name: "no client id", client: toolauth.OAuth2Auth{ClientSecret: "secret-1"}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ts := newTokenServer(t)
			cfg := ccConfig(ts.URL)
			cfg.RawAuthCredential = oauthCred(tt.client)

			cred, err := newManager(t, cfg).GetAuthCredential(newToolContext(t, nil))
			reqs := ts.seen()
			if !tt.wantRequest {
				if err == nil || len(reqs) != 0 {
					t.Fatalf("GetAuthCredential() error = %v with %d token requests, want an error and none", err, len(reqs))
				}
				if diff := cmp.Diff(oauthCred(tt.client), cred); diff != "" {
					t.Errorf("GetAuthCredential() credential mismatch (-want +got):\n%s", diff)
				}
				return
			}
			if err != nil || accessToken(cred) != "cc-access" || len(reqs) != 1 {
				t.Fatalf("GetAuthCredential() = %s, %v with %d token requests, want cc-access, nil and 1", accessToken(cred), err, len(reqs))
			}
			if diff := cmp.Diff(tt.wantForm, reqs[0].form); diff != "" {
				t.Errorf("token request form mismatch (-want +got):\n%s", diff)
			}
			if reqs[0].authorization != "" {
				t.Errorf("token request has an Authorization header, want none")
			}
		})
	}
}

// Test 24. Fails if an OpenID Connect scheme is not dispatched on
// grant_types_supported.
func TestGetAuthCredential_OIDCGrantSelection(t *testing.T) {
	tests := []struct {
		name      string
		grants    []string
		wantToken string
	}{
		{name: "client credentials supported", grants: []string{"client_credentials"}, wantToken: "cc-access"},
		{name: "authorization code only", grants: []string{"authorization_code"}},
		{name: "no grant types", grants: nil},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ts := newTokenServer(t)
			cfg := &toolauth.Config{
				AuthScheme: toolauth.Scheme{
					Type:                  toolauth.SchemeTypeOpenIDConnect,
					AuthorizationEndpoint: "https://idp.example/auth",
					TokenEndpoint:         ts.URL,
					GrantTypesSupported:   tt.grants,
				},
				RawAuthCredential: &toolauth.Credential{AuthType: toolauth.CredentialTypeOpenIDConnect, OAuth2: &toolauth.OAuth2Auth{
					ClientID: "client-1", ClientSecret: "secret-1",
				}},
				CredentialKey: "key-1",
			}
			cred, err := newManager(t, cfg).GetAuthCredential(newToolContext(t, nil))
			if err != nil {
				t.Fatalf("GetAuthCredential() error = %v", err)
			}
			wantRequests := 0
			if tt.wantToken != "" {
				wantRequests = 1
				if accessToken(cred) != tt.wantToken {
					t.Errorf("GetAuthCredential() token = %s, want %s", accessToken(cred), tt.wantToken)
				}
			} else if cred != nil {
				t.Errorf("GetAuthCredential() = %s, want nil", accessToken(cred))
			}
			if n := len(ts.seen()); n != wantRequests {
				t.Errorf("token requests = %d, want %d", n, wantRequests)
			}
		})
	}
}

// Test 25. Ported flaw (auth/exchanger/oauth2_credential_exchanger.py:236-238)
// and amendment A7, a Fixed bug: adk-python also saves the unexchanged raw
// copy (credential_manager.py:284-287). Fails if the credential is nil, the
// error is dropped, or a failed exchange is saved.
func TestGetAuthCredential_FailedExchangeKeepsRaw(t *testing.T) {
	ts := newTokenServer(t)
	ts.set(func(s *tokenServer) { s.status, s.body = 400, invalidClient })
	cfg := ccConfig(ts.URL)
	svc := newFakeService()

	cred, err := newManager(t, cfg).GetAuthCredential(newToolContext(t, svc))
	if err == nil || !strings.Contains(err.Error(), "invalid_client") {
		t.Fatalf("GetAuthCredential() error = %v, want one naming invalid_client", err)
	}
	if strings.Contains(err.Error(), "secret-1") {
		t.Errorf("GetAuthCredential() error quotes the client secret")
	}
	if diff := cmp.Diff(cfg.RawAuthCredential, cred); diff != "" {
		t.Errorf("GetAuthCredential() credential mismatch (-want +got):\n%s", diff)
	}
	if _, saves := svc.counts(); saves != 0 {
		t.Errorf("service saves = %d, want 0", saves)
	}
}

// Test 26. Fails if a missing credential service becomes an error.
func TestGetAuthCredential_ExchangeWithoutService(t *testing.T) {
	ts := newTokenServer(t)
	cred, err := newManager(t, ccConfig(ts.URL)).GetAuthCredential(newToolContext(t, nil))
	if err != nil || accessToken(cred) != "cc-access" {
		t.Fatalf("GetAuthCredential() = %s, %v, want cc-access, nil", accessToken(cred), err)
	}
}

// Test 27. Fails if a held token is exchanged again or saved on every call.
func TestGetAuthCredential_TokenHeldIsNotExchangedOrSaved(t *testing.T) {
	ts := newTokenServer(t)
	cfg := codeConfig(ts.URL)
	svc := newFakeService()
	svc.put(t, cfg, oauthCred(toolauth.OAuth2Auth{AccessToken: "svc-access", RefreshToken: "r-old"}))

	cred, err := newManager(t, cfg).GetAuthCredential(newToolContext(t, svc))
	if err != nil || accessToken(cred) != "svc-access" {
		t.Fatalf("GetAuthCredential() = %s, %v, want svc-access, nil", accessToken(cred), err)
	}
	if n := len(ts.seen()); n != 0 {
		t.Errorf("token requests = %d, want 0", n)
	}
	if _, saves := svc.counts(); saves != 0 {
		t.Errorf("service saves = %d, want 0", saves)
	}
}

// fixedNow is the clock of the refresh tests, on a whole second.
var fixedNow = time.Unix(1_800_000_000, 0)

// Test 28. authlib's OAuth2Token.is_expired with its 60-second leeway, where a
// missing expires_at is built from expires_in. Fails if the comparison is off
// by one, zero counts as expired, ExpiresIn is ignored, or the refresh token
// is lost.
func TestGetAuthCredential_RefreshNearExpiry(t *testing.T) {
	now := fixedNow.Unix()
	tests := []struct {
		name        string
		expiresAt   int64
		expiresIn   int64
		wantRefresh bool
	}{
		{name: "expires in 30 seconds", expiresAt: now + 30, wantRefresh: true},
		{name: "expired an hour ago", expiresAt: now - 3600, wantRefresh: true},
		{name: "expires in an hour", expiresAt: now + 3600},
		{name: "expires in 59 seconds", expiresAt: now + 59, wantRefresh: true},
		{name: "expires in 60 seconds", expiresAt: now + 60},
		{name: "no expiry"},
		{name: "expires_in 30 seconds", expiresIn: 30, wantRefresh: true},
		{name: "expires_in an hour", expiresIn: 3600},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ts := newTokenServer(t)
			ts.set(func(s *tokenServer) { s.body = refreshTokenBody })
			cfg := codeConfig(ts.URL)
			svc := newFakeService()
			svc.put(t, cfg, oauthCred(toolauth.OAuth2Auth{
				ClientID: "client-1", ClientSecret: "secret-1",
				AccessToken: "stale-access", RefreshToken: "r-old", ExpiresAt: tt.expiresAt, ExpiresIn: tt.expiresIn,
			}))
			m, err := authtool.NewCredentialManager(cfg, func() time.Time { return fixedNow })
			if err != nil {
				t.Fatalf("NewCredentialManager() error = %v", err)
			}

			cred, err := m.GetAuthCredential(newToolContext(t, svc))
			if err != nil {
				t.Fatalf("GetAuthCredential() error = %v", err)
			}
			reqs := ts.seen()
			if !tt.wantRefresh {
				if len(reqs) != 0 || accessToken(cred) != "stale-access" {
					t.Fatalf("GetAuthCredential() = %s with %d token requests, want stale-access and none", accessToken(cred), len(reqs))
				}
				return
			}
			if len(reqs) != 1 {
				t.Fatalf("token requests = %d, want 1", len(reqs))
			}
			wantForm := url.Values{"grant_type": {"refresh_token"}, "refresh_token": {"r-old"}}
			if diff := cmp.Diff(wantForm, reqs[0].form); diff != "" {
				t.Errorf("refresh request form mismatch (-want +got):\n%s", diff)
			}
			if accessToken(cred) != "cc-access" || cred.OAuth2.RefreshToken != "r-new" {
				t.Errorf("GetAuthCredential() = %s, refresh token %s, want cc-access, r-new", accessToken(cred), cred.OAuth2.RefreshToken)
			}
			if got := svc.stored(t, cfg); accessToken(got) != "cc-access" {
				t.Errorf("service holds %s, want cc-access", accessToken(got))
			}
		})
	}
	t.Run("response without a refresh token", func(t *testing.T) {
		ts := newTokenServer(t)
		cfg := codeConfig(ts.URL)
		svc := newFakeService()
		svc.put(t, cfg, oauthCred(toolauth.OAuth2Auth{
			ClientID: "client-1", ClientSecret: "secret-1", AccessToken: "stale-access", RefreshToken: "r-old", ExpiresAt: now - 3600,
		}))
		m, err := authtool.NewCredentialManager(cfg, func() time.Time { return fixedNow })
		if err != nil {
			t.Fatalf("NewCredentialManager() error = %v", err)
		}
		cred, err := m.GetAuthCredential(newToolContext(t, svc))
		if err != nil || accessToken(cred) != "cc-access" || cred.OAuth2.RefreshToken != "r-old" {
			t.Fatalf("GetAuthCredential() = %s, %v, want cc-access with refresh token r-old", accessToken(cred), err)
		}
	})
}

// Test 29. Ported flaw (auth/refresher/oauth2_credential_refresher.py:124-128):
// the stale credential is returned with the error, and saved once because
// adk-python reports a needed refresh as done (credential_manager.py:376-392).
// Fails if the stale credential is dropped or the error vanishes.
func TestGetAuthCredential_FailedRefreshKeepsStale(t *testing.T) {
	tests := []struct {
		name         string
		refreshToken string
		wantRequests int
	}{
		{name: "endpoint refuses", refreshToken: "r-old", wantRequests: 1},
		{name: "no refresh token", refreshToken: "", wantRequests: 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ts := newTokenServer(t)
			ts.set(func(s *tokenServer) { s.status, s.body = 400, invalidGrant })
			cfg := codeConfig(ts.URL)
			svc := newFakeService()
			svc.put(t, cfg, oauthCred(toolauth.OAuth2Auth{
				ClientID: "client-1", ClientSecret: "secret-1",
				AccessToken: "stale-access", RefreshToken: tt.refreshToken, ExpiresAt: fixedNow.Unix() - 3600,
			}))
			m, err := authtool.NewCredentialManager(cfg, func() time.Time { return fixedNow })
			if err != nil {
				t.Fatalf("NewCredentialManager() error = %v", err)
			}

			cred, err := m.GetAuthCredential(newToolContext(t, svc))
			if err == nil || accessToken(cred) != "stale-access" {
				t.Fatalf("GetAuthCredential() = %s, %v, want stale-access and an error", accessToken(cred), err)
			}
			if n := len(ts.seen()); n != tt.wantRequests {
				t.Errorf("token requests = %d, want %d", n, tt.wantRequests)
			}
			if _, saves := svc.counts(); saves != 1 {
				t.Errorf("service saves = %d, want 1", saves)
			}
		})
	}
}

// Test 30. Fails if the refresher runs on a credential that was just
// exchanged.
func TestGetAuthCredential_NoRefreshAfterExchange(t *testing.T) {
	ts := newTokenServer(t)
	ts.set(func(s *tokenServer) {
		s.body = `{"access_token":"cc-access","token_type":"Bearer","expires_in":20,"refresh_token":"r-new"}`
	})
	cred, err := newManager(t, ccConfig(ts.URL)).GetAuthCredential(newToolContext(t, newFakeService()))
	if err != nil || accessToken(cred) != "cc-access" {
		t.Fatalf("GetAuthCredential() = %s, %v, want cc-access, nil", accessToken(cred), err)
	}
	if n := len(ts.seen()); n != 1 {
		t.Errorf("token requests = %d, want 1", n)
	}
}

// The lead's note on u2409: a credential with no oauth2 section comes back
// unchanged and not exchanged, with an error. Fails if Exchange fails the
// call, makes a request, or exchanges a credential that holds a token.
func TestOAuth2CredentialExchanger_Exchange(t *testing.T) {
	ts := newTokenServer(t)
	noOAuth2 := &toolauth.Credential{AuthType: toolauth.CredentialTypeOAuth2}
	held := oauthCred(toolauth.OAuth2Auth{ClientID: "client-1", AccessToken: "held-access"})
	tests := []struct {
		name         string
		cfg          *toolauth.Config
		cred         *toolauth.Credential
		want         *toolauth.Credential
		wantErr      bool
		wantExchange bool
	}{
		{name: "nil config", cfg: nil, cred: held, want: nil, wantErr: true},
		{name: "scheme without type", cfg: &toolauth.Config{}, cred: held, want: nil, wantErr: true},
		{name: "no oauth2 section", cfg: ccConfig(ts.URL), cred: noOAuth2, want: noOAuth2, wantErr: true},
		{name: "token held", cfg: ccConfig(ts.URL), cred: held, want: held},
		{
			name:         "client credentials",
			cfg:          ccConfig(ts.URL),
			cred:         oauthCred(toolauth.OAuth2Auth{ClientID: "client-1", ClientSecret: "secret-1"}),
			want:         oauthCred(toolauth.OAuth2Auth{ClientID: "client-1", ClientSecret: "secret-1", AccessToken: "cc-access", ExpiresIn: 3600}),
			wantExchange: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			before := len(ts.seen())
			got, didExchange, err := authtool.OAuth2CredentialExchanger{}.Exchange(newToolContext(t, nil), tt.cfg, tt.cred)
			if gotErr := err != nil; gotErr != tt.wantErr {
				t.Fatalf("Exchange() error = %v, want error %v", err, tt.wantErr)
			}
			if didExchange != tt.wantExchange {
				t.Errorf("Exchange() didExchange = %v, want %v", didExchange, tt.wantExchange)
			}
			if diff := cmp.Diff(tt.want, got, cmpopts.IgnoreFields(toolauth.OAuth2Auth{}, "ExpiresAt")); diff != "" {
				t.Errorf("Exchange() credential mismatch (-want +got):\n%s", diff)
			}
			wantRequests := 0
			if tt.wantExchange {
				wantRequests = 1
			}
			if n := len(ts.seen()) - before; n != wantRequests {
				t.Errorf("token requests = %d, want %d", n, wantRequests)
			}
		})
	}
}

// The authorization-code grant of the exchanger, for a code the manager
// loaded from the credential service. Fails if the code in the redirect URL
// is not redeemed, or the PKCE verifier and redirect URI are not sent.
func TestGetAuthCredential_ExchangesStoredCode(t *testing.T) {
	ts := newTokenServer(t)
	cfg := codeConfig(ts.URL)
	svc := newFakeService()
	svc.put(t, cfg, oauthCred(toolauth.OAuth2Auth{
		ClientID: "client-1", ClientSecret: "secret-1", RedirectURI: "https://app.example/cb", State: "st-1",
		AuthResponseURI: "https://app.example/cb?code=code-1&state=st-1", CodeVerifier: "verifier-1",
	}))

	cred, err := newManager(t, cfg).GetAuthCredential(newToolContext(t, svc))
	if err != nil || accessToken(cred) != "cc-access" {
		t.Fatalf("GetAuthCredential() = %s, %v, want cc-access, nil", accessToken(cred), err)
	}
	reqs := ts.seen()
	if len(reqs) != 1 {
		t.Fatalf("token requests = %d, want 1", len(reqs))
	}
	want := url.Values{
		"grant_type": {"authorization_code"}, "code": {"code-1"},
		"redirect_uri": {"https://app.example/cb"}, "code_verifier": {"verifier-1"},
	}
	if diff := cmp.Diff(want, reqs[0].form); diff != "" {
		t.Errorf("token request form mismatch (-want +got):\n%s", diff)
	}
	if got := svc.stored(t, cfg); accessToken(got) != "cc-access" {
		t.Errorf("service holds %s, want cc-access", accessToken(got))
	}
}

// nilExchanger fails every exchange by returning no credential.
type nilExchanger struct{}

func (nilExchanger) Exchange(agent.Context, *toolauth.Config, *toolauth.Credential) (*toolauth.Credential, bool, error) {
	return nil, true, nil
}

// Test 31. Ruling R7: registration is per instance. Fails if registration is
// process-wide, ignored, or the refresher runs after the registered exchange.
func TestRegisterCredentialExchanger_PerInstance(t *testing.T) {
	ts := newTokenServer(t)
	cfg := ccConfig(ts.URL)
	fake := &countingExchanger{
		result:      oauthCred(toolauth.OAuth2Auth{ClientID: "client-1", AccessToken: "fake-access", RefreshToken: "r-old", ExpiresAt: 1}),
		didExchange: true,
	}
	a := newManager(t, cfg)
	a.RegisterCredentialExchanger(toolauth.CredentialTypeOAuth2, fake)
	b := newManager(t, cfg)

	credA, err := a.GetAuthCredential(newToolContext(t, nil))
	if err != nil || accessToken(credA) != "fake-access" {
		t.Fatalf("A.GetAuthCredential() = %s, %v, want fake-access, nil", accessToken(credA), err)
	}
	if fake.count() != 1 || fake.cfgs[0] != cfg {
		t.Errorf("fake exchanger calls = %d, saw the manager's config = %v, want 1, true", fake.count(), fake.count() == 1 && fake.cfgs[0] == cfg)
	}
	if diff := cmp.Diff(cfg.RawAuthCredential, fake.creds[0]); diff != "" {
		t.Errorf("fake exchanger credential mismatch (-want +got):\n%s", diff)
	}
	if n := len(ts.seen()); n != 0 {
		t.Errorf("token requests after A = %d, want 0", n)
	}

	credB, err := b.GetAuthCredential(newToolContext(t, nil))
	if err != nil || accessToken(credB) != "cc-access" {
		t.Fatalf("B.GetAuthCredential() = %s, %v, want cc-access, nil", accessToken(credB), err)
	}
	if fake.count() != 1 {
		t.Errorf("fake exchanger calls after B = %d, want 1", fake.count())
	}
}

// Test 32. Fails if a nil credential from an exchanger is returned as success
// or saved: saving a config with no credential erases the stored one.
func TestGetAuthCredential_ExchangerReturnsNil(t *testing.T) {
	cfg := ccConfig("https://idp.example/token")
	svc := newFakeService()
	m := newManager(t, cfg)
	m.RegisterCredentialExchanger(toolauth.CredentialTypeOAuth2, nilExchanger{})

	cred, err := m.GetAuthCredential(newToolContext(t, svc))
	if cred != nil || err == nil {
		t.Fatalf("GetAuthCredential() = %+v, %v, want nil and an error", cred, err)
	}
	if _, saves := svc.counts(); saves != 0 {
		t.Errorf("service saves = %d, want 0", saves)
	}
}

// Test 33. Fails if the manager writes the credential into the shared config
// (user-2 would then get user-1's token) or does not save a credential that
// came from the auth response.
func TestGetAuthCredential_SavesCopyNotSharedConfig(t *testing.T) {
	tests := []struct {
		name   string
		answer toolauth.OAuth2Auth
		want   string
	}{
		{name: "answer with a token", answer: toolauth.OAuth2Auth{AccessToken: "state-access"}, want: "state-access"},
		{name: "answer with a code", answer: toolauth.OAuth2Auth{AuthCode: "code-1"}, want: "cc-access"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ts := newTokenServer(t)
			cfg := codeConfig(ts.URL)
			svc := newFakeService()
			m := newManager(t, cfg)
			ctx := newToolContext(t, svc)
			putAuthResponse(t, ctx, *oauthCred(tt.answer))

			cred, err := m.GetAuthCredential(ctx)
			if err != nil || accessToken(cred) != tt.want {
				t.Fatalf("GetAuthCredential() = %s, %v, want %s, nil", accessToken(cred), err, tt.want)
			}
			if got := svc.stored(t, cfg); accessToken(got) != tt.want {
				t.Errorf("service holds %s, want %s", accessToken(got), tt.want)
			}
			if cfg.ExchangedAuthCredential != nil {
				t.Errorf("shared config holds %s, want no credential", accessToken(cfg.ExchangedAuthCredential))
			}
			other := newToolContext(t, svc)
			other.userID = "user-2"
			if cred, err := m.GetAuthCredential(other); cred != nil || err != nil {
				t.Errorf("GetAuthCredential(user-2) = %s, %v, want nil, nil", accessToken(cred), err)
			}
		})
	}
}

// Test 34. Fails if a credential service error is dropped.
func TestGetAuthCredential_ServiceErrorsPropagate(t *testing.T) {
	errSvc := errors.New("service unavailable")
	tests := []struct {
		name         string
		loadErr      error
		saveErr      error
		wantRequests int
	}{
		{name: "load fails", loadErr: errSvc, wantRequests: 0},
		{name: "save fails after an exchange", saveErr: errSvc, wantRequests: 1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ts := newTokenServer(t)
			svc := newFakeService()
			svc.loadErr, svc.saveErr = tt.loadErr, tt.saveErr

			cred, err := newManager(t, ccConfig(ts.URL)).GetAuthCredential(newToolContext(t, svc))
			if cred != nil || !errors.Is(err, errSvc) {
				t.Fatalf("GetAuthCredential() = %s, %v, want nil and errSvc", accessToken(cred), err)
			}
			if n := len(ts.seen()); n != tt.wantRequests {
				t.Errorf("token requests = %d, want %d", n, tt.wantRequests)
			}
		})
	}
}

// Test 41. Fails if the request is recorded under another key, carries a
// secret, or an empty function call ID is accepted.
func TestRequestCredential_RecordsForFunctionCall(t *testing.T) {
	tests := []struct {
		name    string
		callID  string
		wantErr bool
	}{
		{name: "tool call", callID: "call-1"},
		{name: "no function call", callID: "", wantErr: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := newToolContext(t, nil)
			ctx.callID = tt.callID

			err := newManager(t, codeConfig("https://idp.example/token")).RequestCredential(ctx)
			got := ctx.actions.RequestedAuthConfigs
			if tt.wantErr {
				if err == nil || len(got) != 0 {
					t.Fatalf("RequestCredential() error = %v with %d requests, want an error and none", err, len(got))
				}
				return
			}
			if err != nil {
				t.Fatalf("RequestCredential() error = %v", err)
			}
			req, ok := got["call-1"]
			if len(got) != 1 || !ok {
				t.Fatalf("RequestedAuthConfigs has %d entries, call-1 present = %v, want only call-1", len(got), ok)
			}
			if req.AuthScheme.Type != toolauth.SchemeTypeOAuth2 {
				t.Errorf("request scheme = %s, want oauth2", req.AuthScheme.Type)
			}
			b, err := json.Marshal(req)
			if err != nil {
				t.Fatalf("json.Marshal() error = %v", err)
			}
			if strings.Contains(string(b), "secret-1") {
				t.Errorf("request carries the client secret")
			}
		})
	}
}

// authlib refuses a redirect URL whose state is not the request's. Fails if
// the exchanger redeems such a code.
func TestGetAuthCredential_StoredCodeWithOtherState(t *testing.T) {
	ts := newTokenServer(t)
	cfg := codeConfig(ts.URL)
	stored := oauthCred(toolauth.OAuth2Auth{
		ClientID: "client-1", ClientSecret: "secret-1", State: "st-1",
		AuthResponseURI: "https://app.example/cb?code=code-1&state=st-2",
	})
	svc := newFakeService()
	svc.put(t, cfg, stored)

	cred, err := newManager(t, cfg).GetAuthCredential(newToolContext(t, svc))
	if err == nil || !strings.Contains(err.Error(), "state") {
		t.Fatalf("GetAuthCredential() error = %v, want one about the state", err)
	}
	if diff := cmp.Diff(stored, cred); diff != "" {
		t.Errorf("GetAuthCredential() credential mismatch (-want +got):\n%s", diff)
	}
	if n := len(ts.seen()); n != 0 {
		t.Errorf("token requests = %d, want 0", n)
	}
}

// Fails if a flow's URL check is dropped from adk-python's _missing_oauth_info
// (credential_manager.py): each of these fails the call with the same error
// instead of going on with an unexchanged credential.
func TestGetAuthCredential_MissingURLOfEachFlow(t *testing.T) {
	const want = "OAuth scheme info is missing, and auto-discovery has failed to fill them in."
	tests := []struct {
		name  string
		flows toolauth.OAuthFlows
	}{
		{name: "implicit without authorization URL", flows: toolauth.OAuthFlows{Implicit: &toolauth.OAuthFlow{}}},
		{name: "password without token URL", flows: toolauth.OAuthFlows{Password: &toolauth.OAuthFlow{}}},
		{name: "client credentials without token URL", flows: toolauth.OAuthFlows{ClientCredentials: &toolauth.OAuthFlow{}}},
		{name: "authorization code without authorization URL", flows: toolauth.OAuthFlows{
			AuthorizationCode: &toolauth.OAuthFlow{TokenURL: "https://idp.example/token"},
		}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := ccConfig("https://idp.example/token")
			cfg.AuthScheme.Flows = &tt.flows
			cred, err := newManager(t, cfg).GetAuthCredential(newToolContext(t, newFakeService()))
			if err == nil || err.Error() != want || cred != nil {
				t.Errorf("GetAuthCredential() = %v, %v, want nil and %q", cred, err, want)
			}
		})
	}
}

// Fails if an OpenID Connect credential is never refreshed (the refresher is
// registered for oauth2 only).
func TestGetAuthCredential_RefreshesOpenIDConnectCredential(t *testing.T) {
	ts := newTokenServer(t)
	ts.set(func(s *tokenServer) { s.body = refreshTokenBody })
	cfg := &toolauth.Config{
		AuthScheme: toolauth.Scheme{
			Type: toolauth.SchemeTypeOpenIDConnect, AuthorizationEndpoint: "https://idp.example/auth", TokenEndpoint: ts.URL,
		},
		RawAuthCredential: &toolauth.Credential{AuthType: toolauth.CredentialTypeOpenIDConnect, OAuth2: &toolauth.OAuth2Auth{
			ClientID: "client-1", ClientSecret: "secret-1",
		}},
		CredentialKey: "key-1",
	}
	svc := newFakeService()
	svc.put(t, cfg, &toolauth.Credential{AuthType: toolauth.CredentialTypeOpenIDConnect, OAuth2: &toolauth.OAuth2Auth{
		ClientID: "client-1", ClientSecret: "secret-1", AccessToken: "stale-access", RefreshToken: "r-old", ExpiresAt: fixedNow.Unix() - 3600,
	}})
	m, err := authtool.NewCredentialManager(cfg, func() time.Time { return fixedNow })
	if err != nil {
		t.Fatalf("NewCredentialManager() error = %v", err)
	}
	cred, err := m.GetAuthCredential(newToolContext(t, svc))
	if err != nil || accessToken(cred) != "cc-access" {
		t.Fatalf("GetAuthCredential() = %s, %v, want cc-access, nil", accessToken(cred), err)
	}
	if n := len(ts.seen()); n != 1 {
		t.Errorf("token requests = %d, want 1", n)
	}
}

// Fails if a refresh that cannot even be built (a stored credential with no
// client ID) fails the call instead of returning the stale credential with
// the error, as a refresh that the endpoint refuses does.
func TestGetAuthCredential_RefreshWithoutClientIDKeepsStale(t *testing.T) {
	ts := newTokenServer(t)
	cfg := codeConfig(ts.URL)
	svc := newFakeService()
	svc.put(t, cfg, oauthCred(toolauth.OAuth2Auth{AccessToken: "stale-access", RefreshToken: "r-old", ExpiresAt: fixedNow.Unix() - 3600}))
	m, err := authtool.NewCredentialManager(cfg, func() time.Time { return fixedNow })
	if err != nil {
		t.Fatalf("NewCredentialManager() error = %v", err)
	}
	cred, err := m.GetAuthCredential(newToolContext(t, svc))
	if err == nil || accessToken(cred) != "stale-access" {
		t.Fatalf("GetAuthCredential() = %s, %v, want stale-access and an error", accessToken(cred), err)
	}
	if n := len(ts.seen()); n != 0 {
		t.Errorf("token requests = %d, want 0", n)
	}
}

// Fails if Exchange panics on a nil credential, or fails the call (a nil
// credential) for a grant it does not support, where adk-python returns the
// credential unchanged.
func TestOAuth2CredentialExchanger_NilCredentialAndUnsupportedGrant(t *testing.T) {
	ts := newTokenServer(t)
	password := ccConfig(ts.URL)
	password.AuthScheme.Flows = &toolauth.OAuthFlows{Password: &toolauth.OAuthFlow{TokenURL: ts.URL}}
	bare := oauthCred(toolauth.OAuth2Auth{ClientID: "client-1", ClientSecret: "secret-1"})

	got, didExchange, err := authtool.OAuth2CredentialExchanger{}.Exchange(newToolContext(t, nil), ccConfig(ts.URL), nil)
	if got != nil || didExchange || err == nil {
		t.Errorf("Exchange(nil credential) = %v, %v, %v, want nil, false and an error", got, didExchange, err)
	}

	got, didExchange, err = authtool.OAuth2CredentialExchanger{}.Exchange(newToolContext(t, nil), password, bare)
	if diff := cmp.Diff(bare, got); diff != "" || didExchange || err == nil {
		t.Errorf("Exchange(password grant) = %v, %v, %v, want the credential unchanged, false and an error\n%s", got, didExchange, err, diff)
	}
	if n := len(ts.seen()); n != 0 {
		t.Errorf("token requests = %d, want 0", n)
	}
}

// Fails if a token request does not stop when the call's context ends: the
// caller would wait for the endpoint's answer (up to the ten-second limit).
func TestGetAuthCredential_CancelledDuringTokenRequest(t *testing.T) {
	for _, tt := range []struct {
		name string
		cfg  func(url string) *toolauth.Config
		prep func(t *testing.T, ctx *fakeToolContext, svc *fakeService, cfg *toolauth.Config)
	}{
		{name: "client credentials", cfg: ccConfig, prep: func(*testing.T, *fakeToolContext, *fakeService, *toolauth.Config) {}},
		{name: "authorization code", cfg: codeConfig, prep: func(t *testing.T, _ *fakeToolContext, svc *fakeService, cfg *toolauth.Config) {
			svc.put(t, cfg, oauthCred(toolauth.OAuth2Auth{
				ClientID: "client-1", ClientSecret: "secret-1", RedirectURI: "https://app.example/cb", State: "st-1",
				AuthResponseURI: "https://app.example/cb?code=code-1&state=st-1",
			}))
		}},
		{name: "refresh", cfg: codeConfig, prep: func(t *testing.T, _ *fakeToolContext, svc *fakeService, cfg *toolauth.Config) {
			svc.put(t, cfg, oauthCred(toolauth.OAuth2Auth{
				ClientID: "client-1", ClientSecret: "secret-1", AccessToken: "stale-access", RefreshToken: "r-old", ExpiresAt: 1,
			}))
		}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			ts := newTokenServer(t)
			release := make(chan struct{})
			ts.set(func(s *tokenServer) { s.release = release })
			t.Cleanup(func() { close(release) })
			cfg := tt.cfg(ts.URL)
			svc := newFakeService()
			parent, cancel := context.WithCancel(t.Context())
			t.Cleanup(cancel)
			ctx := newToolContextFrom(t, parent, svc)
			tt.prep(t, ctx, svc, cfg)
			m := newManager(t, cfg)

			done := make(chan error, 1)
			go func() { _, err := m.GetAuthCredential(ctx); done <- err }()
			<-ts.arrived
			cancel()
			select {
			case err := <-done:
				if err == nil || !errors.Is(err, parent.Err()) {
					t.Errorf("GetAuthCredential() error = %v, want one wrapping the context's error", err)
				}
			case <-time.After(3 * time.Second):
				t.Fatal("GetAuthCredential() still waiting for the token endpoint 3 s after the context ended")
			}
		})
	}
}

// Fails if a token response that gives expires_in as a string loses it, so
// that the credential never counts as about to expire.
func TestOAuth2CredentialExchanger_StringExpiresIn(t *testing.T) {
	ts := newTokenServer(t)
	ts.set(func(s *tokenServer) {
		s.body = `{"access_token":"cc-access","token_type":"Bearer","expires_in":"3600"}`
	})
	got, _, err := authtool.OAuth2CredentialExchanger{}.Exchange(newToolContext(t, nil), ccConfig(ts.URL),
		oauthCred(toolauth.OAuth2Auth{ClientID: "client-1", ClientSecret: "secret-1"}))
	if err != nil || got.OAuth2.ExpiresIn != 3600 {
		t.Fatalf("Exchange() = %+v, %v, want ExpiresIn 3600", got, err)
	}
}

// Fails if the credential the manager hands out when the exchange failed is
// the config's own raw credential, which a function could then change for every
// user.
func TestGetAuthCredential_FallbackCredentialIsACopy(t *testing.T) {
	ts := newTokenServer(t)
	ts.set(func(s *tokenServer) { s.status, s.body = 400, invalidClient })
	cfg := ccConfig(ts.URL)
	cred, err := newManager(t, cfg).GetAuthCredential(newToolContext(t, nil))
	if err == nil || cred == nil {
		t.Fatalf("GetAuthCredential() = %v, %v, want the raw copy and an error", cred, err)
	}
	cred.OAuth2.ClientID = "changed"
	if got := cfg.RawAuthCredential.OAuth2.ClientID; got != "client-1" {
		t.Errorf("config raw credential ClientID = %q after the caller changed the copy, want client-1", got)
	}
}

// Fails if a credential of a type with no registered exchanger, held in the
// credential service, panics the manager instead of being returned as it is.
func TestGetAuthCredential_StoredCredentialWithoutExchangerIsReturned(t *testing.T) {
	cfg := codeConfig("https://idp.example/token")
	svc := newFakeService()
	svc.put(t, cfg, &toolauth.Credential{AuthType: toolauth.CredentialTypeAPIKey, APIKey: "test-api-key-1"})
	cred, err := newManager(t, cfg).GetAuthCredential(newToolContext(t, svc))
	if err != nil || cred == nil || cred.APIKey != "test-api-key-1" {
		t.Errorf("GetAuthCredential() = %+v, %v, want the stored API key credential", cred, err)
	}
}
