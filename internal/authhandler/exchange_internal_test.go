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
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"golang.org/x/oauth2"

	"google.golang.org/adk/v2/tool/toolauth"
)

func TestAuthCode(t *testing.T) {
	tests := []struct {
		name string
		o    toolauth.OAuth2Auth
		want string
	}{
		{
			name: "redirect with the matching state",
			o:    toolauth.OAuth2Auth{State: "Xy7", AuthResponseURI: "https://adk.example/cb?state=Xy7&code=4/0Ab"},
			want: "4/0Ab",
		},
		{
			name: "redirect ending in #",
			o:    toolauth.OAuth2Auth{State: "Xy7", AuthResponseURI: "https://adk.example/cb?state=Xy7&code=4/0Ab#"},
			want: "4/0Ab",
		},
		{
			name: "no state on the credential, so none is checked",
			o:    toolauth.OAuth2Auth{AuthResponseURI: "https://adk.example/cb?state=anything&code=4/0Ab"},
			want: "4/0Ab",
		},
		{
			name: "redirect without a code falls back to AuthCode",
			o: toolauth.OAuth2Auth{
				State:           "Xy7",
				AuthResponseURI: "https://adk.example/cb?error=access_denied&state=Xy7",
				AuthCode:        "sent-alone",
			},
			want: "sent-alone",
		},
		{
			name: "no redirect, only AuthCode",
			o:    toolauth.OAuth2Auth{AuthCode: "sent-alone"},
			want: "sent-alone",
		},
		{
			name: "the redirect's code wins over AuthCode",
			o: toolauth.OAuth2Auth{
				State:           "Xy7",
				AuthResponseURI: "https://adk.example/cb?state=Xy7&code=4/0Ab",
				AuthCode:        "sent-alone",
			},
			want: "4/0Ab",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := authCode(&tt.o)
			if err != nil {
				t.Fatalf("authCode() error = %v", err)
			}
			if got != tt.want {
				t.Errorf("authCode() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestAuthCodeRejects(t *testing.T) {
	tests := []struct {
		name string
		o    toolauth.OAuth2Auth
	}{
		{
			name: "a different state",
			o:    toolauth.OAuth2Auth{State: "Xy7", AuthResponseURI: "https://adk.example/cb?state=Q2w&code=9-zzz"},
		},
		{
			name: "state missing from the redirect",
			o:    toolauth.OAuth2Auth{State: "Xy7", AuthResponseURI: "https://adk.example/cb?code=9-zzz"},
		},
		{
			name: "no code anywhere",
			o:    toolauth.OAuth2Auth{State: "Xy7", AuthResponseURI: "https://adk.example/cb?state=Xy7"},
		},
		{
			// AuthCode is set so that skipping the URL would still find a code.
			name: "redirect that does not parse",
			o: toolauth.OAuth2Auth{
				State:           "Xy7",
				AuthResponseURI: "http://[::1/cb?state=Xy7&code=9-zzz",
				AuthCode:        "sent-alone",
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := authCode(&tt.o)
			if err == nil {
				t.Fatalf("authCode() error = nil, want an error")
			}
			for _, secret := range []string{"9-zzz", "Xy7", "Q2w"} {
				if strings.Contains(err.Error(), secret) {
					t.Errorf("error quotes %q from the answer: %v", secret, err)
				}
			}
		})
	}
}

func codeFlow(tokenURL string) toolauth.Scheme {
	return toolauth.Scheme{
		Type: toolauth.SchemeTypeOAuth2,
		Flows: &toolauth.OAuthFlows{AuthorizationCode: &toolauth.OAuthFlow{
			AuthorizationURL: "https://idp.example/authorize",
			TokenURL:         tokenURL,
		}},
	}
}

func TestTokenConfig(t *testing.T) {
	withSecret := toolauth.OAuth2Auth{ClientID: "app-123", ClientSecret: "s3cret", RedirectURI: "https://adk.example/cb"}
	public := toolauth.OAuth2Auth{ClientID: "app-123", RedirectURI: "https://adk.example/cb"}
	tests := []struct {
		name   string
		scheme toolauth.Scheme
		o      toolauth.OAuth2Auth
		want   *oauth2.Config
	}{
		{
			name:   "code flow, client with a secret, header",
			scheme: codeFlow("https://idp.example/token"),
			o:      withSecret,
			want: &oauth2.Config{
				ClientID: "app-123", ClientSecret: "s3cret", RedirectURL: "https://adk.example/cb",
				Endpoint: oauth2.Endpoint{TokenURL: "https://idp.example/token", AuthStyle: oauth2.AuthStyleInHeader},
			},
		},
		{
			name:   "client_secret_post puts the secret in the body",
			scheme: codeFlow("https://idp.example/token"),
			o: toolauth.OAuth2Auth{
				ClientID: "app-123", ClientSecret: "s3cret", RedirectURI: "https://adk.example/cb",
				TokenEndpointAuthMethod: "client_secret_post",
			},
			want: &oauth2.Config{
				ClientID: "app-123", ClientSecret: "s3cret", RedirectURL: "https://adk.example/cb",
				Endpoint: oauth2.Endpoint{TokenURL: "https://idp.example/token", AuthStyle: oauth2.AuthStyleInParams},
			},
		},
		{
			name:   "public client sends only its ID in the body",
			scheme: codeFlow("https://idp.example/token"),
			o:      public,
			want: &oauth2.Config{
				ClientID: "app-123", RedirectURL: "https://adk.example/cb",
				Endpoint: oauth2.Endpoint{TokenURL: "https://idp.example/token", AuthStyle: oauth2.AuthStyleInParams},
			},
		},
		{
			name:   "public client named client_secret_jwt is still public",
			scheme: codeFlow("https://idp.example/token"),
			o:      toolauth.OAuth2Auth{ClientID: "app-123", TokenEndpointAuthMethod: "client_secret_jwt"},
			want: &oauth2.Config{
				ClientID: "app-123",
				Endpoint: oauth2.Endpoint{TokenURL: "https://idp.example/token", AuthStyle: oauth2.AuthStyleInParams},
			},
		},
		{
			name: "openid connect token endpoint",
			scheme: toolauth.Scheme{
				Type:                  toolauth.SchemeTypeOpenIDConnect,
				AuthorizationEndpoint: "https://oidc.example/auth",
				TokenEndpoint:         "https://oidc.example/token",
			},
			o: withSecret,
			want: &oauth2.Config{
				ClientID: "app-123", ClientSecret: "s3cret", RedirectURL: "https://adk.example/cb",
				Endpoint: oauth2.Endpoint{TokenURL: "https://oidc.example/token", AuthStyle: oauth2.AuthStyleInHeader},
			},
		},
		{
			name: "code flow token URL wins over client credentials",
			scheme: toolauth.Scheme{
				Type: toolauth.SchemeTypeOAuth2,
				Flows: &toolauth.OAuthFlows{
					AuthorizationCode: &toolauth.OAuthFlow{
						AuthorizationURL: "https://idp.example/authorize",
						TokenURL:         "https://idp.example/token",
					},
					ClientCredentials: &toolauth.OAuthFlow{TokenURL: "https://idp.example/cc-token"},
				},
			},
			o: withSecret,
			want: &oauth2.Config{
				ClientID: "app-123", ClientSecret: "s3cret", RedirectURL: "https://adk.example/cb",
				Endpoint: oauth2.Endpoint{TokenURL: "https://idp.example/token", AuthStyle: oauth2.AuthStyleInHeader},
			},
		},
		{
			name: "client credentials token URL when the code flow has none",
			scheme: toolauth.Scheme{
				Type: toolauth.SchemeTypeOAuth2,
				Flows: &toolauth.OAuthFlows{
					AuthorizationCode: &toolauth.OAuthFlow{AuthorizationURL: "https://idp.example/authorize"},
					ClientCredentials: &toolauth.OAuthFlow{TokenURL: "https://idp.example/cc-token"},
				},
			},
			o: withSecret,
			want: &oauth2.Config{
				ClientID: "app-123", ClientSecret: "s3cret", RedirectURL: "https://adk.example/cb",
				Endpoint: oauth2.Endpoint{TokenURL: "https://idp.example/cc-token", AuthStyle: oauth2.AuthStyleInHeader},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := tokenConfig(&tt.scheme, &tt.o)
			if err != nil {
				t.Fatalf("tokenConfig() error = %v", err)
			}
			if diff := cmp.Diff(tt.want, got, cmpopts.IgnoreUnexported(oauth2.Config{})); diff != "" {
				t.Errorf("tokenConfig() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestTokenConfigRejects(t *testing.T) {
	withSecret := toolauth.OAuth2Auth{ClientID: "app-123", ClientSecret: "s3cret"}
	tests := []struct {
		name   string
		scheme toolauth.Scheme
		o      toolauth.OAuth2Auth
	}{
		{
			name: "implicit flow only",
			scheme: toolauth.Scheme{
				Type:  toolauth.SchemeTypeOAuth2,
				Flows: &toolauth.OAuthFlows{Implicit: &toolauth.OAuthFlow{AuthorizationURL: "https://idp.example/authorize"}},
			},
			o: withSecret,
		},
		{
			name: "openid connect with only a discovery URL",
			scheme: toolauth.Scheme{
				Type:             toolauth.SchemeTypeOpenIDConnect,
				OpenIDConnectURL: "https://oidc.example/.well-known/openid-configuration",
			},
			o: withSecret,
		},
		{
			name: "custom scheme type",
			scheme: toolauth.Scheme{
				Type:  "gcpAuthProviderScheme",
				Flows: codeFlow("https://idp.example/token").Flows,
			},
			o: withSecret,
		},
		{
			name:   "no client ID",
			scheme: codeFlow("https://idp.example/token"),
			o:      toolauth.OAuth2Auth{ClientSecret: "s3cret"},
		},
		{
			name:   "private_key_jwt",
			scheme: codeFlow("https://idp.example/token"),
			o:      toolauth.OAuth2Auth{ClientID: "app-123", TokenEndpointAuthMethod: "private_key_jwt"},
		},
		{
			name:   "client_secret_jwt with a secret",
			scheme: codeFlow("https://idp.example/token"),
			o:      toolauth.OAuth2Auth{ClientID: "app-123", ClientSecret: "s3cret", TokenEndpointAuthMethod: "client_secret_jwt"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := tokenConfig(&tt.scheme, &tt.o); err == nil {
				t.Errorf("tokenConfig() error = nil, want an error")
			}
		})
	}
}

const tokenResponse = `{"access_token":"at-1","token_type":"Bearer","refresh_token":"rt-1","expires_in":3600,"id_token":"idt-1"}`

// tokenEndpointStub records the requests a fake token endpoint receives.
type tokenEndpointStub struct {
	mu         sync.Mutex
	calls      int
	form       url.Values
	authHeader string
}

func (s *tokenEndpointStub) seen() (int, url.Values) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.calls, s.form
}

// seenAuthHeader returns the Authorization header of the last request.
func (s *tokenEndpointStub) seenAuthHeader() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.authHeader
}

// newTokenEndpoint serves a token endpoint that answers every request with
// status and body, and returns its URL.
func newTokenEndpoint(t *testing.T, status int, body string) (*tokenEndpointStub, string) {
	t.Helper()
	stub := &tokenEndpointStub{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := r.ParseForm(); err != nil {
			t.Errorf("ParseForm() error = %v", err)
		}
		stub.mu.Lock()
		stub.calls++
		stub.form = r.PostForm
		stub.authHeader = r.Header.Get("Authorization")
		stub.mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		if _, err := w.Write([]byte(body)); err != nil {
			t.Errorf("Write() error = %v", err)
		}
	}))
	t.Cleanup(srv.Close)
	return stub, srv.URL + "/token"
}

// answeredCredential is a public client's credential after the user signed in.
func answeredCredential() *toolauth.Credential {
	return &toolauth.Credential{
		AuthType: toolauth.CredentialTypeOAuth2,
		OAuth2: &toolauth.OAuth2Auth{
			ClientID:            "app-123",
			State:               "Xy7",
			AuthResponseURI:     "https://adk.example/cb?state=Xy7&code=4/0Ab",
			RedirectURI:         "https://adk.example/cb",
			CodeVerifier:        "v-secret",
			CodeChallengeMethod: "S256",
		},
	}
}

func TestExchangeCodeSendsTheTokenRequest(t *testing.T) {
	tests := []struct {
		name string
		cred func() *toolauth.Credential
		want url.Values
	}{
		{
			name: "public client with a PKCE verifier",
			cred: answeredCredential,
			want: url.Values{
				"grant_type":    {"authorization_code"},
				"code":          {"4/0Ab"},
				"redirect_uri":  {"https://adk.example/cb"},
				"code_verifier": {"v-secret"},
				"client_id":     {"app-123"},
			},
		},
		{
			// The client ID and secret go in the Authorization header.
			name: "client with a secret and no verifier sends no code_verifier",
			cred: func() *toolauth.Credential {
				c := answeredCredential()
				c.OAuth2.ClientSecret = "s3cret"
				c.OAuth2.CodeVerifier, c.OAuth2.CodeChallengeMethod = "", ""
				return c
			},
			want: url.Values{
				"grant_type":   {"authorization_code"},
				"code":         {"4/0Ab"},
				"redirect_uri": {"https://adk.example/cb"},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			stub, tokenURL := newTokenEndpoint(t, http.StatusOK, tokenResponse)
			scheme := codeFlow(tokenURL)
			if _, err := exchangeCode(t.Context(), &scheme, tt.cred()); err != nil {
				t.Fatalf("exchangeCode() error = %v", err)
			}
			if _, form := stub.seen(); !cmp.Equal(tt.want, form) {
				t.Errorf("token request mismatch (-want +got):\n%s", cmp.Diff(tt.want, form))
			}
		})
	}
}

func TestExchangeCodeWithoutExpiresInLeavesTheExpiryUnset(t *testing.T) {
	_, tokenURL := newTokenEndpoint(t, http.StatusOK, `{"access_token":"at-1","token_type":"Bearer"}`)
	scheme := codeFlow(tokenURL)
	got, err := exchangeCode(t.Context(), &scheme, answeredCredential())
	if err != nil {
		t.Fatalf("exchangeCode() error = %v", err)
	}
	if got.OAuth2.ExpiresAt != 0 || got.OAuth2.ExpiresIn != 0 {
		t.Errorf("ExpiresAt, ExpiresIn = %d, %d; want 0, 0", got.OAuth2.ExpiresAt, got.OAuth2.ExpiresIn)
	}
}

func TestExchangeCodeFillsInTheTokens(t *testing.T) {
	_, tokenURL := newTokenEndpoint(t, http.StatusOK, tokenResponse)
	scheme := codeFlow(tokenURL)
	before := time.Now().Unix()
	got, err := exchangeCode(t.Context(), &scheme, answeredCredential())
	after := time.Now().Unix()
	if err != nil {
		t.Fatalf("exchangeCode() error = %v", err)
	}
	want := answeredCredential()
	want.OAuth2.AccessToken = "at-1"
	want.OAuth2.RefreshToken = "rt-1"
	want.OAuth2.IDToken = "idt-1"
	want.OAuth2.ExpiresIn = 3600
	if diff := cmp.Diff(want, got, cmpopts.IgnoreFields(toolauth.OAuth2Auth{}, "ExpiresAt")); diff != "" {
		t.Errorf("exchanged credential mismatch (-want +got):\n%s", diff)
	}
	if at := got.OAuth2.ExpiresAt; at < before+3600 || at > after+3600 {
		t.Errorf("ExpiresAt = %d, want between %d and %d", at, before+3600, after+3600)
	}
}

func TestExchangeCodeKeepsACredentialThatHasAToken(t *testing.T) {
	stub, tokenURL := newTokenEndpoint(t, http.StatusOK, tokenResponse)
	scheme := codeFlow(tokenURL)
	cred := answeredCredential()
	cred.OAuth2.AccessToken = "already"
	got, err := exchangeCode(t.Context(), &scheme, cred)
	if err != nil {
		t.Fatalf("exchangeCode() error = %v", err)
	}
	want := answeredCredential()
	want.OAuth2.AccessToken = "already"
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("credential changed (-want +got):\n%s", diff)
	}
	if calls, _ := stub.seen(); calls != 0 {
		t.Errorf("token endpoint got %d requests, want 0", calls)
	}
}

func TestExchangeCodeErrorLeavesOutTheResponseBody(t *testing.T) {
	_, tokenURL := newTokenEndpoint(t, http.StatusBadRequest,
		`{"error":"invalid_grant","error_description":"code was issued to alice@example.com"}`)
	scheme := codeFlow(tokenURL)
	_, err := exchangeCode(t.Context(), &scheme, answeredCredential())
	if err == nil {
		t.Fatalf("exchangeCode() error = nil, want an error")
	}
	if msg := err.Error(); !strings.Contains(msg, "400") || !strings.Contains(msg, "invalid_grant") {
		t.Errorf("error = %q, want the status and the OAuth error code", msg)
	}
	if strings.Contains(err.Error(), "alice@example.com") {
		t.Errorf("error quotes the provider's description: %v", err)
	}
}

func TestExchangeCodeRefusesOtherGrants(t *testing.T) {
	tests := []struct {
		name   string
		scheme func(tokenURL string) toolauth.Scheme
	}{
		{"client credentials flow", func(tokenURL string) toolauth.Scheme {
			return toolauth.Scheme{
				Type:  toolauth.SchemeTypeOAuth2,
				Flows: &toolauth.OAuthFlows{ClientCredentials: &toolauth.OAuthFlow{TokenURL: tokenURL}},
			}
		}},
		{"openid connect that supports client credentials", func(tokenURL string) toolauth.Scheme {
			return toolauth.Scheme{
				Type:                  toolauth.SchemeTypeOpenIDConnect,
				AuthorizationEndpoint: "https://oidc.example/auth",
				TokenEndpoint:         tokenURL,
				GrantTypesSupported:   []string{"authorization_code", "client_credentials"},
			}
		}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			stub, tokenURL := newTokenEndpoint(t, http.StatusOK, tokenResponse)
			scheme := tt.scheme(tokenURL)
			if _, err := exchangeCode(t.Context(), &scheme, answeredCredential()); err == nil {
				t.Errorf("exchangeCode() error = nil, want an error")
			}
			if calls, _ := stub.seen(); calls != 0 {
				t.Errorf("token endpoint got %d requests, want 0", calls)
			}
		})
	}
}

func TestExchangeCodeLeavesInputUnchanged(t *testing.T) {
	_, tokenURL := newTokenEndpoint(t, http.StatusOK, tokenResponse)
	scheme := codeFlow(tokenURL)
	cred := answeredCredential()
	if _, err := exchangeCode(t.Context(), &scheme, cred); err != nil {
		t.Fatalf("exchangeCode() error = %v", err)
	}
	if diff := cmp.Diff(answeredCredential(), cred); diff != "" {
		t.Errorf("input credential was modified (-want +got):\n%s", diff)
	}
}
