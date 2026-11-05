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
	"errors"
	"iter"
	"net/http"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"

	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool/toolauth"
)

// stateWith returns a real in-memory session state holding values.
func stateWith(t *testing.T, values map[string]any) session.State {
	t.Helper()
	resp, err := session.InMemoryService().Create(t.Context(), &session.CreateRequest{AppName: "app", UserID: "user"})
	if err != nil {
		t.Fatalf("Create() error = %v", err)
	}
	state := resp.Session.State()
	for k, v := range values {
		if err := state.Set(k, v); err != nil {
			t.Fatalf("state.Set(%q) error = %v", k, err)
		}
	}
	return state
}

var (
	apiKeyScheme = toolauth.Scheme{Type: toolauth.SchemeTypeAPIKey, In: "header", Name: "X-Api-Key"}
	httpScheme   = toolauth.Scheme{Type: toolauth.SchemeTypeHTTP, Scheme: "bearer"}
	oauth2Scheme = toolauth.Scheme{Type: toolauth.SchemeTypeOAuth2}
	signedIn     = toolauth.Credential{
		AuthType: toolauth.CredentialTypeOAuth2,
		OAuth2:   &toolauth.OAuth2Auth{ClientID: "app-123", AuthResponseURI: "https://adk.example/cb?code=4/0Ab"},
	}
)

func TestReadStoredCredential(t *testing.T) {
	tests := []struct {
		name         string
		scheme       toolauth.Scheme
		values       map[string]any
		wantStateKey string
		want         *toolauth.Credential
	}{
		{
			name:         "credential under temp, as StoreAuthResponse writes it",
			scheme:       oauth2Scheme,
			values:       map[string]any{"temp:k": signedIn},
			wantStateKey: "temp:k",
			want:         &signedIn,
		},
		{
			name:   "JSON object under the bare key",
			scheme: oauth2Scheme,
			values: map[string]any{"k": map[string]any{
				"authType": "oauth2",
				"oauth2":   map[string]any{"clientId": "app-123", "accessToken": "at-1"},
			}},
			wantStateKey: "k",
			want: &toolauth.Credential{
				AuthType: toolauth.CredentialTypeOAuth2,
				OAuth2:   &toolauth.OAuth2Auth{ClientID: "app-123", AccessToken: "at-1"},
			},
		},
		{
			name:         "token string for an API key scheme",
			scheme:       apiKeyScheme,
			values:       map[string]any{"k": "tok"},
			wantStateKey: "k",
			want:         &toolauth.Credential{AuthType: toolauth.CredentialTypeAPIKey, APIKey: "tok"},
		},
		{
			name:         "token string for an HTTP scheme",
			scheme:       httpScheme,
			values:       map[string]any{"k": "tok"},
			wantStateKey: "k",
			want: &toolauth.Credential{AuthType: toolauth.CredentialTypeHTTP, HTTP: &toolauth.HTTPAuth{
				Scheme:      "bearer",
				Credentials: toolauth.HTTPCredentials{Token: "tok"},
			}},
		},
		{
			name:         "token string for an HTTP scheme with no scheme name is a bearer token",
			scheme:       toolauth.Scheme{Type: toolauth.SchemeTypeHTTP},
			values:       map[string]any{"k": "tok"},
			wantStateKey: "k",
			want: &toolauth.Credential{AuthType: toolauth.CredentialTypeHTTP, HTTP: &toolauth.HTTPAuth{
				Scheme:      "bearer",
				Credentials: toolauth.HTTPCredentials{Token: "tok"},
			}},
		},
		{
			name:         "token string for an OAuth2 scheme",
			scheme:       oauth2Scheme,
			values:       map[string]any{"k": "tok"},
			wantStateKey: "k",
			want:         &toolauth.Credential{AuthType: toolauth.CredentialTypeOAuth2, OAuth2: &toolauth.OAuth2Auth{AccessToken: "tok"}},
		},
		{
			name:         "temp wins when both hold one",
			scheme:       oauth2Scheme,
			values:       map[string]any{"temp:k": signedIn, "k": "older-token"},
			wantStateKey: "temp:k",
			want:         &signedIn,
		},
		{
			name:         "empty string under temp falls through to the bare key",
			scheme:       oauth2Scheme,
			values:       map[string]any{"temp:k": "", "k": "tok"},
			wantStateKey: "k",
			want:         &toolauth.Credential{AuthType: toolauth.CredentialTypeOAuth2, OAuth2: &toolauth.OAuth2Auth{AccessToken: "tok"}},
		},
		{
			name:         "pointer to a credential, as GetAuthResponse returns it",
			scheme:       apiKeyScheme,
			values:       map[string]any{"k": &toolauth.Credential{AuthType: toolauth.CredentialTypeAPIKey, APIKey: "user-key"}},
			wantStateKey: "k",
			want:         &toolauth.Credential{AuthType: toolauth.CredentialTypeAPIKey, APIKey: "user-key"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			stateKey, got, err := readStoredCredential(stateWith(t, tt.values), &tt.scheme, "k")
			if err != nil {
				t.Fatalf("readStoredCredential() error = %v", err)
			}
			if stateKey != tt.wantStateKey {
				t.Errorf("state key = %q, want %q", stateKey, tt.wantStateKey)
			}
			if diff := cmp.Diff(tt.want, got); diff != "" {
				t.Errorf("credential mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestReadStoredCredentialFindsNothing(t *testing.T) {
	tests := []struct {
		name   string
		values map[string]any
	}{
		{"nothing stored", nil},
		{"a nil value under the bare key", map[string]any{"k": nil}},
		{"a number", map[string]any{"temp:k": 42}},
		{"a nil credential pointer", map[string]any{"temp:k": (*toolauth.Credential)(nil)}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			stateKey, got, err := readStoredCredential(stateWith(t, tt.values), &oauth2Scheme, "k")
			if err != nil || got != nil || stateKey != "" {
				t.Errorf("readStoredCredential() = %q, %v, %v; want nothing found and no error", stateKey, got, err)
			}
		})
	}
}

// failingState is a session.State whose reads fail.
type failingState struct{}

func (failingState) Get(string) (any, error)     { return nil, errors.New("storage unavailable") }
func (failingState) Set(string, any) error       { return nil }
func (failingState) All() iter.Seq2[string, any] { return func(func(string, any) bool) {} }

func TestReadStoredCredentialErrors(t *testing.T) {
	const planted = "PLANTED-SECRET"
	tests := []struct {
		name  string
		state func(t *testing.T) session.State
	}{
		{"JSON object that is not a credential", func(t *testing.T) session.State {
			return stateWith(t, map[string]any{"temp:k": map[string]any{"authType": planted}})
		}},
		{"state that fails to read", func(*testing.T) session.State { return failingState{} }},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, _, err := readStoredCredential(tt.state(t), &oauth2Scheme, "k")
			if err == nil {
				t.Fatalf("readStoredCredential() error = nil, want an error")
			}
			if strings.Contains(err.Error(), planted) {
				t.Errorf("error quotes the stored value: %v", err)
			}
		})
	}
}

// toolConfig is a tool's own config for a client with a secret.
func toolConfig(tokenURL string) *toolauth.Config {
	return &toolauth.Config{
		AuthScheme: codeFlow(tokenURL),
		RawAuthCredential: &toolauth.Credential{
			AuthType: toolauth.CredentialTypeOAuth2,
			OAuth2:   &toolauth.OAuth2Auth{ClientID: "app-123", ClientSecret: "s3cret"},
		},
		CredentialKey: "k",
	}
}

// storedCode is a signed-in answer as StoreAuthResponse would leave it, but
// naming another app, so the configured one has to win.
func storedCode() toolauth.Credential {
	c := *answeredCredential()
	c.OAuth2.ClientID = "other-app"
	return c
}

// exchangedCredential is answeredCredential after the token endpoint answered
// with tokenResponse, with the client secret set to secret.
func exchangedCredential(secret string) toolauth.Credential {
	c := *answeredCredential()
	c.OAuth2.ClientSecret = secret
	c.OAuth2.AccessToken = "at-1"
	c.OAuth2.RefreshToken = "rt-1"
	c.OAuth2.IDToken = "idt-1"
	c.OAuth2.ExpiresIn = 3600
	return c
}

var ignoreExpiresAt = cmpopts.IgnoreFields(toolauth.OAuth2Auth{}, "ExpiresAt")

func TestGetAuthResponseExchangesAStoredCode(t *testing.T) {
	tests := []struct {
		name string
		cfg  func(tokenURL string) *toolauth.Config
	}{
		{"oauth2 code flow", toolConfig},
		{"openid connect", func(tokenURL string) *toolauth.Config {
			cfg := toolConfig(tokenURL)
			cfg.AuthScheme = toolauth.Scheme{
				Type:                  toolauth.SchemeTypeOpenIDConnect,
				AuthorizationEndpoint: "https://oidc.example/auth",
				TokenEndpoint:         tokenURL,
			}
			return cfg
		}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, tokenURL := newTokenEndpoint(t, http.StatusOK, tokenResponse)
			state := stateWith(t, map[string]any{"temp:k": storedCode()})
			got, err := GetAuthResponse(t.Context(), state, tt.cfg(tokenURL))
			if err != nil {
				t.Fatalf("GetAuthResponse() error = %v", err)
			}
			want := exchangedCredential("s3cret")
			if diff := cmp.Diff(&want, got, ignoreExpiresAt); diff != "" {
				t.Errorf("GetAuthResponse() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// The tokens go back where the code was found, so the spent code is replaced.
func TestGetAuthResponseWritesTheTokensBackWithoutTheSecret(t *testing.T) {
	tests := []struct {
		name               string
		storedAt, otherKey string
	}{
		{"code under the temp key", "temp:k", "k"},
		{"code under the bare key", "k", "temp:k"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, tokenURL := newTokenEndpoint(t, http.StatusOK, tokenResponse)
			state := stateWith(t, map[string]any{tt.storedAt: storedCode()})
			if _, err := GetAuthResponse(t.Context(), state, toolConfig(tokenURL)); err != nil {
				t.Fatalf("GetAuthResponse() error = %v", err)
			}
			got, err := state.Get(tt.storedAt)
			if err != nil {
				t.Fatalf("state.Get(%q) error = %v", tt.storedAt, err)
			}
			if diff := cmp.Diff(exchangedCredential(""), got, ignoreExpiresAt); diff != "" {
				t.Errorf("stored credential mismatch (-want +got):\n%s", diff)
			}
			if _, err := state.Get(tt.otherKey); !errors.Is(err, session.ErrStateKeyNotExist) {
				t.Errorf("state.Get(%q) error = %v, want nothing written there", tt.otherKey, err)
			}
		})
	}
}

func TestGetAuthResponseNothingStored(t *testing.T) {
	stub, tokenURL := newTokenEndpoint(t, http.StatusOK, tokenResponse)
	got, err := GetAuthResponse(t.Context(), stateWith(t, nil), toolConfig(tokenURL))
	if got != nil || err != nil {
		t.Errorf("GetAuthResponse() = %v, %v; want nil, nil", got, err)
	}
	if calls, _ := stub.seen(); calls != 0 {
		t.Errorf("token endpoint got %d requests, want 0", calls)
	}
}

func TestGetAuthResponseNoExchangeNeeded(t *testing.T) {
	tests := []struct {
		name   string
		cfg    func(tokenURL string) *toolauth.Config
		stored toolauth.Credential
		want   *toolauth.Credential
	}{
		{
			name: "a stored token comes back with the configured client",
			cfg:  toolConfig,
			stored: toolauth.Credential{
				AuthType: toolauth.CredentialTypeOAuth2,
				OAuth2:   &toolauth.OAuth2Auth{ClientID: "app-123", AccessToken: "at-0"},
			},
			want: &toolauth.Credential{
				AuthType: toolauth.CredentialTypeOAuth2,
				OAuth2:   &toolauth.OAuth2Auth{ClientID: "app-123", ClientSecret: "s3cret", AccessToken: "at-0"},
			},
		},
		{
			name: "an API key comes back as stored",
			cfg: func(string) *toolauth.Config {
				return &toolauth.Config{
					AuthScheme:        apiKeyScheme,
					RawAuthCredential: &toolauth.Credential{AuthType: toolauth.CredentialTypeAPIKey},
					CredentialKey:     "k",
				}
			},
			stored: toolauth.Credential{AuthType: toolauth.CredentialTypeAPIKey, APIKey: "user-key"},
			want:   &toolauth.Credential{AuthType: toolauth.CredentialTypeAPIKey, APIKey: "user-key"},
		},
		{
			// adk-python redeems codes only for OAuth2 and OpenID Connect schemes.
			name: "a code under a custom scheme comes back as stored",
			cfg: func(tokenURL string) *toolauth.Config {
				return &toolauth.Config{
					AuthScheme:    toolauth.Scheme{Type: "gcpAuthProviderScheme", Flows: codeFlow(tokenURL).Flows},
					CredentialKey: "k",
				}
			},
			stored: storedCode(),
			want: func() *toolauth.Credential {
				c := storedCode()
				return &c
			}(),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			stub, tokenURL := newTokenEndpoint(t, http.StatusOK, tokenResponse)
			state := stateWith(t, map[string]any{"temp:k": tt.stored})
			got, err := GetAuthResponse(t.Context(), state, tt.cfg(tokenURL))
			if err != nil {
				t.Fatalf("GetAuthResponse() error = %v", err)
			}
			if diff := cmp.Diff(tt.want, got); diff != "" {
				t.Errorf("GetAuthResponse() mismatch (-want +got):\n%s", diff)
			}
			if calls, _ := stub.seen(); calls != 0 {
				t.Errorf("token endpoint got %d requests, want 0", calls)
			}
		})
	}
}

func TestGetAuthResponseFailedExchangeKeepsTheCode(t *testing.T) {
	_, tokenURL := newTokenEndpoint(t, http.StatusBadRequest, `{"error":"invalid_grant"}`)
	state := stateWith(t, map[string]any{"temp:k": storedCode()})
	if _, err := GetAuthResponse(t.Context(), state, toolConfig(tokenURL)); err == nil {
		t.Fatalf("GetAuthResponse() error = nil, want an error")
	}
	got, err := state.Get("temp:k")
	if err != nil {
		t.Fatalf("state.Get(temp key) error = %v", err)
	}
	if diff := cmp.Diff(storedCode(), got); diff != "" {
		t.Errorf("stored code changed (-want +got):\n%s", diff)
	}
}

// The tool's config decides how its secret is sent, whatever the answer names.
// An empty method is client_secret_basic, the secret in the header.
func TestGetAuthResponseUsesTheConfiguredAuthMethod(t *testing.T) {
	tests := []struct {
		name             string
		configured       string
		answered         string
		wantSecretInBody bool
	}{
		{"configured client_secret_post beats the answer's basic", "client_secret_post", "client_secret_basic", true},
		{"an empty config means basic, whatever the answer says", "", "client_secret_post", false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			stub, tokenURL := newTokenEndpoint(t, http.StatusOK, tokenResponse)
			cfg := toolConfig(tokenURL)
			cfg.RawAuthCredential.OAuth2.TokenEndpointAuthMethod = tt.configured
			stored := storedCode()
			stored.OAuth2.TokenEndpointAuthMethod = tt.answered
			state := stateWith(t, map[string]any{"temp:k": stored})
			if _, err := GetAuthResponse(t.Context(), state, cfg); err != nil {
				t.Fatalf("GetAuthResponse() error = %v", err)
			}
			_, form := stub.seen()
			inBody := form.Get("client_secret") == "s3cret"
			inHeader := strings.HasPrefix(stub.seenAuthHeader(), "Basic ")
			if inBody != tt.wantSecretInBody || inHeader == tt.wantSecretInBody {
				t.Errorf("secret in body = %t, in Basic header = %t; want in body = %t, in header = %t",
					inBody, inHeader, tt.wantSecretInBody, !tt.wantSecretInBody)
			}
		})
	}
}
