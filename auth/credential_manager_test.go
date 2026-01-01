// Copyright 2025 Google LLC
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

package auth

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"
)

type failingCredentialService struct {
	loadErr error
}

func (f *failingCredentialService) LoadCredential(context.Context, *AuthConfig) (*AuthCredential, error) {
	return nil, f.loadErr
}

func (f *failingCredentialService) SaveCredential(context.Context, *AuthConfig) error {
	return nil
}

type stubRefresher struct {
	shouldRefresh bool
	err           error
	refreshed     *AuthCredential
}

func (s *stubRefresher) IsRefreshNeeded(*AuthCredential, AuthScheme) bool {
	return s.shouldRefresh
}

func (s *stubRefresher) Refresh(context.Context, *AuthCredential, AuthScheme) (*AuthCredential, error) {
	if s.err != nil {
		return nil, s.err
	}
	return s.refreshed, nil
}

type stubExchanger struct {
	calls  int
	result *ExchangeResult
	err    error
}

func (s *stubExchanger) Exchange(ctx context.Context, cred *AuthCredential, scheme AuthScheme) (*ExchangeResult, error) {
	s.calls++
	if s.err != nil {
		return nil, s.err
	}
	return s.result, nil
}

type stubCredentialService struct {
	loadResp *AuthCredential
	loadErr  error
	saved    []*AuthConfig
	saveErr  error
}

func (s *stubCredentialService) LoadCredential(ctx context.Context, cfg *AuthConfig) (*AuthCredential, error) {
	if s.loadErr != nil {
		return nil, s.loadErr
	}
	return s.loadResp, nil
}

func (s *stubCredentialService) SaveCredential(ctx context.Context, cfg *AuthConfig) error {
	if s.saveErr != nil {
		return s.saveErr
	}
	s.saved = append(s.saved, cfg.Copy())
	return nil
}

func TestCredentialManager_GetAuthCredential_LoadCredentialError(t *testing.T) {
	cfg := &AuthConfig{
		AuthScheme: &OAuth2Scheme{
			Flows: &OAuthFlows{
				ClientCredentials: &OAuthFlowClientCredentials{
					TokenURL: "https://example.com/token",
				},
			},
		},
		RawAuthCredential: &AuthCredential{
			AuthType: AuthCredentialTypeOAuth2,
			OAuth2: &OAuth2Auth{
				ClientID:     "client-id",
				ClientSecret: "client-secret",
				AccessToken:  "existing-token",
			},
		},
	}

	manager := NewCredentialManager(cfg)

	svc := &failingCredentialService{loadErr: errors.New("database offline")}

	if _, err := manager.GetAuthCredential(context.Background(), nil, svc); err == nil || !strings.Contains(err.Error(), "failed to load credential") {
		t.Fatalf("GetAuthCredential() error = %v, want load credential error", err)
	}
}

func TestCredentialManager_GetAuthCredential_RefreshError(t *testing.T) {
	cfg := &AuthConfig{
		AuthScheme: &OAuth2Scheme{
			Flows: &OAuthFlows{
				AuthorizationCode: &OAuthFlowAuthorizationCode{
					AuthorizationURL: "https://example.com/auth",
					TokenURL:         "https://example.com/token",
				},
			},
		},
		RawAuthCredential: &AuthCredential{
			AuthType: AuthCredentialTypeOAuth2,
			OAuth2: &OAuth2Auth{
				ClientID:     "client-id",
				ClientSecret: "client-secret",
			},
		},
		ExchangedAuthCredential: &AuthCredential{
			AuthType: AuthCredentialTypeOAuth2,
			OAuth2: &OAuth2Auth{
				ClientID:     "client-id",
				ClientSecret: "client-secret",
				AccessToken:  "expired-token",
				RefreshToken: "refresh",
				ExpiresAt:    time.Now().Add(-time.Minute).Unix(),
			},
		},
	}

	manager := NewCredentialManager(cfg)
	manager.refresherRegistry.Register(AuthCredentialTypeOAuth2, &stubRefresher{
		shouldRefresh: true,
		err:           errors.New("refresh failed"),
	})

	if _, err := manager.GetAuthCredential(context.Background(), nil); err == nil || !strings.Contains(err.Error(), "failed to refresh credential") {
		t.Fatalf("GetAuthCredential() error = %v, want refresh error", err)
	}
}

func TestCredentialManager_GetAuthCredential_ReadyAPIKey(t *testing.T) {
	cfg := &AuthConfig{
		RawAuthCredential: &AuthCredential{
			AuthType: AuthCredentialTypeAPIKey,
			APIKey:   "key-123",
		},
	}

	manager := NewCredentialManager(cfg)
	cred, err := manager.GetAuthCredential(context.Background(), nil)
	if err != nil {
		t.Fatalf("GetAuthCredential() unexpected error: %v", err)
	}
	if cred != cfg.RawAuthCredential {
		t.Fatalf("got %v, want raw credential", cred)
	}
}

func TestCredentialManager_GetAuthCredential_ClientCredentialsFlow(t *testing.T) {
	cfg := &AuthConfig{
		CredentialKey: "adk_client",
		AuthScheme: &OAuth2Scheme{
			Flows: &OAuthFlows{
				ClientCredentials: &OAuthFlowClientCredentials{
					TokenURL: "https://example.com/token",
				},
			},
		},
		RawAuthCredential: &AuthCredential{
			AuthType: AuthCredentialTypeOAuth2,
			OAuth2: &OAuth2Auth{
				ClientID:     "client",
				ClientSecret: "secret",
			},
		},
	}

	manager := NewCredentialManager(cfg)
	exchanger := &stubExchanger{
		result: &ExchangeResult{
			Credential: &AuthCredential{
				AuthType: AuthCredentialTypeOAuth2,
				OAuth2: &OAuth2Auth{
					AccessToken: "token-123",
				},
			},
			WasExchanged: true,
		},
	}
	manager.exchangerRegistry = NewExchangerRegistry()
	manager.exchangerRegistry.Register(AuthCredentialTypeOAuth2, exchanger)

	svc := &stubCredentialService{}
	cred, err := manager.GetAuthCredential(context.Background(), nil, svc)
	if err != nil {
		t.Fatalf("GetAuthCredential() unexpected error: %v", err)
	}
	if exchanger.calls != 1 {
		t.Fatalf("Exchange called %d times, want 1", exchanger.calls)
	}
	if cred.OAuth2.AccessToken != "token-123" {
		t.Fatalf("AccessToken = %q, want %q", cred.OAuth2.AccessToken, "token-123")
	}
	if len(svc.saved) != 1 {
		t.Fatalf("expected credential to be saved, got %d saves", len(svc.saved))
	}
}

func TestCredentialManager_GetAuthCredential_AuthCodeExchange(t *testing.T) {
	cfg := &AuthConfig{
		CredentialKey: "adk_auth_code",
		AuthScheme: &OAuth2Scheme{
			Flows: &OAuthFlows{
				AuthorizationCode: &OAuthFlowAuthorizationCode{
					AuthorizationURL: "https://example.com/auth",
					TokenURL:         "https://example.com/token",
				},
			},
		},
		RawAuthCredential: &AuthCredential{
			AuthType: AuthCredentialTypeOAuth2,
			OAuth2: &OAuth2Auth{
				ClientID:     "client",
				ClientSecret: "secret",
			},
		},
	}

	manager := NewCredentialManager(cfg)
	exchanger := &stubExchanger{
		result: &ExchangeResult{
			Credential: &AuthCredential{
				AuthType: AuthCredentialTypeOAuth2,
				OAuth2: &OAuth2Auth{
					AccessToken: "new-token",
				},
			},
			WasExchanged: true,
		},
	}
	manager.exchangerRegistry = NewExchangerRegistry()
	manager.exchangerRegistry.Register(AuthCredentialTypeOAuth2, exchanger)

	stateGetter := func(key string) interface{} {
		if key == "temp:"+cfg.CredentialKey {
			return &AuthCredential{
				AuthType: AuthCredentialTypeOAuth2,
				OAuth2: &OAuth2Auth{
					ClientID:     "client",
					ClientSecret: "secret",
					AuthCode:     "code-123",
				},
			}
		}
		return nil
	}

	cred, err := manager.GetAuthCredential(context.Background(), stateGetter)
	if err != nil {
		t.Fatalf("GetAuthCredential() unexpected error: %v", err)
	}
	if exchanger.calls != 1 {
		t.Fatalf("Exchange called %d times, want 1", exchanger.calls)
	}
	if cred.OAuth2.AccessToken != "new-token" {
		t.Fatalf("AccessToken = %q, want %q", cred.OAuth2.AccessToken, "new-token")
	}
}

func TestCredentialManager_GetAuthCredential_LoadsFromCredentialService(t *testing.T) {
	cfg := &AuthConfig{
		AuthScheme: &OAuth2Scheme{
			Flows: &OAuthFlows{
				ClientCredentials: &OAuthFlowClientCredentials{
					TokenURL: "https://example.com/token",
				},
			},
		},
		RawAuthCredential: &AuthCredential{
			AuthType: AuthCredentialTypeOAuth2,
		},
	}

	manager := NewCredentialManager(cfg)
	svc := &stubCredentialService{
		loadResp: &AuthCredential{
			AuthType: AuthCredentialTypeOAuth2,
			OAuth2: &OAuth2Auth{
				AccessToken: "svc-token",
			},
		},
	}

	cred, err := manager.GetAuthCredential(context.Background(), nil, svc)
	if err != nil {
		t.Fatalf("GetAuthCredential() unexpected error: %v", err)
	}
	if cred.OAuth2.AccessToken != "svc-token" {
		t.Fatalf("AccessToken = %q, want %q", cred.OAuth2.AccessToken, "svc-token")
	}
	if len(svc.saved) != 0 {
		t.Fatalf("expected no saves, got %d", len(svc.saved))
	}
}

func TestCredentialManager_GetAuthCredential_LoadsFromAuthResponse(t *testing.T) {
	cfg := &AuthConfig{
		CredentialKey: "adk_temp",
		AuthScheme: &OAuth2Scheme{
			Flows: &OAuthFlows{
				AuthorizationCode: &OAuthFlowAuthorizationCode{
					AuthorizationURL: "https://example.com/auth",
					TokenURL:         "https://example.com/token",
				},
			},
		},
	}

	manager := NewCredentialManager(cfg)
	stateCred := &AuthCredential{
		AuthType: AuthCredentialTypeOAuth2,
		OAuth2: &OAuth2Auth{
			AccessToken: "state-token",
		},
	}
	stateGetter := func(key string) interface{} {
		if key == "temp:"+cfg.CredentialKey {
			return stateCred
		}
		return nil
	}

	cred, err := manager.GetAuthCredential(context.Background(), stateGetter)
	if err != nil {
		t.Fatalf("GetAuthCredential() unexpected error: %v", err)
	}
	if cred.OAuth2.AccessToken != "state-token" {
		t.Fatalf("AccessToken = %q, want %q", cred.OAuth2.AccessToken, "state-token")
	}
}

func TestCredentialManager_GetAuthCredential_ReturnsNilWhenAuthorizationNeeded(t *testing.T) {
	cfg := &AuthConfig{
		AuthScheme: &OAuth2Scheme{
			Flows: &OAuthFlows{
				AuthorizationCode: &OAuthFlowAuthorizationCode{
					AuthorizationURL: "https://example.com/auth",
					TokenURL:         "https://example.com/token",
				},
			},
		},
	}

	manager := NewCredentialManager(cfg)
	cred, err := manager.GetAuthCredential(context.Background(), nil)
	if err != nil {
		t.Fatalf("GetAuthCredential() unexpected error: %v", err)
	}
	if cred != nil {
		t.Fatalf("expected nil credential when user auth is required, got %v", cred)
	}
}

func TestCredentialManager_GetAuthCredential_RefreshesCredential(t *testing.T) {
	cfg := &AuthConfig{
		CredentialKey: "adk_refresh",
		AuthScheme:    &OAuth2Scheme{},
		ExchangedAuthCredential: &AuthCredential{
			AuthType: AuthCredentialTypeOAuth2,
			OAuth2: &OAuth2Auth{
				AccessToken:  "expired",
				RefreshToken: "refresh",
				ExpiresAt:    time.Now().Add(-time.Minute).Unix(),
			},
		},
	}

	manager := NewCredentialManager(cfg)
	manager.refresherRegistry = NewRefresherRegistry()
	manager.refresherRegistry.Register(AuthCredentialTypeOAuth2, &stubRefresher{
		shouldRefresh: true,
		refreshed: &AuthCredential{
			AuthType: AuthCredentialTypeOAuth2,
			OAuth2: &OAuth2Auth{
				AccessToken: "fresh",
			},
		},
	})

	svc := &stubCredentialService{}
	cred, err := manager.GetAuthCredential(context.Background(), nil, svc)
	if err != nil {
		t.Fatalf("GetAuthCredential() unexpected error: %v", err)
	}
	if cred.OAuth2.AccessToken != "fresh" {
		t.Fatalf("AccessToken = %q, want %q", cred.OAuth2.AccessToken, "fresh")
	}
	if len(svc.saved) != 1 {
		t.Fatalf("expected refreshed credential to be saved, got %d", len(svc.saved))
	}
}
