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
