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
	"fmt"
)

// CredentialManager orchestrates the complete lifecycle of authentication
// credentials, from initial loading to final preparation for use.
type CredentialManager struct {
	authConfig        *AuthConfig
	exchangerRegistry *ExchangerRegistry
	refresherRegistry *RefresherRegistry
}

// NewCredentialManager creates a new CredentialManager with default exchangers
// and refreshers registered.
func NewCredentialManager(cfg *AuthConfig) *CredentialManager {
	m := &CredentialManager{
		authConfig:        cfg,
		exchangerRegistry: NewExchangerRegistry(),
		refresherRegistry: NewRefresherRegistry(),
	}

	// Register default exchangers
	oauth2Exchanger := NewOAuth2Exchanger()
	m.exchangerRegistry.Register(AuthCredentialTypeOAuth2, oauth2Exchanger)
	m.exchangerRegistry.Register(AuthCredentialTypeOpenIDConnect, oauth2Exchanger)

	// Register default refreshers
	oauth2Refresher := NewOAuth2Refresher()
	m.refresherRegistry.Register(AuthCredentialTypeOAuth2, oauth2Refresher)
	m.refresherRegistry.Register(AuthCredentialTypeOpenIDConnect, oauth2Refresher)

	return m
}

// GetAuthCredential loads and prepares authentication credential through a
// structured workflow:
// 1. Validate credential configuration
// 2. Check if credential is already ready (no processing needed)
// 3. Try to load existing processed credential from CredentialService
// 4. If no existing credential, load from auth response (temp: prefix)
// 5. For client credentials flow, use raw credentials directly
// 6. Exchange credential if needed (e.g., auth code -> access token)
// 7. Refresh credential if expired
// 8. Save credential to CredentialService if it was modified
func (m *CredentialManager) GetAuthCredential(ctx context.Context, stateGetter func(key string) interface{}, credentialService ...CredentialService) (*AuthCredential, error) {
	// Step 1: Validate credential configuration
	if err := m.validate(); err != nil {
		return nil, err
	}

	// Step 2: Check if credential is already ready (no processing needed)
	if m.isCredentialReady() {
		return m.authConfig.RawAuthCredential, nil
	}

	// Step 3: Try to load existing processed credential
	credential := m.loadExistingCredential()

	// Step 3b: Try to load from credential service (persistent storage)
	var svc CredentialService
	if len(credentialService) > 0 && credentialService[0] != nil {
		svc = credentialService[0]
		if credential == nil {
			loaded, err := svc.LoadCredential(ctx, m.authConfig)
			if err != nil {
				return nil, fmt.Errorf("failed to load credential: %w", err)
			}
			if loaded != nil && loaded.OAuth2 != nil && loaded.OAuth2.AccessToken != "" {
				credential = loaded
			}
		}
	}

	// Step 4: If no existing credential, load from auth response (session state)
	wasFromAuthResponse := false
	if credential == nil && stateGetter != nil {
		credential = m.loadFromAuthResponse(stateGetter)
		if credential != nil {
			wasFromAuthResponse = true
		}
	}

	// Step 5: If still no credential available, check if client credentials
	if credential == nil {
		if m.isClientCredentialsFlow() {
			credential = m.authConfig.RawAuthCredential
		} else {
			// For authorization code flow, return nil to trigger user authorization
			return nil, nil
		}
	}

	// Step 6: Exchange credential if needed
	credential, wasExchanged, err := m.exchangeCredential(ctx, credential)
	if err != nil {
		return nil, err
	}

	// Step 7: Refresh credential if expired
	wasRefreshed := false
	if !wasExchanged {
		var err error
		credential, wasRefreshed, err = m.refreshCredential(ctx, credential)
		if err != nil {
			return nil, err
		}
	}

	// Step 8: Save credential if it was modified
	if wasFromAuthResponse || wasExchanged || wasRefreshed {
		m.authConfig.ExchangedAuthCredential = credential
		// Save to credential service for persistence across requests
		if svc != nil {
			if err := svc.SaveCredential(ctx, m.authConfig); err != nil {
				return nil, fmt.Errorf("failed to save credential: %w", err)
			}
		}
	}

	return credential, nil
}

// validate checks that the auth configuration is valid.
func (m *CredentialManager) validate() error {
	// For OAuth2/OIDC, raw_auth_credential is required
	if m.authConfig.AuthScheme != nil {
		schemeType := m.authConfig.AuthScheme.GetType()
		if schemeType == SecuritySchemeTypeOAuth2 || schemeType == SecuritySchemeTypeOpenIDConnect {
			if m.authConfig.RawAuthCredential == nil {
				return nil // Will need user auth
			}
		}
	}
	return nil
}

// isCredentialReady checks if credential is ready to use without further processing.
func (m *CredentialManager) isCredentialReady() bool {
	raw := m.authConfig.RawAuthCredential
	if raw == nil {
		return false
	}

	// Simple credentials that don't need exchange or refresh
	switch raw.AuthType {
	case AuthCredentialTypeAPIKey, AuthCredentialTypeHTTP:
		return true
	}

	return false
}

// loadExistingCredential loads credential from exchanged cache.
func (m *CredentialManager) loadExistingCredential() *AuthCredential {
	if m.authConfig.ExchangedAuthCredential != nil {
		return m.authConfig.ExchangedAuthCredential
	}
	return nil
}

// loadFromAuthResponse loads credential from session state (auth response).
func (m *CredentialManager) loadFromAuthResponse(stateGetter func(key string) interface{}) *AuthCredential {
	key := "temp:" + m.authConfig.CredentialKey
	if val := stateGetter(key); val != nil {
		if cred, ok := val.(*AuthCredential); ok {
			return cred
		}
	}
	return nil
}

// isClientCredentialsFlow checks if the auth scheme uses client credentials flow.
func (m *CredentialManager) isClientCredentialsFlow() bool {
	switch scheme := m.authConfig.AuthScheme.(type) {
	case *OAuth2Scheme:
		if scheme.Flows == nil {
			return false
		}
		return scheme.Flows.ClientCredentials != nil
	case *OpenIDConnectScheme:
		return grantSupported(scheme.GrantTypesSupported, "client_credentials")
	default:
		return false
	}
}

// exchangeCredential exchanges credential if needed.
func (m *CredentialManager) exchangeCredential(ctx context.Context, cred *AuthCredential) (*AuthCredential, bool, error) {
	ex := m.exchangerRegistry.Get(cred.AuthType)
	if ex == nil {
		return cred, false, nil
	}

	result, err := ex.Exchange(ctx, cred, m.authConfig.AuthScheme)
	if err != nil {
		return nil, false, err
	}

	return result.Credential, result.WasExchanged, nil
}

// refreshCredential refreshes credential if expired.
func (m *CredentialManager) refreshCredential(ctx context.Context, cred *AuthCredential) (*AuthCredential, bool, error) {
	ref := m.refresherRegistry.Get(cred.AuthType)
	if ref == nil {
		return cred, false, nil
	}

	if !ref.IsRefreshNeeded(cred, m.authConfig.AuthScheme) {
		return cred, false, nil
	}

	refreshed, err := ref.Refresh(ctx, cred, m.authConfig.AuthScheme)
	if err != nil {
		return cred, false, fmt.Errorf("failed to refresh credential: %w", err)
	}

	return refreshed, true, nil
}

// GetAuthConfig accessor
func (m *CredentialManager) GetAuthConfig() *AuthConfig {
	return m.authConfig
}
