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
	"time"

	"golang.org/x/oauth2"
)

// OAuth2Exchanger exchanges OAuth2 credentials.
// It handles both authorization code and client credentials flows.
type OAuth2Exchanger struct{}

// NewOAuth2Exchanger creates a new OAuth2 exchanger.
func NewOAuth2Exchanger() *OAuth2Exchanger {
	return &OAuth2Exchanger{}
}

// Exchange exchanges OAuth2 credentials for access tokens.
func (e *OAuth2Exchanger) Exchange(ctx context.Context, cred *AuthCredential, scheme AuthScheme) (*ExchangeResult, error) {
	if scheme == nil {
		return nil, fmt.Errorf("auth scheme is required for OAuth2 credential exchange")
	}

	// If already has access token, no exchange needed
	if cred.OAuth2 != nil && cred.OAuth2.AccessToken != "" {
		return &ExchangeResult{Credential: cred, WasExchanged: false}, nil
	}

	// Determine grant type from scheme
	grantType := e.determineGrantType(scheme)

	switch grantType {
	case GrantTypeClientCredentials:
		return e.exchangeClientCredentials(ctx, cred, scheme)
	case GrantTypeAuthorizationCode:
		return e.exchangeAuthorizationCode(ctx, cred, scheme)
	default:
		// Unknown grant type, return unchanged
		return &ExchangeResult{Credential: cred, WasExchanged: false}, nil
	}
}

// GrantType represents OAuth2 grant types.
type GrantType string

const (
	GrantTypeAuthorizationCode GrantType = "authorization_code"
	GrantTypeClientCredentials GrantType = "client_credentials"
)

func (e *OAuth2Exchanger) determineGrantType(scheme AuthScheme) GrantType {
	oauth2Scheme, ok := scheme.(*OAuth2Scheme)
	if !ok {
		return ""
	}

	if oauth2Scheme.Flows == nil {
		return ""
	}

	if oauth2Scheme.Flows.ClientCredentials != nil {
		return GrantTypeClientCredentials
	}
	if oauth2Scheme.Flows.AuthorizationCode != nil {
		return GrantTypeAuthorizationCode
	}

	return ""
}

func (e *OAuth2Exchanger) exchangeClientCredentials(ctx context.Context, cred *AuthCredential, scheme AuthScheme) (*ExchangeResult, error) {
	oauth2Scheme := scheme.(*OAuth2Scheme)
	if oauth2Scheme.Flows == nil || oauth2Scheme.Flows.ClientCredentials == nil {
		return nil, fmt.Errorf("client credentials flow not configured in scheme")
	}

	if cred.OAuth2 == nil {
		return nil, fmt.Errorf("oauth2 credentials required")
	}

	config := &oauth2.Config{
		ClientID:     cred.OAuth2.ClientID,
		ClientSecret: cred.OAuth2.ClientSecret,
		Endpoint: oauth2.Endpoint{
			TokenURL: oauth2Scheme.Flows.ClientCredentials.TokenURL,
		},
		Scopes: scopeKeys(oauth2Scheme.Flows.ClientCredentials.Scopes),
	}

	// Use client credentials grant
	token, err := config.Exchange(ctx, "", oauth2.SetAuthURLParam("grant_type", "client_credentials"))
	if err != nil {
		return nil, fmt.Errorf("failed to exchange client credentials: %w", err)
	}

	// Update credential with tokens
	newCred := cred.Copy()
	newCred.OAuth2.AccessToken = token.AccessToken
	newCred.OAuth2.RefreshToken = token.RefreshToken
	if !token.Expiry.IsZero() {
		newCred.OAuth2.ExpiresAt = token.Expiry.Unix()
		newCred.OAuth2.ExpiresIn = int64(time.Until(token.Expiry).Seconds())
	}

	return &ExchangeResult{Credential: newCred, WasExchanged: true}, nil
}

func (e *OAuth2Exchanger) exchangeAuthorizationCode(ctx context.Context, cred *AuthCredential, scheme AuthScheme) (*ExchangeResult, error) {
	oauth2Scheme := scheme.(*OAuth2Scheme)
	if oauth2Scheme.Flows == nil || oauth2Scheme.Flows.AuthorizationCode == nil {
		return nil, fmt.Errorf("authorization code flow not configured in scheme")
	}

	if cred.OAuth2 == nil {
		return nil, fmt.Errorf("oauth2 credentials required")
	}

	// Need auth_code to exchange
	if cred.OAuth2.AuthCode == "" {
		return &ExchangeResult{Credential: cred, WasExchanged: false}, nil
	}

	config := &oauth2.Config{
		ClientID:     cred.OAuth2.ClientID,
		ClientSecret: cred.OAuth2.ClientSecret,
		Endpoint: oauth2.Endpoint{
			AuthURL:  oauth2Scheme.Flows.AuthorizationCode.AuthorizationURL,
			TokenURL: oauth2Scheme.Flows.AuthorizationCode.TokenURL,
		},
		RedirectURL: cred.OAuth2.RedirectURI,
		Scopes:      scopeKeys(oauth2Scheme.Flows.AuthorizationCode.Scopes),
	}

	token, err := config.Exchange(ctx, cred.OAuth2.AuthCode)
	if err != nil {
		return nil, fmt.Errorf("failed to exchange authorization code: %w", err)
	}

	// Update credential with tokens
	newCred := cred.Copy()
	newCred.OAuth2.AccessToken = token.AccessToken
	newCred.OAuth2.RefreshToken = token.RefreshToken
	newCred.OAuth2.AuthCode = "" // Clear the code after use
	if !token.Expiry.IsZero() {
		newCred.OAuth2.ExpiresAt = token.Expiry.Unix()
		newCred.OAuth2.ExpiresIn = int64(time.Until(token.Expiry).Seconds())
	}

	return &ExchangeResult{Credential: newCred, WasExchanged: true}, nil
}

func scopeKeys(scopes map[string]string) []string {
	if scopes == nil {
		return nil
	}
	keys := make([]string, 0, len(scopes))
	for k := range scopes {
		keys = append(keys, k)
	}
	return keys
}

// OAuth2Refresher refreshes OAuth2 access tokens using refresh tokens.
type OAuth2Refresher struct{}

// NewOAuth2Refresher creates a new OAuth2 refresher.
func NewOAuth2Refresher() *OAuth2Refresher {
	return &OAuth2Refresher{}
}

// IsRefreshNeeded checks if the OAuth2 token is expired or about to expire.
func (r *OAuth2Refresher) IsRefreshNeeded(cred *AuthCredential, scheme AuthScheme) bool {
	if cred.OAuth2 == nil {
		return false
	}

	// No expiry info, assume valid
	if cred.OAuth2.ExpiresAt == 0 {
		return false
	}

	// Check if expired (with 60 second buffer)
	expiresAt := time.Unix(cred.OAuth2.ExpiresAt, 0)
	return time.Now().Add(60 * time.Second).After(expiresAt)
}

// Refresh refreshes the OAuth2 access token using the refresh token.
func (r *OAuth2Refresher) Refresh(ctx context.Context, cred *AuthCredential, scheme AuthScheme) (*AuthCredential, error) {
	if cred.OAuth2 == nil || cred.OAuth2.RefreshToken == "" {
		// No refresh token, return original
		return cred, nil
	}

	oauth2Scheme, ok := scheme.(*OAuth2Scheme)
	if !ok || oauth2Scheme.Flows == nil {
		return cred, nil
	}

	// Get token URL from appropriate flow
	var tokenURL string
	if oauth2Scheme.Flows.AuthorizationCode != nil {
		tokenURL = oauth2Scheme.Flows.AuthorizationCode.TokenURL
	} else if oauth2Scheme.Flows.ClientCredentials != nil {
		tokenURL = oauth2Scheme.Flows.ClientCredentials.TokenURL
	}

	if tokenURL == "" {
		return cred, nil
	}

	config := &oauth2.Config{
		ClientID:     cred.OAuth2.ClientID,
		ClientSecret: cred.OAuth2.ClientSecret,
		Endpoint: oauth2.Endpoint{
			TokenURL: tokenURL,
		},
	}

	// Create token source from existing token
	oldToken := &oauth2.Token{
		AccessToken:  cred.OAuth2.AccessToken,
		RefreshToken: cred.OAuth2.RefreshToken,
		Expiry:       time.Unix(cred.OAuth2.ExpiresAt, 0),
	}

	tokenSource := config.TokenSource(ctx, oldToken)
	newToken, err := tokenSource.Token()
	if err != nil {
		return cred, err
	}

	// Update credential with new tokens
	newCred := cred.Copy()
	newCred.OAuth2.AccessToken = newToken.AccessToken
	if newToken.RefreshToken != "" {
		newCred.OAuth2.RefreshToken = newToken.RefreshToken
	}
	if !newToken.Expiry.IsZero() {
		newCred.OAuth2.ExpiresAt = newToken.Expiry.Unix()
		newCred.OAuth2.ExpiresIn = int64(time.Until(newToken.Expiry).Seconds())
	}

	return newCred, nil
}
