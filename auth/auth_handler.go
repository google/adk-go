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
	"crypto/rand"
	"encoding/hex"
	"fmt"

	"golang.org/x/oauth2"
)

var randRead = rand.Read

// AuthHandler handles the OAuth flow orchestration including auth URI generation
// and response parsing.
type AuthHandler struct {
	authConfig *AuthConfig
}

// NewAuthHandler creates a new AuthHandler.
func NewAuthHandler(config *AuthConfig) *AuthHandler {
	return &AuthHandler{
		authConfig: config,
	}
}

// GenerateAuthRequest generates an AuthConfig with the auth_uri populated for
// OAuth2/OIDC flows. The client uses this to redirect users for authorization.
func (h *AuthHandler) GenerateAuthRequest() (*AuthConfig, error) {
	// For non-OAuth schemes, return a copy as-is
	if h.authConfig.AuthScheme == nil {
		return h.authConfig.Copy(), nil
	}

	schemeType := h.authConfig.AuthScheme.GetType()
	if schemeType != SecuritySchemeTypeOAuth2 && schemeType != SecuritySchemeTypeOpenIDConnect {
		return h.authConfig.Copy(), nil
	}

	// If auth_uri already exists in exchanged credential, return as-is
	if h.authConfig.ExchangedAuthCredential != nil &&
		h.authConfig.ExchangedAuthCredential.OAuth2 != nil &&
		h.authConfig.ExchangedAuthCredential.OAuth2.AuthURI != "" {
		return h.authConfig.Copy(), nil
	}

	// Check if raw_auth_credential exists with client credentials
	if h.authConfig.RawAuthCredential == nil ||
		h.authConfig.RawAuthCredential.OAuth2 == nil ||
		h.authConfig.RawAuthCredential.OAuth2.ClientID == "" {
		return h.authConfig.Copy(), nil
	}

	// If auth_uri already in raw credential, copy to exchanged
	if h.authConfig.RawAuthCredential.OAuth2.AuthURI != "" {
		return &AuthConfig{
			AuthScheme:              h.authConfig.AuthScheme,
			RawAuthCredential:       h.authConfig.RawAuthCredential,
			ExchangedAuthCredential: h.authConfig.RawAuthCredential.Copy(),
			CredentialKey:           h.authConfig.CredentialKey,
		}, nil
	}

	// Generate new auth URI
	exchangedCred, err := h.generateAuthURI()
	if err != nil {
		return nil, err
	}
	if exchangedCred == nil {
		return h.authConfig.Copy(), nil
	}

	return &AuthConfig{
		AuthScheme:              h.authConfig.AuthScheme,
		RawAuthCredential:       h.authConfig.RawAuthCredential,
		ExchangedAuthCredential: exchangedCred,
		CredentialKey:           h.authConfig.CredentialKey,
	}, nil
}

// generateAuthURI generates the OAuth authorization URI.
func (h *AuthHandler) generateAuthURI() (*AuthCredential, error) {
	cred := h.authConfig.RawAuthCredential
	if cred == nil || cred.OAuth2 == nil {
		return nil, nil
	}

	authURL, scopes := authorizationMetadata(h.authConfig.AuthScheme)
	if authURL == "" {
		return nil, nil
	}

	config := &oauth2.Config{
		ClientID:     cred.OAuth2.ClientID,
		ClientSecret: cred.OAuth2.ClientSecret,
		Endpoint: oauth2.Endpoint{
			AuthURL: authURL,
		},
		RedirectURL: cred.OAuth2.RedirectURI,
		Scopes:      scopes,
	}
	opts := []oauth2.AuthCodeOption{oauth2.AccessTypeOffline}
	if cred.OAuth2.Audience != "" {
		opts = append(opts, oauth2.SetAuthURLParam("audience", cred.OAuth2.Audience))
	}

	state, err := generateRandomState()
	if err != nil {
		return nil, err
	}
	authURI := config.AuthCodeURL(state, opts...)

	exchanged := cred.Copy()
	exchanged.OAuth2.AuthURI = authURI
	exchanged.OAuth2.State = state

	return exchanged, nil
}

// GetAuthResponse retrieves the auth response from session state.
func (h *AuthHandler) GetAuthResponse(stateGetter func(key string) interface{}) *AuthCredential {
	key := "temp:" + h.authConfig.CredentialKey
	if val := stateGetter(key); val != nil {
		if cred, ok := val.(*AuthCredential); ok {
			return cred
		}
	}
	return nil
}

// generateRandomState generates a random state string for OAuth CSRF protection.
func generateRandomState() (string, error) {
	b := make([]byte, 16)
	if _, err := randRead(b); err != nil {
		return "", fmt.Errorf("failed to generate random OAuth state: %w", err)
	}
	return hex.EncodeToString(b), nil
}

// authorizationMetadata returns the authorization endpoint and scopes for OAuth2/OIDC schemes.
func authorizationMetadata(scheme AuthScheme) (string, []string) {
	switch v := scheme.(type) {
	case *OAuth2Scheme:
		if v == nil || v.Flows == nil {
			return "", nil
		}
		if v.Flows.AuthorizationCode != nil {
			return v.Flows.AuthorizationCode.AuthorizationURL, scopeKeys(v.Flows.AuthorizationCode.Scopes)
		}
		if v.Flows.Implicit != nil {
			return v.Flows.Implicit.AuthorizationURL, scopeKeys(v.Flows.Implicit.Scopes)
		}
	case *OpenIDConnectScheme:
		if v == nil {
			return "", nil
		}
		if len(v.Scopes) == 0 {
			return v.AuthorizationEndpoint, []string{"openid"}
		}
		return v.AuthorizationEndpoint, append([]string{}, v.Scopes...)
	}
	return "", nil
}
