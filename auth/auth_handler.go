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

	"golang.org/x/oauth2"
)

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
func (h *AuthHandler) GenerateAuthRequest() *AuthConfig {
	// For non-OAuth schemes, return a copy as-is
	if h.authConfig.AuthScheme == nil {
		return h.authConfig.Copy()
	}

	schemeType := h.authConfig.AuthScheme.GetType()
	if schemeType != SecuritySchemeTypeOAuth2 && schemeType != SecuritySchemeTypeOpenIDConnect {
		return h.authConfig.Copy()
	}

	// If auth_uri already exists in exchanged credential, return as-is
	if h.authConfig.ExchangedAuthCredential != nil &&
		h.authConfig.ExchangedAuthCredential.OAuth2 != nil &&
		h.authConfig.ExchangedAuthCredential.OAuth2.AuthURI != "" {
		return h.authConfig.Copy()
	}

	// Check if raw_auth_credential exists with client credentials
	if h.authConfig.RawAuthCredential == nil ||
		h.authConfig.RawAuthCredential.OAuth2 == nil ||
		h.authConfig.RawAuthCredential.OAuth2.ClientID == "" {
		return h.authConfig.Copy()
	}

	// If auth_uri already in raw credential, copy to exchanged
	if h.authConfig.RawAuthCredential.OAuth2.AuthURI != "" {
		return &AuthConfig{
			AuthScheme:              h.authConfig.AuthScheme,
			RawAuthCredential:       h.authConfig.RawAuthCredential,
			ExchangedAuthCredential: h.authConfig.RawAuthCredential.Copy(),
			CredentialKey:           h.authConfig.CredentialKey,
		}
	}

	// Generate new auth URI
	exchangedCred := h.generateAuthURI()
	if exchangedCred == nil {
		return h.authConfig.Copy()
	}

	return &AuthConfig{
		AuthScheme:              h.authConfig.AuthScheme,
		RawAuthCredential:       h.authConfig.RawAuthCredential,
		ExchangedAuthCredential: exchangedCred,
		CredentialKey:           h.authConfig.CredentialKey,
	}
}

// generateAuthURI generates the OAuth authorization URI.
func (h *AuthHandler) generateAuthURI() *AuthCredential {
	oauth2Scheme, ok := h.authConfig.AuthScheme.(*OAuth2Scheme)
	if !ok || oauth2Scheme.Flows == nil {
		return nil
	}

	cred := h.authConfig.RawAuthCredential
	if cred.OAuth2 == nil {
		return nil
	}

	// Get authorization URL and scopes from flows
	var authURL string
	var scopes []string

	if oauth2Scheme.Flows.AuthorizationCode != nil {
		authURL = oauth2Scheme.Flows.AuthorizationCode.AuthorizationURL
		scopes = scopeKeys(oauth2Scheme.Flows.AuthorizationCode.Scopes)
	} else if oauth2Scheme.Flows.Implicit != nil {
		authURL = oauth2Scheme.Flows.Implicit.AuthorizationURL
		scopes = scopeKeys(oauth2Scheme.Flows.Implicit.Scopes)
	}

	if authURL == "" {
		return nil
	}

	// Create oauth2 config
	config := &oauth2.Config{
		ClientID:     cred.OAuth2.ClientID,
		ClientSecret: cred.OAuth2.ClientSecret,
		Endpoint: oauth2.Endpoint{
			AuthURL: authURL,
		},
		RedirectURL: cred.OAuth2.RedirectURI,
		Scopes:      scopes,
	}

	// Generate random state
	state := generateRandomState()

	// Generate auth URL with offline access
	authURI := config.AuthCodeURL(state, oauth2.AccessTypeOffline)

	// Create exchanged credential with auth_uri
	exchanged := cred.Copy()
	exchanged.OAuth2.AuthURI = authURI
	exchanged.OAuth2.State = state

	return exchanged
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
func generateRandomState() string {
	b := make([]byte, 16)
	rand.Read(b)
	return hex.EncodeToString(b)
}
