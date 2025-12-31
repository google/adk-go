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
	"encoding/json"
)

// TokenToSchemeCredential creates an API key auth scheme and credential.
// This is a helper function similar to Python ADK's token_to_scheme_credential.
//
// Parameters:
//   - schemeType: The scheme type ("apikey" or "http")
//   - location: Where the token is sent ("header", "query", or "cookie")
//   - name: The name of the header/query parameter
//   - token: The actual token/API key value
//
// Returns the auth scheme and credential pair.
func TokenToSchemeCredential(schemeType, location, name, token string) (AuthScheme, *AuthCredential) {
	switch schemeType {
	case "apikey":
		var in APIKeyIn
		switch location {
		case "query":
			in = APIKeyInQuery
		case "cookie":
			in = APIKeyInCookie
		default:
			in = APIKeyInHeader
		}
		scheme := &APIKeyScheme{
			In:   in,
			Name: name,
		}
		cred := &AuthCredential{
			AuthType: AuthCredentialTypeAPIKey,
			APIKey:   token,
		}
		return scheme, cred
	case "http":
		scheme := &HTTPScheme{
			Scheme: "bearer",
		}
		cred := &AuthCredential{
			AuthType: AuthCredentialTypeHTTP,
			HTTP: &HTTPAuth{
				Scheme: "bearer",
				Credentials: &HTTPCredentials{
					Token: token,
				},
			},
		}
		return scheme, cred
	default:
		// Default to API key in header
		scheme := &APIKeyScheme{
			In:   APIKeyInHeader,
			Name: name,
		}
		cred := &AuthCredential{
			AuthType: AuthCredentialTypeAPIKey,
			APIKey:   token,
		}
		return scheme, cred
	}
}

// BearerTokenCredential creates an HTTP Bearer token auth scheme and credential.
func BearerTokenCredential(token string) (AuthScheme, *AuthCredential) {
	scheme := &HTTPScheme{
		Scheme:       "bearer",
		BearerFormat: "JWT",
	}
	cred := &AuthCredential{
		AuthType: AuthCredentialTypeHTTP,
		HTTP: &HTTPAuth{
			Scheme: "bearer",
			Credentials: &HTTPCredentials{
				Token: token,
			},
		},
	}
	return scheme, cred
}

// OAuth2ClientCredentials creates an OAuth2 client credentials auth scheme and credential.
func OAuth2ClientCredentials(clientID, clientSecret, tokenURL string, scopes map[string]string) (AuthScheme, *AuthCredential) {
	scheme := &OAuth2Scheme{
		Flows: &OAuthFlows{
			ClientCredentials: &OAuthFlowClientCredentials{
				TokenURL: tokenURL,
				Scopes:   scopes,
			},
		},
	}
	cred := &AuthCredential{
		AuthType: AuthCredentialTypeOAuth2,
		OAuth2: &OAuth2Auth{
			ClientID:     clientID,
			ClientSecret: clientSecret,
		},
	}
	return scheme, cred
}

// OAuth2AuthorizationCode creates an OAuth2 authorization code auth scheme and credential.
func OAuth2AuthorizationCode(clientID, clientSecret, authURL, tokenURL string, scopes map[string]string) (AuthScheme, *AuthCredential) {
	scheme := &OAuth2Scheme{
		Flows: &OAuthFlows{
			AuthorizationCode: &OAuthFlowAuthorizationCode{
				AuthorizationURL: authURL,
				TokenURL:         tokenURL,
				Scopes:           scopes,
			},
		},
	}
	cred := &AuthCredential{
		AuthType: AuthCredentialTypeOAuth2,
		OAuth2: &OAuth2Auth{
			ClientID:     clientID,
			ClientSecret: clientSecret,
		},
	}
	return scheme, cred
}

// ServiceAccountCredentials creates a service account auth scheme and credential.
func ServiceAccountCredentials(credentialJSON []byte, scopes []string) (AuthScheme, *AuthCredential) {
	scheme := &HTTPScheme{
		Scheme:       "bearer",
		BearerFormat: "JWT",
	}
	cred := &AuthCredential{
		AuthType: AuthCredentialTypeServiceAccount,
		ServiceAccount: &ServiceAccount{
			Scopes: scopes,
		},
	}
	// Parse the JSON if provided
	if len(credentialJSON) > 0 {
		var saCred ServiceAccountCredential
		if err := json.Unmarshal(credentialJSON, &saCred); err == nil {
			cred.ServiceAccount.ServiceAccountCredential = &saCred
		}
	}
	return scheme, cred
}

// DefaultCredentials creates a credential using Application Default Credentials.
func DefaultCredentials(scopes []string) (AuthScheme, *AuthCredential) {
	scheme := &HTTPScheme{
		Scheme:       "bearer",
		BearerFormat: "JWT",
	}
	cred := &AuthCredential{
		AuthType: AuthCredentialTypeServiceAccount,
		ServiceAccount: &ServiceAccount{
			Scopes:               scopes,
			UseDefaultCredential: true,
		},
	}
	return scheme, cred
}
