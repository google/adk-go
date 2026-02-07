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

// Package auth provides authentication types and utilities for ADK tools.
// It follows OpenAPI 3.0 Security Scheme specifications.
package auth

// SecuritySchemeType defines the type of security scheme.
// See: https://swagger.io/specification/#security-scheme-object
type SecuritySchemeType string

const (
	// SecuritySchemeTypeAPIKey for API key authentication.
	SecuritySchemeTypeAPIKey SecuritySchemeType = "apiKey"
	// SecuritySchemeTypeHTTP for HTTP authentication (Basic, Bearer, etc).
	SecuritySchemeTypeHTTP SecuritySchemeType = "http"
	// SecuritySchemeTypeOAuth2 for OAuth 2.0 authentication.
	SecuritySchemeTypeOAuth2 SecuritySchemeType = "oauth2"
	// SecuritySchemeTypeOpenIDConnect for OpenID Connect authentication.
	SecuritySchemeTypeOpenIDConnect SecuritySchemeType = "openIdConnect"
)

// APIKeyIn defines where the API key is located.
type APIKeyIn string

const (
	// APIKeyInQuery for API key in query parameter.
	APIKeyInQuery APIKeyIn = "query"
	// APIKeyInHeader for API key in HTTP header.
	APIKeyInHeader APIKeyIn = "header"
	// APIKeyInCookie for API key in cookie.
	APIKeyInCookie APIKeyIn = "cookie"
)

// AuthScheme is the interface for all security schemes.
type AuthScheme interface {
	// GetType returns the security scheme type.
	GetType() SecuritySchemeType
}

// APIKeyScheme represents API Key authentication.
// See: https://swagger.io/docs/specification/authentication/api-keys/
type APIKeyScheme struct {
	In          APIKeyIn `json:"in"`
	Name        string   `json:"name"`
	Description string   `json:"description,omitempty"`
}

// GetType implements AuthScheme.
func (s *APIKeyScheme) GetType() SecuritySchemeType {
	return SecuritySchemeTypeAPIKey
}

// HTTPScheme represents HTTP authentication (Basic, Bearer, etc).
// See: https://swagger.io/docs/specification/authentication/basic-authentication/
type HTTPScheme struct {
	Scheme       string `json:"scheme"`                 // "basic", "bearer", "digest", etc.
	BearerFormat string `json:"bearerFormat,omitempty"` // e.g., "JWT"
	Description  string `json:"description,omitempty"`
}

// GetType implements AuthScheme.
func (s *HTTPScheme) GetType() SecuritySchemeType {
	return SecuritySchemeTypeHTTP
}

// OAuth2Scheme represents OAuth 2.0 authentication.
// See: https://swagger.io/docs/specification/authentication/oauth2/
type OAuth2Scheme struct {
	Flows       *OAuthFlows `json:"flows"`
	Description string      `json:"description,omitempty"`
}

// GetType implements AuthScheme.
func (s *OAuth2Scheme) GetType() SecuritySchemeType {
	return SecuritySchemeTypeOAuth2
}

// OAuthFlows contains OAuth2 flow configurations.
type OAuthFlows struct {
	Implicit          *OAuthFlowImplicit          `json:"implicit,omitempty"`
	Password          *OAuthFlowPassword          `json:"password,omitempty"`
	ClientCredentials *OAuthFlowClientCredentials `json:"clientCredentials,omitempty"`
	AuthorizationCode *OAuthFlowAuthorizationCode `json:"authorizationCode,omitempty"`
}

// OAuthFlowImplicit represents the OAuth2 Implicit flow.
type OAuthFlowImplicit struct {
	AuthorizationURL string            `json:"authorizationUrl"`
	RefreshURL       string            `json:"refreshUrl,omitempty"`
	Scopes           map[string]string `json:"scopes"`
}

// OAuthFlowPassword represents the OAuth2 Resource Owner Password flow.
type OAuthFlowPassword struct {
	TokenURL   string            `json:"tokenUrl"`
	RefreshURL string            `json:"refreshUrl,omitempty"`
	Scopes     map[string]string `json:"scopes"`
}

// OAuthFlowClientCredentials represents the OAuth2 Client Credentials flow.
type OAuthFlowClientCredentials struct {
	TokenURL   string            `json:"tokenUrl"`
	RefreshURL string            `json:"refreshUrl,omitempty"`
	Scopes     map[string]string `json:"scopes"`
}

// OAuthFlowAuthorizationCode represents the OAuth2 Authorization Code flow.
type OAuthFlowAuthorizationCode struct {
	AuthorizationURL string            `json:"authorizationUrl"`
	TokenURL         string            `json:"tokenUrl"`
	RefreshURL       string            `json:"refreshUrl,omitempty"`
	Scopes           map[string]string `json:"scopes"`
}

// OpenIDConnectScheme represents OpenID Connect authentication.
// This is an extended version that includes flattened OIDC configuration,
// similar to Python ADK's OpenIdConnectWithConfig.
type OpenIDConnectScheme struct {
	// OpenIDConnectURL is the standard OIDC discovery URL.
	OpenIDConnectURL string `json:"openIdConnectUrl,omitempty"`
	// Flattened OIDC configuration (for when discovery is not available).
	AuthorizationEndpoint string   `json:"authorizationEndpoint,omitempty"`
	TokenEndpoint         string   `json:"tokenEndpoint,omitempty"`
	UserInfoEndpoint      string   `json:"userinfoEndpoint,omitempty"`
	RevocationEndpoint    string   `json:"revocationEndpoint,omitempty"`
	GrantTypesSupported   []string `json:"grantTypesSupported,omitempty"`
	Scopes                []string `json:"scopes,omitempty"`
	Description           string   `json:"description,omitempty"`
}

// GetType implements AuthScheme.
func (s *OpenIDConnectScheme) GetType() SecuritySchemeType {
	return SecuritySchemeTypeOpenIDConnect
}
