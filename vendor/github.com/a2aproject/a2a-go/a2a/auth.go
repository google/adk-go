// Copyright 2025 The A2A Authors
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

package a2a

import (
	"encoding/gob"
	"encoding/json"
	"fmt"
)

// SecuritySchemeName is a string used to describe a security scheme in AgentCard.SecuritySchemes
// and reference it the AgentCard.Security requirements.
type SecuritySchemeName string

// SecuritySchemeScopes is a list of scopes a security credential must be covering.
type SecuritySchemeScopes []string

// NamedSecuritySchemes is a declaration of the security schemes available to authorize requests.
// The key is the scheme name. Follows the OpenAPI 3.0 Security Scheme Object.
type NamedSecuritySchemes map[SecuritySchemeName]SecurityScheme

func (s *NamedSecuritySchemes) UnmarshalJSON(b []byte) error {
	var schemes map[SecuritySchemeName]json.RawMessage
	if err := json.Unmarshal(b, &schemes); err != nil {
		return err
	}

	result := make(map[SecuritySchemeName]SecurityScheme, len(schemes))
	for k, v := range schemes {
		type typedScheme struct {
			Type string `json:"type"`
		}
		var ts typedScheme
		if err := json.Unmarshal(v, &ts); err != nil {
			return err
		}

		switch ts.Type {
		case "apiKey":
			var scheme APIKeySecurityScheme
			if err := json.Unmarshal(v, &scheme); err != nil {
				return err
			}
			result[k] = scheme
		case "http":
			var scheme HTTPAuthSecurityScheme
			if err := json.Unmarshal(v, &scheme); err != nil {
				return err
			}
			result[k] = scheme
		case "mutualTLS":
			var scheme MutualTLSSecurityScheme
			if err := json.Unmarshal(v, &scheme); err != nil {
				return err
			}
			result[k] = scheme
		case "oauth2":
			var scheme OAuth2SecurityScheme
			if err := json.Unmarshal(v, &scheme); err != nil {
				return err
			}
			result[k] = scheme
		case "openIdConnect":
			var scheme OpenIDConnectSecurityScheme
			if err := json.Unmarshal(v, &scheme); err != nil {
				return err
			}
			result[k] = scheme
		default:
			return fmt.Errorf("unknown security scheme type %s", ts.Type)
		}
	}

	*s = result
	return nil
}

// SecurityScheme is a sealed discriminated type union for supported security schemes.
type SecurityScheme interface {
	isSecurityScheme()
}

func (APIKeySecurityScheme) isSecurityScheme()        {}
func (HTTPAuthSecurityScheme) isSecurityScheme()      {}
func (OpenIDConnectSecurityScheme) isSecurityScheme() {}
func (MutualTLSSecurityScheme) isSecurityScheme()     {}
func (OAuth2SecurityScheme) isSecurityScheme()        {}

func init() {
	gob.Register(APIKeySecurityScheme{})
	gob.Register(HTTPAuthSecurityScheme{})
	gob.Register(OpenIDConnectSecurityScheme{})
	gob.Register(MutualTLSSecurityScheme{})
	gob.Register(OAuth2SecurityScheme{})
}

// APIKeySecurityScheme defines a security scheme using an API key.
type APIKeySecurityScheme struct {
	// An optional description for the security scheme.
	Description string `json:"description,omitempty" yaml:"description,omitempty" mapstructure:"description,omitempty"`

	// The location of the API key.
	In APIKeySecuritySchemeIn `json:"in" yaml:"in" mapstructure:"in"`

	// The name of the header, query, or cookie parameter to be used.
	Name string `json:"name" yaml:"name" mapstructure:"name"`
}

func (s APIKeySecurityScheme) MarshalJSON() ([]byte, error) {
	type wrapped APIKeySecurityScheme
	type withType struct {
		Type string `json:"type"`
		wrapped
	}
	return json.Marshal(withType{Type: "apiKey", wrapped: wrapped(s)})
}

// APIKeySecuritySchemeIn defines a set of permitted values for the expected API key location in APIKeySecurityScheme.
type APIKeySecuritySchemeIn string

const (
	APIKeySecuritySchemeInCookie APIKeySecuritySchemeIn = "cookie"
	APIKeySecuritySchemeInHeader APIKeySecuritySchemeIn = "header"
	APIKeySecuritySchemeInQuery  APIKeySecuritySchemeIn = "query"
)

// AuthorizationCodeOAuthFlow defines configuration details for the OAuth 2.0 Authorization Code flow.
type AuthorizationCodeOAuthFlow struct {
	// AuthorizationURL is the authorization URL to be used for this flow.
	// This MUST be a URL and use TLS.
	AuthorizationURL string `json:"authorizationUrl" yaml:"authorizationUrl" mapstructure:"authorizationUrl"`

	// RefreshURL is an optional URL to be used for obtaining refresh tokens.
	// This MUST be a URL and use TLS.
	RefreshURL string `json:"refreshUrl,omitempty" yaml:"refreshUrl,omitempty" mapstructure:"refreshUrl,omitempty"`

	// Scopes are the available scopes for the OAuth2 security scheme. A map between the scope
	// name and a short description for it.
	Scopes map[string]string `json:"scopes" yaml:"scopes" mapstructure:"scopes"`

	// TokenURL is the URL to be used for this flow. This MUST be a URL and use TLS.
	TokenURL string `json:"tokenUrl" yaml:"tokenUrl" mapstructure:"tokenUrl"`
}

// ClientCredentialsOAuthFlow defines configuration details for the OAuth 2.0 Client Credentials flow.
type ClientCredentialsOAuthFlow struct {
	// RefreshURL is an optional URL to be used for obtaining refresh tokens. This MUST be a URL.
	RefreshURL string `json:"refreshUrl,omitempty" yaml:"refreshUrl,omitempty" mapstructure:"refreshUrl,omitempty"`

	// Scopes are the available scopes for the OAuth2 security scheme. A map between the scope
	// name and a short description for it.
	Scopes map[string]string `json:"scopes" yaml:"scopes" mapstructure:"scopes"`

	// TokenURL is the token URL to be used for this flow. This MUST be a URL.
	TokenURL string `json:"tokenUrl" yaml:"tokenUrl" mapstructure:"tokenUrl"`
}

// HTTPAuthSecurityScheme defines a security scheme using HTTP authentication.
type HTTPAuthSecurityScheme struct {
	// BearerFormat is an optional hint to the client to identify how the bearer token is formatted (e.g.,
	// "JWT"). This is primarily for documentation purposes.
	BearerFormat string `json:"bearerFormat,omitempty" yaml:"bearerFormat,omitempty" mapstructure:"bearerFormat,omitempty"`

	// Description is an optional description for the security scheme.
	Description string `json:"description,omitempty" yaml:"description,omitempty" mapstructure:"description,omitempty"`

	// Scheme is the name of the HTTP Authentication scheme to be used in the Authorization
	// header, as defined in RFC7235 (e.g., "Bearer").
	// This value should be registered in the IANA Authentication Scheme registry.
	Scheme string `json:"scheme" yaml:"scheme" mapstructure:"scheme"`
}

func (s HTTPAuthSecurityScheme) MarshalJSON() ([]byte, error) {
	type wrapped HTTPAuthSecurityScheme
	type withType struct {
		Type string `json:"type"`
		wrapped
	}
	return json.Marshal(withType{Type: "http", wrapped: wrapped(s)})
}

// ImplicitOAuthFlow defines configuration details for the OAuth 2.0 Implicit flow.
type ImplicitOAuthFlow struct {
	// AuthorizationURL is the authorization URL to be used for this flow. This MUST be a URL.
	AuthorizationURL string `json:"authorizationUrl" yaml:"authorizationUrl" mapstructure:"authorizationUrl"`

	// RefreshURL is an optional URL to be used for obtaining refresh tokens. This MUST be a URL.
	RefreshURL string `json:"refreshUrl,omitempty" yaml:"refreshUrl,omitempty" mapstructure:"refreshUrl,omitempty"`

	// Scopes are the available scopes for the OAuth2 security scheme. A map between the scope
	// name and a short description for it.
	Scopes map[string]string `json:"scopes" yaml:"scopes" mapstructure:"scopes"`
}

// MutualTLSSecurityScheme defines a security scheme using mTLS authentication.
type MutualTLSSecurityScheme struct {
	// Description is an optional description for the security scheme.
	Description string `json:"description,omitempty" yaml:"description,omitempty" mapstructure:"description,omitempty"`
}

func (s MutualTLSSecurityScheme) MarshalJSON() ([]byte, error) {
	type wrapped MutualTLSSecurityScheme
	type withType struct {
		Type string `json:"type"`
		wrapped
	}
	return json.Marshal(withType{Type: "mutualTLS", wrapped: wrapped(s)})
}

// OAuth2SecurityScheme defines a security scheme using OAuth 2.0.
type OAuth2SecurityScheme struct {
	// Description is an optional description for the security scheme.
	Description string `json:"description,omitempty" yaml:"description,omitempty" mapstructure:"description,omitempty"`

	// Flows is an object containing configuration information for the supported OAuth 2.0 flows.
	Flows OAuthFlows `json:"flows" yaml:"flows" mapstructure:"flows"`

	// Oauth2MetadataURL is an optional URL to the oauth2 authorization server metadata
	// [RFC8414](https://datatracker.ietf.org/doc/html/rfc8414). TLS is required.
	Oauth2MetadataURL string `json:"oauth2MetadataUrl,omitempty" yaml:"oauth2MetadataUrl,omitempty" mapstructure:"oauth2MetadataUrl,omitempty"`
}

func (s OAuth2SecurityScheme) MarshalJSON() ([]byte, error) {
	type wrapped OAuth2SecurityScheme
	type withType struct {
		Type string `json:"type"`
		wrapped
	}
	return json.Marshal(withType{Type: "oauth2", wrapped: wrapped(s)})
}

// OAuthFlows defines the configuration for the supported OAuth 2.0 flows.
type OAuthFlows struct {
	// AuthorizationCode is a configuration for the OAuth Authorization Code flow.
	// Previously called accessCode in OpenAPI 2.0.
	AuthorizationCode *AuthorizationCodeOAuthFlow `json:"authorizationCode,omitempty" yaml:"authorizationCode,omitempty" mapstructure:"authorizationCode,omitempty"`

	// ClientCredentials is a configuration for the OAuth Client Credentials flow. Previously called
	// application in OpenAPI 2.0.
	ClientCredentials *ClientCredentialsOAuthFlow `json:"clientCredentials,omitempty" yaml:"clientCredentials,omitempty" mapstructure:"clientCredentials,omitempty"`

	// Implicit is a configuration for the OAuth Implicit flow.
	Implicit *ImplicitOAuthFlow `json:"implicit,omitempty" yaml:"implicit,omitempty" mapstructure:"implicit,omitempty"`

	// Password is a configuration for the OAuth Resource Owner Password flow.
	Password *PasswordOAuthFlow `json:"password,omitempty" yaml:"password,omitempty" mapstructure:"password,omitempty"`
}

// OpenIDConnectSecurityScheme defines a security scheme using OpenID Connect.
type OpenIDConnectSecurityScheme struct {
	// Description is an optional description for the security scheme.
	Description string `json:"description,omitempty" yaml:"description,omitempty" mapstructure:"description,omitempty"`

	// OpenIDConnectURL is the OpenID Connect Discovery URL for the OIDC provider's metadata.
	OpenIDConnectURL string `json:"openIdConnectUrl" yaml:"openIdConnectUrl" mapstructure:"openIdConnectUrl"`
}

func (s OpenIDConnectSecurityScheme) MarshalJSON() ([]byte, error) {
	type wrapped OpenIDConnectSecurityScheme
	type withType struct {
		Type string `json:"type"`
		wrapped
	}
	return json.Marshal(withType{Type: "openIdConnect", wrapped: wrapped(s)})
}

// PasswordOAuthFlow defines configuration details for the OAuth 2.0 Resource Owner Password flow.
type PasswordOAuthFlow struct {
	// RefreshURL is an optional URL to be used for obtaining refresh tokens. This MUST be a URL.
	RefreshURL string `json:"refreshUrl,omitempty" yaml:"refreshUrl,omitempty" mapstructure:"refreshUrl,omitempty"`

	// Scopes are еру available scopes for the OAuth2 security scheme. A map between the scope
	// name and a short description for it.
	Scopes map[string]string `json:"scopes" yaml:"scopes" mapstructure:"scopes"`

	// TokenURL is the token URL to be used for this flow. This MUST be a URL.
	TokenURL string `json:"tokenUrl" yaml:"tokenUrl" mapstructure:"tokenUrl"`
}
