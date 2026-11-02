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

package toolauth

import (
	"errors"
	"fmt"
)

// OAuthGrantType is an OAuth2 grant type, spelled as it goes on the wire in
// the grant_type parameter.
type OAuthGrantType string

// The OAuth2 grant types, named as in RFC 6749.
const (
	OAuthGrantTypeClientCredentials OAuthGrantType = "client_credentials"
	OAuthGrantTypeAuthorizationCode OAuthGrantType = "authorization_code"
	OAuthGrantTypeImplicit          OAuthGrantType = "implicit"
	OAuthGrantTypePassword          OAuthGrantType = "password"
)

// OAuthFlow is one OAuth2 flow of an OpenAPI security scheme. Which URLs it
// needs depends on the flow; see [OAuthFlows].
type OAuthFlow struct {
	AuthorizationURL string `json:"authorizationUrl,omitempty"`
	TokenURL         string `json:"tokenUrl,omitempty"`
	RefreshURL       string `json:"refreshUrl,omitempty"`
	// Scopes maps each scope to its description.
	Scopes map[string]string `json:"scopes"`
	// Extra holds JSON keys that match no field, under the names they arrived
	// with, such as OpenAPI x- extensions. It is written back after the fields.
	Extra map[string]any `json:"-"`
}

// UnmarshalJSON implements [encoding/json.Unmarshaler]. It reads each field
// under its exact OpenAPI name and keeps other keys in Extra. Which URLs are
// required depends on the flow, so [OAuthFlows.UnmarshalJSON] checks them.
func (f *OAuthFlow) UnmarshalJSON(data []byte) error {
	type plain OAuthFlow
	return decodeOpenAPIModel(data, (*plain)(f), &f.Extra, "OAuthFlow")
}

// MarshalJSON implements [encoding/json.Marshaler]. It writes nil Scopes as {},
// as adk-python does, and writes Extra after the fields.
func (f OAuthFlow) MarshalJSON() ([]byte, error) {
	type plain OAuthFlow
	p := plain(f)
	if p.Scopes == nil {
		p.Scopes = map[string]string{}
	}
	return encodeOpenAPIModel(p, f.Extra)
}

// OAuthFlows lists the OAuth2 flows a scheme supports. The implicit flow needs
// AuthorizationURL, the password and client credentials flows need TokenURL,
// and the authorization code flow needs both.
type OAuthFlows struct {
	Implicit          *OAuthFlow `json:"implicit,omitempty"`
	Password          *OAuthFlow `json:"password,omitempty"`
	ClientCredentials *OAuthFlow `json:"clientCredentials,omitempty"`
	AuthorizationCode *OAuthFlow `json:"authorizationCode,omitempty"`
	// Extra holds JSON keys that match no field. See [OAuthFlow.Extra].
	Extra map[string]any `json:"-"`
}

// UnmarshalJSON implements [encoding/json.Unmarshaler] the way
// [OAuthFlow.UnmarshalJSON] does, and rejects a flow without the URLs it
// needs. An empty URL counts as missing: Go cannot tell the two apart after
// decoding, and neither can be used.
func (f *OAuthFlows) UnmarshalJSON(data []byte) error {
	type plain OAuthFlows
	if err := decodeOpenAPIModel(data, (*plain)(f), &f.Extra, "OAuthFlows"); err != nil {
		return err
	}
	for _, c := range []struct {
		name                string
		flow                *OAuthFlow
		needAuth, needToken bool
	}{
		{"implicit", f.Implicit, true, false},
		{"password", f.Password, false, true},
		{"clientCredentials", f.ClientCredentials, false, true},
		{"authorizationCode", f.AuthorizationCode, true, true},
	} {
		switch {
		case c.flow == nil:
		case c.needAuth && c.flow.AuthorizationURL == "":
			return fmt.Errorf("toolauth: OAuthFlows: %s flow needs authorizationUrl", c.name)
		case c.needToken && c.flow.TokenURL == "":
			return fmt.Errorf("toolauth: OAuthFlows: %s flow needs tokenUrl", c.name)
		}
	}
	return nil
}

// MarshalJSON implements [encoding/json.Marshaler]. It writes Extra after the
// fields.
func (f OAuthFlows) MarshalJSON() ([]byte, error) {
	type plain OAuthFlows
	return encodeOpenAPIModel(plain(f), f.Extra)
}

// GrantType returns the grant type of the first flow that is set, checking
// client credentials, authorization code, implicit and password in that
// order, as adk-python's OAuthGrantType.from_flow does. It returns "" when no
// flow is set.
func (f *OAuthFlows) GrantType() OAuthGrantType {
	switch {
	case f.ClientCredentials != nil:
		return OAuthGrantTypeClientCredentials
	case f.AuthorizationCode != nil:
		return OAuthGrantTypeAuthorizationCode
	case f.Implicit != nil:
		return OAuthGrantTypeImplicit
	case f.Password != nil:
		return OAuthGrantTypePassword
	}
	return ""
}

// SchemeType is the type of a [Scheme]: one of the four OpenAPI
// security scheme types, or any other string for a custom scheme.
type SchemeType string

// The OpenAPI security scheme types.
const (
	SchemeTypeAPIKey        SchemeType = "apiKey"
	SchemeTypeHTTP          SchemeType = "http"
	SchemeTypeOAuth2        SchemeType = "oauth2"
	SchemeTypeOpenIDConnect SchemeType = "openIdConnect"
)

// Scheme says how a resource expects to be authenticated. It is an OpenAPI
// security scheme object as one flat struct, plus the two extensions
// adk-python adds: OpenID Connect endpoints given inline, and an issuer URL
// for OAuth2 discovery. Type decides which fields apply. A Type outside the
// four OpenAPI ones is a custom scheme, whose own fields live in Extra.
type Scheme struct {
	Type        SchemeType `json:"type"`
	Description string     `json:"description,omitempty"`

	// In and Name place an apiKey. In is query, header or cookie.
	In   string `json:"in,omitempty"`
	Name string `json:"name,omitempty"`

	// Scheme is the http scheme, such as basic or bearer. BearerFormat hints
	// at how a bearer token is formatted.
	Scheme       string `json:"scheme,omitempty"`
	BearerFormat string `json:"bearerFormat,omitempty"`

	// Flows are the oauth2 flows. IssuerURL, when set, lets the endpoints be
	// discovered, as adk-python's ExtendedOAuth2 does.
	Flows     *OAuthFlows `json:"flows,omitempty"`
	IssuerURL string      `json:"issuer_url,omitempty"`

	// OpenIDConnectURL is the openIdConnect discovery URL.
	OpenIDConnectURL string `json:"openIdConnectUrl,omitempty"`

	// The fields below give openIdConnect endpoints inline instead of a
	// discovery URL, as adk-python's OpenIdConnectWithConfig does. adk-python
	// writes them in snake_case.
	AuthorizationEndpoint             string   `json:"authorization_endpoint,omitempty"`
	TokenEndpoint                     string   `json:"token_endpoint,omitempty"`
	UserinfoEndpoint                  string   `json:"userinfo_endpoint,omitempty"`
	RevocationEndpoint                string   `json:"revocation_endpoint,omitempty"`
	TokenEndpointAuthMethodsSupported []string `json:"token_endpoint_auth_methods_supported,omitzero"`
	GrantTypesSupported               []string `json:"grant_types_supported,omitzero"`
	Scopes                            []string `json:"scopes,omitzero"`

	// Extra holds JSON keys that match no field. See [OAuthFlow.Extra].
	Extra map[string]any `json:"-"`
}

// UnmarshalJSON implements [encoding/json.Unmarshaler]. It reads each field
// under its exact name and keeps other keys in Extra. It does not check that
// the fields fit Type: adk-python reads a scheme that fits no OpenAPI type as
// a custom scheme and writes it back unchanged, and this does the same. Flows
// are the exception: a flow without the URLs it needs is rejected, as
// [OAuthFlows.UnmarshalJSON] says, where adk-python reads the scheme as a
// custom one. A missing Type is inferred from the fields, as adk-python's
// models infer it. An object with no type and no field to infer one from is
// rejected, where adk-python reads it as an http bearer scheme only because
// fastapi's HTTPBearer has a default for every field.
func (s *Scheme) UnmarshalJSON(data []byte) error {
	type plain Scheme
	if err := decodeOpenAPIModel(data, (*plain)(s), &s.Extra, "Scheme"); err != nil {
		return err
	}
	if s.Type == "" {
		switch {
		case s.In != "" && s.Name != "":
			s.Type = SchemeTypeAPIKey
		case s.Flows != nil:
			s.Type = SchemeTypeOAuth2
		case s.OpenIDConnectURL != "", s.AuthorizationEndpoint != "" && s.TokenEndpoint != "":
			s.Type = SchemeTypeOpenIDConnect
		case s.Scheme != "":
			s.Type = SchemeTypeHTTP
		default:
			return errors.New("toolauth: Scheme: missing type, and no field says which")
		}
	}
	return nil
}

// MarshalJSON implements [encoding/json.Marshaler]. It writes Extra after the
// fields.
func (s Scheme) MarshalJSON() ([]byte, error) {
	type plain Scheme
	return encodeOpenAPIModel(plain(s), s.Extra)
}
