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

import "errors"

// CredentialType is the kind of credential a [Credential] holds.
type CredentialType string

const (
	// CredentialTypeAPIKey is an API key:
	// https://swagger.io/docs/specification/v3_0/authentication/api-keys/
	CredentialTypeAPIKey CredentialType = "apiKey"
	// CredentialTypeHTTP is a credential for an HTTP auth scheme, such as
	// basic or bearer:
	// https://www.iana.org/assignments/http-authschemes/http-authschemes.xhtml
	CredentialTypeHTTP CredentialType = "http"
	// CredentialTypeOAuth2 is an OAuth2 credential:
	// https://swagger.io/docs/specification/v3_0/authentication/oauth2/
	CredentialTypeOAuth2 CredentialType = "oauth2"
	// CredentialTypeOpenIDConnect is an OpenID Connect credential:
	// https://swagger.io/docs/specification/v3_0/authentication/openid-connect-discovery/
	CredentialTypeOpenIDConnect CredentialType = "openIdConnect"
	// CredentialTypeServiceAccount is a Google service account credential:
	// https://cloud.google.com/iam/docs/service-account-creds
	CredentialTypeServiceAccount CredentialType = "serviceAccount"
)

// HTTPCredentials is the secret for an HTTP auth scheme: a username and
// password, or a token.
type HTTPCredentials struct {
	Username string `json:"username,omitempty"`
	Password string `json:"password,omitempty"`
	Token    string `json:"token,omitempty"`
	// Extra holds JSON keys that match no field. See [Credential.Extra].
	Extra map[string]any `json:"-"`
}

// UnmarshalJSON implements [encoding/json.Unmarshaler] the way
// [Credential.UnmarshalJSON] does.
func (c *HTTPCredentials) UnmarshalJSON(data []byte) error {
	type plain HTTPCredentials
	return decodeModel(data, (*plain)(c), &c.Extra, "HTTPCredentials")
}

// MarshalJSON implements [encoding/json.Marshaler]. It writes Extra after the
// fields.
func (c HTTPCredentials) MarshalJSON() ([]byte, error) {
	type plain HTTPCredentials
	return encodeModel(plain(c), c.Extra)
}

// HTTPAuth is the credential and metadata for an HTTP auth scheme.
type HTTPAuth struct {
	// Scheme is the RFC 7235 scheme name for the Authorization header, such as
	// "basic" or "bearer". It should be one registered with IANA.
	Scheme string `json:"scheme"`
	// Credentials is a value, so it is written as {} even when empty, the way
	// adk-python writes it.
	Credentials HTTPCredentials `json:"credentials"`
	// AdditionalHeaders are set on the request alongside the credential.
	AdditionalHeaders map[string]string `json:"additionalHeaders,omitzero"`
	// Extra holds JSON keys that match no field. See [Credential.Extra].
	Extra map[string]any `json:"-"`
}

// UnmarshalJSON implements [encoding/json.Unmarshaler] the way
// [Credential.UnmarshalJSON] does. Scheme and Credentials are required.
func (h *HTTPAuth) UnmarshalJSON(data []byte) error {
	type plain HTTPAuth
	return decodeModel(data, (*plain)(h), &h.Extra, "HTTPAuth", "scheme", "credentials")
}

// MarshalJSON implements [encoding/json.Marshaler]. It writes Extra after the
// fields.
func (h HTTPAuth) MarshalJSON() ([]byte, error) {
	type plain HTTPAuth
	return encodeModel(plain(h), h.Extra)
}

// OAuth2Auth is an OAuth2 or OpenID Connect credential, together with the
// state of the flow that obtains it.
type OAuth2Auth struct {
	ClientID     string `json:"clientId,omitempty"`
	ClientSecret string `json:"clientSecret,omitempty"`
	// AuthURI is the authorization URI. ADK or the tool can generate it with
	// the state in it, so the client can verify the state.
	AuthURI string `json:"authUri,omitempty"`
	// Nonce binds the user's session to the authorization request. It is
	// generated at the start of the flow, usually stored with the session, and
	// checked by the backend.
	Nonce string `json:"nonce,omitempty"`
	State string `json:"state,omitempty"`
	// RedirectURI is set when ADK or the tool chooses the redirect URI instead
	// of the client.
	RedirectURI     string `json:"redirectUri,omitempty"`
	AuthResponseURI string `json:"authResponseUri,omitempty"`
	AuthCode        string `json:"authCode,omitempty"`
	AccessToken     string `json:"accessToken,omitempty"`
	RefreshToken    string `json:"refreshToken,omitempty"`
	IDToken         string `json:"idToken,omitempty"`
	// ExpiresAt is when the access token expires, in Unix seconds.
	ExpiresAt int64 `json:"expiresAt,omitempty"`
	// ExpiresIn is the access token lifetime the token endpoint reported, in
	// seconds.
	ExpiresIn           int64  `json:"expiresIn,omitempty"`
	Audience            string `json:"audience,omitempty"`
	Prompt              string `json:"prompt,omitempty"`
	CodeVerifier        string `json:"codeVerifier,omitempty"`
	CodeChallengeMethod string `json:"codeChallengeMethod,omitempty"`
	// TokenEndpointAuthMethod is how the client authenticates to the token
	// endpoint: client_secret_basic, client_secret_post, client_secret_jwt or
	// private_key_jwt. Empty means client_secret_basic. A public client has no
	// ClientSecret; adk-python sends none to the token endpoint for it, but
	// does not accept none here.
	TokenEndpointAuthMethod string `json:"tokenEndpointAuthMethod,omitempty"`
	// Extra holds JSON keys that match no field, such as a provider-specific
	// field of a token response. See [Credential.Extra].
	Extra map[string]any `json:"-"`
}

// UnmarshalJSON implements [encoding/json.Unmarshaler] the way
// [Credential.UnmarshalJSON] does. It rejects a TokenEndpointAuthMethod
// other than the four listed on the field.
func (o *OAuth2Auth) UnmarshalJSON(data []byte) error {
	type plain OAuth2Auth
	if err := decodeModel(data, (*plain)(o), &o.Extra, "OAuth2Auth"); err != nil {
		return err
	}
	switch o.TokenEndpointAuthMethod {
	case "", "client_secret_basic", "client_secret_post", "client_secret_jwt", "private_key_jwt":
		return nil
	}
	return errors.New("toolauth: OAuth2Auth: unsupported tokenEndpointAuthMethod")
}

// MarshalJSON implements [encoding/json.Marshaler]. It writes
// client_secret_basic for an empty TokenEndpointAuthMethod, which is what
// adk-python sends, and writes Extra after the fields. A JSON null read into
// TokenEndpointAuthMethod leaves it empty, so it comes back out as
// client_secret_basic, where adk-python would write nothing.
func (o OAuth2Auth) MarshalJSON() ([]byte, error) {
	// plain has no methods, so json.Marshal does not call back into this one.
	type plain OAuth2Auth
	p := plain(o)
	if p.TokenEndpointAuthMethod == "" {
		p.TokenEndpointAuthMethod = "client_secret_basic"
	}
	return encodeModel(p, o.Extra)
}

// ServiceAccountCredential holds the fields of a Google service account JSON
// key. Every field is written even when empty, as adk-python does, because
// Python requires them all.
type ServiceAccountCredential struct {
	// Type is "service_account" in a key file.
	Type                    string `json:"type"`
	ProjectID               string `json:"projectId"`
	PrivateKeyID            string `json:"privateKeyId"`
	PrivateKey              string `json:"privateKey"`
	ClientEmail             string `json:"clientEmail"`
	ClientID                string `json:"clientId"`
	AuthURI                 string `json:"authUri"`
	TokenURI                string `json:"tokenUri"`
	AuthProviderX509CertURL string `json:"authProviderX509CertUrl"`
	ClientX509CertURL       string `json:"clientX509CertUrl"`
	UniverseDomain          string `json:"universeDomain"`
	// Extra holds JSON keys that match no field. See [Credential.Extra].
	Extra map[string]any `json:"-"`
}

// UnmarshalJSON implements [encoding/json.Unmarshaler] the way
// [Credential.UnmarshalJSON] does, so it also reads a key file, whose
// names are snake_case. Every field but Type is required.
func (s *ServiceAccountCredential) UnmarshalJSON(data []byte) error {
	type plain ServiceAccountCredential
	return decodeModel(data, (*plain)(s), &s.Extra, "ServiceAccountCredential",
		"projectId", "privateKeyId", "privateKey", "clientEmail", "clientId", "authUri",
		"tokenUri", "authProviderX509CertUrl", "clientX509CertUrl", "universeDomain")
}

// MarshalJSON implements [encoding/json.Marshaler]. It writes Extra after the
// fields.
func (s ServiceAccountCredential) MarshalJSON() ([]byte, error) {
	type plain ServiceAccountCredential
	return encodeModel(plain(s), s.Extra)
}

// ServiceAccount configures a Google service account credential, from a
// key or from Application Default Credentials.
type ServiceAccount struct {
	// ServiceAccountCredential is the key. It is required unless
	// UseDefaultCredential is set.
	ServiceAccountCredential *ServiceAccountCredential `json:"serviceAccountCredential,omitempty"`
	// Scopes are the OAuth2 scopes to request. With UseDefaultCredential and no
	// scopes, the cloud-platform scope is used.
	Scopes []string `json:"scopes,omitzero"`
	// UseDefaultCredential uses Application Default Credentials instead of a
	// key.
	UseDefaultCredential bool `json:"useDefaultCredential"`
	// UseIDToken asks for an ID token instead of an access token, which
	// service-to-service calls to Cloud Run and Cloud Functions need. Audience
	// is required with it.
	UseIDToken bool `json:"useIdToken"`
	// Audience is the ID token's audience, usually the receiving service's URL,
	// such as https://my-service-xyz.run.app.
	Audience string `json:"audience,omitempty"`
	// Extra holds JSON keys that match no field. See [Credential.Extra].
	Extra map[string]any `json:"-"`
}

// UnmarshalJSON implements [encoding/json.Unmarshaler] the way
// [Credential.UnmarshalJSON] does. It rejects a configuration with neither
// a ServiceAccountCredential nor UseDefaultCredential, and one with UseIDToken
// but no Audience.
func (s *ServiceAccount) UnmarshalJSON(data []byte) error {
	type plain ServiceAccount
	if err := decodeModel(data, (*plain)(s), &s.Extra, "ServiceAccount"); err != nil {
		return err
	}
	if !s.UseDefaultCredential && s.ServiceAccountCredential == nil {
		return errors.New("toolauth: ServiceAccount: serviceAccountCredential is required unless useDefaultCredential is set")
	}
	if s.UseIDToken && s.Audience == "" {
		return errors.New("toolauth: ServiceAccount: audience is required with useIdToken")
	}
	return nil
}

// MarshalJSON implements [encoding/json.Marshaler]. It writes Extra after the
// fields.
func (s ServiceAccount) MarshalJSON() ([]byte, error) {
	type plain ServiceAccount
	return encodeModel(plain(s), s.Extra)
}

// Credential is a credential in adk-python's wire format. AuthType says
// what kind of credential it is, and the field for that kind holds it. An
// OpenID Connect credential is held in OAuth2.
//
// An API key, an HTTP bearer token, and an OAuth2 client that still needs the
// user's consent look like this:
//
//	toolauth.Credential{AuthType: toolauth.CredentialTypeAPIKey, APIKey: "1234"}
//
//	toolauth.Credential{
//		AuthType: toolauth.CredentialTypeHTTP,
//		HTTP: &toolauth.HTTPAuth{
//			Scheme:      "bearer",
//			Credentials: toolauth.HTTPCredentials{Token: "eyAkaknabna"},
//		},
//	}
//
//	toolauth.Credential{
//		AuthType: toolauth.CredentialTypeOAuth2,
//		OAuth2:   &toolauth.OAuth2Auth{ClientID: "1234", ClientSecret: "secret"},
//	}
//
// The JSON matches what adk-python writes with by_alias and exclude_none, with
// two exceptions. Optional strings and numbers are plain values, and Go cannot
// tell "" or 0 from unset, so an optional field set to "" or 0 is not written,
// where adk-python writes it. A nil list or map is not written and an empty
// one is, as in adk-python. [OAuth2Auth.MarshalJSON] describes the other
// exception.
type Credential struct {
	AuthType CredentialType `json:"authType"`
	// ResourceRef names a resource that holds the credential. adk-python
	// reserves it for future use.
	ResourceRef    string          `json:"resourceRef,omitempty"`
	APIKey         string          `json:"apiKey,omitempty"`
	HTTP           *HTTPAuth       `json:"http,omitempty"`
	ServiceAccount *ServiceAccount `json:"serviceAccount,omitempty"`
	OAuth2         *OAuth2Auth     `json:"oauth2,omitempty"`
	// Extra holds the JSON keys that match no field, under the names they
	// arrived with. They are written back after the fields, in sorted order, so
	// a credential passing through Go keeps keys that only adk-python, a client
	// or a provider knows. Numbers in it are [encoding/json.Number] after
	// decoding. A key that is a field's camelCase or snake_case name is not
	// written.
	Extra map[string]any `json:"-"`
}

// UnmarshalJSON implements [encoding/json.Unmarshaler]. It reads each field
// under its camelCase name or its snake_case Python name, preferring camelCase
// when an object has both, and keeps the other keys in Extra. Like adk-python,
// it rejects a missing or unknown AuthType and an invalid credential in any
// field. Its errors never quote the value that caused them.
//
// When an object has both names for a field, the snake_case value is dropped,
// as adk-python's model_validate_json drops it. adk-python's model_validate,
// which reads a client's response, keeps it as an unknown key and writes it
// back; it may be a secret meant for that field.
//
// JSON types must match the fields exactly. pydantic's lax mode also accepts
// "3600" for an integer and "true" for a bool, and this does not.
func (c *Credential) UnmarshalJSON(data []byte) error {
	type plain Credential
	if err := decodeModel(data, (*plain)(c), &c.Extra, "Credential", "authType"); err != nil {
		return err
	}
	switch c.AuthType {
	case CredentialTypeAPIKey, CredentialTypeHTTP, CredentialTypeOAuth2,
		CredentialTypeOpenIDConnect, CredentialTypeServiceAccount:
		return nil
	}
	return errors.New("toolauth: Credential: unknown authType")
}

// MarshalJSON implements [encoding/json.Marshaler]. It writes Extra after the
// fields.
func (c Credential) MarshalJSON() ([]byte, error) {
	type plain Credential
	return encodeModel(plain(c), c.Extra)
}
