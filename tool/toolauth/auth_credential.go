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
	// Username is the user name for the basic scheme.
	Username string `json:"username,omitempty"`
	// Password is the password for the basic scheme. [Config.WithoutSecrets]
	// removes it.
	Password string `json:"password,omitempty"`
	// Token is the token for the bearer scheme. [Config.WithoutSecrets]
	// removes it.
	Token string `json:"token,omitempty"`
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
//
// The developer sets the client's fields (ClientID, ClientSecret and,
// optionally, RedirectURI, Nonce, Audience, Prompt, CodeChallengeMethod and
// TokenEndpointAuthMethod) in a config's RawAuthCredential. The other fields
// change during a sign-in. When a tool asks the user for the credential, ADK
// sets AuthURI and State, and for PKCE CodeVerifier and CodeChallengeMethod,
// in the config it sends to the client. The client sets AuthResponseURI or
// AuthCode in its answer. ADK then redeems the code at the token endpoint and
// sets AccessToken, RefreshToken, IDToken, ExpiresAt and ExpiresIn from the
// response; a client credentials grant or a refresh sets the same five.
type OAuth2Auth struct {
	// ClientID is the OAuth2 client ID of the agent's app. When ADK stores
	// the client's answer or redeems its code, it uses the ClientID of the
	// tool's RawAuthCredential, so the client cannot choose which app redeems
	// the code.
	ClientID string `json:"clientId,omitempty"`
	// ClientSecret is the client secret of the agent's app. Empty means a
	// public client, which ADK signs in with PKCE. ADK never sends it to the
	// client ([Config.WithoutSecrets]) or writes it to session state; it takes
	// it from the tool's RawAuthCredential when it calls the token endpoint.
	ClientSecret string `json:"clientSecret,omitempty"`
	// AuthURI is the authorization URI, where the client sends the user to
	// sign in. ADK or the tool can generate it with the state in it, so the
	// client can verify the state. When the tool's RawAuthCredential has
	// none, ADK generates it, with State, in the ExchangedAuthCredential of
	// the config it sends in an adk_request_credential call.
	AuthURI string `json:"authUri,omitempty"`
	// Nonce binds the user's session to the authorization request. It is
	// generated at the start of the flow, usually stored with the session, and
	// checked by the backend. ADK does not generate it; when it is set, ADK
	// adds it to the authorization URI it generates.
	Nonce string `json:"nonce,omitempty"`
	// State is the OAuth2 state parameter. ADK sets a new random value each
	// time it generates AuthURI, and puts it in that URI. When the client
	// answers with AuthResponseURI or AuthCode and no access token, ADK
	// replaces the answer's State with the one it issued, and refuses a
	// redirect URL whose state parameter differs from a non-empty State.
	State string `json:"state,omitempty"`
	// RedirectURI is set when ADK or the tool chooses the redirect URI instead
	// of the client. ADK puts it in the authorization URI it generates and
	// sends it to the token endpoint with the code. When the client's answer
	// has none, ADK takes the one of the request it sent.
	RedirectURI string `json:"redirectUri,omitempty"`
	// AuthResponseURI is the full URL the provider redirected the user to
	// after sign-in, with the code and the state in its query. The client
	// sets it in its answer, and ADK redeems the code it carries. It stays
	// set in the stored credential after the code is redeemed.
	AuthResponseURI string `json:"authResponseUri,omitempty"`
	// AuthCode is the authorization code, which the client may set in its
	// answer instead of AuthResponseURI. ADK redeems the code of
	// AuthResponseURI when that URL carries one, and AuthCode otherwise. A
	// bare AuthCode carries no state, so ADK cannot check it against State.
	AuthCode string `json:"authCode,omitempty"`
	// AccessToken is the access token. ADK sets it from the token endpoint's
	// response when it redeems a code, uses the client credentials grant or
	// refreshes the token. A credential that has one is not exchanged again.
	AccessToken string `json:"accessToken,omitempty"`
	// RefreshToken is the refresh token of the token endpoint's response, set
	// with AccessToken. ADK uses it to get a new access token when the current
	// one expires.
	RefreshToken string `json:"refreshToken,omitempty"`
	// IDToken is the OpenID Connect ID token, the id_token member of the
	// token endpoint's response, set with AccessToken.
	IDToken string `json:"idToken,omitempty"`
	// ExpiresAt is when the access token expires, in Unix seconds. ADK sets it
	// with AccessToken, and leaves it zero when the response gives no
	// lifetime.
	ExpiresAt int64 `json:"expiresAt,omitempty"`
	// ExpiresIn is the access token lifetime the token endpoint reported, in
	// seconds. ADK sets it with AccessToken.
	ExpiresIn int64 `json:"expiresIn,omitempty"`
	// Audience is the audience parameter ADK adds to the authorization URI it
	// generates, for providers that need one.
	Audience string `json:"audience,omitempty"`
	// Prompt is the prompt parameter of the authorization URI ADK generates.
	// Empty means consent.
	Prompt string `json:"prompt,omitempty"`
	// CodeVerifier is the PKCE code verifier. When ADK generates AuthURI with
	// PKCE, it uses this one, or generates one when it is empty, and it sends
	// it to the token endpoint with the code. When the client's answer has
	// none, ADK takes the one of the request it sent.
	CodeVerifier string `json:"codeVerifier,omitempty"`
	// CodeChallengeMethod is the PKCE code challenge method. Only S256 is
	// supported. When it is empty, ADK uses PKCE with S256 for a client with
	// no ClientSecret, sets this field to S256, and uses no PKCE for a client
	// with a secret.
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
	Type string `json:"type"`
	// ProjectID is the project_id of the key file.
	ProjectID string `json:"projectId"`
	// PrivateKeyID is the private_key_id of the key file.
	// [Config.WithoutSecrets] removes it.
	PrivateKeyID string `json:"privateKeyId"`
	// PrivateKey is the private_key of the key file, in PEM.
	// [Config.WithoutSecrets] removes it.
	PrivateKey string `json:"privateKey"`
	// ClientEmail is the client_email of the key file, the service account's
	// email address.
	ClientEmail string `json:"clientEmail"`
	// ClientID is the client_id of the key file.
	ClientID string `json:"clientId"`
	// AuthURI is the auth_uri of the key file.
	AuthURI string `json:"authUri"`
	// TokenURI is the token_uri of the key file.
	TokenURI string `json:"tokenUri"`
	// AuthProviderX509CertURL is the auth_provider_x509_cert_url of the key
	// file.
	AuthProviderX509CertURL string `json:"authProviderX509CertUrl"`
	// ClientX509CertURL is the client_x509_cert_url of the key file.
	ClientX509CertURL string `json:"clientX509CertUrl"`
	// UniverseDomain is the universe_domain of the key file.
	UniverseDomain string `json:"universeDomain"`
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
	// AuthType is the kind of credential, and says which field below holds
	// it. It is required.
	AuthType CredentialType `json:"authType"`
	// ResourceRef names a resource that holds the credential. adk-python
	// reserves it for future use.
	ResourceRef string `json:"resourceRef,omitempty"`
	// APIKey is the key of a [CredentialTypeAPIKey] credential.
	// [Config.WithoutSecrets] removes it.
	APIKey string `json:"apiKey,omitempty"`
	// HTTP holds a [CredentialTypeHTTP] credential.
	HTTP *HTTPAuth `json:"http,omitempty"`
	// ServiceAccount holds a [CredentialTypeServiceAccount] credential.
	ServiceAccount *ServiceAccount `json:"serviceAccount,omitempty"`
	// OAuth2 holds a [CredentialTypeOAuth2] or [CredentialTypeOpenIDConnect]
	// credential, and the state of the flow that obtains it.
	OAuth2 *OAuth2Auth `json:"oauth2,omitempty"`
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

// Clone returns a copy of c that a caller can change without changing c, or
// nil when c is nil. Every struct, slice and map it reaches is copied,
// HTTP.AdditionalHeaders included. Values in an Extra map are copied deeply
// when they are JSON objects or arrays as encoding/json decodes them,
// map[string]any and []any; any other value is shared with c. A nil map or
// slice stays nil, so the copy equals c under [reflect.DeepEqual].
//
// Clone only reads c: it may run concurrently with other reads of c, but not
// with a change to it.
func (c *Credential) Clone() *Credential {
	if c == nil {
		return nil
	}
	out := *c
	out.HTTP = c.HTTP.clone()
	out.OAuth2 = c.OAuth2.clone()
	out.ServiceAccount = c.ServiceAccount.clone()
	out.Extra = cloneExtra(c.Extra)
	return &out
}
