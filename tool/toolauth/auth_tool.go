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
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"maps"
	"slices"
)

// RequestCredentialFunctionCallName is the name of the function call ADK sends
// to the client to ask the end user for a credential.
const RequestCredentialFunctionCallName = "adk_request_credential"

// Config is what a tool needs in order to get a credential: how the
// resource authenticates, the credential the tool was configured with, and the
// credential the flow produces. ADK sends it to the client in an
// adk_request_credential call, and the client fills ExchangedAuthCredential in
// and sends it back.
type Config struct {
	AuthScheme Scheme `json:"authScheme"`
	// RawAuthCredential is the credential the tool was configured with, such
	// as an OAuth2 client ID and secret. A scheme that needs no exchange, such
	// as an API key, may leave it nil.
	RawAuthCredential *Credential `json:"rawAuthCredential,omitempty"`
	// ExchangedAuthCredential is filled in by ADK and the client together. For
	// OAuth2 and OpenID Connect, ADK adds the authorization URI and state, and
	// the client adds the auth response.
	ExchangedAuthCredential *Credential `json:"exchangedAuthCredential,omitempty"`
	// CredentialKey names the credential in a credential service.
	CredentialKey string `json:"credentialKey,omitempty"`
	// Extra holds JSON keys that match no field. See [Credential.Extra].
	Extra map[string]any `json:"-"`
}

// UnmarshalJSON implements [encoding/json.Unmarshaler] the way
// [Credential.UnmarshalJSON] does. AuthScheme is required.
func (c *Config) UnmarshalJSON(data []byte) error {
	type plain Config
	return decodeModel(data, (*plain)(c), &c.Extra, "Config", "authScheme")
}

// MarshalJSON implements [encoding/json.Marshaler]. It writes credentialKey as
// [Config.Key], since adk-python's dump always carries the key, and writes
// Extra after the fields.
func (c Config) MarshalJSON() ([]byte, error) {
	type plain Config
	key, err := c.Key()
	if err != nil {
		return nil, err
	}
	p := plain(c)
	p.CredentialKey = key
	return encodeModel(p, c.Extra)
}

// Key returns the key that names this config's credential in a credential
// service.
//
// It is CredentialKey when that is set. Otherwise it is the first non-empty
// string under credential_key or credentialKey among the unknown keys of
// RawAuthCredential, then of AuthScheme, which is where adk-python's
// AuthConfig looks. Otherwise it is derived the way adk-python's
// get_credential_key derives it (auth_tool.py): adk_<type>_<digest>_ followed,
// when there is a raw credential, by <authType>_<digest>. Each digest covers
// the JSON of the scheme or the credential without its own unknown keys. The
// OAuth2 fields that change during a flow, which are AuthURI, State,
// AuthResponseURI, AuthCode, AccessToken, RefreshToken, ExpiresAt, ExpiresIn
// and RedirectURI, are left out, so the key stays put while the flow runs.
//
// Every field of [Scheme] counts towards the key, whatever Type is, but
// Scheme.Extra does not. A custom scheme's own fields are in Extra, so two
// schemes of one custom type that differ only there get the same key; set
// CredentialKey to tell them apart. adk-python counts only the fields of the
// model it reads a scheme into, so for a scheme that reaches it as JSON
// without a credentialKey the two keys can differ. It reads
// {"type":"apiKey","in":"body","name":"X"} as a custom scheme and leaves in
// and name out, and it counts the fields a custom scheme class declares, such
// as continueUri on GcpAuthProviderScheme, which are in Extra here.
//
// Key trusts the config it is called on. A config that a client sends back in
// an adk_request_credential response can carry any credentialKey, so take
// only its ExchangedAuthCredential and store the credential under the key of
// the config ADK sent, as adk-python's store_auth_response (_auth_resume.py)
// does.
//
// Key fails only when a value in an Extra map cannot be encoded as JSON.
func (c *Config) Key() (string, error) {
	if c.CredentialKey != "" {
		return c.CredentialKey, nil
	}
	var credExtra map[string]any
	if c.RawAuthCredential != nil {
		credExtra = c.RawAuthCredential.Extra
	}
	for _, extra := range []map[string]any{credExtra, c.AuthScheme.Extra} {
		for _, name := range []string{"credential_key", "credentialKey"} {
			if k, ok := extra[name].(string); ok && k != "" {
				return k, nil
			}
		}
	}
	scheme := c.AuthScheme
	scheme.Extra = nil
	d, err := digest(scheme)
	if err != nil {
		return "", err
	}
	key := "adk_" + string(scheme.Type) + "_" + d + "_"
	if c.RawAuthCredential == nil {
		return key, nil
	}
	cred := *c.RawAuthCredential
	cred.Extra = nil
	if cred.OAuth2 != nil {
		o := *cred.OAuth2
		o.AuthURI, o.State, o.AuthResponseURI, o.AuthCode = "", "", "", ""
		o.AccessToken, o.RefreshToken, o.RedirectURI = "", "", ""
		o.ExpiresAt, o.ExpiresIn = 0, 0
		cred.OAuth2 = &o
	}
	if d, err = digest(cred); err != nil {
		return "", err
	}
	return key + string(cred.AuthType) + "_" + d, nil
}

// digest returns the first 16 hex digits of the SHA-256 of v's JSON, written
// the way Python's json.dumps(sort_keys=True, ensure_ascii=False,
// separators=(",", ":")) writes it: keys sorted, no spaces, and no escaping of
// &, <, > or non-ASCII. Two differences remain. Go always escapes U+2028 and
// U+2029, and it writes a number in an Extra map as it arrived, where Python
// rewrites a float such as 1e3 as 1000.0.
func digest(v any) (string, error) {
	b, err := json.Marshal(v)
	if err != nil {
		return "", fmt.Errorf("toolauth: credential key: %w", err)
	}
	dec := json.NewDecoder(bytes.NewReader(b))
	dec.UseNumber()
	var tree any
	if err := dec.Decode(&tree); err != nil {
		return "", fmt.Errorf("toolauth: credential key: %w", err)
	}
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(tree); err != nil {
		return "", fmt.Errorf("toolauth: credential key: %w", err)
	}
	sum := sha256.Sum256(bytes.TrimSuffix(buf.Bytes(), []byte("\n")))
	return hex.EncodeToString(sum[:8]), nil
}

// WithoutSecrets returns a deep copy of c without the secrets the agent was
// configured with, for sending to the client and keeping in the session: the
// OAuth2 client secret, the API key, the HTTP password, token and additional
// headers, and a service account's private key and key ID, in both
// RawAuthCredential and ExchangedAuthCredential. It is adk-python's
// _without_configured_secrets (auth_handler.py), which leaves the service
// account key in. The copy carries c's key in CredentialKey, since its
// stripped content would derive a different one.
//
// Values in Extra maps are copied deeply when they are JSON objects or arrays
// as encoding/json decodes them, map[string]any and []any; any other value is
// shared with c. WithoutSecrets fails only when [Config.Key] does.
func (c *Config) WithoutSecrets() (*Config, error) {
	key, err := c.Key()
	if err != nil {
		return nil, err
	}
	out := *c
	out.AuthScheme = c.AuthScheme.clone()
	out.RawAuthCredential = c.RawAuthCredential.clone()
	out.ExchangedAuthCredential = c.ExchangedAuthCredential.clone()
	out.CredentialKey = key
	out.Extra = cloneExtra(c.Extra)
	for _, cred := range []*Credential{out.RawAuthCredential, out.ExchangedAuthCredential} {
		if cred == nil {
			continue
		}
		cred.APIKey = ""
		if cred.OAuth2 != nil {
			cred.OAuth2.ClientSecret = ""
		}
		if cred.HTTP != nil {
			cred.HTTP.Credentials.Password = ""
			cred.HTTP.Credentials.Token = ""
			cred.HTTP.AdditionalHeaders = nil
		}
		if sa := cred.ServiceAccount; sa != nil && sa.ServiceAccountCredential != nil {
			sa.ServiceAccountCredential.PrivateKey = ""
			sa.ServiceAccountCredential.PrivateKeyID = ""
		}
	}
	return &out, nil
}

// The clone methods return deep copies for [Config.WithoutSecrets]. Each
// pointer receiver may be nil, and then the copy is nil too.

func (s Scheme) clone() Scheme {
	s.Flows = s.Flows.clone()
	s.TokenEndpointAuthMethodsSupported = slices.Clone(s.TokenEndpointAuthMethodsSupported)
	s.GrantTypesSupported = slices.Clone(s.GrantTypesSupported)
	s.Scopes = slices.Clone(s.Scopes)
	s.Extra = cloneExtra(s.Extra)
	return s
}

func (f *OAuthFlows) clone() *OAuthFlows {
	if f == nil {
		return nil
	}
	out := *f
	out.Implicit = f.Implicit.clone()
	out.Password = f.Password.clone()
	out.ClientCredentials = f.ClientCredentials.clone()
	out.AuthorizationCode = f.AuthorizationCode.clone()
	out.Extra = cloneExtra(f.Extra)
	return &out
}

func (f *OAuthFlow) clone() *OAuthFlow {
	if f == nil {
		return nil
	}
	out := *f
	out.Scopes = maps.Clone(f.Scopes)
	out.Extra = cloneExtra(f.Extra)
	return &out
}

func (c *Credential) clone() *Credential {
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

// clone leaves AdditionalHeaders shared, since WithoutSecrets drops it.
func (h *HTTPAuth) clone() *HTTPAuth {
	if h == nil {
		return nil
	}
	out := *h
	out.Credentials.Extra = cloneExtra(h.Credentials.Extra)
	out.Extra = cloneExtra(h.Extra)
	return &out
}

func (o *OAuth2Auth) clone() *OAuth2Auth {
	if o == nil {
		return nil
	}
	out := *o
	out.Extra = cloneExtra(o.Extra)
	return &out
}

func (s *ServiceAccount) clone() *ServiceAccount {
	if s == nil {
		return nil
	}
	out := *s
	out.ServiceAccountCredential = s.ServiceAccountCredential.clone()
	out.Scopes = slices.Clone(s.Scopes)
	out.Extra = cloneExtra(s.Extra)
	return &out
}

func (s *ServiceAccountCredential) clone() *ServiceAccountCredential {
	if s == nil {
		return nil
	}
	out := *s
	out.Extra = cloneExtra(s.Extra)
	return &out
}

// cloneExtra returns a copy of an Extra map, copying the JSON objects and
// arrays in it deeply.
func cloneExtra(m map[string]any) map[string]any {
	if m == nil {
		return nil
	}
	out := make(map[string]any, len(m))
	for k, v := range m {
		out[k] = cloneJSONValue(v)
	}
	return out
}

func cloneJSONValue(v any) any {
	switch v := v.(type) {
	case map[string]any:
		return cloneExtra(v)
	case []any:
		out := slices.Clone(v)
		for i := range out {
			out[i] = cloneJSONValue(out[i])
		}
		return out
	}
	return v
}

// ToolArguments are the arguments of an adk_request_credential call: the
// ID of the tool call that needs the credential, and its config.
type ToolArguments struct {
	FunctionCallID string `json:"functionCallId"`
	AuthConfig     Config `json:"authConfig"`
	// Extra holds JSON keys that match no field. See [Credential.Extra].
	Extra map[string]any `json:"-"`
}

// UnmarshalJSON implements [encoding/json.Unmarshaler] the way
// [Credential.UnmarshalJSON] does. FunctionCallID and AuthConfig are
// required.
func (a *ToolArguments) UnmarshalJSON(data []byte) error {
	type plain ToolArguments
	return decodeModel(data, (*plain)(a), &a.Extra, "ToolArguments", "functionCallId", "authConfig")
}

// MarshalJSON implements [encoding/json.Marshaler]. It writes Extra after the
// fields.
func (a ToolArguments) MarshalJSON() ([]byte, error) {
	type plain ToolArguments
	return encodeModel(plain(a), a.Extra)
}
