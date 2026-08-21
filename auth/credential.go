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

package auth

import (
	"encoding/base64"
	"fmt"
	"net/http"

	"golang.org/x/oauth2"
)

// Credential is a resolved credential that writes itself onto an outbound HTTP
// request.
type Credential interface {
	// Apply writes the credential's auth headers onto h.
	Apply(h http.Header) error
}

// APIKeyCredential is a header-based API key, e.g. {Name: "X-Api-Key"}.
type APIKeyCredential struct {
	Name  string
	Value string
}

// Apply implements [Credential].
func (c APIKeyCredential) Apply(h http.Header) error {
	if c.Name == "" {
		return fmt.Errorf("auth: api key credential missing header name")
	}
	h.Set(c.Name, c.Value)
	return nil
}

// BearerCredential is an HTTP bearer token.
type BearerCredential struct {
	Token string
}

// Apply implements [Credential].
func (c BearerCredential) Apply(h http.Header) error {
	if c.Token == "" {
		return fmt.Errorf("auth: bearer credential missing token")
	}
	h.Set("Authorization", "Bearer "+c.Token)
	return nil
}

// BasicCredential is an HTTP basic username/password credential.
type BasicCredential struct {
	Username string
	Password string
}

// Apply implements [Credential].
func (c BasicCredential) Apply(h http.Header) error {
	// An empty username or password alone is allowed; reject only when both are empty.
	if c.Username == "" && c.Password == "" {
		return fmt.Errorf("auth: basic credential missing username and password")
	}
	raw := base64.StdEncoding.EncodeToString([]byte(c.Username + ":" + c.Password))
	h.Set("Authorization", "Basic "+raw)
	return nil
}

// OAuth2Credential mints a fresh access token from TokenSource on each Apply.
// For a static token, wrap it in [oauth2.StaticTokenSource].
type OAuth2Credential struct {
	TokenSource oauth2.TokenSource
}

// Apply implements [Credential].
func (c OAuth2Credential) Apply(h http.Header) error {
	if c.TokenSource == nil {
		return fmt.Errorf("auth: oauth2 credential missing token source")
	}
	tok, err := c.TokenSource.Token()
	if err != nil {
		return fmt.Errorf("auth: mint oauth2 token: %w", err)
	}
	h.Set("Authorization", tok.Type()+" "+tok.AccessToken)
	return nil
}

// WithHeaders wraps inner to also set headers verbatim (overriding inner on
// conflict) — e.g. x-goog-user-project alongside an OAuth2 token.
func WithHeaders(inner Credential, headers map[string]string) Credential {
	return withHeaders{inner: inner, headers: headers}
}

type withHeaders struct {
	inner   Credential
	headers map[string]string
}

// Apply implements [Credential].
func (c withHeaders) Apply(h http.Header) error {
	if c.inner == nil {
		return fmt.Errorf("auth: WithHeaders has nil inner credential")
	}
	if err := c.inner.Apply(h); err != nil {
		return err
	}
	for k, v := range c.headers {
		h.Set(k, v)
	}
	return nil
}
