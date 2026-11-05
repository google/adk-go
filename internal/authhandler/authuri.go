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

// Package authhandler runs the server side of the end-user OAuth2 consent
// flow: it builds the sign-in request ADK sends to the client and, in later
// files, stores and exchanges the client's answer.
package authhandler

import (
	"cmp"
	"crypto/rand"
	"encoding/base64"
	"errors"
	"maps"
	"slices"

	"golang.org/x/oauth2"

	"google.golang.org/adk/v2/tool/toolauth"
)

const (
	pkceMethodS256 = "S256"
	// 36 random bytes encode to a 48-character verifier.
	verifierBytes = 36
	stateBytes    = 24
)

// GenerateAuthURI returns a copy of cfg's raw OAuth2 credential with the
// authorization URI and state filled in. A client with no secret is public,
// so it gets PKCE with S256 unless it names another method, which is an
// error. cfg is not modified.
func GenerateAuthURI(cfg *toolauth.Config) (*toolauth.Credential, error) {
	raw := cfg.RawAuthCredential
	if raw == nil || raw.OAuth2 == nil {
		return nil, errors.New("authhandler: raw auth credential has no oauth2 section")
	}
	endpoint, scopes, err := authorizationEndpoint(&cfg.AuthScheme)
	if err != nil {
		return nil, err
	}

	o := *raw.OAuth2
	method := o.CodeChallengeMethod
	if method == "" && o.ClientSecret == "" {
		method = pkceMethodS256
	}
	if method != "" && method != pkceMethodS256 {
		return nil, errors.New("authhandler: unsupported code challenge method, only S256 is supported")
	}

	opts := []oauth2.AuthCodeOption{
		oauth2.AccessTypeOffline,
		oauth2.SetAuthURLParam("prompt", cmp.Or(o.Prompt, "consent")),
	}
	if o.Audience != "" {
		opts = append(opts, oauth2.SetAuthURLParam("audience", o.Audience))
	}
	if o.Nonce != "" {
		opts = append(opts, oauth2.SetAuthURLParam("nonce", o.Nonce))
	}
	if method != "" {
		o.CodeVerifier = cmp.Or(o.CodeVerifier, randomToken(verifierBytes))
		o.CodeChallengeMethod = method
		opts = append(opts, oauth2.S256ChallengeOption(o.CodeVerifier))
	}

	o.State = randomToken(stateBytes)
	conf := oauth2.Config{
		ClientID:    o.ClientID,
		RedirectURL: o.RedirectURI,
		Scopes:      scopes,
		Endpoint:    oauth2.Endpoint{AuthURL: endpoint},
	}
	o.AuthURI = conf.AuthCodeURL(o.State, opts...)

	out := *raw
	out.OAuth2 = &o
	return &out, nil
}

// authorizationEndpoint returns the sign-in URL and the scopes of the first
// flow that is set, sorted. Like adk-python, it falls back to a token URL when
// no flow has an authorization URL.
func authorizationEndpoint(s *toolauth.Scheme) (string, []string, error) {
	if s.Type == toolauth.SchemeTypeOpenIDConnect && s.AuthorizationEndpoint != "" {
		return s.AuthorizationEndpoint, s.Scopes, nil
	}
	f := s.Flows
	if f == nil {
		return "", nil, errors.New("authhandler: auth scheme carries no OAuth2 flows")
	}
	var endpoint string
	if f.Implicit != nil {
		endpoint = f.Implicit.AuthorizationURL
	}
	if endpoint == "" && f.AuthorizationCode != nil {
		endpoint = f.AuthorizationCode.AuthorizationURL
	}
	if endpoint == "" && f.ClientCredentials != nil {
		endpoint = f.ClientCredentials.TokenURL
	}
	if endpoint == "" && f.Password != nil {
		endpoint = f.Password.TokenURL
	}
	if endpoint == "" {
		return "", nil, errors.New("authhandler: no OAuth2 flow declares an authorization endpoint")
	}
	// A flow is set here, since the endpoint came from one.
	first := cmp.Or(f.Implicit, f.AuthorizationCode, f.ClientCredentials, f.Password)
	return endpoint, slices.Sorted(maps.Keys(first.Scopes)), nil
}

// randomToken returns n random bytes as unpadded base64url.
func randomToken(n int) string {
	b := make([]byte, n)
	// rand.Read never returns an error; it ends the program if the system
	// has no randomness.
	_, _ = rand.Read(b)
	return base64.RawURLEncoding.EncodeToString(b)
}
