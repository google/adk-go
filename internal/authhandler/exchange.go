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

package authhandler

import (
	"cmp"
	"context"
	"errors"
	"fmt"
	"net/url"
	"slices"
	"time"

	"golang.org/x/oauth2"

	"google.golang.org/adk/v2/tool/toolauth"
)

// TokenRequestTimeout matches adk-python's _TOKEN_REQUEST_TIMEOUT_SECONDS. It
// bounds each request to a token endpoint.
const TokenRequestTimeout = 10 * time.Second

// AuthCode returns the authorization code to redeem from o: the code in the
// redirect URL the client sent back (o.AuthResponseURI), or o.AuthCode when
// the URL carries none. A bare AuthCode comes with no redirect state, so
// nothing ties it to the request. Like authlib, it returns an error when the
// redirect URL does not parse or its state is not o.State (an empty o.State
// is not checked), and when o holds no code at all.
func AuthCode(o *toolauth.OAuth2Auth) (string, error) {
	code, err := codeFromRedirect(o.AuthResponseURI, o.State)
	if err != nil {
		return "", err
	}
	code = cmp.Or(code, o.AuthCode)
	if code == "" {
		return "", errors.New("authhandler: auth response carries no authorization code")
	}
	return code, nil
}

// codeFromRedirect returns the code in the redirect URL raw, or "" when it has
// none. Like authlib, it refuses a code whose state is not the request's.
func codeFromRedirect(raw, state string) (string, error) {
	if raw == "" {
		return "", nil
	}
	u, err := url.Parse(raw)
	if err != nil {
		return "", errors.New("authhandler: auth response URI does not parse")
	}
	q := u.Query()
	code := q.Get("code")
	if code == "" {
		return "", nil
	}
	if state != "" && q.Get("state") != state {
		return "", errors.New("authhandler: auth response state does not match the request")
	}
	return code, nil
}

// TokenConfig returns the settings for redeeming o's code under scheme: the
// token URL ([TokenEndpoint]), o's client ID, secret and redirect URI, and how
// the client authenticates there ([AuthStyle]). It returns an error when scheme
// declares no token endpoint, o has no client ID, or o's client
// authentication method is not supported.
func TokenConfig(scheme *toolauth.Scheme, o *toolauth.OAuth2Auth) (*oauth2.Config, error) {
	tokenURL, err := TokenEndpoint(scheme)
	if err != nil {
		return nil, err
	}
	if o.ClientID == "" {
		return nil, errors.New("authhandler: token request needs a client ID")
	}
	style, err := AuthStyle(o)
	if err != nil {
		return nil, err
	}
	return &oauth2.Config{
		ClientID:     o.ClientID,
		ClientSecret: o.ClientSecret,
		RedirectURL:  o.RedirectURI,
		Endpoint:     oauth2.Endpoint{TokenURL: tokenURL, AuthStyle: style},
	}, nil
}

// TokenEndpoint returns the token endpoint from the scheme: an OpenID Connect
// scheme's TokenEndpoint, or the TokenURL of an OAuth2 scheme's
// authorization-code flow, else of its client-credentials flow. It returns an
// error when s declares none.
func TokenEndpoint(s *toolauth.Scheme) (string, error) {
	if s.Type == toolauth.SchemeTypeOpenIDConnect && s.TokenEndpoint != "" {
		return s.TokenEndpoint, nil
	}
	if f := s.Flows; s.Type == toolauth.SchemeTypeOAuth2 && f != nil {
		if f.AuthorizationCode != nil && f.AuthorizationCode.TokenURL != "" {
			return f.AuthorizationCode.TokenURL, nil
		}
		if f.ClientCredentials != nil && f.ClientCredentials.TokenURL != "" {
			return f.ClientCredentials.TokenURL, nil
		}
	}
	return "", errors.New("authhandler: auth scheme declares no token endpoint")
}

// AuthStyle says where the client's credentials go in the token request,
// following o's TokenEndpointAuthMethod: the Authorization header for "" and
// "client_secret_basic", the body for "client_secret_post". A client without a
// secret is public and sends only its ID in the body. It returns an error for
// "private_key_jwt", and for any other method when o has a secret.
func AuthStyle(o *toolauth.OAuth2Auth) (oauth2.AuthStyle, error) {
	switch m := o.TokenEndpointAuthMethod; {
	case m == "private_key_jwt":
		return 0, errors.New("authhandler: token endpoint auth method private_key_jwt is not supported")
	case o.ClientSecret == "":
		return oauth2.AuthStyleInParams, nil
	case m == "" || m == "client_secret_basic":
		return oauth2.AuthStyleInHeader, nil
	case m == "client_secret_post":
		return oauth2.AuthStyleInParams, nil
	}
	return 0, errors.New("authhandler: token endpoint auth method client_secret_jwt is not supported")
}

// exchangeCode trades the authorization code in cred for tokens at scheme's
// token endpoint and returns a copy of cred holding them. A credential that
// already has an access token is returned as it is.
func exchangeCode(ctx context.Context, scheme *toolauth.Scheme, cred *toolauth.Credential) (*toolauth.Credential, error) {
	o := cred.OAuth2
	if o == nil {
		return nil, errors.New("authhandler: credential has no oauth2 section")
	}
	if o.AccessToken != "" {
		return cred, nil
	}
	if !usesCodeGrant(scheme) {
		return nil, errors.New("authhandler: only the authorization code grant is exchanged here")
	}
	code, err := AuthCode(o)
	if err != nil {
		return nil, err
	}
	conf, err := TokenConfig(scheme, o)
	if err != nil {
		return nil, err
	}
	var opts []oauth2.AuthCodeOption
	if o.CodeVerifier != "" {
		opts = append(opts, oauth2.VerifierOption(o.CodeVerifier))
	}
	ctx, cancel := context.WithTimeout(ctx, TokenRequestTimeout)
	defer cancel()
	tok, err := conf.Exchange(ctx, code, opts...)
	if err != nil {
		return nil, TokenError(err)
	}
	return WithTokens(cred, tok), nil
}

// usesCodeGrant reports whether scheme is redeemed with the authorization code
// grant, decided as adk-python's _determine_grant_type decides it.
func usesCodeGrant(s *toolauth.Scheme) bool {
	if s.Type == toolauth.SchemeTypeOpenIDConnect {
		return !slices.Contains(s.GrantTypesSupported, "client_credentials")
	}
	return s.Flows != nil && s.Flows.GrantType() == toolauth.OAuthGrantTypeAuthorizationCode
}

// TokenError keeps the status and OAuth error code of a refused token request,
// and wraps any other error from x/oauth2 with %w. x/oauth2's message quotes
// the response, which can hold text about the user, so a refusal's message is
// dropped.
func TokenError(err error) error {
	var re *oauth2.RetrieveError
	if !errors.As(err, &re) {
		return fmt.Errorf("authhandler: token request: %w", err)
	}
	return fmt.Errorf("authhandler: token endpoint refused the code: %s %s", re.Response.Status, re.ErrorCode)
}

// WithTokens returns a copy of cred holding tok, field by field as adk-python's
// update_credential_with_tokens sets them. cred must have an OAuth2 section;
// cred itself is not changed.
func WithTokens(cred *toolauth.Credential, tok *oauth2.Token) *toolauth.Credential {
	o := *cred.OAuth2
	o.AccessToken = tok.AccessToken
	o.RefreshToken = tok.RefreshToken
	o.IDToken, _ = tok.Extra("id_token").(string)
	o.ExpiresIn = tok.ExpiresIn
	o.ExpiresAt = 0
	if !tok.Expiry.IsZero() {
		o.ExpiresAt = tok.Expiry.Unix()
	}
	out := *cred
	out.OAuth2 = &o
	return &out
}
