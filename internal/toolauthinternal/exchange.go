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

package toolauthinternal

import (
	"context"
	"errors"
	"fmt"
	"net/url"
	"slices"
	"strings"
	"time"

	"golang.org/x/oauth2"
	"golang.org/x/oauth2/clientcredentials"

	"google.golang.org/adk/v2/tool/toolauth"
)

// tokenRequestTimeout bounds one request to a token endpoint, so an endpoint
// that accepts the connection and never answers cannot hold the run.
const defaultTokenRequestTimeout = 10 * time.Second

var tokenRequestTimeout = defaultTokenRequestTimeout

// ExchangeToken exchanges cred for an OAuth2 token at the token endpoint of
// scheme and returns a copy of cred with the token in it.
//
// The scheme decides the token endpoint and the grant: an oauth2 scheme's
// authorization code flow or else its client credentials flow, or an OpenID
// Connect scheme's inline token endpoint, with the client credentials grant
// when the scheme supports it and the authorization code grant otherwise. The
// authorization code is taken from the code parameter of AuthResponseURI,
// whose state parameter must then match State, or else from AuthCode. The
// client authenticates with HTTP basic auth, or in the form for
// client_secret_post and for a public client, which sends no secret.
//
// cred is returned as it is when there is nothing to exchange: cred has no
// OAuth2 section, no client ID, or already an access token; the scheme gives
// no token endpoint or a grant other than those two; the client authenticates
// with a signed JWT, which is not supported; or there is no authorization
// code.
//
// adk-python logs a failed exchange and returns the credential unexchanged;
// ADK Go returns the error instead, as it does not log from library code. The
// error never quotes the token endpoint's response, and wraps
// [context.Canceled] or [context.DeadlineExceeded] when the request was cut
// short.
func ExchangeToken(ctx context.Context, cred *toolauth.Credential, scheme toolauth.Scheme) (*toolauth.Credential, error) {
	if cred == nil || cred.OAuth2 == nil || cred.OAuth2.ClientID == "" || cred.OAuth2.AccessToken != "" {
		return cred, nil
	}
	o := cred.OAuth2
	tokenURL, grant := tokenEndpoint(scheme)
	if tokenURL == "" {
		return cred, nil
	}
	style, ok := authStyle(o)
	if !ok {
		return cred, nil
	}

	ctx, cancel := context.WithTimeout(ctx, tokenRequestTimeout)
	defer cancel()
	var tok *oauth2.Token
	var err error
	switch grant {
	case toolauth.OAuthGrantTypeClientCredentials:
		// No scope is sent: RFC 6749 does not need one here, and some
		// providers reject it.
		cc := clientcredentials.Config{
			ClientID:     o.ClientID,
			ClientSecret: o.ClientSecret,
			TokenURL:     tokenURL,
			AuthStyle:    style,
		}
		tok, err = cc.Token(ctx)
	case toolauth.OAuthGrantTypeAuthorizationCode:
		code, codeErr := authorizationCode(o)
		if codeErr != nil {
			return nil, codeErr
		}
		if code == "" {
			return cred, nil
		}
		conf := oauth2.Config{
			ClientID:     o.ClientID,
			ClientSecret: o.ClientSecret,
			Endpoint:     oauth2.Endpoint{TokenURL: tokenURL, AuthStyle: style},
			RedirectURL:  o.RedirectURI,
		}
		var opts []oauth2.AuthCodeOption
		if o.CodeVerifier != "" {
			opts = append(opts, oauth2.VerifierOption(o.CodeVerifier))
		}
		tok, err = conf.Exchange(ctx, code, opts...)
	default:
		return cred, nil
	}
	if err != nil {
		return nil, exchangeError(err)
	}

	out := copyCredential(cred)
	out.OAuth2.AccessToken = tok.AccessToken
	out.OAuth2.RefreshToken = tok.RefreshToken
	out.OAuth2.IDToken, _ = tok.Extra("id_token").(string)
	out.OAuth2.ExpiresAt = 0
	if !tok.Expiry.IsZero() {
		out.OAuth2.ExpiresAt = tok.Expiry.Unix()
	}
	out.OAuth2.ExpiresIn = tok.ExpiresIn
	return out, nil
}

// tokenEndpoint returns the token endpoint and grant type that scheme uses,
// or "" when it gives none.
func tokenEndpoint(scheme toolauth.Scheme) (string, toolauth.OAuthGrantType) {
	switch {
	case isOIDCWithConfig(scheme):
		grant := toolauth.OAuthGrantTypeAuthorizationCode
		if slices.Contains(scheme.GrantTypesSupported, string(toolauth.OAuthGrantTypeClientCredentials)) {
			grant = toolauth.OAuthGrantTypeClientCredentials
		}
		return scheme.TokenEndpoint, grant
	case scheme.Type == toolauth.SchemeTypeOAuth2 && scheme.Flows != nil:
		f := scheme.Flows
		var tokenURL string
		switch {
		case f.AuthorizationCode != nil && f.AuthorizationCode.TokenURL != "":
			tokenURL = f.AuthorizationCode.TokenURL
		case f.ClientCredentials != nil && f.ClientCredentials.TokenURL != "":
			tokenURL = f.ClientCredentials.TokenURL
		}
		return tokenURL, f.GrantType()
	}
	return "", ""
}

// authStyle returns how the client in o authenticates to the token endpoint.
// It is always explicit: x/oauth2's auto-detection retries a failed request
// with the other style, which sends the credentials twice. It reports false
// for a method x/oauth2 cannot do.
func authStyle(o *toolauth.OAuth2Auth) (oauth2.AuthStyle, bool) {
	if o.ClientSecret == "" {
		if o.TokenEndpointAuthMethod == "private_key_jwt" {
			return 0, false
		}
		// A public client sends its client ID in the form and no secret.
		return oauth2.AuthStyleInParams, true
	}
	switch o.TokenEndpointAuthMethod {
	case "", "client_secret_basic":
		return oauth2.AuthStyleInHeader, true
	case "client_secret_post":
		return oauth2.AuthStyleInParams, true
	}
	// client_secret_jwt and private_key_jwt need a signed client assertion,
	// which x/oauth2 does not build.
	return 0, false
}

// authorizationCode returns the authorization code in o: the code parameter
// of AuthResponseURI, after one trailing "#" is dropped, or else AuthCode.
// Like adk-python's authlib, it refuses a response URI whose state parameter
// does not match State when State is set.
func authorizationCode(o *toolauth.OAuth2Auth) (string, error) {
	if o.AuthResponseURI != "" {
		u, err := url.Parse(strings.TrimSuffix(o.AuthResponseURI, "#"))
		if err != nil {
			// url.Parse errors quote the URL, which carries the code.
			return "", errors.New("auth response URI is not a valid URL")
		}
		q := u.Query()
		if code := q.Get("code"); code != "" {
			if o.State != "" && q.Get("state") != o.State {
				return "", errors.New("auth response URI state does not match the state of the auth request")
			}
			return code, nil
		}
	}
	return o.AuthCode, nil
}

// exchangeError returns the error for a failed token request without the
// response body or the URL, which x/oauth2's errors quote.
func exchangeError(err error) error {
	switch {
	case errors.Is(err, context.Canceled):
		return fmt.Errorf("oauth2 token exchange: %w", context.Canceled)
	case errors.Is(err, context.DeadlineExceeded):
		return fmt.Errorf("oauth2 token exchange: %w", context.DeadlineExceeded)
	}
	var re *oauth2.RetrieveError
	if errors.As(err, &re) && re.Response != nil {
		return fmt.Errorf("oauth2 token exchange: token endpoint returned HTTP status %d", re.Response.StatusCode)
	}
	return errors.New("oauth2 token exchange failed")
}
