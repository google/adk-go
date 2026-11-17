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

package authtool

import (
	"context"
	"errors"
	"fmt"
	"strconv"

	"golang.org/x/oauth2"
	"golang.org/x/oauth2/clientcredentials"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/internal/authhandler"
	"google.golang.org/adk/v2/tool/toolauth"
)

// CredentialExchanger exchanges a credential.
//
// Exchange receives the manager's config and a copy of the credential to
// exchange, and returns the credential to use and whether it was exchanged.
// When it returns an error together with a credential, the manager goes on
// with that credential and returns the error beside it; an error with a nil
// credential fails the call, and so does a nil credential with a nil error.
// Exchange must not change cfg, and may run for several calls at once.
//
// This is adk-python's BaseCredentialExchanger
// (auth/exchanger/base_credential_exchanger.py), which adk-python marks
// experimental.
//
// EXPERIMENTAL: CredentialExchanger is experimental; its API and behavior may change or be removed in a later version.
type CredentialExchanger interface {
	Exchange(ctx agent.Context, cfg *toolauth.Config, cred *toolauth.Credential) (exchanged *toolauth.Credential, didExchange bool, err error)
}

// OAuth2CredentialExchanger is the built-in exchanger for OAuth2 and OIDC client-credentials and authorization-code grants.
//
// Its zero value is ready to use. [NewCredentialManager] registers it for
// OAuth2 and OpenID Connect credentials.
//
// This is adk-python's OAuth2CredentialExchanger
// (auth/exchanger/oauth2_credential_exchanger.py), which adk-python marks
// experimental.
//
// EXPERIMENTAL: OAuth2CredentialExchanger is experimental; its API and behavior may change or be removed in a later version.
type OAuth2CredentialExchanger struct{}

// Exchange trades the credential at the scheme's endpoint.
//
// It redeems the authorization code in cred, or requests a token with the
// client-credentials grant, depending on cfg's scheme, and returns a copy of
// cred holding the tokens and true. The grant is the scheme's first OAuth2
// flow in the order client credentials, authorization code; for OpenID
// Connect it is client credentials when the scheme's GrantTypesSupported
// lists it, and authorization code otherwise. The code is the one in cred's
// AuthResponseURI, whose state must match cred's State, or else cred's
// AuthCode. The request sends no scopes and is bounded by ten seconds.
//
// Exchange returns (cred, false, nil) when cred already holds an access
// token. It returns (cred, false, err), with cred unchanged, when cred has no
// OAuth2 section, when no token request can be built (another grant type, no
// token endpoint, no client ID, no code, an unsupported client authentication
// method) or when the token endpoint refuses the request; the error never
// quotes the endpoint's response or a secret. adk-python returns the
// credential unchanged in all of these cases too, after a logged warning. It
// returns (nil, false, err) when cfg or cred is nil or cfg's scheme has no
// type.
//
// This is adk-python's OAuth2CredentialExchanger.exchange
// (auth/exchanger/oauth2_credential_exchanger.py), which adk-python marks
// experimental.
//
// EXPERIMENTAL: Exchange is experimental; its API and behavior may change or be removed in a later version.
func (OAuth2CredentialExchanger) Exchange(ctx agent.Context, cfg *toolauth.Config, cred *toolauth.Credential) (*toolauth.Credential, bool, error) {
	if cfg == nil || cfg.AuthScheme.Type == "" {
		return nil, false, errors.New("authtool: OAuth2 exchange needs an auth scheme")
	}
	if cred == nil {
		return nil, false, errors.New("authtool: OAuth2 exchange needs a credential")
	}
	o := cred.OAuth2
	if o == nil {
		return cred, false, errors.New("authtool: OAuth2 exchange: credential has no oauth2 section")
	}
	if o.AccessToken != "" {
		return cred, false, nil
	}
	scheme := &cfg.AuthScheme
	grant := grantType(scheme)
	var opts []oauth2.AuthCodeOption
	switch grant {
	case toolauth.OAuthGrantTypeClientCredentials:
	case toolauth.OAuthGrantTypeAuthorizationCode:
		if o.CodeVerifier != "" {
			opts = append(opts, oauth2.VerifierOption(o.CodeVerifier))
		}
	default:
		return cred, false, fmt.Errorf("authtool: OAuth2 exchange: grant type %q is not supported", grant)
	}
	conf, err := authhandler.TokenConfig(scheme, o)
	if err != nil {
		return cred, false, fmt.Errorf("authtool: OAuth2 exchange: %w", err)
	}
	ctx2, cancel := context.WithTimeout(ctx, authhandler.TokenRequestTimeout)
	defer cancel()
	var tok *oauth2.Token
	if grant == toolauth.OAuthGrantTypeClientCredentials {
		if tok, err = clientCredentials(conf).Token(ctx2); err == nil {
			// clientcredentials drops expires_in, which adk-python keeps.
			tok.ExpiresIn = expiresIn(tok)
		}
	} else {
		var code string
		if code, err = authhandler.AuthCode(o); err != nil {
			return cred, false, fmt.Errorf("authtool: OAuth2 exchange: %w", err)
		}
		tok, err = conf.Exchange(ctx2, code, opts...)
	}
	if err != nil {
		return cred, false, fmt.Errorf("authtool: OAuth2 %s grant: %w", grant, authhandler.TokenError(err))
	}
	return authhandler.WithTokens(cred, tok), true, nil
}

// grantType returns the grant s is redeemed with, as adk-python's
// _determine_grant_type decides, or "" when s is neither OAuth2 with flows
// nor OpenID Connect.
func grantType(s *toolauth.Scheme) toolauth.OAuthGrantType {
	switch {
	case s.Type == toolauth.SchemeTypeOAuth2 && s.Flows != nil:
		return s.Flows.GrantType()
	case s.Type == toolauth.SchemeTypeOpenIDConnect && usesClientCredentials(s):
		return toolauth.OAuthGrantTypeClientCredentials
	case s.Type == toolauth.SchemeTypeOpenIDConnect:
		return toolauth.OAuthGrantTypeAuthorizationCode
	}
	return ""
}

func clientCredentials(conf *oauth2.Config) *clientcredentials.Config {
	return &clientcredentials.Config{
		ClientID:     conf.ClientID,
		ClientSecret: conf.ClientSecret,
		TokenURL:     conf.Endpoint.TokenURL,
		AuthStyle:    conf.Endpoint.AuthStyle,
	}
}

// expiresIn reads expires_in from the token response, a JSON number or a
// form value, or returns 0 when it has none.
func expiresIn(tok *oauth2.Token) int64 {
	switch v := tok.Extra("expires_in").(type) {
	case float64:
		return int64(v)
	case string:
		n, _ := strconv.ParseInt(v, 10, 64)
		return n
	}
	return 0
}
