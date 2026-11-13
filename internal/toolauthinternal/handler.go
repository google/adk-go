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

// Package toolauthinternal runs the auth flow behind the
// adk_request_credential function call. It turns the [toolauth.Config] a tool
// asks for into the request ADK sends to the client, and it stores, and
// exchanges for a token, the credential the client sends back.
//
// The functions never change the configs and credentials they are given. The
// credentials they return are copies, but a copy shares its HTTP, service
// account and Extra values with the original, so callers must not change
// those in place either.
package toolauthinternal

import (
	"context"
	"crypto/rand"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"slices"
	"strings"

	"golang.org/x/oauth2"

	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool/toolauth"
)

// GenerateRequest returns the auth request for cfg: the config ADK sends to
// the client in an adk_request_credential call and keeps in the session.
//
// For an OAuth2 or OpenID Connect scheme whose exchanged credential has no
// authorization URI yet, the request carries a new one, with its state, built
// from the raw credential by [GenerateURI]. It fails when the raw credential,
// its OAuth2 section or its client ID is missing, or when GenerateURI fails.
// Any other config is returned as it is.
//
// The request never carries the secrets the tool was configured with: it is
// the output of [toolauth.Config.WithoutSecrets], so it carries cfg's
// credential key in CredentialKey.
func GenerateRequest(cfg *toolauth.Config) (*toolauth.Config, error) {
	req, err := generateRequest(cfg)
	if err != nil {
		return nil, err
	}
	return req.WithoutSecrets()
}

func generateRequest(cfg *toolauth.Config) (*toolauth.Config, error) {
	if !isOAuthScheme(cfg.AuthScheme) {
		return cfg, nil
	}
	if ex := cfg.ExchangedAuthCredential; ex != nil && ex.OAuth2 != nil && ex.OAuth2.AuthURI != "" {
		return cfg, nil
	}
	raw := cfg.RawAuthCredential
	if raw == nil {
		return nil, fmt.Errorf("auth scheme %s requires a raw auth credential", cfg.AuthScheme.Type)
	}
	if raw.OAuth2 == nil {
		return nil, fmt.Errorf("auth scheme %s requires oauth2 in the raw auth credential", cfg.AuthScheme.Type)
	}
	if raw.OAuth2.AuthURI != "" {
		return &toolauth.Config{
			AuthScheme:              cfg.AuthScheme,
			RawAuthCredential:       raw,
			ExchangedAuthCredential: copyCredential(raw),
			CredentialKey:           cfg.CredentialKey,
		}, nil
	}
	// A public client, such as one using PKCE, has a client ID and no secret.
	if raw.OAuth2.ClientID == "" {
		return nil, fmt.Errorf("auth scheme %s requires a client ID in the raw auth credential", cfg.AuthScheme.Type)
	}
	exchanged, err := GenerateURI(raw, cfg.AuthScheme)
	if err != nil {
		return nil, err
	}
	return &toolauth.Config{
		AuthScheme:              cfg.AuthScheme,
		RawAuthCredential:       raw,
		ExchangedAuthCredential: exchanged,
		CredentialKey:           cfg.CredentialKey,
	}, nil
}

// GenerateURI returns a copy of raw with an OAuth2 authorization URI for
// scheme in AuthURI and the random state the URI carries in State.
//
// The authorization endpoint and scopes come from an OpenID Connect scheme's
// inline endpoints, or else from the first of the scheme's flows that has
// one. The URI asks for offline access and for the prompt in raw, or consent
// when that is empty, and carries raw's audience and nonce when they are set.
// A client without a secret and without a code challenge method uses PKCE
// with S256; then, or when raw asks for S256, the copy carries the code
// verifier and method, generating the verifier when raw has none.
//
// GenerateURI fails when raw or its OAuth2 section is nil, when the scheme
// gives no authorization endpoint, and when raw asks for a code challenge
// method other than S256.
func GenerateURI(raw *toolauth.Credential, scheme toolauth.Scheme) (*toolauth.Credential, error) {
	if raw == nil || raw.OAuth2 == nil {
		return nil, errors.New("raw auth credential or its oauth2 section is empty")
	}
	var endpoint string
	var scopes []string
	if isOIDCWithConfig(scheme) {
		endpoint = scheme.AuthorizationEndpoint
		scopes = slices.Clone(scheme.Scopes)
	} else {
		// Flows are read off any scheme type, not only oauth2, so that a
		// custom scheme carrying flows joins the consent flow, as in adk-python.
		f := scheme.Flows
		if f == nil {
			return nil, fmt.Errorf("cannot generate an auth uri for auth scheme %s: it carries no OAuth2 flows", scheme.Type)
		}
		for _, url := range []string{
			flowURL(f.Implicit, true),
			flowURL(f.AuthorizationCode, true),
			flowURL(f.ClientCredentials, false),
			flowURL(f.Password, false),
		} {
			if url != "" {
				endpoint = url
				break
			}
		}
		if endpoint == "" {
			return nil, fmt.Errorf("cannot generate an auth uri for auth scheme %s: no flow declares an authorization endpoint", scheme.Type)
		}
		for _, flow := range []*toolauth.OAuthFlow{f.Implicit, f.AuthorizationCode, f.ClientCredentials, f.Password} {
			if flow != nil {
				// adk-python keeps the scopes in the order the scheme declares
				// them; a Go map has no order, so they are sorted to keep the URI
				// the same from one run to the next.
				scopes = slices.Sorted(maps.Keys(flow.Scopes))
				break
			}
		}
	}

	o := raw.OAuth2
	method := o.CodeChallengeMethod
	if o.ClientSecret == "" && method == "" {
		method = "S256"
	}
	verifier := o.CodeVerifier
	if method != "" {
		if method != "S256" {
			return nil, errors.New("unsupported code challenge method: only S256 is supported")
		}
		if verifier == "" {
			verifier = oauth2.GenerateVerifier()
		}
	}

	prompt := o.Prompt
	if prompt == "" {
		prompt = "consent"
	}
	opts := []oauth2.AuthCodeOption{oauth2.AccessTypeOffline, oauth2.SetAuthURLParam("prompt", prompt)}
	if o.Audience != "" {
		opts = append(opts, oauth2.SetAuthURLParam("audience", o.Audience))
	}
	if o.Nonce != "" {
		opts = append(opts, oauth2.SetAuthURLParam("nonce", o.Nonce))
	}
	if method != "" {
		opts = append(opts, oauth2.S256ChallengeOption(verifier))
	}
	conf := oauth2.Config{
		ClientID:     o.ClientID,
		ClientSecret: o.ClientSecret,
		Endpoint:     oauth2.Endpoint{AuthURL: endpoint},
		RedirectURL:  o.RedirectURI,
		Scopes:       scopes,
	}
	state := rand.Text()

	out := copyCredential(raw)
	out.OAuth2.AuthURI = conf.AuthCodeURL(state, opts...)
	out.OAuth2.State = state
	if verifier != "" {
		out.OAuth2.CodeVerifier = verifier
		out.OAuth2.CodeChallengeMethod = method
	}
	return out, nil
}

// flowURL returns the authorization URL of flow when auth is set, and its
// token URL otherwise. It returns "" for a nil flow.
func flowURL(flow *toolauth.OAuthFlow, auth bool) string {
	switch {
	case flow == nil:
		return ""
	case auth:
		return flow.AuthorizationURL
	default:
		return flow.TokenURL
	}
}

// ParseAndStoreResponse stores the credential in cfg's exchanged credential
// in state under "temp:" followed by cfg's key, without its OAuth2 client
// secret, since a client can read session state.
//
// When cfg has a raw credential with an OAuth2 section, its OAuth2 client ID
// and secret replace the ones in the exchanged credential, so the client
// cannot choose which OAuth2 client a token is exchanged for. When the
// credential then has a client secret and no access token, it is exchanged
// for a token first; a failed exchange is returned as an error and nothing is
// stored.
//
// A request that the flow issues carries no client secret, so on that path
// no exchange happens: the stored credential holds the auth code or the auth
// response URI that the client sent, not an access token.
func ParseAndStoreResponse(ctx context.Context, cfg *toolauth.Config, state session.State) error {
	key, err := cfg.Key()
	if err != nil {
		return err
	}
	cred := withConfiguredClient(cfg.ExchangedAuthCredential, cfg.RawAuthCredential)
	if isExchangeable(cfg.AuthScheme, cred, false) {
		cred, err = ExchangeToken(ctx, cred, cfg.AuthScheme)
		if err != nil {
			return err
		}
	}
	var stored any
	if c := withoutClientSecret(cred); c != nil {
		stored = c
	}
	return state.Set(session.KeyPrefixTemp+key, stored)
}

// Response returns the credential stored for cfg in state, under "temp:"
// followed by cfg's key or, failing that, under the key alone. It returns nil
// when neither holds a credential.
//
// A stored value may be a [toolauth.Credential], a pointer to one, a JSON
// object as a map, or a string, which [credentialFromString] turns into a
// credential for cfg's scheme. A value that is none of these is skipped.
//
// The stored credential carries no client secret, so the client ID and secret
// of cfg's raw credential are put back on it. When it still needs a token,
// including for a public client with no secret, it is exchanged for one, and
// the result is written back, without the secret, under the key it was read
// from.
//
// Nothing in ADK calls Response yet: a tool that asked for a credential reads
// it from state itself until tools have a way to call this.
func Response(ctx context.Context, cfg *toolauth.Config, state session.State) (*toolauth.Credential, error) {
	key, cred, err := readStoredCredential(cfg, state)
	if err != nil || cred == nil {
		return nil, err
	}
	cred = withConfiguredClient(cred, cfg.RawAuthCredential)
	if !isExchangeable(cfg.AuthScheme, cred, true) {
		return cred, nil
	}
	exchanged, err := ExchangeToken(ctx, cred, cfg.AuthScheme)
	if err != nil {
		return nil, err
	}
	if err := state.Set(key, withoutClientSecret(exchanged)); err != nil {
		return nil, err
	}
	return exchanged, nil
}

// readStoredCredential returns the state key and the credential stored for
// cfg, or a nil credential when none is stored.
func readStoredCredential(cfg *toolauth.Config, state session.State) (string, *toolauth.Credential, error) {
	credKey, err := cfg.Key()
	if err != nil {
		return "", nil, err
	}
	for _, key := range []string{session.KeyPrefixTemp + credKey, credKey} {
		val, err := state.Get(key)
		if errors.Is(err, session.ErrStateKeyNotExist) {
			continue
		}
		if err != nil {
			return "", nil, err
		}
		// adk-python logs a warning for a malformed value before skipping it;
		// ADK Go has no logger for library code, so the value is skipped
		// silently.
		switch v := val.(type) {
		case *toolauth.Credential:
			if v != nil {
				return key, v, nil
			}
		case toolauth.Credential:
			return key, &v, nil
		case map[string]any:
			b, err := json.Marshal(v)
			if err != nil {
				continue
			}
			var c toolauth.Credential
			if err := json.Unmarshal(b, &c); err != nil {
				continue
			}
			return key, &c, nil
		case string:
			if c, err := credentialFromString(v, cfg.AuthScheme); err == nil {
				return key, c, nil
			}
		}
	}
	return "", nil, nil
}

// credentialFromString returns the credential that val, a raw token or key,
// stands for under scheme: an API key for an apiKey scheme, an HTTP token for
// an http scheme, and an OAuth2 access token otherwise. It fails for an empty
// val and for HTTP basic auth, which needs a username and password.
func credentialFromString(val string, scheme toolauth.Scheme) (*toolauth.Credential, error) {
	if val == "" {
		return nil, errors.New("credential string is empty")
	}
	switch scheme.Type {
	case toolauth.SchemeTypeAPIKey:
		return &toolauth.Credential{AuthType: toolauth.CredentialTypeAPIKey, APIKey: val}, nil
	case toolauth.SchemeTypeHTTP:
		httpScheme := scheme.Scheme
		if httpScheme == "" {
			httpScheme = "bearer"
		}
		if strings.EqualFold(httpScheme, "basic") {
			return nil, errors.New("HTTP basic authentication requires a username and password, not a token string")
		}
		return &toolauth.Credential{
			AuthType: toolauth.CredentialTypeHTTP,
			HTTP: &toolauth.HTTPAuth{
				Scheme:      httpScheme,
				Credentials: toolauth.HTTPCredentials{Token: val},
			},
		}, nil
	}
	return &toolauth.Credential{
		AuthType: toolauth.CredentialTypeOAuth2,
		OAuth2:   &toolauth.OAuth2Auth{AccessToken: val},
	}, nil
}

// isOAuthScheme reports whether scheme is an OAuth2 or OpenID Connect scheme.
func isOAuthScheme(scheme toolauth.Scheme) bool {
	return scheme.Type == toolauth.SchemeTypeOAuth2 || scheme.Type == toolauth.SchemeTypeOpenIDConnect
}

// isOIDCWithConfig reports whether scheme is an OpenID Connect scheme that
// gives its endpoints inline.
func isOIDCWithConfig(scheme toolauth.Scheme) bool {
	return scheme.Type == toolauth.SchemeTypeOpenIDConnect && scheme.AuthorizationEndpoint != "" && scheme.TokenEndpoint != ""
}

// isExchangeable reports whether cred still needs a token exchange under
// scheme, and has what it takes to do one: a client ID, and a client secret
// unless allowPublic is set.
func isExchangeable(scheme toolauth.Scheme, cred *toolauth.Credential, allowPublic bool) bool {
	if !isOAuthScheme(scheme) || cred == nil {
		return false
	}
	o := cred.OAuth2
	if o == nil || o.AccessToken != "" || o.ClientID == "" {
		return false
	}
	return o.ClientSecret != "" || allowPublic
}

// withConfiguredClient returns a copy of cred with the OAuth2 client ID and
// secret of raw, or cred itself when cred has no OAuth2 section. When raw has
// no OAuth2 section, the client secret in cred is cleared so a client response
// cannot supply its own secret to trigger a token exchange.
func withConfiguredClient(cred, raw *toolauth.Credential) *toolauth.Credential {
	if cred == nil || cred.OAuth2 == nil {
		return cred
	}
	if raw == nil || raw.OAuth2 == nil {
		return withoutClientSecret(cred)
	}
	out := copyCredential(cred)
	out.OAuth2.ClientID = raw.OAuth2.ClientID
	out.OAuth2.ClientSecret = raw.OAuth2.ClientSecret
	return out
}

// withoutClientSecret returns a copy of cred without its OAuth2 client
// secret.
func withoutClientSecret(cred *toolauth.Credential) *toolauth.Credential {
	out := copyCredential(cred)
	if out != nil && out.OAuth2 != nil {
		out.OAuth2.ClientSecret = ""
	}
	return out
}

// copyCredential returns a copy of c whose OAuth2 section can be changed
// without changing c. The other sections are shared with c.
func copyCredential(c *toolauth.Credential) *toolauth.Credential {
	if c == nil {
		return nil
	}
	out := *c
	if c.OAuth2 != nil {
		o := *c.OAuth2
		out.OAuth2 = &o
	}
	return &out
}
