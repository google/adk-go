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

package authn

import (
	"context"
	"errors"
	"fmt"
	"log"
	"net/http"
	"slices"
	"strings"

	"google.golang.org/api/idtoken"
)

// googleOIDC authenticates the Google-signed OIDC bearer token that a Pub/Sub
// push subscription or an Eventarc trigger attaches when it is configured with
// a service account.
type googleOIDC struct {
	audience               string
	allowedServiceAccounts []string

	// validate verifies the token. NewGoogleOIDC sets it to idtoken.Validate; a
	// field rather than a package variable so parallel tests can install their
	// own fake without racing and without a live call to Google's certificate
	// endpoint. It has no default: a struct built directly without setting it
	// (only a test does that) makes validateToken panic and recover into a
	// rejection, and the constructor's assignment is what keeps production off
	// that path.
	validate tokenValidator
}

// tokenValidator matches the signature of [idtoken.Validate], which is what
// production uses.
type tokenValidator func(ctx context.Context, idToken, audience string) (*idtoken.Payload, error)

// googleIssuers are the two spellings Google uses in iss. idtoken.Validate
// parses the claim but leaves checking it to the caller.
var googleIssuers = []string{"accounts.google.com", "https://accounts.google.com"}

// GoogleOIDCConfig configures [NewGoogleOIDC].
//
// A struct rather than positional parameters: both fields are required today and
// the verifier is likely to grow more later (an HTTP client for the certificate
// fetch, possibly several accepted audiences), and a struct absorbs those as
// compatible additions where a new positional parameter would be a breaking
// change. This is the house style for a new constructor; see AGENTS.md.
type GoogleOIDCConfig struct {
	// Audience is the OIDC audience a token must carry to be accepted. Required.
	//
	// It does not identify the caller, because it is chosen freely by whoever
	// mints the token: any principal that can call
	// iam.serviceAccounts.getOpenIdToken on a service account of its own can
	// obtain a Google-signed token for a given audience.
	Audience string

	// AllowedServiceAccounts is the set of service account emails permitted to
	// call. Required and non-empty: since the audience does not identify the
	// caller, an authenticator with no allow-list would admit any principal
	// holding a Google-signed token for the audience. Each entry is matched
	// against a token's verified email claim, the identity the subscription or
	// trigger actually delivers as.
	AllowedServiceAccounts []string
}

// NewGoogleOIDC returns an [Authenticator] requiring a Google-signed OIDC bearer
// token issued for cfg.Audience, verified with [idtoken.Validate].
//
// Both fields are required. An empty [GoogleOIDCConfig.Audience] is rejected
// rather than passed through to [idtoken.Validate], which skips the audience
// check when it is given an empty one and would leave an authenticator that
// takes a token minted for any audience at all (leave the [Authenticator]
// unset, or use [NewNoop], to run without authentication). An empty
// [GoogleOIDCConfig.AllowedServiceAccounts] is rejected too. Because the
// allow-list is mandatory, an [Authenticator] this returns can never be
// configured to accept an unlisted principal, including when it is handed
// directly to adkrest.ServerConfig.Authenticator rather than reached through a
// trigger flag.
//
// This mirrors GoogleOidcVerifier(audience, allowed_emails) in adk-python, with
// deliberate differences. adk-python's allow-list is optional (allowed_emails
// defaults to None), whereas here it is mandatory. adk-python also tests
// email_verified for Python truthiness, so every non-empty string passes it,
// "false" included, whereas here the claim must be an actual boolean. And
// adk-python names the failed check in its response detail, where [Middleware]
// answers with the same body for every rejection carrying a given status.
//
// adk-python already answers 403 for a verified-but-unlisted principal, so the
// [ErrForbidden]/403 this returns for that case is not itself a difference, and
// a missing, malformed or unverifiable credential is a 401 either way.
func NewGoogleOIDC(cfg GoogleOIDCConfig) (Authenticator, error) {
	if cfg.Audience == "" {
		return nil, errors.New("authn: NewGoogleOIDC requires an audience")
	}
	if len(cfg.AllowedServiceAccounts) == 0 {
		return nil, errors.New("authn: NewGoogleOIDC requires at least one allowed service account; " +
			"the audience alone does not identify the caller")
	}
	// A blank or space-padded entry is rejected rather than stored. Length alone
	// would pass []string{""} -- what an operator gets from an unset environment
	// variable, or a trailing comma alongside a real entry -- and a "" entry is
	// not inert. Authenticate reads an absent email claim as "", so a payload
	// carrying email_verified true with no email matches a blank entry and is
	// admitted. Refusing the entry here keeps that unreachable.
	//
	// This is defense in depth rather than protection from a routine token.
	// getOpenIdToken emits email and email_verified as a pair, so a token minted
	// without includeEmail carries neither, and !emailVerified in Authenticate
	// refuses that one whatever the list holds.
	for _, account := range cfg.AllowedServiceAccounts {
		if account == "" || account != strings.TrimSpace(account) {
			return nil, fmt.Errorf("authn: NewGoogleOIDC allowed service account %q is empty or has surrounding whitespace", account)
		}
	}
	return &googleOIDC{
		audience: cfg.Audience,
		// Copied so a caller mutating its own slice later cannot change what a
		// running authenticator enforces.
		allowedServiceAccounts: slices.Clone(cfg.AllowedServiceAccounts),
		validate:               idtoken.Validate,
	}, nil
}

// Authenticate implements [Authenticator].
//
// A missing, malformed or unverifiable credential wraps [ErrUnauthenticated],
// so [Middleware] answers 401. A token that verifies but names a principal the
// allow-list does not permit wraps [ErrForbidden] instead, so [Middleware]
// answers 403: the credential was fine, the identity is what is refused.
func (g *googleOIDC) Authenticate(r *http.Request) (*Caller, error) {
	token, err := bearerToken(r.Header.Values("Authorization"))
	if err != nil {
		return nil, deny(err)
	}

	payload, err := g.validateToken(r.Context(), token)
	if err != nil {
		return nil, deny(fmt.Errorf("invalid identity token: %w", err))
	}
	if payload == nil {
		return nil, deny(errors.New("identity token verified with no payload"))
	}
	if !slices.Contains(googleIssuers, payload.Issuer) {
		return nil, deny(fmt.Errorf("untrusted issuer %q", payload.Issuer))
	}

	email, _ := payload.Claims["email"].(string)
	// A non-boolean email_verified fails this assertion and so fails closed.
	emailVerified, _ := payload.Claims["email_verified"].(bool)

	// The allow-list is mandatory and carries no blank entry, since NewGoogleOIDC
	// rejects both, so every accepted caller has a verified email naming a listed
	// account. A token minted without includeEmail carries no email -- and so no
	// entry can match -- and an unverified email is not trusted even when its
	// string is listed; both are refused here rather than admitted.
	if !emailVerified || !slices.Contains(g.allowedServiceAccounts, email) {
		return nil, forbid(fmt.Errorf("principal %q (verified=%t) is not an allowed service account", email, emailVerified))
	}

	// sub distinguishes callers the email cannot: a service account deleted and
	// recreated under the same name keeps its email, and so is still admitted by
	// the allow-list above, but is a separate identity carrying a new sub that
	// Google never reuses. Falling back to the email gives that up, and is here
	// only because nothing in this package requires a sub. A caller that reached
	// here always has an email, since it had to match a non-blank entry.
	userID := payload.Subject
	if userID == "" {
		userID = email
	}

	return &Caller{
		UserID: userID,
		Claims: map[string]any{
			"aud":            payload.Audience,
			"email":          email,
			"email_verified": emailVerified,
		},
	}, nil
}

// validateToken runs the validator and turns a panic into an error, so no
// request can crash the handler. idtoken.Validate checks aud and exp before the
// signature and, for a malformed ES256 signature shorter than 32 bytes, slices
// out of range before it can return an error. net/http would recover from that,
// but the caller would see a dropped connection and the log a stack trace per
// request instead of a clean rejection.
//
// g.validate has no fallback: NewGoogleOIDC sets it, so a nil here means a test
// built the struct without one, and the recover turns the resulting nil call
// into a rejection rather than a crash.
func (g *googleOIDC) validateToken(ctx context.Context, token string) (payload *idtoken.Payload, err error) {
	defer func() {
		if r := recover(); r != nil {
			payload, err = nil, fmt.Errorf("identity token validation panicked: %v", r)
		}
	}()
	return g.validate(ctx, token, g.audience)
}

// deny logs the reason and returns it as a credential problem (a 401). The
// reason stays server-side, so a caller cannot tell an expired token from an
// audience mismatch, nor from an untrusted issuer.
//
// A failure to fetch Google's signing certificates comes back from
// idtoken.Validate as an ordinary error and so is reported here as a 401, not a
// 500. idtoken does not distinguish that failure in a way this can key on, and
// the certificate response is cached, so an outage bites at cold start or after
// expiry rather than on every delivery. Surfacing it as a 500 waits for the
// wiring PR, where the constructor gains an http.Client and can observe the
// transport directly.
func deny(err error) error {
	log.Printf("adk: authn: rejected an OIDC-authenticated request: %v", err)
	return fmt.Errorf("%w: %w", ErrUnauthenticated, err)
}

// forbid logs the reason and returns it as an authorization problem (a 403):
// the token verified, but the principal it named is outside the allow-list. The
// reason stays server-side, so the response body never names the refused
// principal and does not distinguish an unlisted one from an unverified or
// absent email. That body is not the one deny's rejection gets -- Middleware
// writes "forbidden" here and "unauthorized" there -- but it differs only in
// the way the status already does.
func forbid(err error) error {
	log.Printf("adk: authn: refused an OIDC-authenticated principal: %v", err)
	return fmt.Errorf("%w: %w", ErrForbidden, err)
}

// bearerToken extracts the credential from the Authorization headers. RFC 9110
// makes the scheme case-insensitive.
//
// More than one header is rejected rather than resolved, so a proxy in front
// that forwards the last value cannot disagree with this authenticator about
// which credential arrived.
func bearerToken(authHeaders []string) (string, error) {
	if len(authHeaders) != 1 {
		if len(authHeaders) == 0 {
			return "", errors.New("no Authorization header")
		}
		return "", fmt.Errorf("request carries %d Authorization headers", len(authHeaders))
	}
	scheme, token, found := strings.Cut(authHeaders[0], " ")
	if !found || !strings.EqualFold(scheme, "Bearer") {
		return "", errors.New("authorization header is not a Bearer credential")
	}
	if token = strings.TrimSpace(token); token == "" {
		return "", errors.New("bearer credential is empty")
	}
	return token, nil
}

var _ Authenticator = &googleOIDC{}
