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

package remoteagent

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"reflect"
	"strings"
	"sync"
	"time"

	"github.com/a2aproject/a2a-go/v2/a2a"
	"github.com/a2aproject/a2a-go/v2/a2aclient"
	"github.com/a2aproject/a2a-go/v2/log"
	"golang.org/x/oauth2"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/auth"
	iremoteagent "google.golang.org/adk/v2/internal/agent/remoteagent"
	"google.golang.org/adk/v2/session"
)

// CredentialScope returns the scope this package attaches to every outgoing
// call when [A2AConfig.Auth] is set, readable by a provider through
// [a2aclient.SessionIDFrom]. agentName is the remote agent's A2AConfig.Name.
//
// It is exported for the case A2AConfig.Auth cannot serve: a caller combining a
// custom ClientProvider with their own [a2aclient.AuthInterceptor] attaches
// this themselves, and gets a key that cannot collide across users, sessions or
// remote agents.
//
// The parts are percent-encoded and joined with "/". A provider that wants them
// back splits on "/" and calls [url.QueryUnescape] on each.
func CredentialScope(s session.Session, agentName string) a2aclient.SessionID {
	return iremoteagent.CredentialScope(s.AppName(), s.UserID(), s.ID(), agentName)
}

// authContext keeps a context an [agent.InvocationContext] while its
// cancellation and values come from somewhere else. A plain context.WithValue
// would hide the ADK context behind an opaque wrapper, and
// [auth.CredentialProvider] promises a provider can recover that context to
// learn who is acting — including from the deferred cleanup, whose context is
// detached from the invocation and separately bounded.
type authContext struct {
	// The embedded value answers the ADK accessors. Everything
	// context.Context declares is overridden below to read from ctx.
	agent.InvocationContext
	ctx context.Context
}

var _ agent.InvocationContext = authContext{}

func (c authContext) Deadline() (time.Time, bool) { return c.ctx.Deadline() }
func (c authContext) Done() <-chan struct{}       { return c.ctx.Done() }
func (c authContext) Err() error                  { return c.ctx.Err() }
func (c authContext) Value(key any) any           { return c.ctx.Value(key) }

// WithContext implements [agent.InvocationContext]. It carries the credential
// scope and the agent card onto the replacement context, which an arbitrary
// caller-supplied one would not have, and re-wraps so a later type assertion
// still finds an agent.InvocationContext.
func (c authContext) WithContext(ctx context.Context) agent.InvocationContext {
	ctx = c.carry(ctx)
	return authContext{InvocationContext: c.InvocationContext.WithContext(ctx), ctx: ctx}
}

// carry copies the credential scope and the agent card onto ctx, so a context
// derived from this one still resolves a credential.
func (c authContext) carry(ctx context.Context) context.Context {
	if sid, ok := a2aclient.SessionIDFrom(c.ctx); ok {
		ctx = a2aclient.AttachSessionID(ctx, sid)
	}
	if card := iremoteagent.AgentCardFrom(c.ctx); card != nil {
		ctx = iremoteagent.WithAgentCard(ctx, card)
	}
	return ctx
}

// WithICDelta implements [agent.InvocationContext]. A delta carrying a context
// replaces the wrapper's too, so the two cannot end up describing different
// lifetimes.
func (c authContext) WithICDelta(d *agent.InvocationContextDelta) agent.InvocationContext {
	ctx := c.ctx
	if d != nil && d.Context != nil {
		ctx = c.carry(*d.Context)
	}
	return authContext{InvocationContext: c.InvocationContext.WithICDelta(d), ctx: ctx}
}

// authSendContext returns the context for every outgoing call of one
// invocation. With Auth unset it is the invocation context untouched, so a
// caller who never opted in sees no change at all — including no change to the
// context's dynamic type, which reaches the exported ClientProvider hook.
//
// With Auth set this package owns the scope, and overwrites one the caller may
// have attached, because it also owns the interceptor that will read it.
func authSendContext(ctx agent.InvocationContext, cfg A2AConfig, card *a2a.AgentCard) context.Context {
	if cfg.Auth == nil {
		return ctx
	}
	values := a2aclient.AttachSessionID(ctx, CredentialScope(ctx.Session(), cfg.Name))
	values = iremoteagent.WithAgentCard(values, card)
	return authContext{InvocationContext: ctx, ctx: values}
}

// reattachInvocation re-wraps derived so it is still an
// [agent.InvocationContext] when orig was one. context.WithoutCancel and
// context.WithTimeout return their own types, which would otherwise break the
// type assertion the Auth doc invites a provider to make.
func reattachInvocation(orig, derived context.Context) context.Context {
	ic, ok := orig.(agent.InvocationContext)
	if !ok {
		return derived
	}
	return authContext{InvocationContext: ic, ctx: derived}
}

// credentialsService adapts an [auth.CredentialProvider] to
// [a2aclient.CredentialsService]. The a2a AuthInterceptor calls Get and places
// the returned value per the agent card's security scheme — it writes the
// "Bearer " prefix or the API-key header itself — so Get returns the raw secret.
//
// The card, not the credential, decides placement, so Get yields a secret only
// for a scheme that can carry it and returns [a2aclient.ErrCredentialNotFound]
// otherwise, which tells the interceptor to try the next scheme. Without that
// check the interceptor picks among the schemes named in one requirement object
// in Go map order, and a bearer token would land in an API-key header on a
// random subset of requests.
type credentialsService struct {
	provider auth.CredentialProvider
	// warnMismatch fires the "nothing on this card can carry it" warning once.
	// The interceptor asks per scheme per request, and a credential that fits
	// nothing keeps not fitting, so warning every time would bury the operator
	// in duplicates of one fact.
	warnMismatch *sync.Once
}

var _ a2aclient.CredentialsService = credentialsService{}

// Get implements [a2aclient.CredentialsService].
func (s credentialsService) Get(ctx context.Context, _ a2aclient.SessionID, scheme a2a.SecuritySchemeName) (a2aclient.AuthCredential, error) {
	if s.provider == nil {
		return "", errors.New("remoteagent: a2a auth has no credential provider")
	}
	cred, err := s.provider.Credential(ctx)
	if err != nil {
		return "", fmt.Errorf("remoteagent: resolve auth credential: %w", err)
	}
	cred = derefCredential(cred)
	// Reject an untransmittable credential before consulting the card, so the
	// reason reaches the interceptor's log instead of looking like a scheme
	// this session simply has no credential for.
	place, err := credentialPlacement(cred)
	if err != nil {
		return "", err
	}
	card := iremoteagent.AgentCardFrom(ctx)
	if !schemeAccepts(card, scheme, place) {
		// The interceptor swallows this sentinel and moves on, which is right
		// when another scheme can carry the credential — a card may offer
		// alternatives. When none can, the request goes out unauthenticated
		// with nothing said, and this is the only place that can see why.
		if !cardAccepts(card, place) && s.warnMismatch != nil {
			s.warnMismatch.Do(func() {
				log.Warn(ctx, "a2a auth: no security scheme the agent card declares can carry the resolved credential, so the request will go out unauthenticated",
					"credential", fmt.Sprintf("%T", cred))
			})
		}
		return "", a2aclient.ErrCredentialNotFound
	}
	value, err := credentialValue(ctx, cred)
	if err != nil {
		return "", err
	}
	return a2aclient.AuthCredential(value), nil
}

// placement is where the a2a AuthInterceptor would write a credential: into the
// card-named API-key header, or into "Authorization" behind a "Bearer " prefix.
type placement int

const (
	placementNone placement = iota
	placementAPIKey
	placementBearer
)

// credentialPlacement reports how c would be transmitted, and errors for a
// credential this adapter cannot transmit at all.
func credentialPlacement(c auth.Credential) (placement, error) {
	switch c.(type) {
	case nil:
		return placementNone, errors.New("remoteagent: credential provider returned a nil credential")
	case auth.APIKeyCredential:
		return placementAPIKey, nil
	case auth.BearerCredential, auth.OAuth2Credential:
		return placementBearer, nil
	default:
		return placementNone, errUntransmittable(c)
	}
}

func errUntransmittable(c auth.Credential) error {
	return fmt.Errorf("remoteagent: cannot send %T over a2a, where the agent card decides placement; "+
		"only auth.APIKeyCredential, auth.BearerCredential and auth.OAuth2Credential are supported", c)
}

// schemeAccepts reports whether the card's security scheme named name can carry
// a credential written as p. It default-denies: an absent card, or a name the
// card does not declare, yields false, so the interceptor moves on without the
// adapter having minted anything for a scheme that cannot be satisfied.
func schemeAccepts(card *a2a.AgentCard, name a2a.SecuritySchemeName, p placement) bool {
	if card == nil {
		return false
	}
	switch scheme := card.SecuritySchemes[name].(type) {
	case a2a.APIKeySecurityScheme:
		// The interceptor writes the key as a header whatever the card says, so
		// a query- or cookie-located key would go somewhere the card never
		// named and the remote would never read.
		return p == placementAPIKey && scheme.Location == a2a.APIKeySecuritySchemeLocationHeader
	case a2a.HTTPAuthSecurityScheme:
		// The interceptor always writes "Bearer", so a card asking for Basic or
		// any other HTTP scheme would receive a mislabeled credential.
		return p == placementBearer && strings.EqualFold(scheme.Scheme, "bearer")
	case a2a.OAuth2SecurityScheme:
		return p == placementBearer
	default:
		// Mutual TLS, OpenID Connect, and a name the card does not declare:
		// nothing the interceptor knows how to place.
		return false
	}
}

// cardAccepts reports whether any scheme the card requires can carry a
// credential written as p.
func cardAccepts(card *a2a.AgentCard, p placement) bool {
	if card == nil {
		return false
	}
	for _, requirement := range card.SecurityRequirements {
		for name := range requirement {
			if schemeAccepts(card, name, p) {
				return true
			}
		}
	}
	return false
}

// derefCredential unwraps a pointer credential. Every auth.Credential method
// has a value receiver, so *T satisfies the interface too and passing
// &auth.BearerCredential{...} is reasonable. A typed nil pointer is left alone
// and lands in the untransmittable branch instead of panicking.
func derefCredential(c auth.Credential) auth.Credential {
	switch p := c.(type) {
	case *auth.APIKeyCredential:
		if p != nil {
			return *p
		}
	case *auth.BearerCredential:
		if p != nil {
			return *p
		}
	case *auth.OAuth2Credential:
		if p != nil {
			return *p
		}
	}
	return c
}

// credentialValue returns the raw secret the a2a AuthInterceptor transmits: the
// API-key value, the bearer token, or a freshly minted OAuth2 access token.
func credentialValue(ctx context.Context, c auth.Credential) (string, error) {
	switch v := c.(type) {
	case auth.APIKeyCredential:
		if v.Value == "" {
			return "", errors.New("remoteagent: api key credential has an empty value")
		}
		return v.Value, nil
	case auth.BearerCredential:
		if v.Token == "" {
			return "", errors.New("remoteagent: bearer credential has an empty token")
		}
		return v.Token, nil
	case auth.OAuth2Credential:
		return mintAccessToken(ctx, v.TokenSource)
	default:
		return "", errUntransmittable(c)
	}
}

// mintTimeout bounds a token mint when the caller's context has none. The
// runner sets no deadline on an invocation, so without this the send path would
// wait on a hung token endpoint for as long as the whole run. It mirrors the
// auth package's own initTimeout, and is a var so tests need not wait it out.
var mintTimeout = 30 * time.Second

// mintAccessToken returns a fresh access token, bounded by ctx and by
// mintTimeout. [oauth2.TokenSource.Token] takes no context, and real sources —
// the JWT and ADC sources behind auth.ServiceAccount and auth.ADC — post to a
// token endpoint through http.DefaultClient, which has no timeout.
// Interceptors run before the transport call, so nothing else bounds the mint:
// without this it can outlive the invocation and hold the run loop's deferred
// cleanup past the budget that cleanup set for itself.
//
// Only the wait is bounded. Token() cannot be interrupted, so a mint against a
// black-holed endpoint leaves its goroutine parked until the source gives up —
// no worse than the synchronous call it replaces, which parked the caller's own
// goroutine instead, but not something this function can cancel.
func mintAccessToken(ctx context.Context, ts oauth2.TokenSource) (string, error) {
	if ts == nil {
		return "", errors.New("remoteagent: oauth2 credential has no token source")
	}
	ctx, cancel := context.WithTimeout(ctx, mintTimeout)
	defer cancel()

	type result struct {
		tok *oauth2.Token
		err error
	}
	// Buffered, so a mint nobody is waiting for can still complete and let its
	// goroutine exit rather than blocking forever on the send.
	done := make(chan result, 1)
	go func() {
		tok, err := ts.Token()
		done <- result{tok, err}
	}()

	select {
	case <-ctx.Done():
		return "", fmt.Errorf("remoteagent: mint oauth2 token: %w", context.Cause(ctx))
	case r := <-done:
		if r.err != nil {
			return "", fmt.Errorf("remoteagent: mint oauth2 token: %w", redactTokenError(r.err))
		}
		if r.tok == nil || r.tok.AccessToken == "" {
			return "", errors.New("remoteagent: oauth2 token source returned an empty access token")
		}
		// a2a always writes "Bearer", so any other type would go out mislabeled.
		if t := r.tok.Type(); !strings.EqualFold(t, "bearer") {
			return "", fmt.Errorf("remoteagent: oauth2 token type %q cannot be sent over a2a, which always writes a bearer token", t)
		}
		return r.tok.AccessToken, nil
	}
}

// a2aRequestTimeout restates the a2a client's own default. Auth has to build an
// explicit http.Client to install CheckRedirect, and building one drops the
// timeout a2aclient would otherwise have applied.
const a2aRequestTimeout = 3 * time.Minute

// maxRedirects matches net/http's default cap, which a custom CheckRedirect
// replaces rather than extends.
const maxRedirects = 10

// authHTTPClient is the HTTP client used when Auth is set. It refuses a
// redirect that leaves the card's host or downgrades its scheme: the credential
// is attached before the first hop and Go replays request headers on every hop.
// Go strips Authorization only when the host changes, and never strips a
// card-named API-key header, so nothing else bounds where the secret travels.
func authHTTPClient() *http.Client {
	return &http.Client{Timeout: a2aRequestTimeout, CheckRedirect: checkRedirect}
}

// checkRedirect is authHTTPClient's redirect policy. A custom CheckRedirect
// replaces net/http's default rather than extending it, so the hop cap is here
// too.
func checkRedirect(req *http.Request, via []*http.Request) error {
	if len(via) >= maxRedirects {
		return fmt.Errorf("remoteagent: stopped after %d redirects", maxRedirects)
	}
	// Check against the original request and against the hop just taken. The
	// original stops a chain of individually-safe hops laundering the
	// credential away from the card's host; the previous hop stops a chain
	// climbing to https and then dropping back to http, which the original
	// alone would accept.
	for _, from := range []*url.URL{via[0].URL, via[len(via)-1].URL} {
		if !sameCredentialTarget(from, req.URL) {
			return fmt.Errorf("remoteagent: refusing redirect from %s to %s: A2AConfig.Auth is set and the credential would follow",
				requestOrigin(from), requestOrigin(req.URL))
		}
	}
	return nil
}

// sameCredentialTarget reports whether it is safe to replay the credential on a
// redirect from one URL to the other: the host must be identical, and the
// scheme may only stay the same or be upgraded from http to https, which
// strictly improves confidentiality.
func sameCredentialTarget(from, to *url.URL) bool {
	if !strings.EqualFold(from.Hostname(), to.Hostname()) {
		return false
	}
	switch {
	case strings.EqualFold(from.Scheme, to.Scheme):
		return defaultedPort(from) == defaultedPort(to)
	case strings.EqualFold(from.Scheme, "http") && strings.EqualFold(to.Scheme, "https"):
		// The default port differs between the two schemes, so an upgrade
		// cannot be judged by comparing defaulted ports. It keeps the same
		// endpoint when both sides use their own scheme's default, or when both
		// name the same explicit port.
		return isDefaultPort(from) && isDefaultPort(to) || from.Port() == to.Port()
	default:
		return false
	}
}

// defaultedPort returns the URL's port with the scheme's default filled in, so
// that https://example.com and https://example.com:443 compare equal.
func defaultedPort(u *url.URL) string {
	if port := u.Port(); port != "" {
		return port
	}
	switch strings.ToLower(u.Scheme) {
	case "https":
		return "443"
	case "http":
		return "80"
	}
	return ""
}

// isDefaultPort reports whether the URL leaves its scheme's port implicit or
// names the default explicitly.
func isDefaultPort(u *url.URL) bool {
	port := u.Port()
	if port == "" {
		return true
	}
	return port == defaultedPort(&url.URL{Scheme: u.Scheme})
}

func requestOrigin(u *url.URL) string { return u.Scheme + "://" + u.Host }

// redactedError reports msg while keeping cause reachable through errors.Is and
// errors.As. Wrapping with %w would put the cause's own text back in the
// message, which is the thing being redacted.
type redactedError struct {
	msg   string
	cause error
}

func (e *redactedError) Error() string { return e.msg }
func (e *redactedError) Unwrap() error { return e.cause }

// redactTokenError strips the token endpoint's verbatim response body from an
// [oauth2.RetrieveError]. The a2a interceptor logs whatever this adapter
// returns at ERROR level, and what an identity provider echoes into a non-2xx
// body is outside our control — some reflect the request back.
//
// Only the branch that carries the body is rewritten: RetrieveError.Error()
// prints the body exactly when the response was not a well-formed OAuth2 error,
// which is also when its content is least predictable. A response that did
// name an error code already prints only that, and is left alone. The status
// and the error chain survive either way.
func redactTokenError(err error) error {
	var re *oauth2.RetrieveError
	if !errors.As(err, &re) || re.ErrorCode != "" || re.Response == nil {
		return err
	}
	return &redactedError{
		msg:   "oauth2: cannot fetch token: " + re.Response.Status + " (response body redacted)",
		cause: err,
	}
}

// isTypedNil reports whether v is a nil pointer, func or map held in a non-nil
// interface — auth.ProviderFunc(nil), say. Such a value passes an ordinary
// != nil check and then panics on the first call.
func isTypedNil(v any) bool {
	switch rv := reflect.ValueOf(v); rv.Kind() {
	case reflect.Chan, reflect.Func, reflect.Map, reflect.Pointer, reflect.Slice, reflect.UnsafePointer:
		return rv.IsNil()
	default:
		return false
	}
}
