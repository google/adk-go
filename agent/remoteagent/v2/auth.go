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
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/a2aproject/a2a-go/v2/a2a"
	"github.com/a2aproject/a2a-go/v2/a2aclient"
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
//
// s must be non-nil; a nil session panics. There is no safe value to return
// instead, because every scope built without an identity is the same scope, and
// two identities sharing one key is what this function exists to prevent.
func CredentialScope(s session.Session, agentName string) a2aclient.SessionID {
	return iremoteagent.CredentialScope(s.AppName(), s.UserID(), s.ID(), agentName)
}

// authContext keeps a context an [agent.InvocationContext] while its
// cancellation and values come from somewhere else. A plain context.WithValue
// would keep the values — agent.IdentityFromContext, which the
// [auth.CredentialProvider] contract tells a provider to use, would still work
// — but it would hide the invocation behind an opaque type, and the Auth doc
// lets a provider type-assert its context, as RemoteTaskCleanupCallback's doc
// lets that callback.
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

// WithContext implements [agent.InvocationContext]. It carries what the auth
// transport reads onto the replacement context, which an arbitrary
// caller-supplied one would not have, and re-wraps so a later type assertion
// still finds an agent.InvocationContext.
func (c authContext) WithContext(ctx context.Context) agent.InvocationContext {
	ctx = c.carry(ctx)
	return authContext{InvocationContext: c.InvocationContext.WithContext(ctx), ctx: ctx}
}

// carry copies what the auth transport reads — the credential scope, the
// invocation's resolved credential and the invocation itself — onto ctx, so a
// context derived from this one still resolves a credential.
func (c authContext) carry(ctx context.Context) context.Context {
	if sid, ok := a2aclient.SessionIDFrom(c.ctx); ok {
		ctx = a2aclient.AttachSessionID(ctx, sid)
	}
	if cell, ok := c.ctx.Value(credentialCellKey{}).(*credentialCell); ok {
		ctx = context.WithValue(ctx, credentialCellKey{}, cell)
	}
	if ic, ok := c.ctx.Value(invocationKey{}).(agent.InvocationContext); ok {
		ctx = context.WithValue(ctx, invocationKey{}, ic)
	}
	if client := iremoteagent.CardFetchClientFrom(c.ctx); client != nil {
		ctx = iremoteagent.WithCardFetchClient(ctx, client)
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
// invocation, the agent card fetch included. With Auth unset it is the
// invocation context untouched, so a caller who never opted in sees no change
// at all — including no change to the context's dynamic type, which reaches the
// exported ClientProvider hook.
//
// With Auth set this package owns the scope, and overwrites one the caller may
// have attached, because it also owns the transport that will read it.
func authSendContext(ctx agent.InvocationContext, cfg A2AConfig, client *http.Client) context.Context {
	if cfg.Auth == nil {
		return ctx
	}
	values := a2aclient.AttachSessionID(ctx, CredentialScope(ctx.Session(), cfg.Name))
	values = context.WithValue(values, credentialCellKey{}, &credentialCell{})
	values = context.WithValue(values, invocationKey{}, ctx)
	values = iremoteagent.WithCardFetchClient(values, client)
	return authContext{InvocationContext: ctx, ctx: values}
}

// reattachInvocation re-wraps derived so it is still an
// [agent.InvocationContext] when orig was one. Its caller is the cleanup, whose
// context.WithoutCancel returns its own type, and RemoteTaskCleanupCallback's
// doc promises that callback the invocation context. The credential provider
// does not depend on it: the auth transport recovers the invocation from the
// context's values whatever type the context has.
func reattachInvocation(orig, derived context.Context) context.Context {
	ic, ok := orig.(agent.InvocationContext)
	if !ok {
		return derived
	}
	return authContext{InvocationContext: ic, ctx: derived}
}

// authTransport applies the credential A2AConfig.Auth resolves to every request
// the A2A client sends, and to the agent card fetch.
//
// The caller's credential decides where it goes, not the agent card: this
// matches adk-python, whose RemoteA2aAgent writes the header its configured
// auth scheme names and never reads the card's security section. Every
// credential but a bare OAuth2 one writes itself through its own Apply, as
// mcptoolset.Config.Auth applies a credential through auth.Transport. A bare
// OAuth2 credential is minted here instead and sent as a bearer token, see
// apply. So every credential type works, and a card that declares no security
// still gets the credential.
//
// A credential that cannot be resolved or applied fails the request rather than
// letting it go out unauthenticated. adk-python fails closed too, by pausing
// the invocation to ask for consent, which this package does not do yet.
type authTransport struct {
	provider auth.CredentialProvider
	mints    *mintGroup[string]
	applies  *mintGroup[http.Header]
	base     http.RoundTripper
}

var _ http.RoundTripper = (*authTransport)(nil)

// RoundTrip implements [http.RoundTripper].
func (t *authTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	// The RoundTripper contract makes this function responsible for the body on
	// every early return. On success the base transport owns it.
	bodyOwned := req.Body != nil
	defer func() {
		if bodyOwned {
			_ = req.Body.Close()
		}
	}()

	ctx := providerContext(req.Context())
	cred, err := t.credential(ctx)
	if err != nil {
		return nil, err
	}
	out := req.Clone(req.Context())
	if err := t.apply(ctx, cred, out.Header); err != nil {
		return nil, err
	}
	bodyOwned = false
	return t.base.RoundTrip(out)
}

// credential resolves the credential once per invocation and reuses it for
// every request that invocation makes, as adk-python does with its
// per-invocation credential cache. A request with no invocation behind it — the
// cancel the adka2a server issues for an abandoned child task — resolves its
// own. A failure is not cached, so the next request tries again.
func (t *authTransport) credential(ctx context.Context) (auth.Credential, error) {
	cell, _ := ctx.Value(credentialCellKey{}).(*credentialCell)
	if cred := cell.get(); cred != nil {
		return cred, nil
	}
	cred, err := t.provider.Credential(ctx)
	if err != nil {
		return nil, fmt.Errorf("remoteagent: resolve auth credential: %w", err)
	}
	if cred == nil || isTypedNil(cred) {
		return nil, errors.New("remoteagent: credential provider returned a nil credential")
	}
	cell.set(cred)
	return cred, nil
}

// apply writes cred onto h. Both paths run the blocking step on a bounded,
// single-flighted goroutine, because TokenSource.Token takes no context and so
// could otherwise hold the request past every deadline around it. A bare OAuth2
// credential is minted directly, so its token is checked and sent as a bearer
// token. Anything else — a credential wrapping an OAuth2 one included, such as
// auth.WithHeaders — writes itself through its own Apply into a private header,
// copied onto h only once the step has landed, so a caller released by its
// deadline never races the writer.
func (t *authTransport) apply(ctx context.Context, cred auth.Credential, h http.Header) error {
	// The scope keys the single-flight. Every request this package sends
	// carries one, so this fallback is not reached today. It keeps a request
	// that one day arrives without a scope from sharing a step, and so a
	// credential, with every other unscoped request.
	scope, ok := a2aclient.SessionIDFrom(ctx)
	mints, applies := t.mints, t.applies
	if !ok {
		mints, applies = newMintGroup(), newApplyGroup()
	}

	var ts oauth2.TokenSource
	isOAuth2 := true
	switch c := cred.(type) {
	case auth.OAuth2Credential:
		ts = c.TokenSource
	case *auth.OAuth2Credential:
		ts = c.TokenSource
	default:
		isOAuth2 = false
	}
	if isOAuth2 {
		token, err := mints.token(ctx, scope, ts)
		if err != nil {
			return err
		}
		h.Set("Authorization", "Bearer "+token)
		return nil
	}

	written, err := applies.do(ctx, scope, func() (http.Header, error) {
		out := http.Header{}
		if err := cred.Apply(out); err != nil {
			return nil, fmt.Errorf("remoteagent: apply auth credential: %w", redactTokenError(err))
		}
		return out, nil
	})
	if err != nil {
		return err
	}
	for k, v := range written {
		h[k] = append([]string(nil), v...)
	}
	return nil
}

// credentialCell holds the credential one invocation resolved. A nil cell
// holds nothing and ignores writes, which is what a request made outside an
// invocation gets.
type credentialCell struct {
	mu   sync.Mutex
	cred auth.Credential
}

type credentialCellKey struct{}

func (c *credentialCell) get() auth.Credential {
	if c == nil {
		return nil
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.cred
}

func (c *credentialCell) set(cred auth.Credential) {
	if c == nil {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	c.cred = cred
}

type invocationKey struct{}

// providerContext returns ctx as the provider should see it: still an
// agent.InvocationContext when the call was made on behalf of one, which the
// auth.CredentialProvider contract lets a provider rely on. A transport sees
// whatever context the A2A client built the request with, and interceptors or
// a context.WithTimeout on the way down can hide the invocation behind their
// own type, so it is recovered from the value authSendContext attached.
func providerContext(ctx context.Context) context.Context {
	if _, ok := ctx.(agent.InvocationContext); ok {
		return ctx
	}
	if ic, ok := ctx.Value(invocationKey{}).(agent.InvocationContext); ok {
		return authContext{InvocationContext: ic, ctx: ctx}
	}
	return ctx
}

// cardSendsInClear returns every interface the card names that would put the
// credential on the wire unencrypted. It is not a refusal: a card can only be
// trusted as far as its source, and a caller who points Auth at a plaintext
// internal host has said so deliberately. Without a signal, though, that is
// indistinguishable from not having noticed.
//
// Loopback is not reported. A credential that never leaves the machine is not
// exposed by the absence of TLS, and refusing it would rule out every local
// test server. The rule matches validateCardInterfaceOrigins, which enforces it
// on the one card source that can be checked at fetch time.
func cardSendsInClear(card *a2a.AgentCard) []string {
	if card == nil {
		return nil
	}
	var clear []string
	for _, iface := range card.SupportedInterfaces {
		if iface == nil {
			continue
		}
		u, err := url.Parse(iface.URL)
		if err != nil {
			continue
		}
		if !strings.EqualFold(u.Scheme, "https") && !isLoopbackHost(u.Hostname()) {
			clear = append(clear, iface.URL)
		}
	}
	return clear
}

// mintTimeout bounds a token mint when the caller's context has none. The
// runner sets no deadline on an invocation, so without this the send path would
// wait on a hung token endpoint for as long as the whole run. It mirrors the
// auth package's own initTimeout, and is a var so tests need not wait it out.
var mintTimeout = 30 * time.Second

// mintGroup collapses concurrent requests for one credential scope onto a
// single blocking credential step — an OAuth2 token mint, or a credential's own
// Apply — and hands the result to everyone waiting on it. A second step for the
// same scope starts only once the current attempt's deadline has passed, see
// mintCall.
//
// The step has to run in its own goroutine, because [oauth2.TokenSource.Token]
// takes no context and so cannot be interrupted, and a credential's Apply can
// call it: only the caller's wait can be bounded. Releasing the caller is what
// makes the single-flight necessary. Without it, every request arriving while a
// token endpoint hangs starts another step and parks another goroutine, and
// neither ever ends.
//
// The scope is the key because it is already the per-identity credential key:
// a provider resolves one credential for one scope, so two steps under the same
// scope are the same step. A provider that returns a different credential per
// call for one scope would see the first one's result answer both, which is why
// the field doc asks for a credential that depends only on the scope.
type mintGroup[T any] struct {
	// what names the step in errors, e.g. "mint oauth2 token".
	what     string
	mu       sync.Mutex
	inFlight map[a2aclient.SessionID]*mintCall[T]
}

// mintCall is one in-flight step. value and err are written once, before done
// closes, and read only after it.
//
// deadline is the attempt's, not any one waiter's: a caller arriving midway
// through a stuck step waits out what is left of it rather than arming a fresh
// mintTimeout of its own. auth/gcp's provider bounds its waiters the same way.
//
// Past the deadline the two pick opposite costs for the same problem: a step
// that may never return. Token() takes no context, and the JWT and ADC sources
// post through http.DefaultClient, which has no timeout. auth/gcp keeps its hung
// attempt for the rest of the process and fails every later call, accepting a
// permanent lockout so that it never starts uncancellable lookups on a timer —
// its ErrClientUnavailable doc calls that the deliberate trade. This group
// retires the attempt and starts a new step, accepting that while a token
// endpoint hangs one more goroutine per scope stays parked every mintTimeout,
// so that a hang never locks an identity out after the endpoint recovers.
type mintCall[T any] struct {
	done     chan struct{}
	deadline time.Time
	value    T
	err      error
}

// newMintGroup returns the group for OAuth2 token mints.
func newMintGroup() *mintGroup[string] {
	return &mintGroup[string]{what: "mint oauth2 token", inFlight: map[a2aclient.SessionID]*mintCall[string]{}}
}

// newApplyGroup returns the group for a credential's own Apply.
func newApplyGroup() *mintGroup[http.Header] {
	return &mintGroup[http.Header]{what: "apply auth credential", inFlight: map[a2aclient.SessionID]*mintCall[http.Header]{}}
}

// token returns a fresh access token for scope, bounded by ctx and by
// mintTimeout. Real sources — the JWT and ADC sources behind
// auth.ServiceAccount and auth.ADC — post to a token endpoint through
// http.DefaultClient, which has no timeout. The mint runs inside RoundTrip and
// Token takes no context, so neither the request's context nor the client's
// timeout can interrupt it: without mintTimeout it can outlive the invocation
// and hold the run loop's deferred cleanup past the budget that cleanup set for
// itself.
func (g *mintGroup[T]) token(ctx context.Context, scope a2aclient.SessionID, ts oauth2.TokenSource) (T, error) {
	if ts == nil {
		var zero T
		return zero, errors.New("remoteagent: oauth2 credential has no token source")
	}
	return g.do(ctx, scope, func() (T, error) {
		tok, err := mintAccessToken(ts)
		v, _ := any(tok).(T)
		return v, err
	})
}

// do runs step for scope, joining one already in flight, and waits for it
// bounded by ctx and by the attempt's deadline.
func (g *mintGroup[T]) do(ctx context.Context, scope a2aclient.SessionID, step func() (T, error)) (T, error) {
	var zero T
	now := time.Now()
	g.mu.Lock()
	call, joined := g.inFlight[scope]
	if joined && !now.Before(call.deadline) {
		// The attempt has spent its budget. Retire it so this caller starts
		// afresh instead of waiting on one that is already over time; its
		// goroutine still publishes to whoever is on it.
		delete(g.inFlight, scope)
		joined = false
	}
	if !joined {
		call = &mintCall[T]{done: make(chan struct{}), deadline: now.Add(mintTimeout)}
		g.inFlight[scope] = call
		go g.run(scope, call, step)
	}
	g.mu.Unlock()

	// A result that has already landed beats an expired bound. This narrows the
	// race rather than closing it, so the other two arms re-check as well: when
	// two cases are ready at once Go picks between them at random, and
	// discarding a result that did arrive would fail the request for no reason.
	if done, v, err := call.result(); done {
		return v, err
	}
	timer := time.NewTimer(call.deadline.Sub(now))
	defer timer.Stop()
	select {
	case <-call.done:
		return call.value, call.err
	case <-ctx.Done():
		if done, v, err := call.result(); done {
			return v, err
		}
		return zero, fmt.Errorf("remoteagent: %s: %w", g.what, context.Cause(ctx))
	case <-timer.C:
		if done, v, err := call.result(); done {
			return v, err
		}
		return zero, fmt.Errorf("remoteagent: %s: %w", g.what, context.DeadlineExceeded)
	}
}

// result reports the step's outcome if it has landed, without blocking.
func (c *mintCall[T]) result() (bool, T, error) {
	select {
	case <-c.done:
		return true, c.value, c.err
	default:
		var zero T
		return false, zero, nil
	}
}

// run performs one step and publishes its outcome. The entry is removed and
// done closed under the lock, so a caller either joins this call and is woken
// by it or finds no entry and starts a fresh one — a failed step is never
// replayed to a later request. The removal is conditional because an attempt
// that ran past its deadline has already been retired, and a successor may hold
// the entry by now.
func (g *mintGroup[T]) run(scope a2aclient.SessionID, call *mintCall[T], step func() (T, error)) {
	defer func() {
		// The step is third-party code on a goroutine of our own, where a panic
		// is fatal rather than something the runner's recover can turn into an
		// error. auth/gcp's provider guards its background lookup the same way.
		if r := recover(); r != nil {
			call.err = fmt.Errorf("remoteagent: %s: credential panicked: %v", g.what, r)
		}
		g.mu.Lock()
		if g.inFlight[scope] == call {
			delete(g.inFlight, scope)
		}
		close(call.done)
		g.mu.Unlock()
	}()
	call.value, call.err = step()
}

// mintAccessToken reads one access token from ts and checks it can be sent.
func mintAccessToken(ts oauth2.TokenSource) (string, error) {
	tok, err := ts.Token()
	if err != nil {
		return "", fmt.Errorf("remoteagent: mint oauth2 token: %w", redactTokenError(err))
	}
	if tok == nil || tok.AccessToken == "" {
		return "", errors.New("remoteagent: oauth2 token source returned an empty access token")
	}
	// The token is written as a bearer token, as adk-python writes it, so any
	// other type would go out mislabeled.
	if t := tok.Type(); !strings.EqualFold(t, "bearer") {
		return "", fmt.Errorf("remoteagent: oauth2 token type %q cannot be sent as a bearer token", t)
	}
	return tok.AccessToken, nil
}

// a2aRequestTimeout restates the a2a client's own default. Auth has to build an
// explicit http.Client to install CheckRedirect, and building one drops the
// timeout a2aclient would otherwise have applied.
const a2aRequestTimeout = 3 * time.Minute

// maxRedirects matches net/http's default cap, which a custom CheckRedirect
// replaces rather than extends.
const maxRedirects = 10

// authHTTPClient is the HTTP client used when Auth is set. It refuses a
// redirect that leaves the card's host or downgrades its scheme, because the
// transport applies the credential again on every hop, so nothing else bounds
// where the secret travels. adk-python's HTTP client follows no redirects at
// all by default; this one still follows a same-origin redirect.
func authHTTPClient(provider auth.CredentialProvider) *http.Client {
	return &http.Client{
		Timeout:       a2aRequestTimeout,
		CheckRedirect: checkRedirect,
		Transport:     &authTransport{provider: provider, mints: newMintGroup(), applies: newApplyGroup(), base: http.DefaultTransport},
	}
}

// cardFetchTimeout restates the bound a2a-go's default card resolver applies.
// Authenticating the fetch means supplying a client of our own, which would
// otherwise carry the three-minute RPC timeout instead.
const cardFetchTimeout = 30 * time.Second

// cardFetchHTTPClient is authHTTPClient with the card resolver's timeout: the
// same transport and redirect policy, so the fetch carries the credential under
// the same rules as every other call.
func cardFetchHTTPClient(rpc *http.Client) *http.Client {
	return &http.Client{Timeout: cardFetchTimeout, CheckRedirect: rpc.CheckRedirect, Transport: rpc.Transport}
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
// [oauth2.RetrieveError]. The error reaches the invocation's error event and
// any log that records it, and what an identity provider writes into a non-2xx
// body is outside our control.
//
// It never reads the wrapper's message to decide. Deciding from the inner
// error's fields was the first thing tried and it fails: the token source
// behind auth.ServiceAccount with an Audience returns a
// cloud.google.com/go/auth *Error that prints the body itself, wrapping an
// *oauth2.RetrieveError whose ErrorCode the adapter has already parsed out of
// that body — so a fields-based check finds a named code, declines, and lets
// the wrapper print the body anyway. Searching the message for the body fixes
// that one but only for a wrapper that prints the body verbatim: one that
// quotes or re-encodes it would slip past.
//
// So a response that carried a body always gets a message built here, out of
// the status and the error code, and nothing is copied from whatever the
// wrapper wrote. The code is parsed out of the same body, but the OAuth2 spec
// defines it as a single short ASCII code from a registered set, such as
// invalid_grant, and it is the one field a caller needs to tell an expired
// grant from a misconfiguration. error_description
// and error_uri are free text and are dropped with the body. The full error
// stays reachable through the chain for a caller that wants it — only what
// gets logged is rebuilt.
func redactTokenError(err error) error {
	var re *oauth2.RetrieveError
	if !errors.As(err, &re) || re.Response == nil || len(re.Body) == 0 {
		return err
	}
	msg := "oauth2: cannot fetch token: " + re.Response.Status + " (response body redacted)"
	if re.ErrorCode != "" {
		msg += ": " + strconv.Quote(re.ErrorCode)
	}
	return &redactedError{msg: msg, cause: err}
}

// isTypedNil reports whether v is a nil func or a nil pointer held in a non-nil
// interface — auth.ProviderFunc(nil), say. Such a value passes an ordinary
// != nil check, and calling it panics for a nil func always and for a nil
// pointer as soon as the method touches its receiver, which in practice it
// does. Rejecting the rare pointer that would have worked costs a caller one
// clear constructor error.
//
// Only those two kinds. A value receiver on a nil named map, slice or channel
// can be called, and whether it then works depends on what it does — reading a
// nil map is fine, indexing a nil slice panics and receiving from a nil channel
// blocks. No check on the kind can tell those apart, and an empty non-nil slice
// fails the same way, so none of the three is rejected.
func isTypedNil(v any) bool {
	switch rv := reflect.ValueOf(v); rv.Kind() {
	case reflect.Func, reflect.Pointer:
		return rv.IsNil()
	default:
		return false
	}
}
