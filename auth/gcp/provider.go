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

package gcp

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"slices"
	"strconv"
	"sync"
	"time"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/auth"
	"google.golang.org/adk/v2/platform"
)

// ProviderScheme identifies a GCP auth resource and the access it requests. It
// mirrors adk-python's GcpAuthProviderScheme.
type ProviderScheme struct {
	// Name is the full resource name, routed by [Client]: either
	// projects/*/locations/*/connectors/* (IAM Connector) or
	// projects/*/locations/*/authProviders/* (Agent Identity).
	//
	// [NewProvider] accepts only those two shapes. That is stricter than
	// [Client.RetrieveCredential] and than adk-python, both of which send any
	// non-connector name to Agent Identity: at wiring time a name outside the two
	// is a typo, and this type's name invites passing an HTTP auth scheme.
	Name string
	// Scopes are the OAuth scopes requested for the credential.
	Scopes []string
	// ContinueURI is the developer-hosted URI used to finalize managed-OAuth
	// (3-legged) flows. Non-interactive flows do not use it.
	ContinueURI string
}

// ProviderConfig configures a provider built by [NewProvider].
type ProviderConfig struct {
	// Scheme is the resource to mint credentials for. Required.
	Scheme ProviderScheme
	// Client reaches the credential services. When nil, a default client backed
	// by Application Default Credentials is built lazily on first use, against
	// the production endpoints. That adds two failure modes to Credential, since
	// the build can fail or exceed its 30s bound. Pass a [Client] from
	// [NewClient] to reach another endpoint or to tune the poll timeout — note
	// that this trades the lazy path away: unless the Client carries its own
	// HTTPClient, [NewClient] discovers Application Default Credentials
	// synchronously, so the cost and any failure move to startup, where they can
	// at least be reported.
	Client *Client
	// Store caches resolved credentials across requests. When nil, a private
	// in-memory store is used. Caching matters here because each miss is a
	// network round-trip (and up to a ~10s pending poll) to the credential
	// service.
	//
	// One store can safely back several providers: entries are keyed by the app,
	// the end user, and everything that decides what the service returns — the
	// resource, the scopes, the continue URI, and the Client. Providers whose
	// Clients authenticate as different identities do not share entries.
	//
	// The cost of caching is staleness. A credential revoked before it expires
	// keeps being served until the cached entry does, which is the service's
	// expiry or an hour, whichever comes first. Call
	// [auth.CredentialStore.Delete] to invalidate one sooner.
	Store auth.CredentialStore
}

// ErrClientUnavailable means the default Application Default Credentials client
// is not available: discovery failed, or it did not finish inside the bound. The
// lookup is not cancellable, so that bound is on the wait rather than on the
// attempt, which keeps running — a later call may well succeed.
//
// That holds for a lookup that eventually returns. One that never does is
// terminal: the attempt is cleared only when it finishes, so this provider keeps
// exactly one for the rest of the process and every later call fails here
// immediately. Retiring a running attempt instead would start a fresh
// uncancellable lookup on a timer, which is worse, so this is the deliberate
// trade rather than an oversight.
//
// The count is per provider, not per process — pending is a provider field. Build
// the provider once at wiring time and one hung lookup costs one parked
// goroutine. Build one per request and the cost is per request.
var ErrClientUnavailable = errors.New("gcp: default credentials client unavailable")

// ErrNoActingUser means the provider could not determine the acting end user,
// either because the context is not an ADK context or because the invocation
// carries no user. Unlike adk-python, which degrades such a turn into an auth
// request, the Go provider fails the request: no user, no credential.
var ErrNoActingUser = errors.New("gcp: no acting user")

// defaultInitTimeout bounds how long a caller waits for the default client. The
// build itself cannot be bounded by a context — FindDefaultCredentials reads the
// credentials file with os.ReadFile and probes with the context-free
// metadata.OnGCE(), neither of which observes cancellation — so the bound lives
// on the waiting side.
const defaultInitTimeout = 30 * time.Second

// NewProvider returns an [auth.CredentialProvider] that resolves credentials for
// cfg.Scheme via the Agent Identity / IAM Connector services.
//
// The acting user is taken from the ADK context ([agent.IdentityFromContext]) at
// resolve time, so the provider must run within an agent invocation. Three
// requirements then fall on the transport that carries the authenticated
// requests and on the context they run under, none of which this package can
// enforce:
//
//   - Every request must descend from the invoking user's context. A transport
//     that shares one connection across invocations does not qualify —
//     mcptoolset included: its per-call POSTs are per-user, but the MCP session
//     and the standalone server-to-client stream are opened on the context of
//     whichever user connected first and stay bound to it.
//   - The transport must not follow a cross-host redirect. net/http strips
//     Authorization above the RoundTripper, so [auth.Transport] re-resolves and
//     re-applies the end user's credential to the redirect target. Set
//     CheckRedirect on the http.Client that carries the transport. The
//     ADC-backed client [NewClient] builds for itself does the same.
//   - Requests must run under a context ADK derived, and an
//     agent.InvocationContext or agent.Context implemented outside the ADK module
//     must override every method on those interfaces that returns one of them.
//     Otherwise it is dropped by its own promoted methods and the credential is
//     minted for the invocation it wraps. Deriving it is necessary and is not
//     sufficient on its own: agent.Run applies a delta on every run, and the
//     workflow schedulers call WithAgentCancel and WithAgentTimeout on the
//     context a node was handed.
//     [agent.IdentityFromContext] states the limits in full.
//
// Wiring this up also means trusting the embedding server: ADK does not
// authenticate session.UserID, and it now decides whose credential is minted.
//
// ctx is used only to build the default client, and only for its values. Its
// cancellation is not honored, because that client outlives any one request.
// Pass the process-scoped context the rest of the app is wired with, not a
// request's. It is ignored when cfg.Client is set.
func NewProvider(ctx context.Context, cfg ProviderConfig) (auth.CredentialProvider, error) {
	if cfg.Scheme.Name == "" {
		return nil, errors.New("gcp: NewProvider requires a scheme Name")
	}
	// A malformed name is a wiring mistake, so catch it here rather than on every
	// request from inside a transport. Stricter than RetrieveCredential, which
	// routes any non-connector name to Agent Identity: at wiring time a name that
	// is not one of the two known shapes is a typo, not a new collection.
	if err := validateResource(cfg.Scheme.Name); err != nil {
		return nil, fmt.Errorf("gcp: NewProvider: %w", err)
	}
	if !connectorResourceRE.MatchString(cfg.Scheme.Name) && !authProviderResourceRE.MatchString(cfg.Scheme.Name) {
		return nil, fmt.Errorf("gcp: NewProvider: scheme Name %q is neither projects/*/locations/*/connectors/* nor projects/*/locations/*/authProviders/*", cfg.Scheme.Name)
	}
	// A zero Client would nil-deref deep inside net/http on first use.
	if cfg.Client != nil && cfg.Client.httpClient == nil {
		return nil, errors.New("gcp: ProviderConfig.Client must come from NewClient")
	}
	store := cfg.Store
	if store == nil {
		store = auth.NewInMemoryCredentialStore()
	}
	p := &provider{
		scheme:      cfg.Scheme,
		client:      cfg.Client,
		store:       store,
		newClient:   func(ctx context.Context) (*Client, error) { return NewClient(ctx, nil) },
		initTimeout: defaultInitTimeout,
	}
	// The provider outlives this call and re-reads Scopes per request, so it must
	// not alias a caller-mutable slice.
	p.scheme.Scopes = slices.Clone(cfg.Scheme.Scopes)
	if cfg.Client == nil {
		// Captured only where it will be used: it is retained for the life of the
		// provider, and pinning a caller's context graph for nothing is a leak.
		p.initCtx = context.WithoutCancel(ctx)
	}
	return p, nil
}

// maxCachedLifetime caps how long a credential is cached, whatever expiry the
// service reports, so a bad or injected expireTime cannot pin one indefinitely.
const maxCachedLifetime = time.Hour

// cacheSlot is the store slot a credential for s, minted through c, is cached
// under. It covers everything that decides what comes back: the client (which
// fixes both the service asked and the identity the ask is authenticated as),
// the resource, the scopes, and the continue URI. A store shared by several
// providers would otherwise serve one provider's credential to another —
// the broad token to a read-only provider, or one caller identity's token to a
// second identity's provider.
//
// The components are length-prefixed before hashing, so no combination of
// delimiters inside a scope or a URI can collide two different schemes. Sorting
// the scopes makes the slot independent of the caller's ordering. adk-python
// keys its credential service the same way, on a SHA-256 digest of a canonical
// encoding of the whole scheme plus the credential used to obtain it.
func cacheSlot(c *Client, s ProviderScheme) string {
	scopes := slices.Clone(s.Scopes)
	slices.Sort(scopes)
	fields := []string{c.cacheSlot, s.Name, strconv.Itoa(len(scopes))}
	fields = append(fields, scopes...)
	fields = append(fields, s.ContinueURI)
	sum := sha256.Sum256([]byte(joinFields(fields...)))
	return hex.EncodeToString(sum[:])
}

type provider struct {
	scheme ProviderScheme
	store  auth.CredentialStore
	// initCtx roots the lazily built default client, which is why NewProvider
	// asks for a process-scoped context. Nil when a Client was supplied.
	initCtx context.Context
	// newClient and initTimeout are fields, not package constants, so tests can
	// drive the failure and hang paths a real ADC lookup cannot be made to hit.
	newClient   func(context.Context) (*Client, error)
	initTimeout time.Duration

	mu      sync.Mutex
	client  *Client
	pending *clientInit // in-flight lazy init, shared by concurrent callers
}

// clientInit is one attempt at building the default client. Its fields are
// written by the attempt's own goroutine and read by waiters only after done is
// closed.
type clientInit struct {
	done   chan struct{}
	client *Client
	err    error
	// deadline bounds this attempt, and every waiter shares it rather than
	// starting a bound of its own on arrival. The attempt is kept running when it
	// expires, so a per-waiter bound would cost each request that arrived inside
	// the window a full initTimeout of its own. Sharing releases the whole cohort
	// at once, and a caller arriving after it has passed fails immediately.
	deadline time.Time
}

// result reports what the attempt produced. Safe only after done is closed.
func (in *clientInit) result() (*Client, error) {
	if in.err != nil {
		return nil, in.err
	}
	return in.client, nil
}

var _ auth.CredentialProvider = (*provider)(nil)

// Credential implements [auth.CredentialProvider].
func (p *provider) Credential(ctx context.Context) (auth.Credential, error) {
	id, ok := agent.IdentityFromContext(ctx)
	if !ok {
		return nil, fmt.Errorf("%w: no ADK invocation identity on the context — not an agent invocation, or its session is unset", ErrNoActingUser)
	}
	if id.UserID == "" {
		// No ids in the message: this text is fed to the model and persisted in
		// the session, and every id here comes off the request.
		return nil, fmt.Errorf("%w: the invocation's session carries no user", ErrNoActingUser)
	}

	// Before the cache read, because the client is part of the cache key. Costs a
	// mutex on the hot path: the client is built at most once, and a provider
	// that has never built one has nothing cached under its slot either.
	client, err := p.resolveClient(ctx)
	if err != nil {
		// Deliberately not qualified by resource: a client-init failure is about
		// this process's own credentials, not about the resource, and every
		// provider in the process fails it identically. Retrieval errors are a
		// different matter, and RetrieveCredential names the resource on all of
		// them itself — one client serves several, and a direct caller has no
		// provider to do it for them.
		return nil, err
	}

	key := auth.CredentialKey{AppName: id.AppName, UserID: id.UserID, Key: cacheSlot(client, p.scheme)}
	// A store read error is non-fatal: fall through and fetch a fresh credential.
	// A hit carrying no credential is a miss, though the interface forbids one:
	// a third-party store is not worth failing closed over, and returning a nil
	// credential to auth.Transport would fail the request anyway.
	if cred, ok, err := p.store.Get(ctx, key); err == nil && ok && cred != nil {
		return cred, nil
	}

	r, err := client.RetrieveCredential(ctx, Request{
		Resource:    p.scheme.Name,
		UserID:      id.UserID,
		Scopes:      p.scheme.Scopes,
		ContinueURI: p.scheme.ContinueURI,
	})
	if err != nil {
		return nil, err
	}
	p.cache(ctx, key, r)
	return r.Credential, nil
}

// cache stores r under key, best-effort: a store write failure must not fail
// auth.
//
// Nothing is stored unless the service gave a lifetime that is still running.
// That single test covers three cases: the service reported no expiry, meaning
// it cannot say when the token dies, so the credential cannot be vouched for
// later; it reported an unparseable one, which reaches here as the same zero
// time; or it reported one already spent, which a store deriving a TTL from it
// would turn into a negative one.
//
// The far end is clamped rather than rejected, so a wildly distant expiry —
// wrong, or injected — shortens to the cap instead of pinning the entry.
func (p *provider) cache(ctx context.Context, key auth.CredentialKey, r *Retrieval) {
	now := platform.Now(ctx)
	if !r.ExpiresAt.After(now) {
		return
	}
	expiresAt := r.ExpiresAt
	if capped := now.Add(maxCachedLifetime); expiresAt.After(capped) {
		expiresAt = capped
	}
	_ = p.store.Set(ctx, key, r.Credential, expiresAt)
}

// attribute names the resource a retrieval failure belongs to: several providers
// can be wired into one process, and an unattributed error says nothing about
// which.
//
// It names nothing else. This error becomes the tool's error, which is fed to
// the model and persisted in the session, and every id available here is
// supplied by the caller — a user id is commonly an email, and a session id
// arrives unvalidated from the request path. The invocation is already
// identified by the trace and the session the error is stored in.
func (p *provider) attribute(err error) error {
	return fmt.Errorf("gcp: resource %q: %w", p.scheme.Name, err)
}

// resolveClient returns the configured client, building a default one (backed by
// Application Default Credentials) on first use.
//
// Concurrent callers share one attempt and each waits on the earlier of its own
// context and that attempt's remaining bound: auth.Transport resolves a
// credential per outbound request, so a slow cold start must not outlive the
// request that triggered it. The bound is the attempt's rather than each
// waiter's, so a caller arriving midway through a stuck lookup waits out what is
// left of it instead of starting a fresh initTimeout of its own, and one
// arriving after it has passed fails immediately. A failed attempt is not
// cached, so the next call retries.
//
// A hung attempt is not abandoned. The lookup cannot be cancelled, so retiring
// it would start a fresh one every initTimeout, each parked in a syscall pinning
// an OS thread. Waiters get a prompt error instead, and the moment the stuck
// lookup returns its client is published and callers recover.
//
// The bound is charged against elapsed attempt time, not against any one
// waiter's arrival, so a caller that shows up after it has passed fails at once
// even if nobody ever waited. That is the deliberate trade for releasing a whole
// cohort together: the alternative arms a fresh bound per arrival, which is what
// made a stuck lookup cost every request its own full initTimeout.
func (p *provider) resolveClient(ctx context.Context) (*Client, error) {
	p.mu.Lock()
	if c := p.client; c != nil {
		p.mu.Unlock()
		return c, nil
	}
	in := p.pending
	if in == nil {
		in = &clientInit{done: make(chan struct{}), deadline: time.Now().Add(p.initTimeout)}
		p.pending = in
		go p.runInit(in)
	}
	p.mu.Unlock()

	// A result that has already landed wins over an expired bound. This narrows
	// the window rather than closing it, so the timer arm re-checks too.
	select {
	case <-in.done:
		return in.result()
	default:
	}

	timer := time.NewTimer(time.Until(in.deadline))
	defer timer.Stop()
	select {
	case <-in.done:
		return in.result()
	case <-timer.C:
		// The bound and the result can be ready at once — for a caller arriving
		// after the deadline the timer is ready immediately — and select would
		// then choose between them at random. The result wins.
		select {
		case <-in.done:
			return in.result()
		default:
		}
		// A sentinel of its own, not context.DeadlineExceeded: that is what the
		// caller-deadline arm below returns, and the two mean different things.
		return nil, fmt.Errorf("%w: the attempt exceeded %v and is still running", ErrClientUnavailable, p.initTimeout)
	case <-ctx.Done():
		// Same reasoning as the bound: a result that is already there beats a
		// caller that has just given up, and select would pick between them at
		// random.
		select {
		case <-in.done:
			return in.result()
		default:
		}
		return nil, fmt.Errorf("gcp: waiting for the default credentials client: %w", ctx.Err())
	}
}

// runInit builds the default client once and publishes it to the waiters on in.
func (p *provider) runInit(in *clientInit) {
	// This runs on a goroutine the provider owns, so nothing above can recover a
	// panic here and it would take the process down — where an eagerly built
	// client would merely have panicked in the caller's own frame. Report it as
	// this attempt's failure instead, panic value and all, and release the
	// waiters: without this, an abrupt exit leaves pending set with its goroutine
	// dead and every later caller waits out initTimeout forever.
	published := false
	defer func() {
		if published {
			return
		}
		// Every exit carries the sentinel: a caller behind a RoundTripper classifies
		// on errors.Is, and these are client-unavailable outcomes like any other.
		if r := recover(); r != nil {
			in.err = fmt.Errorf("%w: building it panicked: %v", ErrClientUnavailable, r)
		} else if in.err == nil {
			in.err = fmt.Errorf("%w: building it did not complete", ErrClientUnavailable)
		}
		p.publish(in)
	}()

	c, err := p.newClient(p.initCtx)
	switch {
	case err != nil:
		in.err = fmt.Errorf("%w: %w", ErrClientUnavailable, err)
	case c == nil:
		// Caching a nil client would only move the failure to the next retrieval.
		in.err = fmt.Errorf("%w: the builder returned no client", ErrClientUnavailable)
	default:
		in.client = c
	}
	published = true
	p.publish(in)
}

// publish caches a successful client, frees the in-flight slot so a failed
// attempt is retried rather than cached, and releases the waiters.
func (p *provider) publish(in *clientInit) {
	// Deferred for symmetry and against a future early return inside the critical
	// section, not against a panic. runInit is the only caller and sets published
	// before calling, so its recover has already declined by the time this runs —
	// a panic here would escape a goroutine nobody owns and end the process, which
	// leaves no later resolveClient to wedge.
	p.mu.Lock()
	defer p.mu.Unlock()
	if in.err == nil {
		p.client = in.client
	}
	p.pending = nil
	// Closed inside the critical section, which narrows the window rather than
	// closing it: no waiter re-acquires p.mu after its own read, so none of them
	// re-reads p.client and the two can never be atomic from where they stand.
	// What moving the close inside buys is the Unlock's worth of interval in
	// which the client is published and done is still open.
	close(in.done)
}
