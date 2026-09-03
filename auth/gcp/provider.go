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
	"sync/atomic"
	"time"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/auth"
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
	// (3-legged) flows; unused by non-interactive flows.
	ContinueURI string
}

// ProviderConfig configures a provider built by [NewProvider].
type ProviderConfig struct {
	// Scheme is the resource to mint credentials for. Required.
	Scheme ProviderScheme
	// Client reaches the credential services. When nil, a default client backed
	// by Application Default Credentials is built lazily on first use, against
	// the production endpoints; that adds two failure modes to Credential, since
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
	// resource, the scopes, the continue URI, and the Client. Two providers share
	// an entry when they share a Client and agree on all of that, and not
	// otherwise. Building a Client per provider is therefore a way to get no
	// sharing at all; see [Config.HTTPClient].
	//
	// The cost of caching is staleness. A credential revoked before it expires
	// keeps being served until the cached entry does, which is the service's
	// expiry or an hour, whichever comes first. To invalidate one sooner, pass
	// [Client.CacheKey] to [auth.CredentialStore.Delete] — which needs a Client,
	// so set one here rather than leaving it to the lazy default if you intend to
	// invalidate at all. The provider does not expose the Client it builds for
	// itself.
	Store auth.CredentialStore
}

// ErrClientUnavailable means the default Application Default Credentials client
// is not available: discovery failed, or it did not finish inside the bound. The
// lookup is not cancellable, so that bound is on the wait rather than on the
// attempt, which keeps running — a later call may well succeed.
var ErrClientUnavailable = errors.New("gcp: default credentials client unavailable")

// ErrNoActingUser means the provider could not determine the acting end user,
// either because the context is not an ADK context or because the invocation
// carries no user. No user, no credential — the same call adk-python makes,
// which raises on a missing user id rather than degrading the turn.
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
// resolve time, so the provider must run within an agent invocation. Two
// requirements then fall on the transport that carries the authenticated
// requests, neither of which this package can enforce:
//
//   - Every request must descend from the invoking user's context. A transport
//     that shares one connection across invocations does not qualify —
//     mcptoolset included: its per-call POSTs are per-user, but the MCP session
//     and the standalone server-to-client stream are opened on the context of
//     whichever user connected first and stay bound to it.
//   - The transport must not follow a cross-host redirect. net/http strips
//     Authorization above the RoundTripper, so [auth.Transport] re-resolves and
//     re-applies the end user's credential to the redirect target. Set
//     CheckRedirect on the http.Client that carries the transport; the
//     ADC-backed client [NewClient] builds for itself does the same.
//
// Wiring this up also means trusting the embedding server: ADK does not
// authenticate session.UserID, and it now decides whose credential is minted.
//
// ctx is used only to build the default client, and only for its values; its
// cancellation is not honored, because that client outlives any one request.
// Pass the process-scoped context the rest of the app is wired with, not a
// request's. It is ignored when cfg.Client is set.
func NewProvider(ctx context.Context, cfg ProviderConfig) (auth.CredentialProvider, error) {
	if cfg.Scheme.Name == "" {
		return nil, errors.New("gcp: NewProvider requires a scheme Name")
	}
	// A malformed name is a wiring mistake; catch it here rather than on every
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
// It also bounds how long a credential revoked before its expiry keeps being
// served, which the service warns can happen at any time.
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
// the scopes makes the slot independent of the caller's ordering.
//
// There is no adk-python original to match. Python's credential service is not
// on this provider's path at all: GcpAuthProviderScheme is a CustomAuthScheme,
// and CredentialManager returns the provider's credential directly without ever
// loading or saving one (credential_manager.py). Caching GCP credentials is a Go
// addition, and this slot is its own design. The nearest analogue is
// AuthConfig.get_credential_key (auth_tool.py), which joins two digests of
// canonical JSON — one of the auth scheme, one of the credential used to obtain
// it. Go cannot produce the second, a Client's credentials being opaque to this
// package, so it names the Client instead.
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
	// blown is set by the first waiter whose bound expires. Later waiters fail
	// fast instead of each paying the bound again: the attempt is kept running,
	// so without this a stuck lookup costs every outbound request its full
	// initTimeout for as long as it is stuck.
	blown atomic.Bool
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

	// Before the cache read, because the Client is part of the cache key and a
	// provider that has not built one yet has no slot, so nothing can be cached
	// under it. Costs one mutex on the hot path; the client is built at most
	// once.
	client, err := p.resolveClient(ctx)
	if err != nil {
		// Not attributed: a client-init failure is about this process's own
		// credentials, not about the resource, and every provider in the process
		// fails it identically. Attributing it would also stack the package prefix.
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
		return nil, p.attribute(err)
	}
	p.cache(ctx, key, r)
	return r.Credential, nil
}

// cache stores r under key, best-effort: a store write failure must not fail
// auth.
//
// Nothing is stored unless the service gave a lifetime with enough left to be
// worth reading back. That single test covers four cases: the service reported
// no expiry, meaning it cannot say when the token dies, so the credential cannot
// be vouched for later; it reported an unparseable one, which reaches here as
// the same zero time; it reported one already spent, which a store deriving a
// TTL from it would turn into a negative one; or it reported one so close that
// a store applying the standard margin would refuse to serve the entry the
// moment it was written.
//
// The far end is clamped rather than rejected, so a wildly distant expiry —
// wrong, or injected — shortens to the cap instead of pinning the entry.
//
// Wall clock, not platform.Now: what is being decided is when a real credential
// stops working, which no simulated clock changes. [auth.CredentialStore.Set]
// says the same of the value written here.
func (p *provider) cache(ctx context.Context, key auth.CredentialKey, r *Retrieval) {
	now := time.Now()
	if !r.ExpiresAt.After(now.Add(auth.ExpirySkew)) {
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
// context and initTimeout: auth.Transport resolves a credential per outbound
// request, so a slow cold start must not outlive the request that triggered it.
// A failed attempt is not cached; the next call retries.
//
// A hung attempt is not abandoned. The lookup cannot be cancelled, so retiring
// it would start a fresh one every initTimeout, each parked in a syscall pinning
// an OS thread. Waiters get a prompt error instead, and the moment the stuck
// lookup returns its client is published and callers recover.
func (p *provider) resolveClient(ctx context.Context) (*Client, error) {
	p.mu.Lock()
	if c := p.client; c != nil {
		p.mu.Unlock()
		return c, nil
	}
	in := p.pending
	if in == nil {
		in = &clientInit{done: make(chan struct{})}
		p.pending = in
		go p.runInit(in)
	}
	p.mu.Unlock()

	if in.blown.Load() {
		select {
		case <-in.done: // landed after the bound blew; fall through to the result
		default:
			return nil, fmt.Errorf("%w: an earlier attempt exceeded %v and is still running", ErrClientUnavailable, p.initTimeout)
		}
	}

	timer := time.NewTimer(p.initTimeout)
	defer timer.Stop()
	select {
	case <-in.done:
		if in.err != nil {
			return nil, in.err
		}
		return in.client, nil
	case <-timer.C:
		// A sentinel of its own, not context.DeadlineExceeded: that is what the
		// caller-deadline arm below returns, and the two mean different things.
		in.blown.Store(true)
		return nil, fmt.Errorf("%w after %v", ErrClientUnavailable, p.initTimeout)
	case <-ctx.Done():
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
		if r := recover(); r != nil {
			in.err = fmt.Errorf("gcp: building the default credentials client panicked: %v", r)
		} else if in.err == nil {
			in.err = errors.New("gcp: building the default credentials client did not complete")
		}
		p.publish(in)
	}()

	c, err := p.newClient(p.initCtx)
	switch {
	case err != nil:
		in.err = fmt.Errorf("%w: %w", ErrClientUnavailable, err)
	case c == nil:
		// Caching a nil client would only move the failure to the next retrieval.
		in.err = errors.New("gcp: default credentials client builder returned no client")
	default:
		in.client = c
	}
	published = true
	p.publish(in)
}

// publish caches a successful client, frees the in-flight slot so a failed
// attempt is retried rather than cached, and releases the waiters.
func (p *provider) publish(in *clientInit) {
	p.mu.Lock()
	if in.err == nil {
		p.client = in.client
	}
	p.pending = nil
	p.mu.Unlock()
	close(in.done)
}
