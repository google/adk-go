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
	"errors"
	"fmt"
	"slices"
	"sync"
	"time"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/auth"
)

// ProviderScheme identifies a GCP auth resource and the access it requests. It
// mirrors adk-python's GcpAuthProviderScheme.
type ProviderScheme struct {
	// Name is the full resource name, routed by [Client]: either
	// projects/*/locations/*/connectors/* (IAM Connector) or
	// projects/*/locations/*/authProviders/* (Agent Identity). [NewProvider]
	// accepts only those two shapes.
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
}

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
//     re-applies the end user's credential to the redirect target. [NewClient]
//     refuses redirects for the same reason.
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
	p := &provider{
		scheme:      cfg.Scheme,
		client:      cfg.Client,
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

type provider struct {
	scheme ProviderScheme
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
}

var _ auth.CredentialProvider = (*provider)(nil)

// Credential implements [auth.CredentialProvider].
func (p *provider) Credential(ctx context.Context) (auth.Credential, error) {
	id, ok := agent.IdentityFromContext(ctx)
	if !ok {
		return nil, fmt.Errorf("%w: no ADK invocation identity on the context — not an agent invocation, or its session is unset", ErrNoActingUser)
	}
	if id.UserID == "" {
		return nil, fmt.Errorf("%w: invocation for app %q session %q has an empty UserID", ErrNoActingUser, id.AppName, id.SessionID)
	}

	client, err := p.resolveClient(ctx)
	if err != nil {
		return nil, p.attribute(id, err)
	}
	cred, err := client.RetrieveCredential(ctx, Request{
		Resource:    p.scheme.Name,
		UserID:      id.UserID,
		Scopes:      p.scheme.Scopes,
		ContinueURI: p.scheme.ContinueURI,
	})
	if err != nil {
		return nil, p.attribute(id, err)
	}
	return cred, nil
}

// attribute names the invocation and resource a failure belongs to: several
// providers can be wired into one process, and an unattributed error says
// nothing about which. It names the session rather than the user, because this
// error becomes the tool's error — fed to the model and persisted in the
// session — and UserID is commonly an email.
func (p *provider) attribute(id agent.Identity, err error) error {
	return fmt.Errorf("gcp: credential for session %q on %q: %w", id.SessionID, p.scheme.Name, err)
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

	timer := time.NewTimer(p.initTimeout)
	defer timer.Stop()
	select {
	case <-in.done:
		if in.err != nil {
			return nil, in.err
		}
		return in.client, nil
	case <-timer.C:
		// Wrap DeadlineExceeded so a caller can tell a bounded wait from a hard
		// failure without matching on the message.
		return nil, fmt.Errorf("gcp: default credentials client not ready after %v: %w", p.initTimeout, context.DeadlineExceeded)
	case <-ctx.Done():
		return nil, fmt.Errorf("gcp: waiting for the default credentials client: %w", ctx.Err())
	}
}

// runInit builds the default client once and publishes it to the waiters on in.
func (p *provider) runInit(in *clientInit) {
	c, err := p.newClient(p.initCtx)
	switch {
	case err != nil:
		in.err = fmt.Errorf("gcp: build default credentials client: %w", err)
	case c == nil:
		// Caching a nil client would only move the failure to the next retrieval.
		in.err = errors.New("gcp: default credentials client builder returned no client")
	default:
		in.client = c
	}

	p.mu.Lock()
	if in.err == nil {
		p.client = in.client
	}
	p.pending = nil // so a failed attempt is retried, not cached
	p.mu.Unlock()
	close(in.done)
}
