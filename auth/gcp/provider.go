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
	// projects/*/locations/*/authProviders/* (Agent Identity).
	Name string
	// Scopes are the OAuth scopes requested for the credential.
	Scopes []string
	// ContinueURI is the developer-hosted URI used to finalize managed-OAuth
	// (3-legged) flows; unused by non-interactive flows.
	ContinueURI string
}

// ProviderConfig configures a provider built by [NewProvider]. A nil
// *ProviderConfig, or any zero-valued field, uses the corresponding default.
type ProviderConfig struct {
	// Client reaches the credential services. When nil, a default client (backed
	// by Application Default Credentials) is created lazily on first use. That
	// default targets the production services, so pass a [Client] built by
	// [NewClient] to reach a test endpoint or to tune the poll timeout.
	Client *Client
}

// ErrNoActingUser means the provider could not determine the acting end user,
// either because the context is not an ADK context or because the invocation
// carries no user.
var ErrNoActingUser = errors.New("gcp: no acting user")

// defaultInitTimeout retires a hung default-client init so the next call starts
// a fresh one, instead of every caller attaching to a flight that never lands.
// The lookup cannot be bounded by a context: FindDefaultCredentials reads the
// credentials file with os.ReadFile and probes with the context-free
// metadata.OnGCE(), neither of which observes cancellation.
const defaultInitTimeout = 30 * time.Second

// NewProvider returns an [auth.CredentialProvider] that resolves credentials for
// the given GCP resource via the Agent Identity / IAM Connector services.
//
// The acting user is taken from the ADK context ([agent.IdentityFromContext]) at
// resolve time, so the provider must run within an agent invocation — and every
// request it authenticates must descend from the invoking user's context. That
// holds for mcptoolset's per-call requests, but not for a transport that shares
// one connection across invocations.
//
// Whoever wires this up is trusting the embedding server: ADK does not
// authenticate session.UserID, and it now decides whose credential is minted.
//
// ctx is used only to build the default client when cfg.Client is nil, and only
// for its values; its cancellation is not honored, because the client outlives
// any one request. Pass the process-scoped context the rest of the app is wired
// with, not a request's.
func NewProvider(ctx context.Context, scheme ProviderScheme, cfg *ProviderConfig) (auth.CredentialProvider, error) {
	if scheme.Name == "" {
		return nil, errors.New("gcp: NewProvider requires a scheme Name")
	}
	// The same check RetrieveCredential makes per request: a malformed name is a
	// wiring mistake, so fail here rather than on every call inside a transport.
	if err := validateResource(scheme.Name); err != nil {
		return nil, fmt.Errorf("gcp: NewProvider: %w", err)
	}
	if cfg == nil {
		cfg = &ProviderConfig{}
	}
	// A zero Client would nil-deref deep inside net/http on first use.
	if cfg.Client != nil && cfg.Client.httpClient == nil {
		return nil, errors.New("gcp: ProviderConfig.Client must come from NewClient")
	}
	// Defensive copy: the provider outlives this call and re-reads Scopes per request, so it must not alias a caller-mutable slice.
	scheme.Scopes = slices.Clone(scheme.Scopes)
	return &provider{
		scheme:      scheme,
		client:      cfg.Client,
		initCtx:     context.WithoutCancel(ctx),
		newClient:   func(ctx context.Context) (*Client, error) { return NewClient(ctx, nil) },
		initTimeout: defaultInitTimeout,
	}, nil
}

type provider struct {
	scheme ProviderScheme
	// initCtx roots the lazily built default client. It is held for the life of
	// the provider, which is why NewProvider asks for a process-scoped one.
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
		return nil, fmt.Errorf("%w: provider must run within an agent invocation", ErrNoActingUser)
	}
	if id.UserID == "" {
		return nil, fmt.Errorf("%w: invocation for app %q session %q has an empty UserID", ErrNoActingUser, id.AppName, id.SessionID)
	}

	client, err := p.resolveClient(ctx)
	if err != nil {
		return nil, err
	}
	cred, err := client.RetrieveCredential(ctx, Request{
		Resource:    p.scheme.Name,
		UserID:      id.UserID,
		Scopes:      p.scheme.Scopes,
		ContinueURI: p.scheme.ContinueURI,
	})
	if err != nil {
		// Name the user and resource: several providers can be wired into one
		// process, and the failure otherwise says nothing about which.
		return nil, fmt.Errorf("gcp: credential for user %q on %q: %w", id.UserID, p.scheme.Name, err)
	}
	return cred, nil
}

// resolveClient returns the configured client, creating a default one (backed by
// Application Default Credentials) on first use.
//
// Concurrent first callers share one attempt, and each waits on its own ctx:
// auth.Transport resolves a credential per outbound request, so a slow cold
// start must not outlive the request that triggered it. A failed or timed-out
// attempt is not cached; the next call starts a new one.
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

	select {
	case <-in.done:
		if in.err != nil {
			return nil, in.err
		}
		return in.client, nil
	case <-ctx.Done():
		return nil, fmt.Errorf("gcp: waiting for the default credentials client: %w", ctx.Err())
	}
}

// runInit runs one attempt at building the default client and publishes it to
// the waiters on in.
func (p *provider) runInit(in *clientInit) {
	type result struct {
		client *Client
		err    error
	}
	// Buffered, because the lookup cannot be cancelled: once the attempt is
	// retired nobody receives, and an unbuffered send would leak the goroutine.
	res := make(chan result, 1)
	go func() {
		c, err := p.newClient(p.initCtx)
		res <- result{c, err}
	}()

	timer := time.NewTimer(p.initTimeout)
	defer timer.Stop()
	select {
	case r := <-res:
		in.client = r.client
		if r.err != nil {
			in.err = fmt.Errorf("gcp: build default credentials client: %w", r.err)
		}
	case <-timer.C:
		in.err = fmt.Errorf("gcp: building the default credentials client exceeded %v", p.initTimeout)
	}

	p.mu.Lock()
	if in.err == nil {
		p.client = in.client
	}
	p.pending = nil // so a failed or retired attempt is retried, not wedged
	p.mu.Unlock()
	close(in.done)
}
