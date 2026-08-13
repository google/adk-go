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

	"golang.org/x/sync/singleflight"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/auth"
)

// Scheme identifies a GCP auth resource and the access it requests. It mirrors
// adk-python's GcpAuthProviderScheme.
type Scheme struct {
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
	// by Application Default Credentials) is created lazily on first use.
	Client *Client
}

// ErrNoActingUser means the provider could not determine the acting end user,
// either because the context is not an ADK context or because the invocation
// carries no user.
var ErrNoActingUser = errors.New("gcp: no acting user")

// NewProvider returns an [auth.CredentialProvider] that resolves credentials for
// the given GCP resource via the Agent Identity / IAM Connector services.
//
// The acting user is taken from the ADK context ([agent.IdentityFromContext]) at
// resolve time, so the provider must run within an agent invocation — and every
// request it authenticates must descend from the invoking user's context. That
// holds for mcptoolset's per-call requests, but not for a transport that shares
// one connection across invocations.
func NewProvider(scheme Scheme, cfg *ProviderConfig) (auth.CredentialProvider, error) {
	if scheme.Name == "" {
		return nil, errors.New("gcp: NewProvider requires a scheme Name")
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
	return &provider{scheme: scheme, client: cfg.Client}, nil
}

type provider struct {
	scheme Scheme

	mu         sync.Mutex
	client     *Client
	clientInit singleflight.Group // coalesces concurrent first-time client init
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
	return client.RetrieveCredential(ctx, Request{
		Resource:    p.scheme.Name,
		UserID:      id.UserID,
		Scopes:      p.scheme.Scopes,
		ContinueURI: p.scheme.ContinueURI,
	})
}

// resolveClient returns the configured client, creating a default one (backed by
// Application Default Credentials) on first use.
//
// Concurrent first callers are coalesced via singleflight so the ADC lookup runs
// once, off the mutex, and each waits on its own ctx: auth.Transport resolves a
// credential per outbound request, so a slow cold start must not outlive the
// request that triggered it. Init runs on a context detached from the caller's
// cancellation (values kept) so the shared client isn't bound to one request. A
// failed init is not cached; the next call retries.
//
// The lookup cannot be bounded by a context: FindDefaultCredentials reads the
// credentials file with os.ReadFile and probes with the context-free
// metadata.OnGCE(), neither of which observes cancellation. The bound therefore
// lives on the waiting side, not on the context handed to NewClient.
func (p *provider) resolveClient(ctx context.Context) (*Client, error) {
	p.mu.Lock()
	c := p.client
	p.mu.Unlock()
	if c != nil {
		return c, nil
	}

	ch := p.clientInit.DoChan("client", func() (any, error) {
		// A prior winner may have set the client while we waited.
		p.mu.Lock()
		c := p.client
		p.mu.Unlock()
		if c != nil {
			return c, nil
		}
		nc, err := NewClient(context.WithoutCancel(ctx), nil)
		if err != nil {
			return nil, err
		}
		p.mu.Lock()
		p.client = nc
		p.mu.Unlock()
		return nc, nil
	})
	select {
	case r := <-ch:
		if r.Err != nil {
			return nil, r.Err
		}
		return r.Val.(*Client), nil
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}
