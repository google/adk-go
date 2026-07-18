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
	// Store caches resolved credentials across requests (keyed by app, user, and
	// resource). When nil, an in-memory store is used. Caching matters here
	// because each miss is a network round-trip (and up to a ~10s pending poll)
	// to the credential service.
	Store auth.CredentialStore
}

// NewProvider returns an [auth.CredentialProvider] that resolves credentials for
// the given GCP resource via the Agent Identity / IAM Connector services.
//
// The acting user is taken from the ADK context ([agent.FromContext]) at resolve
// time, so the provider must run within an agent invocation (e.g. wired into
// mcptoolset or remoteagent).
func NewProvider(scheme Scheme, cfg *ProviderConfig) (auth.CredentialProvider, error) {
	if scheme.Name == "" {
		return nil, errors.New("gcp: NewProvider requires a scheme Name")
	}
	if cfg == nil {
		cfg = &ProviderConfig{}
	}
	// Defensive copy: the provider outlives this call and reads Scopes on every
	// request, so it must not alias a slice the caller can mutate later.
	scheme.Scopes = slices.Clone(scheme.Scopes)
	store := cfg.Store
	if store == nil {
		store = auth.NewInMemoryCredentialStore()
	}
	return &provider{scheme: scheme, store: store, client: cfg.Client}, nil
}

type provider struct {
	scheme Scheme
	store  auth.CredentialStore

	mu     sync.Mutex
	client *Client
}

var (
	_ auth.CredentialProvider = (*provider)(nil)
	_ auth.RefreshingProvider = (*provider)(nil)
)

// Credential implements [auth.CredentialProvider].
func (p *provider) Credential(ctx context.Context) (auth.Credential, error) {
	key, err := p.resolveKey(ctx)
	if err != nil {
		return nil, err
	}
	// A store read error is non-fatal: fall through and fetch a fresh credential.
	if cred, ok, err := p.store.Get(ctx, key); err == nil && ok {
		return cred, nil
	}
	return p.fetch(ctx, key, p.request(key, "", false))
}

// Refresh implements [auth.RefreshingProvider]: it discards the cached
// credential and forces the service to mint a new one, passing the prior
// (rejected) token where the service needs it.
func (p *provider) Refresh(ctx context.Context) (auth.Credential, error) {
	key, err := p.resolveKey(ctx)
	if err != nil {
		return nil, err
	}
	var prior string
	if cred, ok, _ := p.store.Get(ctx, key); ok {
		prior = credentialToken(cred)
	}
	return p.fetch(ctx, key, p.request(key, prior, true))
}

// resolveKey builds the store key from the acting user's identity.
func (p *provider) resolveKey(ctx context.Context) (auth.CredentialKey, error) {
	rc, err := agent.RequireContext(ctx)
	if err != nil {
		return auth.CredentialKey{}, fmt.Errorf("gcp: %w", err)
	}
	userID := rc.UserID()
	if userID == "" {
		return auth.CredentialKey{}, errors.New("gcp: ADK context has no user id")
	}
	return auth.CredentialKey{AppName: rc.AppName(), UserID: userID, Key: p.scheme.Name}, nil
}

func (p *provider) request(key auth.CredentialKey, priorToken string, forceRefresh bool) Request {
	return Request{
		Resource:     p.scheme.Name,
		UserID:       key.UserID,
		Scopes:       p.scheme.Scopes,
		ContinueURI:  p.scheme.ContinueURI,
		ForceRefresh: forceRefresh,
		PriorToken:   priorToken,
	}
}

// fetch retrieves a credential and (best-effort) caches it; a store write
// failure must not fail auth.
func (p *provider) fetch(ctx context.Context, key auth.CredentialKey, req Request) (auth.Credential, error) {
	client, err := p.resolveClient(ctx)
	if err != nil {
		return nil, err
	}
	cred, expiresAt, err := client.retrieve(ctx, req)
	if err != nil {
		return nil, err
	}
	// Cache only when the service reported an expiry: a zero time means "never
	// expires" to the store, and the GCP services omit it only when the lifetime
	// is unknown — caching that would risk serving a stale credential forever.
	// Best-effort: a store write failure must not fail auth.
	if !expiresAt.IsZero() {
		_ = p.store.Set(ctx, key, cred, expiresAt)
	}
	return cred, nil
}

// credentialToken returns the bearer/API-key token carried by a resolved GCP
// credential, used as the prior token on refresh. A custom-header credential
// (wrapped for extra headers) yields "", so the refresh proceeds without the
// prior-token hint; GCP issues bearer tokens on this path in practice.
func credentialToken(c auth.Credential) string {
	switch v := c.(type) {
	case auth.BearerCredential:
		return v.Token
	case auth.APIKeyCredential:
		return v.Value
	}
	return ""
}

// resolveClient returns the configured client, creating a default one (backed by
// Application Default Credentials) on first use. The client's lifetime is
// detached from this one request with [context.WithoutCancel] (it is cached and
// reused) while keeping ctx's values.
func (p *provider) resolveClient(ctx context.Context) (*Client, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.client == nil {
		c, err := NewClient(context.WithoutCancel(ctx), nil)
		if err != nil {
			return nil, err
		}
		p.client = c
	}
	return p.client, nil
}
