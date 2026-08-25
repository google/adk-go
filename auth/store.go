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

package auth

import (
	"context"
	"errors"
	"sync"
	"time"

	"google.golang.org/adk/v2/platform"
)

// expirySkew is how long before its stated expiry a cached credential is treated
// as expired, to absorb clock skew and in-flight latency.
const expirySkew = 10 * time.Second

// CredentialKey identifies a cached credential: the app, the acting user, and a
// caller-chosen slot (typically the target resource or scheme name).
type CredentialKey struct {
	// AppName is the ADK application name.
	AppName string
	// UserID is the acting end user's identity.
	UserID string
	// Key is a caller-chosen slot, typically the target resource or scheme name.
	Key string
}

// CredentialStore caches resolved credentials across calls, keyed by
// [CredentialKey]. It exists so network-backed providers (e.g. auth/gcp) avoid a
// credential-service round-trip on every request; self-caching providers
// (oauth2.TokenSource) do not need it. Implementations must be safe for
// concurrent use.
type CredentialStore interface {
	// Get returns the cached, unexpired credential for key.
	//
	// The bool reports a usable hit, so a miss is (nil, false, nil). An
	// implementation must not report a hit carrying a nil credential, and must
	// not return a credential alongside a non-nil error: callers discard the
	// credential whenever the error is non-nil, so a store reporting a degraded
	// backend that way would have its result silently dropped.
	//
	// An entry should be treated as expired shortly before its stated expiry, so
	// a credential is never handed out with too little lifetime left to complete
	// the request it was fetched for. [InMemoryCredentialStore] uses a 10s
	// margin.
	Get(ctx context.Context, key CredentialKey) (Credential, bool, error)
	// Set stores cred for key until expiresAt. Both cred and expiresAt are
	// required: a caller that cannot establish a lifetime must not cache, rather
	// than cache forever.
	Set(ctx context.Context, key CredentialKey, cred Credential, expiresAt time.Time) error
	// Delete removes any entry for key. Removing an absent key is not an error.
	//
	// It is the invalidation hook for callers that learn a credential is no
	// longer good before it expires — on consent revocation or logout. ADK does
	// not call it: a credential rejected downstream is refreshed by the provider
	// that issued it, not deleted from the store, and until a provider implements
	// that refresh a revoked credential is served until its cached expiry. The
	// auth/gcp provider bounds that window to an hour.
	Delete(ctx context.Context, key CredentialKey) error
}

// sweepInterval is the shortest time between sweeps of expired entries. Nothing
// else evicts a principal that resolves once and never returns, and the sweep is
// O(entries), so it is paced by wall clock rather than by call count: a large
// store under heavy traffic would otherwise pay for a scan on a fixed fraction
// of its calls.
const sweepInterval = time.Minute

// InMemoryCredentialStore is a concurrency-safe, process-local [CredentialStore]
// (per app+user+key, across sessions). It serves the same role as adk-python's
// InMemoryCredentialService, which buckets the same way, and adds per-entry
// expiry. The zero value is ready to use.
//
// It holds one entry per key seen, and it is unbounded: an expired entry is
// dropped when its key is read, and otherwise by a sweep that any call at least
// [sweepInterval] after the last one performs. Steady-state size is therefore
// the number of distinct keys active within one credential lifetime. A process
// that stops calling the store entirely retains whatever it last held, so a
// caller that must not keep credential material past its use should
// [InMemoryCredentialStore.Delete] it.
type InMemoryCredentialStore struct {
	// mu guards every field below. Not an RWMutex: Get evicts on expiry and can
	// sweep, so readers write.
	mu        sync.Mutex
	lastSweep time.Time
	entries   map[CredentialKey]cacheEntry
}

type cacheEntry struct {
	cred      Credential
	expiresAt time.Time
}

// NewInMemoryCredentialStore returns an empty [InMemoryCredentialStore].
func NewInMemoryCredentialStore() *InMemoryCredentialStore {
	return &InMemoryCredentialStore{entries: make(map[CredentialKey]cacheEntry)}
}

// Get implements [CredentialStore]. Expiry is evaluated against the context's
// clock ([platform.Now]), so tests can drive it deterministically.
func (s *InMemoryCredentialStore) Get(ctx context.Context, key CredentialKey) (Credential, bool, error) {
	// Resolved before the lock: platform.Now is caller-supplied, and a clock that
	// reaches back into the store would deadlock on this non-reentrant mutex.
	now := platform.Now(ctx)

	s.mu.Lock()
	defer s.mu.Unlock()
	s.sweep(now)
	e, ok := s.entries[key]
	if !ok {
		return nil, false, nil
	}
	if expired(now, e.expiresAt) {
		delete(s.entries, key)
		return nil, false, nil
	}
	return e.cred, true, nil
}

// Set implements [CredentialStore].
func (s *InMemoryCredentialStore) Set(ctx context.Context, key CredentialKey, cred Credential, expiresAt time.Time) error {
	if cred == nil {
		return errors.New("auth: Set requires a credential")
	}
	// The key names the app and the end user, so it stays out of the message: an
	// error text is logged far more freely than the store's contents.
	if expiresAt.IsZero() {
		return errors.New("auth: Set requires an expiry")
	}
	now := platform.Now(ctx)

	s.mu.Lock()
	defer s.mu.Unlock()
	if s.entries == nil {
		s.entries = make(map[CredentialKey]cacheEntry)
	}
	s.sweep(now)
	s.entries[key] = cacheEntry{cred: cred, expiresAt: expiresAt}
	return nil
}

// expired reports whether an entry expiring at expiresAt is spent as of now,
// counting the skew margin.
func expired(now, expiresAt time.Time) bool {
	return now.Add(expirySkew).After(expiresAt)
}

// sweep drops every expired entry, at most once per [sweepInterval]. The caller
// holds s.mu.
func (s *InMemoryCredentialStore) sweep(now time.Time) {
	if now.Sub(s.lastSweep) < sweepInterval {
		return
	}
	s.lastSweep = now
	for k, e := range s.entries {
		if expired(now, e.expiresAt) {
			delete(s.entries, k)
		}
	}
}

// Delete implements [CredentialStore].
func (s *InMemoryCredentialStore) Delete(_ context.Context, key CredentialKey) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.entries, key)
	return nil
}

var _ CredentialStore = (*InMemoryCredentialStore)(nil)
