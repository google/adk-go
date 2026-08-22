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
	"fmt"
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
	// Get returns the cached, unexpired credential for key, if present.
	Get(ctx context.Context, key CredentialKey) (Credential, bool, error)
	// Set stores cred for key until expiresAt. Both cred and expiresAt are
	// required: a caller that cannot establish a lifetime must not cache, rather
	// than cache forever.
	Set(ctx context.Context, key CredentialKey, cred Credential, expiresAt time.Time) error
	// Delete removes any entry for key, so a credential can be invalidated ahead
	// of its expiry — on consent revocation, logout, or a downstream 401.
	// Removing an absent key is not an error.
	Delete(ctx context.Context, key CredentialKey) error
}

// sweepEvery is how many Set calls pass between sweeps of expired entries.
// Nothing else evicts a principal that resolves once and never returns.
const sweepEvery = 256

// InMemoryCredentialStore is a concurrency-safe, process-local [CredentialStore]
// (per app+user+key, across sessions). It serves the same role as adk-python's
// InMemoryCredentialService, which buckets the same way, and adds per-entry
// expiry. The zero value is ready to use.
type InMemoryCredentialStore struct {
	// mu guards entries. Not an RWMutex: Get evicts on expiry, so readers write.
	mu      sync.Mutex
	sets    uint64
	entries map[CredentialKey]cacheEntry
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
	e, ok := s.entries[key]
	if !ok {
		return nil, false, nil
	}
	if now.Add(expirySkew).After(e.expiresAt) {
		delete(s.entries, key)
		return nil, false, nil
	}
	return e.cred, true, nil
}

// Set implements [CredentialStore].
func (s *InMemoryCredentialStore) Set(ctx context.Context, key CredentialKey, cred Credential, expiresAt time.Time) error {
	if cred == nil {
		return fmt.Errorf("auth: Set requires a credential")
	}
	if expiresAt.IsZero() {
		return fmt.Errorf("auth: Set requires an expiry for %+v", key)
	}
	now := platform.Now(ctx)

	s.mu.Lock()
	defer s.mu.Unlock()
	if s.entries == nil {
		s.entries = make(map[CredentialKey]cacheEntry)
	}
	s.sets++
	if s.sets%sweepEvery == 0 {
		for k, e := range s.entries {
			if now.Add(expirySkew).After(e.expiresAt) {
				delete(s.entries, k)
			}
		}
	}
	s.entries[key] = cacheEntry{cred: cred, expiresAt: expiresAt}
	return nil
}

// Delete implements [CredentialStore].
func (s *InMemoryCredentialStore) Delete(_ context.Context, key CredentialKey) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.entries, key)
	return nil
}

var _ CredentialStore = (*InMemoryCredentialStore)(nil)
