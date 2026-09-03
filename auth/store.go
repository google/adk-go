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
)

// ExpirySkew is how long before its stated expiry [InMemoryCredentialStore]
// treats a cached credential as expired, so a credential is never handed out
// with too little lifetime left to complete the request it was fetched for.
//
// It absorbs two things. Time passes between the store read and the outbound
// request the credential authenticates. And the issuer's clock is not the
// caller's: Google's credential services warn that a token "may expire slightly
// earlier" than the expiry they report, for exactly that reason
// (https://agentidentitycredentials.googleapis.com/$discovery/rest?version=v1,
// the Success.expireTime field).
//
// It is exported so another [CredentialStore] can apply the same margin, and so
// a producer can decline to cache a credential with this much left or less —
// which is the boundary the reference producer and this store both use, so that
// the store serves exactly what a producer will write.
const ExpirySkew = 10 * time.Second

// CredentialKey identifies a cached credential: the app, the acting user, and a
// slot chosen by whatever produced the credential.
type CredentialKey struct {
	// AppName is the ADK application name.
	AppName string
	// UserID is the acting end user's identity.
	UserID string
	// Key is the producer's slot. It must distinguish every input that changes
	// which credential comes back — the endpoint, the identity the request for it
	// is authenticated as, the resource, the scopes, any redirect URI — and it
	// must distinguish the producer itself, since one store can back several. A
	// bare resource name does none of that: two providers for one resource that
	// differ only in scope would share an entry, and the narrower one would be
	// served the broader token.
	//
	// Derive it from a collision-resistant digest of those inputs rather than by
	// joining them, so that no delimiter appearing inside one of them can make
	// two different sets encode alike. auth/gcp is the reference construction.
	Key string
}

// The same warning applies to a store that flattens a CredentialKey into one
// identifier of its own — a row key, a filename, a cache line. AppName and
// UserID come off the request and ADK does not authenticate either, so joining
// the three fields on a separator lets {app "acme", user "bob|X"} and
// {app "acme|bob", user "X"} name one entry, and one end user is then served
// another's credential. Length-prefix or digest them, as auth/gcp does for the
// slot itself.

// CredentialStore caches resolved credentials across calls, keyed by
// [CredentialKey]. It exists so network-backed providers (e.g. auth/gcp) avoid a
// credential-service round-trip on every request; self-caching providers
// (oauth2.TokenSource) do not need it. Implementations must be safe for
// concurrent use.
//
// It carries no unexported method, so anyone can implement it and its method set
// is frozen: a method added later breaks every implementation outside this
// module. That is the price of the seam, and it is why Delete is here from the
// start rather than added when something first needs it.
type CredentialStore interface {
	// Get returns the cached, unexpired credential for key.
	//
	// The bool reports a usable hit, so a miss is (nil, false, nil). An
	// implementation must not report a hit carrying a nil credential, and must
	// not return a credential alongside a non-nil error: callers discard the
	// credential whenever the error is non-nil, so a store reporting a degraded
	// backend that way would have its result silently dropped.
	//
	// An entry should be treated as expired shortly before its stated expiry, by
	// no more than [ExpirySkew]. Producers decline to cache anything with that
	// much left or less, so a store applying a wider margin would be handed
	// credentials it then refuses to serve, and the cache would do nothing.
	//
	// The returned credential must be safe for concurrent [Credential.Apply] and
	// must not be mutated afterwards, the same requirement Set places on the
	// value handed in — a store that materializes credentials on read owes it
	// too, since one value is handed to every request that hits.
	Get(ctx context.Context, key CredentialKey) (Credential, bool, error)
	// Set stores cred for key until expiresAt, an absolute wall-clock time. Both
	// arguments are required: a caller that cannot establish a lifetime must not
	// cache, rather than cache forever.
	//
	// Wall clock, and not platform.Now, deliberately. That seam is scoped to one
	// call tree so concurrent runs can hold independent clocks, and a credential's
	// expiry is not that kind of quantity: the token dies when its issuer says it
	// does, on everybody's clock at once. Judging it on a simulated clock would
	// serve a dead credential to a real request, or drop a live one — and the
	// sweep, which judges entries the caller does not own, would let one call tree
	// empty another's.
	//
	// cred must be safe for concurrent [Credential.Apply] and must not be mutated
	// after Set returns, since every hit for key hands the same value to a
	// different request. A store that persists entries outside the process must
	// cope with a cred it cannot encode — several of this package's credentials
	// are opaque, and one wraps a live token source — by rejecting it, which
	// costs a round trip rather than correctness.
	//
	// Both arguments are secret. cred is bearer material, and key names the
	// application and the acting end user, so neither belongs in a log line or an
	// error string, and a store that persists them owes them the protection the
	// credentials themselves get.
	Set(ctx context.Context, key CredentialKey, cred Credential, expiresAt time.Time) error
	// Delete removes any entry for key. Removing an absent key is not an error.
	//
	// It is the invalidation hook for a caller that learns a credential is no
	// longer good before it expires — on consent revocation or logout. ADK does
	// not call it: a credential rejected downstream is refreshed by the provider
	// that issued it, not deleted from the store, and until a provider implements
	// that refresh a revoked credential is served until its cached entry expires.
	// The auth/gcp provider bounds that to an hour and exposes the key to delete
	// as gcp.Client.CacheKey.
	//
	// Delete does not cancel a retrieval already in flight, and it reports success
	// whether or not an entry was there. A retrieval that began before the Delete
	// writes its result afterwards, so if it began before the revocation it
	// restores exactly the credential the caller was removing, for the full
	// cached lifetime, and the nil return said nothing about it. Invalidate again
	// once the request that failed has returned.
	Delete(ctx context.Context, key CredentialKey) error
}

// sweepInterval is the shortest time between sweeps of expired entries. Nothing
// else evicts a principal that resolves once and never returns, and the sweep is
// O(entries), so it is paced by elapsed time rather than by call count: a large
// store under heavy traffic would otherwise pay for a scan on a fixed fraction
// of its calls.
const sweepInterval = time.Minute

// InMemoryCredentialStore is a concurrency-safe, process-local [CredentialStore]
// (per app+user+key, across sessions). It serves the same role as adk-python's
// InMemoryCredentialService, which buckets the same way, and adds per-entry
// expiry. The zero value is ready to use.
//
// It holds one entry per key ever written and has no size limit. An expired
// entry is dropped when its key is read, and otherwise by a sweep that any call
// at least [sweepInterval] after the last one performs, so the resident count is
// the number of distinct keys written within one credential lifetime.
//
// That is a count of keys, not of requests — unless the producer makes it one.
// The auth/gcp provider keys on the Client among other things, so a program that
// builds a Client per request writes an entry per request that nothing will ever
// read again, and the store then grows with request rate until the sweep catches
// up. Build long-lived Clients, or supply a store of your own that bounds
// itself.
//
// A process that stops calling the store entirely retains whatever it last held:
// nothing here runs on its own. A caller that must not keep credential material
// past its use should [InMemoryCredentialStore.Delete] it.
type InMemoryCredentialStore struct {
	// mu guards every field below. Not an RWMutex: Get evicts on expiry and can
	// sweep, so readers write.
	mu sync.Mutex
	// lastSweep is on the store's own clock, never a caller's. See sweep.
	lastSweep time.Time
	// sweepEvery overrides sweepInterval; only tests set it. Zero means the
	// default, so it cannot be used to ask for a sweep on every call.
	sweepEvery time.Duration
	entries    map[CredentialKey]cacheEntry
}

type cacheEntry struct {
	cred      Credential
	expiresAt time.Time
}

// NewInMemoryCredentialStore returns an empty [InMemoryCredentialStore].
func NewInMemoryCredentialStore() *InMemoryCredentialStore {
	return &InMemoryCredentialStore{entries: make(map[CredentialKey]cacheEntry)}
}

// Get implements [CredentialStore].
func (s *InMemoryCredentialStore) Get(_ context.Context, key CredentialKey) (Credential, bool, error) {
	now := time.Now()

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
func (s *InMemoryCredentialStore) Set(_ context.Context, key CredentialKey, cred Credential, expiresAt time.Time) error {
	if cred == nil {
		return errors.New("auth: Set requires a credential")
	}
	// The key names the app and the end user, so it stays out of the message: an
	// error text is logged far more freely than the store's contents.
	if expiresAt.IsZero() {
		return errors.New("auth: Set requires an expiry")
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	if s.entries == nil {
		s.entries = make(map[CredentialKey]cacheEntry)
	}
	s.sweep(time.Now())
	// Round(0) strips the monotonic reading a caller derived expiresAt from its
	// own clock would carry. Comparing two entries in different clock domains
	// diverges across a suspend, when the monotonic clock stops but the wall
	// clock does not.
	s.entries[key] = cacheEntry{cred: cred, expiresAt: expiresAt.Round(0)}
	return nil
}

// expired reports whether an entry expiring at expiresAt is spent as of now,
// counting the skew margin. Exactly [ExpirySkew] left counts as spent, so this
// is the precise complement of the test a producer applies before caching — the
// auth/gcp provider stores exactly what this will serve, and nothing that it
// won't.
func expired(now, expiresAt time.Time) bool {
	return !expiresAt.After(now.Add(ExpirySkew))
}

// sweep drops every expired entry, at most once per [sweepInterval]. The caller
// holds s.mu and supplies now, which is always the wall clock — see Set on why
// this type reads no other.
func (s *InMemoryCredentialStore) sweep(now time.Time) {
	every := s.sweepEvery
	if every == 0 {
		every = sweepInterval
	}
	if !s.lastSweep.IsZero() && now.Sub(s.lastSweep) < every {
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
	s.sweep(time.Now())
	delete(s.entries, key)
	return nil
}

var _ CredentialStore = (*InMemoryCredentialStore)(nil)
