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
	"strconv"
	"sync"
	"testing"
	"time"

	"google.golang.org/adk/v2/platform"
)

// Expired entries for principals that never come back must not accumulate:
// nothing but the sweep evicts them, and a read of some other key is enough to
// trigger one.
func TestInMemoryCredentialStoreSweepsExpired(t *testing.T) {
	s := NewInMemoryCredentialStore()
	cred := BearerCredential{Token: "t"}
	for i := range 300 {
		key := CredentialKey{UserID: "gone-" + strconv.Itoa(i), Key: "res"}
		if err := s.Set(t.Context(), key, cred, time.Now().Add(-time.Hour)); err != nil {
			t.Fatalf("Set() error = %v", err)
		}
	}
	if got := s.size(); got != 300 {
		t.Fatalf("store holds %d entries before any sweep is due, want 300", got)
	}

	s.arm()
	if _, _, err := s.Get(t.Context(), CredentialKey{UserID: "someone-else", Key: "res"}); err != nil {
		t.Fatalf("Get() error = %v", err)
	}
	if got := s.size(); got != 0 {
		t.Errorf("store holds %d entries, want the expired ones swept", got)
	}
}

// The sweep is paced: it costs O(entries), so it must not run on every call.
func TestInMemoryCredentialStoreSweepIsPaced(t *testing.T) {
	s := NewInMemoryCredentialStore()
	key := CredentialKey{UserID: "u", Key: "res"}
	if err := s.Set(t.Context(), key, BearerCredential{Token: "t"}, time.Now().Add(-time.Hour)); err != nil {
		t.Fatalf("Set() error = %v", err)
	}

	// Expired, but the next sweep is a minute away, so a Get for some other key
	// must leave the entry alone.
	swept := s.swept()
	if _, _, err := s.Get(t.Context(), CredentialKey{UserID: "other", Key: "res"}); err != nil {
		t.Fatalf("Get() error = %v", err)
	}
	if !s.swept().Equal(swept) {
		t.Error("a sweep ran twice inside one sweepInterval")
	}
	if got := s.size(); got != 1 {
		t.Errorf("store holds %d entries, want the expired one still resident until a sweep is due", got)
	}
}

// The sweep decides the fate of every entry in the store, so it must run on the
// store's own clock. platform.Now is scoped to one call tree by design, and one
// request arriving with a clock pinned to next year must not empty a store that
// other call trees share.
func TestInMemoryCredentialStoreSweepIgnoresTheCallersClock(t *testing.T) {
	s := NewInMemoryCredentialStore()
	for i := range 3 {
		key := CredentialKey{UserID: "live-" + strconv.Itoa(i), Key: "res"}
		if err := s.Set(t.Context(), key, BearerCredential{Token: "t"}, time.Now().Add(time.Hour)); err != nil {
			t.Fatalf("Set() error = %v", err)
		}
	}

	s.arm()
	ahead := platform.WithTimeProvider(t.Context(), func() time.Time { return time.Now().AddDate(5, 0, 0) })
	if _, _, err := s.Get(ahead, CredentialKey{UserID: "someone-else", Key: "res"}); err != nil {
		t.Fatalf("Get() error = %v", err)
	}
	if got := s.size(); got != 3 {
		t.Errorf("store holds %d entries, want all 3 — one caller's clock must not expire another's credentials", got)
	}
}

// The store is documented safe for concurrent use, and Get writes to the map on
// eviction while the sweep deletes from it, so a plain RWMutex read lock would
// not do. The pacing is shortened so the sweep body runs while readers are in
// flight — at its default it fires once, on an empty map, before any of these
// goroutines start — but not to zero, which would sweep every expired entry
// before any Get could reach one and leave the eviction branch uncovered.
func TestInMemoryCredentialStoreConcurrent(t *testing.T) {
	ctx := t.Context()
	s := NewInMemoryCredentialStore()
	s.mu.Lock()
	s.sweepEvery = 200 * time.Microsecond
	s.mu.Unlock()

	keyFor := func(i int) CredentialKey {
		return CredentialKey{AppName: "app", UserID: "user-" + strconv.Itoa(i%4), Key: "res"}
	}
	var wg sync.WaitGroup
	for g := range 8 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range 200 {
				key := keyFor(g + i)
				switch (g + i) % 3 {
				case 0:
					// Every other write is already spent, so Get's eviction and the
					// sweep's deletes run while other goroutines read.
					expires := time.Now().Add(time.Hour)
					if (g+i)%2 == 0 {
						expires = time.Now().Add(-time.Hour)
					}
					if err := s.Set(ctx, key, BearerCredential{Token: "t"}, expires); err != nil {
						t.Errorf("Set() error = %v", err)
						return
					}
				case 1:
					if _, _, err := s.Get(ctx, key); err != nil {
						t.Errorf("Get() error = %v", err)
						return
					}
				case 2:
					if err := s.Delete(ctx, key); err != nil {
						t.Errorf("Delete() error = %v", err)
						return
					}
				}
			}
		}()
	}
	wg.Wait()
}

// An entry's expiry is compared against later clock readings, so it must not
// carry a monotonic reading one of them might lack: the monotonic clock stops
// across a suspend and the wall clock does not, and two entries in different
// clock domains then expire on different timelines.
func TestInMemoryCredentialStoreSetStripsMonotonic(t *testing.T) {
	s := NewInMemoryCredentialStore()
	key := CredentialKey{UserID: "u", Key: "res"}
	// time.Now() carries a monotonic reading; Round(0) is the only way to drop it.
	if err := s.Set(t.Context(), key, BearerCredential{Token: "t"}, time.Now().Add(time.Hour)); err != nil {
		t.Fatalf("Set() error = %v", err)
	}
	s.mu.Lock()
	stored := s.entries[key].expiresAt
	s.mu.Unlock()
	// == compares the monotonic reading too, so this holds only once it is gone.
	if stored != stored.Round(0) {
		t.Errorf("stored expiry %v carries a monotonic reading", stored)
	}
}

// arm makes the next call sweep, so a test need not wait out sweepInterval.
func (s *InMemoryCredentialStore) arm() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.sweepEvery = time.Nanosecond
}

func (s *InMemoryCredentialStore) size() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.entries)
}

func (s *InMemoryCredentialStore) swept() time.Time {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.lastSweep
}

// Delete sweeps too. A caller whose only traffic after a burst is invalidation
// would otherwise never run one, and the type's doc says any call does.
func TestInMemoryCredentialStoreDeleteSweeps(t *testing.T) {
	s := NewInMemoryCredentialStore()
	stale := CredentialKey{UserID: "gone", Key: "res"}
	if err := s.Set(t.Context(), stale, BearerCredential{Token: "t"}, time.Now().Add(-time.Hour)); err != nil {
		t.Fatalf("Set() error = %v", err)
	}
	if got := s.size(); got != 1 {
		t.Fatalf("store holds %d entries, want the expired one still resident until a sweep is due", got)
	}

	s.arm()
	if err := s.Delete(t.Context(), CredentialKey{UserID: "unrelated", Key: "res"}); err != nil {
		t.Fatalf("Delete() error = %v", err)
	}
	if got := s.size(); got != 0 {
		t.Errorf("store holds %d entries after an armed Delete, want the expired one swept", got)
	}
}

// Get evicts the expired entry it was asked for, rather than only reporting a
// miss. That write is the reason the store takes a full Mutex instead of an
// RWMutex, so nothing else in the suite distinguishes the two designs.
func TestInMemoryCredentialStoreGetEvicts(t *testing.T) {
	s := NewInMemoryCredentialStore()
	key := CredentialKey{UserID: "u", Key: "res"}
	if err := s.Set(t.Context(), key, BearerCredential{Token: "t"}, time.Now().Add(-time.Hour)); err != nil {
		t.Fatalf("Set() error = %v", err)
	}
	if got := s.size(); got != 1 {
		t.Fatalf("store holds %d entries, want the expired one resident until something removes it", got)
	}

	// No sweep is due, so a miss here can only come from Get's own eviction.
	if _, ok, _ := s.Get(t.Context(), key); ok {
		t.Fatal("Get() of an expired entry = hit, want miss")
	}
	if got := s.size(); got != 0 {
		t.Error("Get() reported a miss without evicting the expired entry")
	}
}

// expired's boundary is the exact complement of the floor a producer applies
// before caching, so the store serves precisely what the auth/gcp provider is
// willing to write and nothing it is not. It is pinned here rather than through
// Get because a wall clock cannot hold still: any reading taken after the write
// has already moved past the equality case.
func TestExpiredBoundary(t *testing.T) {
	now := time.Date(2030, 1, 1, 0, 0, 0, 0, time.UTC)
	tests := []struct {
		name      string
		expiresAt time.Time
		want      bool
	}{
		{"a nanosecond beyond the margin", now.Add(ExpirySkew + time.Nanosecond), false},
		{"exactly the margin", now.Add(ExpirySkew), true},
		{"a nanosecond inside the margin", now.Add(ExpirySkew - time.Nanosecond), true},
		{"already past", now.Add(-time.Nanosecond), true},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := expired(now, tc.expiresAt); got != tc.want {
				t.Errorf("expired(now, now%+v) = %v, want %v", tc.expiresAt.Sub(now), got, tc.want)
			}
		})
	}
}
