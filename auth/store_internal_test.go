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
	"testing"
	"time"

	"google.golang.org/adk/v2/platform"
)

// Expired entries for principals that never come back must not accumulate:
// nothing but the sweep evicts them, and a read of some other key is enough to
// trigger one.
func TestInMemoryCredentialStoreSweepsExpired(t *testing.T) {
	clock := time.Date(2030, 1, 1, 0, 0, 0, 0, time.UTC)
	ctx := platform.WithTimeProvider(t.Context(), func() time.Time { return clock })
	s := NewInMemoryCredentialStore()
	cred := BearerCredential{Token: "t"}

	for i := range 300 {
		key := CredentialKey{UserID: "gone-" + strconv.Itoa(i), Key: "res"}
		if err := s.Set(ctx, key, cred, clock.Add(time.Minute)); err != nil {
			t.Fatalf("Set() error = %v", err)
		}
	}
	if got := s.size(); got != 300 {
		t.Fatalf("store holds %d entries before expiry, want 300", got)
	}

	clock = clock.Add(time.Hour) // every entry above is now expired
	if _, _, err := s.Get(ctx, CredentialKey{UserID: "someone-else", Key: "res"}); err != nil {
		t.Fatalf("Get() error = %v", err)
	}
	if got := s.size(); got != 0 {
		t.Errorf("store holds %d entries, want the expired ones swept", got)
	}
}

// The sweep is paced: it costs O(entries), so it must not run on every call.
func TestInMemoryCredentialStoreSweepIsPaced(t *testing.T) {
	clock := time.Date(2030, 1, 1, 0, 0, 0, 0, time.UTC)
	ctx := platform.WithTimeProvider(t.Context(), func() time.Time { return clock })
	s := NewInMemoryCredentialStore()

	key := CredentialKey{UserID: "u", Key: "res"}
	if err := s.Set(ctx, key, BearerCredential{Token: "t"}, clock.Add(10*time.Second)); err != nil {
		t.Fatalf("Set() error = %v", err)
	}
	// Expired, but the next sweep is not due yet, so a Get for some other key
	// must leave the entry alone.
	clock = clock.Add(sweepInterval / 2)
	swept := s.swept()
	if _, _, err := s.Get(ctx, CredentialKey{UserID: "other", Key: "res"}); err != nil {
		t.Fatalf("Get() error = %v", err)
	}
	if !s.swept().Equal(swept) {
		t.Error("a sweep ran twice inside one sweepInterval")
	}
	if got := s.size(); got != 1 {
		t.Errorf("store holds %d entries, want the expired one still resident until a sweep is due", got)
	}
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

// A clock that jumps forward once must not switch eviction off for good. The
// clock is caller-supplied and need not be monotonic, so one call arriving with
// a far-future reading would otherwise park lastSweep there and suppress every
// later sweep until real time caught up.
func TestInMemoryCredentialStoreSweepSurvivesAClockJump(t *testing.T) {
	base := time.Date(2030, 1, 1, 0, 0, 0, 0, time.UTC)
	clock := base
	ctx := platform.WithTimeProvider(t.Context(), func() time.Time { return clock })
	s := NewInMemoryCredentialStore()

	// One call lands with a clock five years ahead, on an empty store.
	ahead := platform.WithTimeProvider(t.Context(), func() time.Time { return base.AddDate(5, 0, 0) })
	if _, _, err := s.Get(ahead, CredentialKey{UserID: "other", Key: "res"}); err != nil {
		t.Fatalf("Get() error = %v", err)
	}

	// Entries written on the real clock, short-lived.
	for i := range 3 {
		key := CredentialKey{UserID: "gone-" + strconv.Itoa(i), Key: "res"}
		if err := s.Set(ctx, key, BearerCredential{Token: "t"}, clock.Add(time.Minute)); err != nil {
			t.Fatalf("Set() error = %v", err)
		}
	}

	// An hour later on the real clock, they are all expired and a sweep is due.
	// Nobody ever reads their keys, so the sweep is the only thing that can
	// remove them.
	clock = clock.Add(time.Hour)
	if _, _, err := s.Get(ctx, CredentialKey{UserID: "other", Key: "res"}); err != nil {
		t.Fatalf("Get() error = %v", err)
	}
	if got := s.size(); got != 0 {
		t.Errorf("store holds %d entries, want them swept once the clock came back", got)
	}
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
