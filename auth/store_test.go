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

package auth_test

import (
	"context"
	"testing"
	"time"

	"google.golang.org/adk/v2/auth"
	"google.golang.org/adk/v2/platform"
)

func TestInMemoryCredentialStoreGetSet(t *testing.T) {
	ctx := t.Context()
	s := auth.NewInMemoryCredentialStore()
	key := auth.CredentialKey{AppName: "app", UserID: "user", Key: "res"}
	cred := auth.BearerCredential{Token: "tok"}

	if _, ok, err := s.Get(ctx, key); err != nil || ok {
		t.Fatalf("Get() on empty store = (_, %v, %v), want (_, false, nil)", ok, err)
	}
	if err := s.Set(ctx, key, cred, time.Now().Add(time.Hour)); err != nil {
		t.Fatalf("Set() error = %v", err)
	}
	got, ok, err := s.Get(ctx, key)
	if err != nil || !ok || got != cred {
		t.Fatalf("Get() after Set = (%v, %v, %v), want the stored credential", got, ok, err)
	}

	// A different key is a miss.
	if _, ok, _ := s.Get(ctx, auth.CredentialKey{AppName: "app", UserID: "other", Key: "res"}); ok {
		t.Error("Get() for a different user = hit, want miss")
	}
}

func TestInMemoryCredentialStoreExpiry(t *testing.T) {
	ctx := t.Context()
	s := auth.NewInMemoryCredentialStore()
	key := auth.CredentialKey{Key: "res"}
	cred := auth.APIKeyCredential{Name: "X", Value: "v"}

	if err := s.Set(ctx, key, cred, time.Now().Add(time.Hour)); err != nil {
		t.Fatal(err)
	}
	if _, ok, _ := s.Get(ctx, key); !ok {
		t.Error("Get() with far-future expiry = miss, want hit")
	}

	if err := s.Set(ctx, key, cred, time.Now().Add(-time.Second)); err != nil {
		t.Fatal(err)
	}
	if _, ok, _ := s.Get(ctx, key); ok {
		t.Error("Get() with past expiry = hit, want miss")
	}
}

// TestInMemoryCredentialStoreGetHonorsClock verifies Get evaluates expiry
// against the context's clock (platform.Now), so a test can drive time
// deterministically via platform.WithTimeProvider — no real sleeping.
func TestInMemoryCredentialStoreGetHonorsClock(t *testing.T) {
	base := time.Date(2030, 1, 1, 0, 0, 0, 0, time.UTC)
	expiry := base.Add(time.Hour)
	tests := []struct {
		name    string
		clock   time.Time
		wantHit bool
	}{
		{name: "before expiry", clock: base, wantHit: true},
		{name: "past expiry", clock: base.Add(2 * time.Hour), wantHit: false},
		{name: "inside the skew window", clock: expiry.Add(-time.Second), wantHit: false},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ctx := platform.WithTimeProvider(t.Context(), func() time.Time { return tc.clock })
			s := auth.NewInMemoryCredentialStore()
			key := auth.CredentialKey{Key: "res"}
			if err := s.Set(ctx, key, auth.BearerCredential{Token: "tok"}, expiry); err != nil {
				t.Fatal(err)
			}
			if _, ok, _ := s.Get(ctx, key); ok != tc.wantHit {
				t.Errorf("Get() with clock %s = hit %v, want %v", tc.clock, ok, tc.wantHit)
			}
		})
	}
}

func TestInMemoryCredentialStoreOverwrite(t *testing.T) {
	ctx := t.Context()
	s := auth.NewInMemoryCredentialStore()
	key := auth.CredentialKey{Key: "res"}
	first := auth.BearerCredential{Token: "a"}
	second := auth.BearerCredential{Token: "b"}

	expiry := time.Now().Add(time.Hour)
	if err := s.Set(ctx, key, first, expiry); err != nil {
		t.Fatal(err)
	}
	if err := s.Set(ctx, key, second, expiry); err != nil {
		t.Fatal(err)
	}
	if got, _, _ := s.Get(ctx, key); got != second {
		t.Fatalf("Get() after overwrite = %v, want the second credential", got)
	}
}

func TestInMemoryCredentialStoreSetRequiresCredentialAndExpiry(t *testing.T) {
	s := auth.NewInMemoryCredentialStore()
	key := auth.CredentialKey{Key: "res"}
	if err := s.Set(t.Context(), key, nil, time.Now().Add(time.Hour)); err == nil {
		t.Error("Set() with a nil credential = nil error, want error")
	}
	// A zero expiry must not silently mean "cache forever".
	if err := s.Set(t.Context(), key, auth.BearerCredential{Token: "t"}, time.Time{}); err == nil {
		t.Error("Set() with a zero expiry = nil error, want error")
	}
}

func TestInMemoryCredentialStoreDelete(t *testing.T) {
	ctx := t.Context()
	s := auth.NewInMemoryCredentialStore()
	key := auth.CredentialKey{Key: "res"}
	if err := s.Set(ctx, key, auth.BearerCredential{Token: "t"}, time.Now().Add(time.Hour)); err != nil {
		t.Fatal(err)
	}
	if err := s.Delete(ctx, key); err != nil {
		t.Fatalf("Delete() error = %v", err)
	}
	if _, ok, _ := s.Get(ctx, key); ok {
		t.Error("Get() after Delete = hit, want miss")
	}
	if err := s.Delete(ctx, key); err != nil {
		t.Errorf("Delete() of an absent key = %v, want nil", err)
	}
}

// The zero value is usable: Set used to panic on the nil map.
func TestInMemoryCredentialStoreZeroValue(t *testing.T) {
	var s auth.InMemoryCredentialStore
	key := auth.CredentialKey{Key: "res"}
	if err := s.Set(t.Context(), key, auth.BearerCredential{Token: "t"}, time.Now().Add(time.Hour)); err != nil {
		t.Fatalf("Set() on a zero store error = %v", err)
	}
	if _, ok, _ := s.Get(t.Context(), key); !ok {
		t.Error("Get() after Set on a zero store = miss, want hit")
	}
}

// The clock is caller-supplied code; calling it under the store's non-reentrant
// mutex would deadlock a TimeProvider that reaches back into the store.
func TestInMemoryCredentialStoreClockMayTouchStore(t *testing.T) {
	s := auth.NewInMemoryCredentialStore()
	key := auth.CredentialKey{Key: "res"}
	ctx := platform.WithTimeProvider(t.Context(), func() time.Time {
		_, _, _ = s.Get(context.Background(), auth.CredentialKey{Key: "other"})
		return time.Now()
	})
	if err := s.Set(ctx, key, auth.BearerCredential{Token: "t"}, time.Now().Add(time.Hour)); err != nil {
		t.Fatalf("Set() error = %v", err)
	}

	done := make(chan struct{})
	go func() {
		defer close(done)
		_, _, _ = s.Get(ctx, key)
	}()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("Get() deadlocked; the clock must not be called while holding the lock")
	}
}
