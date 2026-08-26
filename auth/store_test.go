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
	"testing"
	"time"

	"google.golang.org/adk/v2/auth"
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

// Expiry is judged on the wall clock, and a test drives it by choosing how much
// life to store rather than by moving a clock: a credential dies when its issuer
// says it does, so this type reads no overridable clock at all.
func TestInMemoryCredentialStoreExpiryBoundaries(t *testing.T) {
	tests := []struct {
		name    string
		left    time.Duration
		wantHit bool
	}{
		{name: "an hour left", left: time.Hour, wantHit: true},
		{name: "long past", left: -time.Hour, wantHit: false},
		{name: "inside the skew window", left: auth.ExpirySkew - time.Second, wantHit: false},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ctx := t.Context()
			s := auth.NewInMemoryCredentialStore()
			key := auth.CredentialKey{Key: "res"}
			if err := s.Set(ctx, key, auth.BearerCredential{Token: "tok"}, time.Now().Add(tc.left)); err != nil {
				t.Fatal(err)
			}
			if _, ok, _ := s.Get(ctx, key); ok != tc.wantHit {
				t.Errorf("Get() with %s left = hit %v, want %v", tc.left, ok, tc.wantHit)
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

// ExpirySkew is public API and the provider's caching floor, so its value is
// pinned from both sides: a wider margin would silently raise that floor, and a
// narrower one would hand out credentials with no usable life left.
func TestExpirySkewBoundary(t *testing.T) {
	tests := []struct {
		name    string
		left    time.Duration
		wantHit bool
	}{
		// Relative to the constant: these pin which side of the boundary is
		// inclusive, so that the store serves exactly what a producer will cache.
		{"just outside the margin", auth.ExpirySkew + time.Second, true},
		{"exactly the margin", auth.ExpirySkew, false},
		{"just inside the margin", auth.ExpirySkew - time.Millisecond, false},
		// Spelled in seconds, so they move if the constant does. A test written
		// only against ExpirySkew measures it against itself and pins nothing.
		{"eleven seconds left", 11 * time.Second, true},
		{"nine seconds left", 9 * time.Second, false},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ctx := t.Context()
			s := auth.NewInMemoryCredentialStore()
			key := auth.CredentialKey{Key: "res"}
			if err := s.Set(ctx, key, auth.BearerCredential{Token: "t"}, time.Now().Add(tc.left)); err != nil {
				t.Fatalf("Set() error = %v", err)
			}
			if _, ok, _ := s.Get(ctx, key); ok != tc.wantHit {
				t.Errorf("Get() with %s left = hit %v, want %v", tc.left, ok, tc.wantHit)
			}
		})
	}
}
