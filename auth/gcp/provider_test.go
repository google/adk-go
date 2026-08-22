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

package gcp_test

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"google.golang.org/adk/v2/auth"
	"google.golang.org/adk/v2/auth/gcp"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/platform"
	"google.golang.org/adk/v2/session"
)

// TestProviderCredential drives two users through one shared provider: the
// provider is long-lived, so serving one user's credential to another is the
// failure that matters. It also pins scopes and continueUri on the wire.
func TestProviderCredential(t *testing.T) {
	var gotUsers []string
	var gotScopes []string
	var gotContinueURI string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			UserID      string   `json:"userId"`
			Scopes      []string `json:"scopes"`
			ContinueURI string   `json:"continueUri"`
		}
		_ = json.NewDecoder(r.Body).Decode(&body)
		gotUsers = append(gotUsers, body.UserID)
		gotScopes, gotContinueURI = body.Scopes, body.ContinueURI
		// Echo the caller back, so a credential served to the wrong user shows up.
		_, _ = io.WriteString(w, `{"success":{"token":"tok-`+body.UserID+`","header":"Authorization: Bearer"}}`)
	}))
	defer srv.Close()

	client, err := gcp.NewClient(t.Context(), &gcp.Config{
		HTTPClient:            srv.Client(),
		AgentIdentityEndpoint: srv.URL,
	})
	if err != nil {
		t.Fatalf("NewClient() error = %v", err)
	}
	scopes := []string{"s1", "s2"}
	p, err := gcp.NewProvider(
		gcp.Scheme{
			Name:        "projects/p/locations/l/authProviders/ap",
			Scopes:      scopes,
			ContinueURI: "https://example.test/continue",
		},
		&gcp.ProviderConfig{Client: client},
	)
	if err != nil {
		t.Fatalf("NewProvider() error = %v", err)
	}
	scopes[0] = "mutated" // the provider must have cloned this

	for _, user := range []string{"alice", "bob"} {
		cred, err := p.Credential(adkContext(t, user))
		if err != nil {
			t.Fatalf("Credential(%q) error = %v", user, err)
		}
		if bc, ok := cred.(auth.BearerCredential); !ok || bc.Token != "tok-"+user {
			t.Errorf("credential for %q = %+v, want bearer %q", user, cred, "tok-"+user)
		}
	}
	if !slices.Equal(gotUsers, []string{"alice", "bob"}) {
		t.Errorf("service saw users %q, want [alice bob]", gotUsers)
	}
	if !slices.Equal(gotScopes, []string{"s1", "s2"}) {
		t.Errorf("body scopes = %q, want [s1 s2] (caller's later mutation must not leak)", gotScopes)
	}
	if gotContinueURI != "https://example.test/continue" {
		t.Errorf("body continueUri = %q, want the scheme's", gotContinueURI)
	}
}

func TestProviderRequiresADKContext(t *testing.T) {
	// Fails the test if reached: the guard must reject before any service call.
	srv := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		t.Error("credentials service must not be called without an ADK identity")
	}))
	defer srv.Close()
	client, err := gcp.NewClient(t.Context(), &gcp.Config{
		HTTPClient:            srv.Client(),
		AgentIdentityEndpoint: srv.URL,
	})
	if err != nil {
		t.Fatalf("NewClient() error = %v", err)
	}
	p, err := gcp.NewProvider(
		gcp.Scheme{Name: "projects/p/locations/l/authProviders/ap"},
		&gcp.ProviderConfig{Client: client},
	)
	if err != nil {
		t.Fatalf("NewProvider() error = %v", err)
	}
	_, err = p.Credential(t.Context())
	if !errors.Is(err, gcp.ErrNoActingUser) {
		t.Fatalf("Credential() error = %v, want gcp.ErrNoActingUser", err)
	}
}

func TestNewProviderRejectsUnconstructedClient(t *testing.T) {
	_, err := gcp.NewProvider(
		gcp.Scheme{Name: "projects/p/locations/l/authProviders/ap"},
		&gcp.ProviderConfig{Client: &gcp.Client{}},
	)
	if err == nil {
		t.Fatal("NewProvider() = nil error, want a zero Client rejected")
	}
}

func TestNewProviderValidatesScheme(t *testing.T) {
	if _, err := gcp.NewProvider(gcp.Scheme{}, nil); err == nil {
		t.Fatal("NewProvider() = nil error, want error for empty scheme Name")
	}
}

func TestProviderCachesCredential(t *testing.T) {
	var calls int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&calls, 1)
		_, _ = io.WriteString(w, `{"success":{"token":"tok","header":"Authorization: Bearer","expireTime":"2999-01-01T00:00:00Z"}}`)
	}))
	defer srv.Close()

	client, err := gcp.NewClient(t.Context(), &gcp.Config{
		HTTPClient:            srv.Client(),
		AgentIdentityEndpoint: srv.URL,
	})
	if err != nil {
		t.Fatalf("NewClient() error = %v", err)
	}
	// Default (in-memory) store; two resolves for the same app+user+resource.
	p, err := gcp.NewProvider(gcp.Scheme{Name: "projects/p/locations/l/authProviders/ap"}, &gcp.ProviderConfig{Client: client})
	if err != nil {
		t.Fatalf("NewProvider() error = %v", err)
	}

	for i := range 2 {
		if _, err := p.Credential(adkContext(t, "user-1")); err != nil {
			t.Fatalf("call %d: Credential() error = %v", i, err)
		}
	}
	if got := atomic.LoadInt32(&calls); got != 1 {
		t.Errorf("service calls = %d, want 1 (second resolve should hit the cache)", got)
	}
}

func TestProviderSkipsCacheWithoutExpiry(t *testing.T) {
	var calls int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&calls, 1)
		// No expireTime: lifetime unknown, so the provider must not cache.
		_, _ = io.WriteString(w, `{"success":{"token":"tok","header":"Authorization: Bearer"}}`)
	}))
	defer srv.Close()

	client, err := gcp.NewClient(t.Context(), &gcp.Config{
		HTTPClient:            srv.Client(),
		AgentIdentityEndpoint: srv.URL,
	})
	if err != nil {
		t.Fatalf("NewClient() error = %v", err)
	}
	p, err := gcp.NewProvider(gcp.Scheme{Name: "projects/p/locations/l/authProviders/ap"}, &gcp.ProviderConfig{Client: client})
	if err != nil {
		t.Fatalf("NewProvider() error = %v", err)
	}

	for i := range 2 {
		if _, err := p.Credential(adkContext(t, "user-1")); err != nil {
			t.Fatalf("call %d: Credential() error = %v", i, err)
		}
	}
	if got := atomic.LoadInt32(&calls); got != 2 {
		t.Errorf("service calls = %d, want 2 (unknown expiry must not be cached)", got)
	}
}

// adkContext returns an ADK invocation context (recoverable via agent.IdentityFromContext)
// for the given user.
func adkContext(t *testing.T, userID string) context.Context {
	t.Helper()
	svc := session.InMemoryService()
	resp, err := svc.Create(t.Context(), &session.CreateRequest{AppName: "app", UserID: userID})
	if err != nil {
		t.Fatalf("session Create() error = %v", err)
	}
	return icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{Session: resp.Session})
}

// TestProviderCacheIsolatesPrincipals is the guard on the cache's headline
// property: a long-lived provider must never serve one principal's credential
// to another. Collapsing the cache key leaves every other test green.
func TestProviderCacheIsolatesPrincipals(t *testing.T) {
	var calls int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&calls, 1)
		var body struct {
			UserID string `json:"userId"`
		}
		_ = json.NewDecoder(r.Body).Decode(&body)
		// Echo the caller, so a credential served to the wrong principal shows up.
		_, _ = io.WriteString(w, `{"success":{"token":"tok-`+body.UserID+`","header":"Authorization: Bearer","expireTime":"2999-01-01T00:00:00Z"}}`)
	}))
	defer srv.Close()

	client, err := gcp.NewClient(t.Context(), &gcp.Config{HTTPClient: srv.Client(), AgentIdentityEndpoint: srv.URL})
	if err != nil {
		t.Fatalf("NewClient() error = %v", err)
	}
	p, err := gcp.NewProvider(gcp.Scheme{Name: "projects/p/locations/l/authProviders/ap"}, &gcp.ProviderConfig{Client: client})
	if err != nil {
		t.Fatalf("NewProvider() error = %v", err)
	}

	// Twice each, so the second pass reads through the cache.
	for range 2 {
		for _, user := range []string{"user-1", "user-2"} {
			cred, err := p.Credential(adkContext(t, user))
			if err != nil {
				t.Fatalf("Credential(%q) error = %v", user, err)
			}
			if bc, ok := cred.(auth.BearerCredential); !ok || bc.Token != "tok-"+user {
				t.Fatalf("%q was served %+v, want bearer %q", user, cred, "tok-"+user)
			}
		}
	}
	if got := atomic.LoadInt32(&calls); got != 2 {
		t.Errorf("service calls = %d, want 2 (one cold read per principal)", got)
	}
}

// A store shared by providers that differ only in scopes must not serve the
// broad credential to the narrow one.
func TestProviderCacheSeparatesScopes(t *testing.T) {
	var gotScopes [][]string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Scopes []string `json:"scopes"`
		}
		_ = json.NewDecoder(r.Body).Decode(&body)
		gotScopes = append(gotScopes, body.Scopes)
		_, _ = io.WriteString(w, `{"success":{"token":"tok-`+strings.Join(body.Scopes, ",")+`","header":"Authorization: Bearer","expireTime":"2999-01-01T00:00:00Z"}}`)
	}))
	defer srv.Close()

	client, err := gcp.NewClient(t.Context(), &gcp.Config{HTTPClient: srv.Client(), AgentIdentityEndpoint: srv.URL})
	if err != nil {
		t.Fatalf("NewClient() error = %v", err)
	}
	store := auth.NewInMemoryCredentialStore()
	newProvider := func(scope string) auth.CredentialProvider {
		t.Helper()
		p, err := gcp.NewProvider(
			gcp.Scheme{Name: "projects/p/locations/l/authProviders/ap", Scopes: []string{scope}},
			&gcp.ProviderConfig{Client: client, Store: store},
		)
		if err != nil {
			t.Fatalf("NewProvider() error = %v", err)
		}
		return p
	}

	for _, scope := range []string{"drive", "drive.readonly"} {
		cred, err := newProvider(scope).Credential(adkContext(t, "user-1"))
		if err != nil {
			t.Fatalf("Credential(%q) error = %v", scope, err)
		}
		if bc, ok := cred.(auth.BearerCredential); !ok || bc.Token != "tok-"+scope {
			t.Errorf("scope %q was served %+v, want bearer %q", scope, cred, "tok-"+scope)
		}
	}
	if len(gotScopes) != 2 {
		t.Errorf("service calls = %d, want 2 (the narrow provider must not reuse the broad entry)", len(gotScopes))
	}
}

// recordingStore reports every call, and can fail writes.
type recordingStore struct {
	inner   auth.CredentialStore
	sets    int
	gets    int
	setErr  error
	nilOnce bool // return a hit carrying no credential on the first Get
}

func (s *recordingStore) Get(ctx context.Context, key auth.CredentialKey) (auth.Credential, bool, error) {
	s.gets++
	if s.nilOnce {
		s.nilOnce = false
		return nil, true, nil
	}
	return s.inner.Get(ctx, key)
}

func (s *recordingStore) Set(ctx context.Context, key auth.CredentialKey, cred auth.Credential, expiresAt time.Time) error {
	s.sets++
	if s.setErr != nil {
		return s.setErr
	}
	return s.inner.Set(ctx, key, cred, expiresAt)
}

func (s *recordingStore) Delete(ctx context.Context, key auth.CredentialKey) error {
	return s.inner.Delete(ctx, key)
}

// A configured store is actually used, a write failure does not fail auth, and
// a hit carrying no credential is treated as a miss rather than returned.
func TestProviderStorePlumbing(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, `{"success":{"token":"tok","header":"Authorization: Bearer","expireTime":"2999-01-01T00:00:00Z"}}`)
	}))
	defer srv.Close()
	client, err := gcp.NewClient(t.Context(), &gcp.Config{HTTPClient: srv.Client(), AgentIdentityEndpoint: srv.URL})
	if err != nil {
		t.Fatalf("NewClient() error = %v", err)
	}

	store := &recordingStore{inner: auth.NewInMemoryCredentialStore(), setErr: errors.New("disk on fire"), nilOnce: true}
	p, err := gcp.NewProvider(
		gcp.Scheme{Name: "projects/p/locations/l/authProviders/ap"},
		&gcp.ProviderConfig{Client: client, Store: store},
	)
	if err != nil {
		t.Fatalf("NewProvider() error = %v", err)
	}
	cred, err := p.Credential(adkContext(t, "user-1"))
	if err != nil {
		t.Fatalf("Credential() error = %v; a store write failure must not fail auth", err)
	}
	if cred == nil {
		t.Fatal("Credential() = nil credential; a hit with no credential must be treated as a miss")
	}
	if store.gets != 1 || store.sets != 1 {
		t.Errorf("store gets/sets = %d/%d, want 1/1 (the configured store must be used)", store.gets, store.sets)
	}
}

// The service's expiry is honoured only up to a bound, so a bad or injected
// expireTime cannot pin a credential in the cache.
func TestProviderClampsServiceExpiry(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, `{"success":{"token":"tok","header":"Authorization: Bearer","expireTime":"9999-12-31T23:59:59Z"}}`)
	}))
	defer srv.Close()
	client, err := gcp.NewClient(t.Context(), &gcp.Config{HTTPClient: srv.Client(), AgentIdentityEndpoint: srv.URL})
	if err != nil {
		t.Fatalf("NewClient() error = %v", err)
	}
	store := &recordingStore{inner: auth.NewInMemoryCredentialStore()}
	p, err := gcp.NewProvider(
		gcp.Scheme{Name: "projects/p/locations/l/authProviders/ap"},
		&gcp.ProviderConfig{Client: client, Store: store},
	)
	if err != nil {
		t.Fatalf("NewProvider() error = %v", err)
	}
	if _, err := p.Credential(adkContext(t, "user-1")); err != nil {
		t.Fatalf("Credential() error = %v", err)
	}
	// Far past the cap: the entry must already be gone a day later.
	future := platform.WithTimeProvider(t.Context(), func() time.Time { return time.Now().Add(24 * time.Hour) })
	key := auth.CredentialKey{AppName: "app", UserID: "user-1", Key: "projects/p/locations/l/authProviders/ap||"}
	if _, ok, _ := store.inner.Get(future, key); ok {
		t.Error("credential still cached a day later, want the service expiry clamped")
	}
}
