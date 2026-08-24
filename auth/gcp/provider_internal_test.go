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
	"net/http"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// TestResolveClientBuildsDefaultClient drives the lazy ADC path end to end:
// discovery, the cached client, and a retrieval whose token is minted after the
// request that triggered construction is already gone.
func TestResolveClientBuildsDefaultClient(t *testing.T) {
	fakeADC(t)
	srv, calls := sequenceServer(`{"success":{"token":"tok","header":"Authorization: Bearer"}}`)
	defer srv.Close()

	p := newTestProvider(t)
	ctx, cancel := context.WithCancel(t.Context())
	c, err := p.resolveClient(ctx)
	if err != nil {
		t.Fatalf("resolveClient() error = %v", err)
	}
	cancel()
	// The default client targets the production endpoint; retarget it so the
	// retrieval below stays offline.
	c.agentIdentityURL = srv.URL

	if _, err := c.RetrieveCredential(t.Context(), Request{Resource: authProviderResource, UserID: "u"}); err != nil {
		t.Fatalf("RetrieveCredential() error = %v", err)
	}
	if got := atomic.LoadInt32(calls); got != 1 {
		t.Errorf("service calls = %d, want 1", got)
	}
	if again, err := p.resolveClient(t.Context()); err != nil || again != c {
		t.Errorf("resolveClient() = %v, %v; want the cached client", again, err)
	}
}

// TestResolveClientSingleFlight pins the concurrency design: many first callers
// share one build and all get the same client. Nothing exercises the mutex
// unless a test runs callers together — the race detector included, which is
// what catches a lock that stops locking.
func TestResolveClientSingleFlight(t *testing.T) {
	var attempts atomic.Int32
	built := &Client{httpClient: http.DefaultClient}
	release := make(chan struct{})
	p := newTestProvider(t)
	p.newClient = func(context.Context) (*Client, error) {
		attempts.Add(1)
		<-release // hold the flight open so every caller piles up on it
		return built, nil
	}

	const callers = 16
	var wg sync.WaitGroup
	got := make([]*Client, callers)
	errs := make([]error, callers)
	for i := range callers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			got[i], errs[i] = p.resolveClient(t.Context())
		}()
	}
	time.Sleep(20 * time.Millisecond)
	close(release)
	wg.Wait()

	if n := attempts.Load(); n != 1 {
		t.Errorf("init attempts = %d, want 1 (concurrent callers must share one flight)", n)
	}
	for i := range callers {
		if errs[i] != nil || got[i] != built {
			t.Fatalf("caller %d = %v, %v; want the single shared client", i, got[i], errs[i])
		}
	}
}

// TestResolveClientHonorsCallerDeadline pins that a caller waiting on a slow
// init is bounded by its own context: auth.Transport resolves a credential per
// outbound request, so one cold start must not stall every concurrent request.
func TestResolveClientHonorsCallerDeadline(t *testing.T) {
	p := newTestProvider(t)
	p.newClient = blockingInit(t)

	const deadline = 50 * time.Millisecond
	ctx, cancel := context.WithTimeout(t.Context(), deadline)
	defer cancel()
	start := time.Now()
	_, err := p.resolveClient(ctx)
	elapsed := time.Since(start)
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("resolveClient() error = %v, want the caller's deadline", err)
	}
	// Loose enough not to flake on a loaded machine, tight enough that a waiter
	// which ignored the caller's ctx — and so sat out the 30s initTimeout —
	// fails here.
	if elapsed >= p.initTimeout {
		t.Errorf("resolveClient() returned after %v, want the caller's %v deadline, not the %v init bound", elapsed, deadline, p.initTimeout)
	}
}

// TestResolveClientBoundsWaitOnHungInit pins the initTimeout: an ADC lookup that
// never returns cannot be cancelled, so a caller with no deadline of its own must
// still be released — while the attempt itself is kept.
func TestResolveClientBoundsWaitOnHungInit(t *testing.T) {
	p := newTestProvider(t)
	p.newClient = blockingInit(t)
	p.initTimeout = 50 * time.Millisecond

	// A sentinel of its own: the caller-deadline arm returns
	// context.DeadlineExceeded, and a caller must be able to tell them apart.
	_, err := p.resolveClient(t.Context())
	if !errors.Is(err, ErrClientUnavailable) {
		t.Fatalf("resolveClient() error = %v, want ErrClientUnavailable", err)
	}
	if errors.Is(err, context.DeadlineExceeded) {
		t.Errorf("resolveClient() error = %v, want it distinguishable from a caller deadline", err)
	}
	// Retiring the attempt would start a fresh lookup — and leak a fresh
	// thread-pinning goroutine — every initTimeout, forever.
	p.mu.Lock()
	pending := p.pending
	p.mu.Unlock()
	if pending == nil {
		t.Error("provider dropped the hung attempt; the next caller would start another lookup")
	}
}

// TestResolveClientPublishesLateClient pins that a lookup which lands after every
// waiter gave up is not wasted: the attempt is kept rather than retired precisely
// so the next caller gets its client instead of starting another lookup.
func TestResolveClientPublishesLateClient(t *testing.T) {
	built := &Client{httpClient: http.DefaultClient}
	release := make(chan struct{})
	var attempts atomic.Int32
	p := newTestProvider(t)
	p.initTimeout = 20 * time.Millisecond
	p.newClient = func(context.Context) (*Client, error) {
		attempts.Add(1)
		<-release
		return built, nil
	}

	if _, err := p.resolveClient(t.Context()); !errors.Is(err, ErrClientUnavailable) {
		t.Fatalf("resolveClient() error = %v, want ErrClientUnavailable", err)
	}
	close(release)

	// The attempt finishes on its own goroutine; wait for it to publish.
	deadline := time.Now().Add(2 * time.Second)
	var got *Client
	for time.Now().Before(deadline) {
		if c, err := p.resolveClient(t.Context()); err == nil {
			got = c
			break
		}
		time.Sleep(time.Millisecond)
	}
	if got != built {
		t.Fatalf("resolveClient() = %v, want the late-landing client", got)
	}
	if n := attempts.Load(); n != 1 {
		t.Errorf("init attempts = %d, want 1 (the late client must be used, not rebuilt)", n)
	}
}

// TestRunInitSurvivesAbruptBuilder pins that a builder which does not return
// normally still releases the waiters and the in-flight slot. Without the
// deferred publish, pending stays set with its goroutine dead and every later
// caller waits out initTimeout, forever.
func TestRunInitSurvivesAbruptBuilder(t *testing.T) {
	for _, tc := range []struct {
		name    string
		build   func(context.Context) (*Client, error)
		wantMsg string
	}{
		{
			name:    "panic",
			build:   func(context.Context) (*Client, error) { panic("credentials discovery blew up") },
			wantMsg: "blew up", // surfaced, not swallowed
		},
		{
			// What a t.Fatal inside a caller-supplied builder does.
			name:    "runtime.Goexit",
			build:   func(context.Context) (*Client, error) { runtime.Goexit(); return nil, nil },
			wantMsg: "did not complete",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p := newTestProvider(t)
			p.initTimeout = 5 * time.Second // long enough that a wedge shows up as one
			p.newClient = tc.build

			done := make(chan error, 1)
			go func() { _, err := p.resolveClient(t.Context()); done <- err }()
			select {
			case err := <-done:
				if err == nil || !strings.Contains(err.Error(), tc.wantMsg) {
					t.Fatalf("resolveClient() error = %v, want it to report %q", err, tc.wantMsg)
				}
			case <-time.After(2 * time.Second):
				t.Fatal("resolveClient() blocked; an abrupt builder wedged the provider")
			}
			p.mu.Lock()
			pending := p.pending
			p.mu.Unlock()
			if pending != nil {
				t.Error("provider kept the dead attempt; the next caller would wait on it forever")
			}
		})
	}
}

// TestResolveClientRetriesFailedInit pins that a failed ADC discovery is not
// cached: the doc promises the next call retries, and a provider that wedged on
// a transient environment failure would never recover.
func TestResolveClientRetriesFailedInit(t *testing.T) {
	var attempts atomic.Int32
	p := newTestProvider(t)
	p.newClient = func(context.Context) (*Client, error) {
		if attempts.Add(1) == 1 {
			return nil, errors.New("no credentials")
		}
		return &Client{httpClient: http.DefaultClient}, nil
	}

	if _, err := p.resolveClient(t.Context()); err == nil {
		t.Fatal("resolveClient() = nil error, want the discovery failure")
	}
	if _, err := p.resolveClient(t.Context()); err != nil {
		t.Errorf("resolveClient() after a failed attempt: error = %v, want a retry", err)
	}
	if got := attempts.Load(); got != 2 {
		t.Errorf("init attempts = %d, want 2 (a failure must not be cached)", got)
	}
}

// TestResolveClientRejectsNilClient pins that a builder returning (nil, nil) is
// an error rather than a cached nil that nil-derefs on the next retrieval, inside
// an http.RoundTripper.
func TestResolveClientRejectsNilClient(t *testing.T) {
	p := newTestProvider(t)
	p.newClient = func(context.Context) (*Client, error) { return nil, nil }

	if _, err := p.resolveClient(t.Context()); err == nil {
		t.Fatal("resolveClient() = nil error, want a nil client rejected")
	}
	p.mu.Lock()
	cached := p.client
	p.mu.Unlock()
	if cached != nil {
		t.Errorf("provider cached %v, want nothing", cached)
	}
}

// TestNewProviderIgnoresWiringContextCancellation pins the documented contract
// that NewProvider's ctx supplies values only: the default client outlives any
// one request, so cancelling what was passed at wiring time must not stop it
// being built.
func TestNewProviderIgnoresWiringContextCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(t.Context())
	p, err := NewProvider(ctx, ProviderConfig{Scheme: ProviderScheme{Name: authProviderResource}})
	if err != nil {
		t.Fatalf("NewProvider() error = %v", err)
	}
	cancel()

	built := &Client{httpClient: http.DefaultClient}
	prov := p.(*provider)
	var sawErr error
	prov.newClient = func(ctx context.Context) (*Client, error) {
		sawErr = ctx.Err()
		return built, nil
	}
	got, err := prov.resolveClient(t.Context())
	if err != nil || got != built {
		t.Fatalf("resolveClient() = %v, %v; want the client despite the cancelled wiring context", got, err)
	}
	if sawErr != nil {
		t.Errorf("the builder saw ctx.Err() = %v, want the cancellation stripped", sawErr)
	}
}

// TestNewProviderKeepsWiringContextOnlyWhenLazy pins that a provider given a
// Client does not pin its caller's context — and with it the caller's whole
// session and event graph — for the life of the process.
func TestNewProviderKeepsWiringContextOnlyWhenLazy(t *testing.T) {
	client, err := NewClient(t.Context(), &Config{HTTPClient: http.DefaultClient})
	if err != nil {
		t.Fatalf("NewClient() error = %v", err)
	}
	cfg := ProviderConfig{Scheme: ProviderScheme{Name: authProviderResource}, Client: client}
	p, err := NewProvider(t.Context(), cfg)
	if err != nil {
		t.Fatalf("NewProvider() error = %v", err)
	}
	if got := p.(*provider).initCtx; got != nil {
		t.Errorf("initCtx = %v, want nil when a Client is supplied", got)
	}
	if got := newTestProvider(t).initCtx; got == nil {
		t.Error("initCtx = nil on the lazy path, want the wiring context")
	}
}

// blockingInit returns a client init that never returns until the test ends —
// the shape the init timeout exists for (os.ReadFile on a stalled mount, a hung
// resolver inside metadata.OnGCE).
func blockingInit(t *testing.T) func(context.Context) (*Client, error) {
	t.Helper()
	release := make(chan struct{})
	t.Cleanup(func() { close(release) })
	return func(context.Context) (*Client, error) {
		<-release
		return nil, errors.New("released")
	}
}

// newTestProvider builds a provider with no configured client, so resolveClient
// takes the lazy default-client path.
func newTestProvider(t *testing.T) *provider {
	t.Helper()
	p, err := NewProvider(t.Context(), ProviderConfig{Scheme: ProviderScheme{Name: authProviderResource}})
	if err != nil {
		t.Fatalf("NewProvider() error = %v", err)
	}
	return p.(*provider)
}
