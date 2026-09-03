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
	"slices"
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

// TestResolveClientFailFastAfterBlownBound pins that a stuck lookup costs the
// bound once, not once per request. The attempt is deliberately kept running, so
// without the latch every outbound request would re-pay the full initTimeout for
// as long as the lookup is stuck.
func TestResolveClientFailFastAfterBlownBound(t *testing.T) {
	p := newTestProvider(t)
	p.newClient = blockingInit(t)
	p.initTimeout = 200 * time.Millisecond

	start := time.Now()
	if _, err := p.resolveClient(t.Context()); !errors.Is(err, ErrClientUnavailable) {
		t.Fatalf("first resolveClient() error = %v, want ErrClientUnavailable", err)
	}
	first := time.Since(start)

	start = time.Now()
	_, err := p.resolveClient(t.Context())
	second := time.Since(start)
	if !errors.Is(err, ErrClientUnavailable) {
		t.Fatalf("second resolveClient() error = %v, want ErrClientUnavailable", err)
	}
	// The fail-fast path does not wait at all, so a quarter of the bound is a
	// generous ceiling and still far under the ~200ms a re-paid bound costs.
	if second > p.initTimeout/4 {
		t.Errorf("second caller waited %v against a %v bound (first paid %v): the blown bound must be latched", second, p.initTimeout, first)
	}
}

// TestResolveClientDiscoveryFailureIsMatchable pins that the common failure —
// discovery not working at all — carries the same sentinel as the timeout. A
// caller behind a RoundTripper cannot switch on a message.
func TestResolveClientDiscoveryFailureIsMatchable(t *testing.T) {
	p := newTestProvider(t)
	p.newClient = func(context.Context) (*Client, error) { return nil, errors.New("no credentials on this host") }

	_, err := p.resolveClient(t.Context())
	if !errors.Is(err, ErrClientUnavailable) {
		t.Fatalf("resolveClient() error = %v, want it to wrap ErrClientUnavailable", err)
	}
	if !strings.Contains(err.Error(), "no credentials on this host") {
		t.Errorf("resolveClient() error = %v, want the underlying cause kept", err)
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

// Two Clients left to Application Default Credentials are still two cache
// dimensions. NewClient resolves ADC afresh on every call and the transport
// underneath it can come from the context, so this package cannot know that two
// of them authenticate as the same principal — and a false miss costs a round
// trip where a false hit discloses one principal's token to another.
func TestADCClientsDoNotShareACacheSlot(t *testing.T) {
	fakeADC(t)
	newADCClient := func() *Client {
		t.Helper()
		c, err := NewClient(t.Context(), nil)
		if err != nil {
			t.Fatalf("NewClient() error = %v", err)
		}
		return c
	}
	first, second := newADCClient(), newADCClient()
	if first.cacheSlot == second.cacheSlot {
		t.Error("two ADC-built Clients share a cache slot, so one would be served the other's credential")
	}
}

// A Client's cache slot must not repeat in another process. A store can outlive
// the process that wrote to it, or be shared by two, and an entry written by a
// Client that no longer exists names an identity nothing can check.
func TestClientCacheSlotIsProcessUnique(t *testing.T) {
	// The width is spelled out rather than derived from nonceBytes: a test that
	// measures the constant against itself moves with it, and a nonce narrowed to
	// a byte would collide between two processes once in 256 while still passing
	// an inequality check almost every run. 128 bits, hex-encoded.
	if len(clientNonce) < 32 {
		t.Fatalf("clientNonce is %d hex characters, want at least 32 (128 bits)", len(clientNonce))
	}
	if clientNonce == newClientNonce() {
		t.Fatal("clientNonce is fixed; two processes constructing Clients in the same order would collide")
	}
	c, err := NewClient(t.Context(), &Config{HTTPClient: &http.Client{}})
	if err != nil {
		t.Fatalf("NewClient() error = %v", err)
	}
	if !strings.Contains(c.cacheSlot, clientNonce) {
		t.Errorf("cache slot %q does not carry the per-process nonce", c.cacheSlot)
	}
}

// joinFields must be injective: no two distinct field lists may encode alike,
// whatever characters the fields contain. This is what stands between a scope
// holding a delimiter and a cross-provider cache hit, and the cache-dimension
// cases in provider_test.go do not pin it on their own — none of the pairs they
// compare uses ":" as a field value, so a ":"-separated join tells them apart.
func TestJoinFieldsIsInjective(t *testing.T) {
	// Every field list that can be built from this alphabet, at every length up to
	// 3. A separator-joining encoding collides inside the set whatever separator
	// it picks, because each separator is itself a field value. The long entries
	// are load-bearing too: with every field under ten bytes the length prefix is
	// a single digit and is self-punctuating, so dropping the ":" would survive.
	long := strings.Repeat("a", 19)
	alphabet := []string{"", "a", ",", "|", ":", "0", "1:", "a,b", "a|b", "23", long, "319" + long}
	var lists [][]string
	var build func(prefix []string, depth int)
	build = func(prefix []string, depth int) {
		lists = append(lists, slices.Clone(prefix))
		if depth == 0 {
			return
		}
		for _, f := range alphabet {
			build(append(prefix, f), depth-1)
		}
	}
	build(nil, 3)

	seen := make(map[string][]string, len(lists))
	for _, l := range lists {
		enc := joinFields(l...)
		if prev, ok := seen[enc]; ok {
			t.Fatalf("joinFields(%q) and joinFields(%q) both encode to %q", prev, l, enc)
		}
		seen[enc] = l
	}
	t.Logf("%d distinct field lists, %d distinct encodings", len(lists), len(seen))
}

// NewClient is called concurrently in production — two providers on the lazy
// path build their default clients on goroutines they own — and a slot handed
// out twice is a cross-principal cache hit.
func TestNewClientSlotsAreUniqueUnderConcurrency(t *testing.T) {
	const n = 32
	slots := make([]string, n)
	var wg sync.WaitGroup
	for i := range n {
		wg.Add(1)
		go func() {
			defer wg.Done()
			c, err := NewClient(t.Context(), &Config{HTTPClient: &http.Client{}})
			if err != nil {
				t.Errorf("NewClient() error = %v", err)
				return
			}
			slots[i] = c.cacheSlot
		}()
	}
	wg.Wait()

	seen := make(map[string]bool, n)
	for _, s := range slots {
		if s == "" {
			t.Fatal("a Client was built with an empty cache slot")
		}
		if seen[s] {
			t.Fatalf("cache slot %q was handed to two Clients", s)
		}
		seen[s] = true
	}
}
