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
	"strings"
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
	// Generous, but far below the initTimeout the attempt itself waits out: a
	// regression that ignores the caller's ctx fails here.
	if elapsed > 20*deadline {
		t.Errorf("resolveClient() returned after %v, want ~the caller's %v deadline", elapsed, deadline)
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

// TestResolveClientRetiresHungInit drives the init timeout: a lookup that never
// returns cannot be cancelled, so the attempt has to be abandoned — otherwise
// every later caller attaches to a flight that never lands.
func TestResolveClientRetiresHungInit(t *testing.T) {
	p := newTestProvider(t)
	p.newClient = blockingInit(t)
	p.initTimeout = 50 * time.Millisecond

	_, err := p.resolveClient(t.Context())
	if err == nil || !strings.Contains(err.Error(), "exceeded") {
		t.Fatalf("resolveClient() error = %v, want the init timeout reported", err)
	}
	p.mu.Lock()
	pending := p.pending
	p.mu.Unlock()
	if pending != nil {
		t.Error("provider still holds the retired attempt; the next caller would wait on it again")
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
	p, err := NewProvider(t.Context(), ProviderScheme{Name: authProviderResource}, nil)
	if err != nil {
		t.Fatalf("NewProvider() error = %v", err)
	}
	return p.(*provider)
}
