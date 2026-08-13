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

	p := &provider{scheme: Scheme{Name: authProviderResource}}
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
	release := make(chan struct{})
	defer close(release)
	started := make(chan struct{})
	p := &provider{scheme: Scheme{Name: authProviderResource}}
	// Occupy the singleflight key with an init that does not return.
	go func() {
		_, _, _ = p.clientInit.Do("client", func() (any, error) {
			close(started)
			<-release
			return nil, context.Canceled
		})
	}()
	<-started

	ctx, cancel := context.WithTimeout(t.Context(), 50*time.Millisecond)
	defer cancel()
	start := time.Now()
	if _, err := p.resolveClient(ctx); err == nil {
		t.Fatal("resolveClient() = nil error, want the caller's deadline honored")
	}
	if elapsed := time.Since(start); elapsed > time.Second {
		t.Errorf("resolveClient() returned after %v, want ~the caller's 50ms deadline", elapsed)
	}
}
