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
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"golang.org/x/oauth2"
)

func TestLazyTokenSourceMemoizesSuccess(t *testing.T) {
	calls := 0
	p := lazyTokenSource(func(context.Context) (oauth2.TokenSource, error) {
		calls++
		return oauth2.StaticTokenSource(&oauth2.Token{AccessToken: "tok"}), nil
	})

	for i := range 3 {
		cred, err := p.Credential(t.Context())
		if err != nil {
			t.Fatalf("call %d: Credential() error = %v", i, err)
		}
		oc, ok := cred.(OAuth2Credential)
		if !ok || oc.TokenSource == nil {
			t.Fatalf("call %d: got %T, want OAuth2Credential with a token source", i, cred)
		}
	}
	if calls != 1 {
		t.Errorf("init called %d times, want 1 (success should be memoized)", calls)
	}
}

func TestLazyTokenSourceRetriesFailure(t *testing.T) {
	calls := 0
	boom := errors.New("boom")
	p := lazyTokenSource(func(context.Context) (oauth2.TokenSource, error) {
		calls++
		if calls == 1 {
			return nil, boom
		}
		return oauth2.StaticTokenSource(&oauth2.Token{AccessToken: "tok"}), nil
	})

	if _, err := p.Credential(t.Context()); !errors.Is(err, boom) {
		t.Fatalf("first call error = %v, want %v", err, boom)
	}
	if _, err := p.Credential(t.Context()); err != nil {
		t.Fatalf("second call error = %v, want nil (failure must not be memoized)", err)
	}
	if calls != 2 {
		t.Errorf("init called %d times, want 2", calls)
	}
}

// TestLazyTokenSourceSingleFlight: concurrent callers must trigger init once.
func TestLazyTokenSourceSingleFlight(t *testing.T) {
	var calls atomic.Int64
	release := make(chan struct{})
	p := lazyTokenSource(func(context.Context) (oauth2.TokenSource, error) {
		calls.Add(1)
		<-release // block in init so concurrent callers pile up waiting for it
		return oauth2.StaticTokenSource(&oauth2.Token{AccessToken: "tok"}), nil
	})

	const n = 50
	var wg sync.WaitGroup
	errs := make([]error, n)
	for i := range n {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_, errs[i] = p.Credential(t.Context())
		}()
	}
	close(release)
	wg.Wait()

	for i, err := range errs {
		if err != nil {
			t.Errorf("goroutine %d: Credential() error = %v", i, err)
		}
	}
	if got := calls.Load(); got != 1 {
		t.Errorf("init called %d times, want 1 (single-flight)", got)
	}
}

// TestLazyTokenSourceHonorsCallerCancellation: a caller whose context is done
// returns promptly instead of blocking on a slow init, while the detached init
// still completes so a later caller reuses it (one init overall).
func TestLazyTokenSourceHonorsCallerCancellation(t *testing.T) {
	var calls atomic.Int64
	release := make(chan struct{})
	p := lazyTokenSource(func(context.Context) (oauth2.TokenSource, error) {
		calls.Add(1)
		<-release // hold init open until the test releases it
		return oauth2.StaticTokenSource(&oauth2.Token{AccessToken: "tok"}), nil
	})

	// A caller with an already-canceled context must not block on the init.
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := p.Credential(canceled); !errors.Is(err, context.Canceled) {
		t.Fatalf("Credential(canceled) error = %v, want context.Canceled", err)
	}

	// Let the detached init finish; a fresh caller then reuses the memoized source.
	close(release)
	cred, err := p.Credential(t.Context())
	if err != nil {
		t.Fatalf("Credential() error = %v", err)
	}
	if oc, ok := cred.(OAuth2Credential); !ok || oc.TokenSource == nil {
		t.Fatalf("got %T, want OAuth2Credential with a token source", cred)
	}
	if got := calls.Load(); got != 1 {
		t.Errorf("init called %d times, want 1 (detached init reused)", got)
	}
}

// TestLazyTokenSourceBoundsHungInit: a hung init is bounded by initTimeout, so it
// fails with a deadline and the provider recovers — a later call retries instead
// of waiting on the same stuck init forever.
func TestLazyTokenSourceBoundsHungInit(t *testing.T) {
	defer func(d time.Duration) { initTimeout = d }(initTimeout)
	initTimeout = 20 * time.Millisecond

	var calls atomic.Int64
	p := lazyTokenSource(func(ctx context.Context) (oauth2.TokenSource, error) {
		calls.Add(1)
		<-ctx.Done() // hang until the bound fires
		return nil, ctx.Err()
	})

	for i := range 2 {
		if _, err := p.Credential(t.Context()); !errors.Is(err, context.DeadlineExceeded) {
			t.Fatalf("call %d: Credential() error = %v, want context.DeadlineExceeded", i, err)
		}
	}
	if got := calls.Load(); got != 2 {
		t.Errorf("init called %d times, want 2 (each call retries after a bounded hang)", got)
	}
}

// TestLazyTokenSourceInitContextOutlivesInit: the context init receives stays
// live after init returns, since the token source refreshes on it.
func TestLazyTokenSourceInitContextOutlivesInit(t *testing.T) {
	defer func(d time.Duration) { initTimeout = d }(initTimeout)
	initTimeout = 100 * time.Millisecond

	var initCtx context.Context
	p := lazyTokenSource(func(ctx context.Context) (oauth2.TokenSource, error) {
		initCtx = ctx
		return oauth2.StaticTokenSource(&oauth2.Token{AccessToken: "tok"}), nil
	})
	if _, err := p.Credential(t.Context()); err != nil {
		t.Fatalf("Credential() error = %v", err)
	}

	select {
	case <-initCtx.Done():
		t.Fatalf("init context done after init returned: %v", initCtx.Err())
	case <-time.After(3 * initTimeout):
	}
}

// TestLazyTokenSourceDiscardsSourceBuiltPastTimeout: a source returned after
// the bound fired is not cached.
func TestLazyTokenSourceDiscardsSourceBuiltPastTimeout(t *testing.T) {
	defer func(d time.Duration) { initTimeout = d }(initTimeout)
	initTimeout = 10 * time.Millisecond

	var calls atomic.Int64
	p := lazyTokenSource(func(ctx context.Context) (oauth2.TokenSource, error) {
		if calls.Add(1) == 1 {
			<-ctx.Done()
		}
		return oauth2.StaticTokenSource(&oauth2.Token{AccessToken: "tok"}), nil
	})

	if _, err := p.Credential(t.Context()); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("first Credential() error = %v, want context.DeadlineExceeded", err)
	}
	if _, err := p.Credential(t.Context()); err != nil {
		t.Fatalf("second Credential() error = %v, want a retried init to succeed", err)
	}
	if got := calls.Load(); got != 2 {
		t.Errorf("init called %d times, want 2 (the late source must not be cached)", got)
	}
}

// TestLazyTokenSourceKeepsInitErrorTextPastTimeout: a timed-out init keeps
// init's error text but does not report as context.Canceled.
func TestLazyTokenSourceKeepsInitErrorTextPastTimeout(t *testing.T) {
	defer func(d time.Duration) { initTimeout = d }(initTimeout)
	initTimeout = 100 * time.Millisecond

	p := lazyTokenSource(func(ctx context.Context) (oauth2.TokenSource, error) {
		<-ctx.Done()
		return nil, fmt.Errorf("auth: find default credentials: %w", ctx.Err())
	})

	_, err := p.Credential(t.Context())
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Errorf("Credential() error = %v, want it to wrap context.DeadlineExceeded", err)
	}
	if errors.Is(err, context.Canceled) {
		t.Errorf("Credential() error = %v, want a timeout not to report as a cancellation", err)
	}
	if !strings.Contains(err.Error(), "find default credentials") {
		t.Errorf("Credential() error = %v, want it to name what init was doing", err)
	}
}

// TestProvidersMintTokensAfterInit: ADC and ServiceAccount mint through an
// authorized_user source, which requests tokens on its construction context.
func TestProvidersMintTokensAfterInit(t *testing.T) {
	var hits atomic.Int64
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		n := hits.Add(1)
		w.Header().Set("Content-Type", "application/json")
		// Inside oauth2's expiry margin, so every Apply refreshes.
		_, _ = fmt.Fprintf(w, `{"access_token":"tok%d","token_type":"Bearer","expires_in":1}`, n)
	}))
	defer srv.Close()

	credsFile := filepath.Join(t.TempDir(), "adc.json")
	creds := fmt.Sprintf(`{"type":"authorized_user","client_id":"id","client_secret":"secret","refresh_token":"refresh","token_uri":%q}`, srv.URL)
	if err := os.WriteFile(credsFile, []byte(creds), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("GOOGLE_APPLICATION_CREDENTIALS", credsFile)

	tests := []struct {
		name     string
		provider CredentialProvider
	}{
		{"ADC", ADC()},
		{"ServiceAccountADCFallback", ServiceAccount(ServiceAccountConfig{})},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			before := hits.Load()
			cred, err := tc.provider.Credential(t.Context())
			if err != nil {
				t.Fatalf("Credential() error = %v", err)
			}
			for i := range 2 {
				h := http.Header{}
				if err := cred.Apply(h); err != nil {
					t.Fatalf("Apply() #%d error = %v", i+1, err)
				}
				if !strings.HasPrefix(h.Get("Authorization"), "Bearer tok") {
					t.Errorf("Apply() #%d Authorization = %q, want a minted bearer token", i+1, h.Get("Authorization"))
				}
			}
			if got := hits.Load() - before; got != 2 {
				t.Errorf("token endpoint hit %d times, want 2 (one mint per Apply)", got)
			}
		})
	}
}
