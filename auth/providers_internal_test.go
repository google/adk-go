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
	"crypto/rand"
	"crypto/rsa"
	"crypto/x509"
	"encoding/base64"
	"encoding/json"
	"encoding/pem"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
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
		wg.Go(func() {
			_, errs[i] = p.Credential(t.Context())
		})
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
// live after init returns, since the source may keep it, as the explicit-key
// one does for the HTTP client it carries.
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

// TestProvidersMintTokensAfterInit: ADC and ServiceAccount mint through a
// service-account source for the scopes they were given, and keep minting once
// the first token has expired, with the context Credential was given canceled
// before the first Apply.
func TestProvidersMintTokensAfterInit(t *testing.T) {
	const tokenTTL = time.Second
	var hits atomic.Int64
	var lastScope atomic.Value
	lastScope.Store("")
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n := hits.Add(1)
		if scope, err := assertionScope(r); err != nil {
			t.Errorf("token request: %v", err)
		} else {
			lastScope.Store(scope)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprintf(w, `{"access_token":"tok%d","token_type":"Bearer","expires_in":%d}`, n, int(tokenTTL.Seconds()))
	}))
	defer srv.Close()

	credsFile := filepath.Join(t.TempDir(), "adc.json")
	if err := os.WriteFile(credsFile, serviceAccountKey(t, srv.URL), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("GOOGLE_APPLICATION_CREDENTIALS", credsFile)
	isolateADCEnv(t)

	const emailScope = "https://www.googleapis.com/auth/userinfo.email"
	tests := []struct {
		name      string
		provider  CredentialProvider
		wantScope string
	}{
		// The literal rather than scopeCloudPlatform, which would move with the
		// constant it is meant to pin.
		{"ADC", ADC(), "https://www.googleapis.com/auth/cloud-platform"},
		{"ADCExplicitScope", ADC(emailScope), emailScope},
		// Empty Scopes on the ADC fallback default to cloud-platform, as the
		// ServiceAccountConfig.Scopes doc promises. Only the explicit JSONKey
		// path rejects them.
		{"ServiceAccountADCFallbackDefaultScope", ServiceAccount(ServiceAccountConfig{}), "https://www.googleapis.com/auth/cloud-platform"},
		{"ServiceAccountADCFallback", ServiceAccount(ServiceAccountConfig{Scopes: []string{emailScope}}), emailScope},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			before := hits.Load()
			ctx, cancel := context.WithCancel(t.Context())
			cred, err := tc.provider.Credential(ctx)
			cancel()
			if err != nil {
				t.Fatalf("Credential() error = %v", err)
			}
			var minted []string
			for i := range 2 {
				if i > 0 {
					// Past the first token's expiry, so this Apply cannot be served
					// from cache. Waiting it out rather than issuing an expired token:
					// the source treats a token with no usable expiry as never expiring.
					time.Sleep(tokenTTL + 100*time.Millisecond)
				}
				h := http.Header{}
				if err := cred.Apply(h); err != nil {
					t.Fatalf("Apply() #%d error = %v", i+1, err)
				}
				if !strings.HasPrefix(h.Get("Authorization"), "Bearer tok") {
					t.Errorf("Apply() #%d Authorization = %q, want a minted bearer token", i+1, h.Get("Authorization"))
				}
				minted = append(minted, h.Get("Authorization"))
			}
			if minted[0] == minted[1] {
				t.Errorf("Apply() reused %q after it expired, want a fresh token", minted[0])
			}
			if got := hits.Load() - before; got != 2 {
				t.Errorf("token endpoint hit %d times, want 2 (one mint per Apply)", got)
			}
			if got := lastScope.Load().(string); got != tc.wantScope {
				t.Errorf("token request asked for scope %q, want %q", got, tc.wantScope)
			}
		})
	}
}

// TestADCKeepsTheTokenExchangeOutOfDebugLogs: GOOGLE_SDK_GO_LOGGING_LEVEL is
// shared with the other Google client libraries in the process, so turning it
// on to debug one of them must not put the ADC provider's signed assertion and
// minted token on stderr.
func TestADCKeepsTheTokenExchangeOutOfDebugLogs(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"access_token":"tok","token_type":"Bearer","expires_in":3600}`)
	}))
	defer srv.Close()
	credsFile := filepath.Join(t.TempDir(), "adc.json")
	if err := os.WriteFile(credsFile, serviceAccountKey(t, srv.URL), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("GOOGLE_APPLICATION_CREDENTIALS", credsFile)
	isolateADCEnv(t)
	t.Setenv("GOOGLE_SDK_GO_LOGGING_LEVEL", "debug")

	for _, tc := range []struct {
		name     string
		provider CredentialProvider
	}{
		{"ADC", ADC()},
		{"ServiceAccountADCFallback", ServiceAccount(ServiceAccountConfig{})},
	} {
		t.Run(tc.name, func(t *testing.T) {
			f, err := os.CreateTemp(t.TempDir(), "stderr")
			if err != nil {
				t.Fatal(err)
			}
			saved := os.Stderr
			os.Stderr = f
			cred, err := tc.provider.Credential(t.Context())
			if err == nil {
				err = cred.Apply(http.Header{})
			}
			os.Stderr = saved
			if err != nil {
				t.Fatalf("minting a token: %v", err)
			}
			if err := f.Close(); err != nil {
				t.Fatal(err)
			}
			out, err := os.ReadFile(f.Name())
			if err != nil {
				t.Fatal(err)
			}
			if len(out) != 0 {
				t.Errorf("stderr with debug logging on = %q, want nothing from the auth library", out)
			}
		})
	}
}

// adcScopesChildEnv marks the child process TestADCDoesNotAliasCallerScopes
// re-executes itself in.
const adcScopesChildEnv = "ADK_AUTH_ADC_SCOPES_CHILD"

// TestADCDoesNotAliasCallerScopes pins the clone in adcTokenSource: the
// metadata-server credential keeps the scopes slice it is given and reads it on
// every refresh, so a caller reusing its slice must not change what later
// tokens are requested for.
//
// It runs in a child process because the metadata-server probe is memoized for
// the life of the process, and this test needs it to answer yes.
func TestADCDoesNotAliasCallerScopes(t *testing.T) {
	if os.Getenv(adcScopesChildEnv) != "1" {
		cmd := exec.CommandContext(t.Context(), os.Args[0], "-test.run=^TestADCDoesNotAliasCallerScopes$", "-test.v", "-test.count=1")
		cmd.Env = append(os.Environ(), adcScopesChildEnv+"=1")
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("child process: %v\n%s", err, out)
		}
		// A child that ran nothing exits 0 too.
		if !strings.Contains(string(out), "--- PASS: TestADCDoesNotAliasCallerScopes") {
			t.Fatalf("child process did not run the test:\n%s", out)
		}
		return
	}

	var mu sync.Mutex
	var asked []string
	md := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !strings.HasSuffix(r.URL.Path, "/token") {
			http.NotFound(w, r)
			return
		}
		mu.Lock()
		asked = append(asked, r.URL.Query().Get("scopes"))
		mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"access_token":"tok","token_type":"Bearer","expires_in":1}`)
	}))
	defer md.Close()
	isolateADCEnv(t)
	// No credentials file anywhere discovery looks, so it falls through to the
	// metadata server, which GCE_METADATA_HOST points at the fake.
	t.Setenv("GOOGLE_APPLICATION_CREDENTIALS", "")
	t.Setenv("HOME", t.TempDir())
	t.Setenv("APPDATA", t.TempDir())
	t.Setenv("GCE_METADATA_HOST", strings.TrimPrefix(md.URL, "http://"))

	const emailScope = "https://www.googleapis.com/auth/userinfo.email"
	scopes := []string{emailScope}
	cred, err := ADC(scopes...).Credential(t.Context())
	if err != nil {
		t.Fatalf("Credential() error = %v", err)
	}
	if err := cred.Apply(http.Header{}); err != nil {
		t.Fatalf("Apply() #1 error = %v", err)
	}
	scopes[0] = "https://example.test/reused-by-the-caller"
	// Past the first token's one-second life, so the second Apply refreshes.
	time.Sleep(1100 * time.Millisecond)
	if err := cred.Apply(http.Header{}); err != nil {
		t.Fatalf("Apply() #2 error = %v", err)
	}

	mu.Lock()
	defer mu.Unlock()
	if want := []string{emailScope, emailScope}; !slices.Equal(asked, want) {
		t.Errorf("token requests asked for scopes %q, want %q", asked, want)
	}
}

// isolateADCEnv pins the rest of the environment the auth library reads, the
// same four variables the gcp package's copy pins. With the service-account-key
// fixture these tests use none of them changes what is sent, since the token is
// applied by OAuth2Credential rather than the library's transports. Other
// credential sources do read some of them on this path, so a future fixture is
// covered too.
func isolateADCEnv(t *testing.T) {
	t.Helper()
	t.Setenv("GOOGLE_CLOUD_QUOTA_PROJECT", "")
	t.Setenv("GOOGLE_CLOUD_UNIVERSE_DOMAIN", "")
	t.Setenv("GOOGLE_API_USE_CLIENT_CERTIFICATE", "false")
	t.Setenv("GOOGLE_AUTH_TRUST_BOUNDARY_ENABLED", "")
}

// assertionScope reads the scope claim out of the signed JWT a service-account
// token request carries as its assertion form field.
func assertionScope(r *http.Request) (string, error) {
	if err := r.ParseForm(); err != nil {
		return "", err
	}
	parts := strings.Split(r.PostForm.Get("assertion"), ".")
	if len(parts) != 3 {
		return "", fmt.Errorf("assertion is not a JWT: %d segments", len(parts))
	}
	payload, err := base64.RawURLEncoding.DecodeString(parts[1])
	if err != nil {
		return "", fmt.Errorf("decode assertion payload: %w", err)
	}
	var claims struct {
		Scope string `json:"scope"`
	}
	if err := json.Unmarshal(payload, &claims); err != nil {
		return "", fmt.Errorf("parse assertion payload: %w", err)
	}
	return claims.Scope, nil
}

// serviceAccountKey is a service-account key file whose token requests go to
// tokenURI. A service-account key rather than an authorized_user credential,
// because credential discovery honors a service-account key's token_uri but
// sends a user credential's token requests to Google's endpoint regardless.
func serviceAccountKey(t *testing.T, tokenURI string) []byte {
	t.Helper()
	key, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		t.Fatal(err)
	}
	der, err := x509.MarshalPKCS8PrivateKey(key)
	if err != nil {
		t.Fatal(err)
	}
	b, err := json.Marshal(map[string]string{
		"type":           "service_account",
		"project_id":     "p",
		"private_key_id": "key-id",
		"private_key":    string(pem.EncodeToMemory(&pem.Block{Type: "PRIVATE KEY", Bytes: der})),
		"client_email":   "sa@p.iam.gserviceaccount.com",
		"client_id":      "1",
		"token_uri":      tokenURI,
	})
	if err != nil {
		t.Fatal(err)
	}
	return b
}
