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

package googlemtls

import (
	"context"
	"crypto/tls"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"testing/synctest"
	"time"
)

func TestRoute(t *testing.T) {
	for _, tc := range []struct {
		url        string
		never      bool
		wantGoogle bool
		wantHost   string // "" means not rewritten
	}{
		{url: "https://aiplatform.googleapis.com/v1", wantGoogle: true, wantHost: "aiplatform.mtls.googleapis.com"},
		{url: "https://AIPlatform.GoogleAPIs.com/v1", wantGoogle: true, wantHost: "aiplatform.mtls.googleapis.com"},
		{url: "HTTPS://bigquery.googleapis.com/", wantGoogle: true, wantHost: "bigquery.mtls.googleapis.com"},
		{url: "https://us-central1-aiplatform.googleapis.com:8443/x", wantGoogle: true, wantHost: "us-central1-aiplatform.mtls.googleapis.com:8443"},
		{url: "https://a.b.googleapis.com/", wantGoogle: true, wantHost: "a.b.mtls.googleapis.com"},
		{url: "https://aiplatform.googleapis.com/v1", never: true, wantGoogle: true},
		{url: "https://aiplatform.mtls.googleapis.com/v1", wantGoogle: true},
		{url: "https://AIPLATFORM.MTLS.googleapis.com/v1", wantGoogle: true},
		{url: "https://mtls.googleapis.com/", wantGoogle: true},
		{url: "https://aiplatform.us.rep.googleapis.com/", wantGoogle: true},
		{url: "https://aiplatform.eu.rep.googleapis.com/", wantGoogle: true},
		{url: "https://aiplatform.asia.rep.googleapis.com/", wantGoogle: true, wantHost: "aiplatform.asia.rep.mtls.googleapis.com"},
		{url: "https://googleapis.com/"},
		{url: "https://.googleapis.com/"},
		{url: "http://aiplatform.googleapis.com/"},
		{url: "https://aiplatform.googleapis.com.evil.test/"},
		{url: "https://evil.test/aiplatform.googleapis.com"},
		{url: "https://notgoogleapis.com/"},
		{url: "https://127.0.0.1/"},
		{url: "https://[::1]:443/"},
		// Only an ASCII name counts. strings.ToLower folds U+0130 into i and
		// the Kelvin sign into k, while net/http dials the IDNA form of the
		// name as written, which for the first two is outside
		// googleapis.com.
		{url: "https://aiplatform.mtls.googleap\u0130s.com/"},
		{url: "https://aiplatform.googleap\u0130s.com/"},
		{url: "https://aiplatform\u212a.googleapis.com/"},
		// An IPv6 literal, whose zone can end in any text.
		{url: "https://[2001:db8::1%25.googleapis.com]/"},
		{url: "https://[2001:db8::1%25a.mtls.googleapis.com]/"},
	} {
		t.Run(tc.url, func(t *testing.T) {
			if tc.never {
				t.Setenv("GOOGLE_API_USE_MTLS_ENDPOINT", "Never")
			} else {
				t.Setenv("GOOGLE_API_USE_MTLS_ENDPOINT", "")
			}
			u, err := url.Parse(tc.url)
			if err != nil {
				t.Fatal(err)
			}
			if got := isGoogleAPI(u); got != tc.wantGoogle {
				t.Errorf("isGoogleAPI(%q) = %v, want %v", tc.url, got, tc.wantGoogle)
			}
			if !tc.wantGoogle {
				return
			}
			host, ok := mtlsHost(u)
			if ok != (tc.wantHost != "") || host != tc.wantHost {
				t.Errorf("mtlsHost(%q) = %q, %v, want %q", tc.url, host, ok, tc.wantHost)
			}
		})
	}
}

// failingTransport fails the test when a request reaches it.
type failingTransport struct{ t *testing.T }

func (f failingTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	f.t.Errorf("request to %s reached the plain transport", req.URL)
	return nil, errors.New("unexpected plain request")
}

// recordingTransport answers every request with 204 and records its URL.
type recordingTransport struct{ urls []string }

func (r *recordingTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	r.urls = append(r.urls, req.URL.String())
	if req.Body != nil {
		_ = req.Body.Close()
	}
	return &http.Response{StatusCode: http.StatusNoContent, Body: http.NoBody, Request: req}, nil
}

func countingSource(p provider, err error) (source, *atomic.Int32) {
	var n atomic.Int32
	return func() (provider, error) {
		n.Add(1)
		return p, err
	}, &n
}

func get(t *testing.T, rt http.RoundTripper, rawURL string) (*http.Response, error) {
	t.Helper()
	req, err := http.NewRequestWithContext(t.Context(), http.MethodGet, rawURL, nil)
	if err != nil {
		t.Fatal(err)
	}
	resp, err := rt.RoundTrip(req)
	if err == nil {
		t.Cleanup(func() { _ = resp.Body.Close() })
	}
	return resp, err
}

func TestNonGoogleRequestSkipsSource(t *testing.T) {
	rec := &recordingTransport{}
	src, calls := countingSource(nil, errors.New("must not be called"))
	tr := &transport{base: rec, source: src}
	urls := []string{
		"https://example.test/a",
		"http://aiplatform.googleapis.com/b",
		// Names with a non-ASCII letter.
		"https://aiplatform.mtls.googleap\u0130s.com/c",
		"https://aiplatform.googleap\u0130s.com/d",
		"https://aiplatform\u212a.googleapis.com/e",
		// An IP address whose zone ends like one.
		"https://[2001:db8::1%25.googleapis.com]/f",
	}
	for _, u := range urls {
		if _, err := get(t, tr, u); err != nil {
			t.Fatalf("RoundTrip(%s) error = %v", u, err)
		}
	}
	if n := calls.Load(); n != 0 {
		t.Errorf("source called %d times for non-Google requests, want 0", n)
	}
	if len(rec.urls) != len(urls) {
		t.Errorf("plain transport got %v, want all %d requests", rec.urls, len(urls))
	}
}

func TestNoProviderPassesThroughUnchanged(t *testing.T) {
	t.Setenv("GOOGLE_API_USE_MTLS_ENDPOINT", "")
	rec := &recordingTransport{}
	src, calls := countingSource(nil, nil)
	tr := &transport{base: rec, source: src}
	const u = "https://aiplatform.googleapis.com/v1"
	for range 2 {
		if _, err := get(t, tr, u); err != nil {
			t.Fatalf("RoundTrip error = %v", err)
		}
	}
	if len(rec.urls) != 2 || rec.urls[0] != u {
		t.Errorf("plain transport got %v, want %q twice", rec.urls, u)
	}
	// The source is asked again for the second request, so a rule turned
	// off or on later in the process takes effect.
	if n := calls.Load(); n != 2 {
		t.Errorf("source called %d times, want 2", n)
	}
}

func TestSourceErrorFailsRequestAndClosesBody(t *testing.T) {
	srcErr := errors.New("bad cert config")
	src, _ := countingSource(nil, srcErr)
	tr := &transport{base: failingTransport{t}, source: src}
	body := &closeRecorder{Reader: strings.NewReader("{}")}
	req, err := http.NewRequestWithContext(t.Context(), http.MethodPost, "https://aiplatform.googleapis.com/v1", body)
	if err != nil {
		t.Fatal(err)
	}
	_, err = tr.RoundTrip(req)
	if !errors.Is(err, srcErr) {
		t.Fatalf("RoundTrip error = %v, want the source error", err)
	}
	if !body.closed.Load() {
		t.Error("request body not closed after the source failed")
	}
}

type closeRecorder struct {
	io.Reader
	closed atomic.Bool
}

func (c *closeRecorder) Close() error {
	c.closed.Store(true)
	return nil
}

func TestHangingSourceReleasesRequestOnContextEnd(t *testing.T) {
	release := make(chan struct{})
	t.Cleanup(func() { close(release) })
	base := &countingTransport{}
	tr := &transport{base: base, source: func() (provider, error) {
		<-release
		return nil, nil
	}}
	ctx, cancel := context.WithTimeout(t.Context(), 50*time.Millisecond)
	defer cancel()
	body := &closeRecorder{Reader: strings.NewReader("{}")}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, "https://aiplatform.googleapis.com/v1", body)
	if err != nil {
		t.Fatal(err)
	}
	done := make(chan error, 1)
	go func() {
		_, err := tr.RoundTrip(req)
		done <- err
	}()
	select {
	case err := <-done:
		if !errors.Is(err, context.DeadlineExceeded) || !strings.Contains(err.Error(), "GOOGLE_API_USE_CLIENT_CERTIFICATE=false") {
			t.Errorf("RoundTrip error = %v, want context.DeadlineExceeded with the opt-out", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("RoundTrip still waits for the source 5s after its context ended")
	}
	if n := base.n.Load(); n != 0 {
		t.Errorf("plain transport got %d requests after the context ended, want 0", n)
	}
	if !body.closed.Load() {
		t.Error("request body not closed after the context ended")
	}
}

type countingTransport struct{ n atomic.Int32 }

func (c *countingTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	c.n.Add(1)
	return &http.Response{StatusCode: http.StatusNoContent, Body: http.NoBody, Request: req}, nil
}

// Requests that arrive while a lookup is under way wait for it, so a
// certificate helper that hangs holds one goroutine and not one per request.
func TestConcurrentRequestsShareOneLookup(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		release := make(chan struct{})
		var calls atomic.Int32
		base := &countingTransport{}
		tr := &transport{base: base, source: func() (provider, error) {
			calls.Add(1)
			<-release
			return nil, nil
		}}
		const n = 8
		var wg sync.WaitGroup
		for range n {
			wg.Go(func() {
				req, err := http.NewRequestWithContext(t.Context(), http.MethodGet, "https://aiplatform.googleapis.com/v1", nil)
				if err != nil {
					t.Error(err)
					return
				}
				if _, err := tr.RoundTrip(req); err != nil {
					t.Errorf("RoundTrip error = %v", err)
				}
			})
		}
		synctest.Wait()
		if got := calls.Load(); got != 1 {
			t.Errorf("source called %d times for %d concurrent requests, want 1", got, n)
		}
		close(release)
		wg.Wait()
		if got := base.n.Load(); got != n {
			t.Errorf("plain transport got %d requests, want %d", got, n)
		}
	})
}

func TestPanickingSourceFailsRequest(t *testing.T) {
	tr := &transport{base: failingTransport{t}, source: func() (provider, error) { panic("boom") }}
	_, err := get(t, tr, "https://aiplatform.googleapis.com/v1")
	if err == nil || !strings.Contains(err.Error(), "boom") || !strings.Contains(err.Error(), "GOOGLE_API_USE_CLIENT_CERTIFICATE=false") {
		t.Errorf("RoundTrip error = %v, want one naming the panic and the opt-out", err)
	}
}

func TestGuardProxy(t *testing.T) {
	req := httptest.NewRequest(http.MethodGet, "https://aiplatform.mtls.googleapis.com/", nil)
	for _, tc := range []struct {
		proxy   string
		wantErr bool
	}{
		{proxy: "", wantErr: false},
		{proxy: "http://proxy.test:3128", wantErr: false},
		{proxy: "https://proxy.test:3129", wantErr: true},
		{proxy: "HTTPS://proxy.test:3129", wantErr: true},
	} {
		t.Run(tc.proxy, func(t *testing.T) {
			guarded := guardProxy(func(*http.Request) (*url.URL, error) {
				if tc.proxy == "" {
					return nil, nil
				}
				return url.Parse(tc.proxy)
			})
			u, err := guarded(req)
			if (err != nil) != tc.wantErr {
				t.Fatalf("guarded proxy = %v, %v; want error %v", u, err, tc.wantErr)
			}
			if !tc.wantErr && tc.proxy != "" && (u == nil || u.String() != tc.proxy) {
				t.Errorf("guarded proxy = %v, want %s", u, tc.proxy)
			}
		})
	}
}

// The mTLS transport keeps http.DefaultTransport's proxy setting, guarded.
func TestMTLSTransportGuardsInheritedProxy(t *testing.T) {
	orig := http.DefaultTransport
	t.Cleanup(func() { http.DefaultTransport = orig })
	req := httptest.NewRequest(http.MethodGet, "https://aiplatform.mtls.googleapis.com/", nil)
	p := func(*tls.CertificateRequestInfo) (*tls.Certificate, error) { return nil, nil }

	http.DefaultTransport = &http.Transport{Proxy: func(*http.Request) (*url.URL, error) {
		return url.Parse("https://proxy.test:3129")
	}}
	tr := newMTLSTransport(p)
	if u, err := tr.Proxy(req); err == nil {
		t.Errorf("Proxy() = %v, nil; want the https proxy refused", u)
	}

	http.DefaultTransport = &http.Transport{}
	if tr := newMTLSTransport(p); tr.Proxy != nil {
		t.Error("Proxy is set although http.DefaultTransport has none")
	}

	// An application's wrapper hides the proxy setting, so the transport
	// takes the environment's, as net/http's own default does.
	http.DefaultTransport = &recordingTransport{}
	if tr := newMTLSTransport(p); tr.Proxy == nil {
		t.Error("Proxy is not set although http.DefaultTransport is a wrapper")
	}
}

func TestPlainHost(t *testing.T) {
	for _, tc := range []struct{ host, want string }{
		{"agent.mtls.googleapis.com", "agent.googleapis.com"},
		{"Agent.MTLS.GoogleAPIs.com", "agent.googleapis.com"},
		{"a.b.mtls.googleapis.com", "a.b.googleapis.com"},
		{"agent.asia.rep.mtls.googleapis.com", "agent.asia.rep.googleapis.com"},
		{"Agent.GoogleAPIs.com", "agent.googleapis.com"},
		{"agent.mtls.example.com", "agent.mtls.example.com"},
		// Not the mTLS name of any host: mtlsHost never produces these.
		{"mtls.googleapis.com", "mtls.googleapis.com"},
		{".mtls.googleapis.com", ".mtls.googleapis.com"},
		{"mtls.mtls.googleapis.com", "mtls.mtls.googleapis.com"},
		{"agent.mtls.mtls.googleapis.com", "agent.mtls.mtls.googleapis.com"},
		{"agent.us.rep.mtls.googleapis.com", "agent.us.rep.mtls.googleapis.com"},
		{"agent.eu.rep.mtls.googleapis.com", "agent.eu.rep.mtls.googleapis.com"},
	} {
		if got := plainHost(tc.host); got != tc.want {
			t.Errorf("plainHost(%q) = %q, want %q", tc.host, got, tc.want)
		}
	}
}

func TestTwins(t *testing.T) {
	for _, tc := range []struct {
		a, b string
		want bool
	}{
		{"agent.googleapis.com", "agent.mtls.googleapis.com", true},
		{"Agent.GoogleAPIs.com", "agent.MTLS.googleapis.com", true},
		{"example.com", "Example.COM", true},
		{"agent.googleapis.com", "other.mtls.googleapis.com", false},
		{"agent.example.com", "agent.mtls.example.com", false},
		{"agent.us.rep.googleapis.com", "agent.us.rep.mtls.googleapis.com", false},
		{"mtls.googleapis.com", "mtls.mtls.googleapis.com", false},
		{".googleapis.com", ".mtls.googleapis.com", false},
		// Only ASCII names: U+0130 and the Kelvin sign lowercase into i and
		// k, and other letters lowercase alike without being one host.
		{"agent.googleapis.com", "agent.mtls.googleap\u0130s.com", false},
		{"agent.mtls.googleapis.com", "agent.googleap\u0130s.com", false},
		{"agentk.googleapis.com", "agent\u212a.mtls.googleapis.com", false},
		{"agent\u212a.googleapis.com", "agentk.mtls.googleapis.com", false},
		{"B\u00dcCHER.example", "b\u00fccher.example", false},
		// An IPv6 literal, whose zone can end in any text.
		{"2001:db8::1%a.googleapis.com", "2001:db8::1%a.mtls.googleapis.com", false},
		{"::1", "::1", false},
	} {
		if got := Twins(tc.a, tc.b); got != tc.want {
			t.Errorf("Twins(%q, %q) = %v, want %v", tc.a, tc.b, got, tc.want)
		}
		if got := Twins(tc.b, tc.a); got != tc.want {
			t.Errorf("Twins(%q, %q) = %v, want %v", tc.b, tc.a, got, tc.want)
		}
	}
}

// plainHost undoes exactly what mtlsHost does: a name maps back to a host
// only when the transport sends that host's requests to it. The hosts are
// every sequence of up to three labels, out of the ones either function looks
// for and two others, under each suffix. TestPlainHost and TestTwins have
// what this leaves out: upper case, an empty label, non-ASCII names and IP
// literals.
func TestPlainHostInvertsMTLSHost(t *testing.T) {
	t.Setenv("GOOGLE_API_USE_MTLS_ENDPOINT", "")
	labels := []string{"a", "mtls", "us", "eu", "asia", "rep"}
	var prefixes []string
	var build func(prefix string, depth int)
	build = func(prefix string, depth int) {
		prefixes = append(prefixes, prefix)
		if depth == 3 {
			return
		}
		for _, l := range labels {
			build(prefix+l+".", depth+1)
		}
	}
	build("", 0)
	mapped := 0
	for _, prefix := range prefixes {
		for _, suffix := range []string{"googleapis.com", "mtls.googleapis.com", "example.com"} {
			host := prefix + suffix
			u := &url.URL{Scheme: "https", Host: host}
			if isGoogleAPI(u) {
				if m, ok := mtlsHost(u); ok {
					mapped++
					if got := plainHost(m); got != host {
						t.Errorf("mtlsHost(%q) = %q, but plainHost(%q) = %q", host, m, m, got)
					}
				}
			}
			if plain := plainHost(host); plain != host {
				pu := &url.URL{Scheme: "https", Host: plain}
				if m, ok := mtlsHost(pu); !isGoogleAPI(pu) || !ok || m != host {
					t.Errorf("plainHost(%q) = %q, but mtlsHost(%q) = %q, %v", host, plain, plain, m, ok)
				}
			}
		}
	}
	// A guard that the loop reaches hosts the transport rewrites.
	if mapped == 0 {
		t.Error("no generated host has an mTLS name")
	}
}

func TestNewTransportNilBaseUsesCurrentDefaultTransport(t *testing.T) {
	rec := &recordingTransport{}
	orig := http.DefaultTransport
	http.DefaultTransport = rec
	t.Cleanup(func() { http.DefaultTransport = orig })
	tr := NewTransport(nil)
	if _, err := get(t, tr, "https://example.test/"); err != nil {
		t.Fatalf("RoundTrip error = %v", err)
	}
	if len(rec.urls) != 1 {
		t.Errorf("http.DefaultTransport got %v, want the request", rec.urls)
	}
}
