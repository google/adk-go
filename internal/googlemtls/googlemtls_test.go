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
	"sync/atomic"
	"testing"
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
	for _, u := range []string{"https://example.test/a", "http://aiplatform.googleapis.com/b"} {
		if _, err := get(t, tr, u); err != nil {
			t.Fatalf("RoundTrip(%s) error = %v", u, err)
		}
	}
	if n := calls.Load(); n != 0 {
		t.Errorf("source called %d times for non-Google requests, want 0", n)
	}
	if len(rec.urls) != 2 {
		t.Errorf("plain transport got %v, want both requests", rec.urls)
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
	src, _ := countingSource(nil, errors.New("bad cert config"))
	tr := &transport{base: failingTransport{t}, source: src}
	body := &closeRecorder{Reader: strings.NewReader("{}")}
	req, err := http.NewRequestWithContext(t.Context(), http.MethodPost, "https://aiplatform.googleapis.com/v1", body)
	if err != nil {
		t.Fatal(err)
	}
	_, err = tr.RoundTrip(req)
	if err == nil {
		t.Fatal("RoundTrip error = nil, want the source error")
	}
	for _, want := range []string{"bad cert config", "GOOGLE_API_USE_CLIENT_CERTIFICATE=false"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("RoundTrip error = %q, want it to contain %q", err, want)
		}
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
	tr := &transport{base: failingTransport{t}, source: func() (provider, error) {
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
	start := time.Now()
	_, err = tr.RoundTrip(req)
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Errorf("RoundTrip error = %v, want context.DeadlineExceeded", err)
	}
	if d := time.Since(start); d > 5*time.Second {
		t.Errorf("RoundTrip returned after %v, want it released by the context", d)
	}
	if !body.closed.Load() {
		t.Error("request body not closed after the context ended")
	}
}

func TestPanickingSourceFailsRequest(t *testing.T) {
	tr := &transport{base: failingTransport{t}, source: func() (provider, error) { panic("boom") }}
	_, err := get(t, tr, "https://aiplatform.googleapis.com/v1")
	if err == nil || !strings.Contains(err.Error(), "boom") {
		t.Errorf("RoundTrip error = %v, want one naming the panic", err)
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

func TestAllHostsRoute(t *testing.T) {
	t.Setenv("GOOGLE_API_USE_MTLS_ENDPOINT", "")
	p := func(*tls.CertificateRequestInfo) (*tls.Certificate, error) { return nil, nil }
	for _, tc := range []struct {
		name     string
		url      string
		provider provider
		wantMTLS bool
		wantHost string
	}{
		{name: "other https host with a certificate", url: "https://proxy.example.test/v1", provider: p, wantMTLS: true, wantHost: "proxy.example.test"},
		{name: "Google host with a certificate", url: "https://aiplatform.googleapis.com/v1", provider: p, wantMTLS: true, wantHost: "aiplatform.mtls.googleapis.com"},
		{name: "other https host without a certificate", url: "https://proxy.example.test/v1"},
		{name: "plain http host with a certificate", url: "http://proxy.example.test/v1", provider: p},
	} {
		t.Run(tc.name, func(t *testing.T) {
			plain, mtls := &recordingTransport{}, &recordingTransport{}
			src, _ := countingSource(tc.provider, nil)
			tr := &transport{base: plain, source: src, allHosts: true}
			// A transport built from a provider is cached, so it can be
			// replaced by one that records what reaches it.
			tr.mtls = &http.Transport{}
			tr.mtls.RegisterProtocol("https", mtls)
			if _, err := get(t, tr, tc.url); err != nil {
				t.Fatalf("RoundTrip error = %v", err)
			}
			if !tc.wantMTLS {
				if len(plain.urls) != 1 || plain.urls[0] != tc.url || len(mtls.urls) != 0 {
					t.Errorf("plain got %v, mTLS got %v; want %s on the plain transport only", plain.urls, mtls.urls, tc.url)
				}
				return
			}
			if len(plain.urls) != 0 || len(mtls.urls) != 1 {
				t.Fatalf("plain got %v, mTLS got %v; want one request on the mTLS transport", plain.urls, mtls.urls)
			}
			if u, err := url.Parse(mtls.urls[0]); err != nil || u.Host != tc.wantHost {
				t.Errorf("mTLS transport got %s, want host %s", mtls.urls[0], tc.wantHost)
			}
		})
	}
}

func TestAllHostsSourceErrorFailsOtherHost(t *testing.T) {
	src, _ := countingSource(nil, errors.New("bad cert config"))
	tr := &transport{base: failingTransport{t}, source: src, allHosts: true}
	if _, err := get(t, tr, "https://proxy.example.test/v1"); err == nil || !strings.Contains(err.Error(), "bad cert config") {
		t.Errorf("RoundTrip error = %v, want the source error", err)
	}
}
