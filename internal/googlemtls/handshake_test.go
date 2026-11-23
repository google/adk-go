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

package googlemtls_test

import (
	"context"
	"crypto/tls"
	"errors"
	"net"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"google.golang.org/adk/v2/internal/clientcert/clientcerttest"
	"google.golang.org/adk/v2/internal/googlemtls"
	"google.golang.org/adk/v2/internal/googlemtls/googlemtlstest"
)

type unusedTransport struct{ t *testing.T }

func (u unusedTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	u.t.Errorf("request to %s reached the plain transport", req.URL)
	return nil, errors.New("unexpected plain request")
}

func roundTrip(t *testing.T, rt http.RoundTripper, rawURL string) (*http.Response, error) {
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

func TestHandshakePresentsCertificateToMTLSHost(t *testing.T) {
	for _, tc := range []struct {
		name, url, mtlsEndpoint, wantHost string
	}{
		{name: "plain host is rewritten", url: "https://aiplatform.googleapis.com/v1", wantHost: "aiplatform.mtls.googleapis.com"},
		{name: "port is kept", url: "https://aiplatform.googleapis.com:8443/v1", wantHost: "aiplatform.mtls.googleapis.com:8443"},
		{name: "explicit mtls host is kept", url: "https://aiplatform.mtls.googleapis.com/v1", wantHost: "aiplatform.mtls.googleapis.com"},
		{name: "never keeps the host and still presents", url: "https://aiplatform.mtls.googleapis.com/v1", mtlsEndpoint: "never", wantHost: "aiplatform.mtls.googleapis.com"},
		{name: "never on a plain host", url: "https://aiplatform.googleapis.com/v1", mtlsEndpoint: "never", wantHost: "aiplatform.googleapis.com"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Setenv("GOOGLE_API_USE_MTLS_ENDPOINT", tc.mtlsEndpoint)
			srv := googlemtlstest.NewServer(t)
			srv.Install(t)
			resp, err := roundTrip(t, googlemtls.NewTransport(unusedTransport{t}), tc.url)
			if err != nil {
				t.Fatalf("RoundTrip error = %v", err)
			}
			if resp.StatusCode != http.StatusNoContent {
				t.Fatalf("status = %d, want 204", resp.StatusCode)
			}
			got := <-srv.Requests
			want := googlemtlstest.Request{Host: tc.wantHost, ServerName: strings.Split(tc.wantHost, ":")[0], PeerCerts: 1, Leaf: srv.Leaf()}
			if got != want {
				t.Errorf("server saw %+v, want %+v", got, want)
			}
		})
	}
}

func TestHandshakeProviderErrorNamesOptOut(t *testing.T) {
	t.Setenv("GOOGLE_API_USE_MTLS_ENDPOINT", "")
	srv := googlemtlstest.NewServer(t)
	t.Cleanup(googlemtls.SetTestHooks(func() (func(*tls.CertificateRequestInfo) (*tls.Certificate, error), error) {
		return func(*tls.CertificateRequestInfo) (*tls.Certificate, error) {
			return nil, errors.New("open /certs/cert.pem: no such file")
		}, nil
	}, srv.Configure))
	_, err := roundTrip(t, googlemtls.NewTransport(unusedTransport{t}), "https://aiplatform.googleapis.com/v1")
	if err == nil {
		t.Fatal("RoundTrip error = nil, want the provider error")
	}
	for _, want := range []string{"no such file", "GOOGLE_API_USE_CLIENT_CERTIFICATE=false"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("RoundTrip error = %q, want it to contain %q", err, want)
		}
	}
}

// countingProvider presents srv's client certificate and counts the handshakes
// that asked for it.
func countingProvider(t *testing.T, srv *googlemtlstest.Server) *atomic.Int32 {
	var handshakes atomic.Int32
	t.Cleanup(googlemtls.SetTestHooks(func() (func(*tls.CertificateRequestInfo) (*tls.Certificate, error), error) {
		return func(info *tls.CertificateRequestInfo) (*tls.Certificate, error) {
			handshakes.Add(1)
			return srv.Provider(info)
		}, nil
	}, srv.Configure))
	return &handshakes
}

// The mTLS transport is built once, so a second request finds the first
// one's connection instead of running another handshake.
func TestMTLSConnectionIsReused(t *testing.T) {
	t.Setenv("GOOGLE_API_USE_MTLS_ENDPOINT", "")
	srv := googlemtlstest.NewServer(t)
	handshakes := countingProvider(t, srv)
	tr := googlemtls.NewTransport(unusedTransport{t})
	for range 2 {
		req, err := http.NewRequestWithContext(t.Context(), http.MethodGet, "https://aiplatform.googleapis.com/v1", nil)
		if err != nil {
			t.Fatal(err)
		}
		resp, err := tr.RoundTrip(req)
		if err != nil {
			t.Fatalf("RoundTrip error = %v", err)
		}
		_ = resp.Body.Close()
		<-srv.Requests
	}
	if got := handshakes.Load(); got != 1 {
		t.Errorf("two requests ran %d handshakes, want 1", got)
	}
}

// Concurrent first requests all present the certificate. Run with -race,
// which is what reports an mTLS transport built without the lock.
func TestConcurrentFirstRequestsPresentCertificate(t *testing.T) {
	t.Setenv("GOOGLE_API_USE_MTLS_ENDPOINT", "")
	srv := googlemtlstest.NewServer(t)
	srv.Install(t)
	tr := googlemtls.NewTransport(unusedTransport{t})
	const n = 8
	var wg sync.WaitGroup
	for range n {
		wg.Go(func() {
			req, err := http.NewRequestWithContext(t.Context(), http.MethodGet, "https://aiplatform.googleapis.com/v1", nil)
			if err != nil {
				t.Error(err)
				return
			}
			resp, err := tr.RoundTrip(req)
			if err != nil {
				t.Errorf("RoundTrip error = %v", err)
				return
			}
			_ = resp.Body.Close()
		})
	}
	wg.Wait()
	want := googlemtlstest.Request{Host: "aiplatform.mtls.googleapis.com", ServerName: "aiplatform.mtls.googleapis.com", PeerCerts: 1, Leaf: srv.Leaf()}
	for range n {
		select {
		case got := <-srv.Requests:
			if got != want {
				t.Errorf("server saw %+v, want %+v", got, want)
			}
		default:
			// The server records a request before it answers it.
			t.Fatalf("fewer than %d requests reached the server", n)
		}
	}
}

// A request that is not for a Google API host goes to base, which presents
// nothing, even though a certificate is configured.
func TestNonGoogleHostGetsNoCertificate(t *testing.T) {
	srv := googlemtlstest.NewServer(t)
	srv.Install(t)
	addr := srv.Listener.Addr().String()
	base := &http.Transport{
		DialContext: func(ctx context.Context, network, _ string) (net.Conn, error) {
			var d net.Dialer
			return d.DialContext(ctx, network, addr)
		},
		TLSClientConfig: &tls.Config{RootCAs: srv.Roots, ServerName: "other.googleapis.com"},
	}
	t.Cleanup(base.CloseIdleConnections)
	if _, err := roundTrip(t, googlemtls.NewTransport(base), "https://example.test/v1"); err != nil {
		t.Fatalf("RoundTrip error = %v", err)
	}
	if got := <-srv.Requests; got.PeerCerts != 0 || got.Host != "example.test" {
		t.Errorf("server saw %+v, want Host example.test and no client certificate", got)
	}
}

// The default source is the environment's. With the rule off the transport
// sends Google API requests to base. With it on through the workload config at
// gcloud's default path, which Isolate's child home holds, it presents that
// certificate. The auth library keeps its source for the life of the process,
// so this runs in a child process of its own and loads one source only.
func TestNewTransportUsesEnvironmentCertificate(t *testing.T) {
	if !clientcerttest.Isolate(t) {
		return
	}
	srv := googlemtlstest.NewServer(t)
	srv.Route(t)
	clientcerttest.Setenv(t, "GOOGLE_API_USE_MTLS_ENDPOINT", clientcerttest.Unset)
	clientcerttest.Setenv(t, "GOOGLE_API_CERTIFICATE_CONFIG", clientcerttest.Unset)

	for _, use := range []string{"false", "FALSE", "no"} {
		clientcerttest.Setenv(t, "GOOGLE_API_USE_CLIENT_CERTIFICATE", use)
		base := &countingBase{}
		if _, err := roundTrip(t, googlemtls.NewTransport(base), "https://aiplatform.googleapis.com/v1"); err != nil {
			t.Fatalf("variable %q: RoundTrip error = %v", use, err)
		}
		if base.n != 1 {
			t.Errorf("variable %q: base got %d requests, want 1", use, base.n)
		}
	}

	clientcerttest.Setenv(t, "GOOGLE_API_USE_CLIENT_CERTIFICATE", clientcerttest.Unset)
	if _, err := roundTrip(t, googlemtls.NewTransport(unusedTransport{t}), "https://aiplatform.googleapis.com/v1"); err != nil {
		t.Fatalf("RoundTrip error = %v", err)
	}
	got := <-srv.Requests
	want := googlemtlstest.Request{Host: "aiplatform.mtls.googleapis.com", ServerName: "aiplatform.mtls.googleapis.com", PeerCerts: 1, Leaf: googlemtlstest.Fingerprint(clientcerttest.HomeCertificate(t))}
	if got != want {
		t.Errorf("server saw %+v, want %+v", got, want)
	}
}

// A workload config the auth library cannot load fails a Google API request
// and leaves other requests alone.
func TestNewTransportBrokenEnvironmentCertificate(t *testing.T) {
	if !clientcerttest.Isolate(t) {
		return
	}
	clientcerttest.Setenv(t, "GOOGLE_API_USE_CLIENT_CERTIFICATE", clientcerttest.Unset)
	clientcerttest.Setenv(t, "GOOGLE_API_CERTIFICATE_CONFIG", clientcerttest.BrokenWorkload(t))
	base := &countingBase{}
	tr := googlemtls.NewTransport(base)
	_, err := roundTrip(t, tr, "https://aiplatform.googleapis.com/v1")
	if err == nil || strings.Count(err.Error(), "GOOGLE_API_USE_CLIENT_CERTIFICATE=false") != 1 {
		t.Errorf("RoundTrip error = %v, want the load error naming the opt-out once", err)
	}
	if _, err := roundTrip(t, tr, "https://example.test/"); err != nil || base.n != 1 {
		t.Errorf("non-Google request: error = %v, base got %d requests; want nil, 1", err, base.n)
	}
}

type countingBase struct{ n int }

func (c *countingBase) RoundTrip(req *http.Request) (*http.Response, error) {
	c.n++
	return &http.Response{StatusCode: http.StatusNoContent, Body: http.NoBody, Request: req}, nil
}
