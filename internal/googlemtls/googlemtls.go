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

// Package googlemtls sends requests for Google API hosts over mutual TLS when
// the environment provides a client certificate.
//
// A certificate-bound access token is accepted only on a connection that
// presented the matching certificate, and Google API hosts ask for one only on
// their *.mtls.googleapis.com names. This matches adk-python's
// google.adk.utils._mtls_utils, which certificate-enabled clients use to route
// *.googleapis.com requests to the mTLS hosts.
package googlemtls

import (
	"context"
	"crypto/tls"
	"fmt"
	"net"
	"net/http"
	"net/url"
	"os"
	"strings"
	"sync"
	"time"

	"google.golang.org/adk/v2/internal/clientcert"
)

const (
	googleAPIsSuffix = ".googleapis.com"
	mtlsSuffix       = ".mtls.googleapis.com"
	optOut           = "set GOOGLE_API_USE_CLIENT_CERTIFICATE=false to send Google API requests without it"
)

// Multi-region hosts have no mTLS names, so they are never rewritten. Same
// list as adk-python's _NO_MTLS_GOOGLEAPIS_SUFFIXES.
var noMTLSSuffixes = []string{".us.rep.googleapis.com", ".eu.rep.googleapis.com"}

// provider follows crypto/tls.Config.GetClientCertificate.
type provider = func(*tls.CertificateRequestInfo) (*tls.Certificate, error)

// source returns the provider to present to Google API hosts, nil when none
// is to be presented, or an error when a certificate the environment
// configures cannot be loaded. A transport keeps the first provider it gets:
// the auth library resolves its source once per process, so later calls
// return the same one.
type source func() (provider, error)

// defaultSource is the environment's source, under the rule clientcert.Source
// documents.
var defaultSource source = clientcert.Source

// testConfigure is applied to the mTLS transport of every transport created
// while it is set. See SetTestHooks.
var testConfigure func(*http.Transport)

// SetTestHooks makes transports created by NewTransport after it returns use
// src as their certificate source, the environment's when src is nil, and
// apply configure to their mTLS transport, until restore is called. It is for
// tests in other packages, which must not run in parallel while the hooks are
// set.
func SetTestHooks(src func() (func(*tls.CertificateRequestInfo) (*tls.Certificate, error), error), configure func(*http.Transport)) (restore func()) {
	prevSource, prevConfigure := defaultSource, testConfigure
	if src != nil {
		defaultSource = src
	}
	testConfigure = configure
	return func() { defaultSource, testConfigure = prevSource, prevConfigure }
}

// NewTransport returns a RoundTripper that sends an https request for a
// *.googleapis.com host over mutual TLS when the environment provides a client
// certificate, to the host's .mtls.googleapis.com name. Every other request,
// and every Google API request while the environment provides no certificate,
// goes to base unchanged. A Google API request fails when a certificate the
// environment configures cannot be loaded. A nil base stands for
// http.DefaultTransport, looked up on each request.
//
// The mTLS requests do not go through base: they use a clone of
// http.DefaultTransport, taken at the first such request, that presents the
// certificate and refuses an https proxy, which would be offered the
// certificate in its own handshake.
func NewTransport(base http.RoundTripper) http.RoundTripper {
	return &transport{base: base, source: defaultSource, configure: testConfigure}
}

// NewTransportAllHosts is [NewTransport], except that while the environment
// provides a certificate every https request goes over mutual TLS, so any
// host that asks for the certificate gets it. Only *.googleapis.com hosts are
// moved to their mTLS names. This is what google-genai Python, which
// adk-python's Gemini model uses, does for a Vertex AI client.
func NewTransportAllHosts(base http.RoundTripper) http.RoundTripper {
	return &transport{base: base, source: defaultSource, configure: testConfigure, allHosts: true}
}

type transport struct {
	base     http.RoundTripper
	source   source
	allHosts bool
	// configure, when set, is applied to the mTLS transport after it is
	// built. Tests use it to reach a local server.
	configure func(*http.Transport)

	mu   sync.Mutex
	call *lookup
	mtls *http.Transport
}

// lookup is one run of source that concurrent requests share.
type lookup struct {
	done chan struct{}
	p    provider
	err  error
}

func (t *transport) RoundTrip(req *http.Request) (*http.Response, error) {
	google := isGoogleAPI(req.URL)
	if !google && (!t.allHosts || !isHTTPS(req.URL)) {
		return t.plain().RoundTrip(req)
	}
	p, err := t.provider(req.Context())
	if err != nil {
		if req.Body != nil {
			_ = req.Body.Close()
		}
		return nil, err
	}
	if p == nil {
		return t.plain().RoundTrip(req)
	}
	mtls := t.mtlsTransport(p)
	if host, ok := mtlsHost(req.URL); google && ok {
		req = req.Clone(req.Context())
		req.URL.Host = host
		// net/http writes the Host header and the HTTP/2 :authority from
		// req.Host, which NewRequest fills in, so the URL alone would leave the
		// plain name there. adk-python sets the header the same way.
		req.Host = host
	}
	return mtls.RoundTrip(req)
}

func (t *transport) plain() http.RoundTripper {
	if t.base != nil {
		return t.base
	}
	return http.DefaultTransport
}

// provider runs source unless a run is already under way, which concurrent
// requests share, so the rule that decides whether a certificate is used is
// read again rather than cached. It runs source on a goroutine of its own and
// waits for it under ctx: a certificate helper runs with no context and must
// not hold a request past its deadline.
func (t *transport) provider(ctx context.Context) (provider, error) {
	t.mu.Lock()
	c := t.call
	if c == nil {
		c = &lookup{done: make(chan struct{})}
		t.call = c
		go t.run(c)
	}
	t.mu.Unlock()
	select {
	case <-c.done:
		return c.p, c.err
	case <-ctx.Done():
		return nil, fmt.Errorf("googlemtls: waiting for the client certificate source: %w", context.Cause(ctx))
	}
}

func (t *transport) run(c *lookup) {
	defer func() {
		if r := recover(); r != nil {
			c.p, c.err = nil, fmt.Errorf("googlemtls: the client certificate source panicked: %v", r)
		}
		t.mu.Lock()
		t.call = nil
		t.mu.Unlock()
		close(c.done)
	}()
	p, err := t.source()
	if err != nil {
		c.err = fmt.Errorf("googlemtls: loading the client certificate: %w (%s)", err, optOut)
		return
	}
	c.p = p
}

func (t *transport) mtlsTransport(p provider) *http.Transport {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.mtls == nil {
		t.mtls = newMTLSTransport(p)
		if t.configure != nil {
			t.configure(t.mtls)
		}
	}
	return t.mtls
}

func newMTLSTransport(p provider) *http.Transport {
	var tr *http.Transport
	if dt, ok := http.DefaultTransport.(*http.Transport); ok {
		tr = dt.Clone()
	} else {
		// The settings net/http creates http.DefaultTransport with, for an
		// application that replaced it with a wrapper.
		tr = &http.Transport{
			Proxy:                 http.ProxyFromEnvironment,
			DialContext:           (&net.Dialer{Timeout: 30 * time.Second, KeepAlive: 30 * time.Second}).DialContext,
			ForceAttemptHTTP2:     true,
			MaxIdleConns:          100,
			IdleConnTimeout:       90 * time.Second,
			TLSHandshakeTimeout:   10 * time.Second,
			ExpectContinueTimeout: time.Second,
		}
	}
	if tr.TLSClientConfig == nil {
		tr.TLSClientConfig = &tls.Config{}
	}
	tr.TLSClientConfig.GetClientCertificate = func(info *tls.CertificateRequestInfo) (*tls.Certificate, error) {
		c, err := p(info)
		if err != nil {
			return nil, fmt.Errorf("googlemtls: loading the client certificate: %w (%s)", err, optOut)
		}
		return c, nil
	}
	if tr.Proxy != nil {
		tr.Proxy = guardProxy(tr.Proxy)
	}
	return tr
}

// guardProxy refuses an https proxy. net/http runs the handshake with such a
// proxy using the transport's TLSClientConfig, so the proxy could ask for the
// client certificate and get it. An http proxy only tunnels the connection.
func guardProxy(proxy func(*http.Request) (*url.URL, error)) func(*http.Request) (*url.URL, error) {
	return func(req *http.Request) (*url.URL, error) {
		u, err := proxy(req)
		if u != nil && strings.EqualFold(u.Scheme, "https") {
			return nil, fmt.Errorf("googlemtls: refusing https proxy %s for a request that presents the client certificate", u.Redacted())
		}
		return u, err
	}
}

// isGoogleAPI reports whether u is an https URL on a *.googleapis.com host.
func isGoogleAPI(u *url.URL) bool {
	if !isHTTPS(u) {
		return false
	}
	h := strings.ToLower(u.Hostname())
	return len(h) > len(googleAPIsSuffix) && strings.HasSuffix(h, googleAPIsSuffix)
}

func isHTTPS(u *url.URL) bool {
	return u != nil && strings.EqualFold(u.Scheme, "https")
}

// PlainHost returns host lowercased, with a *.mtls.googleapis.com name
// replaced by the Google API host it serves. A transport from NewTransport
// may send a request for one to the other, so a check comparing hosts
// should compare these.
func PlainHost(host string) string {
	h := strings.ToLower(host)
	if label, ok := strings.CutSuffix(h, mtlsSuffix); ok {
		return label + googleAPIsSuffix
	}
	return h
}

// mtlsHost returns the mTLS name for u's host, keeping its port, and false
// when u is already on one, has none, or GOOGLE_API_USE_MTLS_ENDPOINT=never
// asks to keep the host as given.
func mtlsHost(u *url.URL) (string, bool) {
	if strings.EqualFold(os.Getenv("GOOGLE_API_USE_MTLS_ENDPOINT"), "never") {
		return "", false
	}
	h := strings.ToLower(u.Hostname())
	if h == "mtls.googleapis.com" || strings.HasSuffix(h, mtlsSuffix) {
		return "", false
	}
	for _, s := range noMTLSSuffixes {
		if strings.HasSuffix(h, s) {
			return "", false
		}
	}
	h = strings.TrimSuffix(h, googleAPIsSuffix) + mtlsSuffix
	if port := u.Port(); port != "" {
		return net.JoinHostPort(h, port), true
	}
	return h, true
}
