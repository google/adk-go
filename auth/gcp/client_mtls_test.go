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
	"bytes"
	"crypto/tls"
	"crypto/x509"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"google.golang.org/adk/v2/internal/clientcert/clientcerttest"
)

// TestNewClientMTLSEndpointMode covers the endpoint choice with no certificate
// source, where only GOOGLE_API_USE_MTLS_ENDPOINT can select the mTLS hosts.
func TestNewClientMTLSEndpointMode(t *testing.T) {
	const aiOverride, connOverride = "https://ai.example.test", "https://conn.example.test"
	for _, tc := range []struct {
		name     string
		mode     string
		cfg      Config
		wantAI   string
		wantConn string
	}{
		{name: "unset", mode: "", wantAI: defaultAgentIdentityURL, wantConn: defaultConnectorURL},
		{name: "auto", mode: "auto", wantAI: defaultAgentIdentityURL, wantConn: defaultConnectorURL},
		{name: "never", mode: "never", wantAI: defaultAgentIdentityURL, wantConn: defaultConnectorURL},
		// Literals, so a typo in the constants shows up here.
		{
			name: "always", mode: "always",
			wantAI:   "https://agentidentitycredentials.mtls.googleapis.com",
			wantConn: "https://iamconnectorcredentials.mtls.googleapis.com",
		},
		{name: "case-insensitive", mode: "Always", wantAI: defaultAgentIdentityMTLSURL, wantConn: defaultConnectorMTLSURL},
		{
			name: "agent identity override wins", mode: "always",
			cfg:    Config{AgentIdentityEndpoint: aiOverride},
			wantAI: aiOverride, wantConn: defaultConnectorMTLSURL,
		},
		{
			name: "connector override wins", mode: "always",
			cfg:    Config{ConnectorEndpoint: connOverride},
			wantAI: defaultAgentIdentityMTLSURL, wantConn: connOverride,
		},
		{
			// The caller owns transport and endpoints, so the variable is not
			// read at all: neither always nor an invalid value has an effect.
			name: "HTTPClient set ignores always", mode: "always",
			cfg:    Config{HTTPClient: http.DefaultClient},
			wantAI: defaultAgentIdentityURL, wantConn: defaultConnectorURL,
		},
		{
			name: "HTTPClient set ignores an invalid value", mode: "sometimes",
			cfg:    Config{HTTPClient: http.DefaultClient},
			wantAI: defaultAgentIdentityURL, wantConn: defaultConnectorURL,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			fakeADC(t)
			t.Setenv("GOOGLE_API_USE_MTLS_ENDPOINT", tc.mode)
			c, err := NewClient(t.Context(), &tc.cfg)
			if err != nil {
				t.Fatalf("NewClient() error = %v", err)
			}
			if c.agentIdentityURL != tc.wantAI || c.connectorURL != tc.wantConn {
				t.Errorf("endpoints = %q / %q, want %q / %q", c.agentIdentityURL, c.connectorURL, tc.wantAI, tc.wantConn)
			}
		})
	}
}

func TestNewClientRejectsInvalidMTLSEndpointMode(t *testing.T) {
	fakeADC(t)
	t.Setenv("GOOGLE_API_USE_MTLS_ENDPOINT", "sometimes")
	c, err := NewClient(t.Context(), nil)
	if err == nil {
		t.Fatalf("NewClient() = %+v, nil error; want an error for the invalid mode", c)
	}
	if !strings.Contains(err.Error(), `"sometimes"`) {
		t.Errorf("NewClient() error = %v, want it to name the value", err)
	}
}

// TestNewClientWithCertificateSource runs with a workload certificate config,
// the certificate source that needs nothing but files. See
// [clientcerttest.Isolate] for why it runs in a child process.
func TestNewClientWithCertificateSource(t *testing.T) {
	if !clientcerttest.Isolate(t) {
		return
	}
	workload, leaf := clientcerttest.Workload(t)
	srv, peers := clientcerttest.Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, `{"success":{"token":"tok","header":"Authorization: Bearer"}}`)
	}))
	fakeADC(t)

	// useCert is GOOGLE_API_USE_CLIENT_CERTIFICATE, and config is
	// GOOGLE_API_CERTIFICATE_CONFIG. Without it the only config is the one at
	// gcloud's default path, which the auth library uses and the rule does not.
	for _, tc := range []struct {
		name          string
		useCert       string
		config        string
		mode          string
		wantMTLSHosts bool
		wantCert      bool
	}{
		// First, so that the auth library, which resolves its source once per
		// process, resolves the certificate the cases below compare against.
		{name: "unset, workload config", useCert: clientcerttest.Unset, config: workload, wantMTLSHosts: true, wantCert: true},
		{name: "true", useCert: "true", config: workload, wantMTLSHosts: true, wantCert: true},
		{name: "never keeps the plain hosts and the certificate", useCert: clientcerttest.Unset, config: workload, mode: "never", wantCert: true},
		{name: "false", useCert: "false", config: workload},
		// The auth library alone turns the certificate on for these two.
		{name: "unset, config only at the default path", useCert: clientcerttest.Unset, config: clientcerttest.Unset},
		{name: "1", useCert: "1", config: workload},
	} {
		t.Run(tc.name, func(t *testing.T) {
			clientcerttest.Setenv(t, "GOOGLE_API_USE_CLIENT_CERTIFICATE", tc.useCert)
			clientcerttest.Setenv(t, "GOOGLE_API_CERTIFICATE_CONFIG", tc.config)
			t.Setenv("GOOGLE_API_USE_MTLS_ENDPOINT", tc.mode)

			c, err := NewClient(t.Context(), nil)
			if err != nil {
				t.Fatalf("NewClient() error = %v", err)
			}
			wantAI, wantConn := defaultAgentIdentityURL, defaultConnectorURL
			if tc.wantMTLSHosts {
				wantAI, wantConn = defaultAgentIdentityMTLSURL, defaultConnectorMTLSURL
			}
			if c.agentIdentityURL != wantAI || c.connectorURL != wantConn {
				t.Errorf("endpoints = %q / %q, want %q / %q", c.agentIdentityURL, c.connectorURL, wantAI, wantConn)
			}

			// The same environment, against a server that requests a certificate.
			before := len(peers())
			c, err = NewClient(t.Context(), &Config{AgentIdentityEndpoint: srv.URL, ConnectorEndpoint: srv.URL})
			if err != nil {
				t.Fatalf("NewClient() error = %v", err)
			}
			if _, err := c.RetrieveCredential(t.Context(), Request{Resource: authProviderResource, UserID: "u"}); err != nil {
				t.Fatalf("RetrieveCredential() error = %v", err)
			}
			got := peers()[before:]
			if !tc.wantCert {
				if len(got) != 0 {
					t.Errorf("server saw %d client certificates, want none", len(got))
				}
				return
			}
			if len(got) != 1 || !bytes.Equal(got[0], leaf) {
				t.Errorf("server saw %d client certificates, want exactly the configured one", len(got))
			}
		})
	}
}

// TestNewClientTokenExchangeFollowsTheRule watches where gcloud user
// credentials fetch their token. With the certificate on, that request goes
// to the auth library's mTLS token host. See [clientcerttest.RecordingProxy]
// for how the test sees the host without a request leaving the machine.
func TestNewClientTokenExchangeFollowsTheRule(t *testing.T) {
	if !clientcerttest.Isolate(t) {
		return
	}
	hosts := clientcerttest.RecordingProxy(t)

	workload, _ := clientcerttest.Workload(t)
	isolateADCEnv(t)
	adc := filepath.Join(t.TempDir(), "adc.json")
	clientcerttest.WriteFile(t, adc, []byte(`{"type":"authorized_user","client_id":"id","client_secret":"secret","refresh_token":"refresh"}`))
	t.Setenv("GOOGLE_APPLICATION_CREDENTIALS", adc)

	for _, tc := range []struct {
		name    string
		useCert string
		config  string
		want    string
	}{
		{name: "unset, workload config", useCert: clientcerttest.Unset, config: workload, want: "oauth2.mtls.googleapis.com:443"},
		{name: "false", useCert: "false", config: workload, want: "oauth2.googleapis.com:443"},
		// The auth library alone uses the certificate, and the mTLS token host, here.
		{name: "unset, config only at the default path", useCert: clientcerttest.Unset, config: clientcerttest.Unset, want: "oauth2.googleapis.com:443"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			clientcerttest.Setenv(t, "GOOGLE_API_USE_CLIENT_CERTIFICATE", tc.useCert)
			clientcerttest.Setenv(t, "GOOGLE_API_CERTIFICATE_CONFIG", tc.config)
			before := len(hosts())

			// A loopback endpoint, which the proxy settings never apply to.
			c, err := NewClient(t.Context(), &Config{AgentIdentityEndpoint: "http://127.0.0.1:1"})
			if err != nil {
				t.Fatalf("NewClient() error = %v", err)
			}
			if _, err := c.RetrieveCredential(t.Context(), Request{Resource: authProviderResource, UserID: "u"}); err == nil {
				t.Fatal("RetrieveCredential() succeeded, want the refused token request to fail it")
			}
			got := hosts()[before:]
			if len(got) == 0 {
				t.Fatal("no token request reached the proxy")
			}
			for _, h := range got {
				if h != tc.want {
					t.Errorf("token request went to %q, want %q", h, tc.want)
				}
			}
		})
	}
}

// TestNewClientWithTheCertificateOff covers what the auth library would do on
// its own while the rule leaves the certificate off. The certificate config at
// gcloud's default path is one the library fails on, so NewClient succeeds
// only if the library never looks. The test server is trusted only through
// the TLS configuration of http.DefaultTransport, so a request succeeds only
// if the client keeps that configuration.
func TestNewClientWithTheCertificateOff(t *testing.T) {
	if !clientcerttest.Isolate(t) {
		return
	}
	clientcerttest.WriteFile(t, filepath.Join(os.Getenv("HOME"), ".config/gcloud/certificate_config.json"), []byte(`{"cert_configs":{"workload":{}}}`))

	var certs int
	srv := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		certs += len(r.TLS.PeerCertificates)
		_, _ = io.WriteString(w, `{"success":{"token":"tok","header":"Authorization: Bearer"}}`)
	}))
	srv.TLS = &tls.Config{ClientAuth: tls.RequestClientCert}
	srv.StartTLS()
	t.Cleanup(srv.Close)
	roots := x509.NewCertPool()
	roots.AddCert(srv.Certificate())
	// Safe to change: this child process runs no other test.
	http.DefaultTransport.(*http.Transport).TLSClientConfig = &tls.Config{RootCAs: roots}

	fakeADC(t)
	clientcerttest.Setenv(t, "GOOGLE_API_USE_CLIENT_CERTIFICATE", clientcerttest.Unset)
	clientcerttest.Setenv(t, "GOOGLE_API_CERTIFICATE_CONFIG", clientcerttest.Unset)

	c, err := NewClient(t.Context(), &Config{AgentIdentityEndpoint: srv.URL, ConnectorEndpoint: srv.URL})
	if err != nil {
		t.Fatalf("NewClient() error = %v", err)
	}
	if _, err := c.RetrieveCredential(t.Context(), Request{Resource: authProviderResource, UserID: "u"}); err != nil {
		t.Fatalf("RetrieveCredential() error = %v", err)
	}
	if certs != 0 {
		t.Errorf("server saw %d client certificates, want none", certs)
	}
}

// TestNewClientWithAReplacedDefaultTransport covers an application that put
// something other than an http.Transport in http.DefaultTransport. The auth
// library then builds the base transport itself and asks the client's
// certificate source, which has to answer with no certificate. The home's
// certificate config is valid here, so the library would find a certificate
// if it looked.
func TestNewClientWithAReplacedDefaultTransport(t *testing.T) {
	if !clientcerttest.Isolate(t) {
		return
	}
	srv, peers := clientcerttest.Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, `{"success":{"token":"tok","header":"Authorization: Bearer"}}`)
	}))
	fakeADC(t)
	clientcerttest.Setenv(t, "GOOGLE_API_USE_CLIENT_CERTIFICATE", clientcerttest.Unset)
	clientcerttest.Setenv(t, "GOOGLE_API_CERTIFICATE_CONFIG", clientcerttest.Unset)
	// Safe to change: this child process runs no other test.
	http.DefaultTransport = roundTripperFunc(http.DefaultTransport.RoundTrip)

	c, err := NewClient(t.Context(), &Config{AgentIdentityEndpoint: srv.URL, ConnectorEndpoint: srv.URL})
	if err != nil {
		t.Fatalf("NewClient() error = %v", err)
	}
	if _, err := c.RetrieveCredential(t.Context(), Request{Resource: authProviderResource, UserID: "u"}); err != nil {
		t.Fatalf("RetrieveCredential() error = %v", err)
	}
	if got := peers(); len(got) != 0 {
		t.Errorf("server saw %d client certificates, want none", len(got))
	}
}

func TestPlainTransport(t *testing.T) {
	if got := plainTransport(roundTripperFunc(nil)); got != nil {
		t.Errorf("plainTransport() = %v for a default transport that is not an http.Transport, want nil", got)
	}

	roots := x509.NewCertPool()
	def := &http.Transport{
		TLSClientConfig: &tls.Config{RootCAs: roots},
		HTTP2:           &http.HTTP2Config{PingTimeout: time.Second},
	}
	got, ok := plainTransport(def).(*http.Transport)
	if !ok || got == def {
		t.Fatalf("plainTransport() = %v, want a copy of the default transport", got)
	}
	if got.TLSClientConfig == nil || got.TLSClientConfig.RootCAs != roots {
		t.Error("plainTransport() dropped the default transport's TLS configuration")
	}
	// The values the auth library sets on its own base transport.
	if got.MaxIdleConnsPerHost != 100 || got.HTTP2.SendPingTimeout != 31*time.Second {
		t.Errorf("plainTransport() idle connections per host = %d, ping after = %v, want 100 and 31s", got.MaxIdleConnsPerHost, got.HTTP2.SendPingTimeout)
	}
	if got.HTTP2.PingTimeout != time.Second {
		t.Errorf("plainTransport() ping timeout = %v, want the default transport's 1s kept", got.HTTP2.PingTimeout)
	}
	if def.MaxIdleConnsPerHost != 0 || def.HTTP2.SendPingTimeout != 0 {
		t.Error("plainTransport() changed the default transport")
	}
	if plain, ok := plainTransport(&http.Transport{}).(*http.Transport); !ok || plain.HTTP2 == nil || plain.HTTP2.SendPingTimeout != 31*time.Second {
		t.Error("plainTransport() set no ping for a default transport without HTTP/2 settings")
	}
}

type roundTripperFunc func(*http.Request) (*http.Response, error)

func (f roundTripperFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

// TestNewClientReportsABrokenCertificateConfig runs apart from the tests above
// because the auth library keeps the failure for the rest of the process.
func TestNewClientReportsABrokenCertificateConfig(t *testing.T) {
	if !clientcerttest.Isolate(t) {
		return
	}
	fakeADC(t)
	clientcerttest.Setenv(t, "GOOGLE_API_USE_CLIENT_CERTIFICATE", clientcerttest.Unset)
	t.Setenv("GOOGLE_API_CERTIFICATE_CONFIG", clientcerttest.BrokenWorkload(t))

	c, err := NewClient(t.Context(), nil)
	if err == nil {
		t.Fatalf("NewClient() = %+v, nil error; want an error for a workload section without paths", c)
	}
	if !strings.Contains(err.Error(), "client certificate") {
		t.Errorf("NewClient() error = %q, want it to name the client certificate", err)
	}
}
