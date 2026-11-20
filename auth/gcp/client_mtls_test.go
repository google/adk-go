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
	"sync/atomic"
	"testing"

	"google.golang.org/adk/v2/internal/clientcert/clientcerttest"
)

// NewClient builds its default client through the Google auth library, which
// loads a client certificate when the shell running the tests turns one on,
// and that can run a certificate helper installed on the machine.
func TestMain(m *testing.M) {
	clientcerttest.PinOff()
	os.Exit(m.Run())
}

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
	home := clientcerttest.HomeCertificate(t)
	srv, peers := clientcerttest.Server(t, credentialHandler)
	fakeADC(t)

	unset := clientcerttest.Unset
	for _, tc := range []struct {
		name          string
		useCert       string // GOOGLE_API_USE_CLIENT_CERTIFICATE
		gcloud        string // CLOUDSDK_CONTEXT_AWARE_USE_CLIENT_CERTIFICATE
		config        string // GOOGLE_API_CERTIFICATE_CONFIG
		mode          string
		wantMTLSHosts bool
		wantCert      []byte
	}{
		{name: "workload config", useCert: unset, gcloud: unset, config: workload, wantMTLSHosts: true, wantCert: leaf},
		{name: "workload config at the default path", useCert: unset, gcloud: unset, config: unset, wantMTLSHosts: true, wantCert: home},
		{name: "true", useCert: "true", gcloud: unset, config: workload, wantMTLSHosts: true, wantCert: leaf},
		{name: "gcloud's variable", useCert: unset, gcloud: "true", config: workload, wantMTLSHosts: true, wantCert: leaf},
		{name: "never keeps the plain hosts and the certificate", useCert: unset, gcloud: unset, config: workload, mode: "never", wantCert: leaf},
		{name: "false", useCert: "false", gcloud: unset, config: workload},
		// The auth library alone turns the certificate on for these two.
		{name: "gcloud's variable false", useCert: unset, gcloud: "false", config: workload},
		{name: "1", useCert: "1", gcloud: unset, config: workload},
	} {
		t.Run(tc.name, func(t *testing.T) {
			clientcerttest.Setenv(t, "GOOGLE_API_USE_CLIENT_CERTIFICATE", tc.useCert)
			clientcerttest.Setenv(t, "CLOUDSDK_CONTEXT_AWARE_USE_CLIENT_CERTIFICATE", tc.gcloud)
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
			if tc.wantCert == nil {
				if len(got) != 0 {
					t.Errorf("server saw %d client certificates, want none", len(got))
				}
				return
			}
			if len(got) != 1 || !bytes.Equal(got[0], tc.wantCert) {
				t.Errorf("server saw %d client certificates, want exactly the configured one", len(got))
			}
		})
	}

	// The certificate does not stand in for credentials.
	t.Run("no credentials", func(t *testing.T) {
		clientcerttest.Setenv(t, "GOOGLE_API_USE_CLIENT_CERTIFICATE", unset)
		t.Setenv("GOOGLE_API_CERTIFICATE_CONFIG", workload)
		t.Setenv("GOOGLE_APPLICATION_CREDENTIALS", filepath.Join(t.TempDir(), "absent.json"))
		if c, err := NewClient(t.Context(), nil); err == nil {
			t.Errorf("NewClient() = %+v, nil error without credentials; want an error", c)
		}
	})
}

// credentialHandler answers a credentials:retrieve call with a token.
var credentialHandler = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
	_, _ = io.WriteString(w, `{"success":{"token":"tok","header":"Authorization: Bearer"}}`)
})

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

	const plain, mtls = "oauth2.googleapis.com:443", "oauth2.mtls.googleapis.com:443"
	unset := clientcerttest.Unset
	for _, tc := range []struct {
		name    string
		useCert string
		gcloud  string
		want    string
	}{
		{name: "workload config", useCert: unset, gcloud: unset, want: mtls},
		{name: "false", useCert: "false", gcloud: unset, want: plain},
		// The auth library alone uses the certificate, and the mTLS token
		// host, for these two.
		{name: "gcloud's variable false", useCert: unset, gcloud: "false", want: plain},
		{name: "1", useCert: "1", gcloud: unset, want: plain},
	} {
		t.Run(tc.name, func(t *testing.T) {
			clientcerttest.Setenv(t, "GOOGLE_API_USE_CLIENT_CERTIFICATE", tc.useCert)
			clientcerttest.Setenv(t, "CLOUDSDK_CONTEXT_AWARE_USE_CLIENT_CERTIFICATE", tc.gcloud)
			t.Setenv("GOOGLE_API_CERTIFICATE_CONFIG", workload)
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

// TestNewClientTokenExchangePresentsTheCertificate has a service-account key
// fetch its token from a server that requests a client certificate, where
// TestNewClientTokenExchangeFollowsTheRule only sees which host is asked.
func TestNewClientTokenExchangePresentsTheCertificate(t *testing.T) {
	if !clientcerttest.Isolate(t) {
		return
	}
	workload, leaf := clientcerttest.Workload(t)
	// One server for the token and for the credential service.
	var scope atomic.Value
	srv, peers := clientcerttest.Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.ParseForm() == nil && r.PostForm.Get("assertion") != "" {
			got, err := assertionScope(r)
			if err != nil {
				t.Errorf("token request: %v", err)
			}
			scope.Store(got)
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, `{"access_token":"ADC-TOKEN","token_type":"Bearer","expires_in":3600}`)
			return
		}
		credentialHandler(w, r)
	}))
	isolateADCEnv(t)
	writeFakeADC(t, srv.URL, "")
	clientcerttest.Setenv(t, "GOOGLE_API_USE_CLIENT_CERTIFICATE", clientcerttest.Unset)
	t.Setenv("GOOGLE_API_CERTIFICATE_CONFIG", workload)

	for _, tc := range []struct {
		name   string
		gcloud string
		// The token request and the call to the service each open a connection.
		want int
	}{
		{name: "workload config", gcloud: clientcerttest.Unset, want: 2},
		// The auth library alone presents the certificate on both.
		{name: "gcloud's variable false", gcloud: "false"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			clientcerttest.Setenv(t, "CLOUDSDK_CONTEXT_AWARE_USE_CLIENT_CERTIFICATE", tc.gcloud)
			before := len(peers())
			scope.Store("")
			c, err := NewClient(t.Context(), &Config{AgentIdentityEndpoint: srv.URL, ConnectorEndpoint: srv.URL})
			if err != nil {
				t.Fatalf("NewClient() error = %v", err)
			}
			if _, err := c.RetrieveCredential(t.Context(), Request{Resource: authProviderResource, UserID: "u"}); err != nil {
				t.Fatalf("RetrieveCredential() error = %v", err)
			}
			// Each of the two branches in NewClient asks for the scope itself.
			if got, want := scope.Load(), "https://www.googleapis.com/auth/cloud-platform"; got != want {
				t.Errorf("token request asked for scope %q, want %q", got, want)
			}
			got := peers()[before:]
			if len(got) != tc.want {
				t.Errorf("server saw %d client certificates, want %d", len(got), tc.want)
			}
			for _, c := range got {
				if !bytes.Equal(c, leaf) {
					t.Error("server saw a certificate other than the configured one")
				}
			}
		})
	}
}

// TestNewClientWithTheCertificateOff covers what the auth library would do on
// its own while the rule leaves the certificate off. The certificate config at
// gcloud's default path is one the library fails on, so NewClient succeeds
// only if neither the rule nor the library reads it. The test server is
// trusted only through the TLS configuration of http.DefaultTransport, so a
// request succeeds only if the client keeps that configuration.
func TestNewClientWithTheCertificateOff(t *testing.T) {
	if !clientcerttest.Isolate(t) {
		return
	}
	clientcerttest.WriteFile(t, filepath.Join(os.Getenv("HOME"), ".config/gcloud/certificate_config.json"), []byte(`{"cert_configs":{"workload":{}}}`))

	var certs int
	srv := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		certs += len(r.TLS.PeerCertificates)
		credentialHandler(w, r)
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
	// With this the auth library alone still has the certificate on.
	t.Setenv("CLOUDSDK_CONTEXT_AWARE_USE_CLIENT_CERTIFICATE", "false")

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
	srv, peers := clientcerttest.Server(t, credentialHandler)
	fakeADC(t)
	clientcerttest.Setenv(t, "GOOGLE_API_USE_CLIENT_CERTIFICATE", clientcerttest.Unset)
	// With this the auth library alone still has the certificate on.
	t.Setenv("CLOUDSDK_CONTEXT_AWARE_USE_CLIENT_CERTIFICATE", "false")
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

// TestNewClientLeavesSecureConnectAlone covers the certificate source that
// runs a helper program: with no variable set the helper must not run, where
// the auth library by itself runs it. Where [clientcerttest.SecureConnect]
// skips the test, the cases above that the auth library alone turns on cover
// the same code with a workload certificate.
func TestNewClientLeavesSecureConnectAlone(t *testing.T) {
	if !clientcerttest.Isolate(t) {
		return
	}
	leaf, ran := clientcerttest.SecureConnect(t)
	srv, peers := clientcerttest.Server(t, credentialHandler)
	fakeADC(t)
	clientcerttest.Setenv(t, "GOOGLE_API_USE_CLIENT_CERTIFICATE", clientcerttest.Unset)

	retrieve := func() *Client {
		t.Helper()
		c, err := NewClient(t.Context(), &Config{AgentIdentityEndpoint: srv.URL, ConnectorEndpoint: srv.URL})
		if err != nil {
			t.Fatalf("NewClient() error = %v", err)
		}
		if _, err := c.RetrieveCredential(t.Context(), Request{Resource: authProviderResource, UserID: "u"}); err != nil {
			t.Fatalf("RetrieveCredential() error = %v", err)
		}
		c, err = NewClient(t.Context(), nil)
		if err != nil {
			t.Fatalf("NewClient() error = %v", err)
		}
		return c
	}

	if c := retrieve(); c.agentIdentityURL != defaultAgentIdentityURL || len(peers()) != 0 || ran() {
		t.Fatalf("with no variable set: agent identity endpoint %q, %d client certificates, helper ran: %v; want the plain endpoint, none and false", c.agentIdentityURL, len(peers()), ran())
	}

	t.Setenv("CLOUDSDK_CONTEXT_AWARE_USE_CLIENT_CERTIFICATE", "true")
	c := retrieve()
	if got := peers(); c.agentIdentityURL != defaultAgentIdentityMTLSURL || len(got) != 1 || !bytes.Equal(got[0], leaf) {
		t.Errorf("with gcloud's variable true: agent identity endpoint %q, %d client certificates; want the mTLS endpoint and the helper's certificate", c.agentIdentityURL, len(got))
	}
}

type roundTripperFunc func(*http.Request) (*http.Response, error)

func (f roundTripperFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

// TestNewClientReportsABrokenCertificateConfig runs in a child process so that
// a NewClient that went on past the error would find only the test's files.
func TestNewClientReportsABrokenCertificateConfig(t *testing.T) {
	if !clientcerttest.Isolate(t) {
		return
	}
	fakeADC(t)
	notJSON := filepath.Join(t.TempDir(), "certificate_config.json")
	clientcerttest.WriteFile(t, notJSON, []byte("{"))
	for _, tc := range []struct {
		name   string
		config string
	}{
		{name: "workload section without paths", config: clientcerttest.BrokenWorkload(t)},
		{name: "not JSON", config: notJSON},
	} {
		t.Run(tc.name, func(t *testing.T) {
			clientcerttest.Setenv(t, "GOOGLE_API_USE_CLIENT_CERTIFICATE", clientcerttest.Unset)
			t.Setenv("GOOGLE_API_CERTIFICATE_CONFIG", tc.config)

			c, err := NewClient(t.Context(), nil)
			if err == nil {
				t.Fatalf("NewClient() = %+v, nil error; want an error", c)
			}
			// The file, and the way out.
			for _, want := range []string{tc.config, "GOOGLE_API_USE_CLIENT_CERTIFICATE=false"} {
				if !strings.Contains(err.Error(), want) {
					t.Errorf("NewClient() error = %q, want it to contain %q", err, want)
				}
			}

			t.Setenv("GOOGLE_API_USE_CLIENT_CERTIFICATE", "false")
			if _, err := NewClient(t.Context(), nil); err != nil {
				t.Errorf("NewClient() error = %v with the certificate turned off, want nil", err)
			}
		})
	}
}
