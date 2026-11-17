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
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"fmt"
	"io"
	"math/big"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"
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

// certChildEnv marks the child process TestNewClientWithCertificateSource
// re-executes itself in.
const certChildEnv = "ADK_GCP_CERT_SOURCE_CHILD"

// TestNewClientWithCertificateSource runs with a workload certificate config,
// the certificate source that needs nothing but files.
//
// It runs in a child process because two lookups it depends on are cached for
// the life of the process on first use: the auth library's default certificate
// source, and the system roots, which SSL_CERT_FILE points at the test server's
// certificate.
func TestNewClientWithCertificateSource(t *testing.T) {
	if runtime.GOOS == "darwin" || runtime.GOOS == "windows" {
		t.Skip("crypto/x509 reads SSL_CERT_FILE only on Unix-like systems other than darwin")
	}
	if os.Getenv(certChildEnv) != "1" {
		cmd := exec.CommandContext(t.Context(), os.Args[0], "-test.run=^TestNewClientWithCertificateSource$", "-test.v", "-test.count=1")
		cmd.Env = append(os.Environ(), certChildEnv+"=1")
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("child process: %v\n%s", err, out)
		}
		// A child that ran nothing exits 0 too.
		if !bytes.Contains(out, []byte("--- PASS: TestNewClientWithCertificateSource")) {
			t.Fatalf("child process did not run the test:\n%s", out)
		}
		return
	}

	certPEM, keyPEM, leaf := selfSignedCert(t)
	dir := t.TempDir()
	certPath, keyPath := filepath.Join(dir, "cert.pem"), filepath.Join(dir, "key.pem")
	writeFile(t, certPath, certPEM)
	writeFile(t, keyPath, keyPEM)
	configPath := filepath.Join(dir, "certificate_config.json")
	writeFile(t, configPath, fmt.Appendf(nil, `{"cert_configs":{"workload":{"cert_path":%q,"key_path":%q}}}`, certPath, keyPath))

	fakeADC(t)
	t.Setenv("GOOGLE_API_CERTIFICATE_CONFIG", configPath)
	// Unset rather than true: the certificate is used by default for the
	// googleapis.com universe, and that default is what this exercises.
	t.Setenv("GOOGLE_API_USE_CLIENT_CERTIFICATE", "")
	if err := os.Unsetenv("GOOGLE_API_USE_CLIENT_CERTIFICATE"); err != nil {
		t.Fatal(err)
	}

	var (
		mu    sync.Mutex
		peers [][]byte
	)
	srv := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		for _, c := range r.TLS.PeerCertificates {
			peers = append(peers, c.Raw)
		}
		mu.Unlock()
		_, _ = io.WriteString(w, `{"success":{"token":"tok","header":"Authorization: Bearer"}}`)
	}))
	srv.TLS = &tls.Config{ClientAuth: tls.RequestClientCert}
	srv.StartTLS()
	t.Cleanup(srv.Close)
	rootsPath := filepath.Join(dir, "roots.pem")
	writeFile(t, rootsPath, pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: srv.Certificate().Raw}))
	t.Setenv("SSL_CERT_FILE", rootsPath)

	for _, tc := range []struct {
		mode, useCert    string
		wantAI, wantConn string
	}{
		{mode: "", wantAI: defaultAgentIdentityMTLSURL, wantConn: defaultConnectorMTLSURL},
		{mode: "never", wantAI: defaultAgentIdentityURL, wantConn: defaultConnectorURL},
		{mode: "", useCert: "false", wantAI: defaultAgentIdentityURL, wantConn: defaultConnectorURL},
	} {
		t.Run(fmt.Sprintf("mode=%q,use_cert=%q", tc.mode, tc.useCert), func(t *testing.T) {
			t.Setenv("GOOGLE_API_USE_MTLS_ENDPOINT", tc.mode)
			if tc.useCert != "" {
				t.Setenv("GOOGLE_API_USE_CLIENT_CERTIFICATE", tc.useCert)
			}
			c, err := NewClient(t.Context(), nil)
			if err != nil {
				t.Fatalf("NewClient() error = %v", err)
			}
			if c.agentIdentityURL != tc.wantAI || c.connectorURL != tc.wantConn {
				t.Errorf("endpoints = %q / %q, want %q / %q", c.agentIdentityURL, c.connectorURL, tc.wantAI, tc.wantConn)
			}
		})
	}

	t.Run("presents the certificate to an endpoint that requests one", func(t *testing.T) {
		c, err := NewClient(t.Context(), &Config{AgentIdentityEndpoint: srv.URL, ConnectorEndpoint: srv.URL})
		if err != nil {
			t.Fatalf("NewClient() error = %v", err)
		}
		if _, err := c.RetrieveCredential(t.Context(), Request{Resource: authProviderResource, UserID: "u"}); err != nil {
			t.Fatalf("RetrieveCredential() error = %v", err)
		}
		mu.Lock()
		defer mu.Unlock()
		if len(peers) != 1 || !bytes.Equal(peers[0], leaf) {
			t.Errorf("server saw %d client certificates, want exactly the configured one", len(peers))
		}
	})
}

// selfSignedCert returns a PEM certificate and PKCS #8 key, and the
// certificate's DER bytes.
func selfSignedCert(t *testing.T) (certPEM, keyPEM, der []byte) {
	t.Helper()
	key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	tmpl := &x509.Certificate{
		SerialNumber: big.NewInt(1),
		Subject:      pkix.Name{CommonName: "workload"},
		NotBefore:    time.Now().Add(-time.Hour),
		NotAfter:     time.Now().Add(time.Hour),
		ExtKeyUsage:  []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth},
	}
	der, err = x509.CreateCertificate(rand.Reader, tmpl, tmpl, &key.PublicKey, key)
	if err != nil {
		t.Fatal(err)
	}
	pkcs8, err := x509.MarshalPKCS8PrivateKey(key)
	if err != nil {
		t.Fatal(err)
	}
	return pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: der}),
		pem.EncodeToMemory(&pem.Block{Type: "PRIVATE KEY", Bytes: pkcs8}), der
}

func writeFile(t *testing.T, path string, data []byte) {
	t.Helper()
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}
}
