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

// Package clientcerttest gives tests a client certificate source that is safe
// to load on any machine.
package clientcerttest

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
	"math/big"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"
)

const (
	childEnv = "ADK_CLIENTCERT_TEST_CHILD"
	// Where the auth library looks for a certificate config when
	// GOOGLE_API_CERTIFICATE_CONFIG is unset, relative to $HOME.
	defaultConfig = ".config/gcloud/certificate_config.json"
	// A file Isolate leaves in the child's home, which no real one has.
	homeMarker = ".adk-clientcerttest-home"
)

// cleared are the variables a child starts without, so that each test states
// the ones it depends on.
var cleared = []string{
	"GOOGLE_API_USE_CLIENT_CERTIFICATE",
	"GOOGLE_API_CERTIFICATE_CONFIG",
	"GOOGLE_API_USE_MTLS_ENDPOINT",
	"GOOGLE_API_USE_MTLS",
	"GOOGLE_CLOUD_UNIVERSE_DOMAIN",
	"GOOGLE_API_GO_EXPERIMENTAL_DISABLE_NEW_AUTH_LIB",
	"GOOGLE_API_GO_EXPERIMENTAL_ENABLE_NEW_AUTH_LIB",
	"EXPERIMENTAL_GOOGLE_API_USE_S2A",
}

// Isolate re-runs the calling top-level test in a child process and reports
// whether the caller is that child, so a test starts with
//
//	if !clientcerttest.Isolate(t) {
//		return
//	}
//
// A test needs the child when it lets the auth library load a certificate
// source, for two reasons. The library resolves its source once per process
// and keeps the outcome, an error included. And with the variables above
// unset it looks beyond the test's own files: after a workload config comes an
// enterprise certificate proxy and then SecureConnect, whose helper a
// developer machine may really have.
//
// The child therefore gets a home directory holding a workload config at
// gcloud's default path. The library reads $HOME for that path and tries it
// first, so its search ends there whatever the test, or a mutant of the code
// under test, leaves unset.
func Isolate(t *testing.T) bool {
	t.Helper()
	if strings.Contains(t.Name(), "/") {
		t.Fatal("clientcerttest.Isolate: call it from a top-level test")
	}
	if runtime.GOOS == "windows" {
		t.Skip("the auth library takes the default certificate config from %APPDATA% on Windows")
	}
	if os.Getenv(childEnv) == t.Name() {
		// The marker tells the home Isolate made from a real one, should the
		// variable above come from somewhere else.
		for _, name := range []string{homeMarker, defaultConfig} {
			if _, err := os.Stat(filepath.Join(os.Getenv("HOME"), name)); err != nil {
				t.Fatalf("clientcerttest.Isolate: $HOME is not a home Isolate prepared: %v", err)
			}
		}
		return true
	}

	home := t.TempDir()
	path := filepath.Join(home, defaultConfig)
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		t.Fatal(err)
	}
	writeWorkload(t, filepath.Dir(path), path)
	WriteFile(t, filepath.Join(home, homeMarker), nil)

	cmd := exec.CommandContext(t.Context(), os.Args[0], "-test.run=^"+t.Name()+"$", "-test.v", "-test.count=1")
	for _, kv := range os.Environ() {
		if name, _, _ := strings.Cut(kv, "="); !slices.Contains(cleared, name) {
			cmd.Env = append(cmd.Env, kv)
		}
	}
	// Last, so that they win over the inherited values.
	cmd.Env = append(cmd.Env, "HOME="+home, childEnv+"="+t.Name())
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("child process: %v\n%s", err, out)
	}
	// A child that ran nothing exits 0 too, and so does one that skipped, as
	// [Server] does where it cannot make the process trust it.
	switch {
	case bytes.Contains(out, []byte("\n--- PASS: "+t.Name()+" (")):
	case bytes.Contains(out, []byte("\n--- SKIP: "+t.Name()+" (")):
		t.Skipf("the child process skipped the test:\n%s", out)
	default:
		t.Fatalf("child process did not pass the test:\n%s", out)
	}
	return false
}

// Unset, as a value for [Setenv], leaves the variable out of the environment.
// The certificate rule tells that apart from a variable set to the empty
// string.
const Unset = "\x00unset"

// Setenv is [testing.T.Setenv], except that [Unset] removes the variable for
// the duration of the test.
func Setenv(t *testing.T, name, value string) {
	t.Helper()
	// Setenv first, so that the variable is restored when the test ends.
	t.Setenv(name, "")
	if value == Unset {
		if err := os.Unsetenv(name); err != nil {
			t.Fatal(err)
		}
		return
	}
	t.Setenv(name, value)
}

// HomeCertificate returns the DER bytes of the certificate that the config at
// gcloud's default path names, in a child [Isolate] started.
func HomeCertificate(t *testing.T) []byte {
	t.Helper()
	data, err := os.ReadFile(filepath.Join(os.Getenv("HOME"), filepath.Dir(defaultConfig), "cert.pem"))
	if err != nil {
		t.Fatal(err)
	}
	block, _ := pem.Decode(data)
	if block == nil {
		t.Fatal("clientcerttest.HomeCertificate: the certificate file holds no PEM block")
	}
	return block.Bytes
}

// Workload writes a client certificate, its key and a workload certificate
// config naming them. It returns the config's path, for
// GOOGLE_API_CERTIFICATE_CONFIG, and the certificate's DER bytes.
func Workload(t *testing.T) (configPath string, der []byte) {
	t.Helper()
	dir := t.TempDir()
	configPath = filepath.Join(dir, "certificate_config.json")
	return configPath, writeWorkload(t, dir, configPath)
}

// BrokenWorkload writes a certificate config whose workload section names no
// certificate, and returns its path. The auth library reports it as an error
// and keeps that error for the rest of the process.
func BrokenWorkload(t *testing.T) (configPath string) {
	t.Helper()
	configPath = filepath.Join(t.TempDir(), "certificate_config.json")
	WriteFile(t, configPath, []byte(`{"cert_configs":{"workload":{}}}`))
	return configPath
}

func writeWorkload(t *testing.T, dir, configPath string) (der []byte) {
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
	certPath, keyPath := filepath.Join(dir, "cert.pem"), filepath.Join(dir, "key.pem")
	WriteFile(t, certPath, pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: der}))
	WriteFile(t, keyPath, pem.EncodeToMemory(&pem.Block{Type: "PRIVATE KEY", Bytes: pkcs8}))
	WriteFile(t, configPath, fmt.Appendf(nil, `{"cert_configs":{"workload":{"cert_path":%q,"key_path":%q}}}`, certPath, keyPath))
	return der
}

// WriteFile writes data to path, readable by the owner only.
func WriteFile(t *testing.T, path string, data []byte) {
	t.Helper()
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}
}

// Server starts a TLS server that requests a client certificate without
// requiring one, and makes the process trust it. peers returns the DER bytes
// of every certificate clients have presented so far.
//
// Call it before the process makes its first TLS connection: the system roots,
// which SSL_CERT_FILE points at the server's certificate, are read once.
func Server(t *testing.T, h http.Handler) (srv *httptest.Server, peers func() [][]byte) {
	t.Helper()
	if runtime.GOOS == "darwin" || runtime.GOOS == "windows" {
		t.Skip("crypto/x509 reads SSL_CERT_FILE only on Unix-like systems other than darwin")
	}
	var (
		mu   sync.Mutex
		seen [][]byte
	)
	srv = httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		for _, c := range r.TLS.PeerCertificates {
			seen = append(seen, c.Raw)
		}
		mu.Unlock()
		h.ServeHTTP(w, r)
	}))
	srv.TLS = &tls.Config{ClientAuth: tls.RequestClientCert}
	srv.StartTLS()
	t.Cleanup(srv.Close)
	roots := filepath.Join(t.TempDir(), "roots.pem")
	WriteFile(t, roots, pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: srv.Certificate().Raw}))
	t.Setenv("SSL_CERT_FILE", roots)
	return srv, func() [][]byte {
		mu.Lock()
		defer mu.Unlock()
		return append([][]byte(nil), seen...)
	}
}

// RecordingProxy makes the process send its HTTPS requests to a local proxy
// that refuses each one, so that none leaves the machine. hosts returns the
// host and port every request so far asked the proxy for. Requests to a
// loopback address bypass any proxy.
//
// Call it before the process makes its first request: net/http reads the
// proxy variables once.
func RecordingProxy(t *testing.T) (hosts func() []string) {
	t.Helper()
	var (
		mu   sync.Mutex
		seen []string
	)
	proxy := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		seen = append(seen, r.Host)
		mu.Unlock()
		http.Error(w, "refused by the test", http.StatusBadGateway)
	}))
	t.Cleanup(proxy.Close)
	t.Setenv("HTTPS_PROXY", proxy.URL)
	t.Setenv("NO_PROXY", "")
	t.Setenv("no_proxy", "")
	return func() []string {
		mu.Lock()
		defer mu.Unlock()
		return append([]string(nil), seen...)
	}
}
