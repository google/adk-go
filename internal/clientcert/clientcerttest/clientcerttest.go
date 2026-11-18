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
	"os/user"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"
)

const (
	childEnv = "ADK_CLIENTCERT_TEST_CHILD"
	// Where gcloud keeps the certificate config, relative to $HOME.
	defaultConfig = ".config/gcloud/certificate_config.json"
	// A file Isolate leaves in the child's home, which no real one has.
	homeMarker = ".adk-clientcerttest-home"
)

// Isolate re-runs the calling top-level test in a child process and reports
// whether the caller is that child, so a test starts with
//
//	if !clientcerttest.Isolate(t) {
//		return
//	}
//
// A test needs the child when it lets the auth library handle a certificate
// source. The library resolves its own source once per process and keeps the
// outcome, an error included. And once the certificate is on, the library
// looks beyond the test's files: at a certificate config the shell names, at
// an enterprise certificate proxy and at SecureConnect, whose helper programs
// a developer machine may really have.
//
// The child therefore starts with none of the parent's environment, and with a
// home directory of its own that holds a workload certificate config at
// gcloud's default path. With nothing set, the certificate is on and is
// [HomeCertificate], and the library's own search, should a mutant of the code
// under test start it, ends at that config.
//
// That holds while GOOGLE_API_CERTIFICATE_CONFIG is unset or names a config
// with a workload section, as [Workload] and [BrokenWorkload] return. The
// library does not go back to the default path from a named config without
// one, so a test must not name such a config with the certificate on.
func Isolate(t *testing.T) bool {
	t.Helper()
	if strings.Contains(t.Name(), "/") {
		t.Fatal("clientcerttest.Isolate: call it from a top-level test")
	}
	if runtime.GOOS == "windows" {
		t.Skip("gcloud's default certificate config is under %APPDATA% on Windows, not under the home directory")
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
	writeWorkload(t, path)
	WriteFile(t, filepath.Join(home, homeMarker), nil)

	// The deadline is the child's own: it outlives a parent that is killed.
	cmd := exec.CommandContext(t.Context(), os.Args[0], "-test.run=^"+t.Name()+"$", "-test.v", "-test.count=1", "-test.timeout=1m")
	// Nothing inherited, and no filter over the parent's variables that could
	// let one through.
	cmd.Env = []string{"HOME=" + home, childEnv + "=" + t.Name()}
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

// PinOff turns the client certificate off for the whole test process, for a
// TestMain. A package needs it when a test outside an [Isolate] child can
// reach the certificate rule: on a machine whose shell turns the certificate
// on, that test would otherwise run the machine's certificate helper.
//
// It leaves a child [Isolate] started alone, so that a test there starts with
// nothing set.
func PinOff() {
	if _, err := os.Stat(filepath.Join(os.Getenv("HOME"), homeMarker)); os.Getenv(childEnv) != "" && err == nil {
		return
	}
	if err := os.Setenv("GOOGLE_API_USE_CLIENT_CERTIFICATE", "false"); err != nil {
		panic(err)
	}
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
	configPath = filepath.Join(t.TempDir(), "certificate_config.json")
	return configPath, writeWorkload(t, configPath)
}

// Rotate replaces the certificate and key that the config from [Workload]
// names, and returns the new certificate's DER bytes.
func Rotate(t *testing.T, configPath string) (der []byte) {
	t.Helper()
	_, _, der = writeCertificate(t, filepath.Dir(configPath))
	return der
}

// BrokenWorkload writes a certificate config whose workload section names no
// certificate, and returns its path.
func BrokenWorkload(t *testing.T) (configPath string) {
	t.Helper()
	configPath = filepath.Join(t.TempDir(), "certificate_config.json")
	WriteFile(t, configPath, []byte(`{"cert_configs":{"workload":{}}}`))
	return configPath
}

// writeWorkload writes a workload certificate config at configPath, and the
// certificate and key it names next to it.
func writeWorkload(t *testing.T, configPath string) (der []byte) {
	t.Helper()
	certPath, keyPath, der := writeCertificate(t, filepath.Dir(configPath))
	WriteFile(t, configPath, fmt.Appendf(nil, `{"cert_configs":{"workload":{"cert_path":%q,"key_path":%q}}}`, certPath, keyPath))
	return der
}

// writeCertificate writes a new client certificate and its key into dir.
func writeCertificate(t *testing.T, dir string) (certPath, keyPath string, der []byte) {
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
	certPath, keyPath = filepath.Join(dir, "cert.pem"), filepath.Join(dir, "key.pem")
	WriteFile(t, certPath, pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: der}))
	WriteFile(t, keyPath, pem.EncodeToMemory(&pem.Block{Type: "PRIVATE KEY", Bytes: pkcs8}))
	return certPath, keyPath, der
}

// SecureConnect replaces the certificate config in the child's home with a
// SecureConnect setup, the certificate source that runs a helper program. It
// returns the DER bytes of the certificate the helper prints, and ran, which
// reports whether the helper has run.
//
// It skips the test unless the auth library would look for SecureConnect in
// that home. The library takes the home directory from the user database when
// it can, as it does under -race, and would then run the machine's real
// helper, if it has one.
func SecureConnect(t *testing.T) (der []byte, ran func() bool) {
	t.Helper()
	home := os.Getenv("HOME")
	if _, err := os.Stat(filepath.Join(home, homeMarker)); err != nil {
		t.Fatalf("clientcerttest.SecureConnect: call it in a child Isolate started: %v", err)
	}
	// Without a user database entry the library needs a user name to take the
	// home directory from $HOME, and the child starts without one.
	t.Setenv("USER", "clientcerttest")
	if u, err := user.Current(); err != nil || u.HomeDir != home {
		t.Skip("the auth library would look for SecureConnect in the machine's home directory, not the test's")
	}
	if err := os.Remove(filepath.Join(home, defaultConfig)); err != nil {
		t.Fatal(err)
	}
	dir := filepath.Join(home, ".secureConnect")
	if err := os.Mkdir(dir, 0o700); err != nil {
		t.Fatal(err)
	}
	certPath, keyPath, der := writeCertificate(t, dir)
	marker, helper := filepath.Join(dir, "ran"), filepath.Join(dir, "helper.sh")
	// The child has no PATH, so the helper names its one program in full.
	script := fmt.Sprintf("#!/bin/sh\necho ran >> %q\n/bin/cat %q %q\n", marker, certPath, keyPath)
	if err := os.WriteFile(helper, []byte(script), 0o700); err != nil {
		t.Fatal(err)
	}
	WriteFile(t, filepath.Join(dir, "context_aware_metadata.json"), fmt.Appendf(nil, `{"cert_provider_command":[%q]}`, helper))
	return der, func() bool {
		_, err := os.Stat(marker)
		return err == nil
	}
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
