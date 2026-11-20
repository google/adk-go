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

// Package clientcert decides when ADK's own Google API clients use the
// environment's client certificate.
package clientcert

import (
	"crypto/tls"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"cloud.google.com/go/auth/httptransport"
	"golang.org/x/net/http2"
)

const (
	envUseClientCert       = "GOOGLE_API_USE_CLIENT_CERTIFICATE"
	envGcloudUseClientCert = "CLOUDSDK_CONTEXT_AWARE_USE_CLIENT_CERTIFICATE"
	envCertConfig          = "GOOGLE_API_CERTIFICATE_CONFIG"
	envUniverseDomain      = "GOOGLE_CLOUD_UNIVERSE_DOMAIN"

	// Ends the error for a certificate config that cannot be used.
	optOut = envUseClientCert + "=false turns the client certificate off"
)

type source = func(*tls.CertificateRequestInfo) (*tls.Certificate, error)

// Source returns the environment's client certificate source when the rule
// below turns the certificate on. It returns nil when the rule is off, and
// when it is on but the environment has no certificate. An error means a
// certificate config that cannot be used, or a certificate source the auth
// library fails on.
//
// The rule is adk-python's, as far as the auth library can load the
// certificate:
//
//   - GOOGLE_API_USE_CLIENT_CERTIFICATE, when set, decides: on for true in any
//     letter case, off for anything else.
//   - Otherwise CLOUDSDK_CONTEXT_AWARE_USE_CLIENT_CERTIFICATE, the variable
//     gcloud reads, decides in the same way unless it is empty.
//   - Otherwise the certificate is on only if a certificate config has a
//     workload section. The config is the file GOOGLE_API_CERTIFICATE_CONFIG
//     names, or the one at gcloud's default path.
//
// A workload section names the certificate, and that certificate is the one
// returned, so the decision and the certificate come from one file. Without
// such a section, a variable that turns the certificate on leaves the choice
// to the auth library: an enterprise certificate proxy, then SecureConnect.
//
// The auth library's own gate comes first, because the library ignores a
// certificate it is handed while that gate is off, and then no config is
// read. It parses a set GOOGLE_API_USE_CLIENT_CERTIFICATE with
// [strconv.ParseBool], so of the spellings of true only true, TRUE and True
// pass. With that variable unset it leaves the certificate off outside the
// googleapis.com universe.
//
// Left to itself the auth library is wider: with the variable unset it uses a
// proxy or SecureConnect without being asked. A caller that builds its
// transport through that library has to keep it from doing so when Source
// returns nil.
//
// Nothing is cached here: every call reads the environment and the config
// again. The auth library resolves its own source once per process and keeps
// the outcome, an error included.
func Source() (func(*tls.CertificateRequestInfo) (*tls.Certificate, error), error) {
	on, decided := switched()
	if decided && !on {
		return nil, nil
	}
	// Asked with a certificate in hand, the library answers for its gate
	// alone and searches for nothing.
	if gate, err := authLibrary(NoCertificate); gate == nil || err != nil {
		return nil, err
	}
	workload, err := workloadSource()
	if workload != nil || err != nil {
		return workload, err
	}
	if !decided {
		return nil, nil
	}
	return authLibrary(nil)
}

// NoCertificate answers a server's certificate request with no certificate.
// Handed to the auth library as a source, it also stops the library's own
// search for one, which can run a certificate helper.
func NoCertificate(*tls.CertificateRequestInfo) (*tls.Certificate, error) {
	return &tls.Certificate{}, nil
}

// PlainTransport returns the base transport the auth library builds from def,
// the default transport, when it has no certificate source. Given a source,
// even [NoCertificate], the library replaces the TLS configuration an
// application set on the default transport.
//
// It returns nil when def is not an [http.Transport]. The library then starts
// from a transport of its own, which has no configuration to lose. The second
// result is the HTTP/2 side of the transport, for the tests.
func PlainTransport(def http.RoundTripper) (http.RoundTripper, *http2.Transport) {
	t, ok := def.(*http.Transport)
	if !ok {
		return nil, nil
	}
	// The same steps as the library's (cloud.google.com/go/auth v0.24.4,
	// httptransport.defaultBaseTransport): a larger idle pool, HTTP/2
	// configured by golang.org/x/net/http2, and a health check that drops a
	// connection gone quiet.
	t = t.Clone()
	t.MaxIdleConnsPerHost = 100
	h2, err := http2.ConfigureTransports(t)
	if err == nil {
		h2.ReadIdleTimeout = 31 * time.Second
	}
	return t, h2
}

// switched reports what the two variables say, and whether either says
// anything.
func switched() (on, decided bool) {
	if v, ok := os.LookupEnv(envUseClientCert); ok {
		return strings.EqualFold(v, "true"), true
	}
	if v := os.Getenv(envGcloudUseClientCert); v != "" {
		return strings.EqualFold(v, "true"), true
	}
	return false, false
}

// workloadSource returns the certificate the certificate config's workload
// section names, and nil when no config has such a section. A config that is
// missing or cannot be read counts as no config, as it does to adk-python.
func workloadSource() (source, error) {
	path := os.Getenv(envCertConfig)
	if path == "" {
		path = defaultConfigPath(runtime.GOOS)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, nil
	}
	var config struct {
		CertConfigs struct {
			// A pointer, so that a null section reads as absent, as it does to
			// the auth library.
			Workload *struct {
				CertPath string `json:"cert_path"`
				KeyPath  string `json:"key_path"`
			} `json:"workload"`
		} `json:"cert_configs"`
	}
	if err := json.Unmarshal(data, &config); err != nil {
		return nil, fmt.Errorf("clientcert: certificate config %s: %w (%s)", path, err, optOut)
	}
	w := config.CertConfigs.Workload
	if w == nil {
		return nil, nil
	}
	if w.CertPath == "" || w.KeyPath == "" {
		return nil, fmt.Errorf("clientcert: certificate config %s: the workload section needs cert_path and key_path (%s)", path, optOut)
	}
	// Read for every handshake, so that a rotated certificate is picked up.
	return func(*tls.CertificateRequestInfo) (*tls.Certificate, error) {
		cert, err := tls.LoadX509KeyPair(w.CertPath, w.KeyPath)
		if err != nil {
			return nil, fmt.Errorf("clientcert: load the workload certificate: %w", err)
		}
		return &cert, nil
	}, nil
}

// defaultConfigPath is where gcloud keeps the certificate config on the
// system goos names, or "" when the directory it lives under is not known.
func defaultConfigPath(goos string) string {
	dir := os.Getenv("APPDATA")
	if goos != "windows" {
		home, err := os.UserHomeDir()
		if err != nil {
			return ""
		}
		dir = filepath.Join(home, ".config")
	}
	if dir == "" {
		return ""
	}
	return filepath.Join(dir, "gcloud", "certificate_config.json")
}

// authLibrary returns the certificate source the auth library gives a
// transport that is handed given: given itself, the source of its own search
// when given is nil, or nil when its gate is off.
//
// It is a variable so that the tests of the rule can stand in for the
// library, whose own search can run a certificate helper.
var authLibrary = func(given source) (source, error) {
	// The library hands out its certificate source only by building a client,
	// into whose options it writes the source back.
	opts := &httptransport.Options{
		DisableAuthentication: true,
		// The universe the Google API clients pass, so the answer is theirs.
		UniverseDomain:     os.Getenv(envUniverseDomain),
		ClientCertProvider: given,
	}
	if _, err := httptransport.NewClient(opts); err != nil {
		return nil, fmt.Errorf("clientcert: load the client certificate source: %w (%s)", err, optOut)
	}
	return opts.ClientCertProvider, nil
}
