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
	"os"
	"strings"

	"cloud.google.com/go/auth/httptransport"
)

const (
	envUseClientCert  = "GOOGLE_API_USE_CLIENT_CERTIFICATE"
	envCertConfig     = "GOOGLE_API_CERTIFICATE_CONFIG"
	envUniverseDomain = "GOOGLE_CLOUD_UNIVERSE_DOMAIN"
)

// Source returns the environment's client certificate source when the rule
// below turns the certificate on. It returns nil when the rule is off, and
// when it is on but the environment has no certificate. An error means the
// rule is on and the configured source is broken.
//
// The rule is adk-python's, as far as the auth library allows:
//
//   - GOOGLE_API_USE_CLIENT_CERTIFICATE, when set, decides: on for true in any
//     letter case, off for anything else.
//   - When it is unset, the certificate is on only if
//     GOOGLE_API_CERTIFICATE_CONFIG names a JSON file with a
//     cert_configs.workload section.
//
// The certificate itself comes from the auth library, whose own gate applies
// on top: it parses a set variable with [strconv.ParseBool], so of the
// spellings of true only true, TRUE and True pass, and it leaves an unset one
// off outside the googleapis.com universe. It ignores a certificate it is
// handed while that gate is off, so the rule cannot be wider than it.
//
// Left to itself the auth library is wider: with the variable unset it also
// uses an enterprise certificate proxy, SecureConnect, or a workload config at
// gcloud's default path. A caller that builds its transport through that
// library has to keep it from doing so when Source returns nil.
//
// Nothing is cached here. The auth library resolves its source once per
// process and keeps the outcome, an error included.
func Source() (func(*tls.CertificateRequestInfo) (*tls.Certificate, error), error) {
	if !enabled() {
		return nil, nil
	}
	// The auth library hands out its certificate source only by building a
	// client, into whose options it writes the source back.
	opts := &httptransport.Options{
		DisableAuthentication: true,
		// The universe the Google API clients pass, so the answer is theirs.
		UniverseDomain: os.Getenv(envUniverseDomain),
	}
	if _, err := httptransport.NewClient(opts); err != nil {
		return nil, fmt.Errorf("clientcert: load the client certificate source: %w", err)
	}
	return opts.ClientCertProvider, nil
}

// enabled is the rule in the [Source] doc, before the auth library's gate.
func enabled() bool {
	if v, ok := os.LookupEnv(envUseClientCert); ok {
		return strings.EqualFold(v, "true")
	}
	data, err := os.ReadFile(os.Getenv(envCertConfig))
	if err != nil {
		return false
	}
	var config struct {
		CertConfigs struct {
			// A pointer, so that a null section reads as absent, as it does to
			// the auth library.
			Workload *json.RawMessage `json:"workload"`
		} `json:"cert_configs"`
	}
	return json.Unmarshal(data, &config) == nil && config.CertConfigs.Workload != nil
}
