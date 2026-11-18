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

package clientcert

import (
	"bytes"
	"path/filepath"
	"strings"
	"testing"

	"google.golang.org/adk/v2/internal/clientcert/clientcerttest"
)

// TestEnabled covers the rule itself, which reads the environment and one
// file and never asks the auth library for a certificate.
func TestEnabled(t *testing.T) {
	workload, _ := clientcerttest.Workload(t)
	dir := t.TempDir()
	file := func(name, content string) string {
		path := filepath.Join(dir, name)
		clientcerttest.WriteFile(t, path, []byte(content))
		return path
	}
	for _, tc := range []struct {
		name    string
		useCert string
		config  string
		want    bool
	}{
		{name: "true", useCert: "true", config: clientcerttest.Unset, want: true},
		{name: "upper case", useCert: "TRUE", config: clientcerttest.Unset, want: true},
		{name: "mixed case", useCert: "tRuE", config: clientcerttest.Unset, want: true},
		{name: "false", useCert: "false", config: clientcerttest.Unset},
		{name: "false wins over a workload config", useCert: "false", config: workload},
		// strconv.ParseBool reads these as true. adk-python does not.
		{name: "1", useCert: "1", config: workload},
		{name: "t", useCert: "t", config: workload},
		{name: "not a boolean", useCert: "yes", config: workload},
		{name: "set but empty", useCert: "", config: workload},

		{name: "unset, workload config", useCert: clientcerttest.Unset, config: workload, want: true},
		{name: "unset, no config", useCert: clientcerttest.Unset, config: clientcerttest.Unset},
		{name: "unset, empty config path", useCert: clientcerttest.Unset, config: ""},
		{name: "unset, missing file", useCert: clientcerttest.Unset, config: filepath.Join(dir, "missing.json")},
		{name: "unset, not JSON", useCert: clientcerttest.Unset, config: file("bad.json", "{")},
		{name: "unset, no workload section", useCert: clientcerttest.Unset, config: file("ecp.json", `{"cert_configs":{"pkcs11":{}},"libs":{}}`)},
		{name: "unset, null workload section", useCert: clientcerttest.Unset, config: file("null.json", `{"cert_configs":{"workload":null}}`)},
		{name: "unset, cert_configs is not an object", useCert: clientcerttest.Unset, config: file("list.json", `{"cert_configs":[]}`)},
		{name: "unset, workload section without paths", useCert: clientcerttest.Unset, config: file("empty.json", `{"cert_configs":{"workload":{}}}`), want: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			clientcerttest.Setenv(t, envUseClientCert, tc.useCert)
			clientcerttest.Setenv(t, envCertConfig, tc.config)
			if got := enabled(); got != tc.want {
				t.Errorf("enabled() = %v, want %v", got, tc.want)
			}
		})
	}
}

// TestSource covers what the rule and the auth library decide together. The
// child's home holds a workload config at gcloud's default path, so with the
// variable unset the auth library alone would always find a certificate.
func TestSource(t *testing.T) {
	if !clientcerttest.Isolate(t) {
		return
	}
	workload, der := clientcerttest.Workload(t)
	// The certificate does not depend on credentials, so Source must not look
	// for any. These cannot be loaded.
	t.Setenv("GOOGLE_APPLICATION_CREDENTIALS", filepath.Join(t.TempDir(), "absent.json"))
	for _, tc := range []struct {
		name     string
		useCert  string
		config   string
		universe string
		want     bool
	}{
		// First, so that the auth library, which resolves its source once per
		// process, resolves the certificate the cases below compare against.
		{name: "unset, workload config", useCert: clientcerttest.Unset, config: workload, want: true},
		{name: "true", useCert: "true", config: workload, want: true},
		{name: "upper case", useCert: "TRUE", config: workload, want: true},
		{name: "capitalized", useCert: "True", config: workload, want: true},
		{name: "true in another universe", useCert: "true", config: workload, universe: "example.com", want: true},

		{name: "false", useCert: "false", config: workload},
		// The auth library alone turns the certificate on for each of these.
		{name: "unset, config only at the default path", useCert: clientcerttest.Unset, config: clientcerttest.Unset},
		{name: "1", useCert: "1", config: workload},
		// The rule is on, and the auth library does not parse the value.
		{name: "mixed case", useCert: "tRuE", config: workload},
		{name: "unset, another universe", useCert: clientcerttest.Unset, config: workload, universe: "example.com"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			clientcerttest.Setenv(t, envUseClientCert, tc.useCert)
			clientcerttest.Setenv(t, envCertConfig, tc.config)
			t.Setenv(envUniverseDomain, tc.universe)
			src, err := Source()
			if err != nil {
				t.Fatalf("Source() error = %v", err)
			}
			if !tc.want {
				if src != nil {
					t.Fatal("Source() returned a certificate source, want nil")
				}
				return
			}
			if src == nil {
				t.Fatal("Source() = nil, want the workload certificate")
			}
			cert, err := src(nil)
			if err != nil {
				t.Fatalf("certificate source: %v", err)
			}
			if len(cert.Certificate) != 1 || !bytes.Equal(cert.Certificate[0], der) {
				t.Error("the source returned a certificate other than the configured one")
			}
		})
	}
}

// TestSourceTakesOtherSourcesOnlyWhenTrue uses the config at gcloud's default
// path as the certificate source that the rule leaves alone while the
// variable is unset. It runs apart from TestSource so that the auth library
// resolves that source and not one named by GOOGLE_API_CERTIFICATE_CONFIG.
func TestSourceTakesOtherSourcesOnlyWhenTrue(t *testing.T) {
	if !clientcerttest.Isolate(t) {
		return
	}
	src, err := Source()
	if err != nil || src != nil {
		t.Fatalf("Source() = (non-nil: %v), %v with the variable unset; want nil, nil", src != nil, err)
	}

	t.Setenv(envUseClientCert, "true")
	src, err = Source()
	if err != nil {
		t.Fatalf("Source() error = %v", err)
	}
	if src == nil {
		t.Fatal("Source() = nil with the variable true, want the certificate at the default path")
	}
	cert, err := src(nil)
	if err != nil {
		t.Fatalf("certificate source: %v", err)
	}
	if len(cert.Certificate) != 1 || !bytes.Equal(cert.Certificate[0], clientcerttest.HomeCertificate(t)) {
		t.Error("the source returned a certificate other than the one at the default path")
	}
}

// TestSourceReportsABrokenConfig runs apart from TestSource because the auth
// library keeps the failure for the rest of the process.
func TestSourceReportsABrokenConfig(t *testing.T) {
	if !clientcerttest.Isolate(t) {
		return
	}
	t.Setenv(envCertConfig, clientcerttest.BrokenWorkload(t))

	src, err := Source()
	if err == nil {
		t.Fatalf("Source() = (non-nil: %v), nil error; want an error for a workload section without paths", src != nil)
	}
	if !strings.HasPrefix(err.Error(), "clientcert: ") {
		t.Errorf("Source() error = %q, want it to name the package", err)
	}
}
