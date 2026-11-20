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
	"net/http"
	"testing"

	"google.golang.org/api/option"
	htransport "google.golang.org/api/transport/http"

	"google.golang.org/adk/v2/internal/clientcert/clientcerttest"
)

// TestAPIOptions checks the options by what a client built with them does.
//
// No case sets GOOGLE_API_USE_CLIENT_CERTIFICATE to true, here or in the
// packages that use the options. With it google.golang.org/api consults its
// own certificate sources whenever the options hand it no certificate, and
// those end at the SecureConnect of the machine's real home directory.
// TestSource covers true, and APIOptions does not look at how the rule came
// to be on.
func TestAPIOptions(t *testing.T) {
	if !clientcerttest.Isolate(t) {
		return
	}
	workload, der := clientcerttest.Workload(t)
	home := clientcerttest.HomeCertificate(t)
	srv, peers := clientcerttest.Server(t, http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
	unset := clientcerttest.Unset
	for _, tc := range []struct {
		name       string
		useCert    string
		gcloud     string
		config     string
		mode       string
		oldMode    string
		newAuthOff string
		caller     []option.ClientOption
		universe   string
		wantCert   []byte
		wantMTLS   bool
	}{
		{name: "workload config", useCert: unset, gcloud: unset, config: workload, wantCert: der, wantMTLS: true},
		{name: "workload config at the default path", useCert: unset, gcloud: unset, config: unset, wantCert: home, wantMTLS: true},
		{name: "gcloud's variable", useCert: unset, gcloud: "true", config: workload, wantCert: der, wantMTLS: true},
		{name: "auto", useCert: unset, gcloud: unset, config: workload, mode: "AUTO", wantCert: der, wantMTLS: true},
		{name: "never", useCert: unset, gcloud: unset, config: workload, mode: "never", wantCert: der},
		// The library picks the mTLS endpoint itself.
		{name: "always", useCert: unset, gcloud: unset, config: workload, mode: "always", wantCert: der},
		{name: "not a mode", useCert: unset, gcloud: unset, config: workload, mode: "sometimes", wantCert: der},
		{name: "old variable", useCert: unset, gcloud: unset, config: workload, oldMode: "never", wantCert: der},
		{name: "new variable wins over the old one", useCert: unset, gcloud: unset, config: workload, mode: "auto", oldMode: "never", wantCert: der, wantMTLS: true},
		{name: "newer auth library not switched off", useCert: unset, gcloud: unset, config: workload, newAuthOff: "false", wantCert: der, wantMTLS: true},
		{name: "newer auth library switched off", useCert: unset, gcloud: unset, config: workload, newAuthOff: "true"},
		{name: "an HTTP client from the caller", useCert: unset, gcloud: unset, config: workload, caller: []option.ClientOption{option.WithHTTPClient(&http.Client{})}},
		{name: "other options from the caller", useCert: unset, gcloud: unset, config: workload, caller: []option.ClientOption{option.WithQuotaProject("p"), option.WithEndpoint("https://example.test")}, wantCert: der, wantMTLS: true},
		{name: "another universe from the caller", useCert: unset, gcloud: unset, config: workload, caller: []option.ClientOption{option.WithUniverseDomain("example.com")}},
		{name: "another universe from the environment", useCert: unset, gcloud: unset, config: workload, universe: "example.com"},

		{name: "false", useCert: "false", gcloud: unset, config: workload},
		{name: "gcloud's variable false", useCert: unset, gcloud: "false", config: workload},
	} {
		t.Run(tc.name, func(t *testing.T) {
			clientcerttest.Setenv(t, envUseClientCert, tc.useCert)
			clientcerttest.Setenv(t, envGcloudUseClientCert, tc.gcloud)
			clientcerttest.Setenv(t, envCertConfig, tc.config)
			t.Setenv(envMTLSEndpoint, tc.mode)
			t.Setenv(envMTLSEndpointOld, tc.oldMode)
			t.Setenv(envNewAuthLibOff, tc.newAuthOff)
			t.Setenv(envUniverseDomain, tc.universe)

			opts, mtls, err := APIOptions(t.Context(), tc.caller...)
			if err != nil {
				t.Fatalf("APIOptions() error = %v", err)
			}
			if mtls != tc.wantMTLS {
				t.Errorf("APIOptions() mtlsEndpoint = %v, want %v", mtls, tc.wantMTLS)
			}
			// As a client passes them: the returned options, then its caller's.
			client, _, err := htransport.NewClient(t.Context(), append(append(opts, tc.caller...), option.WithoutAuthentication(), option.WithEndpoint(srv.URL))...)
			if err != nil {
				t.Fatalf("NewClient() error = %v", err)
			}
			before := len(peers())
			resp, err := client.Get(srv.URL)
			if err != nil {
				t.Fatalf("Get() error = %v", err)
			}
			_ = resp.Body.Close()
			got := peers()[before:]
			if tc.wantCert == nil {
				if len(opts) != 0 {
					t.Errorf("APIOptions() returned %d options, want none", len(opts))
				}
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
}

// TestAPIOptionsReportsABrokenConfig runs in a child process so that an
// APIOptions that went on past the error would find only the test's files.
func TestAPIOptionsReportsABrokenConfig(t *testing.T) {
	if !clientcerttest.Isolate(t) {
		return
	}
	t.Setenv(envCertConfig, clientcerttest.BrokenWorkload(t))

	if opts, mtls, err := APIOptions(t.Context()); err == nil {
		t.Errorf("APIOptions() = %d options, %v, nil error; want an error for a workload section without paths", len(opts), mtls)
	}
	// A client the caller built is left alone, so its config is not read.
	if _, _, err := APIOptions(t.Context(), option.WithHTTPClient(&http.Client{})); err != nil {
		t.Errorf("APIOptions() error = %v with an HTTP client from the caller, want nil", err)
	}
}

func TestCallerSettings(t *testing.T) {
	// The probe goes through the api library, which looks for a certificate of
	// its own when this is true.
	t.Setenv(envUseClientCert, "false")
	t.Setenv(envMTLSEndpoint, "")
	t.Setenv(envMTLSEndpointOld, "")
	client := option.WithHTTPClient(&http.Client{})
	for _, tc := range []struct {
		name         string
		opts         []option.ClientOption
		env          string // GOOGLE_CLOUD_UNIVERSE_DOMAIN
		wantClient   bool
		wantUniverse string
	}{
		{name: "no options", wantUniverse: "googleapis.com"},
		{name: "no options, another universe in the environment", env: "example.com", wantUniverse: "example.com"},
		{name: "an HTTP client", opts: []option.ClientOption{client}, wantClient: true, wantUniverse: "googleapis.com"},
		{name: "an HTTP client among others", opts: []option.ClientOption{option.WithEndpoint("https://example.test"), client, option.WithUserAgent("ua")}, wantClient: true, wantUniverse: "googleapis.com"},
		{name: "credentials", opts: []option.ClientOption{option.WithAPIKey("key")}, wantUniverse: "googleapis.com"},
		{name: "no authentication", opts: []option.ClientOption{option.WithoutAuthentication()}, wantUniverse: "googleapis.com"},
		// No HTTP client goes with a quota project, so the probe's own fails.
		{name: "a quota project", opts: []option.ClientOption{option.WithQuotaProject("p")}, wantUniverse: "googleapis.com"},
		{name: "a quota project, another universe in the environment", opts: []option.ClientOption{option.WithQuotaProject("p")}, env: "example.com", wantUniverse: "example.com"},
		{name: "another universe", opts: []option.ClientOption{option.WithUniverseDomain("example.com")}, wantUniverse: "example.com"},
		{name: "another universe in the environment", opts: []option.ClientOption{option.WithUserAgent("ua")}, env: "example.com", wantUniverse: "example.com"},
		{name: "the option wins over the environment", opts: []option.ClientOption{option.WithUniverseDomain("googleapis.com")}, env: "example.com", wantUniverse: "googleapis.com"},
		{name: "an endpoint of the caller's hides the option's universe", opts: []option.ClientOption{option.WithUniverseDomain("example.com"), option.WithEndpoint("https://example.test")}, wantUniverse: "googleapis.com"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Setenv(envUniverseDomain, tc.env)
			gotClient, gotUniverse := callerSettings(t.Context(), tc.opts)
			if gotClient != tc.wantClient || gotUniverse != tc.wantUniverse {
				t.Errorf("callerSettings() = %v, %q, want %v, %q", gotClient, gotUniverse, tc.wantClient, tc.wantUniverse)
			}
		})
	}
}
