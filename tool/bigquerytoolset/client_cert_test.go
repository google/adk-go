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

package bigquerytoolset

import (
	"bytes"
	"crypto/tls"
	"net/http"
	"os"
	"testing"

	"golang.org/x/oauth2"
	"google.golang.org/api/impersonate"
	"google.golang.org/api/option"

	"google.golang.org/adk/v2/internal/clientcert/clientcerttest"
)

// The toolset builds its client through the Google auth library, which loads
// a client certificate when the shell running the tests turns one on, and
// that can run a certificate helper installed on the machine.
func TestMain(m *testing.M) {
	clientcerttest.PinOff()
	os.Exit(m.Run())
}

// TestNewClientFollowsTheCertificateRule builds the BigQuery client with a
// workload client certificate configured. See [clientcerttest.Isolate] for why
// it runs in a child process, and TestAPIOptions in internal/clientcert for
// why no case sets GOOGLE_API_USE_CLIENT_CERTIFICATE to true.
func TestNewClientFollowsTheCertificateRule(t *testing.T) {
	if !clientcerttest.Isolate(t) {
		return
	}
	workload, der := clientcerttest.Workload(t)
	srv, peers := clientcerttest.Server(t, http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
	const (
		plain    = "https://bigquery.googleapis.com/bigquery/v2/"
		mtls     = "https://bigquery.mtls.googleapis.com/bigquery/v2/"
		explicit = "https://bigquery.example.test/bigquery/v2/"
	)

	own := &http.Client{}
	callerCert, callerDER := clientcerttest.Certificate(t)
	callerSource := func(*tls.CertificateRequestInfo) (*tls.Certificate, error) { return &callerCert, nil }
	unset := clientcerttest.Unset
	for _, tc := range []struct {
		name         string
		useCert      string // GOOGLE_API_USE_CLIENT_CERTIFICATE
		gcloud       string // CLOUDSDK_CONTEXT_AWARE_USE_CLIENT_CERTIFICATE
		mode         string
		opts         []option.ClientOption
		wantEndpoint string
		wantCert     bool
		wantDER      []byte // the workload certificate when nil
		wantClient   *http.Client
	}{
		{name: "workload config", useCert: unset, gcloud: unset, wantEndpoint: mtls, wantCert: true},
		{name: "gcloud's variable", useCert: unset, gcloud: "true", wantEndpoint: mtls, wantCert: true},
		{name: "never keeps the default endpoint and the certificate", useCert: unset, gcloud: unset, mode: "never", wantEndpoint: plain, wantCert: true},
		{name: "an endpoint from ClientOptions wins", useCert: unset, gcloud: unset, opts: []option.ClientOption{option.WithEndpoint(explicit)}, wantEndpoint: explicit, wantCert: true},
		{name: "a certificate from ClientOptions wins", useCert: unset, gcloud: unset, opts: []option.ClientOption{option.WithClientCertSource(callerSource)}, wantEndpoint: mtls, wantCert: true, wantDER: callerDER},
		{name: "an HTTP client from ClientOptions is left alone", useCert: unset, gcloud: unset, opts: []option.ClientOption{option.WithHTTPClient(own)}, wantEndpoint: plain, wantClient: own},
		// The api library accepts the deprecated option only with a scope,
		// which the toolset adds.
		{name: "an HTTP client next to option.ImpersonateCredentials is left alone", useCert: unset, gcloud: unset, opts: []option.ClientOption{option.ImpersonateCredentials("target@example-project.iam.gserviceaccount.com"), option.WithHTTPClient(own)}, wantEndpoint: plain, wantClient: own},
		// The auth library leaves the certificate off outside googleapis.com.
		{name: "another universe keeps its endpoint", useCert: unset, gcloud: unset, opts: []option.ClientOption{option.WithUniverseDomain("example.com")}, wantEndpoint: "https://bigquery.example.com/bigquery/v2/"},
		{name: "false", useCert: "false", gcloud: unset, wantEndpoint: plain},
		{name: "gcloud's variable false", useCert: unset, gcloud: "false", wantEndpoint: plain},
	} {
		t.Run(tc.name, func(t *testing.T) {
			clientcerttest.Setenv(t, "GOOGLE_API_USE_CLIENT_CERTIFICATE", tc.useCert)
			clientcerttest.Setenv(t, "CLOUDSDK_CONTEXT_AWARE_USE_CLIENT_CERTIFICATE", tc.gcloud)
			t.Setenv("GOOGLE_API_CERTIFICATE_CONFIG", workload)
			t.Setenv("GOOGLE_API_USE_MTLS_ENDPOINT", tc.mode)

			c, err := newClient(t.Context(), Config{ClientOptions: append(tc.opts, option.WithoutAuthentication())})
			if err != nil {
				t.Fatalf("newClient() error = %v", err)
			}
			if c.endpoint != tc.wantEndpoint {
				t.Errorf("endpoint = %q, want %q", c.endpoint, tc.wantEndpoint)
			}
			if tc.wantClient != nil && c.httpClient != tc.wantClient {
				t.Error("the client is not the one from ClientOptions")
			}

			before := len(peers())
			resp, err := c.httpClient.Get(srv.URL)
			if err != nil {
				t.Fatalf("Get() error = %v", err)
			}
			_ = resp.Body.Close()
			got := peers()[before:]
			if !tc.wantCert {
				if len(got) != 0 {
					t.Errorf("server saw %d client certificates, want none", len(got))
				}
				return
			}
			want, whose := der, "the workload certificate"
			if tc.wantDER != nil {
				want, whose = tc.wantDER, "the certificate from ClientOptions"
			}
			if len(got) != 1 {
				t.Fatalf("server saw %d client certificates, want one", len(got))
			}
			if !bytes.Equal(got[0], want) {
				t.Errorf("server saw a certificate other than %s", whose)
			}
		})
	}
}

// TestNewClientImpersonation runs what Config.ClientOptions says about
// impersonation. The proxy refuses every request, so the host a call asks for
// first shows whose token it wanted: the IAM credentials service for the
// impersonated one, BigQuery itself for the base one.
func TestNewClientImpersonation(t *testing.T) {
	if !clientcerttest.Isolate(t) {
		return
	}
	hosts := clientcerttest.RecordingProxy(t)
	workload, _ := clientcerttest.Workload(t)
	const (
		iam    = "iamcredentials.googleapis.com:443"
		target = "target@example-project.iam.gserviceaccount.com"
	)
	base := option.WithTokenSource(oauth2.StaticTokenSource(&oauth2.Token{AccessToken: "base"}))
	source, err := impersonate.CredentialsTokenSource(t.Context(), impersonate.CredentialsConfig{TargetPrincipal: target, Scopes: []string{bigqueryScope}}, base)
	if err != nil {
		t.Fatalf("impersonate.CredentialsTokenSource() error = %v", err)
	}
	deprecated := []option.ClientOption{base, option.ImpersonateCredentials(target)}
	replacement := []option.ClientOption{option.WithTokenSource(source)}

	for _, tc := range []struct {
		name    string
		useCert string // GOOGLE_API_USE_CLIENT_CERTIFICATE
		opts    []option.ClientOption
		want    string
	}{
		{name: "certificate off, option.ImpersonateCredentials", useCert: "false", opts: deprecated, want: iam},
		{name: "certificate on, option.ImpersonateCredentials is not applied", useCert: clientcerttest.Unset, opts: deprecated, want: "bigquery.mtls.googleapis.com:443"},
		{name: "certificate off, a token source from the impersonate package", useCert: "false", opts: replacement, want: iam},
		{name: "certificate on, a token source from the impersonate package", useCert: clientcerttest.Unset, opts: replacement, want: iam},
	} {
		t.Run(tc.name, func(t *testing.T) {
			clientcerttest.Setenv(t, "GOOGLE_API_USE_CLIENT_CERTIFICATE", tc.useCert)
			t.Setenv("GOOGLE_API_CERTIFICATE_CONFIG", workload)

			c, err := newClient(t.Context(), Config{ClientOptions: tc.opts})
			if err != nil {
				t.Fatalf("newClient() error = %v", err)
			}
			before := len(hosts())
			if resp, err := c.httpClient.Get(c.endpoint + "projects/p/datasets"); err == nil {
				_ = resp.Body.Close()
				t.Fatal("Get() through the refusing proxy succeeded")
			}
			if got := hosts()[before:]; len(got) != 1 || got[0] != tc.want {
				t.Errorf("the call asked for %q, want only %q", got, tc.want)
			}
		})
	}
}

// TestNewClientReportsABrokenCertificateConfig runs in a child process so that
// a newClient that went on past the error would find only the test's files.
func TestNewClientReportsABrokenCertificateConfig(t *testing.T) {
	if !clientcerttest.Isolate(t) {
		return
	}
	clientcerttest.Setenv(t, "GOOGLE_API_USE_CLIENT_CERTIFICATE", clientcerttest.Unset)
	t.Setenv("GOOGLE_API_CERTIFICATE_CONFIG", clientcerttest.BrokenWorkload(t))

	if c, err := newClient(t.Context(), Config{ClientOptions: []option.ClientOption{option.WithoutAuthentication()}}); err == nil {
		t.Fatalf("newClient() = %+v, nil error; want an error for a workload section without paths", c)
	}
	// A client the caller built is left alone, so the config is not read.
	if _, err := newClient(t.Context(), Config{ClientOptions: []option.ClientOption{option.WithHTTPClient(&http.Client{})}}); err != nil {
		t.Errorf("newClient() error = %v with an HTTP client from ClientOptions, want nil", err)
	}
}
