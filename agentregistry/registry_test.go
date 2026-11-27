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

package agentregistry

import (
	"bytes"
	"context"
	"net/http"
	"os"
	"strings"
	"testing"

	"golang.org/x/oauth2/google"
	"google.golang.org/api/option"
	"google.golang.org/api/option/internaloption"
	htransport "google.golang.org/api/transport"

	"google.golang.org/adk/v2/internal/clientcert/clientcerttest"
)

// New builds its default client through the Google auth library, which loads
// a client certificate when the shell running the tests turns one on, and
// that can run a certificate helper installed on the machine.
func TestMain(m *testing.M) {
	clientcerttest.PinOff()
	os.Exit(m.Run())
}

func TestNew_Validation(t *testing.T) {
	tests := []struct {
		name string
		cfg  Config
	}{
		{name: "missing project", cfg: Config{Location: "us-central1", HTTPClient: &http.Client{}}},
		{name: "missing location", cfg: Config{ProjectID: "p", HTTPClient: &http.Client{}}},
		{name: "missing both", cfg: Config{HTTPClient: &http.Client{}}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := New(context.Background(), tc.cfg); err == nil {
				t.Errorf("New(%+v) error = nil, want an error", tc.cfg)
			}
		})
	}
}

func TestNew_WiresRestRequester(t *testing.T) {
	tests := []struct {
		name            string
		mtlsEnv         string
		quotaEnv        string
		wantBaseURL     string
		wantUserProject string
	}{
		{name: "default endpoint", mtlsEnv: "", wantBaseURL: baseURLProd, wantUserProject: "my-project"},
		{name: "never keeps standard endpoint", mtlsEnv: "never", wantBaseURL: baseURLProd, wantUserProject: "my-project"},
		{name: "always targets mTLS endpoint", mtlsEnv: "always", wantBaseURL: baseURLMTLS, wantUserProject: "my-project"},
		{name: "ALWAYS is case-insensitive", mtlsEnv: "ALWAYS", wantBaseURL: baseURLMTLS, wantUserProject: "my-project"},
		// Quota env var is honored even with a custom client (adk-python parity).
		{name: "quota project env honored with custom client", mtlsEnv: "", quotaEnv: "quota-proj", wantBaseURL: baseURLProd, wantUserProject: "quota-proj"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Setenv("GOOGLE_API_USE_MTLS_ENDPOINT", tc.mtlsEnv)
			t.Setenv("GOOGLE_CLOUD_QUOTA_PROJECT", tc.quotaEnv)

			// Custom HTTPClient avoids ADC/network in tests.
			c, err := New(context.Background(), Config{
				ProjectID:  "my-project",
				Location:   "us-central1",
				HTTPClient: &http.Client{},
			})
			if err != nil {
				t.Fatalf("New() error = %v", err)
			}

			rt, ok := c.requester.(*restRequester)
			if !ok {
				t.Fatalf("client requester type = %T, want *restRequester", c.requester)
			}
			if rt.baseURL != tc.wantBaseURL {
				t.Errorf("baseURL = %q, want %q", rt.baseURL, tc.wantBaseURL)
			}
			if want := "projects/my-project/locations/us-central1"; rt.basePath != want {
				t.Errorf("basePath = %q, want %q", rt.basePath, want)
			}
			if rt.userProject != tc.wantUserProject {
				t.Errorf("userProject = %q, want %q", rt.userProject, tc.wantUserProject)
			}
		})
	}
}

// TestNewFollowsTheCertificateRule builds the ADC-backed client with a workload
// client certificate configured. See [clientcerttest.Isolate] for why it runs
// in a child process, and TestAPIOptions in internal/clientcert for why no
// case sets GOOGLE_API_USE_CLIENT_CERTIFICATE to true.
func TestNewFollowsTheCertificateRule(t *testing.T) {
	if !clientcerttest.Isolate(t) {
		return
	}
	workload, der := clientcerttest.Workload(t)
	srv, peers := clientcerttest.Server(t, http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
	clientcerttest.ADC(t)

	unset := clientcerttest.Unset
	for _, tc := range []struct {
		name        string
		useCert     string // GOOGLE_API_USE_CLIENT_CERTIFICATE
		gcloud      string // CLOUDSDK_CONTEXT_AWARE_USE_CLIENT_CERTIFICATE
		mode        string
		wantBaseURL string
		wantCert    bool
	}{
		{name: "workload config", useCert: unset, gcloud: unset, wantBaseURL: "https://agentregistry.mtls.googleapis.com/v1", wantCert: true},
		{name: "gcloud's variable", useCert: unset, gcloud: "true", wantBaseURL: baseURLMTLS, wantCert: true},
		{name: "never keeps the standard endpoint and the certificate", useCert: unset, gcloud: unset, mode: "never", wantBaseURL: baseURLProd, wantCert: true},
		{name: "false", useCert: "false", gcloud: unset, wantBaseURL: baseURLProd},
		{name: "gcloud's variable false", useCert: unset, gcloud: "false", wantBaseURL: baseURLProd},
		{name: "always without a certificate", useCert: "false", gcloud: unset, mode: "always", wantBaseURL: baseURLMTLS},
	} {
		t.Run(tc.name, func(t *testing.T) {
			clientcerttest.Setenv(t, "GOOGLE_API_USE_CLIENT_CERTIFICATE", tc.useCert)
			clientcerttest.Setenv(t, "CLOUDSDK_CONTEXT_AWARE_USE_CLIENT_CERTIFICATE", tc.gcloud)
			t.Setenv("GOOGLE_API_CERTIFICATE_CONFIG", workload)
			t.Setenv("GOOGLE_API_USE_MTLS_ENDPOINT", tc.mode)

			c, err := New(t.Context(), Config{ProjectID: "my-project", Location: "us-central1"})
			if err != nil {
				t.Fatalf("New() error = %v", err)
			}
			if got := c.requester.(*restRequester).baseURL; got != tc.wantBaseURL {
				t.Errorf("baseURL = %q, want %q", got, tc.wantBaseURL)
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
			if len(got) != 1 || !bytes.Equal(got[0], der) {
				t.Errorf("server saw %d client certificates, want exactly the configured one", len(got))
			}
		})
	}
}

// TestNewReportsABrokenCertificateConfig runs in a child process so that a
// New that went on past the error would find only the test's files.
func TestNewReportsABrokenCertificateConfig(t *testing.T) {
	if !clientcerttest.Isolate(t) {
		return
	}
	clientcerttest.ADC(t)
	clientcerttest.Setenv(t, "GOOGLE_API_USE_CLIENT_CERTIFICATE", clientcerttest.Unset)
	t.Setenv("GOOGLE_API_CERTIFICATE_CONFIG", clientcerttest.BrokenWorkload(t))

	if c, err := New(t.Context(), Config{ProjectID: "my-project", Location: "us-central1"}); err == nil {
		t.Fatalf("New() = %+v, nil error; want an error for a workload section without paths", c)
	}
}

// New's ADC path relies on htransport.NewHTTPClient returning the default
// endpoint verbatim, "/v1" included (restRequester joins baseURL+"/"+path).
// That branch needs credentials, so exercise the transport directly.
func TestNewHTTPClientEndpointKeepsVersionPath(t *testing.T) {
	tests := []struct {
		name    string
		mtlsEnv string
		want    string
	}{
		{name: "standard endpoint", mtlsEnv: "", want: baseURLProd},
		{name: "mTLS endpoint", mtlsEnv: "always", want: baseURLMTLS},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Setenv("GOOGLE_API_USE_MTLS_ENDPOINT", tc.mtlsEnv)
			_, endpoint, err := htransport.NewHTTPClient(context.Background(),
				option.WithoutAuthentication(),
				internaloption.WithDefaultEndpoint(baseURLProd),
				internaloption.WithDefaultMTLSEndpoint(baseURLMTLS),
			)
			if err != nil {
				t.Fatalf("NewHTTPClient() error = %v", err)
			}
			if endpoint != tc.want {
				t.Errorf("endpoint = %q, want %q", endpoint, tc.want)
			}
			if !strings.HasSuffix(endpoint, "/v1") {
				t.Errorf("endpoint = %q, want %q suffix (version path must not be stripped)", endpoint, "/v1")
			}
		})
	}
}

func TestCustomClientBaseURL(t *testing.T) {
	tests := []struct {
		env  string
		want string
	}{
		{env: "", want: baseURLProd},
		{env: "never", want: baseURLProd},
		{env: "auto", want: baseURLProd},
		{env: "always", want: baseURLMTLS},
		{env: "ALWAYS", want: baseURLMTLS},
	}
	for _, tc := range tests {
		t.Run(tc.env, func(t *testing.T) {
			t.Setenv("GOOGLE_API_USE_MTLS_ENDPOINT", tc.env)
			if got := customClientBaseURL(); got != tc.want {
				t.Errorf("customClientBaseURL() with %q = %q, want %q", tc.env, got, tc.want)
			}
		})
	}
}

func TestQuotaProjectID(t *testing.T) {
	t.Run("env var takes precedence", func(t *testing.T) {
		t.Setenv("GOOGLE_CLOUD_QUOTA_PROJECT", "env-project")
		creds := &google.Credentials{JSON: []byte(`{"quota_project_id":"json-project"}`)}
		if got := quotaProjectID(creds); got != "env-project" {
			t.Errorf("quotaProjectID() = %q, want env-project", got)
		}
	})

	t.Run("from credentials JSON", func(t *testing.T) {
		t.Setenv("GOOGLE_CLOUD_QUOTA_PROJECT", "")
		creds := &google.Credentials{JSON: []byte(`{"quota_project_id":"json-project"}`)}
		if got := quotaProjectID(creds); got != "json-project" {
			t.Errorf("quotaProjectID() = %q, want json-project", got)
		}
	})

	t.Run("JSON without quota project", func(t *testing.T) {
		t.Setenv("GOOGLE_CLOUD_QUOTA_PROJECT", "")
		creds := &google.Credentials{JSON: []byte(`{"type":"service_account"}`)}
		if got := quotaProjectID(creds); got != "" {
			t.Errorf("quotaProjectID() = %q, want empty", got)
		}
	})

	t.Run("nil credentials", func(t *testing.T) {
		t.Setenv("GOOGLE_CLOUD_QUOTA_PROJECT", "")
		if got := quotaProjectID(nil); got != "" {
			t.Errorf("quotaProjectID(nil) = %q, want empty", got)
		}
	})
}

func TestListValues(t *testing.T) {
	got := listValues(WithFilter("type=A2A"), WithPageSize(25), WithPageToken("tok"))
	if got.Get("filter") != "type=A2A" {
		t.Errorf("filter = %q, want type=A2A", got.Get("filter"))
	}
	if got.Get("pageSize") != "25" {
		t.Errorf("pageSize = %q, want 25", got.Get("pageSize"))
	}
	if got.Get("pageToken") != "tok" {
		t.Errorf("pageToken = %q, want tok", got.Get("pageToken"))
	}

	// Zero/empty options should not set parameters.
	empty := listValues(WithFilter(""), WithPageSize(0), WithPageToken(""))
	if len(empty) != 0 {
		t.Errorf("listValues with empty options = %v, want no parameters", empty)
	}
}
