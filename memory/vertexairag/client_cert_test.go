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

package vertexairag

import (
	"bytes"
	"crypto/tls"
	"io"
	"net/http"
	"os"
	"slices"
	"strings"
	"sync"
	"testing"

	aiplatform "cloud.google.com/go/aiplatform/apiv1beta1"
	"cloud.google.com/go/aiplatform/apiv1beta1/aiplatformpb"
	"golang.org/x/oauth2"
	"google.golang.org/api/impersonate"
	"google.golang.org/api/option"

	"google.golang.org/adk/v2/internal/clientcert/clientcerttest"
)

// The generated Vertex AI clients build their transport through the Google
// auth library, which otherwise loads whatever client certificate source the
// machine running the tests has.
func TestMain(m *testing.M) {
	clientcerttest.PinOff()
	os.Exit(m.Run())
}

// hostRecorder answers every request with an empty JSON object and records
// the host it was addressed to.
type hostRecorder struct {
	mu    sync.Mutex
	hosts []string
}

func (r *hostRecorder) RoundTrip(req *http.Request) (*http.Response, error) {
	r.mu.Lock()
	r.hosts = append(r.hosts, req.URL.Host)
	r.mu.Unlock()
	return &http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{"Content-Type": []string{"application/json"}},
		Body:       io.NopCloser(strings.NewReader("{}")),
		Request:    req,
	}, nil
}

// callAll makes one request through each of the three clients.
func callAll(t *testing.T, c *vertexRAGClient) (upload, list, retrieve error) {
	t.Helper()
	upload = c.uploadFile(t.Context(), testCorpus, "name", strings.NewReader("content"))
	_, _, list = c.listFiles(t.Context(), &aiplatformpb.ListRagFilesRequest{Parent: testCorpus})
	_, retrieve = c.retrieveContexts(t.Context(), &aiplatformpb.RetrieveContextsRequest{Parent: "projects/test-proj/locations/us-central1"})
	return upload, list, retrieve
}

// TestNewVertexRAGClientFollowsTheCertificateRule builds the three Vertex AI
// clients with a workload client certificate configured. See
// [clientcerttest.Isolate] for why it runs in a child process, and
// TestAPIOptions in internal/clientcert for why no case sets
// GOOGLE_API_USE_CLIENT_CERTIFICATE to true.
func TestNewVertexRAGClientFollowsTheCertificateRule(t *testing.T) {
	if !clientcerttest.Isolate(t) {
		return
	}
	hosts := clientcerttest.RecordingProxy(t)
	workload, der := clientcerttest.Workload(t)
	srv, peers := clientcerttest.Server(t, &fakeVertexServer{t: t})
	clientcerttest.ADC(t)
	const (
		plain = "us-central1-aiplatform.googleapis.com:443"
		mtls  = "us-central1-aiplatform.mtls.googleapis.com:443"
	)

	unset := clientcerttest.Unset
	for _, tc := range []struct {
		name     string
		useCert  string // GOOGLE_API_USE_CLIENT_CERTIFICATE
		gcloud   string // CLOUDSDK_CONTEXT_AWARE_USE_CLIENT_CERTIFICATE
		mode     string
		wantHost string
		// Whether the upload client presents the certificate, and how many
		// of the two generated clients do.
		wantUpload    bool
		wantGenerated int
	}{
		{name: "workload config", useCert: unset, gcloud: unset, wantHost: mtls, wantUpload: true, wantGenerated: 2},
		{name: "gcloud's variable", useCert: unset, gcloud: "true", wantHost: mtls, wantUpload: true, wantGenerated: 2},
		{name: "never keeps the regional endpoint and the certificate", useCert: unset, gcloud: unset, mode: "never", wantHost: plain, wantUpload: true, wantGenerated: 2},
		{name: "false", useCert: "false", gcloud: unset, wantHost: plain},
		{name: "always without a certificate", useCert: "false", gcloud: unset, mode: "always", wantHost: mtls},
		// The rule is off. The generated clients take the auth library's
		// default, which is on here, so they present the certificate to a
		// server that requests one. Their default host, the plain one, does
		// not.
		{name: "gcloud's variable false", useCert: unset, gcloud: "false", wantHost: plain, wantGenerated: 2},
	} {
		t.Run(tc.name, func(t *testing.T) {
			clientcerttest.Setenv(t, "GOOGLE_API_USE_CLIENT_CERTIFICATE", tc.useCert)
			clientcerttest.Setenv(t, "CLOUDSDK_CONTEXT_AWARE_USE_CLIENT_CERTIFICATE", tc.gcloud)
			t.Setenv("GOOGLE_API_CERTIFICATE_CONFIG", workload)
			t.Setenv("GOOGLE_API_USE_MTLS_ENDPOINT", tc.mode)

			// The host each of the three clients picks by default. The proxy
			// refuses the requests, so each call fails.
			c, err := newVertexRAGClient(t.Context(), "us-central1")
			if err != nil {
				t.Fatalf("newVertexRAGClient() error = %v", err)
			}
			before := len(hosts())
			if upload, list, retrieve := callAll(t, c); upload == nil || list == nil || retrieve == nil {
				t.Fatalf("calls through the refusing proxy = %v, %v, %v, want three errors", upload, list, retrieve)
			}
			got := hosts()[before:]
			if len(got) != 3 {
				t.Fatalf("the proxy saw %d requests (%q), want one from each of the three clients", len(got), got)
			}
			for _, h := range got {
				if h != tc.wantHost {
					t.Errorf("request went to %q, want %q", h, tc.wantHost)
				}
			}

			// The certificate each presents to a server that requests one.
			c, err = newVertexRAGClient(t.Context(), "us-central1", option.WithoutAuthentication(), option.WithEndpoint(srv.URL))
			if err != nil {
				t.Fatalf("newVertexRAGClient() error = %v", err)
			}
			seen := len(peers())
			if err := c.uploadFile(t.Context(), testCorpus, "name", strings.NewReader("content")); err != nil {
				t.Fatalf("uploadFile() error = %v", err)
			}
			wantUpload := 0
			if tc.wantUpload {
				wantUpload = 1
			}
			if got := len(peers()) - seen; got != wantUpload {
				t.Errorf("server saw %d client certificates from the upload client, want %d", got, wantUpload)
			}
			if _, _, err := c.listFiles(t.Context(), &aiplatformpb.ListRagFilesRequest{Parent: testCorpus}); err != nil {
				t.Fatalf("listFiles() error = %v", err)
			}
			if _, err := c.retrieveContexts(t.Context(), &aiplatformpb.RetrieveContextsRequest{Parent: "projects/test-proj/locations/us-central1"}); err != nil {
				t.Fatalf("retrieveContexts() error = %v", err)
			}
			certs := peers()[seen:]
			if got := len(certs) - wantUpload; got != tc.wantGenerated {
				t.Errorf("server saw %d client certificates from the generated clients, want %d", got, tc.wantGenerated)
			}
			for _, c := range certs {
				if !bytes.Equal(c, der) {
					t.Error("server saw a certificate other than the configured one")
				}
			}
		})
	}

	// The api library refuses a certificate outside googleapis.com, so the
	// clients must not move to the mTLS host there.
	for _, tc := range []struct {
		name string
		opts []option.ClientOption
	}{
		{name: "another universe from the caller keeps the regional host"},
		// Options the api library takes with no HTTP client next to them,
		// which must not hide the universe.
		{name: "another universe with a quota project", opts: []option.ClientOption{option.WithQuotaProject("p")}},
		{name: "another universe with a request reason", opts: []option.ClientOption{option.WithRequestReason("r")}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			clientcerttest.Setenv(t, "GOOGLE_API_USE_CLIENT_CERTIFICATE", clientcerttest.Unset)
			t.Setenv("GOOGLE_API_CERTIFICATE_CONFIG", workload)
			opts := append([]option.ClientOption{option.WithoutAuthentication(), option.WithUniverseDomain("example.com")}, tc.opts...)
			c, err := newVertexRAGClient(t.Context(), "us-central1", opts...)
			if err != nil {
				t.Fatalf("newVertexRAGClient() error = %v", err)
			}
			before := len(hosts())
			if upload, list, retrieve := callAll(t, c); upload == nil || list == nil || retrieve == nil {
				t.Fatalf("calls through the refusing proxy = %v, %v, %v, want three errors", upload, list, retrieve)
			}
			got := hosts()[before:]
			if len(got) != 3 {
				t.Fatalf("the proxy saw %d requests (%q), want one from each of the three clients", len(got), got)
			}
			for _, h := range got {
				if h != plain {
					t.Errorf("request went to %q, want %q", h, plain)
				}
			}
		})
	}

	t.Run("a certificate from the caller wins", func(t *testing.T) {
		clientcerttest.Setenv(t, "GOOGLE_API_USE_CLIENT_CERTIFICATE", clientcerttest.Unset)
		t.Setenv("GOOGLE_API_CERTIFICATE_CONFIG", workload)
		callerCert, callerDER := clientcerttest.Certificate(t)
		callerSource := func(*tls.CertificateRequestInfo) (*tls.Certificate, error) { return &callerCert, nil }
		c, err := newVertexRAGClient(t.Context(), "us-central1", option.WithoutAuthentication(), option.WithEndpoint(srv.URL), option.WithClientCertSource(callerSource))
		if err != nil {
			t.Fatalf("newVertexRAGClient() error = %v", err)
		}
		before := len(peers())
		if upload, list, retrieve := callAll(t, c); upload != nil || list != nil || retrieve != nil {
			t.Fatalf("calls = %v, %v, %v, want no error", upload, list, retrieve)
		}
		got := peers()[before:]
		if len(got) != 3 {
			t.Fatalf("server saw %d client certificates, want one from each of the three clients", len(got))
		}
		for _, c := range got {
			if !bytes.Equal(c, callerDER) {
				t.Error("server saw a certificate other than the caller's")
			}
		}
	})

	t.Run("an HTTP client from the caller is left alone", func(t *testing.T) {
		clientcerttest.Setenv(t, "GOOGLE_API_USE_CLIENT_CERTIFICATE", clientcerttest.Unset)
		t.Setenv("GOOGLE_API_CERTIFICATE_CONFIG", workload)
		rec := &hostRecorder{}
		c, err := newVertexRAGClient(t.Context(), "us-central1", option.WithHTTPClient(&http.Client{Transport: rec}))
		if err != nil {
			t.Fatalf("newVertexRAGClient() error = %v", err)
		}
		if upload, list, retrieve := callAll(t, c); upload != nil || list != nil || retrieve != nil {
			t.Fatalf("calls = %v, %v, %v, want no error", upload, list, retrieve)
		}
		if len(rec.hosts) != 3 {
			t.Fatalf("recorded %d requests, want one from each of the three clients", len(rec.hosts))
		}
		for _, h := range rec.hosts {
			if want := "us-central1-aiplatform.googleapis.com"; h != want {
				t.Errorf("request went to %q, want %q", h, want)
			}
		}
	})
}

// TestNewVertexRAGClientImpersonation runs what NewService says about
// impersonation. The proxy refuses every request, so the host a call asks for
// shows whose token it wanted: the IAM credentials service for the
// impersonated one, Vertex AI itself for the base one.
func TestNewVertexRAGClientImpersonation(t *testing.T) {
	if !clientcerttest.Isolate(t) {
		return
	}
	hosts := clientcerttest.RecordingProxy(t)
	workload, _ := clientcerttest.Workload(t)
	const (
		iam    = "iamcredentials.googleapis.com:443"
		plain  = "us-central1-aiplatform.googleapis.com:443"
		mtls   = "us-central1-aiplatform.mtls.googleapis.com:443"
		target = "target@example-project.iam.gserviceaccount.com"
	)
	scopes := aiplatform.DefaultAuthScopes()
	base := option.WithTokenSource(oauth2.StaticTokenSource(&oauth2.Token{AccessToken: "base"}))
	source, err := impersonate.CredentialsTokenSource(t.Context(), impersonate.CredentialsConfig{TargetPrincipal: target, Scopes: scopes}, base)
	if err != nil {
		t.Fatalf("impersonate.CredentialsTokenSource() error = %v", err)
	}
	deprecated := []option.ClientOption{base, option.WithScopes(scopes...), option.ImpersonateCredentials(target)}
	replacement := []option.ClientOption{option.WithTokenSource(source)}

	for _, tc := range []struct {
		name    string
		useCert string // GOOGLE_API_USE_CLIENT_CERTIFICATE
		opts    []option.ClientOption
		// The host the upload, list and retrieve clients ask for, in that
		// order.
		want []string
	}{
		// Only the upload client ever applied the deprecated option.
		{name: "certificate off, option.ImpersonateCredentials", useCert: "false", opts: deprecated, want: []string{iam, plain, plain}},
		{name: "certificate on, option.ImpersonateCredentials is not applied", useCert: clientcerttest.Unset, opts: deprecated, want: []string{mtls, mtls, mtls}},
		{name: "certificate off, a token source from the impersonate package", useCert: "false", opts: replacement, want: []string{iam, iam, iam}},
		{name: "certificate on, a token source from the impersonate package", useCert: clientcerttest.Unset, opts: replacement, want: []string{iam, iam, iam}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			clientcerttest.Setenv(t, "GOOGLE_API_USE_CLIENT_CERTIFICATE", tc.useCert)
			t.Setenv("GOOGLE_API_CERTIFICATE_CONFIG", workload)

			c, err := newVertexRAGClient(t.Context(), "us-central1", tc.opts...)
			if err != nil {
				t.Fatalf("newVertexRAGClient() error = %v", err)
			}
			before := len(hosts())
			if upload, list, retrieve := callAll(t, c); upload == nil || list == nil || retrieve == nil {
				t.Fatalf("calls through the refusing proxy = %v, %v, %v, want three errors", upload, list, retrieve)
			}
			if got := hosts()[before:]; !slices.Equal(got, tc.want) {
				t.Errorf("the calls asked for %q, want %q", got, tc.want)
			}
		})
	}
}

// TestNewVertexRAGClientInAnotherUniverse pins that the regional host does not
// follow the universe, as it did not when it was passed as the endpoint. The
// library refuses to force an mTLS endpoint outside googleapis.com.
func TestNewVertexRAGClientInAnotherUniverse(t *testing.T) {
	t.Setenv("GOOGLE_CLOUD_UNIVERSE_DOMAIN", "example.com")
	t.Setenv("GOOGLE_API_USE_MTLS_ENDPOINT", "")
	t.Setenv("GOOGLE_API_USE_MTLS", "")
	rec := &hostRecorder{}
	c, err := newVertexRAGClient(t.Context(), "us-central1", option.WithHTTPClient(&http.Client{Transport: rec}))
	if err != nil {
		t.Fatalf("newVertexRAGClient() error = %v", err)
	}
	if upload, list, retrieve := callAll(t, c); upload != nil || list != nil || retrieve != nil {
		t.Fatalf("calls = %v, %v, %v, want no error", upload, list, retrieve)
	}
	if len(rec.hosts) != 3 {
		t.Fatalf("recorded %d requests, want one from each of the three clients", len(rec.hosts))
	}
	for _, h := range rec.hosts {
		if want := "us-central1-aiplatform.googleapis.com"; h != want {
			t.Errorf("request went to %q, want %q", h, want)
		}
	}

	t.Setenv("GOOGLE_API_USE_MTLS_ENDPOINT", "always")
	if c, err := newVertexRAGClient(t.Context(), "us-central1", option.WithHTTPClient(&http.Client{Transport: rec})); err == nil {
		t.Errorf("newVertexRAGClient() = %+v, nil error with the mTLS endpoint forced in another universe; want an error", c)
	}
}

// TestNewVertexRAGClientGlobalEndpoint covers the one location whose host has
// no region prefix.
func TestNewVertexRAGClientGlobalEndpoint(t *testing.T) {
	t.Setenv("GOOGLE_API_USE_MTLS_ENDPOINT", "always")
	for _, location := range []string{"", "global"} {
		c, err := newVertexRAGClient(t.Context(), location, option.WithoutAuthentication())
		if err != nil {
			t.Fatalf("newVertexRAGClient(%q) error = %v", location, err)
		}
		if want := "https://aiplatform.mtls.googleapis.com"; c.uploadEndpoint != want {
			t.Errorf("newVertexRAGClient(%q) upload endpoint = %q, want %q", location, c.uploadEndpoint, want)
		}
	}
}

// TestNewVertexRAGClientReportsABrokenCertificateConfig runs in a child process
// so that a newVertexRAGClient that went on past the error would find only
// the test's files.
func TestNewVertexRAGClientReportsABrokenCertificateConfig(t *testing.T) {
	if !clientcerttest.Isolate(t) {
		return
	}
	clientcerttest.Setenv(t, "GOOGLE_API_USE_CLIENT_CERTIFICATE", clientcerttest.Unset)
	t.Setenv("GOOGLE_API_CERTIFICATE_CONFIG", clientcerttest.BrokenWorkload(t))

	// The generated clients fail on the same config a step later, so the
	// message tells whose lookup reported it.
	c, err := newVertexRAGClient(t.Context(), "us-central1", option.WithoutAuthentication())
	if err == nil {
		t.Fatalf("newVertexRAGClient() = %+v, nil error; want an error for a workload section without paths", c)
	}
	if !strings.HasPrefix(err.Error(), "clientcert: ") {
		t.Errorf("newVertexRAGClient() error = %q, want the certificate lookup of this package to report it", err)
	}
	// A client the caller built is left alone, so the config is not read.
	if _, err := newVertexRAGClient(t.Context(), "us-central1", option.WithHTTPClient(&http.Client{Transport: &hostRecorder{}})); err != nil {
		t.Errorf("newVertexRAGClient() error = %v with an HTTP client from the caller, want nil", err)
	}
}
