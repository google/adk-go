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

package gemini

import (
	"context"
	"crypto/rand"
	"crypto/rsa"
	"crypto/tls"
	"crypto/x509"
	"encoding/base64"
	"encoding/json"
	"encoding/pem"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"cloud.google.com/go/auth"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/internal/clientcert/clientcerttest"
	"google.golang.org/adk/v2/internal/googlemtls"
	"google.golang.org/adk/v2/internal/googlemtls/googlemtlstest"
	"google.golang.org/adk/v2/model"
)

// genaiEnv lists the variables the genai SDK resolves a client from.
var genaiEnv = []string{
	"GOOGLE_GENAI_USE_VERTEXAI", "GOOGLE_GENAI_USE_ENTERPRISE", "GOOGLE_API_KEY", "GEMINI_API_KEY",
	"GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_LOCATION", "GOOGLE_CLOUD_REGION",
	"GOOGLE_GEMINI_BASE_URL", "GOOGLE_VERTEX_BASE_URL", "GOOGLE_APPLICATION_CREDENTIALS",
	"GOOGLE_API_USE_MTLS_ENDPOINT",
}

func clearGenaiEnv(t *testing.T) {
	t.Helper()
	for _, name := range genaiEnv {
		clientcerttest.Setenv(t, name, clientcerttest.Unset)
	}
}

type staticToken string

func (s staticToken) Token(context.Context) (*auth.Token, error) {
	return &auth.Token{Value: string(s), Type: "Bearer"}, nil
}

func testCredentials() *auth.Credentials {
	return auth.NewCredentials(&auth.CredentialsOptions{
		TokenProvider: staticToken("tok"),
		QuotaProjectIDProvider: auth.CredentialsPropertyFunc(func(context.Context) (string, error) {
			return "quota-project", nil
		}),
	})
}

// writeServiceAccount points Application Default Credentials at a service
// account key whose token endpoint does not exist. Building a model fetches
// no token.
func writeServiceAccount(t *testing.T) {
	t.Helper()
	writeServiceAccountFor(t, "https://oauth2.invalid/token", "")
}

// writeServiceAccountFor points Application Default Credentials at a service
// account key whose token requests go to tokenURI.
func writeServiceAccountFor(t *testing.T, tokenURI, quotaProject string) {
	t.Helper()
	key, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		t.Fatal(err)
	}
	der, err := x509.MarshalPKCS8PrivateKey(key)
	if err != nil {
		t.Fatal(err)
	}
	file := map[string]string{
		"type":         "service_account",
		"project_id":   "p",
		"private_key":  string(pem.EncodeToMemory(&pem.Block{Type: "PRIVATE KEY", Bytes: der})),
		"client_email": "sa@p.iam.gserviceaccount.com",
		"token_uri":    tokenURI,
	}
	if quotaProject != "" {
		file["quota_project_id"] = quotaProject
	}
	sa, err := json.Marshal(file)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "sa.json")
	clientcerttest.WriteFile(t, path, sa)
	clientcerttest.Setenv(t, "GOOGLE_APPLICATION_CREDENTIALS", path)
}

func newTestModel(t *testing.T, cfg *genai.ClientConfig) *geminiModel {
	t.Helper()
	llm, err := NewModel(t.Context(), "gemini-2.5-flash", cfg)
	if err != nil {
		t.Fatalf("NewModel() error = %v", err)
	}
	return llm.(*geminiModel)
}

// NewModel builds an authenticated transport where the genai SDK would, and
// leaves every other client as the SDK builds it.
func TestNewModelAuthenticatedPath(t *testing.T) {
	callerClient := &http.Client{}
	creds := testCredentials()
	for _, tc := range []struct {
		name      string
		env       map[string]string
		cfg       genai.ClientConfig
		adc       bool
		wantAuthn bool
	}{
		{name: "caller HTTPClient", cfg: genai.ClientConfig{Backend: genai.BackendVertexAI, Project: "p", Location: "us-central1", HTTPClient: callerClient}},
		{name: "Gemini API key", cfg: genai.ClientConfig{Backend: genai.BackendGeminiAPI, APIKey: "k"}},
		{name: "Vertex express mode", cfg: genai.ClientConfig{Backend: genai.BackendVertexAI, APIKey: "k"}},
		{name: "Vertex API key with project", cfg: genai.ClientConfig{Backend: genai.BackendVertexAI, APIKey: "k", Project: "p", Location: "us-central1"}},
		{name: "Vertex project without location, base URL", cfg: genai.ClientConfig{Backend: genai.BackendVertexAI, Project: "p", HTTPOptions: genai.HTTPOptions{BaseURL: "https://proxy.example.test/"}}},
		{name: "Vertex base URL without project or credentials", cfg: genai.ClientConfig{Backend: genai.BackendVertexAI, HTTPOptions: genai.HTTPOptions{BaseURL: "https://proxy.example.test/"}}},
		{name: "Vertex base URL with credentials", cfg: genai.ClientConfig{Backend: genai.BackendVertexAI, Credentials: creds, HTTPOptions: genai.HTTPOptions{BaseURL: "https://proxy.example.test/"}}, wantAuthn: true},
		{name: "Vertex project with credentials", cfg: genai.ClientConfig{Backend: genai.BackendVertexAI, Project: "p", Location: "us-central1", Credentials: creds}, wantAuthn: true},
		{name: "Vertex project with ADC", cfg: genai.ClientConfig{Backend: genai.BackendVertexAI, Project: "p", Location: "us-central1"}, adc: true, wantAuthn: true},
		{name: "Vertex from the environment", env: map[string]string{"GOOGLE_GENAI_USE_VERTEXAI": "true", "GOOGLE_CLOUD_PROJECT": "p", "GOOGLE_CLOUD_LOCATION": "europe-west4"}, adc: true, wantAuthn: true},
		{name: "Vertex global by default", cfg: genai.ClientConfig{Backend: genai.BackendVertexAI, Project: "p"}, adc: true, wantAuthn: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			clearGenaiEnv(t)
			for k, v := range tc.env {
				t.Setenv(k, v)
			}
			if tc.adc {
				writeServiceAccount(t)
			}
			cfg := tc.cfg
			m := newTestModel(t, &cfg)
			cc := m.client.ClientConfig()
			if callerClient.Transport != nil {
				t.Fatalf("NewModel changed the caller's HTTPClient: Transport = %v", callerClient.Transport)
			}
			interceptor, ok := cc.HTTPClient.Transport.(*mergeHeadersInterceptor)
			if !ok {
				t.Fatalf("transport = %T, want *mergeHeadersInterceptor", cc.HTTPClient.Transport)
			}
			if got := interceptor.base != nil; got != tc.wantAuthn {
				t.Errorf("authenticated transport = %v (%T), want %v", got, interceptor.base, tc.wantAuthn)
			}
			// The Live API takes its bearer token from Credentials.
			if got := cc.Credentials != nil; got != (tc.wantAuthn || tc.cfg.Credentials != nil) {
				t.Errorf("ClientConfig().Credentials set = %v, want %v", got, tc.wantAuthn)
			}
			if tc.cfg.Credentials != nil && cc.Credentials != tc.cfg.Credentials {
				t.Error("ClientConfig().Credentials is not the caller's")
			}
		})
	}
}

// offline makes http.DefaultTransport fail every request for the rest of the
// test, so a request that misses the mTLS path fails instead of going to the
// network.
func offline(t *testing.T) {
	orig := http.DefaultTransport
	http.DefaultTransport = offlineTransport{}
	t.Cleanup(func() { http.DefaultTransport = orig })
}

type offlineTransport struct{}

func (offlineTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	return nil, fmt.Errorf("offline: request to %s reached http.DefaultTransport", req.URL.Host)
}

func callModelOnce(t *testing.T, m model.LLM) error {
	t.Helper()
	req := &model.LLMRequest{Contents: genai.Text("hi")}
	for _, err := range m.GenerateContent(t.Context(), req, false) {
		return err
	}
	return nil
}

// receive returns the request the server saw. The stand-in server answers 204,
// which the SDK cannot parse, so the call itself fails.
func receive(t *testing.T, srv *googlemtlstest.Server, callErr error) (googlemtlstest.Request, http.Header) {
	t.Helper()
	select {
	case got := <-srv.Requests:
		h := srv.Headers()
		return got, h[len(h)-1]
	default:
		t.Fatalf("no request reached the server; call error = %v", callErr)
		return googlemtlstest.Request{}, nil
	}
}

func checkAuthHeaders(t *testing.T, h http.Header) {
	t.Helper()
	if got, want := h.Get("Authorization"), "Bearer tok"; got != want {
		t.Errorf("Authorization = %q, want %q", got, want)
	}
	if got, want := h.Get("X-Goog-User-Project"), "quota-project"; got != want {
		t.Errorf("X-Goog-User-Project = %q, want %q", got, want)
	}
}

// With a certificate, a Vertex AI request goes to the regional mTLS host,
// presents it, and keeps its credentials.
func TestNewModelPresentsCertificateToVertexAI(t *testing.T) {
	clearGenaiEnv(t)
	srv := googlemtlstest.NewServer(t)
	srv.Install(t)
	offline(t)
	m := newTestModel(t, &genai.ClientConfig{Backend: genai.BackendVertexAI, Project: "p", Location: "us-central1", Credentials: testCredentials()})
	got, h := receive(t, srv, callModelOnce(t, m))
	want := googlemtlstest.Request{Host: "us-central1-aiplatform.mtls.googleapis.com", ServerName: "us-central1-aiplatform.mtls.googleapis.com", PeerCerts: 1, Leaf: srv.Leaf()}
	if got != want {
		t.Errorf("server saw %+v, want %+v", got, want)
	}
	checkAuthHeaders(t, h)
}

// Like the google-genai Python SDK, a host other than a Google API host gets
// the certificate when it asks, under its own name.
func TestNewModelPresentsCertificateToOtherHost(t *testing.T) {
	clearGenaiEnv(t)
	srv := googlemtlstest.NewServer(t)
	t.Cleanup(googlemtls.SetTestHooks(func() (func(*tls.CertificateRequestInfo) (*tls.Certificate, error), error) {
		return srv.Provider, nil
	}, func(tr *http.Transport) {
		srv.Configure(tr)
		// The server's certificate names only Google API hosts.
		tr.TLSClientConfig.ServerName = "proxy.googleapis.com"
	}))
	offline(t)
	m := newTestModel(t, &genai.ClientConfig{
		Backend: genai.BackendVertexAI, Project: "p", Location: "us-central1", Credentials: testCredentials(),
		HTTPOptions: genai.HTTPOptions{BaseURL: "https://proxy.example.test/"},
	})
	got, h := receive(t, srv, callModelOnce(t, m))
	if got.Host != "proxy.example.test" || got.PeerCerts != 1 || got.Leaf != srv.Leaf() {
		t.Errorf("server saw %+v, want Host proxy.example.test and the client certificate", got)
	}
	checkAuthHeaders(t, h)
}

// The Gemini API never gets the certificate, as in the google-genai Python SDK.
func TestNewModelGeminiAPIGetsNoCertificate(t *testing.T) {
	clearGenaiEnv(t)
	srv := googlemtlstest.NewServer(t)
	srv.Install(t)
	offline(t)
	m := newTestModel(t, &genai.ClientConfig{Backend: genai.BackendGeminiAPI, APIKey: "k"})
	err := callModelOnce(t, m)
	if err == nil || !strings.Contains(err.Error(), "offline: request to generativelanguage.googleapis.com") {
		t.Errorf("GenerateContent error = %v, want the request on http.DefaultTransport", err)
	}
	if n := len(srv.Requests); n != 0 {
		t.Errorf("%d requests reached the mTLS server, want 0", n)
	}
}

// With the rule off, the request goes to the plain host with no certificate
// and with its credentials, through the transport the auth library would
// build from http.DefaultTransport: one that speaks HTTP/2, which a transport
// built by hand with TLS settings of its own does not.
func TestNewModelRuleOff(t *testing.T) {
	clearGenaiEnv(t)
	type seen struct {
		proto, host string
		peerCerts   int
		header      http.Header
	}
	got := make(chan seen, 1)
	srv := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		got <- seen{proto: r.Proto, host: r.Host, peerCerts: len(r.TLS.PeerCertificates), header: r.Header.Clone()}
		w.WriteHeader(http.StatusNoContent)
	}))
	srv.EnableHTTP2 = true
	srv.TLS = &tls.Config{ClientAuth: tls.RequestClientCert}
	srv.StartTLS()
	t.Cleanup(srv.Close)
	roots := x509.NewCertPool()
	roots.AddCert(srv.Certificate())
	addr := srv.Listener.Addr().String()
	def := &http.Transport{
		DialContext: func(ctx context.Context, network, _ string) (net.Conn, error) {
			var d net.Dialer
			return d.DialContext(ctx, network, addr)
		},
		// httptest's certificate names example.com.
		TLSClientConfig: &tls.Config{RootCAs: roots, ServerName: "example.com"},
	}
	t.Cleanup(def.CloseIdleConnections)
	orig := http.DefaultTransport
	http.DefaultTransport = def
	t.Cleanup(func() { http.DefaultTransport = orig })

	m := newTestModel(t, &genai.ClientConfig{Backend: genai.BackendVertexAI, Project: "p", Location: "us-central1", Credentials: testCredentials()})
	callErr := callModelOnce(t, m)
	var r seen
	select {
	case r = <-got:
	default:
		t.Fatalf("no request reached the server; call error = %v", callErr)
	}
	if r.host != "us-central1-aiplatform.googleapis.com" || r.peerCerts != 0 || r.proto != "HTTP/2.0" {
		t.Errorf("server saw host %q, %d client certificates, %s; want the plain host, none, HTTP/2.0", r.host, r.peerCerts, r.proto)
	}
	checkAuthHeaders(t, r.header)
}

// With Application Default Credentials, the token is requested for the
// cloud-platform scope, as the genai SDK requests it, and the request carries
// it and the key's quota project.
func TestNewModelADCToken(t *testing.T) {
	clearGenaiEnv(t)
	t.Setenv("GOOGLE_CLOUD_QUOTA_PROJECT", "")
	t.Setenv("GOOGLE_CLOUD_UNIVERSE_DOMAIN", "")
	t.Setenv("GOOGLE_AUTH_TRUST_BOUNDARY_ENABLED", "")
	var scope string
	var model http.Header
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/token" {
			var err error
			if scope, err = assertionScope(r); err != nil {
				t.Errorf("token request: %v", err)
			}
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, `{"access_token":"adc-token","token_type":"Bearer","expires_in":3600}`)
			return
		}
		model = r.Header.Clone()
		w.WriteHeader(http.StatusNoContent)
	}))
	t.Cleanup(srv.Close)
	writeServiceAccountFor(t, srv.URL+"/token", "adc-quota")

	m := newTestModel(t, &genai.ClientConfig{
		Backend: genai.BackendVertexAI, Project: "p", Location: "us-central1",
		HTTPOptions: genai.HTTPOptions{BaseURL: srv.URL + "/"},
	})
	callErr := callModelOnce(t, m)
	if model == nil {
		t.Fatalf("no model request reached the server; call error = %v", callErr)
	}
	if scope != cloudPlatformScope {
		t.Errorf("token requested for scope %q, want %q", scope, cloudPlatformScope)
	}
	if got := model.Get("Authorization"); got != "Bearer adc-token" {
		t.Errorf("Authorization = %q, want the ADC token", got)
	}
	if got := model.Get("X-Goog-User-Project"); got != "adc-quota" {
		t.Errorf("X-Goog-User-Project = %q, want adc-quota", got)
	}
}

// assertionScope reads the scope claim out of the signed JWT a service-account
// token request carries as its assertion form field.
func assertionScope(r *http.Request) (string, error) {
	if err := r.ParseForm(); err != nil {
		return "", err
	}
	parts := strings.Split(r.PostForm.Get("assertion"), ".")
	if len(parts) != 3 {
		return "", fmt.Errorf("assertion is not a JWT: %d segments", len(parts))
	}
	payload, err := base64.RawURLEncoding.DecodeString(parts[1])
	if err != nil {
		return "", err
	}
	var claims struct {
		Scope string `json:"scope"`
	}
	if err := json.Unmarshal(payload, &claims); err != nil {
		return "", err
	}
	return claims.Scope, nil
}

// newClient takes the authenticated path exactly where the genai SDK builds
// its authenticated transport, and resolves the rest as the SDK does, over
// every combination of the inputs the SDK resolves a client from. A genai
// upgrade that moves the SDK's test turns this red.
func TestNewClientAgreesWithSDK(t *testing.T) {
	clearGenaiEnv(t)
	writeServiceAccount(t)
	// The SDK logs its precedence warnings with the standard logger.
	log.SetOutput(io.Discard)
	t.Cleanup(func() { log.SetOutput(os.Stderr) })
	creds := testCredentials()
	type shape struct {
		err                                      bool
		backend                                  genai.Backend
		key, project, location, baseURL, version string
		creds, authn                             bool
	}
	describe := func(c *genai.Client, err error) shape {
		if err != nil {
			return shape{err: true}
		}
		cc := c.ClientConfig()
		return shape{
			backend: cc.Backend, key: cc.APIKey, project: cc.Project, location: cc.Location,
			baseURL: cc.HTTPOptions.BaseURL, version: cc.HTTPOptions.APIVersion,
			creds: cc.Credentials != nil, authn: cc.HTTPClient != nil && cc.HTTPClient.Transport != nil,
		}
	}
	var total, authn, errs int
	for _, backend := range []genai.Backend{genai.BackendUnspecified, genai.BackendGeminiAPI, genai.BackendVertexAI} {
		for mask := range 1 << 9 {
			pick := func(i int, v string) string {
				if mask&(1<<i) != 0 {
					return v
				}
				return clientcerttest.Unset
			}
			cfg := genai.ClientConfig{Backend: backend}
			for i, f := range []*string{&cfg.APIKey, &cfg.Project, &cfg.Location, &cfg.HTTPOptions.BaseURL} {
				if v := pick([]int{0, 1, 2, 4}[i], []string{"k", "p", "us-central1", "https://proxy.example.test/"}[i]); v != clientcerttest.Unset {
					*f = v
				}
			}
			if mask&(1<<3) != 0 {
				cfg.Credentials = creds
			}
			clientcerttest.Setenv(t, "GOOGLE_GENAI_USE_VERTEXAI", pick(5, "true"))
			clientcerttest.Setenv(t, "GOOGLE_API_KEY", pick(6, "ek"))
			clientcerttest.Setenv(t, "GOOGLE_CLOUD_PROJECT", pick(7, "ep"))
			clientcerttest.Setenv(t, "GOOGLE_CLOUD_LOCATION", pick(8, "europe-west4"))

			sdkCfg, ourCfg := cfg, cfg
			want := describe(genai.NewClient(t.Context(), &sdkCfg))
			got := describe(newClient(t.Context(), &ourCfg))
			total++
			if want.err {
				errs++
			}
			if want.authn {
				authn++
			}
			if got != want {
				t.Errorf("backend %v, inputs %09b: newClient %+v, SDK %+v", backend, mask, got, want)
			}
		}
	}
	// Guards against a table that stopped reaching both paths.
	if authn == 0 || errs == total {
		t.Errorf("%d combinations, %d authenticated, %d errors: want some of each kind", total, authn, errs)
	}
}

// A certificate config the rule cannot use fails NewModel where the client is
// authenticated, as the auth library failed it before, and nowhere else.
func TestNewModelBrokenCertificateConfig(t *testing.T) {
	if !clientcerttest.Isolate(t) {
		return
	}
	clientcerttest.Setenv(t, "GOOGLE_API_CERTIFICATE_CONFIG", clientcerttest.BrokenWorkload(t))
	_, err := NewModel(t.Context(), "gemini-2.5-flash", &genai.ClientConfig{Backend: genai.BackendVertexAI, Project: "p", Location: "us-central1", Credentials: testCredentials()})
	if err == nil || !strings.Contains(err.Error(), "GOOGLE_API_USE_CLIENT_CERTIFICATE=false") {
		t.Errorf("NewModel() on Vertex AI error = %v, want the config error with the opt-out", err)
	}
	for _, cfg := range []*genai.ClientConfig{
		{Backend: genai.BackendGeminiAPI, APIKey: "k"},
		{Backend: genai.BackendVertexAI, APIKey: "k"},
	} {
		if _, err := NewModel(t.Context(), "gemini-2.5-flash", cfg); err != nil {
			t.Errorf("NewModel(%v) error = %v, want nil", cfg.Backend, err)
		}
	}
}

// With the rule off, building a Vertex AI model does not start the helper of
// an enterprise certificate proxy, which the auth library on its own starts
// whenever a config names one. The library keeps its search's outcome for the
// life of the process, so this runs in a child process.
func TestNewModelRuleOffStartsNoCertificateHelper(t *testing.T) {
	if !clientcerttest.Isolate(t) {
		return
	}
	dir := t.TempDir()
	marker, helper := filepath.Join(dir, "ran"), filepath.Join(dir, "ecp.sh")
	// The child has no PATH, so the helper uses a shell builtin only.
	clientcerttest.WriteFile(t, helper, fmt.Appendf(nil, "#!/bin/sh\necho ran >> %q\nexit 1\n", marker))
	if err := os.Chmod(helper, 0o700); err != nil {
		t.Fatal(err)
	}
	config := filepath.Join(dir, "certificate_config.json")
	clientcerttest.WriteFile(t, config, fmt.Appendf(nil, `{"cert_configs":{"pkcs11":{"label":"x","module":"x","slot":"0"}},"libs":{"ecp":%q,"ecp_client":%q,"tls_offload":%q}}`,
		helper, filepath.Join(dir, "none.so"), filepath.Join(dir, "none.so")))
	// A config with no workload section leaves the rule off.
	clientcerttest.Setenv(t, "GOOGLE_API_CERTIFICATE_CONFIG", config)

	newTestModel(t, &genai.ClientConfig{Backend: genai.BackendVertexAI, Project: "p", Location: "us-central1", Credentials: testCredentials()})
	if _, err := os.Stat(marker); err == nil {
		t.Error("building the model started the certificate helper")
	}
}
