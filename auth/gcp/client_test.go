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

package gcp

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"google.golang.org/adk/v2/auth"
)

const (
	authProviderResource = "projects/p/locations/l/authProviders/ap"
	connectorResource    = "projects/p/locations/l/connectors/co"
)

// TestRetrieveCredential drives RetrieveCredential end to end for both services
// via a fake server that replays response bodies in order; each case asserts one
// expected outcome.
func TestRetrieveCredential(t *testing.T) {
	tests := []struct {
		name        string
		resource    string
		bodies      []string
		wantCalls   int       // >0 => assert the number of service calls
		wantBearer  string    // expect a bearer credential carrying this token
		wantExpiry  string    // non-empty => expect this timestamp as ExpiresAt
		wantAPIKey  [2]string // expect an API-key credential {name, value}
		wantConsent [2]string // expect *auth.ConsentRequiredError {authURI, nonce}
		wantErrIs   error     // expect errors.Is(err, target)
		wantErrText string    // expect err to contain this substring
		pollTimeout time.Duration
	}{
		// Agent Identity: synchronous "result" oneof.
		{
			name:       "agent identity bearer",
			resource:   authProviderResource,
			bodies:     []string{`{"success":{"token":"tok","header":"Authorization: Bearer"}}`},
			wantBearer: "tok",
		},
		{
			name:       "agent identity bearer with expiry",
			resource:   authProviderResource,
			bodies:     []string{`{"success":{"token":"tok","header":"Authorization: Bearer","expireTime":"2999-01-01T00:00:00Z"}}`},
			wantBearer: "tok",
			wantExpiry: "2999-01-01T00:00:00Z",
		},
		{
			// An unparseable expiry leaves ExpiresAt zero, which the provider
			// reads as "don't cache"; the credential itself is still usable.
			name:       "agent identity bearer with unparseable expiry",
			resource:   authProviderResource,
			bodies:     []string{`{"success":{"token":"tok","header":"Authorization: Bearer","expireTime":"not-a-time"}}`},
			wantBearer: "tok",
		},
		{
			name:       "agent identity custom header",
			resource:   authProviderResource,
			bodies:     []string{`{"success":{"token":"KEY","header":"X-Goog-Api-Key"}}`},
			wantAPIKey: [2]string{"X-Goog-Api-Key", "KEY"},
		},
		{
			name:        "agent identity consent required",
			resource:    authProviderResource,
			bodies:      []string{`{"uriConsentRequired":{"authorizationUri":"https://consent","consentNonce":"n"}}`},
			wantConsent: [2]string{"https://consent", "n"},
		},
		{
			name:      "agent identity consent rejected",
			resource:  authProviderResource,
			bodies:    []string{`{"consentRejected":{}}`},
			wantErrIs: ErrConsentRejected,
		},
		{
			name:       "agent identity polls pending then succeeds",
			resource:   authProviderResource,
			bodies:     []string{`{"pending":{}}`, `{"success":{"token":"tok","header":"Authorization: Bearer"}}`},
			wantBearer: "tok",
			wantCalls:  2,
		},
		// IAM Connector: google.longrunning.Operation wrapper.
		{
			name:       "connector bearer",
			resource:   connectorResource,
			bodies:     []string{`{"done":true,"response":{"@type":"x","token":"tok","header":"Authorization: Bearer"}}`},
			wantBearer: "tok",
		},
		{
			name:       "connector polls consent pending then succeeds",
			resource:   connectorResource,
			bodies:     []string{`{"metadata":{"@type":"x","consentPending":{}}}`, `{"done":true,"response":{"token":"tok","header":"Authorization: Bearer"}}`},
			wantBearer: "tok",
			wantCalls:  2,
		},
		{
			name:        "connector consent required",
			resource:    connectorResource,
			bodies:      []string{`{"metadata":{"uriConsentRequired":{"authorizationUri":"https://c","consentNonce":"n"}}}`},
			wantConsent: [2]string{"https://c", "n"},
		},
		{
			name:      "connector consent rejected",
			resource:  connectorResource,
			bodies:    []string{`{"metadata":{"consentRejected":{}}}`},
			wantErrIs: ErrConsentRejected,
		},
		{
			name:        "connector operation error",
			resource:    connectorResource,
			bodies:      []string{`{"error":{"message":"boom"}}`},
			wantErrText: "boom",
		},
		{
			// A terminal (done) operation carrying no credential must fail fast,
			// not be treated as pending and polled to the timeout.
			name:        "connector done without credential",
			resource:    connectorResource,
			bodies:      []string{`{"done":true}`},
			wantErrText: "no credential",
		},
		// The two services deliberately disagree on an unrecognised 200: Agent
		// Identity's result is a closed oneof, so an unknown arm can only be a
		// mismatch worth failing on...
		{
			name:        "agent identity unrecognized result fails fast",
			resource:    authProviderResource,
			bodies:      []string{`{}`},
			wantErrText: "empty result",
			wantCalls:   1,
		},
		// ...whereas a connector operation that is merely not done yet is normal,
		// so an unrecognised one keeps being polled until the timeout.
		{
			name:        "connector unrecognized operation polls to timeout",
			resource:    connectorResource,
			bodies:      []string{`{}`},
			wantErrIs:   ErrPollTimeout,
			pollTimeout: 30 * time.Millisecond,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			srv, calls := sequenceServer(tc.bodies...)
			defer srv.Close()

			c := newTestClient(t, srv)
			if tc.pollTimeout > 0 {
				c.pollTimeout = tc.pollTimeout
			}
			got, err := c.RetrieveCredential(t.Context(),
				Request{Resource: tc.resource, UserID: "u"})

			switch {
			case tc.wantBearer != "":
				if err != nil {
					t.Fatalf("RetrieveCredential() error = %v", err)
				}
				wantBearer(t, got.Credential, tc.wantBearer)
				if tc.wantExpiry != "" {
					want, _ := time.Parse(time.RFC3339, tc.wantExpiry)
					if !got.ExpiresAt.Equal(want) {
						t.Errorf("ExpiresAt = %v, want %v", got.ExpiresAt, want)
					}
				} else if !got.ExpiresAt.IsZero() {
					t.Errorf("ExpiresAt = %v, want zero", got.ExpiresAt)
				}
			case tc.wantAPIKey[0] != "":
				if err != nil {
					t.Fatalf("RetrieveCredential() error = %v", err)
				}
				wantAPIKey(t, got.Credential, tc.wantAPIKey[0], tc.wantAPIKey[1])
			case tc.wantConsent[0] != "":
				var consent *auth.ConsentRequiredError
				if !errors.As(err, &consent) {
					t.Fatalf("error = %v, want *auth.ConsentRequiredError", err)
				}
				if consent.AuthURI != tc.wantConsent[0] || consent.Nonce != tc.wantConsent[1] {
					t.Errorf("consent = %+v, want {authURI:%q nonce:%q}", consent, tc.wantConsent[0], tc.wantConsent[1])
				}
			case tc.wantErrIs != nil:
				if !errors.Is(err, tc.wantErrIs) {
					t.Fatalf("error = %v, want errors.Is %v", err, tc.wantErrIs)
				}
			case tc.wantErrText != "":
				if err == nil || !strings.Contains(err.Error(), tc.wantErrText) {
					t.Fatalf("error = %v, want it to contain %q", err, tc.wantErrText)
				}
			default:
				t.Fatalf("test case %q sets no expectation", tc.name)
			}

			if tc.wantCalls != 0 {
				if got := int(atomic.LoadInt32(calls)); got != tc.wantCalls {
					t.Errorf("service calls = %d, want %d", got, tc.wantCalls)
				}
			}
		})
	}
}

func TestRetrieveRoutesByResource(t *testing.T) {
	tests := []struct {
		name       string
		resource   string
		wantPrefix string
	}{
		{"connector", connectorResource, "/v1alpha/"},
		{"auth provider", authProviderResource, "/v1/"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var gotPath, gotMethod, gotUserID, gotContinueURI string
			var gotScopes []string
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				gotPath, gotMethod = r.URL.Path, r.Method
				var body struct {
					UserID      string   `json:"userId"`
					Scopes      []string `json:"scopes"`
					ContinueURI string   `json:"continueUri"`
				}
				_ = json.NewDecoder(r.Body).Decode(&body)
				gotUserID, gotScopes, gotContinueURI = body.UserID, body.Scopes, body.ContinueURI
				_, _ = io.WriteString(w, `{"done":true,"response":{"token":"t","header":"Authorization: Bearer"},"success":{"token":"t","header":"Authorization: Bearer"}}`)
			}))
			defer srv.Close()

			if _, err := newTestClient(t, srv).RetrieveCredential(t.Context(), Request{
				Resource:    tc.resource,
				UserID:      "user-1",
				Scopes:      []string{"scope-a", "scope-b"},
				ContinueURI: "https://example.test/continue",
			}); err != nil {
				t.Fatalf("RetrieveCredential() error = %v", err)
			}
			if gotMethod != http.MethodPost {
				t.Errorf("method = %q, want POST", gotMethod)
			}
			if !strings.HasPrefix(gotPath, tc.wantPrefix) || !strings.Contains(gotPath, tc.resource) || !strings.HasSuffix(gotPath, "/credentials:retrieve") {
				t.Errorf("path = %q, want prefix %q containing %q and suffix :retrieve", gotPath, tc.wantPrefix, tc.resource)
			}
			if gotUserID != "user-1" {
				t.Errorf("body userId = %q, want %q", gotUserID, "user-1")
			}
			if !slices.Equal(gotScopes, []string{"scope-a", "scope-b"}) {
				t.Errorf("body scopes = %q, want [scope-a scope-b]", gotScopes)
			}
			// ContinueURI is what makes the 3-legged flow work, so a wrong tag
			// here would be silent and expensive.
			if gotContinueURI != "https://example.test/continue" {
				t.Errorf("body continueUri = %q, want %q", gotContinueURI, "https://example.test/continue")
			}
		})
	}
}

func TestRetrieveHTTPError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "nope", http.StatusInternalServerError)
	}))
	defer srv.Close()

	_, err := newTestClient(t, srv).RetrieveCredential(t.Context(),
		Request{Resource: authProviderResource, UserID: "u"})
	var apiErr *APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("error = %v, want *APIError", err)
	}
	if apiErr.StatusCode != http.StatusInternalServerError {
		t.Errorf("StatusCode = %d, want %d", apiErr.StatusCode, http.StatusInternalServerError)
	}
	if !strings.Contains(apiErr.Body, "nope") {
		t.Errorf("Body = %q, want it to carry the response body", apiErr.Body)
	}
}

func TestRetrieveValidatesRequest(t *testing.T) {
	tests := []struct {
		name string
		req  Request
	}{
		{name: "missing resource", req: Request{UserID: "u"}},
		{name: "missing user id", req: Request{Resource: authProviderResource}},
		{name: "resource path traversal", req: Request{Resource: "projects/p/../q/authProviders/a", UserID: "u"}},
		{name: "resource query injection", req: Request{Resource: "projects/p/authProviders/a?x=1", UserID: "u"}},
		{name: "resource with space", req: Request{Resource: "projects/p/authProviders/a b", UserID: "u"}},
	}
	// Point at a live server: a client with no endpoint fails at transport for
	// every input, which cannot tell a rejected request from an unreachable one.
	var hits atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		hits.Add(1)
	}))
	defer srv.Close()
	c := newTestClient(t, srv)
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			hits.Store(0)
			_, err := c.RetrieveCredential(t.Context(), tc.req)
			if err == nil {
				t.Fatalf("RetrieveCredential(%+v) = nil error, want error", tc.req)
			}
			if !strings.Contains(err.Error(), "requires a") && !strings.Contains(err.Error(), "invalid characters") {
				t.Errorf("error = %v, want a request-validation error", err)
			}
			if got := hits.Load(); got != 0 {
				t.Errorf("credentials service called %d time(s); a rejected request must not reach the wire", got)
			}
		})
	}
}

func TestNewClient(t *testing.T) {
	t.Run("defaults", func(t *testing.T) {
		// Supply HTTPClient so the constructor skips the ADC lookup (offline test).
		c, err := NewClient(t.Context(), &Config{HTTPClient: http.DefaultClient})
		if err != nil {
			t.Fatalf("NewClient() error = %v", err)
		}
		if c.agentIdentityURL != defaultAgentIdentityURL {
			t.Errorf("agentIdentityURL = %q, want %q", c.agentIdentityURL, defaultAgentIdentityURL)
		}
		if c.connectorURL != defaultConnectorURL {
			t.Errorf("connectorURL = %q, want %q", c.connectorURL, defaultConnectorURL)
		}
		if c.pollTimeout != defaultPollTimeout {
			t.Errorf("pollTimeout = %v, want %v", c.pollTimeout, defaultPollTimeout)
		}
		if c.initialBackoff != defaultInitialBackoff {
			t.Errorf("initialBackoff = %v, want %v", c.initialBackoff, defaultInitialBackoff)
		}
	})
	t.Run("nil config uses defaults", func(t *testing.T) {
		// The nil-Config path the exported doc promises; it takes the ADC branch.
		fakeADC(t)
		c, err := NewClient(t.Context(), nil)
		if err != nil {
			t.Fatalf("NewClient() error = %v", err)
		}
		if c.httpClient == nil {
			t.Error("httpClient = nil, want an ADC-backed client")
		}
		if c.agentIdentityURL != defaultAgentIdentityURL || c.connectorURL != defaultConnectorURL {
			t.Errorf("endpoints = %q / %q, want the defaults", c.agentIdentityURL, c.connectorURL)
		}
		if c.pollTimeout != defaultPollTimeout {
			t.Errorf("pollTimeout = %v, want %v", c.pollTimeout, defaultPollTimeout)
		}
	})
	t.Run("trims endpoint trailing slash", func(t *testing.T) {
		c, err := NewClient(t.Context(), &Config{
			HTTPClient:            http.DefaultClient,
			AgentIdentityEndpoint: "https://ai.example.com/",
			ConnectorEndpoint:     "https://conn.example.com/",
		})
		if err != nil {
			t.Fatalf("NewClient() error = %v", err)
		}
		if c.agentIdentityURL != "https://ai.example.com" {
			t.Errorf("agentIdentityURL = %q, want trailing slash trimmed", c.agentIdentityURL)
		}
		if c.connectorURL != "https://conn.example.com" {
			t.Errorf("connectorURL = %q, want trailing slash trimmed", c.connectorURL)
		}
	})
}

func TestMapCredential(t *testing.T) {
	tests := []struct {
		name       string
		header     string
		token      string
		wantBearer string // non-empty => expect bearer token
		wantAPIKey [2]string
		wantErr    bool
	}{
		{name: "authorization bearer", header: "Authorization: Bearer", token: "t", wantBearer: "t"},
		{name: "authorization bearer lowercase", header: "authorization: bearer", token: "t", wantBearer: "t"},
		{name: "custom header", header: "X-Goog-Api-Key", token: "k", wantAPIKey: [2]string{"X-Goog-Api-Key", "k"}},
		// A name that is NOT X-Goog-Api-Key: with the mirror deleted, the two
		// assertions in wantAPIKey would otherwise read the same header and pass.
		{name: "third-party header is mirrored", header: "X-Acme-Token", token: "k", wantAPIKey: [2]string{"X-Acme-Token", "k"}},
		{name: "empty header", header: "", token: "t", wantErr: true},
		{name: "empty token", header: "Authorization: Bearer", token: "", wantErr: true},
		{name: "header carrying a scheme is not a usable field name", header: "X-Api-Key: Token", token: "k", wantErr: true},
		{name: "bare authorization maps to an api key", header: "Authorization", token: "k", wantAPIKey: [2]string{"Authorization", "k"}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			cred, err := mapCredential(tc.header, tc.token)
			if tc.wantErr {
				if err == nil {
					t.Fatal("mapCredential() = nil error, want error")
				}
				return
			}
			if err != nil {
				t.Fatalf("mapCredential() error = %v", err)
			}
			switch {
			case tc.wantBearer != "":
				wantBearer(t, cred, tc.wantBearer)
			default:
				wantAPIKey(t, cred, tc.wantAPIKey[0], tc.wantAPIKey[1])
			}
		})
	}
}

// TestRetrieveContextCanceledWhilePending verifies that canceling the context
// aborts a pending poll promptly (no hang) and surfaces context.Canceled.
// The rejected header name is service-controlled and reaches the error by a
// third path, separate from a response body and an operation message.
func TestMapCredentialCapsHeaderNameInError(t *testing.T) {
	_, err := mapCredential(strings.Repeat("x", 900_000)+": Token", "SECRET-TOKEN")
	if err == nil {
		t.Fatal("mapCredential() = nil error, want error")
	}
	if len(err.Error()) > 2*maxErrorBody {
		t.Errorf("error is %d bytes, want the header name capped to %d", len(err.Error()), maxErrorBody)
	}
	if strings.Contains(err.Error(), "SECRET-TOKEN") {
		t.Error("error carries the token")
	}
}

func TestRetrieveContextCanceledWhilePending(t *testing.T) {
	srv, _ := sequenceServer(`{"pending":{}}`) // never resolves
	defer srv.Close()

	c := newTestClient(t, srv)
	// A backoff far longer than the window asserted below. Without the ctx arm of
	// the poll wait, cancellation is only noticed on the next request, so the
	// outcome still holds and only the promptness — the point here — is lost.
	c.pollTimeout = time.Minute
	c.initialBackoff = 30 * time.Second

	ctx, cancel := context.WithCancel(t.Context())
	time.AfterFunc(20*time.Millisecond, cancel)

	start := time.Now()
	_, err := c.RetrieveCredential(ctx, Request{Resource: authProviderResource, UserID: "u"})
	elapsed := time.Since(start)

	if !errors.Is(err, context.Canceled) {
		t.Fatalf("RetrieveCredential() error = %v, want context.Canceled", err)
	}
	if elapsed > 5*time.Second {
		t.Errorf("returned after %v, want promptly after cancellation (backoff was %v)", elapsed, c.initialBackoff)
	}
}

// TestRetrievePollTimeout verifies that a service stuck in the non-interactive
// pending state past the poll timeout surfaces ErrPollTimeout (no hang).
func TestRetrievePollTimeout(t *testing.T) {
	srv, _ := sequenceServer(`{"pending":{}}`) // never resolves
	defer srv.Close()

	c := newTestClient(t, srv)
	c.pollTimeout = 30 * time.Millisecond

	_, err := c.RetrieveCredential(t.Context(),
		Request{Resource: authProviderResource, UserID: "u"})
	if !errors.Is(err, ErrPollTimeout) {
		t.Fatalf("RetrieveCredential() error = %v, want ErrPollTimeout", err)
	}
}

// newTestClient points both service endpoints at srv and uses a tiny backoff so
// polling tests are fast.
func newTestClient(t *testing.T, srv *httptest.Server) *Client {
	t.Helper()
	c, err := NewClient(t.Context(), &Config{
		HTTPClient:            srv.Client(),
		AgentIdentityEndpoint: srv.URL,
		ConnectorEndpoint:     srv.URL,
		PollTimeout:           2 * time.Second,
	})
	if err != nil {
		t.Fatalf("NewClient() error = %v", err)
	}
	c.initialBackoff = time.Millisecond
	return c
}

// sequenceServer replies with bodies in order, repeating the last one.
func sequenceServer(bodies ...string) (*httptest.Server, *int32) {
	var n int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		i := int(atomic.AddInt32(&n, 1)) - 1
		if i >= len(bodies) {
			i = len(bodies) - 1
		}
		_, _ = io.WriteString(w, bodies[i])
	}))
	return srv, &n
}

// wantBearer fails t unless cred is an auth.BearerCredential carrying token.
func wantBearer(t *testing.T, cred auth.Credential, token string) {
	t.Helper()
	b, ok := cred.(auth.BearerCredential)
	if !ok {
		t.Fatalf("credential = %#v, want auth.BearerCredential", cred)
	}
	if b.Token != token {
		t.Fatalf("bearer token = %q, want %q", b.Token, token)
	}
}

// wantAPIKey fails t unless applying cred sets the named header and the
// X-Goog-Api-Key mirror (adk-python parity) to value.
func wantAPIKey(t *testing.T, cred auth.Credential, name, value string) {
	t.Helper()
	h := http.Header{}
	if err := cred.Apply(h); err != nil {
		t.Fatalf("cred.Apply() error = %v", err)
	}
	if got := h.Get(name); got != value {
		t.Errorf("header %q = %q, want %q", name, got, value)
	}
	if got := h.Get("X-Goog-Api-Key"); got != value {
		t.Errorf("X-Goog-Api-Key = %q, want %q (adk-python parity)", got, value)
	}
}

// TestNewClientRefusesRedirects pins the ADC client's redirect guard: oauth2's
// transport re-signs every hop below net/http's cross-host stripping, so a
// followed redirect would hand the cloud-platform token to the target and let
// it dictate the returned credential. Drives the real ADC branch of NewClient,
// so deleting the guard fails here.
func TestNewClientRefusesRedirects(t *testing.T) {
	fakeADC(t)

	var targetSawAuth string
	target := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		targetSawAuth = r.Header.Get("Authorization")
		_, _ = io.WriteString(w, `{"success":{"token":"attacker","header":"Authorization: Bearer"}}`)
	}))
	defer target.Close()
	redirector := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, target.URL+r.URL.Path, http.StatusTemporaryRedirect)
	}))
	defer redirector.Close()

	c, err := NewClient(t.Context(), &Config{
		AgentIdentityEndpoint: redirector.URL,
		ConnectorEndpoint:     redirector.URL,
	})
	if err != nil {
		t.Fatalf("NewClient() error = %v", err)
	}
	cred, err := c.RetrieveCredential(t.Context(), Request{Resource: authProviderResource, UserID: "u"})
	if err == nil {
		t.Fatalf("RetrieveCredential() = %#v, nil error; want the 3xx surfaced as an error", cred)
	}
	if targetSawAuth != "" {
		t.Errorf("redirect target received Authorization %q; the token must not leave the configured host", targetSawAuth)
	}
}

// TestNewClientOutlivesConstructionCtx pins the token source's detachment from
// the construction context. Callers build the client inside a bounded,
// request-scoped context (the auth/gcp credential provider does exactly that),
// and every token minted after that context ends must still authenticate.
func TestNewClientOutlivesConstructionCtx(t *testing.T) {
	fakeADC(t)
	srv, _ := sequenceServer(`{"success":{"token":"tok","header":"Authorization: Bearer"}}`)
	defer srv.Close()

	ctx, cancel := context.WithCancel(t.Context())
	c, err := NewClient(ctx, &Config{AgentIdentityEndpoint: srv.URL, ConnectorEndpoint: srv.URL})
	if err != nil {
		t.Fatalf("NewClient() error = %v", err)
	}
	cancel()

	got, err := c.RetrieveCredential(t.Context(), Request{Resource: authProviderResource, UserID: "u"})
	if err != nil {
		t.Fatalf("RetrieveCredential() error = %v", err)
	}
	wantBearer(t, got.Credential, "tok")
}

// fakeADC points Application Default Credentials at a local token server so the
// ADC branch of NewClient runs offline. The token expires immediately, so every
// call mints a fresh one and the token source's own context stays observable.
func fakeADC(t *testing.T) {
	t.Helper()
	tokenSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"access_token":"ADC-TOKEN","token_type":"Bearer","expires_in":1}`)
	}))
	t.Cleanup(tokenSrv.Close)

	adc := filepath.Join(t.TempDir(), "adc.json")
	if err := os.WriteFile(adc, []byte(`{"type":"authorized_user","client_id":"c","client_secret":"s","refresh_token":"r","token_uri":"`+tokenSrv.URL+`"}`), 0o600); err != nil {
		t.Fatalf("write fake ADC: %v", err)
	}
	t.Setenv("GOOGLE_APPLICATION_CREDENTIALS", adc)
}

// TestDoPostOversizeKeepsStatus: an error page big enough to trip the body cap
// must still report its status, the most actionable field.
func TestDoPostOversizeKeepsStatus(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadGateway)
		_, _ = io.WriteString(w, strings.Repeat("x", (1<<20)+10))
	}))
	defer srv.Close()
	_, err := newTestClient(t, srv).RetrieveCredential(t.Context(),
		Request{Resource: authProviderResource, UserID: "u"})
	var apiErr *APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("error = %v, want *APIError", err)
	}
	if apiErr.StatusCode != http.StatusBadGateway {
		t.Errorf("StatusCode = %d, want %d", apiErr.StatusCode, http.StatusBadGateway)
	}
	// Pin the truncation, not just the helper: without it the whole 1 MiB page
	// rides along in the error.
	if len(apiErr.Body) > maxErrorBody+len("...") {
		t.Errorf("Body = %d bytes, want it capped to %d", len(apiErr.Body), maxErrorBody)
	}
}

// A service-controlled operation message must be capped like any response body;
// it reaches the error by a different path than doPost's body.
func TestRetrieveConnectorErrorMessageIsCapped(t *testing.T) {
	srv, _ := sequenceServer(`{"error":{"code":7,"message":"` + strings.Repeat("x", 900_000) + `"}}`)
	defer srv.Close()

	_, err := newTestClient(t, srv).RetrieveCredential(t.Context(),
		Request{Resource: connectorResource, UserID: "u"})
	if err == nil {
		t.Fatal("RetrieveCredential() = nil error, want error")
	}
	if len(err.Error()) > 2*maxErrorBody {
		t.Errorf("error is %d bytes, want the message capped to %d", len(err.Error()), maxErrorBody)
	}
}

// A 2xx body over the cap must be rejected, not handed to json.Unmarshal
// truncated (and thus garbled).
func TestDoPostRejectsOversizeSuccess(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, `{"success":{"token":"t","header":"Authorization: Bearer"}}`+strings.Repeat(" ", 1<<20))
	}))
	defer srv.Close()
	_, err := newTestClient(t, srv).RetrieveCredential(t.Context(),
		Request{Resource: authProviderResource, UserID: "u"})
	if err == nil || !strings.Contains(err.Error(), "exceeded") {
		t.Fatalf("error = %v, want the oversize response rejected", err)
	}
}

// TestDoPostEscapesErrorBody: a service-controlled body must not be able to
// forge log lines through the returned error.
func TestDoPostEscapesErrorBody(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
		_, _ = io.WriteString(w, "unavailable\r\nINFO auth: credential granted user=victim")
	}))
	defer srv.Close()
	_, err := newTestClient(t, srv).RetrieveCredential(t.Context(),
		Request{Resource: authProviderResource, UserID: "u"})
	if err == nil {
		t.Fatal("RetrieveCredential() = nil error, want error")
	}
	if strings.Contains(err.Error(), "\r\n") {
		t.Errorf("error carries raw control bytes: %q", err.Error())
	}
	if !strings.Contains(err.Error(), `\r\n`) {
		t.Errorf("error = %q, want the body escaped", err.Error())
	}
}

func TestTruncateForError(t *testing.T) {
	tests := []struct {
		name string
		in   string
		want string
	}{
		{name: "short is unchanged", in: "nope", want: "nope"},
		{name: "long is cut", in: strings.Repeat("a", 2000), want: strings.Repeat("a", 1024) + "..."},
		// A body need not be UTF-8; an unbounded backup would walk to 0 here and
		// throw away every byte of diagnostic context.
		{name: "non utf8 keeps context", in: strings.Repeat("\x80", 2000), want: strings.Repeat("\x80", 1024) + "..."},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			// Report the tail as well as the length: a body that is cut at the
			// wrong place can still come out the right size.
			if got := truncateForError(tc.in); got != tc.want {
				t.Errorf("truncateForError() = %d bytes ending %q, want %d bytes ending %q",
					len(got), got[max(0, len(got)-8):], len(tc.want), tc.want[max(0, len(tc.want)-8):])
			}
		})
	}
}

func TestNewClientRejectsNegativePollTimeout(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
	defer srv.Close()
	c, err := NewClient(t.Context(), &Config{HTTPClient: srv.Client(), PollTimeout: -time.Second})
	if err != nil {
		t.Fatalf("NewClient() error = %v", err)
	}
	if c.pollTimeout != defaultPollTimeout {
		t.Errorf("pollTimeout = %v, want the default %v (a negative value must not mean 'never retry')",
			c.pollTimeout, defaultPollTimeout)
	}
}
