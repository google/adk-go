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
	"fmt"
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
			cred, err := c.RetrieveCredential(t.Context(),
				Request{Resource: tc.resource, UserID: "u"})

			switch {
			case tc.wantBearer != "":
				if err != nil {
					t.Fatalf("RetrieveCredential() error = %v", err)
				}
				wantBearer(t, cred, tc.wantBearer)
			case tc.wantAPIKey[0] != "":
				if err != nil {
					t.Fatalf("RetrieveCredential() error = %v", err)
				}
				wantAPIKey(t, cred, tc.wantAPIKey[0], tc.wantAPIKey[1])
			case tc.wantConsent[0] != "":
				var consent *auth.ConsentRequiredError
				if !errors.As(err, &consent) {
					t.Fatalf("error = %v, want *auth.ConsentRequiredError", err)
				}
				// Print the fields, not consent: %v on a *ConsentRequiredError
				// goes through Error(), which reports neither of them.
				if consent.AuthURI != tc.wantConsent[0] || consent.Nonce != tc.wantConsent[1] {
					t.Errorf("consent = {authURI:%q nonce:%q}, want {authURI:%q nonce:%q}",
						consent.AuthURI, consent.Nonce, tc.wantConsent[0], tc.wantConsent[1])
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
	// Each row names the message it expects, not a substring true of every error
	// this method returns. Delete validateResource and these requests run past the
	// validation returns to the deferred wrap installed below them, which appends
	// `(resource %q)` to everything after it — so an assertion on "resource "
	// alone stays green with the guard gone, which is the guard against path
	// traversal. Naming the message also keeps the two rejection reasons apart
	// from each other, which is the distinction the charset and the segment check
	// exist to draw.
	const (
		charset = "has invalid characters"
		segment = "has an empty or relative path segment"
		missing = "requires a"
	)
	tests := []struct {
		name    string
		req     Request
		wantMsg string
	}{
		{name: "missing resource", req: Request{UserID: "u"}, wantMsg: missing},
		{name: "missing user id", req: Request{Resource: authProviderResource}, wantMsg: missing},
		{name: "resource path traversal", req: Request{Resource: "projects/p/../q/authProviders/a", UserID: "u"}, wantMsg: segment},
		{name: "resource query injection", req: Request{Resource: "projects/p/authProviders/a?x=1", UserID: "u"}, wantMsg: charset},
		{name: "resource with space", req: Request{Resource: "projects/p/authProviders/a b", UserID: "u"}, wantMsg: charset},
		// A name that normalizes to a different one routes to a different service
		// than the one validateResource inspected.
		{name: "resource empty segment", req: Request{Resource: "projects/p//authProviders/a", UserID: "u"}, wantMsg: segment},
		{name: "resource trailing slash", req: Request{Resource: "projects/p/locations/l/connectors/c/", UserID: "u"}, wantMsg: segment},
		{name: "resource dot segment", req: Request{Resource: "projects/p/locations/l/connectors/c/.", UserID: "u"}, wantMsg: segment},
		// Percent-escapes are rejected by the charset, not decoded: the name is
		// interpolated into a URL, so an escape that survives becomes traversal or
		// a segment break once the server decodes it.
		{name: "resource percent-escaped dot", req: Request{Resource: "projects/p/authProviders/a%2e%2e", UserID: "u"}, wantMsg: charset},
		{name: "resource percent-escaped slash", req: Request{Resource: "projects/p%2flocations/authProviders/a", UserID: "u"}, wantMsg: charset},
		{name: "resource bare percent", req: Request{Resource: "projects/p/authProviders/a%", UserID: "u"}, wantMsg: charset},
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
			if !strings.Contains(err.Error(), tc.wantMsg) {
				t.Errorf("error = %v, want it to say %q", err, tc.wantMsg)
			}
			if got := hits.Load(); got != 0 {
				t.Errorf("credentials service called %d time(s); a rejected request must not reach the wire", got)
			}
		})
	}
}

// TestRetrieveAcceptsResourceNames pins the other side of the boundary
// TestRetrieveValidatesRequest guards. Both of these were widened when the
// per-segment check replaced a substring test for "..", and a widening a
// rejection table cannot see is a widening nothing would notice being undone —
// or being taken further.
func TestRetrieveAcceptsResourceNames(t *testing.T) {
	tests := []struct {
		name, resource string
	}{
		// A domain-scoped project id. The colon is why the charset had to widen,
		// and it is safe only because the name always follows a scheme, a host and
		// a version segment, where a colon cannot begin a scheme.
		{name: "domain-scoped project id", resource: "projects/example.com:my-project/locations/l/authProviders/a"},
		// Dots inside a segment, as opposed to a "." or ".." segment of their own.
		// The old substring check rejected these, while path.Clean leaves them
		// alone, so the name the server resolves is the one validated and routed.
		{name: "dots inside a segment", resource: "projects/p/locations/l/authProviders/a..b"},
		{name: "leading dot in a segment", resource: "projects/p/locations/l/authProviders/.hidden"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var gotPath string
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				gotPath = r.URL.Path
				w.Header().Set("Content-Type", "application/json")
				_, _ = w.Write([]byte(`{"success":{"header":"Authorization: Bearer","token":"tok"}}`))
			}))
			defer srv.Close()

			if _, err := newTestClient(t, srv).RetrieveCredential(t.Context(),
				Request{Resource: tc.resource, UserID: "u"}); err != nil {
				t.Fatalf("RetrieveCredential(%q) error = %v, want it accepted", tc.resource, err)
			}
			// The name must reach the wire unchanged: validation and routing both
			// ran on the string the server is about to resolve.
			if want := "/v1/" + tc.resource + "/credentials:retrieve"; gotPath != want {
				t.Errorf("request path = %q, want %q", gotPath, want)
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

// TestMapCredentialCapsHeaderNameInError pins the cap on a rejected header name.
// It is service-controlled and reaches the error by a third path, separate from
// a response body and an operation message.
//
// Sized inside the examinable length on purpose. Past it the name is withheld
// before the cap is consulted and the assertion below measures the withheld
// sentence, which no change to the cap can lengthen.
func TestMapCredentialCapsHeaderNameInError(t *testing.T) {
	_, err := mapCredential(strings.Repeat("x", maxErrorBody*4)+": Token", "SECRET-TOKEN")
	if err == nil {
		t.Fatal("mapCredential() = nil error, want error")
	}
	if len(err.Error()) > 2*maxErrorBody {
		t.Errorf("error is %d bytes, want the header name capped to %d", len(err.Error()), maxErrorBody)
	}
	// Cannot fail against today's code — no error arm interpolates the token — and
	// kept as a forward guard, since the token is the one value in this function
	// that must never reach an error however the message is later reworded.
	if strings.Contains(err.Error(), "SECRET-TOKEN") {
		t.Error("error carries the token")
	}
}

// TestRetrieveContextCanceledWhilePending verifies that canceling the context
// aborts a pending poll promptly (no hang) and surfaces context.Canceled.
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

	cred, err := c.RetrieveCredential(t.Context(), Request{Resource: authProviderResource, UserID: "u"})
	if err != nil {
		t.Fatalf("RetrieveCredential() error = %v", err)
	}
	wantBearer(t, cred, "tok")
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
//
// Two sizes, because the two paths bound the body by different means and only the
// first can see the cap. A page the whole-text check can examine is shown and
// capped. A page past that length is withheld, and a size assertion on the
// withheld sentence holds however the cap is broken — which is what the single
// 1 MiB row this replaced had quietly become.
func TestDoPostOversizeKeepsStatus(t *testing.T) {
	for _, tc := range []struct {
		name      string
		size      int
		wantShown bool
	}{
		{"a page the check can examine", maxErrorBody * 4, true},
		{"a page past the examinable length", (1 << 20) + 10, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.WriteHeader(http.StatusBadGateway)
				_, _ = io.WriteString(w, strings.Repeat("x", tc.size))
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
			if shown := apiErr.Body != withheldText; shown != tc.wantShown {
				t.Fatalf("Body shown = %v, want %v (Body is %d bytes)", shown, tc.wantShown, len(apiErr.Body))
			}
			if len(apiErr.Body) > maxErrorBody+len("...") {
				t.Errorf("Body = %d bytes, want it capped to %d", len(apiErr.Body), maxErrorBody)
			}
		})
	}
}

// A service-controlled operation message must be capped like any response body,
// and it reaches the error by a different path than doPost's body.
//
// Sized inside the examinable length, for the reason given on
// TestMapCredentialCapsHeaderNameInError.
func TestRetrieveConnectorErrorMessageIsCapped(t *testing.T) {
	srv, _ := sequenceServer(`{"error":{"code":7,"message":"` + strings.Repeat("x", maxErrorBody*4) + `"}}`)
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
			if got, cut := truncateForError(tc.in); got != tc.want || cut != (tc.in != tc.want) {
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

// TestMapCredentialRedactsTheActingUserInError pins the third service-text path
// against the acting user, not only against length.
//
// A rejected header name is service-controlled, so a service that echoes the
// userId into it puts the acting user in an error string. The two sibling paths
// scrub before reporting — doPost through redactedForError, connectorOperation.result
// the same — and this one only capped.
//
// The user is echoed INSIDE a larger name on purpose. With the name equal to the
// user, a scrubbed error and an error that dropped the service text entirely read
// the same, so errors.New("bad header") would pass. The surrounding text has to
// survive for the assertion to be about redaction. The last case is the negative
// control: a rejected name carrying no secret must come back intact, or the scrub
// is a blanket drop rather than something keyed on the acting user.
func TestMapCredentialRedactsTheActingUserInError(t *testing.T) {
	const user = "alice@example.test"
	for _, tc := range []struct {
		name        string
		header      string // "@" and " " are not RFC 9110 token characters, so both are rejected.
		wantAbsent  string
		wantPresent []string
	}{{
		name:        "the user echoed inside a larger name",
		header:      "X-User-" + user + "-Token",
		wantAbsent:  user,
		wantPresent: []string{"X-User-", "-Token", "not a usable HTTP header name"},
	}, {
		name:        "the name is exactly the user",
		header:      user,
		wantAbsent:  user,
		wantPresent: []string{"not a usable HTTP header name"},
	}, {
		name:        "a rejected name with no secret in it survives",
		header:      "not a header",
		wantPresent: []string{"not a header", "not a usable HTTP header name"},
	}} {
		t.Run(tc.name, func(t *testing.T) {
			srv, _ := sequenceServer(`{"success":{"header":"` + tc.header + `","token":"t"}}`)
			defer srv.Close()

			_, err := newTestClient(t, srv).RetrieveCredential(t.Context(), Request{
				Resource: "projects/p/locations/l/authProviders/a",
				UserID:   user,
			})
			if err == nil {
				t.Fatal("RetrieveCredential() = nil error, want the header name rejected")
			}
			if tc.wantAbsent != "" && strings.Contains(err.Error(), tc.wantAbsent) {
				t.Errorf("error carries the acting user %q: %v", tc.wantAbsent, err)
			}
			for _, want := range tc.wantPresent {
				// Case-insensitively: redact lowercases the text it scrubs, so a
				// name it touched comes back lowered. That is the documented price
				// of not mapping offsets between two spellings of the same string.
				if !strings.Contains(strings.ToLower(err.Error()), strings.ToLower(want)) {
					t.Errorf("error lost %q, so it is no longer diagnostic: %v", want, err)
				}
			}
		})
	}
}

// TestServiceErrorsRedactTheActingUser pins the WIRING, which is a different
// claim from the one every other test here makes.
//
// redactedForError is pinned to death — an exact-output table, ~584 generated bodies,
// a 2.4M-combination differential and a fuzz target — and a scrub that is never
// CALLED passes all of it. Two of the four sites that carry service text had no
// end-to-end test, and the ContinueURI argument had none at any of the four:
// replacing doPost's redactedForError call with truncateForError (which is what this
// package did before), replacing connectorOperation.result's, or deleting
// req.ContinueURI from all four secret lists each left the whole package green.
//
// So each row below fails on one of those edits and on nothing else. The
// surviving wording is asserted too, or a blanket drop of all service text would
// pass every row and lose the diagnostic the error exists for.
func TestServiceErrorsRedactTheActingUser(t *testing.T) {
	const user = "alice@example.test"
	// Deliberately free of the user id, so the two secrets can be asserted apart.
	// Where one contains the other they are redacted as one range, which is
	// TestRedactAcrossSeveralSecrets' subject rather than this test's.
	const uri = "https://app.test/cb?state=opaque"

	for _, tc := range []struct {
		name        string
		resource    string
		status      int
		body        string
		wantAbsent  []string
		wantPresent []string
	}{{
		name:        "an error body echoing the acting user",
		resource:    authProviderResource,
		status:      http.StatusForbidden,
		body:        "permission denied for " + user,
		wantAbsent:  []string{user},
		wantPresent: []string{"permission denied for"},
	}, {
		name:        "an error body echoing the continue uri",
		resource:    authProviderResource,
		status:      http.StatusBadRequest,
		body:        "continueUri not registered: " + uri,
		wantAbsent:  []string{uri},
		wantPresent: []string{"continueuri not registered"},
	}, {
		// connectorOperation.result's error arm, whose comment claims "the same
		// treatment doPost gives a response body". Nothing checked that it did.
		name:        "a connector operation message echoing the acting user",
		resource:    connectorResource,
		status:      http.StatusOK,
		body:        `{"done":true,"error":{"code":7,"message":"user ` + user + ` is not permitted"}}`,
		wantAbsent:  []string{user},
		wantPresent: []string{"is not permitted"},
	}, {
		name:        "a connector operation message echoing the continue uri",
		resource:    connectorResource,
		status:      http.StatusOK,
		body:        `{"done":true,"error":{"code":3,"message":"bad continueUri ` + uri + `"}}`,
		wantAbsent:  []string{uri},
		wantPresent: []string{"bad continueuri"},
	}, {
		// The connector's own doPost secret list, which the 200 rows above never
		// reach: they exercise connectorOperation.result instead. Only a non-2xx
		// on a connector resource gets here, and dropping req.ContinueURI at
		// connector.go's doPost call left the package green without this row.
		name:        "a connector error body echoing the continue uri",
		resource:    connectorResource,
		status:      http.StatusForbidden,
		body:        "continueUri not registered: " + uri,
		wantAbsent:  []string{uri},
		wantPresent: []string{"continueuri not registered"},
	}, {
		// mapCredential's secret list, reached only by a 2xx whose header name is
		// unusable. The test that covers this path elsewhere builds its request
		// without a ContinueURI, so that argument was unpinned too.
		name:        "an unusable header name echoing the continue uri",
		resource:    authProviderResource,
		status:      http.StatusOK,
		body:        `{"success":{"header":"X-` + uri + `","token":"t"}}`,
		wantAbsent:  []string{uri},
		wantPresent: []string{"not a usable HTTP header name"},
	}, {
		// The negative control. An error carrying neither identifier has to come
		// back whole, or the rows above are satisfied by dropping all service
		// text rather than by scrubbing it.
		name:        "an error carrying neither identifier survives intact",
		resource:    authProviderResource,
		status:      http.StatusServiceUnavailable,
		body:        "backend overloaded, retry later",
		wantPresent: []string{"backend overloaded, retry later"},
	}} {
		t.Run(tc.name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.WriteHeader(tc.status)
				_, _ = io.WriteString(w, tc.body)
			}))
			defer srv.Close()

			_, err := newTestClient(t, srv).RetrieveCredential(t.Context(), Request{
				Resource: tc.resource, UserID: user, ContinueURI: uri,
			})
			if err == nil {
				t.Fatal("RetrieveCredential() = nil error, want the service's error")
			}
			for _, absent := range tc.wantAbsent {
				if strings.Contains(err.Error(), absent) {
					t.Errorf("error carries %q: %v", absent, err)
				}
			}
			for _, want := range tc.wantPresent {
				// Case-insensitively: redact lowercases the text it scrubbed, which
				// is the documented price of not mapping offsets between two
				// spellings of the same string.
				if !strings.Contains(strings.ToLower(err.Error()), strings.ToLower(want)) {
					t.Errorf("error lost %q, so it is no longer diagnostic: %v", want, err)
				}
			}
		})
	}
}

// TestErrorBodyIsCutBeforeTheScrubNotAfter pins the order the cap and the scrub
// run in, which is not a detail: both of the simpler orders leak.
//
// Cutting the scrubbed text lets redaction's own shortening pull bytes into view
// that the same cut on the raw response would have hidden — every echo of the
// ContinueURI collapses to a ten-byte marker, and whatever followed them rides up
// into the window. Cutting the raw text instead slices an occurrence straddling
// the cut in half, and half an identifier matches nothing, so its head is copied
// straight out.
func TestErrorBodyIsCutBeforeTheScrubNotAfter(t *testing.T) {
	const user = "alice@example.test"
	const uri = "https://app.example.test/callback"

	t.Run("redaction does not promote what the cut hid", func(t *testing.T) {
		// Percent-encoding is outside what the scrub decodes (redactedForError says so),
		// so if this tail reaches the window it is returned as it stands. It must
		// not reach it: past the first kilobyte, it is not this error's to show.
		body := strings.Repeat(uri+" ", 34) + "alice%40example.test"
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.WriteHeader(http.StatusForbidden)
			_, _ = io.WriteString(w, body)
		}))
		defer srv.Close()

		_, err := newTestClient(t, srv).RetrieveCredential(t.Context(), Request{
			Resource: authProviderResource, UserID: user, ContinueURI: uri,
		})
		var apiErr *APIError
		if !errors.As(err, &apiErr) {
			t.Fatalf("error = %v, want *APIError", err)
		}
		if strings.Contains(apiErr.Body, "alice%40example.test") {
			t.Errorf("Body reaches past the cut: %q", apiErr.Body)
		}
		if !strings.Contains(apiErr.Body, redactedMarker) {
			t.Errorf("Body = %q, want the echoed uri redacted rather than the text dropped", apiErr.Body)
		}
	})

	t.Run("an escaped occurrence straddling the cut is not sliced in half", func(t *testing.T) {
		// The escaped spelling is invisible to the scrub, so the cut is what would
		// cut it, leaving a readable head of the address. Each pad puts a different
		// byte of the escape on the boundary.
		for _, pad := range []int{1006, 1010, 1014, 1018, 1020, 1022, 1023} {
			got := redactedForError(strings.Repeat("x", pad)+`alice\u0040example.test`, user)
			for n := len(user); n >= 3; n-- {
				if strings.Contains(strings.ToLower(got), user[:n]) {
					t.Errorf("pad %d: %q of the acting user survives: %q", pad, user[:n], got)
					break
				}
			}
		}
	})
}

// TestRedactAcrossSeveralSecrets pins the two ways one pass per value went wrong.
//
// Both were reachable in production, where every call site passes UserID and
// ContinueURI together, and neither was caught by a green suite or by a fuzz of
// the single-secret shape.
func TestRedactAcrossSeveralSecrets(t *testing.T) {
	for _, tc := range []struct {
		name    string
		in      string
		secrets []string
		want    string
	}{{
		// Redacting the URI first inserted "[redacted]", and the pass for "e" then
		// matched the e's inside that marker.
		name:    "a secret that occurs inside the marker",
		in:      "error https://app.example.test/cb user e",
		secrets: []string{"https://app.example.test/cb", "e"},
		want:    "[redacted]rror [redacted] us[redacted]r [redacted]",
	}, {
		// Redacting the user first broke the URI that contains it, so the URI
		// matched nothing afterwards and its head survived.
		name:    "a secret contained in a longer secret",
		in:      "bad continueUri: https://example.test/cb?user=alice",
		secrets: []string{"alice", "https://example.test/cb?user=alice"},
		want:    "bad continueuri: [redacted]",
	}, {
		// A straddle, as opposed to containment: the URI's occurrence starts before
		// the user's and ends inside it. Choosing the earliest match covers the
		// contained and equal-start cases on its own. Without merging the ranges,
		// this one left the tail of the address in the error.
		name:    "a secret straddling the end of another",
		in:      "invalid: https://app.example.test/cb?login=alice@example.test",
		secrets: []string{"alice@example.test", "https://app.example.test/cb?login=al"},
		want:    "invalid: [redacted]",
	}, {
		name:    "the shortest straddle",
		in:      "xalice",
		secrets: []string{"xa", "alice"},
		want:    "[redacted]",
	}, {
		// A SECOND occurrence of the same secret starting inside the range the
		// first choice covered. Tracking one upcoming match per secret cannot see
		// it, and the refresh only looks forward from the cursor, so it was neither
		// redacted nor found again: this returned "[redacted]lice@example.test",
		// keeping 17 of the address's 18 bytes.
		name:    "a second occurrence inside the chosen range",
		in:      "https://cb.test/u/alice@example.test/alice@example.test",
		secrets: []string{"alice@example.test", "https://cb.test/u/alice@example.test/a"},
		want:    "[redacted]",
	}, {
		// The same shape with one secret: any value whose prefix equals its suffix
		// overlaps itself, and the overlap used to survive.
		name:    "a secret that overlaps itself",
		in:      "aaa",
		secrets: []string{"aa"},
		want:    "[redacted]",
	}, {
		// Longer than the exhaustive test's four-byte bodies, with a secret that
		// tiles them. Bounding the extension walk to a fixed lookahead — a
		// plausible way to answer its cost — passes every one of the exhaustive
		// test's 2,463,725 combinations and leaves a byte of the secret here.
		name:    "a repeated secret longer than the exhaustive bodies",
		in:      "aaaaaa",
		secrets: []string{"aa"},
		want:    "[redacted]",
	}, {
		name:    "the same with the secret embedded in text",
		in:      "before ababababab after",
		secrets: []string{"ab"},
		want:    "before [redacted] after",
	}, {
		// The no-match branch, which nothing else distinguishes: every other test
		// that reaches it uses an already-lowercase body or compares
		// case-insensitively, so deleting `if !hit { return s }` failed nothing.
		name:    "no match leaves the text alone, case and all",
		in:      "Bad Request: NOPE",
		secrets: []string{"alice", "https://example.test/cb"},
		want:    "Bad Request: NOPE",
	}, {
		name:    "order does not matter",
		in:      "bad continueUri: https://example.test/cb?user=alice",
		secrets: []string{"https://example.test/cb?user=alice", "alice"},
		want:    "bad continueuri: [redacted]",
	}} {
		t.Run(tc.name, func(t *testing.T) {
			if got := redact(tc.in, tc.secrets...); got != tc.want {
				t.Errorf("redact(%q, %q) = %q, want %q", tc.in, tc.secrets, got, tc.want)
			}
		})
	}
}

// TestRedactStaysLinear guards the scan against becoming quadratic, and the
// extension walk against being bounded away.
//
// Two shapes, because they exercise different loops. A one-character secret makes
// every range one byte wide, so the walk at the heart of the merge never runs and
// only the outer loop and the next[] refresh are on the clock — that was the whole
// of this test before, which meant the walk it is named for was untested. A
// repeated multi-byte secret is the opposite: every range is extended, so the walk
// runs for the length of the body.
//
// Both bodies are a megabyte where every byte matches, which is what a service
// echoing the acting user back at length produces, and both must collapse to one
// marker.
//
// This test measures cost and nothing else. Deleting the extension walk leaves it
// green, because a body that tiles the secret produces adjacent ranges that merge
// whatever the walk does — the walk matters where an occurrence starts inside a
// range and ends after it, which a tiling never produces.
// TestRedactMatchesReferenceExhaustively is what guards that.
func TestRedactStaysLinear(t *testing.T) {
	// doPost's own read cap, declared local to it, so it is spelled out here.
	const maxBody = 1 << 20

	for _, tc := range []struct {
		name   string
		secret string
	}{
		{"a one-character secret: the walk never runs", "e"},
		{"a repeated multi-byte secret: the walk runs for the whole body", "ab"},
		{"a long repeated secret: the walk runs and each step compares more", strings.Repeat("ab", 16)},
		// The second factor. The walk costs O(body x secret length) on a body that
		// tiles the secret, and the three shapes above are all short enough to hide
		// it — they run in single-digit milliseconds whatever the walk does.
		// Measured here: 4 KiB is ~40ms, 64 KiB ~730ms, 512 KiB ~4.2s. redact is
		// reachable with any of those, so the shape is pinned here; what a caller
		// can actually provoke through an error is capped separately, by
		// maxScrubbableSecret, and TestServiceTextCostIsBounded is where that
		// ceiling lives.
		{"a secret long enough to show the second factor", strings.Repeat("ab", maxScrubbableSecret/2)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			body := strings.Repeat(tc.secret, maxBody/len(tc.secret))
			start := time.Now()
			got := redact(body, tc.secret, "https://app.example.test/cb")
			elapsed := time.Since(start)

			// One marker for the whole run, not one per match. Before adjacent
			// ranges were merged this produced 10 MiB to be thrown away by a 1 KiB
			// cap two calls later.
			if want := len("[redacted]"); len(got) != want {
				t.Errorf("redact() produced %d bytes, want %d — every byte matches, so the "+
					"whole body is one redacted run", len(got), want)
			}
			// Generous against the measurements above, so it fails on a rewrite that
			// makes the cost worse and not on a slow machine.
			if elapsed > 10*time.Second {
				t.Errorf("redact() over %d bytes took %v; the scan is meant to be linear in the body",
					maxBody, elapsed)
			}
			t.Logf("%d bytes, secret %d bytes, every byte a match: %v", len(body), len(tc.secret), elapsed)
		})
	}
}

// TestServiceTextCostIsBounded pins the ceiling on what one failed call can
// spend, which TestRedactStaysLinear cannot see because it drives redact rather
// than the path an error actually takes.
//
// Three factors compounded before the bound existed, and all three needed a
// caller-supplied value with no length limit. The scan is O(body x value). Two
// of the values are decoded to a fixpoint, and decodeFully is quadratic on an
// escape that re-forms its own introducer — "\u005c" decodes to a backslash, so
// each pass shortens by five bytes and there are len/5 of them. And recoverable
// rebuilt both per marker-separated part, a count the SERVICE picks by writing
// the marker into its own response. Together, measured end to end through
// RetrieveCredential against a 403 with a 1020-byte body: 3m33.6s, uncancellable,
// for one call.
//
// The budget below is deliberately loose. It is here to fail on a rewrite that
// puts an unbounded factor back, not to measure a machine.
func TestServiceTextCostIsBounded(t *testing.T) {
	selfRegen := func(n int) string { return `\u005c` + strings.Repeat("u005c", n) }

	for _, tc := range []struct {
		name         string
		user         string
		body         string
		wantWithheld bool
	}{{
		// The shape that measured 3m33.6s. The value is past the bound, so it is
		// not matched at all and nothing is shown.
		name:         "a value past the bound is not matched and nothing is shown",
		user:         selfRegen(12799),
		body:         strings.Repeat(redactedMarker, 102),
		wantWithheld: true,
	}, {
		// The service's own markers no longer multiply anything: the per-value
		// forms are built once, not once per part.
		name: "a body of service-written markers does not multiply the value cost",
		user: strings.Repeat("ab", 2048),
		body: strings.Repeat(redactedMarker, 102),
	}, {
		// The worst case still reachable on the path that SHOWS something: a value
		// at the bound against the largest body the whole-text check can examine,
		// every byte of which matches.
		name: "a value at the bound against a body that tiles the window",
		user: strings.Repeat("ab", maxScrubbableSecret/2),
		body: strings.Repeat("ab", maxStraddleWindow/2),
	}, {
		// The same shape at doPost's read cap. Past the window no whole occurrence
		// can be seen, so this is withheld rather than examined in part. The scan
		// over the megabyte happens first either way, and that is what is timed.
		name:         "a value at the bound against a megabyte that tiles it",
		user:         strings.Repeat("ab", maxScrubbableSecret/2),
		body:         strings.Repeat("ab", (1<<20)/2),
		wantWithheld: true,
	}} {
		t.Run(tc.name, func(t *testing.T) {
			start := time.Now()
			got := redactedForError(tc.body, tc.user, "https://app.test/cb")
			elapsed := time.Since(start)

			if withheld := got == withheldText; withheld != tc.wantWithheld {
				t.Errorf("redactedForError() withheld = %v, want %v", withheld, tc.wantWithheld)
			}
			if elapsed > 5*time.Second {
				t.Errorf("redactedForError() over a %d-byte body with a %d-byte value took %v",
					len(tc.body), len(tc.user), elapsed)
			}
			t.Logf("body %d bytes, value %d bytes: %v", len(tc.body), len(tc.user), elapsed)
		})
	}
}

// TestResponseBodyIsBoundedToo walks a self-regenerating body of growing size
// end to end, on the service's side of the cost ceiling rather than the caller's.
//
// decodeFully shortens by as little as five bytes a pass, and `\u005c` re-forms
// its own introducer, so an unbounded fixpoint decode is one pass per five bytes
// of a response this package reads a megabyte of. Before the pass bound: 20 KB
// took 233ms, 80 KB 3.5s, and a megabyte over nine minutes of CPU that no
// caller's deadline could interrupt, because nothing below doPost reads ctx.
//
// It attributes no kill of its own, and is kept for the end-to-end shape rather
// than for what it pins. Two bounds hold this budget and either alone is enough:
// remove the pass bound and the straddle window still keeps the decoded input
// small, remove the window and 64 passes still finish in tens of milliseconds. It
// fires only with both gone, which is a state no single edit reaches. The pass
// bound is pinned directly by TestDecodeFullyStopsAtItsPassBound and the window by
// TestWholeTextCheckReadsABoundedWindow and TestBodyInsideTheWindowIsStillShown.
func TestResponseBodyIsBoundedToo(t *testing.T) {
	for _, n := range []int{4000, 40000, 200000} {
		body := `\` + strings.Repeat("u005c", n)
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.WriteHeader(http.StatusForbidden)
			_, _ = io.WriteString(w, body)
		}))
		start := time.Now()
		_, err := newTestClient(t, srv).RetrieveCredential(t.Context(), Request{
			Resource: authProviderResource, UserID: "alice@example.test",
		})
		elapsed := time.Since(start)
		srv.Close()
		if err == nil {
			t.Fatal("RetrieveCredential() = nil error, want the 403")
		}
		if elapsed > 2*time.Second {
			t.Errorf("a %d-byte self-regenerating body took %v", len(body), elapsed)
		}
		t.Logf("body %d bytes: %v", len(body), elapsed)
	}
}

// TestWholeTextCheckReadsABoundedWindow is the window's UPPER bound, which no
// timing test can see: the pass bound already keeps an unbounded window fast
// enough. What the window buys is volume, and volume is what this measures.
// TestBodyInsideTheWindowIsStillShown is the lower bound, so a change to the
// constant in either direction turns something red.
//
// Only an occurrence straddling the cut can hide behind it, so the check needs
// the visible window plus room for one occurrence — not the megabyte doPost
// admits. Handed the whole body it ran redact over all of it and split the result
// on the marker: a one-character value against a megabyte measured 45 MB
// allocated against 4.6 MB with the window, for the same one boolean.
//
// This body is past the window, so today the response is withheld before the
// check runs at all. Growing the window past a megabyte puts the megabyte back
// inside it, the check runs over all of it, and the ceiling below catches that.
func TestWholeTextCheckReadsABoundedWindow(t *testing.T) {
	body := strings.Repeat("ab", 1<<19) // 1 MiB, every byte a match for "a"
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusForbidden)
		_, _ = io.WriteString(w, body)
	}))
	defer srv.Close()
	c := newTestClient(t, srv)

	r := testing.Benchmark(func(b *testing.B) {
		for b.Loop() {
			_, _ = c.RetrieveCredential(t.Context(), Request{Resource: authProviderResource, UserID: "a"})
		}
	})
	// Sits between the two measurements with room on both sides, so it fails on
	// the window being dropped rather than on a machine being slow.
	const budget = 20 << 20
	if got := r.AllocedBytesPerOp(); got > budget {
		t.Errorf("one call over a %d-byte body allocated %d bytes, want under %d — the whole-text "+
			"check is reading more than the window", len(body), got, budget)
	}
	t.Logf("%d bytes allocated per call, %d allocations", r.AllocedBytesPerOp(), r.AllocsPerOp())
}

// TestTruncationFlagDescribesTheTextReturned pins that the two come from one
// string.
//
// The flag arms the whole-text check, and the check is the only thing that sees
// an escaped occurrence the cut sliced in half. Taking the flag from the lowered
// copy and the text from the original let them disagree: Go folds U+0130 to a
// one-byte "i", so the lowered copy sat under the cap while the original crossed
// it, the check never ran, and the first candidate reported success before the
// second one could decode and scrub. That put 14 of an 18-byte address in the
// returned error in cleartext.
//
// The ASCII row is the control. Same byte length, no length-changing fold, so it
// leaked nothing even before the fix — which is what makes the fold the variable.
//
// Each row asserts presence as well as absence. Absence alone is satisfied by
// returning nothing at all: replacing showable's success return with ("", false)
// destroys every error body in the package and still passes an absence-only test.
func TestTruncationFlagDescribesTheTextReturned(t *testing.T) {
	const victim = "alice@example.test"
	for _, tc := range []struct{ name, padding, wantVisible string }{
		{"a fold that shortens the lowered copy past the cap", strings.Repeat("\u0130", 505), strings.Repeat("i", 100)},
		{"the same byte length in ASCII", strings.Repeat("x", 1010), strings.Repeat("x", 100)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			// The address one byte short, with its last byte escaped, so redact
			// cannot match it and only the decode can.
			body := tc.padding + victim[:len(victim)-1] + `\u0074`
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.WriteHeader(http.StatusForbidden)
				_, _ = io.WriteString(w, body)
			}))
			defer srv.Close()

			_, err := newTestClient(t, srv).RetrieveCredential(t.Context(), Request{
				Resource: authProviderResource, UserID: victim,
			})
			if err == nil {
				t.Fatal("RetrieveCredential() = nil error, want the 403")
			}
			for n := len(victim); n >= 5; n-- {
				if strings.Contains(err.Error(), victim[:n]) {
					t.Fatalf("%d of the address's %d bytes survive: %q", n, len(victim), victim[:n])
				}
			}
			if !strings.Contains(err.Error(), tc.wantVisible) {
				t.Errorf("the service's own text did not survive the scrub: %v", err)
			}
		})
	}
}

// TestScrubbableBoundFailsClosed pins which side of the bound gets scrubbed and
// which gets nothing, since the difference is a value one byte longer.
func TestScrubbableBoundFailsClosed(t *testing.T) {
	atBound := strings.Repeat("a", maxScrubbableSecret)
	// The body runs past the error-body cap either way, so the scrubbed answer
	// carries the ellipsis that says the rest was dropped.
	if got, want := redactedForError("denied for "+atBound, atBound), "denied for "+redactedMarker+"..."; got != want {
		t.Errorf("at the bound, redactedForError() = %q, want %q", got, want)
	}
	overBound := strings.Repeat("a", maxScrubbableSecret+1)
	if got := redactedForError("denied for "+overBound, overBound); got != withheldText {
		t.Errorf("past the bound, redactedForError() = %q, want it withheld", got)
	}
}

// TestDecodeErrorScrubsTheActingUser pins the fourth service-text path, which was
// the only one with no test.
//
// A decoder error quotes the token it choked on, so a service echoing the acting
// user where a different type is expected puts it in the message. The connector's
// error code is an int, so a numeric user id echoed there overflows it and the
// literal lands in "cannot unmarshal number …".
func TestDecodeErrorScrubsTheActingUser(t *testing.T) {
	const user = "10355512349999999999"
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(w, `{"done":true,"error":{"code":`+user+`,"message":"x"}}`)
	}))
	defer srv.Close()

	_, err := newTestClient(t, srv).RetrieveCredential(t.Context(), Request{
		Resource: "projects/p/locations/l/connectors/c",
		UserID:   user,
	})
	if err == nil {
		t.Fatal("RetrieveCredential() = nil error, want the decode to fail")
	}
	if strings.Contains(err.Error(), user) {
		t.Errorf("error carries the acting user: %v", err)
	}
	// Keyed on the secret, not a blanket drop.
	if !strings.Contains(err.Error(), "cannot unmarshal number") {
		t.Errorf("error lost the decoder's own wording: %v", err)
	}
}

// TestMalformedResponseIsMatchable pins the sentinel that replaced the %w a
// caller lost when the decode error stopped wrapping the decoder's own.
func TestMalformedResponseIsMatchable(t *testing.T) {
	srv, _ := sequenceServer(`{"success":` + strings.Repeat("9", 40) + `}`)
	defer srv.Close()

	_, err := newTestClient(t, srv).RetrieveCredential(t.Context(), Request{
		Resource: "projects/p/locations/l/authProviders/a",
		UserID:   "u",
	})
	if !errors.Is(err, ErrMalformedResponse) {
		t.Errorf("RetrieveCredential() error = %v, want it to match ErrMalformedResponse", err)
	}
	// The resource decoration must not break the match.
	if !strings.Contains(err.Error(), "resource") {
		t.Errorf("error = %v, want the resource named", err)
	}
}

// TestStraddlingOccurrenceIsWithheldNotShownInPart pins the disclosure half of the
// straddle window. The window bounds what the whole-text check reads, and a value
// spelled so its occurrence runs past that bound cannot be examined — so it is
// withheld rather than shown in part.
//
// The escaped spelling's length is the service's choice, not a multiple of the
// value: one character at nesting level k costs 2^(k-1)+5 bytes and takes k decode
// passes, so a cheap prefix can fill the visible kilobyte while a tail at level 12
// puts the end of the occurrence past any fixed window.
func TestStraddlingOccurrenceIsWithheldNotShownInPart(t *testing.T) {
	uri := "https://app.example.test/oauth/callback?state=" + strings.Repeat("k", 154)
	var body strings.Builder
	for i, r := range uri {
		depth := 1
		if i >= 170 {
			depth = 12
		}
		fmt.Fprintf(&body, "%su%04x", strings.Repeat(`\`, 1<<(depth-1)), r)
	}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusForbidden)
		_, _ = io.WriteString(w, body.String())
	}))
	defer srv.Close()

	_, err := newTestClient(t, srv).RetrieveCredential(t.Context(), Request{
		Resource: authProviderResource, UserID: "alice@example.test", ContinueURI: uri,
	})
	if err == nil {
		t.Fatal("RetrieveCredential() = nil error, want the 403")
	}
	got, _ := decodeFully(err.Error())
	for n := len(uri); n >= 5; n-- {
		if strings.Contains(got, uri[:n]) {
			t.Fatalf("%d of the ContinueURI's %d characters are recoverable: %q", n, len(uri), uri[:n])
		}
	}
}

// TestBodyInsideTheWindowIsStillShown is what fails when the window is too SMALL,
// which no test did: shrinking it only ever withholds more, and an allocation
// ceiling is satisfied by every smaller window.
//
// The size repeats the window's defining expression rather than naming
// maxStraddleWindow, so redefining that constant — the thing under test — moves
// the threshold while the fixture stays put, and shrinking it eightfold turns
// this red. Naming the constant would have shrunk the fixture along with it and
// the test could never have fired. The independence is from the constant, not
// from its terms: changing maxScrubbableSecret moves both together, and that one
// is pinned elsewhere.
func TestBodyInsideTheWindowIsStillShown(t *testing.T) {
	// Just inside what the window exists to cover, and past the visible cap so the
	// truncation path — the one the window guards — is the path under test.
	body := "denied: " + strings.Repeat("z", maxErrorBody+8*maxScrubbableSecret-100)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusForbidden)
		_, _ = io.WriteString(w, body)
	}))
	defer srv.Close()

	_, err := newTestClient(t, srv).RetrieveCredential(t.Context(), Request{
		Resource: authProviderResource, UserID: "alice@example.test",
	})
	if err == nil {
		t.Fatal("RetrieveCredential() = nil error, want the 403")
	}
	if !strings.Contains(err.Error(), "denied") {
		t.Errorf("a body of %d bytes carrying no identifier was withheld: %v", len(body), err)
	}
}

// TestMarkersAreBudgetedAgainstTheCap pins the size of what is returned, not the
// size of what was read. One marker per matched run bounds the marker count
// against the number of runs, and the SERVICE picks that number, so a body of
// alternating one-byte matches inflated the result several times past the cap the
// exported doc promises.
//
// The source is deliberately UNDER the cap. That is the case the output cap was
// added for and the only case that distinguishes it: over the cap the input bound
// has already cut, so a body that arrives too long is capped by either
// implementation and tells them apart not at all. Measured on a 1000-byte source,
// an output cap reachable only once the input bound has fired returns 5500 bytes.
//
// Both halves are asserted. Withholding the response also satisfies a size
// ceiling, so a bound with no accompanying "and it is still shown" is held by
// suppressing everything.
func TestMarkersAreBudgetedAgainstTheCap(t *testing.T) {
	body := strings.Repeat("u ", 500) // 1000 bytes in, 5500 bytes of markers out
	if len(body) > maxErrorBody {
		t.Fatalf("fixture is %d bytes, want it under the %d-byte input cap so the output cap is what fires",
			len(body), maxErrorBody)
	}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusForbidden)
		_, _ = io.WriteString(w, body)
	}))
	defer srv.Close()

	_, err := newTestClient(t, srv).RetrieveCredential(t.Context(), Request{
		Resource: authProviderResource, UserID: "u",
	})
	var apiErr *APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("RetrieveCredential() error = %v, want an *APIError", err)
	}
	// Presence first, and stated as content rather than as "not the withheld
	// sentence": an implementation that suppresses this body returns the sentence,
	// but one that empties it returns neither the sentence nor anything else, and
	// both satisfy a ceiling. What the cap must do is SHORTEN a real result.
	if !strings.Contains(apiErr.Body, redactedMarker) {
		t.Fatalf("Body carries no redacted run, want the body shown and cut rather than suppressed: %q", apiErr.Body)
	}
	if got, floor := len(apiErr.Body), maxErrorBody/2; got < floor {
		t.Errorf("Body = %d bytes, want at least %d — the cap should cut this body, not gut it", got, floor)
	}
	if got, want := len(apiErr.Body), maxErrorBody+len("..."); got > want {
		t.Errorf("Body = %d bytes, want it capped to %d", got, want)
	}
}
