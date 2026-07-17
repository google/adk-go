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
// via a fake server that replays the given response bodies in order. Each case
// sets exactly one expectation (a credential, a consent error, an errors.Is
// target, or an error substring).
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
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			srv, calls := sequenceServer(tc.bodies...)
			defer srv.Close()

			cred, err := newTestClient(t, srv).RetrieveCredential(t.Context(),
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
			var gotPath, gotMethod, gotUserID string
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				gotPath, gotMethod = r.URL.Path, r.Method
				var body struct {
					UserID string `json:"userId"`
				}
				_ = json.NewDecoder(r.Body).Decode(&body)
				gotUserID = body.UserID
				_, _ = io.WriteString(w, `{"done":true,"response":{"token":"t","header":"Authorization: Bearer"},"success":{"token":"t","header":"Authorization: Bearer"}}`)
			}))
			defer srv.Close()

			if _, err := newTestClient(t, srv).RetrieveCredential(t.Context(),
				Request{Resource: tc.resource, UserID: "user-1"}); err != nil {
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
	if err == nil || !strings.Contains(err.Error(), "500") {
		t.Fatalf("error = %v, want it to mention status 500", err)
	}
}

func TestRetrieveValidatesRequest(t *testing.T) {
	tests := []struct {
		name string
		req  Request
	}{
		{name: "missing resource", req: Request{UserID: "u"}},
		{name: "missing user id", req: Request{Resource: authProviderResource}},
	}
	c := &Client{httpClient: http.DefaultClient}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := c.RetrieveCredential(t.Context(), tc.req); err == nil {
				t.Errorf("RetrieveCredential(%+v) = nil error, want error", tc.req)
			}
		})
	}
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
		{name: "empty header", header: "", token: "t", wantErr: true},
		{name: "empty token", header: "Authorization: Bearer", token: "", wantErr: true},
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
func TestRetrieveContextCanceledWhilePending(t *testing.T) {
	srv, _ := sequenceServer(`{"pending":{}}`) // never resolves
	defer srv.Close()

	c := newTestClient(t, srv)
	c.initialBackoff = 50 * time.Millisecond // park in the poll wait, then cancel

	ctx, cancel := context.WithCancel(t.Context())
	time.AfterFunc(10*time.Millisecond, cancel)

	_, err := c.RetrieveCredential(ctx, Request{Resource: authProviderResource, UserID: "u"})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("RetrieveCredential() error = %v, want context.Canceled", err)
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
	c, err := NewClient(t.Context(),
		WithHTTPClient(srv.Client()),
		WithAgentIdentityEndpoint(srv.URL),
		WithConnectorEndpoint(srv.URL),
		WithPollTimeout(2*time.Second),
	)
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
