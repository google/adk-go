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

func TestRetrieveAgentIdentityBearer(t *testing.T) {
	srv, _ := sequenceServer(`{"success":{"token":"tok","header":"Authorization: Bearer"}}`)
	defer srv.Close()

	cred, err := newTestClient(t, srv).RetrieveCredential(t.Context(),
		Request{Resource: authProviderResource, UserID: "u"})
	if err != nil {
		t.Fatalf("RetrieveCredential() error = %v", err)
	}
	wantBearer(t, cred, "tok")
}

func TestRetrieveAgentIdentityCustomHeader(t *testing.T) {
	srv, _ := sequenceServer(`{"success":{"token":"KEY","header":"X-Goog-Api-Key"}}`)
	defer srv.Close()

	cred, err := newTestClient(t, srv).RetrieveCredential(t.Context(),
		Request{Resource: authProviderResource, UserID: "u"})
	if err != nil {
		t.Fatalf("RetrieveCredential() error = %v", err)
	}
	wantAPIKey(t, cred, "X-Goog-Api-Key", "KEY")
}

func TestRetrieveAgentIdentityConsentRequired(t *testing.T) {
	srv, _ := sequenceServer(`{"uriConsentRequired":{"authorizationUri":"https://consent","consentNonce":"n"}}`)
	defer srv.Close()

	_, err := newTestClient(t, srv).RetrieveCredential(t.Context(),
		Request{Resource: authProviderResource, UserID: "u"})
	var consent *auth.ConsentRequiredError
	if !errors.As(err, &consent) {
		t.Fatalf("error = %v, want *auth.ConsentRequiredError", err)
	}
	if consent.AuthURI != "https://consent" || consent.Nonce != "n" {
		t.Errorf("consent = %+v, want auth_uri/nonce set", consent)
	}
}

func TestRetrieveAgentIdentityConsentRejected(t *testing.T) {
	srv, _ := sequenceServer(`{"consentRejected":{}}`)
	defer srv.Close()

	_, err := newTestClient(t, srv).RetrieveCredential(t.Context(),
		Request{Resource: authProviderResource, UserID: "u"})
	if !errors.Is(err, ErrConsentRejected) {
		t.Fatalf("error = %v, want ErrConsentRejected", err)
	}
	var consent *auth.ConsentRequiredError
	if errors.As(err, &consent) {
		t.Fatalf("error = %v, want a plain rejection (not ConsentRequiredError)", err)
	}
}

func TestRetrieveAgentIdentityPollsPending(t *testing.T) {
	srv, calls := sequenceServer(
		`{"pending":{}}`,
		`{"success":{"token":"tok","header":"Authorization: Bearer"}}`,
	)
	defer srv.Close()

	cred, err := newTestClient(t, srv).RetrieveCredential(t.Context(),
		Request{Resource: authProviderResource, UserID: "u"})
	if err != nil {
		t.Fatalf("RetrieveCredential() error = %v", err)
	}
	wantBearer(t, cred, "tok")
	if got := atomic.LoadInt32(calls); got != 2 {
		t.Errorf("service calls = %d, want 2 (pending then success)", got)
	}
}

func TestRetrieveConnectorBearer(t *testing.T) {
	srv, _ := sequenceServer(`{"done":true,"response":{"@type":"x","token":"tok","header":"Authorization: Bearer"}}`)
	defer srv.Close()

	cred, err := newTestClient(t, srv).RetrieveCredential(t.Context(),
		Request{Resource: connectorResource, UserID: "u"})
	if err != nil {
		t.Fatalf("RetrieveCredential() error = %v", err)
	}
	wantBearer(t, cred, "tok")
}

func TestRetrieveConnectorPollsConsentPending(t *testing.T) {
	srv, calls := sequenceServer(
		`{"metadata":{"@type":"x","consentPending":{}}}`,
		`{"done":true,"response":{"token":"tok","header":"Authorization: Bearer"}}`,
	)
	defer srv.Close()

	cred, err := newTestClient(t, srv).RetrieveCredential(t.Context(),
		Request{Resource: connectorResource, UserID: "u"})
	if err != nil {
		t.Fatalf("RetrieveCredential() error = %v", err)
	}
	wantBearer(t, cred, "tok")
	if got := atomic.LoadInt32(calls); got != 2 {
		t.Errorf("service calls = %d, want 2 (pending then success)", got)
	}
}

func TestRetrieveConnectorConsentRequired(t *testing.T) {
	srv, _ := sequenceServer(`{"metadata":{"uriConsentRequired":{"authorizationUri":"https://c","consentNonce":"n"}}}`)
	defer srv.Close()

	_, err := newTestClient(t, srv).RetrieveCredential(t.Context(),
		Request{Resource: connectorResource, UserID: "u"})
	var consent *auth.ConsentRequiredError
	if !errors.As(err, &consent) {
		t.Fatalf("error = %v, want *auth.ConsentRequiredError", err)
	}
}

func TestRetrieveConnectorOperationError(t *testing.T) {
	srv, _ := sequenceServer(`{"error":{"message":"boom"}}`)
	defer srv.Close()

	_, err := newTestClient(t, srv).RetrieveCredential(t.Context(),
		Request{Resource: connectorResource, UserID: "u"})
	if err == nil || !strings.Contains(err.Error(), "boom") {
		t.Fatalf("error = %v, want it to contain %q", err, "boom")
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
	c := &Client{httpClient: http.DefaultClient}
	if _, err := c.RetrieveCredential(t.Context(), Request{UserID: "u"}); err == nil {
		t.Error("missing Resource: got nil error, want error")
	}
	if _, err := c.RetrieveCredential(t.Context(), Request{Resource: authProviderResource}); err == nil {
		t.Error("missing UserID: got nil error, want error")
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

// TestRetrieveConnectorDoneWithoutResponse verifies that a terminal (done)
// connector operation carrying no credential fails fast with an error, rather
// than being treated as pending and polled until the timeout.
func TestRetrieveConnectorDoneWithoutResponse(t *testing.T) {
	srv, _ := sequenceServer(`{"done":true}`)
	defer srv.Close()

	_, err := newTestClient(t, srv).RetrieveCredential(t.Context(),
		Request{Resource: connectorResource, UserID: "u"})
	if err == nil || !strings.Contains(err.Error(), "no credential") {
		t.Fatalf("error = %v, want it to mention %q", err, "no credential")
	}
	if errors.Is(err, ErrPollTimeout) {
		t.Fatalf("error = %v, want a done-without-credential error, not a poll timeout", err)
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

// wantAPIKey fails t unless cred is an auth.APIKeyCredential with name and value.
func wantAPIKey(t *testing.T, cred auth.Credential, name, value string) {
	t.Helper()
	k, ok := cred.(auth.APIKeyCredential)
	if !ok {
		t.Fatalf("credential = %#v, want auth.APIKeyCredential", cred)
	}
	if k.Name != name || k.Value != value {
		t.Fatalf("api key = {name:%q value:%q}, want {name:%q value:%q}", k.Name, k.Value, name, value)
	}
}
