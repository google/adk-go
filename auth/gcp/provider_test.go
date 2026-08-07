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

package gcp_test

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"google.golang.org/adk/v2/auth"
	"google.golang.org/adk/v2/auth/gcp"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/session"
)

func TestProviderCredential(t *testing.T) {
	var gotUserID string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			UserID string `json:"userId"`
		}
		_ = json.NewDecoder(r.Body).Decode(&body)
		gotUserID = body.UserID
		_, _ = io.WriteString(w, `{"success":{"token":"tok","header":"Authorization: Bearer"}}`)
	}))
	defer srv.Close()

	client, err := gcp.NewClient(t.Context(), &gcp.Config{
		HTTPClient:            srv.Client(),
		AgentIdentityEndpoint: srv.URL,
	})
	if err != nil {
		t.Fatalf("NewClient() error = %v", err)
	}
	p, err := gcp.NewProvider(
		gcp.Scheme{Name: "projects/p/locations/l/authProviders/ap", Scopes: []string{"s1"}},
		&gcp.ProviderConfig{Client: client},
	)
	if err != nil {
		t.Fatalf("NewProvider() error = %v", err)
	}

	cred, err := p.Credential(adkContext(t, "user-1"))
	if err != nil {
		t.Fatalf("Credential() error = %v", err)
	}
	if bc, ok := cred.(auth.BearerCredential); !ok || bc.Token != "tok" {
		t.Fatalf("credential = %+v, want bearer token %q", cred, "tok")
	}
	if gotUserID != "user-1" {
		t.Errorf("service saw userId = %q, want %q (identity from agent.IdentityFromContext)", gotUserID, "user-1")
	}
}

func TestProviderRequiresADKContext(t *testing.T) {
	p, err := gcp.NewProvider(
		gcp.Scheme{Name: "projects/p/locations/l/authProviders/ap"},
		&gcp.ProviderConfig{Client: &gcp.Client{}},
	)
	if err != nil {
		t.Fatalf("NewProvider() error = %v", err)
	}
	// Plain context: no ADK identity to resolve, so no service call is made.
	if _, err := p.Credential(t.Context()); err == nil {
		t.Fatal("Credential() = nil error, want error for missing ADK context")
	}
}

func TestNewProviderValidatesScheme(t *testing.T) {
	if _, err := gcp.NewProvider(gcp.Scheme{}, nil); err == nil {
		t.Fatal("NewProvider() = nil error, want error for empty scheme Name")
	}
}

func TestProviderCachesCredential(t *testing.T) {
	var calls int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&calls, 1)
		_, _ = io.WriteString(w, `{"success":{"token":"tok","header":"Authorization: Bearer","expireTime":"2999-01-01T00:00:00Z"}}`)
	}))
	defer srv.Close()

	client, err := gcp.NewClient(t.Context(), &gcp.Config{
		HTTPClient:            srv.Client(),
		AgentIdentityEndpoint: srv.URL,
	})
	if err != nil {
		t.Fatalf("NewClient() error = %v", err)
	}
	// Default (in-memory) store; two resolves for the same app+user+resource.
	p, err := gcp.NewProvider(gcp.Scheme{Name: "projects/p/locations/l/authProviders/ap"}, &gcp.ProviderConfig{Client: client})
	if err != nil {
		t.Fatalf("NewProvider() error = %v", err)
	}

	for i := range 2 {
		if _, err := p.Credential(adkContext(t, "user-1")); err != nil {
			t.Fatalf("call %d: Credential() error = %v", i, err)
		}
	}
	if got := atomic.LoadInt32(&calls); got != 1 {
		t.Errorf("service calls = %d, want 1 (second resolve should hit the cache)", got)
	}
}

func TestProviderSkipsCacheWithoutExpiry(t *testing.T) {
	var calls int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&calls, 1)
		// No expireTime: lifetime unknown, so the provider must not cache.
		_, _ = io.WriteString(w, `{"success":{"token":"tok","header":"Authorization: Bearer"}}`)
	}))
	defer srv.Close()

	client, err := gcp.NewClient(t.Context(), &gcp.Config{
		HTTPClient:            srv.Client(),
		AgentIdentityEndpoint: srv.URL,
	})
	if err != nil {
		t.Fatalf("NewClient() error = %v", err)
	}
	p, err := gcp.NewProvider(gcp.Scheme{Name: "projects/p/locations/l/authProviders/ap"}, &gcp.ProviderConfig{Client: client})
	if err != nil {
		t.Fatalf("NewProvider() error = %v", err)
	}

	for i := range 2 {
		if _, err := p.Credential(adkContext(t, "user-1")); err != nil {
			t.Fatalf("call %d: Credential() error = %v", i, err)
		}
	}
	if got := atomic.LoadInt32(&calls); got != 2 {
		t.Errorf("service calls = %d, want 2 (unknown expiry must not be cached)", got)
	}
}

// adkContext returns an ADK invocation context (recoverable via agent.IdentityFromContext)
// for the given user.
func adkContext(t *testing.T, userID string) context.Context {
	t.Helper()
	svc := session.InMemoryService()
	resp, err := svc.Create(t.Context(), &session.CreateRequest{AppName: "app", UserID: userID})
	if err != nil {
		t.Fatalf("session Create() error = %v", err)
	}
	return icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{Session: resp.Session})
}
