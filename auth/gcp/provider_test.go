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
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"slices"
	"testing"

	"google.golang.org/adk/v2/auth"
	"google.golang.org/adk/v2/auth/gcp"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/session"
)

// TestProviderCredential drives two users through one shared provider: the
// provider is long-lived, so serving one user's credential to another is the
// failure that matters. It also pins scopes and continueUri on the wire.
func TestProviderCredential(t *testing.T) {
	var gotUsers []string
	var gotScopes []string
	var gotContinueURI string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			UserID      string   `json:"userId"`
			Scopes      []string `json:"scopes"`
			ContinueURI string   `json:"continueUri"`
		}
		_ = json.NewDecoder(r.Body).Decode(&body)
		gotUsers = append(gotUsers, body.UserID)
		gotScopes, gotContinueURI = body.Scopes, body.ContinueURI
		// Echo the caller back, so a credential served to the wrong user shows up.
		_, _ = io.WriteString(w, `{"success":{"token":"tok-`+body.UserID+`","header":"Authorization: Bearer"}}`)
	}))
	defer srv.Close()

	client, err := gcp.NewClient(t.Context(), &gcp.Config{
		HTTPClient:            srv.Client(),
		AgentIdentityEndpoint: srv.URL,
	})
	if err != nil {
		t.Fatalf("NewClient() error = %v", err)
	}
	scopes := []string{"s1", "s2"}
	p, err := gcp.NewProvider(
		gcp.Scheme{
			Name:        "projects/p/locations/l/authProviders/ap",
			Scopes:      scopes,
			ContinueURI: "https://example.test/continue",
		},
		&gcp.ProviderConfig{Client: client},
	)
	if err != nil {
		t.Fatalf("NewProvider() error = %v", err)
	}
	scopes[0] = "mutated" // the provider must have cloned this

	for _, user := range []string{"alice", "bob"} {
		cred, err := p.Credential(adkContext(t, user))
		if err != nil {
			t.Fatalf("Credential(%q) error = %v", user, err)
		}
		if bc, ok := cred.(auth.BearerCredential); !ok || bc.Token != "tok-"+user {
			t.Errorf("credential for %q = %+v, want bearer %q", user, cred, "tok-"+user)
		}
	}
	if !slices.Equal(gotUsers, []string{"alice", "bob"}) {
		t.Errorf("service saw users %q, want [alice bob]", gotUsers)
	}
	if !slices.Equal(gotScopes, []string{"s1", "s2"}) {
		t.Errorf("body scopes = %q, want [s1 s2] (caller's later mutation must not leak)", gotScopes)
	}
	if gotContinueURI != "https://example.test/continue" {
		t.Errorf("body continueUri = %q, want the scheme's", gotContinueURI)
	}
}

func TestProviderRequiresADKContext(t *testing.T) {
	// Fails the test if reached: the guard must reject before any service call.
	srv := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		t.Error("credentials service must not be called without an ADK identity")
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
		gcp.Scheme{Name: "projects/p/locations/l/authProviders/ap"},
		&gcp.ProviderConfig{Client: client},
	)
	if err != nil {
		t.Fatalf("NewProvider() error = %v", err)
	}
	_, err = p.Credential(t.Context())
	if !errors.Is(err, gcp.ErrNoActingUser) {
		t.Fatalf("Credential() error = %v, want gcp.ErrNoActingUser", err)
	}
}

func TestNewProviderRejectsUnconstructedClient(t *testing.T) {
	_, err := gcp.NewProvider(
		gcp.Scheme{Name: "projects/p/locations/l/authProviders/ap"},
		&gcp.ProviderConfig{Client: &gcp.Client{}},
	)
	if err == nil {
		t.Fatal("NewProvider() = nil error, want a zero Client rejected")
	}
}

func TestNewProviderValidatesScheme(t *testing.T) {
	if _, err := gcp.NewProvider(gcp.Scheme{}, nil); err == nil {
		t.Fatal("NewProvider() = nil error, want error for empty scheme Name")
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
