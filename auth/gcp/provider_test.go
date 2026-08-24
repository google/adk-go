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
	"strings"
	"sync"
	"testing"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/auth"
	"google.golang.org/adk/v2/auth/gcp"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/session"
)

const testResource = "projects/p/locations/l/authProviders/ap"

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

	scopes := []string{"s1", "s2"}
	p := newProvider(t, srv, gcp.ProviderScheme{
		Name:        testResource,
		Scopes:      scopes,
		ContinueURI: "https://example.test/continue",
	})
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

// TestProviderErrorAttribution pins that a failed retrieval says which invocation
// and which resource failed — several providers can be wired into one process —
// without naming the user, whose id is commonly an email and whose error text
// reaches the model. Sentinels must stay matchable through the wrap.
// TestProviderCredentialConcurrent drives two users through one shared provider
// at the same time. The sequential case above pins the wire contract; this one
// pins that concurrency cannot cross the streams.
func TestProviderCredentialConcurrent(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			UserID string `json:"userId"`
		}
		_ = json.NewDecoder(r.Body).Decode(&body)
		_, _ = io.WriteString(w, `{"success":{"token":"tok-`+body.UserID+`","header":"Authorization: Bearer"}}`)
	}))
	defer srv.Close()

	p := newProvider(t, srv, gcp.ProviderScheme{Name: testResource})
	users := []string{"alice", "bob", "carol", "dave"}
	ctxs := make([]context.Context, len(users))
	for i, u := range users {
		ctxs[i] = adkContext(t, u)
	}

	var wg sync.WaitGroup
	got := make([]auth.Credential, len(users)*8)
	errs := make([]error, len(users)*8)
	for i := range got {
		wg.Add(1)
		go func() {
			defer wg.Done()
			got[i], errs[i] = p.Credential(ctxs[i%len(users)])
		}()
	}
	wg.Wait()

	for i, cred := range got {
		want := "tok-" + users[i%len(users)]
		if errs[i] != nil {
			t.Fatalf("Credential(%s) error = %v", users[i%len(users)], errs[i])
		}
		if bc, ok := cred.(auth.BearerCredential); !ok || bc.Token != want {
			t.Errorf("credential %d = %+v, want bearer %q", i, cred, want)
		}
	}
}

func TestProviderErrorAttribution(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusForbidden)
		_, _ = io.WriteString(w, `denied`)
	}))
	defer srv.Close()

	p := newProvider(t, srv, gcp.ProviderScheme{Name: testResource})
	ctx := adkContext(t, "alice@example.test")
	id, _ := agent.IdentityFromContext(ctx)
	_, err := p.Credential(ctx)
	if err == nil {
		t.Fatal("Credential() = nil error, want the service failure")
	}
	for _, want := range []string{id.SessionID, testResource} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("Credential() error = %v, want it to name %s", err, want)
		}
	}
	if strings.Contains(err.Error(), "alice@example.test") {
		t.Errorf("Credential() error = %v, want the user id kept out of it", err)
	}
	var apiErr *gcp.APIError
	if !errors.As(err, &apiErr) || apiErr.StatusCode != http.StatusForbidden {
		t.Errorf("Credential() error = %v, want a wrapped *gcp.APIError with status 403", err)
	}
}

// TestProviderNoActingUser covers both identity failures: the guard must reject
// before any service call, and the two cases must stay distinguishable.
func TestProviderNoActingUser(t *testing.T) {
	tests := []struct {
		name    string
		ctx     func(t *testing.T) context.Context
		wantMsg string
	}{
		{
			name:    "not an ADK context",
			ctx:     func(t *testing.T) context.Context { return t.Context() },
			wantMsg: "no ADK invocation identity",
		},
		{
			name:    "invocation without a user",
			ctx:     userlessADKContext,
			wantMsg: "empty UserID",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Fails the test if reached: no identity means no service call.
			srv := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
				t.Error("credentials service must not be called without an ADK identity")
			}))
			defer srv.Close()

			p := newProvider(t, srv, gcp.ProviderScheme{Name: testResource})
			_, err := p.Credential(tt.ctx(t))
			if !errors.Is(err, gcp.ErrNoActingUser) {
				t.Fatalf("Credential() error = %v, want gcp.ErrNoActingUser", err)
			}
			if !strings.Contains(err.Error(), tt.wantMsg) {
				t.Errorf("Credential() error = %v, want it to mention %q", err, tt.wantMsg)
			}
		})
	}
}

func TestNewProviderValidatesScheme(t *testing.T) {
	// Everything here is a wiring mistake, and each must fail at construction
	// rather than on every request from inside an http.RoundTripper.
	bad := []struct {
		name string
		cfg  gcp.ProviderConfig
	}{
		{"empty name", gcp.ProviderConfig{}},
		{"path traversal", cfgFor("projects/p/locations/l/authProviders/../../secret")},
		{"empty path segment", cfgFor("projects/p/locations/l/authProviders//ap")},
		{"trailing slash routes differently after normalization", cfgFor("projects/p/locations/l/connectors/c/")},
		{"not a resource name at all", cfgFor("Bearer")},
		{"unknown collection", cfgFor("projects/p/locations/l/authProvidrs/ap")},
		{"truncated", cfgFor("projects/p")},
		{"unconstructed client", gcp.ProviderConfig{Scheme: gcp.ProviderScheme{Name: testResource}, Client: &gcp.Client{}}},
	}
	for _, tt := range bad {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := gcp.NewProvider(t.Context(), tt.cfg); err == nil {
				t.Fatal("NewProvider() = nil error, want the config rejected")
			}
		})
	}

	good := []string{testResource, "projects/p/locations/l/connectors/c"}
	for _, name := range good {
		t.Run("accepts "+name, func(t *testing.T) {
			if _, err := gcp.NewProvider(t.Context(), cfgFor(name)); err != nil {
				t.Fatalf("NewProvider(%q) error = %v", name, err)
			}
		})
	}
}

func cfgFor(name string) gcp.ProviderConfig {
	return gcp.ProviderConfig{Scheme: gcp.ProviderScheme{Name: name}}
}

// newProvider builds a provider whose client targets srv.
func newProvider(t *testing.T, srv *httptest.Server, scheme gcp.ProviderScheme) auth.CredentialProvider {
	t.Helper()
	client, err := gcp.NewClient(t.Context(), &gcp.Config{
		HTTPClient:            srv.Client(),
		AgentIdentityEndpoint: srv.URL,
	})
	if err != nil {
		t.Fatalf("NewClient() error = %v", err)
	}
	p, err := gcp.NewProvider(t.Context(), gcp.ProviderConfig{Scheme: scheme, Client: client})
	if err != nil {
		t.Fatalf("NewProvider() error = %v", err)
	}
	return p
}

// adkContext returns an ADK invocation context (recoverable via
// agent.IdentityFromContext) for the given user.
func adkContext(t *testing.T, userID string) context.Context {
	t.Helper()
	svc := session.InMemoryService()
	resp, err := svc.Create(t.Context(), &session.CreateRequest{AppName: "app", UserID: userID})
	if err != nil {
		t.Fatalf("session Create() error = %v", err)
	}
	return icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{Session: resp.Session})
}

// userlessADKContext returns an invocation whose session carries no user — the
// shape session.InMemoryService refuses to create, but that a custom session
// service can produce.
func userlessADKContext(t *testing.T) context.Context {
	t.Helper()
	return icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{Session: userlessSession{}})
}

// userlessSession embeds a nil session.Session for the accessors the identity
// path never reaches.
type userlessSession struct{ session.Session }

func (userlessSession) ID() string      { return "sid" }
func (userlessSession) AppName() string { return "app" }
func (userlessSession) UserID() string  { return "" }
