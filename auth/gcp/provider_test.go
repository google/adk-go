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
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"slices"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

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

// TestProviderCredentialConcurrent drives two users through one shared provider
// at the same time. The sequential case above pins the wire contract; this one
// pins that concurrency cannot cross the streams.
func TestProviderCredentialConcurrent(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			UserID string `json:"userId"`
		}
		_ = json.NewDecoder(r.Body).Decode(&body)
		// With an expiry, so these goroutines drive the cache write and the cache
		// read, not only the retrieval. Without one the provider declines to cache
		// and store.Set is never reached under concurrency at all.
		_, _ = io.WriteString(w, `{"success":{"token":"tok-`+body.UserID+`","header":"Authorization: Bearer","expireTime":"2999-01-01T00:00:00Z"}}`)
	}))
	defer srv.Close()

	// Several scopes, so the clone-before-sort in the slot derivation is exercised
	// concurrently: sorting p.scheme.Scopes in place instead would write a shared
	// backing array from every one of these goroutines, and a nil Scopes makes
	// that mutant a no-op.
	p := newProvider(t, srv, gcp.ProviderScheme{Name: testResource, Scopes: []string{"b", "a", "c"}})
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

// TestProviderErrorAttribution pins that a failed retrieval says which resource
// failed — several providers can be wired into one process — and names no
// caller-supplied id, since this text is fed to the model and persisted in the
// session. Sentinels must stay matchable through the wrap.
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
	if !strings.Contains(err.Error(), testResource) {
		t.Errorf("Credential() error = %v, want it to name the resource", err)
	}
	for _, unwanted := range []string{"alice@example.test", id.SessionID} {
		if strings.Contains(err.Error(), unwanted) {
			t.Errorf("Credential() error = %v, want the caller-supplied %q kept out of it", err, unwanted)
		}
	}
	var apiErr *gcp.APIError
	if !errors.As(err, &apiErr) || apiErr.StatusCode != http.StatusForbidden {
		t.Errorf("Credential() error = %v, want a wrapped *gcp.APIError with status 403", err)
	}
}

// TestProviderNoActingUser covers both identity failures: the guard must reject
// before any service call, the two cases must stay distinguishable, and neither
// message may carry a caller-supplied id — this text reaches the model and is
// persisted in the session.
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
			wantMsg: "carries no user",
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
			for _, unwanted := range []string{"app", "sid"} {
				if strings.Contains(err.Error(), unwanted) {
					t.Errorf("Credential() error = %v, want the caller-supplied %q kept out of it", err, unwanted)
				}
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

	good := []string{
		testResource,
		"projects/p/locations/l/connectors/c",
		// Domain-scoped project ids carry a colon.
		"projects/example.com:my-project/locations/l/authProviders/ap",
	}
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

func TestProviderRefreshForcesNewToken(t *testing.T) {
	// Both credential services mint a fresh token when the caller passes the
	// prior (rejected) token as forceRefreshToken. Refresh must read the cached
	// token and send it on both routes; the connector's Operation-wrapped
	// response and Agent Identity's inline response only differ in the envelope.
	tests := []struct {
		name     string
		resource string
		endpoint func(cfg *gcp.Config, url string)
		// success builds the service-specific success envelope carrying tok.
		success func(tok string) string
		// forced reports whether a request body asked for a forced refresh. The
		// two services spell that differently, which is the point of the split.
		forced func(body forceRefreshFields) bool
	}{
		{
			name:     "agent identity",
			resource: "projects/p/locations/l/authProviders/ap",
			endpoint: func(cfg *gcp.Config, url string) { cfg.AgentIdentityEndpoint = url },
			success: func(tok string) string {
				return fmt.Sprintf(`{"success":{"token":%q,"header":"Authorization: Bearer","expireTime":"2999-01-01T00:00:00Z"}}`, tok)
			},
			forced: func(b forceRefreshFields) bool { return b.ForceRefreshToken != "" },
		},
		{
			name:     "iam connector",
			resource: "projects/p/locations/l/connectors/c",
			endpoint: func(cfg *gcp.Config, url string) { cfg.ConnectorEndpoint = url },
			success: func(tok string) string {
				return fmt.Sprintf(`{"done":true,"response":{"token":%q,"header":"Authorization: Bearer","expireTime":"2999-01-01T00:00:00Z"}}`, tok)
			},
			forced: func(b forceRefreshFields) bool { return b.ForceRefresh },
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var mu sync.Mutex
			var lastBody forceRefreshFields
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				var body forceRefreshFields
				_ = json.NewDecoder(r.Body).Decode(&body)
				mu.Lock()
				lastBody = body
				mu.Unlock()
				tok := "tok1"
				if tc.forced(body) {
					tok = "tok2" // a forced refresh mints a new token
				}
				// Include an expiry so the credential is cached; Refresh reads the
				// prior (cached) token from the store to send as forceRefreshToken.
				_, _ = io.WriteString(w, tc.success(tok))
			}))
			defer srv.Close()

			cfg := &gcp.Config{HTTPClient: srv.Client()}
			tc.endpoint(cfg, srv.URL)
			client, err := gcp.NewClient(t.Context(), cfg)
			if err != nil {
				t.Fatalf("NewClient() error = %v", err)
			}
			p, err := gcp.NewProvider(t.Context(), gcp.ProviderConfig{Scheme: gcp.ProviderScheme{Name: tc.resource}, Client: client})
			if err != nil {
				t.Fatalf("NewProvider() error = %v", err)
			}
			rp, ok := p.(auth.RefreshingProvider)
			if !ok {
				t.Fatal("gcp provider does not implement auth.RefreshingProvider")
			}

			ctx := adkContext(t, "user-1")

			// Prime the cache with the initial token.
			first, err := p.Credential(ctx)
			if err != nil {
				t.Fatalf("Credential() error = %v", err)
			}
			if bc, ok := first.(auth.BearerCredential); !ok || bc.Token != "tok1" {
				t.Fatalf("initial credential = %+v, want tok1", first)
			}

			// Refresh sends the rejected token back and returns a new one.
			cred, err := rp.Refresh(ctx, first)
			if err != nil {
				t.Fatalf("Refresh() error = %v", err)
			}
			mu.Lock()
			got := lastBody
			mu.Unlock()
			if !tc.forced(got) {
				t.Errorf("request body %+v did not ask for a forced refresh", got)
			}
			// Agent Identity is told which token to replace; the connector is not
			// given the field, so it must not be sent one.
			wantToken := ""
			if strings.Contains(tc.resource, "/authProviders/") {
				wantToken = "tok1"
			}
			if got.ForceRefreshToken != wantToken {
				t.Errorf("forceRefreshToken = %q, want %q", got.ForceRefreshToken, wantToken)
			}
			if bc, ok := cred.(auth.BearerCredential); !ok || bc.Token != "tok2" {
				t.Errorf("refreshed credential = %+v, want tok2", cred)
			}

			// The refreshed credential replaces the cached one.
			if cred, err := p.Credential(ctx); err != nil {
				t.Fatalf("Credential() after refresh error = %v", err)
			} else if bc, ok := cred.(auth.BearerCredential); !ok || bc.Token != "tok2" {
				t.Errorf("cached credential after refresh = %+v, want tok2", cred)
			}
		})
	}
}

// adkContext returns an ADK invocation context (recoverable via
// agent.IdentityFromContext) for the given user, under the app name "app".
func adkContext(t *testing.T, userID string) context.Context {
	t.Helper()
	return adkContextIn(t, "app", userID)
}

// adkContextIn is adkContext with the app name spelled out, for the tests that
// vary it.
func adkContextIn(t *testing.T, appName, userID string) context.Context {
	t.Helper()
	svc := session.InMemoryService()
	resp, err := svc.Create(t.Context(), &session.CreateRequest{AppName: appName, UserID: userID})
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

// echoServer answers every retrieval with a token naming what the request asked
// for, so a credential served from the wrong cache entry is visible in the token
// rather than only in a call count. It records each request.
type echoServer struct {
	*httptest.Server
	mu       sync.Mutex
	requests []echoRequest
}

type echoRequest struct {
	path        string // carries the resource name, which is not in the body
	userID      string
	scopes      []string
	continueURI string
	caller      string // X-Caller-Identity, set by the identity transports below
}

// newEchoServer starts an echoServer whose tokens expire at expireTime (RFC
// 3339; empty for a response that reports no expiry).
func newEchoServer(t *testing.T, expireTime string) *echoServer {
	t.Helper()
	e := &echoServer{}
	e.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			UserID      string   `json:"userId"`
			Scopes      []string `json:"scopes"`
			ContinueURI string   `json:"continueUri"`
		}
		_ = json.NewDecoder(r.Body).Decode(&body)
		req := echoRequest{
			path:        r.URL.Path,
			userID:      body.UserID,
			scopes:      body.Scopes,
			continueURI: body.ContinueURI,
			caller:      r.Header.Get("X-Caller-Identity"),
		}
		e.mu.Lock()
		e.requests = append(e.requests, req)
		e.mu.Unlock()

		resp := map[string]any{"token": req.token(), "header": "Authorization: Bearer"}
		if expireTime != "" {
			resp["expireTime"] = expireTime
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"success": resp})
	}))
	t.Cleanup(e.Close)
	return e
}

// token is the token this request is answered with: everything the service was
// told, so any two requests that should not share a cache entry get different
// tokens.
//
// Length-prefixed for the same reason the cache slot is. Joining on a delimiter
// made the token for {scopes:["x"], continueURI:"y|z"} byte-identical to the one
// for {scopes:["x|y"], continueURI:"z"} — so the two cases written to catch
// exactly that collision in the cache key could not see it in the token, and
// rested on the call count alone.
func (r echoRequest) token() string {
	fields := []string{"tok", r.path, r.caller, r.userID, strconv.Itoa(len(r.scopes))}
	fields = append(fields, r.scopes...)
	fields = append(fields, r.continueURI)
	var b strings.Builder
	for _, f := range fields {
		b.WriteString(strconv.Itoa(len(f)))
		b.WriteByte(':')
		b.WriteString(f)
	}
	return b.String()
}

func (e *echoServer) calls() int {
	e.mu.Lock()
	defer e.mu.Unlock()
	return len(e.requests)
}

// identityTransport stamps a caller identity on every request, standing in for
// the distinct service-account credentials a caller-supplied HTTPClient carries.
type identityTransport struct {
	base http.RoundTripper
	who  string
}

func (t identityTransport) RoundTrip(r *http.Request) (*http.Response, error) {
	r = r.Clone(r.Context())
	r.Header.Set("X-Caller-Identity", t.who)
	return t.base.RoundTrip(r)
}

// newSharingProvider builds a provider against e that shares store with every
// other provider built the same way. who names the caller identity its client
// authenticates as; two providers given the same who share one *Client.
func newSharingProvider(t *testing.T, e *echoServer, store auth.CredentialStore, clients map[string]*gcp.Client, who string, scheme gcp.ProviderScheme) auth.CredentialProvider {
	t.Helper()
	client, ok := clients[who]
	if !ok {
		var err error
		client, err = gcp.NewClient(t.Context(), &gcp.Config{
			HTTPClient:            &http.Client{Transport: identityTransport{base: e.Client().Transport, who: who}},
			AgentIdentityEndpoint: e.URL,
		})
		if err != nil {
			t.Fatalf("NewClient() error = %v", err)
		}
		clients[who] = client
	}
	p, err := gcp.NewProvider(t.Context(), gcp.ProviderConfig{Scheme: scheme, Client: client, Store: store})
	if err != nil {
		t.Fatalf("NewProvider() error = %v", err)
	}
	return p
}

func TestProviderCachesCredential(t *testing.T) {
	e := newEchoServer(t, "2999-01-01T00:00:00Z")

	// Default (in-memory) store; two resolves for the same app+user+resource.
	p := newProvider(t, e.Server, gcp.ProviderScheme{Name: testResource})
	for i := range 2 {
		if _, err := p.Credential(adkContext(t, "user-1")); err != nil {
			t.Fatalf("call %d: Credential() error = %v", i, err)
		}
	}
	if got := e.calls(); got != 1 {
		t.Errorf("service calls = %d, want 1 (second resolve should hit the cache)", got)
	}
}

// TestProviderCacheDimensions is the guard on the cache's headline property: a
// credential must never be served to a request that would not have been minted
// the same one. Each case runs two resolves that differ in exactly one thing,
// through one shared store, and requires that both reach the service and each
// gets its own token. Collapsing any single dimension of the cache key leaves
// every other test in the package green.
func TestProviderCacheDimensions(t *testing.T) {
	const res2 = "projects/p/locations/l/authProviders/other"
	// resolve names one call: which caller identity mints, for which end user,
	// under which app, for which scheme.
	type resolve struct {
		who, app, user string
		scheme         gcp.ProviderScheme
	}
	base := resolve{who: "sa-alpha", app: "app", user: "user-1", scheme: gcp.ProviderScheme{Name: testResource, Scopes: []string{"drive"}}}
	with := func(f func(*resolve)) resolve {
		r := base
		r.scheme.Scopes = slices.Clone(base.scheme.Scopes)
		f(&r)
		return r
	}

	tests := []struct {
		name          string
		first, second resolve
	}{
		{name: "end user", second: with(func(r *resolve) { r.user = "user-2" })},
		{name: "app", second: with(func(r *resolve) { r.app = "other-app" })},
		{name: "caller identity", second: with(func(r *resolve) { r.who = "sa-beta" })},
		{name: "resource", second: with(func(r *resolve) { r.scheme.Name = res2 })},
		{name: "scopes", second: with(func(r *resolve) { r.scheme.Scopes = []string{"drive.readonly"} })},
		{name: "continue URI", second: with(func(r *resolve) { r.scheme.ContinueURI = "https://example.com/finish" })},
		// The two pairs below are what an encoding that joins the components on a
		// delimiter collides, since neither "," nor "|" is escaped or barred from a
		// scope or a URI: both members slot alike under
		// name + "|" + join(scopes, ",") + "|" + continueURI.
		{
			name:   "one scope holding the scope separator",
			first:  with(func(r *resolve) { r.scheme.Scopes = []string{"a,b"} }),
			second: with(func(r *resolve) { r.scheme.Scopes = []string{"a", "b"} }),
		},
		{
			name:   "field separator shifted between scope and continue URI",
			first:  with(func(r *resolve) { r.scheme.Scopes, r.scheme.ContinueURI = []string{"x"}, "y|z" }),
			second: with(func(r *resolve) { r.scheme.Scopes, r.scheme.ContinueURI = []string{"x|y"}, "z" }),
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			first := tc.first
			if first.who == "" {
				first = base
			}
			// The pair is symmetric, so also run it reversed: broad-then-narrow hands
			// out more authority than was asked for, narrow-then-broad only produces a
			// confusing failure, and one pass checks only one of the two orders.
			for _, order := range [][2]resolve{{first, tc.second}, {tc.second, first}} {
				e := newEchoServer(t, "2999-01-01T00:00:00Z")
				store := auth.NewInMemoryCredentialStore()
				clients := map[string]*gcp.Client{}
				for _, r := range order {
					p := newSharingProvider(t, e, store, clients, r.who, r.scheme)
					cred, err := p.Credential(adkContextIn(t, r.app, r.user))
					if err != nil {
						t.Fatalf("Credential() error = %v", err)
					}
					want := echoRequest{
						path:        "/v1/" + r.scheme.Name + "/credentials:retrieve",
						userID:      r.user,
						scopes:      r.scheme.Scopes,
						continueURI: r.scheme.ContinueURI,
						caller:      r.who,
					}.token()
					if bc, ok := cred.(auth.BearerCredential); !ok || bc.Token != want {
						t.Errorf("%+v was served %+v, want bearer %q", r, cred, want)
					}
				}
				if got := e.calls(); got != 2 {
					t.Errorf("service calls = %d, want 2 (the second resolve must not reuse the first entry)", got)
				}
			}
		})
	}
}

// The same resolve twice through a shared store is one call, so the isolation
// above is not just the cache never hitting at all.
func TestProviderCacheHitsAcrossProviders(t *testing.T) {
	e := newEchoServer(t, "2999-01-01T00:00:00Z")
	store := auth.NewInMemoryCredentialStore()
	clients := map[string]*gcp.Client{}
	scheme := gcp.ProviderScheme{Name: testResource, Scopes: []string{"drive"}}
	for range 2 {
		p := newSharingProvider(t, e, store, clients, "sa-alpha", scheme)
		if _, err := p.Credential(adkContext(t, "user-1")); err != nil {
			t.Fatalf("Credential() error = %v", err)
		}
	}
	if got := e.calls(); got != 1 {
		t.Errorf("service calls = %d, want 1 (a second provider with the same client and scheme should hit the entry)", got)
	}
}

// Scope order is the caller's, not a cache dimension.
func TestProviderCacheIgnoresScopeOrder(t *testing.T) {
	e := newEchoServer(t, "2999-01-01T00:00:00Z")
	store := auth.NewInMemoryCredentialStore()
	clients := map[string]*gcp.Client{}
	for _, scopes := range [][]string{{"a", "b"}, {"b", "a"}} {
		p := newSharingProvider(t, e, store, clients, "sa-alpha", gcp.ProviderScheme{Name: testResource, Scopes: scopes})
		if _, err := p.Credential(adkContext(t, "user-1")); err != nil {
			t.Fatalf("Credential() error = %v", err)
		}
	}
	if got := e.calls(); got != 1 {
		t.Errorf("service calls = %d, want 1 (reordered scopes are the same credential)", got)
	}
}

// recordingStore reports every call and what it was handed, and can fail either
// direction.
type recordingStore struct {
	inner  auth.CredentialStore
	getErr error
	setErr error

	// CredentialStore is documented safe for concurrent use, so the double is too
	// — otherwise the first test to drive one from two goroutines reports a race
	// in the harness rather than a finding about the code.
	mu      sync.Mutex
	sets    int
	gets    int
	nilOnce bool // return a hit carrying no credential on the first Get

	lastKey     auth.CredentialKey
	lastExpires time.Time
}

func (s *recordingStore) Get(ctx context.Context, key auth.CredentialKey) (auth.Credential, bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.gets++
	if s.getErr != nil {
		return nil, false, s.getErr
	}
	if s.nilOnce {
		s.nilOnce = false
		return nil, true, nil
	}
	return s.inner.Get(ctx, key)
}

func (s *recordingStore) Set(ctx context.Context, key auth.CredentialKey, cred auth.Credential, expiresAt time.Time) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.sets++
	s.lastKey, s.lastExpires = key, expiresAt
	if s.setErr != nil {
		return s.setErr
	}
	return s.inner.Set(ctx, key, cred, expiresAt)
}

func (s *recordingStore) Delete(ctx context.Context, key auth.CredentialKey) error {
	return s.inner.Delete(ctx, key)
}

// newStoreProvider builds a provider against e backed by a recordingStore.
func newStoreProvider(t *testing.T, e *echoServer, store auth.CredentialStore) auth.CredentialProvider {
	t.Helper()
	client, err := gcp.NewClient(t.Context(), &gcp.Config{HTTPClient: e.Client(), AgentIdentityEndpoint: e.URL})
	if err != nil {
		t.Fatalf("NewClient() error = %v", err)
	}
	p, err := gcp.NewProvider(t.Context(), gcp.ProviderConfig{
		Scheme: gcp.ProviderScheme{Name: testResource},
		Client: client,
		Store:  store,
	})
	if err != nil {
		t.Fatalf("NewProvider() error = %v", err)
	}
	return p
}

// TestProviderStoreDegradesRatherThanFails pins that a store which misbehaves
// costs a round trip and nothing else. Each case breaks the store a different
// way and requires a usable credential back.
func TestProviderStoreDegradesRatherThanFails(t *testing.T) {
	tests := []struct {
		name    string
		breakIt func(*recordingStore)
	}{
		{"the read fails", func(s *recordingStore) { s.getErr = errors.New("backend unreachable") }},
		{"the write fails", func(s *recordingStore) { s.setErr = errors.New("disk on fire") }},
		{"a hit carries no credential", func(s *recordingStore) { s.nilOnce = true }},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			e := newEchoServer(t, "2999-01-01T00:00:00Z")
			store := &recordingStore{inner: auth.NewInMemoryCredentialStore()}
			tc.breakIt(store)
			p := newStoreProvider(t, e, store)

			cred, err := p.Credential(adkContext(t, "user-1"))
			if err != nil {
				t.Fatalf("Credential() error = %v; a broken store must not fail auth", err)
			}
			if cred == nil {
				t.Fatal("Credential() = nil credential")
			}
			if store.gets != 1 {
				t.Errorf("store gets = %d, want 1 (the configured store must be consulted)", store.gets)
			}
		})
	}
}

// The configured store is written to, under a key whose app and user land in
// their own fields.
func TestProviderStoreWritesTheKeyItRead(t *testing.T) {
	e := newEchoServer(t, "2999-01-01T00:00:00Z")
	store := &recordingStore{inner: auth.NewInMemoryCredentialStore()}
	p := newStoreProvider(t, e, store)

	if _, err := p.Credential(adkContext(t, "user-1")); err != nil {
		t.Fatalf("Credential() error = %v", err)
	}
	if store.gets != 1 || store.sets != 1 {
		t.Errorf("store gets/sets = %d/%d, want 1/1 (the configured store must be used)", store.gets, store.sets)
	}
	// The app and the user must land in their own fields, not merely in some
	// distinct pair: a store that buckets by app and then by user — the shape
	// auth.CredentialStore is documented for — files them separately, so swapping
	// the two would file alice's credential under an app named "alice".
	if store.lastKey.AppName != "app" || store.lastKey.UserID != "user-1" {
		t.Errorf("store key = %+v, want AppName \"app\" and UserID \"user-1\"", store.lastKey)
	}
	if store.lastKey.Key == "" {
		t.Error("store key has an empty slot")
	}
}

// The key the provider writes under is the one Client.CacheKey names, so a
// caller told to invalidate a credential with it can actually reach the entry.
func TestClientCacheKeyMatchesWhatTheProviderWrote(t *testing.T) {
	e := newEchoServer(t, "2999-01-01T00:00:00Z")
	client, err := gcp.NewClient(t.Context(), &gcp.Config{HTTPClient: e.Client(), AgentIdentityEndpoint: e.URL})
	if err != nil {
		t.Fatalf("NewClient() error = %v", err)
	}
	scheme := gcp.ProviderScheme{Name: testResource, Scopes: []string{"drive"}}
	store := auth.NewInMemoryCredentialStore()
	p, err := gcp.NewProvider(t.Context(), gcp.ProviderConfig{Scheme: scheme, Client: client, Store: store})
	if err != nil {
		t.Fatalf("NewProvider() error = %v", err)
	}
	if _, err := p.Credential(adkContext(t, "user-1")); err != nil {
		t.Fatalf("Credential() error = %v", err)
	}

	key := client.CacheKey(scheme, "app", "user-1")
	if _, ok, _ := store.Get(t.Context(), key); !ok {
		t.Fatal("Client.CacheKey() names no cached entry, so a caller cannot invalidate one")
	}
	if err := store.Delete(t.Context(), key); err != nil {
		t.Fatalf("Delete() error = %v", err)
	}
	if _, err := p.Credential(adkContext(t, "user-1")); err != nil {
		t.Fatalf("Credential() after Delete error = %v", err)
	}
	if got := e.calls(); got != 2 {
		t.Errorf("service calls = %d, want 2 (Delete must actually invalidate)", got)
	}
}

// Two Clients are two cache dimensions even when built from one config: this
// package cannot see what identity a Client authenticates as, so it never
// assumes two of them agree.
func TestProviderCacheSeparatesClientInstances(t *testing.T) {
	e := newEchoServer(t, "2999-01-01T00:00:00Z")
	store := auth.NewInMemoryCredentialStore()
	scheme := gcp.ProviderScheme{Name: testResource}
	for range 2 {
		client, err := gcp.NewClient(t.Context(), &gcp.Config{HTTPClient: e.Client(), AgentIdentityEndpoint: e.URL})
		if err != nil {
			t.Fatalf("NewClient() error = %v", err)
		}
		p, err := gcp.NewProvider(t.Context(), gcp.ProviderConfig{Scheme: scheme, Client: client, Store: store})
		if err != nil {
			t.Fatalf("NewProvider() error = %v", err)
		}
		if _, err := p.Credential(adkContext(t, "user-1")); err != nil {
			t.Fatalf("Credential() error = %v", err)
		}
	}
	if got := e.calls(); got != 2 {
		t.Errorf("service calls = %d, want 2 (a second Client must not read the first one's entries)", got)
	}
}

// TestProviderCachedExpiry pins what lifetime the provider is willing to cache
// for. The store is asked what it was handed rather than inferring from a call
// count, so each case fails loudly if the provider stops calling Set at all.
//
// Expiries are relative to the wall clock, because that is the clock the whole
// path now uses: a credential dies when the issuer says it does, not when a
// simulated clock says so.
func TestProviderCachedExpiry(t *testing.T) {
	const noCache = time.Duration(0)
	tests := []struct {
		name string
		// left is how much life the service reports, from now. The zero value
		// means the service reports no expiry at all.
		left string
		// want is the lifetime the store should be handed, or noCache. When clamped
		// is set it is measured from the provider's clock; otherwise the service's
		// own expiry must be passed through untouched.
		want    time.Duration
		clamped bool
	}{
		{name: "honored as reported", left: "5m", want: 5 * time.Minute},
		// A short-lived credential is still worth caching. With the boundary cases
		// below this pins auth.ExpirySkew under a minute: widen it and this stops
		// being cached at all.
		{name: "a minute of life left", left: "1m", want: time.Minute},
		// Twice the margin, so widening the floor to any multiple of it stops
		// caching this. The floor is what keeps a guaranteed-dead entry out; it is
		// not a licence to refuse short-lived credentials.
		{name: "twice the store's margin", left: "20s", want: 20 * time.Second},
		{name: "clamped to the cap", left: "8760h", want: maxCachedLifetimeForTest, clamped: true},
		// At or inside the margin the store applies, the entry would be written and
		// then refused on the very next read: a guaranteed-dead write.
		{name: "less left than the store's margin", left: "5s", want: noCache},
		{name: "exactly the store's margin", left: "10s", want: noCache},
		{name: "already past", left: "-1h", want: noCache},
		{name: "absent", left: "", want: noCache},
		{name: "unparseable", left: "garbage", want: noCache},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			issued := time.Now()
			expireTime := tc.left
			if d, err := time.ParseDuration(tc.left); err == nil {
				expireTime = issued.Add(d).Format(time.RFC3339Nano)
			}
			e := newEchoServer(t, expireTime)
			store := &recordingStore{inner: auth.NewInMemoryCredentialStore()}
			p := newStoreProvider(t, e, store)

			before := time.Now()
			if _, err := p.Credential(adkContext(t, "user-1")); err != nil {
				t.Fatalf("Credential() error = %v", err)
			}
			after := time.Now()

			if tc.want == noCache {
				if store.sets != 0 {
					t.Errorf("store Set called with expiry %s, want no cache write for expireTime %q", store.lastExpires, expireTime)
				}
				return
			}
			if store.sets != 1 {
				t.Fatalf("store Set calls = %d, want 1", store.sets)
			}
			if !tc.clamped {
				// Passed through untouched, so this is exact.
				want, err := time.Parse(time.RFC3339Nano, expireTime)
				if err != nil {
					t.Fatalf("parse %q: %v", expireTime, err)
				}
				if !store.lastExpires.Equal(want) {
					t.Errorf("cached until %s, want the service's own %s", store.lastExpires, want)
				}
				return
			}
			// Clamped to the cap measured from the provider's own clock read, which
			// happens somewhere between these two. The window is the test's
			// scheduling jitter, not a tolerance on the arithmetic.
			lo, hi := before.Add(tc.want), after.Add(tc.want)
			if store.lastExpires.Before(lo) || store.lastExpires.After(hi) {
				t.Errorf("cached until %s, want %s from now (between %s and %s)", store.lastExpires, tc.want, lo, hi)
			}
		})
	}
}

// maxCachedLifetimeForTest mirrors the provider's cap. Duplicated rather than
// exported: the cap is a policy the test should notice changing.
const maxCachedLifetimeForTest = time.Hour

// refreshFixture wires a provider whose service answers every retrieval with a
// token naming the forceRefreshToken it was sent, so a refresh that lost the
// hint is visible in the credential rather than only in a call count.
func refreshFixture(t *testing.T, header string) (auth.RefreshingProvider, auth.CredentialStore, *gcp.Client, func() (int, string)) {
	t.Helper()
	var mu sync.Mutex
	var calls int
	var lastHint string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			ForceRefreshToken string `json:"forceRefreshToken"`
		}
		_ = json.NewDecoder(r.Body).Decode(&body)
		mu.Lock()
		calls++
		n := calls
		lastHint = body.ForceRefreshToken
		mu.Unlock()
		_, _ = io.WriteString(w, fmt.Sprintf(
			`{"success":{"token":"tok%d","header":%q,"expireTime":%q}}`,
			n, header, time.Now().Add(time.Hour).Format(time.RFC3339Nano)))
	}))
	t.Cleanup(srv.Close)

	client, err := gcp.NewClient(t.Context(), &gcp.Config{HTTPClient: srv.Client(), AgentIdentityEndpoint: srv.URL})
	if err != nil {
		t.Fatalf("NewClient() error = %v", err)
	}
	store := auth.NewInMemoryCredentialStore()
	p, err := gcp.NewProvider(t.Context(), gcp.ProviderConfig{
		Scheme: gcp.ProviderScheme{Name: testResource},
		Client: client,
		Store:  store,
	})
	if err != nil {
		t.Fatalf("NewProvider() error = %v", err)
	}
	rp, ok := p.(auth.RefreshingProvider)
	if !ok {
		t.Fatal("the gcp provider does not implement auth.RefreshingProvider")
	}
	return rp, store, client, func() (int, string) { mu.Lock(); defer mu.Unlock(); return calls, lastHint }
}

// A non-bearer credential is wrapped for the extra X-Goog-Api-Key header, so
// reading its concrete type without unwrapping finds nothing — and the refresh
// then loses the hint and is handed back the credential that was just rejected.
func TestProviderRefreshReadsThroughAWrappedCredential(t *testing.T) {
	rp, _, _, seen := refreshFixture(t, "X-Goog-Api-Key")
	ctx := adkContext(t, "user-1")

	first, err := rp.Credential(ctx)
	if err != nil {
		t.Fatalf("Credential() error = %v", err)
	}
	if _, ok := first.(auth.BearerCredential); ok {
		t.Fatal("the fixture produced a bearer credential; this case is about the wrapped one")
	}
	if _, err := rp.Refresh(ctx, first); err != nil {
		t.Fatalf("Refresh() error = %v", err)
	}
	calls, hint := seen()
	if calls != 2 {
		t.Fatalf("service calls = %d, want 2", calls)
	}
	// The hint is the whole point: without it the service is entitled to return
	// the very credential the downstream just rejected.
	if hint != "tok1" {
		t.Errorf("forceRefreshToken = %q, want %q — the token inside the wrapper", hint, "tok1")
	}
}

// A downstream that rejects everything must not set the rate at which this
// process re-mints, and thereby invalidates, an end user's credential.
func TestProviderRefreshIsRateLimited(t *testing.T) {
	rp, store, client, seen := refreshFixture(t, "Authorization: Bearer")
	ctx := adkContext(t, "user-1")

	first, err := rp.Credential(ctx)
	if err != nil {
		t.Fatalf("Credential() error = %v", err)
	}
	fresh, err := rp.Refresh(ctx, first)
	if err != nil {
		t.Fatalf("Refresh() error = %v", err)
	}
	if calls, _ := seen(); calls != 2 {
		t.Fatalf("service calls = %d, want 2 (the cold resolve and one refresh)", calls)
	}

	// A second forced refresh of the same credential inside the cooldown is
	// refused rather than served.
	if _, err := rp.Refresh(ctx, fresh); err == nil {
		t.Error("Refresh() twice inside the cooldown = nil error, want it refused")
	}
	if calls, _ := seen(); calls != 2 {
		t.Errorf("service calls = %d, want 2 (the refused refresh must not reach the service)", calls)
	}
	// Refused or not, the rejected credential is gone: it is known bad.
	key := client.CacheKey(gcp.ProviderScheme{Name: testResource}, "app", "user-1")
	if _, ok, _ := store.Get(ctx, key); ok {
		t.Error("the rejected credential is still cached after a refused refresh")
	}
}

// Another request may have refreshed already. Force-refreshing then destroys a
// token that is working, so a cached credential that is not the rejected one is
// served instead.
func TestProviderRefreshServesACredentialSomebodyElseAlreadyFetched(t *testing.T) {
	rp, _, _, seen := refreshFixture(t, "Authorization: Bearer")
	ctx := adkContext(t, "user-1")

	first, err := rp.Credential(ctx)
	if err != nil {
		t.Fatalf("Credential() error = %v", err)
	}
	fresh, err := rp.Refresh(ctx, first)
	if err != nil {
		t.Fatalf("Refresh() error = %v", err)
	}
	if calls, _ := seen(); calls != 2 {
		t.Fatalf("service calls = %d, want 2", calls)
	}

	// A straggler still holding the first credential asks again. The cache holds
	// the fresh one, which nothing has rejected.
	got, err := rp.Refresh(ctx, first)
	if err != nil {
		t.Fatalf("Refresh() for a straggler error = %v", err)
	}
	if got != fresh {
		t.Errorf("straggler was served %+v, want the credential already fetched, %+v", got, fresh)
	}
	if calls, _ := seen(); calls != 2 {
		t.Errorf("service calls = %d, want 2 (the straggler must not mint again)", calls)
	}
}

// A refresh whose replacement cannot be cached, and a refresh that fails, must
// both leave the rejected credential gone. Leaving it in place means the very
// next request is served the credential the downstream just refused, for the
// rest of its cached life.
func TestProviderRefreshEvictsWhatItCannotReplace(t *testing.T) {
	tests := []struct {
		name string
		// refresh is what the service does on the second call.
		refresh func(w http.ResponseWriter)
		wantErr bool
	}{
		{
			name: "the replacement has no usable lifetime",
			refresh: func(w http.ResponseWriter) {
				_, _ = io.WriteString(w, `{"success":{"token":"tok2","header":"Authorization: Bearer"}}`)
			},
		},
		{
			name:    "the refresh fails",
			refresh: func(w http.ResponseWriter) { w.WriteHeader(http.StatusServiceUnavailable) },
			wantErr: true,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var mu sync.Mutex
			var calls int
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				mu.Lock()
				calls++
				n := calls
				mu.Unlock()
				if n == 1 {
					_, _ = io.WriteString(w, fmt.Sprintf(
						`{"success":{"token":"tok1","header":"Authorization: Bearer","expireTime":%q}}`,
						time.Now().Add(time.Hour).Format(time.RFC3339Nano)))
					return
				}
				tc.refresh(w)
			}))
			defer srv.Close()

			client, err := gcp.NewClient(t.Context(), &gcp.Config{HTTPClient: srv.Client(), AgentIdentityEndpoint: srv.URL})
			if err != nil {
				t.Fatalf("NewClient() error = %v", err)
			}
			store := auth.NewInMemoryCredentialStore()
			p, err := gcp.NewProvider(t.Context(), gcp.ProviderConfig{
				Scheme: gcp.ProviderScheme{Name: testResource},
				Client: client,
				Store:  store,
			})
			if err != nil {
				t.Fatalf("NewProvider() error = %v", err)
			}
			ctx := adkContext(t, "user-1")
			first, err := p.Credential(ctx)
			if err != nil {
				t.Fatalf("Credential() error = %v", err)
			}
			key := client.CacheKey(gcp.ProviderScheme{Name: testResource}, "app", "user-1")
			if _, ok, _ := store.Get(ctx, key); !ok {
				t.Fatal("the first credential was not cached, so this case proves nothing")
			}

			_, err = p.(auth.RefreshingProvider).Refresh(ctx, first)
			if tc.wantErr != (err != nil) {
				t.Fatalf("Refresh() error = %v, wantErr %v", err, tc.wantErr)
			}
			if _, ok, _ := store.Get(ctx, key); ok {
				t.Error("the rejected credential is still cached, so the next request is served it again")
			}
		})
	}
}

// forceRefreshFields is both services' force-refresh spellings in one struct, so
// a test can assert that each route sends its own and not the other's.
type forceRefreshFields struct {
	ForceRefreshToken string `json:"forceRefreshToken"`
	ForceRefresh      bool   `json:"forceRefresh"`
}

// A service that hands back the credential it already had has not refreshed
// anything. Retrying with it is a second guaranteed failure, so Refresh reports
// instead and lets the downstream's own answer stand — and the known-bad
// credential does not survive in the cache either.
func TestProviderRefreshRejectsTheSameCredentialComingBack(t *testing.T) {
	var mu sync.Mutex
	var calls int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		calls++
		mu.Unlock()
		// The same token every time, whatever the request asked for.
		_, _ = io.WriteString(w, fmt.Sprintf(
			`{"success":{"token":"tok1","header":"Authorization: Bearer","expireTime":%q}}`,
			time.Now().Add(time.Hour).Format(time.RFC3339Nano)))
	}))
	defer srv.Close()

	client, err := gcp.NewClient(t.Context(), &gcp.Config{HTTPClient: srv.Client(), AgentIdentityEndpoint: srv.URL})
	if err != nil {
		t.Fatalf("NewClient() error = %v", err)
	}
	store := auth.NewInMemoryCredentialStore()
	p, err := gcp.NewProvider(t.Context(), gcp.ProviderConfig{
		Scheme: gcp.ProviderScheme{Name: testResource},
		Client: client,
		Store:  store,
	})
	if err != nil {
		t.Fatalf("NewProvider() error = %v", err)
	}
	ctx := adkContext(t, "user-1")
	first, err := p.Credential(ctx)
	if err != nil {
		t.Fatalf("Credential() error = %v", err)
	}

	got, err := p.(auth.RefreshingProvider).Refresh(ctx, first)
	if err == nil {
		t.Fatalf("Refresh() = %+v, nil error; want it refused when the service returns the rejected credential", got)
	}
	if got != nil {
		t.Errorf("Refresh() returned %+v alongside the error, want nil", got)
	}
	key := client.CacheKey(gcp.ProviderScheme{Name: testResource}, "app", "user-1")
	if _, ok, _ := store.Get(ctx, key); ok {
		t.Error("the rejected credential is still cached, so the next request is served it again")
	}
}

// The cooldown protects a credential at the service, which is identified by the
// end user and the scheme — not by the app name, which the service never sees.
// Counting app names would multiply the bound by however many a deployment wires
// up.
func TestProviderRefreshCooldownIgnoresTheAppName(t *testing.T) {
	rp, _, _, seen := refreshFixture(t, "Authorization: Bearer")

	first, err := rp.Credential(adkContextIn(t, "app-one", "user-1"))
	if err != nil {
		t.Fatalf("Credential() error = %v", err)
	}
	if _, err := rp.Refresh(adkContextIn(t, "app-one", "user-1"), first); err != nil {
		t.Fatalf("Refresh() error = %v", err)
	}
	callsAfterFirst, _ := seen()

	// A second app, same end user and scheme: one credential at the service, so
	// one forced refresh.
	second, err := rp.Credential(adkContextIn(t, "app-two", "user-1"))
	if err != nil {
		t.Fatalf("Credential() for the second app error = %v", err)
	}
	if _, err := rp.Refresh(adkContextIn(t, "app-two", "user-1"), second); err == nil {
		t.Error("Refresh() under a second app name = nil error, want the cooldown to hold")
	}
	calls, _ := seen()
	if calls != callsAfterFirst+1 {
		t.Errorf("service calls = %d, want %d (only the second app's cold resolve, not another forced refresh)", calls, callsAfterFirst+1)
	}
}

// swappingStore lets a test place a credential change between two of the
// provider's reads, which is what the eviction guard exists for and what a
// sequential test cannot otherwise produce.
type swappingStore struct {
	inner   auth.CredentialStore
	armed   int             // Gets still to answer with pinned before swapping
	pinned  auth.Credential // what the provider sees first
	swapped auth.Credential // what it sees after, as if somebody else had written
	deletes int
}

func (s *swappingStore) Get(ctx context.Context, key auth.CredentialKey) (auth.Credential, bool, error) {
	switch {
	case s.armed > 0:
		s.armed--
		return s.pinned, true, nil
	case s.swapped != nil:
		return s.swapped, true, nil
	}
	return s.inner.Get(ctx, key)
}

func (s *swappingStore) Set(ctx context.Context, key auth.CredentialKey, cred auth.Credential, expiresAt time.Time) error {
	return s.inner.Set(ctx, key, cred, expiresAt)
}

func (s *swappingStore) Delete(ctx context.Context, key auth.CredentialKey) error {
	s.deletes++
	return s.inner.Delete(ctx, key)
}

// Eviction is the last thing Refresh does on every failing path, and by then its
// view of the entry is stale: another request may have replaced it since. That
// replacement has not been rejected, so deleting it would throw away a working
// credential — and leave every straggler still holding the old one with nothing
// to be served.
func TestProviderRefreshDoesNotEvictACredentialSomebodyElseReplaced(t *testing.T) {
	var mu sync.Mutex
	var calls int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		calls++
		n := calls
		mu.Unlock()
		// A new token each time, so the same-token guard never trips here.
		_, _ = io.WriteString(w, fmt.Sprintf(
			`{"success":{"token":"tok%d","header":"Authorization: Bearer","expireTime":%q}}`,
			n, time.Now().Add(time.Hour).Format(time.RFC3339Nano)))
	}))
	defer srv.Close()

	client, err := gcp.NewClient(t.Context(), &gcp.Config{HTTPClient: srv.Client(), AgentIdentityEndpoint: srv.URL})
	if err != nil {
		t.Fatalf("NewClient() error = %v", err)
	}
	store := &swappingStore{inner: auth.NewInMemoryCredentialStore()}
	p, err := gcp.NewProvider(t.Context(), gcp.ProviderConfig{
		Scheme: gcp.ProviderScheme{Name: testResource},
		Client: client,
		Store:  store,
	})
	if err != nil {
		t.Fatalf("NewProvider() error = %v", err)
	}
	rp := p.(auth.RefreshingProvider)
	ctx := adkContext(t, "user-1")

	rejected, err := p.Credential(ctx)
	if err != nil {
		t.Fatalf("Credential() error = %v", err)
	}
	// Burn the cooldown, so the Refresh below takes the throttled path — the
	// shortest route to the eviction.
	if _, err := rp.Refresh(ctx, rejected); err != nil {
		t.Fatalf("Refresh() error = %v", err)
	}

	// One read answers with the rejected credential, so Refresh proceeds past its
	// straggler check; every read after that answers with a replacement, as if
	// another request had landed in between.
	store.pinned = rejected
	store.swapped = auth.BearerCredential{Token: "replaced-by-somebody-else"}
	store.armed = 1
	before := store.deletes

	if _, err := rp.Refresh(ctx, rejected); err == nil {
		t.Fatal("Refresh() inside the cooldown = nil error, want it refused")
	}
	if store.deletes != before {
		t.Error("Refresh deleted an entry another request had already replaced")
	}
}

// The cooldown must separate end users. It is the only dimension of the refresh
// slot that varies within one provider — the scheme is fixed at construction —
// so without this, collapsing the slot to a constant passes every other test in
// the package, and one user's 401-storming downstream would block every other
// user's refresh for the length of the cooldown.
func TestProviderRefreshCooldownSeparatesUsers(t *testing.T) {
	rp, _, _, seen := refreshFixture(t, "Authorization: Bearer")

	refresh := func(user string) error {
		t.Helper()
		ctx := adkContext(t, user)
		cred, err := rp.Credential(ctx)
		if err != nil {
			t.Fatalf("Credential(%q) error = %v", user, err)
		}
		_, err = rp.Refresh(ctx, cred)
		return err
	}
	if err := refresh("user-1"); err != nil {
		t.Fatalf("Refresh() for the first user error = %v", err)
	}
	if err := refresh("user-2"); err != nil {
		t.Errorf("Refresh() for a second user error = %v; the first user's cooldown must not bind them", err)
	}
	// Two cold resolves and two forced refreshes.
	if calls, _ := seen(); calls != 4 {
		t.Errorf("service calls = %d, want 4", calls)
	}
}

// The end-to-end path nothing else drives: a downstream 401 reaching a real
// provider through a real Transport, the rejected credential surviving the trip
// in a form the provider can read a token out of, and the retry going out with
// the replacement. Both halves are covered elsewhere against a stub of the
// other, which is exactly the seam two earlier defects lived on.
func TestTransportRefreshesThroughTheRealProvider(t *testing.T) {
	var mu sync.Mutex
	var hints []string
	credSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			ForceRefreshToken string `json:"forceRefreshToken"`
		}
		_ = json.NewDecoder(r.Body).Decode(&body)
		mu.Lock()
		hints = append(hints, body.ForceRefreshToken)
		n := len(hints)
		mu.Unlock()
		_, _ = io.WriteString(w, fmt.Sprintf(
			`{"success":{"token":"tok%d","header":"Authorization: Bearer","expireTime":%q}}`,
			n, time.Now().Add(time.Hour).Format(time.RFC3339Nano)))
	}))
	defer credSrv.Close()

	client, err := gcp.NewClient(t.Context(), &gcp.Config{HTTPClient: credSrv.Client(), AgentIdentityEndpoint: credSrv.URL})
	if err != nil {
		t.Fatalf("NewClient() error = %v", err)
	}
	p, err := gcp.NewProvider(t.Context(), gcp.ProviderConfig{
		Scheme: gcp.ProviderScheme{Name: testResource},
		Client: client,
		Store:  auth.NewInMemoryCredentialStore(),
	})
	if err != nil {
		t.Fatalf("NewProvider() error = %v", err)
	}

	// The downstream refuses the first credential and accepts the second.
	var seenAuth []string
	downstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		seenAuth = append(seenAuth, r.Header.Get("Authorization"))
		n := len(seenAuth)
		mu.Unlock()
		if n == 1 {
			w.WriteHeader(http.StatusUnauthorized)
			_, _ = io.WriteString(w, "no")
			return
		}
		_, _ = io.WriteString(w, "ok")
	}))
	defer downstream.Close()

	hc := &http.Client{Transport: &auth.Transport{Provider: p, Base: downstream.Client().Transport}}
	req, err := http.NewRequestWithContext(adkContext(t, "user-1"), http.MethodGet, downstream.URL, nil)
	if err != nil {
		t.Fatalf("NewRequest() error = %v", err)
	}
	resp, err := hc.Do(req)
	if err != nil {
		t.Fatalf("Do() error = %v", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		t.Errorf("status = %d, want 200 after the refresh and retry", resp.StatusCode)
	}
	mu.Lock()
	gotAuth, gotHints := slices.Clone(seenAuth), slices.Clone(hints)
	mu.Unlock()
	if want := []string{"Bearer tok1", "Bearer tok2"}; !slices.Equal(gotAuth, want) {
		t.Errorf("downstream saw %q, want %q", gotAuth, want)
	}
	// The second retrieval must carry the rejected token, which is the whole
	// point of passing it through Transport rather than re-reading the cache.
	if want := []string{"", "tok1"}; !slices.Equal(gotHints, want) {
		t.Errorf("credential service saw forceRefreshToken %q, want %q", gotHints, want)
	}
}
