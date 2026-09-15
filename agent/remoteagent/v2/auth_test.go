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

package remoteagent

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"iter"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"net/url"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/a2aproject/a2a-go/v2/a2a"
	"github.com/a2aproject/a2a-go/v2/a2aclient"
	"github.com/a2aproject/a2a-go/v2/a2asrv"
	"github.com/a2aproject/a2a-go/v2/log"
	"golang.org/x/oauth2"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/auth"
	iremoteagent "google.golang.org/adk/v2/internal/agent/remoteagent"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/session"
)

// errResolve is the provider failure asserted on with errors.Is, so the %w
// wrapping in credentialsService.Get stays load-bearing.
var errResolve = errors.New("resolve failed")

// tokenSourceFunc adapts a function to an [oauth2.TokenSource].
type tokenSourceFunc func() (*oauth2.Token, error)

func (f tokenSourceFunc) Token() (*oauth2.Token, error) { return f() }

// anyPlacementCard declares one scheme of each placement the adapter supports,
// so a table row can pick the one its credential belongs in.
var anyPlacementCard = newSecureCard("http://example.invalid",
	a2a.NamedSecuritySchemes{
		"apikey": a2a.APIKeySecurityScheme{Location: a2a.APIKeySecuritySchemeLocationHeader, Name: "X-Api-Key"},
		"bearer": a2a.HTTPAuthSecurityScheme{Scheme: "Bearer"},
	},
	nil,
)

func schemeFor(apiKey bool) a2a.SecuritySchemeName {
	if apiKey {
		return "apikey"
	}
	return "bearer"
}

func TestCredentialsServiceGet(t *testing.T) {
	tests := []struct {
		name string
		// provider is the credential source under test.
		provider auth.CredentialProvider
		// apiKeyScheme picks the card scheme to resolve against; API-key
		// credentials only match the card's apiKey scheme.
		apiKeyScheme bool
		want         a2aclient.AuthCredential
		// wantErrContains pins which branch produced the error; asserting only
		// err != nil would let every branch collapse into one message.
		wantErrContains string
		// wantErrAbsent must not appear in the error, so a secret cannot leak
		// into a message the interceptor logs.
		wantErrAbsent string
	}{
		{
			name:     "static bearer token",
			provider: auth.StaticToken("tok"),
			want:     "tok",
		},
		{
			name:         "api key value",
			provider:     auth.APIKey("X-Api-Key", "secret"),
			apiKeyScheme: true,
			want:         "secret",
		},
		{
			name:     "oauth2 token source",
			provider: auth.TokenSourceProvider(oauth2.StaticTokenSource(&oauth2.Token{AccessToken: "at"})),
			want:     "at",
		},
		{
			name: "pointer to bearer credential",
			provider: auth.ProviderFunc(func(context.Context) (auth.Credential, error) {
				return &auth.BearerCredential{Token: "tok"}, nil
			}),
			want: "tok",
		},
		{
			name: "pointer to api key credential",
			provider: auth.ProviderFunc(func(context.Context) (auth.Credential, error) {
				return &auth.APIKeyCredential{Name: "X-Api-Key", Value: "secret"}, nil
			}),
			apiKeyScheme: true,
			want:         "secret",
		},
		{
			name: "pointer to oauth2 credential",
			provider: auth.ProviderFunc(func(context.Context) (auth.Credential, error) {
				return &auth.OAuth2Credential{TokenSource: oauth2.StaticTokenSource(&oauth2.Token{AccessToken: "at"})}, nil
			}),
			want: "at",
		},
		{
			name: "typed nil pointer credential",
			provider: auth.ProviderFunc(func(context.Context) (auth.Credential, error) {
				return (*auth.BearerCredential)(nil), nil
			}),
			wantErrContains: "cannot send *auth.BearerCredential",
		},
		{
			name: "oauth2 missing token source",
			provider: auth.ProviderFunc(func(context.Context) (auth.Credential, error) {
				return auth.OAuth2Credential{}, nil
			}),
			wantErrContains: "no token source",
		},
		{
			name:            "oauth2 empty access token",
			provider:        auth.TokenSourceProvider(oauth2.StaticTokenSource(&oauth2.Token{AccessToken: ""})),
			wantErrContains: "empty access token",
		},
		{
			name: "oauth2 token source error",
			provider: auth.TokenSourceProvider(tokenSourceFunc(func() (*oauth2.Token, error) {
				return nil, errors.New("token endpoint refused")
			})),
			wantErrContains: "mint oauth2 token",
		},
		{
			name: "oauth2 non-bearer token type",
			provider: auth.TokenSourceProvider(oauth2.StaticTokenSource(&oauth2.Token{
				AccessToken: "super-secret", TokenType: "mac",
			})),
			wantErrContains: "cannot be sent over a2a",
			wantErrAbsent:   "super-secret",
		},
		{
			name:            "empty api key value",
			provider:        auth.APIKey("X-Api-Key", ""),
			apiKeyScheme:    true,
			wantErrContains: "empty value",
		},
		{
			name:            "empty bearer token",
			provider:        auth.StaticToken(""),
			wantErrContains: "empty token",
		},
		{
			name: "nil credential",
			provider: auth.ProviderFunc(func(context.Context) (auth.Credential, error) {
				return nil, nil
			}),
			wantErrContains: "nil credential",
		},
		{
			name: "basic credential is untransmittable",
			provider: auth.ProviderFunc(func(context.Context) (auth.Credential, error) {
				return auth.BasicCredential{Username: "u", Password: "hunter2"}, nil
			}),
			wantErrContains: "cannot send auth.BasicCredential",
			wantErrAbsent:   "hunter2",
		},
		{
			name: "wrapped credential is untransmittable",
			provider: auth.ProviderFunc(func(context.Context) (auth.Credential, error) {
				return auth.WithHeaders(auth.BearerCredential{Token: "tok"}, map[string]string{"x-goog-user-project": "p"}), nil
			}),
			wantErrContains: "cannot send",
		},
		{
			name: "provider error",
			provider: auth.ProviderFunc(func(context.Context) (auth.Credential, error) {
				return nil, errResolve
			}),
			wantErrContains: "resolve auth credential",
		},
		{
			name:            "nil provider",
			provider:        nil,
			wantErrContains: "no credential provider",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			svc := credentialsService{provider: tc.provider}
			// Every row supplies a card naming both placements, so the table
			// exercises credential handling rather than scheme selection.
			ctx := iremoteagent.WithAgentCard(t.Context(), anyPlacementCard)
			got, err := svc.Get(ctx, a2aclient.SessionID("sid"), schemeFor(tc.apiKeyScheme))
			if tc.wantErrContains != "" {
				if err == nil {
					t.Fatalf("Get() = %q, nil error; want error containing %q", got, tc.wantErrContains)
				}
				if !strings.Contains(err.Error(), tc.wantErrContains) {
					t.Errorf("Get() error = %v, want it to contain %q", err, tc.wantErrContains)
				}
				if tc.wantErrAbsent != "" && strings.Contains(err.Error(), tc.wantErrAbsent) {
					t.Errorf("Get() error = %v, want it not to leak %q", err, tc.wantErrAbsent)
				}
				return
			}
			if err != nil {
				t.Fatalf("Get() error = %v", err)
			}
			if got != tc.want {
				t.Errorf("Get() = %q, want %q", got, tc.want)
			}
		})
	}
}

// TestCredentialsServiceGetWrapsProviderError pins that the provider's own
// error survives the wrap, which is the only way an operator sees why
// resolution failed.
// TestCredentialScope pins the documented key format with literal strings. The
// end-to-end tests can only show that two identities differ, which every
// weaker scope also satisfies; only this one rejects dropping a part or the
// percent-encoding that keeps "/" from being ambiguous.
func TestCredentialScope(t *testing.T) {
	tests := []struct {
		name                                string
		appName, userID, sessionID, agentAt string
		want                                a2aclient.SessionID
	}{
		{
			name: "plain", appName: "app", userID: "alice", sessionID: "s1", agentAt: "crm",
			want: "app/alice/s1/crm",
		},
		{
			name: "separator in a part is escaped", appName: "a/b", userID: "c", sessionID: "d", agentAt: "e",
			want: "a%2Fb/c/d/e",
		},
		{
			name: "the ambiguous twin does not collide", appName: "a", userID: "b/c", sessionID: "d", agentAt: "e",
			want: "a/b%2Fc/d/e",
		},
		{
			name: "percent is escaped so escaping cannot be forged", appName: "a%2Fb", userID: "c", sessionID: "d", agentAt: "e",
			want: "a%252Fb/c/d/e",
		},
		{
			name: "empty parts keep their positions", appName: "", userID: "", sessionID: "", agentAt: "",
			want: "///",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := iremoteagent.CredentialScope(tc.appName, tc.userID, tc.sessionID, tc.agentAt)
			if got != tc.want {
				t.Errorf("CredentialScope(%q, %q, %q, %q) = %q, want %q", tc.appName, tc.userID, tc.sessionID, tc.agentAt, got, tc.want)
			}
		})
	}
}

func TestCredentialsServiceGetWrapsProviderError(t *testing.T) {
	svc := credentialsService{provider: auth.ProviderFunc(func(context.Context) (auth.Credential, error) {
		return nil, errResolve
	})}
	_, err := svc.Get(iremoteagent.WithAgentCard(t.Context(), anyPlacementCard), "sid", "bearer")
	if !errors.Is(err, errResolve) {
		t.Errorf("Get() error = %v, want it to wrap %v", err, errResolve)
	}
}

// TestCredentialsServiceGetMatchesScheme covers the card-driven scheme choice:
// Get hands back a secret only for a scheme that can actually carry it, and
// reports ErrCredentialNotFound otherwise so the interceptor moves on instead
// of writing the secret somewhere the remote will not read it.
func TestCredentialsServiceGetMatchesScheme(t *testing.T) {
	card := newSecureCard("http://example.invalid",
		a2a.NamedSecuritySchemes{
			"apikey":       a2a.APIKeySecurityScheme{Location: a2a.APIKeySecuritySchemeLocationHeader, Name: "X-Api-Key"},
			"apikey-query": a2a.APIKeySecurityScheme{Location: a2a.APIKeySecuritySchemeLocationQuery, Name: "api_key"},
			"bearer":       a2a.HTTPAuthSecurityScheme{Scheme: "Bearer"},
			"basic":        a2a.HTTPAuthSecurityScheme{Scheme: "Basic"},
			"oauth2":       a2a.OAuth2SecurityScheme{},
			"mtls":         a2a.MutualTLSSecurityScheme{},
		},
		nil,
	)

	tests := []struct {
		name     string
		provider auth.CredentialProvider
		scheme   string
		want     a2aclient.AuthCredential
		wantSkip bool
	}{
		{name: "api key on apiKey scheme", provider: auth.APIKey("ignored", "secret"), scheme: "apikey", want: "secret"},
		{name: "api key on bearer scheme", provider: auth.APIKey("ignored", "secret"), scheme: "bearer", wantSkip: true},
		{name: "bearer on bearer scheme", provider: auth.StaticToken("tok"), scheme: "bearer", want: "tok"},
		{name: "bearer on oauth2 scheme", provider: auth.StaticToken("tok"), scheme: "oauth2", want: "tok"},
		{name: "bearer on apiKey scheme", provider: auth.StaticToken("tok"), scheme: "apikey", wantSkip: true},
		{name: "bearer on mtls scheme", provider: auth.StaticToken("tok"), scheme: "mtls", wantSkip: true},
		{name: "bearer on scheme the card does not name", provider: auth.StaticToken("tok"), scheme: "absent", wantSkip: true},
		// The interceptor writes an API key as a header whatever the card asks
		// for, so a query-located key would go somewhere the card never named.
		{name: "api key on query-located apiKey scheme", provider: auth.APIKey("ignored", "secret"), scheme: "apikey-query", wantSkip: true},
		// The interceptor always writes "Bearer", so a Basic card would get a
		// mislabeled credential.
		{name: "bearer on basic HTTP scheme", provider: auth.StaticToken("tok"), scheme: "basic", wantSkip: true},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ctx := iremoteagent.WithAgentCard(t.Context(), card)
			got, err := credentialsService{provider: tc.provider}.Get(ctx, "sid", a2a.SecuritySchemeName(tc.scheme))
			if tc.wantSkip {
				if !errors.Is(err, a2aclient.ErrCredentialNotFound) {
					t.Fatalf("Get() error = %v, want %v so the interceptor tries the next scheme", err, a2aclient.ErrCredentialNotFound)
				}
				return
			}
			if err != nil {
				t.Fatalf("Get() error = %v", err)
			}
			if got != tc.want {
				t.Errorf("Get() = %q, want %q", got, tc.want)
			}
		})
	}
}

// TestMintAccessTokenHonorsContext pins that a hung token mint is interruptible.
// oauth2.TokenSource.Token takes no context, so without an explicit bound a
// blocked mint holds the run loop past every timeout around it.
func TestMintAccessTokenHonorsContext(t *testing.T) {
	release := make(chan struct{})
	t.Cleanup(func() { close(release) })
	blocked := tokenSourceFunc(func() (*oauth2.Token, error) {
		<-release
		return &oauth2.Token{AccessToken: "late"}, nil
	})

	ctx, cancel := context.WithTimeout(t.Context(), 50*time.Millisecond)
	defer cancel()

	done := make(chan error, 1)
	go func() {
		_, err := mintAccessToken(ctx, blocked)
		done <- err
	}()

	select {
	case err := <-done:
		if !errors.Is(err, context.DeadlineExceeded) {
			t.Errorf("mintAccessToken() error = %v, want it to wrap %v", err, context.DeadlineExceeded)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("mintAccessToken() did not return once the context expired")
	}
}

func TestNewA2AAuthWithClientProviderIsError(t *testing.T) {
	_, err := NewA2A(A2AConfig{
		Name:           "a2a",
		AgentCard:      &a2a.AgentCard{Name: "a2a"},
		Auth:           auth.StaticToken("tok"),
		ClientProvider: NewA2AClientProvider(a2aclient.NewFactory()),
	})
	if err == nil || !strings.Contains(err.Error(), "cannot be combined with a custom ClientProvider") {
		t.Fatalf("NewA2A() error = %v, want error about Auth combined with a custom ClientProvider", err)
	}
}

// TestNewA2AAuthTypedNilIsError covers a provider that is nil inside a non-nil
// interface: it passes an ordinary != nil check and would otherwise panic on
// the first request, inside the deferred cleanup as well as the send.
func TestNewA2AAuthTypedNilIsError(t *testing.T) {
	_, err := NewA2A(A2AConfig{
		Name:      "a2a",
		AgentCard: &a2a.AgentCard{Name: "a2a"},
		Auth:      auth.ProviderFunc(nil),
	})
	if err == nil || !strings.Contains(err.Error(), "holds a nil") {
		t.Fatalf("NewA2A() error = %v, want error about a nil Auth provider", err)
	}
}

func TestRemoteAgent_AuthAttachesBearerHeader(t *testing.T) {
	var mu sync.Mutex
	var gotAuth string
	srv := serveRecordingA2A(t, func(r *http.Request) {
		mu.Lock()
		gotAuth = r.Header.Get("Authorization")
		mu.Unlock()
	}, a2a.NewMessage(a2a.MessageRoleAgent, a2a.NewTextPart("ok")))

	card := bearerCard(srv.URL)
	provider := func(context.Context) (*a2a.AgentCard, error) { return card, nil }

	// Cover both card sources (static/provider) and both send paths (streaming/not).
	tests := []struct {
		name      string
		cfg       A2AConfig
		streaming agent.StreamingMode
	}{
		{
			name:      "static card, streaming",
			cfg:       A2AConfig{Name: "a2a", AgentCard: card, Auth: auth.StaticToken("secret-token")},
			streaming: agent.StreamingModeSSE,
		},
		{
			name:      "card provider, streaming",
			cfg:       A2AConfig{Name: "a2a", AgentCardProvider: provider, Auth: auth.StaticToken("secret-token")},
			streaming: agent.StreamingModeSSE,
		},
		{
			name:      "static card, non-streaming",
			cfg:       A2AConfig{Name: "a2a", AgentCard: card, Auth: auth.StaticToken("secret-token")},
			streaming: agent.StreamingModeNone,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			mu.Lock()
			gotAuth = ""
			mu.Unlock()

			remoteAgent, err := NewA2A(tc.cfg)
			if err != nil {
				t.Fatalf("NewA2A() error = %v", err)
			}

			ictx := newInvocationContextWithStreamingMode(t, []*session.Event{newUserHello()}, tc.streaming)
			if _, err := runAndCollect(ictx, remoteAgent); err != nil {
				t.Fatalf("agent.Run() error = %v", err)
			}

			mu.Lock()
			defer mu.Unlock()
			if gotAuth != "Bearer secret-token" {
				t.Errorf("server saw Authorization = %q, want %q", gotAuth, "Bearer secret-token")
			}
		})
	}
}

// TestRemoteAgent_AuthFailOpenSendsUnauthenticated pins the fail-open contract:
// when the provider errors, the a2a interceptor drops auth and the request still
// goes out (unauthenticated), rather than failing the call.
func TestRemoteAgent_AuthFailOpenSendsUnauthenticated(t *testing.T) {
	var mu sync.Mutex
	var gotAuth string
	var sawRequest bool
	srv := serveRecordingA2A(t, func(r *http.Request) {
		mu.Lock()
		gotAuth = r.Header.Get("Authorization")
		sawRequest = true
		mu.Unlock()
	}, a2a.NewMessage(a2a.MessageRoleAgent, a2a.NewTextPart("ok")))

	var calls atomic.Int32
	failing := auth.ProviderFunc(func(context.Context) (auth.Credential, error) {
		calls.Add(1)
		return nil, errResolve
	})

	remoteAgent, err := NewA2A(A2AConfig{Name: "a2a", AgentCard: bearerCard(srv.URL), Auth: failing})
	if err != nil {
		t.Fatalf("NewA2A() error = %v", err)
	}

	ictx := newInvocationContext(t, []*session.Event{newUserHello()})
	if _, err := runAndCollect(ictx, remoteAgent); err != nil {
		t.Fatalf("agent.Run() error = %v; fail-open means the request should still succeed", err)
	}

	mu.Lock()
	defer mu.Unlock()
	if !sawRequest {
		t.Fatal("server never received the request")
	}
	if gotAuth != "" {
		t.Errorf("server saw Authorization = %q, want empty (fail-open, no credential)", gotAuth)
	}
	// Without this the test would also pass if the interceptor were never
	// installed, which is not the contract being pinned.
	if got := calls.Load(); got == 0 {
		t.Error("the provider was never called; this test must show resolution ran and failed, not that it never ran")
	}
}

// TestRemoteAgent_AuthAttachesAPIKeyHeader covers the apiKey scheme. The card
// and the credential name different headers on purpose: placement comes from
// the card alone, and APIKeyCredential.Name is ignored.
func TestRemoteAgent_AuthAttachesAPIKeyHeader(t *testing.T) {
	var mu sync.Mutex
	var gotCardKey, gotCallerKey, gotAuth string
	srv := serveRecordingA2A(t, func(r *http.Request) {
		mu.Lock()
		gotCardKey = r.Header.Get("X-Card-Key")
		gotCallerKey = r.Header.Get("X-Caller-Key")
		gotAuth = r.Header.Get("Authorization")
		mu.Unlock()
	}, a2a.NewMessage(a2a.MessageRoleAgent, a2a.NewTextPart("ok")))

	card := newSecureCard(srv.URL,
		a2a.NamedSecuritySchemes{
			"apikey": a2a.APIKeySecurityScheme{Location: a2a.APIKeySecuritySchemeLocationHeader, Name: "X-Card-Key"},
		},
		a2a.SecurityRequirementsOptions{
			{a2a.SecuritySchemeName("apikey"): a2a.SecuritySchemeScopes{}},
		},
	)

	remoteAgent, err := NewA2A(A2AConfig{Name: "a2a", AgentCard: card, Auth: auth.APIKey("X-Caller-Key", "secret")})
	if err != nil {
		t.Fatalf("NewA2A() error = %v", err)
	}

	ictx := newInvocationContext(t, []*session.Event{newUserHello()})
	if _, err := runAndCollect(ictx, remoteAgent); err != nil {
		t.Fatalf("agent.Run() error = %v", err)
	}

	mu.Lock()
	defer mu.Unlock()
	if gotCardKey != "secret" {
		t.Errorf("server saw X-Card-Key = %q, want %q", gotCardKey, "secret")
	}
	if gotCallerKey != "" {
		t.Errorf("server saw X-Caller-Key = %q, want empty; the card names the header, not the credential", gotCallerKey)
	}
	if gotAuth != "" {
		t.Errorf("server saw Authorization = %q, want empty (apiKey uses its own header)", gotAuth)
	}
}

// TestRemoteAgent_AuthMultiSchemeCardIsDeterministic covers a card whose single
// requirement object names two schemes. The interceptor picks among them in Go
// map order, so without the scheme check in Get the bearer token would land in
// the API-key header on a random subset of requests.
func TestRemoteAgent_AuthMultiSchemeCardIsDeterministic(t *testing.T) {
	var mu sync.Mutex
	var placements []string
	srv := serveRecordingA2A(t, func(r *http.Request) {
		mu.Lock()
		switch {
		case r.Header.Get("Authorization") != "":
			placements = append(placements, "Authorization="+r.Header.Get("Authorization"))
		case r.Header.Get("X-Api-Key") != "":
			placements = append(placements, "X-Api-Key="+r.Header.Get("X-Api-Key"))
		default:
			placements = append(placements, "none")
		}
		mu.Unlock()
	}, a2a.NewMessage(a2a.MessageRoleAgent, a2a.NewTextPart("ok")))

	card := newSecureCard(srv.URL,
		a2a.NamedSecuritySchemes{
			"apikey": a2a.APIKeySecurityScheme{Location: a2a.APIKeySecuritySchemeLocationHeader, Name: "X-Api-Key"},
			"bearer": a2a.HTTPAuthSecurityScheme{Scheme: "Bearer"},
		},
		a2a.SecurityRequirementsOptions{{
			a2a.SecuritySchemeName("apikey"): a2a.SecuritySchemeScopes{},
			a2a.SecuritySchemeName("bearer"): a2a.SecuritySchemeScopes{},
		}},
	)

	remoteAgent, err := NewA2A(A2AConfig{Name: "a2a", AgentCard: card, Auth: auth.StaticToken("secret-token")})
	if err != nil {
		t.Fatalf("NewA2A() error = %v", err)
	}

	// Enough runs that a coin-flip over map order would almost certainly show.
	const runs = 40
	for range runs {
		if _, err := runAndCollect(newInvocationContext(t, []*session.Event{newUserHello()}), remoteAgent); err != nil {
			t.Fatalf("agent.Run() error = %v", err)
		}
	}

	mu.Lock()
	defer mu.Unlock()
	const want = "Authorization=Bearer secret-token"
	for i, got := range placements {
		if got != want {
			t.Fatalf("run %d placed the credential as %q, want %q on every run (saw %d runs)", i, got, want, len(placements))
		}
	}
	if len(placements) != runs {
		t.Errorf("server saw %d requests, want %d", len(placements), runs)
	}
}

// TestRemoteAgent_AuthAcceptedByEnforcingServer proves the credential is usable,
// not merely present: a server that requires the right bearer token accepts the
// correct token and rejects a wrong one.
func TestRemoteAgent_AuthAcceptedByEnforcingServer(t *testing.T) {
	const goodToken = "good-token"
	inner := a2asrv.NewJSONRPCHandler(a2asrv.NewHandler(newA2AEventReplay(t, []a2a.Event{
		a2a.NewMessage(a2a.MessageRoleAgent, a2a.NewTextPart("ok")),
	})))
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer "+goodToken {
			http.Error(w, "unauthorized", http.StatusUnauthorized)
			return
		}
		inner.ServeHTTP(w, r)
	}))
	defer srv.Close()

	tests := []struct {
		name    string
		token   string
		wantErr bool
	}{
		{name: "accepted", token: goodToken, wantErr: false},
		{name: "rejected", token: "bad-token", wantErr: true},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			remoteAgent, err := NewA2A(A2AConfig{Name: "a2a", AgentCard: bearerCard(srv.URL), Auth: auth.StaticToken(tc.token)})
			if err != nil {
				t.Fatalf("NewA2A() error = %v", err)
			}

			events, err := runAndCollect(newInvocationContext(t, []*session.Event{newUserHello()}), remoteAgent)
			if err != nil {
				t.Fatalf("agent.Run() error = %v", err)
			}

			errEvent := firstErrorEvent(events)
			if tc.wantErr {
				if errEvent == nil {
					t.Fatal("want an error event from the rejected request, got none")
				}
				if !strings.Contains(errEvent.ErrorMessage, "401") {
					t.Errorf("error event = %q, want it to mention 401", errEvent.ErrorMessage)
				}
				return
			}
			if errEvent != nil {
				t.Fatalf("unexpected error event: %q", errEvent.ErrorMessage)
			}
			if !eventsContainText(events, "ok") {
				t.Errorf("authenticated response missing the remote agent's reply %q", "ok")
			}
		})
	}
}

// TestRemoteAgent_AuthScopeIsPerUser covers the credential scope handed to a
// session-aware provider. The session id alone is caller-supplied and optional,
// so two users each holding a session called "default" must still resolve to
// their own credential.
func TestRemoteAgent_AuthScopeIsPerUser(t *testing.T) {
	var mu sync.Mutex
	var gotAuth string
	srv := serveRecordingA2A(t, func(r *http.Request) {
		mu.Lock()
		gotAuth = r.Header.Get("Authorization")
		mu.Unlock()
	}, a2a.NewMessage(a2a.MessageRoleAgent, a2a.NewTextPart("ok")))

	// Mints one token per distinct scope and caches it, the pattern the scope
	// exists for. A scope that collides hands the second user the first's token.
	var cacheMu sync.Mutex
	cache := map[a2aclient.SessionID]string{}
	perScope := auth.ProviderFunc(func(ctx context.Context) (auth.Credential, error) {
		sid, ok := a2aclient.SessionIDFrom(ctx)
		if !ok {
			return nil, errors.New("no credential scope on the context")
		}
		cacheMu.Lock()
		defer cacheMu.Unlock()
		tok, ok := cache[sid]
		if !ok {
			tok = fmt.Sprintf("tok-%d", len(cache))
			cache[sid] = tok
		}
		return auth.BearerCredential{Token: tok}, nil
	})

	remoteAgent, err := NewA2A(A2AConfig{Name: "a2a", AgentCard: bearerCard(srv.URL), Auth: perScope})
	if err != nil {
		t.Fatalf("NewA2A() error = %v", err)
	}

	// Same app and session id, different users.
	seen := map[string]string{}
	for _, user := range []string{"alice", "bob"} {
		ictx := newInvocationContextFor(t, t.Name(), user, "default")
		if _, err := runAndCollect(ictx, remoteAgent); err != nil {
			t.Fatalf("agent.Run() for %s error = %v", user, err)
		}
		mu.Lock()
		seen[user] = gotAuth
		gotAuth = ""
		mu.Unlock()
	}

	if seen["alice"] == "" || seen["bob"] == "" {
		t.Fatalf("both users should have sent a credential, got %v", seen)
	}
	if seen["alice"] == seen["bob"] {
		t.Errorf("alice and bob both sent %q; the credential scope collides across users", seen["alice"])
	}
}

// TestRemoteAgent_AuthProviderSeesInvocationContext pins that scoping the
// context does not hide the ADK context behind it: auth.CredentialProvider's
// contract is that a provider can recover it to learn who is acting.
func TestRemoteAgent_AuthProviderSeesInvocationContext(t *testing.T) {
	var mu sync.Mutex
	var gotAuth string
	srv := serveRecordingA2A(t, func(r *http.Request) {
		mu.Lock()
		gotAuth = r.Header.Get("Authorization")
		mu.Unlock()
	}, a2a.NewMessage(a2a.MessageRoleAgent, a2a.NewTextPart("ok")))

	perUser := auth.ProviderFunc(func(ctx context.Context) (auth.Credential, error) {
		ictx, ok := ctx.(agent.InvocationContext)
		if !ok {
			return nil, fmt.Errorf("context is %T, not an agent.InvocationContext", ctx)
		}
		return auth.BearerCredential{Token: "tok-" + ictx.Session().UserID()}, nil
	})

	remoteAgent, err := NewA2A(A2AConfig{Name: "a2a", AgentCard: bearerCard(srv.URL), Auth: perUser})
	if err != nil {
		t.Fatalf("NewA2A() error = %v", err)
	}
	if _, err := runAndCollect(newInvocationContextFor(t, t.Name(), "carol", "default"), remoteAgent); err != nil {
		t.Fatalf("agent.Run() error = %v", err)
	}

	mu.Lock()
	defer mu.Unlock()
	if want := "Bearer tok-carol"; gotAuth != want {
		t.Errorf("server saw Authorization = %q, want %q", gotAuth, want)
	}
}

// TestRemoteAgent_AuthClientProviderScopeRemediation covers the remediation
// NewA2A points at when Auth and ClientProvider are combined: the caller wires
// their own interceptor and attaches the exported CredentialScope themselves.
// With Auth unset this package touches the context not at all, so the scope has
// to come from the caller for the interceptor to resolve anything.
func TestRemoteAgent_AuthClientProviderScopeRemediation(t *testing.T) {
	var mu sync.Mutex
	var gotAuth string
	srv := serveRecordingA2A(t, func(r *http.Request) {
		mu.Lock()
		gotAuth = r.Header.Get("Authorization")
		mu.Unlock()
	}, a2a.NewMessage(a2a.MessageRoleAgent, a2a.NewTextPart("ok")))

	ictx := newInvocationContextFor(t, t.Name(), "dave", "default")
	scope := CredentialScope(ictx.Session(), "a2a")
	store := a2aclient.NewInMemoryCredentialsStore()
	store.Set(scope, "bearer", "own-token")
	factory := a2aclient.NewFactory(a2aclient.WithCallInterceptors(&a2aclient.AuthInterceptor{Service: store}))

	remoteAgent, err := NewA2A(A2AConfig{
		Name:      "a2a",
		AgentCard: bearerCard(srv.URL),
		ClientProvider: func(ctx context.Context, card *a2a.AgentCard) (A2AClient, error) {
			client, err := NewA2AClientProvider(factory)(ctx, card)
			if err != nil {
				return nil, err
			}
			// The interceptor reads the context of each call, not the one the
			// provider was built with, so the scope has to go on per call.
			return scopedClient{A2AClient: client, scope: scope}, nil
		},
	})
	if err != nil {
		t.Fatalf("NewA2A() error = %v", err)
	}
	if _, err := runAndCollect(ictx, remoteAgent); err != nil {
		t.Fatalf("agent.Run() error = %v", err)
	}

	mu.Lock()
	defer mu.Unlock()
	if want := "Bearer own-token"; gotAuth != want {
		t.Errorf("server saw Authorization = %q, want %q", gotAuth, want)
	}
}

// scopedClient attaches a credential scope to every call it forwards, which is
// what a caller combining a custom ClientProvider with their own
// a2aclient.AuthInterceptor has to do.
type scopedClient struct {
	A2AClient
	scope a2aclient.SessionID
}

func (c scopedClient) SendMessage(ctx context.Context, req *a2a.SendMessageRequest) (a2a.SendMessageResult, error) {
	return c.A2AClient.SendMessage(a2aclient.AttachSessionID(ctx, c.scope), req)
}

func (c scopedClient) SendStreamingMessage(ctx context.Context, req *a2a.SendMessageRequest) iter.Seq2[a2a.Event, error] {
	return c.A2AClient.SendStreamingMessage(a2aclient.AttachSessionID(ctx, c.scope), req)
}

func (c scopedClient) CancelTask(ctx context.Context, req *a2a.CancelTaskRequest) (*a2a.Task, error) {
	return c.A2AClient.CancelTask(a2aclient.AttachSessionID(ctx, c.scope), req)
}

// TestRemoteAgent_AuthUnsetLeavesTheContextAlone pins that a caller who never
// set Auth sees the invocation context they passed in, unwrapped and with no
// credential scope on it. Attaching one would revive a dormant interceptor a
// caller had installed but never fed.
func TestRemoteAgent_AuthUnsetLeavesTheContextAlone(t *testing.T) {
	ictx := newInvocationContextFor(t, t.Name(), "erin", "default")
	got := authSendContext(ictx, A2AConfig{Name: "a2a"}, bearerCard("http://example.invalid"))
	if got != context.Context(ictx) {
		t.Errorf("authSendContext() = %T, want the invocation context unchanged", got)
	}
	if sid, ok := a2aclient.SessionIDFrom(got); ok {
		t.Errorf("SessionIDFrom() = %q, true; want no scope attached when Auth is unset", sid)
	}
}

// TestRemoteAgent_AuthRefusesCrossOriginRedirect covers redirect hardening. The
// credential is attached before the first hop and Go replays headers on each
// redirect, never stripping a card-named API-key header, so the client must
// refuse to leave the card's origin.
func TestRemoteAgent_AuthRefusesCrossOriginRedirect(t *testing.T) {
	var mu sync.Mutex
	var elsewhereSawKey string
	var elsewhereHits int
	elsewhere := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		elsewhereHits++
		elsewhereSawKey = r.Header.Get("X-Card-Key")
		mu.Unlock()
		w.WriteHeader(http.StatusOK)
	}))
	defer elsewhere.Close()

	declared := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, elsewhere.URL, http.StatusTemporaryRedirect)
	}))
	defer declared.Close()

	card := newSecureCard(declared.URL,
		a2a.NamedSecuritySchemes{
			"apikey": a2a.APIKeySecurityScheme{Location: a2a.APIKeySecuritySchemeLocationHeader, Name: "X-Card-Key"},
		},
		a2a.SecurityRequirementsOptions{
			{a2a.SecuritySchemeName("apikey"): a2a.SecuritySchemeScopes{}},
		},
	)

	remoteAgent, err := NewA2A(A2AConfig{Name: "a2a", AgentCard: card, Auth: auth.APIKey("X-Card-Key", "secret")})
	if err != nil {
		t.Fatalf("NewA2A() error = %v", err)
	}
	events, err := runAndCollect(newInvocationContext(t, []*session.Event{newUserHello()}), remoteAgent)
	if err != nil {
		t.Fatalf("agent.Run() error = %v", err)
	}

	mu.Lock()
	hits, key := elsewhereHits, elsewhereSawKey
	mu.Unlock()
	if hits != 0 {
		t.Errorf("the redirect target received %d requests, want 0; it saw X-Card-Key = %q", hits, key)
	}

	errEvent := firstErrorEvent(events)
	if errEvent == nil {
		t.Fatal("want an error event from the refused redirect, got none")
	}
	if !strings.Contains(errEvent.ErrorMessage, "refusing redirect") {
		t.Errorf("error event = %q, want it to mention the refused redirect", errEvent.ErrorMessage)
	}
}

// TestSameCredentialTarget covers which redirects may carry the credential.
// The refusal test below only exercises the cross-origin case, so without this
// inverting the predicate would keep the suite green while breaking every
// same-origin redirect.
func TestSameCredentialTarget(t *testing.T) {
	tests := []struct {
		name     string
		from, to string
		want     bool
	}{
		{name: "identical", from: "https://h/rpc", to: "https://h/other", want: true},
		{name: "default port made explicit", from: "https://h/rpc", to: "https://h:443/rpc", want: true},
		{name: "default http port made explicit", from: "http://h/rpc", to: "http://h:80/rpc", want: true},
		{name: "host case differs", from: "https://Host/rpc", to: "https://host/rpc", want: true},
		{name: "scheme upgraded to https, same explicit port", from: "http://h:8080/rpc", to: "https://h:8080/rpc", want: true},
		{name: "scheme upgraded to https, both ports implicit", from: "http://h/rpc", to: "https://h/rpc", want: true},
		{name: "scheme upgraded to https, explicit default ports", from: "http://h:80/rpc", to: "https://h:443/rpc", want: true},
		{name: "scheme upgraded to https onto a different port", from: "http://h/rpc", to: "https://h:9443/rpc", want: false},
		{name: "scheme downgraded with implicit ports", from: "https://h/rpc", to: "http://h/rpc", want: false},
		{name: "unrelated scheme", from: "https://h/rpc", to: "ftp://h/rpc", want: false},
		{name: "scheme downgraded to http", from: "https://h/rpc", to: "http://h/rpc", want: false},
		{name: "different host", from: "https://h/rpc", to: "https://other/rpc", want: false},
		{name: "different port", from: "https://h:8443/rpc", to: "https://h:9443/rpc", want: false},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			from, err := url.Parse(tc.from)
			if err != nil {
				t.Fatalf("url.Parse(%q) error = %v", tc.from, err)
			}
			to, err := url.Parse(tc.to)
			if err != nil {
				t.Fatalf("url.Parse(%q) error = %v", tc.to, err)
			}
			if got := sameCredentialTarget(from, to); got != tc.want {
				t.Errorf("sameCredentialTarget(%q, %q) = %v, want %v", tc.from, tc.to, got, tc.want)
			}
		})
	}
}

// TestAuthHTTPClientRedirectCap covers the redirect limit. A custom
// CheckRedirect replaces net/http's default cap rather than extending it, so a
// same-origin redirect loop would otherwise spin forever.
func TestAuthHTTPClientRedirectCap(t *testing.T) {
	var hits atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		http.Redirect(w, r, r.URL.Path, http.StatusTemporaryRedirect)
	}))
	defer srv.Close()

	_, err := authHTTPClient().Get(srv.URL)
	if err == nil {
		t.Fatal("Get() = nil error, want the redirect cap to stop the loop")
	}
	if !strings.Contains(err.Error(), "stopped after") {
		t.Errorf("Get() error = %v, want it to report the redirect cap", err)
	}
	if got := hits.Load(); got != maxRedirects {
		t.Errorf("server saw %d requests, want %d", got, maxRedirects)
	}
}

// TestCheckRedirect covers the redirect policy as a whole, including a chain
// every hop of which looks safe against the original request. Only comparing
// against the previous hop as well catches http climbing to https and dropping
// back down, which would put the credential on the wire in cleartext again.
func TestCheckRedirect(t *testing.T) {
	chain := func(urls ...string) (*http.Request, []*http.Request) {
		t.Helper()
		reqs := make([]*http.Request, len(urls))
		for i, u := range urls {
			r, err := http.NewRequest(http.MethodGet, u, nil)
			if err != nil {
				t.Fatalf("http.NewRequest(%q) error = %v", u, err)
			}
			reqs[i] = r
		}
		return reqs[len(reqs)-1], reqs[:len(reqs)-1]
	}
	tests := []struct {
		name    string
		urls    []string
		wantErr string
	}{
		{name: "same origin", urls: []string{"https://h/a", "https://h/b"}},
		{name: "upgrade to https", urls: []string{"http://h/a", "https://h/a"}},
		{name: "two same-origin hops", urls: []string{"https://h/a", "https://h/b", "https://h/c"}},
		{name: "cross host", urls: []string{"https://h/a", "https://other/a"}, wantErr: "refusing redirect"},
		{name: "downgrade", urls: []string{"https://h/a", "http://h/a"}, wantErr: "refusing redirect"},
		{
			// Each hop is same-origin against the original http request, so
			// only the previous-hop check rejects the last one.
			name:    "upgrade then downgrade",
			urls:    []string{"http://h/a", "https://h/b", "http://h/c"},
			wantErr: "refusing redirect",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			req, via := chain(tc.urls...)
			err := checkRedirect(req, via)
			if tc.wantErr == "" {
				if err != nil {
					t.Errorf("checkRedirect(%v) error = %v, want nil", tc.urls, err)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
				t.Errorf("checkRedirect(%v) error = %v, want it to contain %q", tc.urls, err, tc.wantErr)
			}
		})
	}
}

// TestRemoteAgent_AuthAllowsSameOriginRedirect is the counterpart to the
// refusal test: a redirect the guard should permit must still reach the server
// with the credential attached.
func TestRemoteAgent_AuthAllowsSameOriginRedirect(t *testing.T) {
	var mu sync.Mutex
	var gotAuth string
	var redirected bool
	inner := a2asrv.NewJSONRPCHandler(a2asrv.NewHandler(newA2AEventReplay(t, []a2a.Event{
		a2a.NewMessage(a2a.MessageRoleAgent, a2a.NewTextPart("ok")),
	})))
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/rpc" {
			http.Redirect(w, r, "/rpc", http.StatusTemporaryRedirect)
			return
		}
		mu.Lock()
		gotAuth, redirected = r.Header.Get("Authorization"), true
		mu.Unlock()
		inner.ServeHTTP(w, r)
	}))
	defer srv.Close()

	card := bearerCard(srv.URL + "/start")
	remoteAgent, err := NewA2A(A2AConfig{Name: "a2a", AgentCard: card, Auth: auth.StaticToken("secret-token")})
	if err != nil {
		t.Fatalf("NewA2A() error = %v", err)
	}
	events, err := runAndCollect(newInvocationContext(t, []*session.Event{newUserHello()}), remoteAgent)
	if err != nil {
		t.Fatalf("agent.Run() error = %v", err)
	}
	if e := firstErrorEvent(events); e != nil {
		t.Fatalf("unexpected error event: %q", e.ErrorMessage)
	}

	mu.Lock()
	defer mu.Unlock()
	if !redirected {
		t.Fatal("the redirect target never received the request")
	}
	if want := "Bearer secret-token"; gotAuth != want {
		t.Errorf("after the redirect the server saw Authorization = %q, want %q", gotAuth, want)
	}
}

// TestRemoteAgent_AuthPreservesCallerScope covers a caller who attached their
// own a2aclient session id before the runner and wired their own interceptor.
// Overwriting their key would silently drop their credential, which is the one
// pre-existing way this path could have been used.
func TestRemoteAgent_AuthPreservesCallerScope(t *testing.T) {
	var mu sync.Mutex
	var gotAuth string
	srv := serveRecordingA2A(t, func(r *http.Request) {
		mu.Lock()
		gotAuth = r.Header.Get("Authorization")
		mu.Unlock()
	}, a2a.NewMessage(a2a.MessageRoleAgent, a2a.NewTextPart("ok")))

	store := a2aclient.NewInMemoryCredentialsStore()
	store.Set("my-tenant", "bearer", "tenant-token")
	factory := a2aclient.NewFactory(a2aclient.WithCallInterceptors(&a2aclient.AuthInterceptor{Service: store}))

	remoteAgent, err := NewA2A(A2AConfig{
		Name:           "a2a",
		AgentCard:      bearerCard(srv.URL),
		ClientProvider: NewA2AClientProvider(factory),
	})
	if err != nil {
		t.Fatalf("NewA2A() error = %v", err)
	}

	ictx := newInvocationContextFor(t, t.Name(), "frank", "default")
	scoped := ictx.WithContext(a2aclient.AttachSessionID(ictx, "my-tenant"))
	if _, err := runAndCollect(scoped, remoteAgent); err != nil {
		t.Fatalf("agent.Run() error = %v", err)
	}

	mu.Lock()
	defer mu.Unlock()
	if want := "Bearer tenant-token"; gotAuth != want {
		t.Errorf("server saw Authorization = %q, want %q; the caller's own scope must survive", gotAuth, want)
	}
}

// TestRemoteAgent_AuthCleanupKeepsInvocationContext covers the provider pattern
// the Auth doc invites — recovering the ADK context by type assertion — on the
// cleanup path, where the context is detached from the invocation and rebounded.
// A provider that fails there sends CancelTask unauthenticated and leaks the
// remote task.
func TestRemoteAgent_AuthCleanupKeepsInvocationContext(t *testing.T) {
	executor := &mockA2AExecutor{
		executeFn: func(ctx context.Context, reqCtx *a2asrv.ExecutorContext) iter.Seq2[a2a.Event, error] {
			return func(yield func(a2a.Event, error) bool) {
				if !yield(a2a.NewSubmittedTask(reqCtx, reqCtx.Message), nil) {
					return
				}
				if !yield(a2a.NewArtifactEvent(reqCtx, a2a.NewDataPart(map[string]any{"foo": "bar"})), nil) {
					return
				}
				<-ctx.Done()
			}
		},
		cancelFn: func(ctx context.Context, reqCtx *a2asrv.ExecutorContext) iter.Seq2[a2a.Event, error] {
			return func(yield func(a2a.Event, error) bool) {
				yield(a2a.NewStatusUpdateEvent(reqCtx, a2a.TaskStateCanceled, nil), nil)
			}
		},
	}
	inner := a2asrv.NewJSONRPCHandler(a2asrv.NewHandler(executor))

	var mu sync.Mutex
	authByMethod := make(map[string]string)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		method := jsonRPCMethod(r)
		mu.Lock()
		authByMethod[method] = r.Header.Get("Authorization")
		mu.Unlock()
		inner.ServeHTTP(w, r)
	}))
	defer srv.Close()

	// Resolves from the ADK context rather than the scope, so it fails outright
	// if the cleanup context is no longer an agent.InvocationContext.
	perUser := auth.ProviderFunc(func(ctx context.Context) (auth.Credential, error) {
		ictx, ok := ctx.(agent.InvocationContext)
		if !ok {
			return nil, fmt.Errorf("context is %T, not an agent.InvocationContext", ctx)
		}
		return auth.BearerCredential{Token: "tok-" + ictx.Session().UserID()}, nil
	})

	remoteAgent, err := NewA2A(A2AConfig{Name: "a2a", AgentCard: bearerCard(srv.URL), Auth: perUser})
	if err != nil {
		t.Fatalf("NewA2A() error = %v", err)
	}
	for _, err := range remoteAgent.Run(newInvocationContextFor(t, t.Name(), "frank", "default")) {
		if err != nil {
			t.Fatalf("agent.Run() error = %v", err)
		}
		break
	}

	mu.Lock()
	defer mu.Unlock()
	got, ok := authByMethod["CancelTask"]
	if !ok {
		t.Fatalf("no CancelTask reached the server; cleanup did not run (saw %v)", authByMethod)
	}
	if want := "Bearer tok-frank"; got != want {
		t.Errorf("cleanup CancelTask Authorization = %q, want %q", got, want)
	}
}

// TestRedactTokenError pins that the token endpoint's response body does not
// reach the error string, which the a2a interceptor logs at ERROR level, while
// the error chain still resolves for a caller inspecting it.
func TestRedactTokenError(t *testing.T) {
	const body = "sensitive-echo-of-the-request"
	tests := []struct {
		name string
		// errorCode empty is the branch where RetrieveError.Error() prints the
		// body verbatim; a named code prints only the code.
		errorCode string
		wantKeep  []string
	}{
		{name: "malformed error response carries the body", errorCode: "", wantKeep: []string{"400 Bad Request"}},
		{name: "well-formed error response names a code", errorCode: "invalid_grant", wantKeep: []string{"invalid_grant"}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			src := tokenSourceFunc(func() (*oauth2.Token, error) {
				return nil, &oauth2.RetrieveError{
					Response:  &http.Response{Status: "400 Bad Request", StatusCode: http.StatusBadRequest},
					Body:      []byte(body),
					ErrorCode: tc.errorCode,
				}
			})
			_, err := mintAccessToken(t.Context(), src)
			if err == nil {
				t.Fatal("mintAccessToken() = nil error, want the token source error")
			}
			if strings.Contains(err.Error(), body) {
				t.Errorf("mintAccessToken() error = %v, want the response body redacted", err)
			}
			for _, want := range tc.wantKeep {
				if !strings.Contains(err.Error(), want) {
					t.Errorf("mintAccessToken() error = %v, want it to keep %q", err, want)
				}
			}
			var re *oauth2.RetrieveError
			if !errors.As(err, &re) {
				t.Errorf("mintAccessToken() error = %v, want errors.As to still find *oauth2.RetrieveError", err)
			}
		})
	}
}

// TestMintAccessTokenHasItsOwnBudget covers a caller context with no deadline,
// which is the ordinary case: the runner sets none on an invocation, so without
// an internal budget a hung token endpoint would stall the whole run.
func TestMintAccessTokenHasItsOwnBudget(t *testing.T) {
	prev := mintTimeout
	mintTimeout = 50 * time.Millisecond
	t.Cleanup(func() { mintTimeout = prev })

	release := make(chan struct{})
	t.Cleanup(func() { close(release) })
	blocked := tokenSourceFunc(func() (*oauth2.Token, error) {
		<-release
		return &oauth2.Token{AccessToken: "late"}, nil
	})

	done := make(chan error, 1)
	go func() {
		_, err := mintAccessToken(context.WithoutCancel(t.Context()), blocked)
		done <- err
	}()
	select {
	case err := <-done:
		if !errors.Is(err, context.DeadlineExceeded) {
			t.Errorf("mintAccessToken() error = %v, want it to wrap %v", err, context.DeadlineExceeded)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("mintAccessToken() did not return; a deadline-free caller context left the mint unbounded")
	}
}

// TestAuthContextDerivation covers the two ways an agent.InvocationContext can
// be derived. Both must keep the credential scope and the agent card, and
// WithICDelta must not leave the wrapper's cancellation disagreeing with the
// invocation context it wraps.
func TestAuthContextDerivation(t *testing.T) {
	card := bearerCard("http://example.invalid")
	ictx := newInvocationContextFor(t, t.Name(), "gina", "default")
	cfg := A2AConfig{Name: "a2a", Auth: auth.StaticToken("tok")}
	sendCtx, ok := authSendContext(ictx, cfg, card).(agent.InvocationContext)
	if !ok {
		t.Fatalf("authSendContext() = %T, want an agent.InvocationContext", sendCtx)
	}
	wantScope := CredentialScope(ictx.Session(), cfg.Name)

	t.Run("WithContext carries the scope and card", func(t *testing.T) {
		got := sendCtx.WithContext(context.Background())
		if sid, ok := a2aclient.SessionIDFrom(got); !ok || sid != wantScope {
			t.Errorf("SessionIDFrom(WithContext(...)) = %q, %v, want %q, true", sid, ok, wantScope)
		}
		if iremoteagent.AgentCardFrom(got) != card {
			t.Error("WithContext() dropped the agent card")
		}
	})

	t.Run("WithICDelta honours a replacement context", func(t *testing.T) {
		inner, cancel := context.WithCancel(context.Background())
		got := sendCtx.WithICDelta(&agent.InvocationContextDelta{Context: &inner})
		if sid, ok := a2aclient.SessionIDFrom(got); !ok || sid != wantScope {
			t.Errorf("SessionIDFrom(WithICDelta(...)) = %q, %v, want %q, true", sid, ok, wantScope)
		}
		cancel()
		select {
		case <-got.Done():
		default:
			t.Error("WithICDelta() ignored the replacement context; cancelling it left the derived context live")
		}
		if !errors.Is(got.Err(), context.Canceled) {
			t.Errorf("Err() = %v, want %v", got.Err(), context.Canceled)
		}
	})
}

// TestRemoteAgent_AuthMismatchWarning observes the log rather than the
// predicate behind it. A card offering an alternative the credential does fit
// is correct operation and must stay quiet; a card no scheme of which can carry
// it is a caller misconfiguration nothing else reports, and must warn — once,
// not once per scheme per request.
func TestRemoteAgent_AuthMismatchWarning(t *testing.T) {
	apiKeyOnly := func(url string) *a2a.AgentCard {
		return newSecureCard(url,
			a2a.NamedSecuritySchemes{"apikey": a2a.APIKeySecurityScheme{Location: a2a.APIKeySecuritySchemeLocationHeader, Name: "X-Api-Key"}},
			a2a.SecurityRequirementsOptions{{a2a.SecuritySchemeName("apikey"): a2a.SecuritySchemeScopes{}}},
		)
	}
	alternatives := func(url string) *a2a.AgentCard {
		return newSecureCard(url,
			a2a.NamedSecuritySchemes{
				"apikey": a2a.APIKeySecurityScheme{Location: a2a.APIKeySecuritySchemeLocationHeader, Name: "X-Api-Key"},
				"bearer": a2a.HTTPAuthSecurityScheme{Scheme: "Bearer"},
			},
			// Two alternative requirement objects: the interceptor tries the
			// api key first about half the time and then falls through.
			a2a.SecurityRequirementsOptions{
				{a2a.SecuritySchemeName("apikey"): a2a.SecuritySchemeScopes{}},
				{a2a.SecuritySchemeName("bearer"): a2a.SecuritySchemeScopes{}},
			},
		)
	}
	tests := []struct {
		name     string
		card     func(string) *a2a.AgentCard
		wantAuth string
		wantWarn int
	}{
		{name: "card offers an alternative that fits", card: alternatives, wantAuth: "Bearer secret-token", wantWarn: 0},
		{name: "no scheme on the card can carry it", card: apiKeyOnly, wantAuth: "", wantWarn: 1},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var mu sync.Mutex
			var gotAuth string
			srv := serveRecordingA2A(t, func(r *http.Request) {
				mu.Lock()
				gotAuth = r.Header.Get("Authorization")
				mu.Unlock()
			}, a2a.NewMessage(a2a.MessageRoleAgent, a2a.NewTextPart("ok")))

			remoteAgent, err := NewA2A(A2AConfig{Name: "a2a", AgentCard: tc.card(srv.URL), Auth: auth.StaticToken("secret-token")})
			if err != nil {
				t.Fatalf("NewA2A() error = %v", err)
			}

			warns := &countingHandler{match: "no security scheme the agent card declares can carry"}
			// Three invocations: a per-request or per-scheme warning would
			// show up as more than one line.
			for range 3 {
				ictx := newInvocationContext(t, []*session.Event{newUserHello()})
				scoped := ictx.WithContext(log.AttachLogger(ictx, slog.New(warns)))
				if _, err := runAndCollect(scoped, remoteAgent); err != nil {
					t.Fatalf("agent.Run() error = %v", err)
				}
			}

			mu.Lock()
			defer mu.Unlock()
			if gotAuth != tc.wantAuth {
				t.Errorf("server saw Authorization = %q, want %q", gotAuth, tc.wantAuth)
			}
			if got := warns.count.Load(); got != int32(tc.wantWarn) {
				t.Errorf("mismatch warning logged %d times over 3 invocations, want %d", got, tc.wantWarn)
			}
		})
	}
}

// countingHandler counts WARN records whose message contains match.
type countingHandler struct {
	slog.Handler
	match string
	count atomic.Int32
}

func (h *countingHandler) Enabled(context.Context, slog.Level) bool { return true }

func (h *countingHandler) Handle(_ context.Context, r slog.Record) error {
	if r.Level == slog.LevelWarn && strings.Contains(r.Message, h.match) {
		h.count.Add(1)
	}
	return nil
}

func (h *countingHandler) WithAttrs([]slog.Attr) slog.Handler { return h }
func (h *countingHandler) WithGroup(string) slog.Handler      { return h }

// TestRemoteAgent_AuthConcurrentInvocations exercises the interceptor and the
// provider shared across concurrent invocations of one agent value, which is
// what a real runner does and what the -race detector needs in order to say
// anything about this path.
func TestRemoteAgent_AuthConcurrentInvocations(t *testing.T) {
	var mu sync.Mutex
	var seen []string
	srv := serveFreshA2A(t, "ok", func(r *http.Request) {
		mu.Lock()
		seen = append(seen, r.Header.Get("Authorization"))
		mu.Unlock()
	})

	perUser := auth.ProviderFunc(func(ctx context.Context) (auth.Credential, error) {
		sid, ok := a2aclient.SessionIDFrom(ctx)
		if !ok {
			return nil, errors.New("no credential scope on the context")
		}
		return auth.BearerCredential{Token: "tok-" + string(sid)}, nil
	})

	remoteAgent, err := NewA2A(A2AConfig{Name: "a2a", AgentCard: bearerCard(srv.URL), Auth: perUser})
	if err != nil {
		t.Fatalf("NewA2A() error = %v", err)
	}

	const users = 8
	contexts := make([]agent.InvocationContext, users)
	want := make([]string, users)
	for i := range users {
		user := fmt.Sprintf("user-%d", i)
		contexts[i] = newInvocationContextFor(t, t.Name(), user, "default")
		want[i] = "Bearer tok-" + string(CredentialScope(contexts[i].Session(), "a2a"))
	}

	var wg sync.WaitGroup
	errs := make([]error, users)
	for i := range users {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_, errs[i] = runAndCollect(contexts[i], remoteAgent)
		}()
	}
	wg.Wait()
	for i, err := range errs {
		if err != nil {
			t.Fatalf("agent.Run() for user %d error = %v", i, err)
		}
	}

	mu.Lock()
	defer mu.Unlock()
	slices.Sort(seen)
	slices.Sort(want)
	if !slices.Equal(seen, want) {
		t.Errorf("server saw %v, want %v; a shared provider must not cross credentials between invocations", seen, want)
	}
}

// TestRemoteAgent_AuthAttachedToCleanupCancel pins that the cleanup CancelTask is
// authenticated too, not just the message send — otherwise it goes out
// unauthenticated, is rejected, and leaks the task.
func TestRemoteAgent_AuthAttachedToCleanupCancel(t *testing.T) {
	executor := &mockA2AExecutor{
		// Submit a task and stream one artifact, then stay non-terminal until the
		// client stops consuming (the run loop breaks mid-task), so its deferred
		// cleanup must CancelTask. Block on cancellation — the same signal the
		// old loop polled via ctx.Err() — instead of spinning.
		executeFn: func(ctx context.Context, reqCtx *a2asrv.ExecutorContext) iter.Seq2[a2a.Event, error] {
			return func(yield func(a2a.Event, error) bool) {
				if !yield(a2a.NewSubmittedTask(reqCtx, reqCtx.Message), nil) {
					return
				}
				data := a2a.NewDataPart(map[string]any{"foo": "bar"})
				if !yield(a2a.NewArtifactEvent(reqCtx, data), nil) {
					return
				}
				<-ctx.Done()
			}
		},
		cancelFn: func(ctx context.Context, reqCtx *a2asrv.ExecutorContext) iter.Seq2[a2a.Event, error] {
			return func(yield func(a2a.Event, error) bool) {
				yield(a2a.NewStatusUpdateEvent(reqCtx, a2a.TaskStateCanceled, nil), nil)
			}
		},
	}
	inner := a2asrv.NewJSONRPCHandler(a2asrv.NewHandler(executor))

	var mu sync.Mutex
	authByMethod := make(map[string]string) // JSON-RPC method -> Authorization header
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		method := jsonRPCMethod(r)
		mu.Lock()
		authByMethod[method] = r.Header.Get("Authorization")
		mu.Unlock()
		inner.ServeHTTP(w, r)
	}))
	defer srv.Close()

	remoteAgent, err := NewA2A(A2AConfig{Name: "a2a", AgentCard: bearerCard(srv.URL), Auth: auth.StaticToken("secret-token")})
	if err != nil {
		t.Fatalf("NewA2A() error = %v", err)
	}

	// Stop consuming after the first event so the run loop returns mid-task; its
	// deferred cleanup issues CancelTask synchronously before the range ends.
	for _, err := range remoteAgent.Run(newInvocationContext(t, []*session.Event{newUserHello()})) {
		if err != nil {
			t.Fatalf("agent.Run() error = %v", err)
		}
		break
	}

	mu.Lock()
	defer mu.Unlock()
	got, ok := authByMethod["CancelTask"]
	if !ok {
		t.Fatalf("no CancelTask request reached the server; cleanup did not run (saw %v)", authByMethod)
	}
	if got != "Bearer secret-token" {
		t.Errorf("cleanup CancelTask Authorization = %q, want %q", got, "Bearer secret-token")
	}
}

// serveRecordingA2A starts a JSON-RPC A2A test server that replays events and
// invokes record for every incoming request, so tests can inspect auth headers.
func serveRecordingA2A(t *testing.T, record func(*http.Request), events ...a2a.Event) *httptest.Server {
	t.Helper()
	inner := a2asrv.NewJSONRPCHandler(a2asrv.NewHandler(newA2AEventReplay(t, events)))
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		record(r)
		inner.ServeHTTP(w, r)
	}))
	t.Cleanup(srv.Close)
	return srv
}

// serveFreshA2A is serveRecordingA2A's concurrency-safe twin. newA2AEventReplay
// stamps the per-request task and context ids into one shared event slice, so
// concurrent invocations would race on it; this one builds a reply per request.
func serveFreshA2A(t *testing.T, reply string, record func(*http.Request)) *httptest.Server {
	t.Helper()
	executor := &mockA2AExecutor{
		executeFn: func(ctx context.Context, execCtx *a2asrv.ExecutorContext) iter.Seq2[a2a.Event, error] {
			return func(yield func(a2a.Event, error) bool) {
				msg := a2a.NewMessage(a2a.MessageRoleAgent, a2a.NewTextPart(reply))
				msg.TaskID = execCtx.TaskID
				msg.ContextID = execCtx.ContextID
				yield(msg, nil)
			}
		},
	}
	inner := a2asrv.NewJSONRPCHandler(a2asrv.NewHandler(executor))
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		record(r)
		inner.ServeHTTP(w, r)
	}))
	t.Cleanup(srv.Close)
	return srv
}

// newInvocationContextFor builds an invocation context on a session with an
// explicit identity, so tests can vary one part of the (app, user, session)
// triple at a time.
func newInvocationContextFor(t *testing.T, appName, userID, sessionID string) agent.InvocationContext {
	t.Helper()
	ctx := t.Context()
	service := session.InMemoryService()
	resp, err := service.Create(ctx, &session.CreateRequest{AppName: appName, UserID: userID, SessionID: sessionID})
	if err != nil {
		t.Fatalf("sessionService.Create() error = %v", err)
	}
	if err := service.AppendEvent(ctx, resp.Session, newUserHello()); err != nil {
		t.Fatalf("sessionService.AppendEvent() error = %v", err)
	}
	return icontext.NewInvocationContext(ctx, icontext.InvocationContextParams{
		Session:   resp.Session,
		RunConfig: &agent.RunConfig{StreamingMode: agent.StreamingModeSSE},
	})
}

// newSecureCard builds an agent card pointing at url that declares the given
// security schemes and requirements, so the a2a AuthInterceptor attaches a
// credential (it is a no-op unless the card carries a requirement).
func newSecureCard(url string, schemes a2a.NamedSecuritySchemes, reqs a2a.SecurityRequirementsOptions) *a2a.AgentCard {
	return &a2a.AgentCard{
		Name:                 "a2a",
		SupportedInterfaces:  []*a2a.AgentInterface{a2a.NewAgentInterface(url, a2a.TransportProtocolJSONRPC)},
		Capabilities:         a2a.AgentCapabilities{Streaming: true},
		SecuritySchemes:      schemes,
		SecurityRequirements: reqs,
	}
}

// bearerCard is a card whose single security scheme is HTTP Bearer.
func bearerCard(url string) *a2a.AgentCard {
	return newSecureCard(url,
		a2a.NamedSecuritySchemes{"bearer": a2a.HTTPAuthSecurityScheme{Scheme: "Bearer"}},
		a2a.SecurityRequirementsOptions{{a2a.SecuritySchemeName("bearer"): a2a.SecuritySchemeScopes{}}},
	)
}

func firstErrorEvent(events []*session.Event) *session.Event {
	for _, e := range events {
		if e.ErrorMessage != "" {
			return e
		}
	}
	return nil
}

func eventsContainText(events []*session.Event, want string) bool {
	for _, e := range events {
		if e.LLMResponse.Content == nil {
			continue
		}
		for _, p := range e.LLMResponse.Content.Parts {
			if strings.Contains(p.Text, want) {
				return true
			}
		}
	}
	return false
}

// jsonRPCMethod peeks the JSON-RPC method of an incoming request and restores
// the body so the wrapped handler can still read it.
func jsonRPCMethod(r *http.Request) string {
	body, err := io.ReadAll(r.Body)
	if err != nil {
		return ""
	}
	r.Body = io.NopCloser(bytes.NewReader(body))
	var rpc struct {
		Method string `json:"method"`
	}
	_ = json.Unmarshal(body, &rpc)
	return rpc.Method
}
