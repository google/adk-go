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
	"io"
	"iter"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/a2aproject/a2a-go/v2/a2a"
	"github.com/a2aproject/a2a-go/v2/a2aclient"
	"github.com/a2aproject/a2a-go/v2/a2asrv"
	"golang.org/x/oauth2"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/auth"
	"google.golang.org/adk/v2/session"
)

func TestCredentialsServiceGet(t *testing.T) {
	tests := []struct {
		name     string
		provider auth.CredentialProvider
		want     a2aclient.AuthCredential
		wantErr  bool
	}{
		{
			name:     "static bearer token",
			provider: auth.StaticToken("tok"),
			want:     "tok",
		},
		{
			name:     "api key value",
			provider: auth.APIKey("X-Api-Key", "secret"),
			want:     "secret",
		},
		{
			name:     "oauth2 token source",
			provider: auth.TokenSourceProvider(oauth2.StaticTokenSource(&oauth2.Token{AccessToken: "at"})),
			want:     "at",
		},
		{
			name: "oauth2 missing token source",
			provider: auth.ProviderFunc(func(context.Context) (auth.Credential, error) {
				return auth.OAuth2Credential{}, nil
			}),
			wantErr: true,
		},
		{
			name:     "oauth2 empty access token",
			provider: auth.TokenSourceProvider(oauth2.StaticTokenSource(&oauth2.Token{AccessToken: ""})),
			wantErr:  true,
		},
		{
			name:     "empty api key value",
			provider: auth.APIKey("X-Api-Key", ""),
			wantErr:  true,
		},
		{
			name: "nil credential",
			provider: auth.ProviderFunc(func(context.Context) (auth.Credential, error) {
				return nil, nil
			}),
			wantErr: true,
		},
		{
			name: "unsupported credential kind",
			provider: auth.ProviderFunc(func(context.Context) (auth.Credential, error) {
				return auth.BasicCredential{Username: "u", Password: "p"}, nil
			}),
			wantErr: true,
		},
		{
			name: "provider error",
			provider: auth.ProviderFunc(func(context.Context) (auth.Credential, error) {
				return nil, errors.New("boom")
			}),
			wantErr: true,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			svc := credentialsService{provider: tc.provider}
			got, err := svc.Get(t.Context(), a2aclient.SessionID("sid"), a2a.SecuritySchemeName("scheme"))
			if tc.wantErr {
				if err == nil {
					t.Fatal("Get() = nil error, want error")
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

	failing := auth.ProviderFunc(func(context.Context) (auth.Credential, error) {
		return nil, errors.New("resolve failed")
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
}

// TestRemoteAgent_AuthAttachesAPIKeyHeader covers the apiKey scheme: the raw key
// goes in the card's named header, not "Authorization".
func TestRemoteAgent_AuthAttachesAPIKeyHeader(t *testing.T) {
	var mu sync.Mutex
	var gotKey, gotAuth string
	srv := serveRecordingA2A(t, func(r *http.Request) {
		mu.Lock()
		gotKey = r.Header.Get("X-Api-Key")
		gotAuth = r.Header.Get("Authorization")
		mu.Unlock()
	}, a2a.NewMessage(a2a.MessageRoleAgent, a2a.NewTextPart("ok")))

	card := newSecureCard(srv.URL,
		a2a.NamedSecuritySchemes{
			"apikey": a2a.APIKeySecurityScheme{Location: a2a.APIKeySecuritySchemeLocationHeader, Name: "X-Api-Key"},
		},
		a2a.SecurityRequirementsOptions{
			{a2a.SecuritySchemeName("apikey"): a2a.SecuritySchemeScopes{}},
		},
	)

	remoteAgent, err := NewA2A(A2AConfig{Name: "a2a", AgentCard: card, Auth: auth.APIKey("X-Api-Key", "secret")})
	if err != nil {
		t.Fatalf("NewA2A() error = %v", err)
	}

	ictx := newInvocationContext(t, []*session.Event{newUserHello()})
	if _, err := runAndCollect(ictx, remoteAgent); err != nil {
		t.Fatalf("agent.Run() error = %v", err)
	}

	mu.Lock()
	defer mu.Unlock()
	if gotKey != "secret" {
		t.Errorf("server saw X-Api-Key = %q, want %q", gotKey, "secret")
	}
	if gotAuth != "" {
		t.Errorf("server saw Authorization = %q, want empty (apiKey uses its own header)", gotAuth)
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

// TestRemoteAgent_AuthScopedPerSession verifies the run loop attaches the ADK
// session id to each outgoing call, so a session-aware provider resolves a
// distinct credential per session.
func TestRemoteAgent_AuthScopedPerSession(t *testing.T) {
	var mu sync.Mutex
	var gotAuth string
	srv := serveRecordingA2A(t, func(r *http.Request) {
		mu.Lock()
		gotAuth = r.Header.Get("Authorization")
		mu.Unlock()
	}, a2a.NewMessage(a2a.MessageRoleAgent, a2a.NewTextPart("ok")))

	// Mints a token from the attached session id, proving the id reached the provider.
	perSession := auth.ProviderFunc(func(ctx context.Context) (auth.Credential, error) {
		sid, _ := a2aclient.SessionIDFrom(ctx)
		return auth.BearerCredential{Token: "tok-" + string(sid)}, nil
	})

	remoteAgent, err := NewA2A(A2AConfig{Name: "a2a", AgentCard: bearerCard(srv.URL), Auth: perSession})
	if err != nil {
		t.Fatalf("NewA2A() error = %v", err)
	}

	seen := make(map[string]string) // session id -> Authorization header
	for range 2 {
		ictx := newInvocationContext(t, []*session.Event{newUserHello()})

		mu.Lock()
		gotAuth = ""
		mu.Unlock()

		if _, err := runAndCollect(ictx, remoteAgent); err != nil {
			t.Fatalf("agent.Run() error = %v", err)
		}

		mu.Lock()
		got := gotAuth
		mu.Unlock()

		sid := ictx.Session().ID()
		if want := "Bearer tok-" + sid; got != want {
			t.Errorf("session %s: server saw Authorization = %q, want %q", sid, got, want)
		}
		seen[sid] = got
	}
	if len(seen) != 2 {
		t.Fatalf("want 2 distinct sessions, got %d: %v", len(seen), seen)
	}
	distinct := make(map[string]bool)
	for _, v := range seen {
		distinct[v] = true
	}
	if len(distinct) != 2 {
		t.Errorf("want 2 distinct credentials across sessions, got %v", seen)
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
