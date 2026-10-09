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
	"strconv"
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
	agentinternal "google.golang.org/adk/v2/internal/agent"
	iremoteagent "google.golang.org/adk/v2/internal/agent/remoteagent"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/session"
)

// errResolve is the provider failure asserted on with errors.Is, so the %w
// wrapping in the auth transport stays load-bearing.
var errResolve = errors.New("resolve failed")

// tokenSourceFunc adapts a function to an [oauth2.TokenSource].
type tokenSourceFunc func() (*oauth2.Token, error)

func (f tokenSourceFunc) Token() (*oauth2.Token, error) { return f() }

// TestCredentialScope pins the documented key format with literal strings. The
// end-to-end tests can only show that two identities differ, which every
// weaker scope also satisfies. Several tests pin the four parts against a
// literal, but only this one rejects dropping the percent-encoding that keeps
// "/" from being ambiguous.
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

// TestCredentialScopeExported pins the exported wrapper against a literal, with
// all four parts distinct. Anything computed from the wrapper agrees with it by
// construction, so only a literal can catch the two mutations that matter: a
// swap between the user and session arguments desynchronizes this path from the
// scope server/adka2a/v2 builds from the internal function, and dropping the
// agent name reinstates the cross-agent token leak it is there to prevent.
func TestCredentialScopeExported(t *testing.T) {
	ictx := newInvocationContextFor(t, "shop", "alice", "s1")
	if got, want := CredentialScope(ictx.Session(), "crm"), a2aclient.SessionID("shop/alice/s1/crm"); got != want {
		t.Errorf("CredentialScope() = %q, want %q", got, want)
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
		_, err := newMintGroup().token(ctx, "sid", blocked)
		done <- err
	}()

	select {
	case err := <-done:
		if !errors.Is(err, context.DeadlineExceeded) {
			t.Errorf("mintGroup.token() error = %v, want it to wrap %v", err, context.DeadlineExceeded)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("mintGroup.token() did not return once the context expired")
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
			if tc.wantErr {
				if err == nil {
					t.Fatal("want an error from the rejected request, got none")
				}
				if !strings.Contains(err.Error(), "401") {
					t.Errorf("agent.Run() error = %q, want it to mention 401", err)
				}
				return
			}
			if err != nil {
				t.Fatalf("agent.Run() error = %v", err)
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
	// A task that stays open, so breaking out of the run makes the cleanup
	// issue a CancelTask. That call is the one whose context is a plain
	// detached one rather than an agent.InvocationContext, which is why the
	// remediation has to capture the scope when the client is built.
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
	authByMethod := map[string]string{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		method := jsonRPCMethod(r)
		mu.Lock()
		authByMethod[method] = r.Header.Get("Authorization")
		mu.Unlock()
		inner.ServeHTTP(w, r)
	}))
	defer srv.Close()

	ictx := newInvocationContextFor(t, t.Name(), "dave", "default")
	store := a2aclient.NewInMemoryCredentialsStore()
	store.Set(CredentialScope(ictx.Session(), "a2a"), "bearer", "own-token")
	factory := a2aclient.NewFactory(a2aclient.WithCallInterceptors(&a2aclient.AuthInterceptor{Service: store}))

	// Exactly what NewA2A's error tells a caller to do: compute the scope from
	// the context the provider receives, and attach it on every call.
	remoteAgent, err := NewA2A(A2AConfig{
		Name:      "a2a",
		AgentCard: bearerCard(srv.URL),
		ClientProvider: func(ctx context.Context, card *a2a.AgentCard) (A2AClient, error) {
			invocation, ok := ctx.(agent.InvocationContext)
			if !ok {
				return nil, fmt.Errorf("ClientProvider got %T, want the agent.InvocationContext the error message promises", ctx)
			}
			client, err := NewA2AClientProvider(factory)(ctx, card)
			if err != nil {
				return nil, err
			}
			return scopedClient{A2AClient: client, scope: CredentialScope(invocation.Session(), "a2a")}, nil
		},
	})
	if err != nil {
		t.Fatalf("NewA2A() error = %v", err)
	}
	for _, err := range remoteAgent.Run(ictx) {
		if err != nil {
			t.Fatalf("agent.Run() error = %v", err)
		}
		break
	}

	mu.Lock()
	defer mu.Unlock()
	for _, method := range []string{"SendStreamingMessage", "CancelTask"} {
		got, ok := authByMethod[method]
		if !ok {
			t.Errorf("no %s reached the server (saw %v)", method, authByMethod)
			continue
		}
		if want := "Bearer own-token"; got != want {
			t.Errorf("%s Authorization = %q, want %q", method, got, want)
		}
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
	got := authSendContext(ictx, A2AConfig{Name: "a2a"}, nil)
	if got != context.Context(ictx) {
		t.Errorf("authSendContext() = %T, want the invocation context unchanged", got)
	}
	if sid, ok := a2aclient.SessionIDFrom(got); ok {
		t.Errorf("SessionIDFrom() = %q, true; want no scope attached when Auth is unset", sid)
	}
}

// TestRemoteAgent_AuthRefusesCrossOriginRedirect covers redirect hardening. The
// transport applies the credential again on every hop, whatever header the
// credential names, so the client must refuse to leave the card's origin.
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
	_, err = runAndCollect(newInvocationContext(t, []*session.Event{newUserHello()}), remoteAgent)

	mu.Lock()
	hits, key := elsewhereHits, elsewhereSawKey
	mu.Unlock()
	if hits != 0 {
		t.Errorf("the redirect target received %d requests, want 0; it saw X-Card-Key = %q", hits, key)
	}

	if err == nil {
		t.Fatal("want an error from the refused redirect, got none")
	}
	if !strings.Contains(err.Error(), "refusing redirect") {
		t.Errorf("agent.Run() error = %q, want it to mention the refused redirect", err)
	}
}

// TestSameCredentialTarget covers which redirects may carry the credential,
// across the scheme and port combinations the end-to-end redirect tests above
// do not reach: an upgrade to https, default ports written out or left
// implicit, and schemes that are neither http nor https.
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
		// Only http to https counts as an upgrade: either half alone is not one.
		{name: "http to an unrelated scheme", from: "http://h/rpc", to: "ftp://h/rpc", want: false},
		{name: "an unrelated scheme to https", from: "ftp://h/rpc", to: "https://h/rpc", want: false},
		// An explicit default port on one side and an implicit one on the other
		// is still each scheme's own default.
		{name: "scheme upgraded to https, explicit http default to implicit", from: "http://h:80/rpc", to: "https://h/rpc", want: true},
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

	_, err := authHTTPClient(auth.StaticToken("tok")).Get(srv.URL)
	if err == nil {
		t.Fatal("Get() = nil error, want the redirect cap to stop the loop")
	}
	if !strings.Contains(err.Error(), "stopped after") {
		t.Errorf("Get() error = %v, want it to report the redirect cap", err)
	}
	// The literal, not maxRedirects: an expectation derived from the constant
	// under test shrinks and grows with it, so the test can never fire on a
	// change to the cap. 10 is net/http's own default, which a custom
	// CheckRedirect replaces rather than extends.
	if got := hits.Load(); got != 10 {
		t.Errorf("server saw %d requests, want %d", got, 10)
	}
}

// TestAuthHTTPClientTimeout pins the request timeout against a literal.
// Supplying an explicit http.Client suppresses the one a2aclient would
// otherwise install, so this field is the only thing bounding the message send
// for every Auth user — the cleanup CancelTask has its own 5s budget and the
// card fetch its own client — and nothing else in the suite touches it.
func TestAuthHTTPClientTimeout(t *testing.T) {
	if got := authHTTPClient(auth.StaticToken("tok")).Timeout; got != 3*time.Minute {
		t.Errorf("authHTTPClient().Timeout = %v, want %v", got, 3*time.Minute)
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

// TestRemoteAgent_AuthCleanupReusesTheInvocationCredential pins that the cleanup
// CancelTask carries the credential the send already resolved rather than
// resolving a new one, as adk-python resolves once per invocation. The provider
// is called once, for the send, so its type assertion says nothing about the
// cleanup context — only that the send's context is an invocation.
func TestRemoteAgent_AuthCleanupReusesTheInvocationCredential(t *testing.T) {
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

	// Resolves from the ADK context rather than the scope, so the token names
	// the invocation's user, and a cleanup that resolved its own credential
	// would show up in the call count.
	var calls atomic.Int32
	perUser := auth.ProviderFunc(func(ctx context.Context) (auth.Credential, error) {
		calls.Add(1)
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
	if n := calls.Load(); n != 1 {
		t.Errorf("provider called %d times across the send and the cleanup, want 1", n)
	}
}

// TestRedactTokenError pins that the token endpoint's response body does not
// reach the error string, which ends up in the invocation's error event, while
// the error chain still resolves for a caller inspecting it.
func TestRedactTokenError(t *testing.T) {
	const body = "sensitive-echo-of-the-request"
	tests := []struct {
		name string
		// errorCode empty is the branch where RetrieveError.Error() prints the
		// body verbatim. A named code takes the other branch, which prints the
		// code, the description and the URI, and never the body.
		errorCode   string
		description string
		uri         string
		wantKeep    []string
		wantGone    []string
	}{
		{name: "malformed error response carries the body", errorCode: "", wantKeep: []string{"400 Bad Request"}},
		{
			name: "well-formed error response names a code", errorCode: "invalid_grant",
			description: "assertion=eyJhbGciOi-SECRET", uri: "https://idp.invalid/errors/1",
			wantKeep: []string{"invalid_grant"},
			// Free text parsed out of the body, dropped with it.
			wantGone: []string{"assertion=eyJhbGciOi-SECRET", "https://idp.invalid/errors/1"},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			src := tokenSourceFunc(func() (*oauth2.Token, error) {
				return nil, &oauth2.RetrieveError{
					Response:         &http.Response{Status: "400 Bad Request", StatusCode: http.StatusBadRequest},
					Body:             []byte(body),
					ErrorCode:        tc.errorCode,
					ErrorDescription: tc.description,
					ErrorURI:         tc.uri,
				}
			})
			_, err := newMintGroup().token(t.Context(), "sid", src)
			if err == nil {
				t.Fatal("mintGroup.token() = nil error, want the token source error")
			}
			if strings.Contains(err.Error(), body) {
				t.Errorf("mintGroup.token() error = %v, want the response body redacted", err)
			}
			for _, want := range tc.wantKeep {
				if !strings.Contains(err.Error(), want) {
					t.Errorf("mintGroup.token() error = %v, want it to keep %q", err, want)
				}
			}
			for _, gone := range tc.wantGone {
				if strings.Contains(err.Error(), gone) {
					t.Errorf("mintGroup.token() error = %v, want %q dropped", err, gone)
				}
			}
			var re *oauth2.RetrieveError
			if !errors.As(err, &re) {
				t.Errorf("mintGroup.token() error = %v, want errors.As to still find *oauth2.RetrieveError", err)
			}
		})
	}
}

// TestMintGroupRejectsANilToken covers a source that reports success and hands
// back nothing, which is distinct from the empty-access-token case: without the
// nil guard the next line dereferences it.
func TestMintGroupRejectsANilToken(t *testing.T) {
	src := tokenSourceFunc(func() (*oauth2.Token, error) { return nil, nil })
	_, err := newMintGroup().token(t.Context(), "sid", src)
	if err == nil {
		t.Fatal("mintGroup.token() = nil error, want an error for a nil token")
	}
	// Without the guard the next line dereferences it, and the recover turns
	// that into an error too — so an error alone does not distinguish the two.
	if strings.Contains(err.Error(), "panicked") {
		t.Errorf("mintGroup.token() error = %v, want a nil token rejected rather than dereferenced", err)
	}
}

// TestRedactTokenErrorLeavesOtherErrorsAlone covers the inputs redaction must
// not touch: an error that is not a RetrieveError, one with no response, and
// one with no body.
func TestRedactTokenErrorLeavesOtherErrorsAlone(t *testing.T) {
	tests := []struct {
		name string
		err  error
	}{
		{name: "not a retrieve error", err: errResolve},
		{name: "no response", err: &oauth2.RetrieveError{Body: []byte("body")}},
		{name: "no body", err: &oauth2.RetrieveError{Response: &http.Response{Status: "400 Bad Request"}}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := redactTokenError(tc.err); got != tc.err {
				t.Errorf("redactTokenError() = %v, want the error returned unchanged", got)
			}
		})
	}
}

// wrappedRetrieveError reproduces the shape that defeats a fields-based
// redaction check, without taking a dependency on the package that produces it:
// the token source behind auth.ServiceAccount with an Audience returns a
// cloud.google.com/go/auth *Error that prints the body itself, wrapping an
// *oauth2.RetrieveError whose ErrorCode the adapter parsed out of that body.
// Deciding from the inner error's fields finds a named code, declines, and lets
// the outer error print the body anyway.
type wrappedRetrieveError struct {
	inner *oauth2.RetrieveError
}

func (e *wrappedRetrieveError) Error() string {
	return fmt.Sprintf("auth: cannot fetch token: %d\nResponse: %s", e.inner.Response.StatusCode, e.inner.Body)
}
func (e *wrappedRetrieveError) Unwrap() error { return e.inner }

// quotingRetrieveError re-encodes the body instead of printing it verbatim.
// No wrapper on any path this package uses does that today, which is exactly
// why redaction must not depend on recognising the body in the message.
type quotingRetrieveError struct {
	inner *oauth2.RetrieveError
}

func (e *quotingRetrieveError) Error() string {
	return "auth: cannot fetch token: " + strconv.Quote(string(e.inner.Body))
}
func (e *quotingRetrieveError) Unwrap() error { return e.inner }

func TestRedactTokenErrorSeesThroughAReencodingWrapper(t *testing.T) {
	const body = "line-one\nassertion=eyJhbGciOi-SECRET"
	err := &quotingRetrieveError{inner: &oauth2.RetrieveError{
		Response: &http.Response{Status: "400 Bad Request", StatusCode: http.StatusBadRequest},
		Body:     []byte(body),
	}}
	if got := redactTokenError(err).Error(); strings.Contains(got, "eyJhbGciOi-SECRET") {
		t.Errorf("redactTokenError() = %q, want the body dropped even when the wrapper re-encodes it", got)
	}
}

func TestRedactTokenErrorSeesThroughAWrapper(t *testing.T) {
	const body = `{"error":"invalid_grant","error_description":"assertion=eyJhbGciOi-SECRET"}`
	err := &wrappedRetrieveError{inner: &oauth2.RetrieveError{
		Response:         &http.Response{Status: "400 Bad Request", StatusCode: http.StatusBadRequest},
		Body:             []byte(body),
		ErrorCode:        "invalid_grant",
		ErrorDescription: "assertion=eyJhbGciOi-SECRET",
	}}
	got := redactTokenError(err).Error()
	if strings.Contains(got, body) {
		t.Errorf("redactTokenError() = %q, want the verbatim response body redacted", got)
	}
	// The description is parsed out of that same body, so asserting only on
	// the whole JSON string would pass while the secret inside it survived.
	if strings.Contains(got, "eyJhbGciOi-SECRET") {
		t.Errorf("redactTokenError() = %q, want the assertion the endpoint echoed into error_description dropped too", got)
	}
	if !strings.Contains(got, "400 Bad Request") || !strings.Contains(got, "invalid_grant") {
		t.Errorf("redactTokenError() = %q, want it to keep the status and the error code", got)
	}
	var re *oauth2.RetrieveError
	if !errors.As(redactTokenError(err), &re) {
		t.Error("redactTokenError() lost the error chain")
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
		_, err := newMintGroup().token(context.WithoutCancel(t.Context()), "sid", blocked)
		done <- err
	}()
	select {
	case err := <-done:
		if !errors.Is(err, context.DeadlineExceeded) {
			t.Errorf("mintGroup.token() error = %v, want it to wrap %v", err, context.DeadlineExceeded)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("mintGroup.token() did not return; a deadline-free caller context left the mint unbounded")
	}
}

// TestAuthContextDerivation covers the two ways an agent.InvocationContext can
// be derived. Both must keep everything the auth transport reads — the scope,
// the invocation's resolved credential and the card-fetch client — and
// WithICDelta must not leave the wrapper's cancellation disagreeing with the
// invocation context it wraps.
func TestAuthContextDerivation(t *testing.T) {
	ictx := newInvocationContextFor(t, t.Name(), "gina", "default")
	cfg := A2AConfig{Name: "a2a", Auth: auth.StaticToken("tok")}
	client := authHTTPClient(cfg.Auth)
	sendCtx, ok := authSendContext(ictx, cfg, client).(agent.InvocationContext)
	if !ok {
		t.Fatalf("authSendContext() = %T, want an agent.InvocationContext", sendCtx)
	}
	wantScope := CredentialScope(ictx.Session(), cfg.Name)
	wantCell := sendCtx.Value(credentialCellKey{})

	t.Run("WithContext carries what the transport reads", func(t *testing.T) {
		got := sendCtx.WithContext(context.Background())
		if sid, ok := a2aclient.SessionIDFrom(got); !ok || sid != wantScope {
			t.Errorf("SessionIDFrom(WithContext(...)) = %q, %v, want %q, true", sid, ok, wantScope)
		}
		if got.Value(credentialCellKey{}) != wantCell || wantCell == nil {
			t.Error("WithContext() dropped the invocation's credential cell, so the credential would be resolved again")
		}
		if iremoteagent.CardFetchClientFrom(got) != client {
			t.Error("WithContext() dropped the card-fetch client")
		}
		// The HTTP client wraps each request's context in a deadline of its
		// own, which hides the invocation behind its own type. The provider
		// must still see one through that.
		wrapped, cancel := context.WithTimeout(got, time.Minute)
		defer cancel()
		if _, ok := wrapped.(agent.InvocationContext); ok {
			t.Fatal("context.WithTimeout returned an agent.InvocationContext; the check below would prove nothing")
		}
		if _, ok := providerContext(wrapped).(agent.InvocationContext); !ok {
			t.Error("the provider would not see an agent.InvocationContext through a derived, then wrapped, context")
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

// TestRemoteAgent_AuthConcurrentInvocations exercises the auth transport and
// the provider shared across concurrent invocations of one agent value, which is
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
// authenticated too, not just the message send — otherwise a secured remote
// rejects it and the task leaks.
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
// security schemes and requirements. The auth transport never reads them, so a
// test uses this to show the card is ignored, or where it wants a realistic card.
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

// TestNewA2AOwnsAuthScope pins the one line joining the user-facing Auth field
// to the server-side cancel path. server/adka2a/v2 reads OwnsAuthScope to
// decide whether to scope the cancel it issues for an abandoned child task, and
// neither that package nor its in-package tests can call NewA2A — the import
// would cycle — so nothing else asserts the derivation. If it stopped tracking
// Auth, every Auth user's adka2a-issued cancel would be resolved with no
// identity: a provider keyed on the scope fails it, so the child task is left
// running, and a card fetched for it goes out unauthenticated.
func TestNewA2AOwnsAuthScope(t *testing.T) {
	for _, tc := range []struct {
		name string
		auth auth.CredentialProvider
		want bool
	}{
		{name: "auth set", auth: auth.StaticToken("tok"), want: true},
		{name: "auth unset", auth: nil, want: false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			a, err := NewA2A(A2AConfig{Name: "a2a", AgentCard: bearerCard("http://example.invalid"), Auth: tc.auth})
			if err != nil {
				t.Fatalf("NewA2A() error = %v", err)
			}
			internalAgent, ok := a.(agentinternal.Agent)
			if !ok {
				t.Fatalf("NewA2A() returned %T, want an agentinternal.Agent", a)
			}
			state, ok := agentinternal.Reveal(internalAgent).Config.(iremoteagent.RemoteAgentState)
			if !ok {
				t.Fatalf("agent state Config is %T, want iremoteagent.RemoteAgentState", agentinternal.Reveal(internalAgent).Config)
			}
			if got := state.A2A.OwnsAuthScope; got != tc.want {
				t.Errorf("OwnsAuthScope = %v, want %v", got, tc.want)
			}
			// The adka2a server authenticates a card it fetches for a cancel
			// through this client, so it must be set exactly when Auth is.
			if got := state.A2A.CardFetchClient != nil; got != tc.want {
				t.Errorf("CardFetchClient set = %v, want %v", got, tc.want)
			}
			// The literal, not cardFetchTimeout: a2a-go's own card resolver
			// bounds a fetch at 30s, and authenticating it must not widen that
			// to the three-minute RPC timeout.
			if c := state.A2A.CardFetchClient; c != nil && c.Timeout != 30*time.Second {
				t.Errorf("CardFetchClient.Timeout = %v, want %v", c.Timeout, 30*time.Second)
			}
		})
	}
}

// TestMintGroupSingleFlight pins that a request arriving while a mint is in
// flight joins it instead of starting another. Token() cannot be cancelled, so
// a caller released by its own deadline leaves the mint running; without the
// single-flight, every request arriving while a token endpoint hangs would
// start another one and park another goroutine.
//
// The joining callers carry an already-cancelled context, which makes the test
// deterministic: joining happens under the lock before the wait, so each one
// returns at once and can be driven from the test goroutine with no window in
// which it might have missed the in-flight call.
func TestMintGroupSingleFlight(t *testing.T) {
	var calls atomic.Int32
	entered := make(chan struct{})
	release := make(chan struct{})
	var once sync.Once
	src := tokenSourceFunc(func() (*oauth2.Token, error) {
		calls.Add(1)
		once.Do(func() { close(entered) })
		<-release
		return &oauth2.Token{AccessToken: "tok"}, nil
	})

	g := newMintGroup()
	first := make(chan string, 1)
	go func() {
		tok, err := g.token(context.WithoutCancel(t.Context()), "scope-a", src)
		if err != nil {
			t.Errorf("mintGroup.token() error = %v", err)
		}
		first <- tok
	}()
	<-entered

	g.mu.Lock()
	inFlight := g.inFlight["scope-a"]
	g.mu.Unlock()
	if inFlight == nil {
		t.Fatal("no mint recorded as in flight while the token source is blocked")
	}

	cancelled, cancel := context.WithCancel(t.Context())
	cancel()
	for range 3 {
		if _, err := g.token(cancelled, "scope-a", src); !errors.Is(err, context.Canceled) {
			t.Fatalf("mintGroup.token() error = %v, want %v from a joined call", err, context.Canceled)
		}
	}

	// Checked here rather than through the call count, which a straggling
	// goroutine may not have incremented yet: a caller that did not join left a
	// different mint in the map, whatever its goroutine has got round to doing.
	g.mu.Lock()
	still, n := g.inFlight["scope-a"], len(g.inFlight)
	g.mu.Unlock()
	if still != inFlight {
		t.Error("a second request for the same scope replaced the in-flight mint instead of joining it")
	}
	if n != 1 {
		t.Errorf("mintGroup holds %d in-flight mints for one scope, want 1", n)
	}

	close(release)
	if tok := <-first; tok != "tok" {
		t.Errorf("mintGroup.token() = %q, want %q", tok, "tok")
	}
	if got := calls.Load(); got != 1 {
		t.Errorf("token source called %d times for 4 requests on one scope, want 1", got)
	}
}

// TestMintGroupSeparatesScopes pins that a token minted for one identity never
// reaches another. The two mints must overlap: a second call made after the
// first has finished finds no entry whatever the map is keyed on, so a
// sequential version of this test passes under a key that merges identities.
// The scopes are real ones, each differing from the first in exactly one
// segment — user, session or agent — so a key that merged on any prefix, or
// dropped any one part, would also be caught. Two single-segment keys have
// nothing in common to merge on.
func TestMintGroupSeparatesScopes(t *testing.T) {
	scopes := []a2aclient.SessionID{
		iremoteagent.CredentialScope("shop", "alice", "s1", "crm"),
		iremoteagent.CredentialScope("shop", "bob", "s1", "crm"),
		iremoteagent.CredentialScope("shop", "alice", "s2", "crm"),
		iremoteagent.CredentialScope("shop", "alice", "s1", "billing"),
	}
	g := newMintGroup()
	release := make(chan struct{})
	entered := make(chan struct{})
	first := tokenSourceFunc(func() (*oauth2.Token, error) {
		close(entered)
		<-release
		return &oauth2.Token{AccessToken: "tok-0"}, nil
	})
	firstDone := make(chan string, 1)
	go func() {
		tok, _ := g.token(context.WithoutCancel(t.Context()), scopes[0], first)
		firstDone <- tok
	}()
	<-entered

	// While the first mint is still in flight, every other identity must get
	// its own token. One that joined the first mint would block on release,
	// so a bounded wait tells the two apart without depending on timing.
	for i, scope := range scopes[1:] {
		want := fmt.Sprintf("tok-%d", i+1)
		src := tokenSourceFunc(func() (*oauth2.Token, error) { return &oauth2.Token{AccessToken: want}, nil })
		got := make(chan string, 1)
		go func() {
			tok, _ := g.token(context.WithoutCancel(t.Context()), scope, src)
			got <- tok
		}()
		select {
		case tok := <-got:
			if tok != want {
				t.Errorf("%s got %q, want %q", scope, tok, want)
			}
		case <-time.After(5 * time.Second):
			t.Errorf("%s joined the in-flight mint for %s instead of minting its own", scope, scopes[0])
		}
	}
	close(release)
	if tok := <-firstDone; tok != "tok-0" {
		t.Errorf("%s got %q, want %q", scopes[0], tok, "tok-0")
	}
}

// TestMintGroupRetriesAfterFailure pins that a failed mint is not replayed to
// the next request. The entry is dropped and the channel closed under one lock,
// so a later caller either joins the call it can still be woken by or starts a
// fresh one.
func TestMintGroupRetriesAfterFailure(t *testing.T) {
	var calls atomic.Int32
	src := tokenSourceFunc(func() (*oauth2.Token, error) {
		if calls.Add(1) == 1 {
			return nil, errResolve
		}
		return &oauth2.Token{AccessToken: "tok"}, nil
	})
	g := newMintGroup()
	if _, err := g.token(t.Context(), "scope-a", src); !errors.Is(err, errResolve) {
		t.Fatalf("mintGroup.token() error = %v, want it to wrap %v", err, errResolve)
	}
	got, err := g.token(t.Context(), "scope-a", src)
	if err != nil {
		t.Fatalf("mintGroup.token() error = %v, want the retry to succeed", err)
	}
	if got != "tok" {
		t.Errorf("mintGroup.token() = %q, want %q", got, "tok")
	}
}

// TestMintGroupRecoversPanic pins that a panicking token source becomes an
// error. The mint runs on a goroutine of ours, where a panic is fatal rather
// than something the runner can turn into an error, and the source is
// third-party code.
func TestMintGroupRecoversPanic(t *testing.T) {
	src := tokenSourceFunc(func() (*oauth2.Token, error) { panic("token source exploded") })
	g := newMintGroup()
	_, err := g.token(t.Context(), "scope-a", src)
	if err == nil {
		t.Fatal("mintGroup.token() = nil error, want the panic reported as one")
	}
	if !strings.Contains(err.Error(), "token source exploded") {
		t.Errorf("mintGroup.token() error = %v, want it to name the panic value", err)
	}
	// Same group and same scope: a fresh one could not observe a wedge.
	if got, err := g.token(t.Context(), "scope-a", tokenSourceFunc(func() (*oauth2.Token, error) {
		return &oauth2.Token{AccessToken: "tok"}, nil
	})); err != nil || got != "tok" {
		t.Errorf("mintGroup.token() = %q, %v; want %q and no error after a panicking mint", got, err, "tok")
	}
}

// nilMapProvider is a credential provider whose receiver is a map. A nil one is
// callable and reads the nil map fine, which is why isTypedNil must not reject
// it.
type nilMapProvider map[string]string

func (p nilMapProvider) Credential(context.Context) (auth.Credential, error) {
	return auth.BearerCredential{Token: p["token"]}, nil
}

// nilPtrProvider is the shape isTypedNil exists for: a nil pointer whose method
// dereferences it.
type nilPtrProvider struct{ token string }

func (p *nilPtrProvider) Credential(context.Context) (auth.Credential, error) {
	return auth.BearerCredential{Token: p.token}, nil
}

func TestIsTypedNil(t *testing.T) {
	tests := []struct {
		name     string
		provider auth.CredentialProvider
		want     bool
	}{
		{name: "nil func", provider: auth.ProviderFunc(nil), want: true},
		{name: "nil pointer", provider: (*nilPtrProvider)(nil), want: true},
		{name: "nil map with a value receiver is callable", provider: nilMapProvider(nil), want: false},
		{name: "ordinary provider", provider: auth.StaticToken("tok"), want: false},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := isTypedNil(tc.provider); got != tc.want {
				t.Errorf("isTypedNil(%T) = %v, want %v", tc.provider, got, tc.want)
			}
			_, err := NewA2A(A2AConfig{Name: "a2a", AgentCard: bearerCard("http://example.invalid"), Auth: tc.provider})
			if gotErr := err != nil; gotErr != tc.want {
				t.Errorf("NewA2A() error = %v, want an error: %v", err, tc.want)
			}
		})
	}
}

// TestRemoteAgent_CleanupContextTypeTracksAuth pins what RemoteTaskCleanupCallback
// receives. With Auth set its doc promises the invocation context and the
// scope. With Auth unset it promises the plain context.WithoutCancel a caller
// had before this field existed.
func TestRemoteAgent_CleanupContextTypeTracksAuth(t *testing.T) {
	tests := []struct {
		name string
		auth auth.CredentialProvider
		want bool
	}{
		{name: "auth set", auth: auth.StaticToken("tok"), want: true},
		{name: "auth unset", auth: nil, want: false},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
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
			}
			srv := httptest.NewServer(a2asrv.NewJSONRPCHandler(a2asrv.NewHandler(executor)))
			defer srv.Close()

			var (
				mu             sync.Mutex
				called         bool
				gotInvocation  bool
				gotCancellable bool
				gotScope       bool
			)
			remoteAgent, err := NewA2A(A2AConfig{
				Name:      "a2a",
				AgentCard: bearerCard(srv.URL),
				Auth:      tc.auth,
				RemoteTaskCleanupCallback: func(ctx context.Context, _ *a2a.AgentCard, _ A2AClient, _ a2a.TaskInfo, _ error) {
					mu.Lock()
					defer mu.Unlock()
					called = true
					_, gotInvocation = ctx.(agent.InvocationContext)
					gotCancellable = ctx.Done() != nil
					_, gotScope = a2aclient.SessionIDFrom(ctx)
				},
			})
			if err != nil {
				t.Fatalf("NewA2A() error = %v", err)
			}

			for _, err := range remoteAgent.Run(newInvocationContextFor(t, t.Name(), "hana", "default")) {
				if err != nil {
					t.Fatalf("agent.Run() error = %v", err)
				}
				break
			}

			mu.Lock()
			defer mu.Unlock()
			if !called {
				t.Fatal("RemoteTaskCleanupCallback was not called")
			}
			if gotInvocation != tc.want {
				t.Errorf("cleanup context is an agent.InvocationContext: %v, want %v", gotInvocation, tc.want)
			}
			// The ClientProvider doc promises an opted-out caller sees no scope
			// anywhere, and this is the one context that is not the send one.
			if gotScope != tc.want {
				t.Errorf("cleanup context carries a credential scope: %v, want %v", gotScope, tc.want)
			}
			// Either way the cleanup context outlives the invocation: that is
			// what context.WithoutCancel is for, and the wrapper must not
			// reintroduce the cancellation it removed.
			if gotCancellable {
				t.Error("cleanup context is still cancellable; context.WithoutCancel was undone")
			}
		})
	}
}

// TestRemoteAgent_AuthOverwritesACallerScope pins the documented ownership
// rule: with Auth set this package owns the scope. Leaving a caller-attached
// one in place would resolve every user of that process under one credential
// key, which is the collision CredentialScope exists to prevent.
func TestRemoteAgent_AuthOverwritesACallerScope(t *testing.T) {
	ictx := newInvocationContextFor(t, "shop", "iris", "s7")
	tenant := ictx.WithContext(a2aclient.AttachSessionID(ictx, "one-tenant-for-everyone"))
	got := authSendContext(tenant, A2AConfig{Name: "crm", Auth: auth.StaticToken("tok")}, nil)
	sid, ok := a2aclient.SessionIDFrom(got)
	if !ok {
		t.Fatal("no credential scope on the send context")
	}
	if want := a2aclient.SessionID("shop/iris/s7/crm"); sid != want {
		t.Errorf("scope = %q, want %q; this package owns the scope when Auth is set", sid, want)
	}
}

// TestMintGroupRetiresAnOverdueAttempt covers the attempt's own deadline. The
// mint cannot be cancelled, so without retiring an overdue one its map entry
// would never be removed and every later request for that scope would join a
// mint that can never finish — a token endpoint that hangs once would lock that
// identity out for the life of the process.
func TestMintGroupRetiresAnOverdueAttempt(t *testing.T) {
	prev := mintTimeout
	mintTimeout = 30 * time.Millisecond
	t.Cleanup(func() { mintTimeout = prev })

	var calls atomic.Int32
	release := make(chan struct{})
	t.Cleanup(func() { close(release) })
	first := make(chan struct{})
	var once sync.Once
	hung := tokenSourceFunc(func() (*oauth2.Token, error) {
		calls.Add(1)
		once.Do(func() { close(first) })
		<-release
		return nil, errResolve
	})

	g := newMintGroup()
	done := make(chan struct{})
	go func() {
		defer close(done)
		if _, err := g.token(context.WithoutCancel(t.Context()), "scope-a", hung); !errors.Is(err, context.DeadlineExceeded) {
			t.Errorf("mintGroup.token() error = %v, want it to wrap %v", err, context.DeadlineExceeded)
		}
	}()
	<-first
	<-done

	// The hung mint still holds its goroutine, but its budget is spent, so the
	// next caller must mint afresh rather than wait on it again.
	got, err := g.token(t.Context(), "scope-a", tokenSourceFunc(func() (*oauth2.Token, error) {
		return &oauth2.Token{AccessToken: "recovered"}, nil
	}))
	if err != nil {
		t.Fatalf("mintGroup.token() error = %v, want the overdue attempt retired and a fresh mint", err)
	}
	if got != "recovered" {
		t.Errorf("mintGroup.token() = %q, want %q", got, "recovered")
	}
	if n := calls.Load(); n != 1 {
		t.Errorf("hung token source called %d times, want 1", n)
	}
}

// TestRemoteAgent_AuthRefusesADifferentHostname isolates the host dimension of
// the redirect policy end to end. The sibling cross-origin test points the card
// at one httptest server and redirects to another, and both bind 127.0.0.1 — so
// it varies the port, and an implementation that compared only ports would pass
// it. Here the card and the redirect target are the same listener on the same
// port and differ only in the hostname the URL spells, so the credential is
// refused for the one reason under test.
func TestRemoteAgent_AuthRefusesADifferentHostname(t *testing.T) {
	var mu sync.Mutex
	var hostsSeen []string
	var port string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		hostsSeen = append(hostsSeen, r.Host)
		mu.Unlock()
		// Same scheme, same port, same listener — only the name differs.
		http.Redirect(w, r, "http://localhost:"+port+r.URL.Path, http.StatusTemporaryRedirect)
	}))
	defer srv.Close()
	u, err := url.Parse(srv.URL)
	if err != nil {
		t.Fatalf("url.Parse() error = %v", err)
	}
	port = u.Port()

	remoteAgent, err := NewA2A(A2AConfig{Name: "a2a", AgentCard: bearerCard(srv.URL), Auth: auth.StaticToken("secret-token")})
	if err != nil {
		t.Fatalf("NewA2A() error = %v", err)
	}
	_, err = runAndCollect(newInvocationContext(t, []*session.Event{newUserHello()}), remoteAgent)

	mu.Lock()
	defer mu.Unlock()
	for _, host := range hostsSeen {
		if strings.HasPrefix(host, "localhost") {
			t.Errorf("the redirect was followed to %q; the credential must not leave the hostname the card named (saw %q)", host, hostsSeen)
		}
	}
	if err == nil {
		t.Fatal("want an error from the refused redirect, got none")
	}
	if !strings.Contains(err.Error(), "refusing redirect") {
		t.Errorf("agent.Run() error = %q, want it to mention the refused redirect", err)
	}
}

// TestRemoteAgent_AuthWarnsOnCleartextInterface covers the card that would put
// the credential on the wire unencrypted. Only a fetched card is checked at
// resolution time, so a static one like this reaches the send path
// unvalidated, and without the warning the cleartext send is silent.
func TestRemoteAgent_AuthWarnsOnCleartextInterface(t *testing.T) {
	srv := serveRecordingA2A(t, func(*http.Request) {}, a2a.NewMessage(a2a.MessageRoleAgent, a2a.NewTextPart("ok")))
	tests := []struct {
		name     string
		url      string
		second   string
		noAuth   bool
		wantWarn int32
	}{
		{name: "loopback is not reported", url: srv.URL, wantWarn: 0},
		{name: "non-loopback http is reported once", url: "http://remote.invalid:8080", wantWarn: 1},
		{name: "each cleartext interface is reported", url: "http://remote.invalid:8080", second: "http://other.invalid:8080", wantWarn: 2},
		{name: "https is not reported", url: "https://remote.invalid", wantWarn: 0},
		{name: "auth unset stays silent", url: "http://remote.invalid:8080", noAuth: true, wantWarn: 0},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var provider auth.CredentialProvider
			if !tc.noAuth {
				provider = auth.StaticToken("secret-token")
			}
			card := bearerCard(tc.url)
			if tc.second != "" {
				card.SupportedInterfaces = append(card.SupportedInterfaces, a2a.NewAgentInterface(tc.second, a2a.TransportProtocolJSONRPC))
			}
			remoteAgent, err := NewA2A(A2AConfig{Name: "a2a", AgentCard: card, Auth: provider})
			if err != nil {
				t.Fatalf("NewA2A() error = %v", err)
			}
			warns := &countingHandler{match: "will be sent in cleartext"}
			// Three times. Two would not tell "warn the first time" from "warn
			// every time but the first": both log once. The unreachable hosts
			// make the run fail, which is not what is under test — the warning
			// is emitted before the first request.
			for range 3 {
				ictx := newInvocationContext(t, []*session.Event{newUserHello()})
				scoped := ictx.WithContext(log.AttachLogger(ictx, slog.New(warns)))
				runAndCollect(scoped, remoteAgent) //nolint:errcheck // the unreachable hosts make the run fail; the warning is emitted before the first request
			}
			if got := warns.count.Load(); got != tc.wantWarn {
				t.Errorf("cleartext warning logged %d times over 3 invocations, want %d", got, tc.wantWarn)
			}
		})
	}
}

// TestMintGroupJoinerWaitsTheAttemptsRemainder pins that the bound belongs to
// the attempt, not the waiter. Arming a fresh mintTimeout per arrival is what
// makes a stuck token endpoint cost every request its own full budget, which is
// the alternative auth/gcp's provider records having rejected.
func TestMintGroupJoinerWaitsTheAttemptsRemainder(t *testing.T) {
	prev := mintTimeout
	mintTimeout = 2 * time.Second
	t.Cleanup(func() { mintTimeout = prev })

	release := make(chan struct{})
	t.Cleanup(func() { close(release) })
	first := make(chan struct{})
	var once sync.Once
	hung := tokenSourceFunc(func() (*oauth2.Token, error) {
		once.Do(func() { close(first) })
		<-release
		return nil, errResolve
	})

	g := newMintGroup()
	go g.token(context.WithoutCancel(t.Context()), "scope-a", hung) //nolint:errcheck // the waiter's own result is not under test
	<-first

	// Join halfway through the attempt. A joiner that armed its own full
	// budget would take about mintTimeout from here rather than the half that
	// is left, and the bound below separates the two by a wide margin.
	time.Sleep(mintTimeout / 2)
	g.mu.Lock()
	call := g.inFlight["scope-a"]
	g.mu.Unlock()
	if call == nil || !time.Now().Before(call.deadline) {
		t.Skip("the first attempt expired before this goroutine could join it; the machine is too loaded to time this")
	}

	start := time.Now()
	if _, err := g.token(context.WithoutCancel(t.Context()), "scope-a", hung); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("mintGroup.token() error = %v, want it to wrap %v", err, context.DeadlineExceeded)
	}
	// The remainder is about mintTimeout/2; a fresh bound would be mintTimeout.
	if waited := time.Since(start); waited > mintTimeout*3/4 {
		t.Errorf("a joiner waited %v, want at most the attempt's remainder rather than a fresh %v", waited, mintTimeout)
	}
}

// TestMintGroupRetiredAttemptDoesNotEvictItsSuccessor pins the conditional
// delete. A mint that ran past its deadline has already been retired and a
// successor may hold the entry, so an unconditional delete would drop the live
// one: the next request would start yet another mint instead of joining it.
func TestMintGroupRetiredAttemptDoesNotEvictItsSuccessor(t *testing.T) {
	prev := mintTimeout
	mintTimeout = 40 * time.Millisecond
	t.Cleanup(func() { mintTimeout = prev })

	releaseA, releaseB := make(chan struct{}), make(chan struct{})
	t.Cleanup(func() { close(releaseB) })
	enteredA, enteredB := make(chan struct{}), make(chan struct{})
	var onceA, onceB sync.Once
	srcA := tokenSourceFunc(func() (*oauth2.Token, error) {
		onceA.Do(func() { close(enteredA) })
		<-releaseA
		return &oauth2.Token{AccessToken: "a"}, nil
	})
	srcB := tokenSourceFunc(func() (*oauth2.Token, error) {
		onceB.Do(func() { close(enteredB) })
		<-releaseB
		return &oauth2.Token{AccessToken: "b"}, nil
	})

	g := newMintGroup()
	done := make(chan struct{})
	go func() {
		defer close(done)
		g.token(context.WithoutCancel(t.Context()), "scope-a", srcA) //nolint:errcheck // this waiter is expected to time out
	}()
	<-enteredA
	g.mu.Lock()
	callA := g.inFlight["scope-a"]
	g.mu.Unlock()
	<-done // the attempt's budget is now spent

	// A second caller retires the overdue attempt and starts its own.
	go g.token(context.WithoutCancel(t.Context()), "scope-a", srcB) //nolint:errcheck // still in flight at the end of the test
	<-enteredB
	g.mu.Lock()
	callB := g.inFlight["scope-a"]
	g.mu.Unlock()
	if callB == nil || callB == callA {
		t.Fatalf("the overdue attempt was not replaced: in-flight call is %p, want a new one (old %p)", callB, callA)
	}

	// Now let the abandoned first mint finish. Its entry is gone, so it must
	// leave the successor's alone. callA.done closes last inside the same
	// locked section as the delete, so receiving from it orders this check
	// after whatever that goroutine did to the map.
	close(releaseA)
	<-callA.done

	g.mu.Lock()
	still := g.inFlight["scope-a"]
	g.mu.Unlock()
	if still != callB {
		t.Errorf("in-flight call is %p after the retired attempt finished, want the successor %p", still, callB)
	}
}

// TestRemoteAgent_AuthCredentialDecidesPlacement pins the placement rule that
// matches adk-python: the caller's credential writes itself, and the agent
// card's security section is never consulted. Every row but the last uses a
// card that asks for an API key in X-Card-Key, so a row passes only if the
// card was ignored. The same card was run through adk-python's
// RemoteA2aAgent configured with a bearer scheme, and it sent the bearer
// token and no X-Card-Key.
func TestRemoteAgent_AuthCredentialDecidesPlacement(t *testing.T) {
	cardWantsAPIKey := func(url string) *a2a.AgentCard {
		return newSecureCard(url,
			a2a.NamedSecuritySchemes{"k": a2a.APIKeySecurityScheme{Location: a2a.APIKeySecuritySchemeLocationHeader, Name: "X-Card-Key"}},
			a2a.SecurityRequirementsOptions{{a2a.SecuritySchemeName("k"): a2a.SecuritySchemeScopes{}}},
		)
	}
	cardWantsNothing := func(url string) *a2a.AgentCard { return newSecureCard(url, nil, nil) }
	tests := []struct {
		name string
		card func(string) *a2a.AgentCard
		cred auth.Credential
		want map[string]string
	}{
		{
			name: "api key goes in the header the credential names",
			card: cardWantsAPIKey, cred: auth.APIKeyCredential{Name: "X-Caller-Key", Value: "secret"},
			want: map[string]string{"X-Caller-Key": "secret", "X-Card-Key": "", "Authorization": ""},
		},
		{
			name: "bearer is sent although the card asks for an api key",
			card: cardWantsAPIKey, cred: auth.BearerCredential{Token: "tok"},
			want: map[string]string{"Authorization": "Bearer tok", "X-Card-Key": ""},
		},
		{
			name: "basic works",
			card: cardWantsAPIKey, cred: auth.BasicCredential{Username: "u", Password: "p"},
			want: map[string]string{"Authorization": "Basic dTpw"},
		},
		{
			name: "extra headers ride along",
			card: cardWantsAPIKey, cred: auth.WithHeaders(auth.BearerCredential{Token: "tok"}, map[string]string{"X-Goog-User-Project": "proj"}),
			want: map[string]string{"Authorization": "Bearer tok", "X-Goog-User-Project": "proj"},
		},
		{
			name: "a pointer credential works",
			card: cardWantsAPIKey, cred: &auth.BearerCredential{Token: "tok"},
			want: map[string]string{"Authorization": "Bearer tok"},
		},
		{
			name: "oauth2 is minted and sent as a bearer token",
			card: cardWantsAPIKey, cred: auth.OAuth2Credential{TokenSource: oauth2.StaticTokenSource(&oauth2.Token{AccessToken: "at"})},
			want: map[string]string{"Authorization": "Bearer at"},
		},
		{
			name: "a card that declares no security still gets the credential",
			card: cardWantsNothing, cred: auth.BearerCredential{Token: "tok"},
			want: map[string]string{"Authorization": "Bearer tok"},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var mu sync.Mutex
			var got http.Header
			srv := serveRecordingA2A(t, func(r *http.Request) {
				mu.Lock()
				got = r.Header.Clone()
				mu.Unlock()
			}, a2a.NewMessage(a2a.MessageRoleAgent, a2a.NewTextPart("ok")))
			cred := tc.cred
			remoteAgent, err := NewA2A(A2AConfig{
				Name:      "a2a",
				AgentCard: tc.card(srv.URL),
				Auth:      auth.ProviderFunc(func(context.Context) (auth.Credential, error) { return cred, nil }),
			})
			if err != nil {
				t.Fatalf("NewA2A() error = %v", err)
			}
			if _, err := runAndCollect(newInvocationContext(t, []*session.Event{newUserHello()}), remoteAgent); err != nil {
				t.Fatalf("agent.Run() error = %v", err)
			}
			mu.Lock()
			defer mu.Unlock()
			for header, want := range tc.want {
				if v := got.Get(header); v != want {
					t.Errorf("server saw %s = %q, want %q", header, v, want)
				}
			}
		})
	}
}

// TestRemoteAgent_AuthFailsClosed pins that a credential which cannot be
// resolved or applied stops the call instead of letting it go out
// unauthenticated, as adk-python stops it. The request must not reach the
// server at all, and the run must say why.
func TestRemoteAgent_AuthFailsClosed(t *testing.T) {
	tests := []struct {
		name     string
		provider auth.CredentialProvider
		wantErr  string
		// absent must not appear anywhere the caller sees, so a secret cannot
		// leak into the error.
		absent string
	}{
		{name: "provider error", provider: auth.ProviderFunc(func(context.Context) (auth.Credential, error) { return nil, errResolve }), wantErr: "resolve auth credential"},
		{name: "consent required", provider: auth.ProviderFunc(func(context.Context) (auth.Credential, error) {
			return nil, &auth.ConsentRequiredError{}
		}), wantErr: "resolve auth credential"},
		{name: "nil credential", provider: auth.ProviderFunc(func(context.Context) (auth.Credential, error) { return nil, nil }), wantErr: "nil credential"},
		{name: "typed nil credential", provider: auth.ProviderFunc(func(context.Context) (auth.Credential, error) {
			return (*auth.BearerCredential)(nil), nil
		}), wantErr: "nil credential"},
		{name: "credential that cannot apply", provider: auth.StaticToken(""), wantErr: "apply auth credential"},
		{name: "oauth2 mint failure", provider: auth.TokenSourceProvider(tokenSourceFunc(func() (*oauth2.Token, error) {
			return nil, errors.New("token endpoint refused")
		})), wantErr: "mint oauth2 token"},
		{name: "oauth2 non-bearer token", provider: auth.TokenSourceProvider(oauth2.StaticTokenSource(&oauth2.Token{
			AccessToken: "super-secret", TokenType: "mac",
		})), wantErr: "cannot be sent as a bearer token", absent: "super-secret"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var hits atomic.Int32
			srv := serveRecordingA2A(t, func(*http.Request) { hits.Add(1) }, a2a.NewMessage(a2a.MessageRoleAgent, a2a.NewTextPart("ok")))
			remoteAgent, err := NewA2A(A2AConfig{Name: "a2a", AgentCard: bearerCard(srv.URL), Auth: tc.provider})
			if err != nil {
				t.Fatalf("NewA2A() error = %v", err)
			}
			_, err = runAndCollect(newInvocationContext(t, []*session.Event{newUserHello()}), remoteAgent)
			if n := hits.Load(); n != 0 {
				t.Errorf("the server received %d requests, want 0: a credential failure must stop the request", n)
			}
			if err == nil {
				t.Fatal("want an error explaining the failure, got none")
			}
			if !strings.Contains(err.Error(), tc.wantErr) {
				t.Errorf("agent.Run() error = %q, want it to contain %q", err, tc.wantErr)
			}
			if tc.absent != "" && strings.Contains(err.Error(), tc.absent) {
				t.Errorf("agent.Run() error = %q, want it not to leak %q", err, tc.absent)
			}
		})
	}
}

// TestRemoteAgent_AuthCoversTheCardFetchOnce pins two adk-python behaviors
// together: the agent card fetch carries the credential, and the credential is
// resolved once per invocation and reused across its calls — here the fetch and
// the send. TestRemoteAgent_AuthCleanupReusesTheInvocationCredential shows the
// reuse on the cleanup cancel too.
func TestRemoteAgent_AuthCoversTheCardFetchOnce(t *testing.T) {
	var mu sync.Mutex
	authByPath := map[string]string{}
	var srv *httptest.Server
	mux := http.NewServeMux()
	mux.Handle("/invoke", a2asrv.NewJSONRPCHandler(a2asrv.NewHandler(newA2AEventReplay(t,
		[]a2a.Event{a2a.NewMessage(a2a.MessageRoleAgent, a2a.NewTextPart("ok"))}))))
	mux.HandleFunc("/.well-known/agent-card.json", func(w http.ResponseWriter, _ *http.Request) {
		card := &a2a.AgentCard{
			SupportedInterfaces: []*a2a.AgentInterface{a2a.NewAgentInterface(srv.URL+"/invoke", a2a.TransportProtocolJSONRPC)},
			Capabilities:        a2a.AgentCapabilities{Streaming: true},
		}
		if err := json.NewEncoder(w).Encode(card); err != nil {
			t.Errorf("json.Encode(agentCard) error = %v", err)
		}
	})
	srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		authByPath[r.URL.Path] = r.Header.Get("Authorization")
		mu.Unlock()
		mux.ServeHTTP(w, r)
	}))
	defer srv.Close()

	var calls atomic.Int32
	provider := auth.ProviderFunc(func(context.Context) (auth.Credential, error) {
		return auth.BearerCredential{Token: fmt.Sprintf("tok-%d", calls.Add(1))}, nil
	})
	remoteAgent, err := NewA2A(A2AConfig{Name: "a2a", AgentCardProvider: NewAgentCardProvider(srv.URL), Auth: provider})
	if err != nil {
		t.Fatalf("NewA2A() error = %v", err)
	}
	if _, err := runAndCollect(newInvocationContext(t, []*session.Event{newUserHello()}), remoteAgent); err != nil {
		t.Fatalf("agent.Run() error = %v", err)
	}

	mu.Lock()
	defer mu.Unlock()
	for _, path := range []string{"/.well-known/agent-card.json", "/invoke"} {
		if got, want := authByPath[path], "Bearer tok-1"; got != want {
			t.Errorf("%s Authorization = %q, want %q", path, got, want)
		}
	}
	if n := calls.Load(); n != 1 {
		t.Errorf("provider called %d times in one invocation, want 1", n)
	}
}

// TestRemoteAgent_AuthRefusesAnInsecureCardSource pins that the scheme of the
// card source is checked before the fetch, because the fetch now carries the
// credential. Checking it afterwards, with the card's interfaces, would be too
// late: the credential would already be on the wire. adk-python orders the two
// checks the same way.
func TestRemoteAgent_AuthRefusesAnInsecureCardSource(t *testing.T) {
	var calls atomic.Int32
	provider := auth.ProviderFunc(func(context.Context) (auth.Credential, error) {
		calls.Add(1)
		return auth.BearerCredential{Token: "tok"}, nil
	})
	remoteAgent, err := NewA2A(A2AConfig{
		Name: "a2a",
		// Nothing listens here. A fetch attempt would fail too, but with a
		// connection error rather than the refusal under test.
		AgentCardProvider: NewAgentCardProvider("http://remote.invalid"),
		Auth:              provider,
	})
	if err != nil {
		t.Fatalf("NewA2A() error = %v", err)
	}
	_, err = runAndCollect(newInvocationContext(t, []*session.Event{newUserHello()}), remoteAgent)
	if err == nil || !strings.Contains(err.Error(), "must use https") {
		t.Fatalf("agent.Run() error = %v, want the insecure card source refused", err)
	}
	if n := calls.Load(); n != 0 {
		t.Errorf("provider called %d times, want 0: nothing should be resolved for a fetch that is refused", n)
	}
}

// TestRemoteAgent_AuthCleanupFailureSendsNoCancel pins the fail-closed rule on
// the cleanup path. With the credential unavailable the CancelTask must not go
// out unauthenticated. Leaving the remote task running is the documented cost.
func TestRemoteAgent_AuthCleanupFailureSendsNoCancel(t *testing.T) {
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
	methods := map[string]int{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		m := jsonRPCMethod(r)
		mu.Lock()
		methods[m]++
		mu.Unlock()
		inner.ServeHTTP(w, r)
	}))
	defer srv.Close()

	// The send resolves the credential and caches it for the invocation, so
	// the cleanup would reuse it. Making the cached credential fail on its
	// second Apply is how the cleanup's failure is produced without touching
	// the send.
	var applies atomic.Int32
	cred := applyFunc(func(h http.Header) error {
		if applies.Add(1) > 1 {
			return errResolve
		}
		h.Set("Authorization", "Bearer tok")
		return nil
	})
	remoteAgent, err := NewA2A(A2AConfig{
		Name:      "a2a",
		AgentCard: bearerCard(srv.URL),
		Auth:      auth.ProviderFunc(func(context.Context) (auth.Credential, error) { return cred, nil }),
	})
	if err != nil {
		t.Fatalf("NewA2A() error = %v", err)
	}
	for _, err := range remoteAgent.Run(newInvocationContextFor(t, t.Name(), "ivan", "default")) {
		if err != nil {
			t.Fatalf("agent.Run() error = %v", err)
		}
		break
	}

	mu.Lock()
	defer mu.Unlock()
	if methods["SendStreamingMessage"] != 1 {
		t.Fatalf("SendStreamingMessage reached the server %d times, want 1 (saw %v)", methods["SendStreamingMessage"], methods)
	}
	if n := methods["CancelTask"]; n != 0 {
		t.Errorf("CancelTask reached the server %d times, want 0: without a credential it must not go out", n)
	}
	if n := applies.Load(); n < 2 {
		t.Errorf("the credential was applied %d times, want the cleanup to have tried; the test did not reach the path it is named for", n)
	}
}

// applyFunc adapts a function to an [auth.Credential].
type applyFunc func(http.Header) error

func (f applyFunc) Apply(h http.Header) error { return f(h) }

// TestAuthTransportUnscopedMintsDoNotShare pins the fallback for a request that
// reaches the transport without a scope. No path in this SDK sends one today —
// every call made with Auth set carries a scope — so this guards a future one:
// keyed on the empty scope, every such request would share one mint and one
// token.
func TestAuthTransportUnscopedMintsDoNotShare(t *testing.T) {
	release := make(chan struct{})
	entered := make(chan struct{})
	first := tokenSourceFunc(func() (*oauth2.Token, error) {
		close(entered)
		<-release
		return &oauth2.Token{AccessToken: "first"}, nil
	})
	tr := &authTransport{mints: newMintGroup()}
	go func() {
		_ = tr.apply(context.WithoutCancel(t.Context()), auth.OAuth2Credential{TokenSource: first}, http.Header{})
	}()
	<-entered
	defer close(release)

	h := http.Header{}
	done := make(chan error, 1)
	go func() {
		done <- tr.apply(context.WithoutCancel(t.Context()), auth.OAuth2Credential{TokenSource: oauth2.StaticTokenSource(&oauth2.Token{AccessToken: "second"})}, h)
	}()
	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("apply() error = %v", err)
		}
		if got := h.Get("Authorization"); got != "Bearer second" {
			t.Errorf("Authorization = %q, want %q", got, "Bearer second")
		}
	case <-time.After(5 * time.Second):
		t.Fatal("an unscoped request joined another unscoped request's mint")
	}
}

// TestAuthTransportUnscopedAppliesDoNotShare is the Apply-path twin of the test
// above, for the same future request: two unscoped requests with different
// credentials must each write their own, however long the other's Apply takes.
func TestAuthTransportUnscopedAppliesDoNotShare(t *testing.T) {
	release := make(chan struct{})
	entered := make(chan struct{})
	slow := applyFunc(func(h http.Header) error {
		close(entered)
		<-release
		h.Set("Authorization", "Bearer slow")
		return nil
	})
	tr := &authTransport{mints: newMintGroup(), applies: newApplyGroup()}
	go func() { _ = tr.apply(context.WithoutCancel(t.Context()), slow, http.Header{}) }()
	<-entered
	defer close(release)

	h := http.Header{}
	done := make(chan error, 1)
	go func() { done <- tr.apply(context.WithoutCancel(t.Context()), auth.BearerCredential{Token: "fast"}, h) }()
	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("apply() error = %v", err)
		}
		if got := h.Get("Authorization"); got != "Bearer fast" {
			t.Errorf("Authorization = %q, want %q", got, "Bearer fast")
		}
	case <-time.After(5 * time.Second):
		t.Fatal("an unscoped request joined another unscoped request's Apply")
	}
}

// TestAuthTransportBoundsAPointerOAuth2Mint pins that a *auth.OAuth2Credential
// is minted through the bounded path like the value form. Left to its own
// Apply, a hung token endpoint would hold the request with nothing to stop it.
func TestAuthTransportBoundsAPointerOAuth2Mint(t *testing.T) {
	prev := mintTimeout
	mintTimeout = 50 * time.Millisecond
	t.Cleanup(func() { mintTimeout = prev })
	release := make(chan struct{})
	t.Cleanup(func() { close(release) })
	hung := tokenSourceFunc(func() (*oauth2.Token, error) {
		<-release
		return &oauth2.Token{AccessToken: "late"}, nil
	})
	tr := &authTransport{mints: newMintGroup()}
	done := make(chan error, 1)
	go func() {
		done <- tr.apply(context.WithoutCancel(t.Context()), &auth.OAuth2Credential{TokenSource: hung}, http.Header{})
	}()
	select {
	case err := <-done:
		if !errors.Is(err, context.DeadlineExceeded) {
			t.Errorf("apply() error = %v, want it to wrap %v", err, context.DeadlineExceeded)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("apply() did not return; a pointer OAuth2 credential bypassed the mint bound")
	}
}

// TestAuthTransportBoundsAWrappedOAuth2Credential covers the first-party shape
// auth.WithHeaders documents — an OAuth2 credential with an extra header. It
// cannot be recognized as OAuth2 from outside the auth package, so it writes
// itself through its own Apply, which calls TokenSource.Token. That call must
// still be bounded, and a token endpoint's response body must still be kept
// out of the error.
func TestAuthTransportBoundsAWrappedOAuth2Credential(t *testing.T) {
	prev := mintTimeout
	mintTimeout = 50 * time.Millisecond
	t.Cleanup(func() { mintTimeout = prev })

	t.Run("a hung token endpoint is bounded", func(t *testing.T) {
		release := make(chan struct{})
		t.Cleanup(func() { close(release) })
		hung := tokenSourceFunc(func() (*oauth2.Token, error) {
			<-release
			return &oauth2.Token{AccessToken: "late"}, nil
		})
		cred := auth.WithHeaders(auth.OAuth2Credential{TokenSource: hung}, map[string]string{"X-Goog-User-Project": "p"})
		tr := &authTransport{mints: newMintGroup(), applies: newApplyGroup()}
		done := make(chan error, 1)
		go func() { done <- tr.apply(context.WithoutCancel(t.Context()), cred, http.Header{}) }()
		select {
		case err := <-done:
			if !errors.Is(err, context.DeadlineExceeded) {
				t.Errorf("apply() error = %v, want it to wrap %v", err, context.DeadlineExceeded)
			}
		case <-time.After(5 * time.Second):
			t.Fatal("apply() did not return; a wrapped OAuth2 credential bypassed the bound")
		}
	})

	t.Run("the response body is redacted", func(t *testing.T) {
		const body = `{"error":"invalid_grant","error_description":"assertion=eyJhbGciOi-SECRET"}`
		failing := tokenSourceFunc(func() (*oauth2.Token, error) {
			return nil, &oauth2.RetrieveError{
				Response:         &http.Response{Status: "400 Bad Request", StatusCode: http.StatusBadRequest},
				Body:             []byte(body),
				ErrorCode:        "invalid_grant",
				ErrorDescription: "assertion=eyJhbGciOi-SECRET",
			}
		})
		cred := auth.WithHeaders(auth.OAuth2Credential{TokenSource: failing}, map[string]string{"X-Goog-User-Project": "p"})
		tr := &authTransport{mints: newMintGroup(), applies: newApplyGroup()}
		err := tr.apply(t.Context(), cred, http.Header{})
		if err == nil {
			t.Fatal("apply() = nil error, want the token endpoint's refusal")
		}
		if strings.Contains(err.Error(), "eyJhbGciOi-SECRET") {
			t.Errorf("apply() error = %q, want the response body and description redacted", err)
		}
		if !strings.Contains(err.Error(), "invalid_grant") {
			t.Errorf("apply() error = %q, want it to keep the error code", err)
		}
	})

	t.Run("the result is written only once it has landed", func(t *testing.T) {
		cred := auth.WithHeaders(auth.BearerCredential{Token: "tok"}, map[string]string{"X-Goog-User-Project": "p"})
		tr := &authTransport{mints: newMintGroup(), applies: newApplyGroup()}
		h := http.Header{}
		if err := tr.apply(t.Context(), cred, h); err != nil {
			t.Fatalf("apply() error = %v", err)
		}
		if h.Get("Authorization") != "Bearer tok" || h.Get("X-Goog-User-Project") != "p" {
			t.Errorf("headers = %v, want both the bearer token and the extra header", h)
		}
	})
}

// TestAuthTransportKeysEachStepOnTheRequestScope pins that apply hands the
// request's own scope to both single-flight groups. The groups are tested
// directly elsewhere, and this drives both the mint and the Apply path through
// the transport with two identities overlapping: keyed on anything shared,
// bob's request would join alice's in-flight step and go out with her token.
// TestRemoteAgent_AuthConcurrentInvocations covers the Apply path end to end
// too.
func TestAuthTransportKeysEachStepOnTheRequestScope(t *testing.T) {
	alice := a2aclient.AttachSessionID(context.WithoutCancel(t.Context()), iremoteagent.CredentialScope("shop", "alice", "s1", "crm"))
	bob := a2aclient.AttachSessionID(context.WithoutCancel(t.Context()), iremoteagent.CredentialScope("shop", "bob", "s1", "crm"))

	tests := []struct {
		name       string
		held, fast func(release, entered chan struct{}) auth.Credential
	}{
		{
			name: "oauth2 mint",
			held: func(release, entered chan struct{}) auth.Credential {
				return auth.OAuth2Credential{TokenSource: tokenSourceFunc(func() (*oauth2.Token, error) {
					close(entered)
					<-release
					return &oauth2.Token{AccessToken: "alice"}, nil
				})}
			},
			fast: func(chan struct{}, chan struct{}) auth.Credential {
				return auth.OAuth2Credential{TokenSource: oauth2.StaticTokenSource(&oauth2.Token{AccessToken: "bob"})}
			},
		},
		{
			name: "credential Apply",
			held: func(release, entered chan struct{}) auth.Credential {
				return applyFunc(func(h http.Header) error {
					close(entered)
					<-release
					h.Set("Authorization", "Bearer alice")
					return nil
				})
			},
			fast: func(chan struct{}, chan struct{}) auth.Credential { return auth.BearerCredential{Token: "bob"} },
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			release, entered := make(chan struct{}), make(chan struct{})
			tr := &authTransport{mints: newMintGroup(), applies: newApplyGroup()}
			go func() { _ = tr.apply(alice, tc.held(release, entered), http.Header{}) }()
			<-entered
			defer close(release)

			h := http.Header{}
			done := make(chan error, 1)
			go func() { done <- tr.apply(bob, tc.fast(nil, nil), h) }()
			select {
			case err := <-done:
				if err != nil {
					t.Fatalf("apply() error = %v", err)
				}
				if got := h.Get("Authorization"); got != "Bearer bob" {
					t.Errorf("bob's Authorization = %q, want %q", got, "Bearer bob")
				}
			case <-time.After(5 * time.Second):
				t.Fatal("bob's request joined alice's in-flight step")
			}
		})
	}
}

// TestAuthContextWithICDeltaWithoutContext covers a delta that replaces
// something other than the context, and no delta at all. Both must keep the
// wrapper's own context rather than dereferencing a context that is not there.
func TestAuthContextWithICDeltaWithoutContext(t *testing.T) {
	ictx := newInvocationContextFor(t, t.Name(), "hana", "default")
	cfg := A2AConfig{Name: "a2a", Auth: auth.StaticToken("tok")}
	sendCtx, ok := authSendContext(ictx, cfg, nil).(agent.InvocationContext)
	if !ok {
		t.Fatalf("authSendContext() = %T, want an agent.InvocationContext", sendCtx)
	}
	want := CredentialScope(ictx.Session(), cfg.Name)
	branch := "other-branch"
	for name, d := range map[string]*agent.InvocationContextDelta{
		"delta replacing only the branch": {Branch: &branch},
		"nil delta":                       nil,
	} {
		t.Run(name, func(t *testing.T) {
			got := sendCtx.WithICDelta(d)
			if sid, ok := a2aclient.SessionIDFrom(got); !ok || sid != want {
				t.Errorf("SessionIDFrom(WithICDelta(...)) = %q, %v, want %q, true", sid, ok, want)
			}
		})
	}
}

// closeRecorder is a request body that records whether it was closed.
type closeRecorder struct {
	io.Reader
	closed atomic.Bool
}

func (c *closeRecorder) Close() error { c.closed.Store(true); return nil }

// TestAuthTransportClosesTheBodyOnFailure pins the RoundTripper contract on the
// two paths that return before the base transport takes the request — the
// credential failing to resolve, and failing to apply. The body is ours to
// close on both, and leaving it open leaks it.
func TestAuthTransportClosesTheBodyOnFailure(t *testing.T) {
	failResolve := auth.ProviderFunc(func(context.Context) (auth.Credential, error) { return nil, errResolve })
	failApply := auth.ProviderFunc(func(context.Context) (auth.Credential, error) {
		return applyFunc(func(http.Header) error { return errResolve }), nil
	})
	for name, provider := range map[string]auth.CredentialProvider{"resolve fails": failResolve, "apply fails": failApply} {
		t.Run(name, func(t *testing.T) {
			tr := &authTransport{provider: provider, mints: newMintGroup(), applies: newApplyGroup(), base: http.DefaultTransport}
			body := &closeRecorder{Reader: strings.NewReader("{}")}
			req, err := http.NewRequestWithContext(t.Context(), http.MethodPost, "http://example.invalid", body)
			if err != nil {
				t.Fatalf("http.NewRequest() error = %v", err)
			}
			if _, err := tr.RoundTrip(req); !errors.Is(err, errResolve) {
				t.Fatalf("RoundTrip() error = %v, want it to wrap %v", err, errResolve)
			}
			if !body.closed.Load() {
				t.Error("RoundTrip() returned early without closing the request body")
			}
		})
	}

	tr := &authTransport{provider: failResolve, mints: newMintGroup(), applies: newApplyGroup(), base: http.DefaultTransport}

	// A request with no body must not trip over the missing one either.
	req, err := http.NewRequestWithContext(t.Context(), http.MethodGet, "http://example.invalid", nil)
	if err != nil {
		t.Fatalf("http.NewRequest() error = %v", err)
	}
	if _, err := tr.RoundTrip(req); !errors.Is(err, errResolve) {
		t.Errorf("RoundTrip() error = %v, want it to wrap %v", err, errResolve)
	}
}

// TestAuthTransportSharesAStepWithinAScope pins the other half of the
// single-flight: two requests for the same scope share one Apply step. Building
// a fresh Apply group per request would bring back what the mintGroup doc says
// the single-flight exists to stop — another parked goroutine for every request
// that arrives while a token endpoint hangs. The OAuth2 twin below does the
// same for the mint group.
func TestAuthTransportSharesAStepWithinAScope(t *testing.T) {
	scope := a2aclient.AttachSessionID(context.WithoutCancel(t.Context()), iremoteagent.CredentialScope("shop", "alice", "s1", "crm"))
	var calls atomic.Int32
	release, entered := make(chan struct{}), make(chan struct{})
	var once sync.Once
	cred := applyFunc(func(h http.Header) error {
		calls.Add(1)
		once.Do(func() { close(entered) })
		<-release
		h.Set("Authorization", "Bearer alice")
		return nil
	})
	tr := &authTransport{mints: newMintGroup(), applies: newApplyGroup()}
	first := make(chan error, 1)
	go func() { first <- tr.apply(scope, cred, http.Header{}) }()
	<-entered
	sid, _ := a2aclient.SessionIDFrom(scope)
	tr.applies.mu.Lock()
	held := tr.applies.inFlight[sid]
	tr.applies.mu.Unlock()
	if held == nil {
		t.Fatal("no step recorded in flight for the scope while its Apply is running; the transport is not using its shared group")
	}

	// The second request carries an already-cancelled context, so it returns
	// as soon as it has joined or started a step, and the count below is read
	// with no window in which it might still be starting one.
	cancelled, cancel := context.WithCancel(scope)
	cancel()
	if err := tr.apply(cancelled, cred, http.Header{}); !errors.Is(err, context.Canceled) {
		t.Fatalf("apply() error = %v, want %v from a request that joined the in-flight step", err, context.Canceled)
	}
	tr.applies.mu.Lock()
	still, n := tr.applies.inFlight[sid], len(tr.applies.inFlight)
	tr.applies.mu.Unlock()
	if still != held || n != 1 {
		t.Errorf("after a second request the group holds %d steps (same one: %v), want the first one only", n, still == held)
	}
	close(release)
	if err := <-first; err != nil {
		t.Fatalf("apply() error = %v", err)
	}
	if n := calls.Load(); n != 1 {
		t.Errorf("Apply ran %d times for two overlapping requests on one scope, want 1", n)
	}
}

// TestAuthTransportSharesAMintWithinAScope is the OAuth2 twin of the test above:
// two requests for one scope share one mint through the transport's own group.
func TestAuthTransportSharesAMintWithinAScope(t *testing.T) {
	scope := a2aclient.AttachSessionID(context.WithoutCancel(t.Context()), iremoteagent.CredentialScope("shop", "alice", "s1", "crm"))
	release, entered := make(chan struct{}), make(chan struct{})
	var once sync.Once
	cred := auth.OAuth2Credential{TokenSource: tokenSourceFunc(func() (*oauth2.Token, error) {
		once.Do(func() { close(entered) })
		<-release
		return &oauth2.Token{AccessToken: "alice"}, nil
	})}
	tr := &authTransport{mints: newMintGroup(), applies: newApplyGroup()}
	first := make(chan error, 1)
	go func() { first <- tr.apply(scope, cred, http.Header{}) }()
	<-entered
	defer func() {
		close(release)
		<-first
	}()

	sid, _ := a2aclient.SessionIDFrom(scope)
	tr.mints.mu.Lock()
	held := tr.mints.inFlight[sid]
	tr.mints.mu.Unlock()
	if held == nil {
		t.Fatal("no mint recorded in flight for the scope while it runs; the transport is not using its shared group")
	}
	cancelled, cancel := context.WithCancel(scope)
	cancel()
	if err := tr.apply(cancelled, cred, http.Header{}); !errors.Is(err, context.Canceled) {
		t.Fatalf("apply() error = %v, want %v from a request that joined the in-flight mint", err, context.Canceled)
	}
	tr.mints.mu.Lock()
	still, n := tr.mints.inFlight[sid], len(tr.mints.inFlight)
	tr.mints.mu.Unlock()
	if still != held || n != 1 {
		t.Errorf("after a second request the group holds %d mints (same one: %v), want the first one only", n, still == held)
	}
}

// TestRemoteAgent_AuthProviderSeesTheIdentity pins the contract the auth package
// itself states: a provider that needs the acting user recovers it with
// agent.IdentityFromContext, as auth/gcp's provider does. It has to work on
// every call the transport makes for an invocation — the card fetch and the
// send here — whatever the HTTP client did to the context on the way down.
func TestRemoteAgent_AuthProviderSeesTheIdentity(t *testing.T) {
	var srv *httptest.Server
	mux := http.NewServeMux()
	mux.Handle("/invoke", a2asrv.NewJSONRPCHandler(a2asrv.NewHandler(newA2AEventReplay(t,
		[]a2a.Event{a2a.NewMessage(a2a.MessageRoleAgent, a2a.NewTextPart("ok"))}))))
	mux.HandleFunc("/.well-known/agent-card.json", func(w http.ResponseWriter, _ *http.Request) {
		card := &a2a.AgentCard{
			SupportedInterfaces: []*a2a.AgentInterface{a2a.NewAgentInterface(srv.URL+"/invoke", a2a.TransportProtocolJSONRPC)},
			Capabilities:        a2a.AgentCapabilities{Streaming: true},
		}
		if err := json.NewEncoder(w).Encode(card); err != nil {
			t.Errorf("json.Encode(agentCard) error = %v", err)
		}
	})
	var mu sync.Mutex
	var gotAuth []string
	srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		gotAuth = append(gotAuth, r.Header.Get("Authorization"))
		mu.Unlock()
		mux.ServeHTTP(w, r)
	}))
	defer srv.Close()

	provider := auth.ProviderFunc(func(ctx context.Context) (auth.Credential, error) {
		id, ok := agent.IdentityFromContext(ctx)
		if !ok {
			return nil, fmt.Errorf("no identity on %T", ctx)
		}
		return auth.BearerCredential{Token: id.AppName + ":" + id.UserID + ":" + id.SessionID}, nil
	})
	remoteAgent, err := NewA2A(A2AConfig{Name: "a2a", AgentCardProvider: NewAgentCardProvider(srv.URL), Auth: provider})
	if err != nil {
		t.Fatalf("NewA2A() error = %v", err)
	}
	if _, err := runAndCollect(newInvocationContextFor(t, "shop", "ivy", "s3"), remoteAgent); err != nil {
		t.Fatalf("agent.Run() error = %v", err)
	}
	mu.Lock()
	defer mu.Unlock()
	if len(gotAuth) < 2 {
		t.Fatalf("server saw %d requests, want the card fetch and the send", len(gotAuth))
	}
	for i, got := range gotAuth {
		if want := "Bearer shop:ivy:s3"; got != want {
			t.Errorf("request %d Authorization = %q, want %q", i, got, want)
		}
	}
}
