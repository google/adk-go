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

package authtool_test

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"sync"
	"testing"
	"time"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/credentialservice"
	icredentialservice "google.golang.org/adk/v2/internal/credentialservice"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool/toolauth"
)

// Token endpoint answers used by the tests. The values are placeholders, not
// shaped like any provider's tokens.
const (
	ccTokenBody      = `{"access_token":"cc-access","token_type":"Bearer","expires_in":3600}`
	refreshTokenBody = `{"access_token":"cc-access","token_type":"Bearer","expires_in":3600,"refresh_token":"r-new"}`
	invalidClient    = `{"error":"invalid_client"}`
	invalidGrant     = `{"error":"invalid_grant"}`
)

// fakeToolContext is the context of one tool call: app "app-1", user
// "user-1", function call "call-1", an in-memory session state and real event
// actions. The credential service, when there is one, travels on the embedded
// context as the runner puts it there.
type fakeToolContext struct {
	agent.ContextMock
	context.Context

	appName, userID, callID string
	state                   session.State
	actions                 *session.EventActions
}

// newToolContext returns the context of a tool call for user-1 whose
// invocation has svc as its credential service; nil means none.
func newToolContext(t *testing.T, svc credentialservice.Service) *fakeToolContext {
	t.Helper()
	return newToolContextFrom(t, t.Context(), svc)
}

// newToolContextFrom is newToolContext over parent, for a test that cancels
// the call.
func newToolContextFrom(t *testing.T, parent context.Context, svc credentialservice.Service) *fakeToolContext {
	t.Helper()
	resp, err := session.InMemoryService().Create(t.Context(), &session.CreateRequest{AppName: "app-1", UserID: "user-1"})
	if err != nil {
		t.Fatalf("Create() error = %v", err)
	}
	ctx := parent
	if svc != nil {
		ctx = icredentialservice.WithService(parent, svc)
	}
	return &fakeToolContext{
		Context: ctx,
		appName: "app-1",
		userID:  "user-1",
		callID:  "call-1",
		state:   resp.Session.State(),
		actions: &session.EventActions{},
	}
}

func (f *fakeToolContext) Deadline() (time.Time, bool)          { return f.Context.Deadline() }
func (f *fakeToolContext) Done() <-chan struct{}                { return f.Context.Done() }
func (f *fakeToolContext) Err() error                           { return f.Context.Err() }
func (f *fakeToolContext) Value(key any) any                    { return f.Context.Value(key) }
func (f *fakeToolContext) AppName() string                      { return f.appName }
func (f *fakeToolContext) UserID() string                       { return f.userID }
func (f *fakeToolContext) FunctionCallID() string               { return f.callID }
func (f *fakeToolContext) State() session.State                 { return f.state }
func (f *fakeToolContext) ReadonlyState() session.ReadonlyState { return f.state }
func (f *fakeToolContext) Actions() *session.EventActions       { return f.actions }

// fakeService is an in-memory credential service that counts its calls and
// fails them on demand.
type fakeService struct {
	inner   credentialservice.Service
	loadErr error
	saveErr error

	mu    sync.Mutex
	loads int
	saves int
}

func newFakeService() *fakeService {
	return &fakeService{inner: credentialservice.InMemoryService()}
}

func (s *fakeService) LoadCredential(ctx context.Context, appName, userID string, config *toolauth.Config) (*toolauth.Credential, error) {
	s.mu.Lock()
	s.loads++
	err := s.loadErr
	s.mu.Unlock()
	if err != nil {
		return nil, err
	}
	return s.inner.LoadCredential(ctx, appName, userID, config)
}

func (s *fakeService) SaveCredential(ctx context.Context, appName, userID string, config *toolauth.Config) error {
	s.mu.Lock()
	s.saves++
	err := s.saveErr
	s.mu.Unlock()
	if err != nil {
		return err
	}
	return s.inner.SaveCredential(ctx, appName, userID, config)
}

func (s *fakeService) counts() (loads, saves int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.loads, s.saves
}

// put stores cred under cfg's key for app-1 and user-1.
func (s *fakeService) put(t *testing.T, cfg *toolauth.Config, cred *toolauth.Credential) {
	t.Helper()
	c := *cfg
	c.ExchangedAuthCredential = cred
	if err := s.inner.SaveCredential(t.Context(), "app-1", "user-1", &c); err != nil {
		t.Fatalf("SaveCredential() error = %v", err)
	}
}

// stored returns what the service holds under cfg's key for app-1 and user-1.
func (s *fakeService) stored(t *testing.T, cfg *toolauth.Config) *toolauth.Credential {
	t.Helper()
	cred, err := s.inner.LoadCredential(t.Context(), "app-1", "user-1", cfg)
	if err != nil {
		t.Fatalf("LoadCredential() error = %v", err)
	}
	return cred
}

// tokenRequest is one request a tokenServer received.
type tokenRequest struct {
	form          url.Values
	authorization string
}

// tokenServer is a fake OAuth2 token endpoint. It answers every request with
// status and body (200 and ccTokenBody unless set), after delay, and records
// the requests and how many were in flight at once. A request also waits
// until holdUntil requests have been in flight at once (for at most two
// seconds), and until release is closed when release is set.
type tokenServer struct {
	*httptest.Server

	mu          sync.Mutex
	status      int
	body        string
	delay       time.Duration
	holdUntil   int
	release     chan struct{}
	requests    []tokenRequest
	inFlight    int
	maxInFlight int
	arrived     chan struct{}
}

func newTokenServer(t *testing.T) *tokenServer {
	t.Helper()
	s := &tokenServer{status: http.StatusOK, body: ccTokenBody, arrived: make(chan struct{}, 100)}
	s.Server = httptest.NewServer(http.HandlerFunc(s.serve))
	t.Cleanup(s.Close)
	return s
}

// set changes the answer and the waiting rules under the server's lock.
func (s *tokenServer) set(f func(s *tokenServer)) {
	s.mu.Lock()
	defer s.mu.Unlock()
	f(s)
}

func (s *tokenServer) serve(w http.ResponseWriter, r *http.Request) {
	if err := r.ParseForm(); err != nil {
		http.Error(w, "bad form", http.StatusBadRequest)
		return
	}
	s.mu.Lock()
	s.requests = append(s.requests, tokenRequest{form: r.PostForm, authorization: r.Header.Get("Authorization")})
	s.inFlight++
	s.maxInFlight = max(s.maxInFlight, s.inFlight)
	status, body, delay, holdUntil, release := s.status, s.body, s.delay, s.holdUntil, s.release
	s.mu.Unlock()
	s.arrived <- struct{}{}
	defer func() {
		s.mu.Lock()
		s.inFlight--
		s.mu.Unlock()
	}()

	deadline := time.Now().Add(2 * time.Second)
	for holdUntil > 0 && time.Now().Before(deadline) {
		s.mu.Lock()
		n := s.maxInFlight
		s.mu.Unlock()
		if n >= holdUntil {
			break
		}
		time.Sleep(5 * time.Millisecond)
	}
	if release != nil {
		<-release
	}
	time.Sleep(delay)
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_, _ = io.WriteString(w, body)
}

func (s *tokenServer) seen() []tokenRequest {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]tokenRequest(nil), s.requests...)
}

func (s *tokenServer) peak() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.maxInFlight
}

// ccConfig is a client-credentials config for the token endpoint tokenURL,
// stored under the key "key-1".
func ccConfig(tokenURL string) *toolauth.Config {
	return &toolauth.Config{
		AuthScheme: toolauth.Scheme{Type: toolauth.SchemeTypeOAuth2, Flows: &toolauth.OAuthFlows{
			ClientCredentials: &toolauth.OAuthFlow{TokenURL: tokenURL, Scopes: map[string]string{"read": "Read"}},
		}},
		RawAuthCredential: &toolauth.Credential{AuthType: toolauth.CredentialTypeOAuth2, OAuth2: &toolauth.OAuth2Auth{
			ClientID: "client-1", ClientSecret: "secret-1",
		}},
		CredentialKey: "key-1",
	}
}

// codeConfig is ccConfig with the authorization-code flow instead.
func codeConfig(tokenURL string) *toolauth.Config {
	return &toolauth.Config{
		AuthScheme: toolauth.Scheme{Type: toolauth.SchemeTypeOAuth2, Flows: &toolauth.OAuthFlows{
			AuthorizationCode: &toolauth.OAuthFlow{
				AuthorizationURL: "https://idp.example/auth",
				TokenURL:         tokenURL,
				Scopes:           map[string]string{"read": "Read"},
			},
		}},
		RawAuthCredential: &toolauth.Credential{AuthType: toolauth.CredentialTypeOAuth2, OAuth2: &toolauth.OAuth2Auth{
			ClientID: "client-1", ClientSecret: "secret-1", RedirectURI: "https://app.example/cb",
		}},
		CredentialKey: "key-1",
	}
}

// apiKeyConfig is an API-key config whose raw credential holds the key.
func apiKeyConfig() *toolauth.Config {
	return &toolauth.Config{
		AuthScheme:        toolauth.Scheme{Type: toolauth.SchemeTypeAPIKey, In: "header", Name: "X-Key"},
		RawAuthCredential: &toolauth.Credential{AuthType: toolauth.CredentialTypeAPIKey, APIKey: "test-api-key-1"},
		CredentialKey:     "key-1",
	}
}
