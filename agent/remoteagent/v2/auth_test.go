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
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	"github.com/a2aproject/a2a-go/v2/a2a"
	"github.com/a2aproject/a2a-go/v2/a2aclient"
	"github.com/a2aproject/a2a-go/v2/a2asrv"
	"golang.org/x/oauth2"

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
			name: "provider error",
			provider: auth.ProviderFunc(func(context.Context) (*auth.Credential, error) {
				return nil, errors.New("boom")
			}),
			wantErr: true,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			svc := credentialsService{provider: tc.provider}
			got, err := svc.Get(context.Background(), a2aclient.SessionID("sid"), a2a.SecuritySchemeName("scheme"))
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
	if err == nil {
		t.Fatal("NewA2A() = nil error, want error for Auth combined with a custom ClientProvider")
	}
}

func TestRemoteAgent_AuthAttachesBearerHeader(t *testing.T) {
	// The interceptor only attaches auth when the card declares a matching
	// security requirement, so the card must carry one.
	hdr := runA2AAuth(t, func(url string) *a2a.AgentCard {
		return &a2a.AgentCard{
			Name:                "a2a",
			SupportedInterfaces: []*a2a.AgentInterface{a2a.NewAgentInterface(url, a2a.TransportProtocolJSONRPC)},
			Capabilities:        a2a.AgentCapabilities{Streaming: true},
			SecuritySchemes: a2a.NamedSecuritySchemes{
				"bearer": a2a.HTTPAuthSecurityScheme{Scheme: "Bearer"},
			},
			SecurityRequirements: a2a.SecurityRequirementsOptions{
				{a2a.SecuritySchemeName("bearer"): a2a.SecuritySchemeScopes{}},
			},
		}
	}, auth.StaticToken("secret-token"))

	if got := hdr.Get("Authorization"); got != "Bearer secret-token" {
		t.Errorf("server saw Authorization = %q, want %q", got, "Bearer secret-token")
	}
}

func TestRemoteAgent_AuthAttachesAPIKeyHeader(t *testing.T) {
	// For an API-key scheme the interceptor sets the header named by the scheme
	// (here X-Api-Key) to the resolved credential value.
	hdr := runA2AAuth(t, func(url string) *a2a.AgentCard {
		return &a2a.AgentCard{
			Name:                "a2a",
			SupportedInterfaces: []*a2a.AgentInterface{a2a.NewAgentInterface(url, a2a.TransportProtocolJSONRPC)},
			Capabilities:        a2a.AgentCapabilities{Streaming: true},
			SecuritySchemes: a2a.NamedSecuritySchemes{
				"apikey": a2a.APIKeySecurityScheme{Location: a2a.APIKeySecuritySchemeLocationHeader, Name: "X-Api-Key"},
			},
			SecurityRequirements: a2a.SecurityRequirementsOptions{
				{a2a.SecuritySchemeName("apikey"): a2a.SecuritySchemeScopes{}},
			},
		}
	}, auth.APIKey("X-Api-Key", "secret-key"))

	if got := hdr.Get("X-Api-Key"); got != "secret-key" {
		t.Errorf("server saw X-Api-Key = %q, want %q", got, "secret-key")
	}
}

// runA2AAuth starts a local A2A server that records the headers of the request
// it receives, builds a remote agent for cardFor(serverURL) with the given Auth
// provider, runs one turn, and returns those headers. It lets the auth tests
// assert, fully offline, that A2AConfig.Auth attaches the credential per the
// card's declared security scheme.
func runA2AAuth(t *testing.T, cardFor func(url string) *a2a.AgentCard, provider auth.CredentialProvider) http.Header {
	t.Helper()

	executor := newA2AEventReplay(t, []a2a.Event{
		a2a.NewMessage(a2a.MessageRoleAgent, a2a.NewTextPart("ok")),
	})
	inner := a2asrv.NewJSONRPCHandler(a2asrv.NewHandler(executor))

	var mu sync.Mutex
	var got http.Header
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		got = r.Header.Clone()
		mu.Unlock()
		inner.ServeHTTP(w, r)
	}))
	defer srv.Close()

	remoteAgent, err := NewA2A(A2AConfig{Name: "a2a", AgentCard: cardFor(srv.URL), Auth: provider})
	if err != nil {
		t.Fatalf("NewA2A() error = %v", err)
	}

	ictx := newInvocationContext(t, []*session.Event{newUserHello()})
	if _, err := runAndCollect(ictx, remoteAgent); err != nil {
		t.Fatalf("agent.Run() error = %v", err)
	}

	mu.Lock()
	defer mu.Unlock()
	return got
}
