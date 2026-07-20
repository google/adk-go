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

package mcptoolset_test

import (
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/modelcontextprotocol/go-sdk/mcp"
	"golang.org/x/oauth2/clientcredentials"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/auth"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/internal/toolinternal"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/mcptoolset"
)

// TestMCPToolSetAuthOverHTTP is a self-contained, offline demonstration that
// mcptoolset.Config.Auth authenticates outgoing MCP requests for different
// credential providers (bearer token, header API key): with the provider the
// credential is attached to both listing and calling; without it the server
// rejects the request with 401.
func TestMCPToolSetAuthOverHTTP(t *testing.T) {
	cases := []struct {
		name     string
		authOK   func(*http.Request) bool // server-side check
		provider auth.CredentialProvider  // client-side provider
	}{
		{
			name:     "bearer token",
			authOK:   func(r *http.Request) bool { return r.Header.Get("Authorization") == "Bearer secret-token" },
			provider: auth.StaticToken("secret-token"),
		},
		{
			name:     "header API key",
			authOK:   func(r *http.Request) bool { return r.Header.Get("X-Api-Key") == "secret-key" },
			provider: auth.APIKey("X-Api-Key", "secret-key"),
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			endpoint := startAuthedMCPServer(t, tc.authOK)

			// With the provider: listing and calling both succeed (each rides the
			// auth RoundTripper).
			ts, err := mcptoolset.New(mcptoolset.Config{Endpoint: endpoint, Auth: tc.provider})
			if err != nil {
				t.Fatalf("New: %v", err)
			}
			callWeatherTool(t, ts)

			// Without the provider: the server rejects with 401.
			noAuth, err := mcptoolset.New(mcptoolset.Config{Endpoint: endpoint})
			if err != nil {
				t.Fatalf("New (no auth): %v", err)
			}
			if _, err := noAuth.Tools(newReadonlyCtx(t)); err == nil {
				t.Fatal("Tools (no auth) succeeded, want an authorization error")
			}
		})
	}
}

// TestMCPToolSetAuthClientCredentials demonstrates, offline, 2-legged
// (client_credentials) OAuth over MCP: a local token endpoint exchanges the
// client id/secret for a bearer token, auth.TokenSourceProvider mints and caches
// it, and the MCP server accepts the resulting Authorization header.
func TestMCPToolSetAuthClientCredentials(t *testing.T) {
	const minted = "cc-minted-access-token"
	tokenURL := startClientCredentialsTokenServer(t, "client-id", "client-secret", minted)

	endpoint := startAuthedMCPServer(t, func(r *http.Request) bool {
		return r.Header.Get("Authorization") == "Bearer "+minted
	})

	cc := clientcredentials.Config{
		ClientID:     "client-id",
		ClientSecret: "client-secret",
		TokenURL:     tokenURL,
	}
	provider := auth.TokenSourceProvider(cc.TokenSource(t.Context()))

	ts, err := mcptoolset.New(mcptoolset.Config{Endpoint: endpoint, Auth: provider})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	callWeatherTool(t, ts)
}

// startAuthedMCPServer starts a local streamable-HTTP MCP server that serves a
// request only when authOK reports the credential is present, rejecting others
// with 401 — on *every* request, tool listing and tool calling alike. It is the
// Go analog of adk-python's
// contributing/samples/mcp/mcp_toolset_auth/oauth_mcp_server.py and lets these
// tests verify, fully offline, that mcptoolset.Config.Auth attaches the
// credential end-to-end over the real HTTP transport.
func startAuthedMCPServer(t *testing.T, authOK func(*http.Request) bool) string {
	t.Helper()

	server := mcp.NewServer(&mcp.Implementation{Name: "authed_weather", Version: "v1.0.0"}, nil)
	mcp.AddTool(server, &mcp.Tool{Name: "get_weather", Description: "returns weather in the given city"}, weatherFunc)

	mcpHandler := mcp.NewStreamableHTTPHandler(func(*http.Request) *mcp.Server { return server }, nil)
	authed := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !authOK(r) {
			http.Error(w, "unauthorized", http.StatusUnauthorized)
			return
		}
		mcpHandler.ServeHTTP(w, r)
	})

	srv := httptest.NewServer(authed)
	// The MCP client keeps a streamable-HTTP stream open; force those connections
	// closed before srv.Close, which otherwise blocks waiting for them.
	t.Cleanup(func() {
		srv.CloseClientConnections()
		srv.Close()
	})
	return srv.URL
}

// newReadonlyCtx builds a minimal read-only context for listing tools.
func newReadonlyCtx(t *testing.T) agent.ReadonlyContext {
	t.Helper()
	return icontext.NewReadonlyContext(icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{}))
}

// callWeatherTool lists the toolset's tools and runs get_weather("london"),
// returning the result. Both requests ride the auth RoundTripper, so this
// succeeds only when Config.Auth attached a credential the server accepts.
func callWeatherTool(t *testing.T, ts tool.Toolset) map[string]any {
	t.Helper()
	tools, err := ts.Tools(newReadonlyCtx(t))
	if err != nil {
		t.Fatalf("Tools: %v", err)
	}
	if len(tools) != 1 || tools[0].Name() != "get_weather" {
		t.Fatalf("got %d tools, want [get_weather]", len(tools))
	}
	ft, ok := tools[0].(toolinternal.FunctionTool)
	if !ok {
		t.Fatalf("tool %T is not a FunctionTool", tools[0])
	}
	invCtx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{})
	got, err := ft.Run(agent.NewToolContext(invCtx, "", nil, nil), map[string]any{"city": "london"})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if got == nil {
		t.Fatal("Run returned a nil result")
	}
	return got
}

// startClientCredentialsTokenServer starts a local OAuth2 token endpoint that
// implements the 2-legged client_credentials grant: it verifies the client id
// and secret (accepting them via HTTP Basic or form body, as x/oauth2 may send
// either) and returns issueToken.
func startClientCredentialsTokenServer(t *testing.T, wantID, wantSecret, issueToken string) string {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := r.ParseForm(); err != nil {
			http.Error(w, "bad form", http.StatusBadRequest)
			return
		}
		id, secret, ok := r.BasicAuth()
		if !ok {
			id, secret = r.PostForm.Get("client_id"), r.PostForm.Get("client_secret")
		}
		if r.PostForm.Get("grant_type") != "client_credentials" || id != wantID || secret != wantSecret {
			http.Error(w, "invalid_client", http.StatusUnauthorized)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"access_token":"`+issueToken+`","token_type":"Bearer","expires_in":3600}`)
	}))
	t.Cleanup(srv.Close)
	return srv.URL
}
