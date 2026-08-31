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

package mcptoolset

import (
	"net/http"
	"testing"
	"time"

	"github.com/modelcontextprotocol/go-sdk/mcp"

	"google.golang.org/adk/v2/auth"
)

const testEndpoint = "https://mcp.example/mcp"

func TestBuildTransport(t *testing.T) {
	// Shared inputs for the wrap/passthrough cases. buildTransport must never
	// mutate a caller-supplied transport or client, so sharing these across
	// cases is safe (and is itself asserted below).
	base := &http.Client{Timeout: 7 * time.Second}
	callerStreamable := &mcp.StreamableClientTransport{
		Endpoint:             testEndpoint,
		HTTPClient:           base,
		MaxRetries:           9,
		DisableStandaloneSSE: true,
	}
	callerCommand := &mcp.CommandTransport{}
	// sentinelRT stands in for a caller's existing HTTP transport, to assert the
	// auth transport chains onto it rather than replacing it.
	sentinelRT := &stubRoundTripper{}

	tests := []struct {
		name    string
		cfg     Config
		wantErr bool
		check   func(t *testing.T, tr mcp.Transport)
	}{
		{
			name: "endpoint with auth builds a streamable client carrying the auth transport",
			cfg:  Config{Endpoint: testEndpoint, Auth: auth.StaticToken("tok")},
			check: func(t *testing.T, tr mcp.Transport) {
				st := mustStreamable(t, tr)
				if st.Endpoint != testEndpoint {
					t.Errorf("Endpoint = %q, want %q", st.Endpoint, testEndpoint)
				}
				wantsAuthTransport(t, st.HTTPClient)
			},
		},
		{
			name: "auth wraps a caller streamable transport on a copy, without mutating it",
			cfg:  Config{Transport: callerStreamable, Auth: auth.StaticToken("tok")},
			check: func(t *testing.T, tr mcp.Transport) {
				st := mustStreamable(t, tr)
				// Non-auth fields and base client settings are preserved on the copy.
				if st.Endpoint != testEndpoint || st.MaxRetries != 9 || !st.DisableStandaloneSSE {
					t.Errorf("preserved fields lost: %+v", st)
				}
				wantsAuthTransport(t, st.HTTPClient)
				if st.HTTPClient.Timeout != 7*time.Second {
					t.Errorf("client Timeout = %v, want 7s (base settings should be copied)", st.HTTPClient.Timeout)
				}
				// The caller's transport and client must be untouched.
				if st == callerStreamable {
					t.Error("caller transport was not copied")
				}
				if st.HTTPClient == base {
					t.Error("caller HTTPClient was not copied")
				}
				if callerStreamable.HTTPClient != base || base.Transport != nil {
					t.Error("caller's transport/client was mutated")
				}
			},
		},
		{
			name: "auth chains onto the caller's existing HTTP transport",
			cfg: Config{
				Transport: &mcp.StreamableClientTransport{
					Endpoint:   testEndpoint,
					HTTPClient: &http.Client{Transport: sentinelRT},
				},
				Auth: auth.StaticToken("tok"),
			},
			check: func(t *testing.T, tr mcp.Transport) {
				st := mustStreamable(t, tr)
				at, ok := st.HTTPClient.Transport.(*auth.Transport)
				if !ok {
					t.Fatalf("HTTPClient.Transport = %T, want *auth.Transport", st.HTTPClient.Transport)
				}
				if at.Base != sentinelRT {
					t.Errorf("auth.Transport.Base = %v, want the caller's original transport", at.Base)
				}
			},
		},
		{
			name:    "auth with a non-streamable transport is an error",
			cfg:     Config{Transport: &mcp.CommandTransport{}, Auth: auth.StaticToken("tok")},
			wantErr: true,
		},
		{
			name:    "auth without an endpoint or transport is an error",
			cfg:     Config{Auth: auth.StaticToken("tok")},
			wantErr: true,
		},
		{
			name: "endpoint without auth builds a streamable client with no HTTP client",
			cfg:  Config{Endpoint: testEndpoint},
			check: func(t *testing.T, tr mcp.Transport) {
				st := mustStreamable(t, tr)
				if st.Endpoint != testEndpoint {
					t.Errorf("Endpoint = %q, want %q", st.Endpoint, testEndpoint)
				}
				if st.HTTPClient != nil {
					t.Errorf("HTTPClient = %v, want nil without Auth", st.HTTPClient)
				}
			},
		},
		{
			name: "no auth passes the caller transport through unchanged",
			cfg:  Config{Transport: callerCommand},
			check: func(t *testing.T, tr mcp.Transport) {
				if tr != callerCommand {
					t.Errorf("transport = %v, want the caller's transport unchanged", tr)
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tr, err := buildTransport(tt.cfg)
			if tt.wantErr {
				if err == nil {
					t.Fatal("buildTransport() = nil error, want error")
				}
				return
			}
			if err != nil {
				t.Fatalf("buildTransport() error = %v", err)
			}
			tt.check(t, tr)
		})
	}
}

func mustStreamable(t *testing.T, tr mcp.Transport) *mcp.StreamableClientTransport {
	t.Helper()
	st, ok := tr.(*mcp.StreamableClientTransport)
	if !ok {
		t.Fatalf("transport = %T, want *mcp.StreamableClientTransport", tr)
	}
	return st
}

func wantsAuthTransport(t *testing.T, c *http.Client) {
	t.Helper()
	if c == nil {
		t.Fatal("HTTPClient = nil, want a client carrying the auth transport")
	}
	if _, ok := c.Transport.(*auth.Transport); !ok {
		t.Errorf("HTTPClient.Transport = %T, want *auth.Transport", c.Transport)
	}
}

// stubRoundTripper is a no-op http.RoundTripper used as a sentinel.
type stubRoundTripper struct{}

func (*stubRoundTripper) RoundTrip(*http.Request) (*http.Response, error) { return nil, nil }
