// Copyright 2025 Google LLC
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
	"context"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/modelcontextprotocol/go-sdk/mcp"
)

func TestGenerateSessionKey(t *testing.T) {
	tests := []struct {
		name     string
		headers1 map[string]string
		headers2 map[string]string
		wantSame bool
	}{
		{
			name:     "empty headers produce default key",
			headers1: map[string]string{},
			headers2: map[string]string{},
			wantSame: true,
		},
		{
			name:     "nil headers produce default key",
			headers1: nil,
			headers2: nil,
			wantSame: true,
		},
		{
			name:     "identical headers produce same key",
			headers1: map[string]string{"Authorization": "Bearer token123"},
			headers2: map[string]string{"Authorization": "Bearer token123"},
			wantSame: true,
		},
		{
			name:     "different header values produce different keys",
			headers1: map[string]string{"Authorization": "Bearer token123"},
			headers2: map[string]string{"Authorization": "Bearer token456"},
			wantSame: false,
		},
		{
			name:     "different header names produce different keys",
			headers1: map[string]string{"Authorization": "Bearer token123"},
			headers2: map[string]string{"X-API-Key": "Bearer token123"},
			wantSame: false,
		},
		{
			name:     "multiple headers in different order produce same key",
			headers1: map[string]string{"Authorization": "Bearer token", "X-API-Key": "key123"},
			headers2: map[string]string{"X-API-Key": "key123", "Authorization": "Bearer token"},
			wantSame: true,
		},
		{
			name:     "subset of headers produces different key",
			headers1: map[string]string{"Authorization": "Bearer token", "X-API-Key": "key123"},
			headers2: map[string]string{"Authorization": "Bearer token"},
			wantSame: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			clientTransport := &mcp.SSEClientTransport{}

			client := mcp.NewClient(&mcp.Implementation{Name: "test_client", Version: "v1.0.0"}, nil)
			sm := NewSessionManager(client, clientTransport)

			key1 := sm.generateSessionKey(tt.headers1)
			key2 := sm.generateSessionKey(tt.headers2)

			if tt.wantSame && key1 != key2 {
				t.Errorf("expected same keys, got %q and %q", key1, key2)
			}
			if !tt.wantSame && key1 == key2 {
				t.Errorf("expected different keys, got %q for both", key1)
			}
		})
	}
}

func TestGenerateSessionKey_IgnoresHeadersForNonHTTPTransport(t *testing.T) {
	ctx := context.Background()
	clientTransport, _ := mcp.NewInMemoryTransports()

	client := mcp.NewClient(&mcp.Implementation{Name: "test_client", Version: "v1.0.0"}, nil)
	sm := NewSessionManager(client, clientTransport)

	key1 := sm.generateSessionKey(map[string]string{"Authorization": "Bearer token123"})
	key2 := sm.generateSessionKey(map[string]string{"Authorization": "Bearer token456"})

	if key1 != "default" || key2 != "default" {
		t.Fatalf("expected default keys for non-HTTP transport, got %q and %q", key1, key2)
	}

	server := mcp.NewServer(&mcp.Implementation{Name: "test_server", Version: "v1.0.0"}, nil)
	_, err := server.Connect(ctx, clientTransport, nil)
	_ = err
}

func TestDefaultKey(t *testing.T) {
	clientTransport, _ := mcp.NewInMemoryTransports()

	client := mcp.NewClient(&mcp.Implementation{Name: "test_client", Version: "v1.0.0"}, nil)
	sm := NewSessionManager(client, clientTransport)

	emptyKey := sm.generateSessionKey(map[string]string{})
	nilKey := sm.generateSessionKey(nil)

	if emptyKey != "default" {
		t.Errorf("expected 'default' key for empty headers, got %q", emptyKey)
	}
	if nilKey != "default" {
		t.Errorf("expected 'default' key for nil headers, got %q", nilKey)
	}
}

func TestGetSession(t *testing.T) {
	ctx := t.Context()

	clientTransport, serverTransport := mcp.NewInMemoryTransports()
	server := mcp.NewServer(&mcp.Implementation{Name: "test_server", Version: "v1.0.0"}, nil)
	_, err := server.Connect(ctx, serverTransport, nil)
	if err != nil {
		t.Fatalf("failed to connect server: %v", err)
	}

	client := mcp.NewClient(&mcp.Implementation{Name: "test_client", Version: "v1.0.0"}, nil)
	sm := NewSessionManager(client, clientTransport)
	defer sm.Close()

	headers := map[string]string{"Authorization": "Bearer token123"}

	session1, err := sm.GetSession(ctx, headers)
	if err != nil {
		t.Fatalf("failed to get session: %v", err)
	}
	if session1 == nil {
		t.Fatal("expected non-nil session")
	}

	session2, err := sm.GetSession(ctx, headers)
	if err != nil {
		t.Fatalf("failed to get session again: %v", err)
	}
	if session2 != session1 {
		t.Error("expected same session instance for identical headers")
	}

	sm.mu.RLock()
	sessionCount := len(sm.sessions)
	sm.mu.RUnlock()
	if sessionCount != 1 {
		t.Errorf("expected 1 session, got %d", sessionCount)
	}
}

func TestGetSessionWithNilHeaders(t *testing.T) {
	ctx := t.Context()

	clientTransport, serverTransport := mcp.NewInMemoryTransports()
	server := mcp.NewServer(&mcp.Implementation{Name: "test_server", Version: "v1.0.0"}, nil)
	_, err := server.Connect(ctx, serverTransport, nil)
	if err != nil {
		t.Fatalf("failed to connect server: %v", err)
	}

	client := mcp.NewClient(&mcp.Implementation{Name: "test_client", Version: "v1.0.0"}, nil)
	sm := NewSessionManager(client, clientTransport)
	defer sm.Close()

	session, err := sm.GetSession(ctx, nil)
	if err != nil {
		t.Fatalf("failed to get session with nil headers: %v", err)
	}
	if session == nil {
		t.Fatal("expected non-nil session")
	}

	key := sm.generateSessionKey(nil)
	if key != "default" {
		t.Errorf("expected default key for nil headers, got %q", key)
	}
}

func TestClose(t *testing.T) {
	ctx := t.Context()

	clientTransport, serverTransport := mcp.NewInMemoryTransports()
	server := mcp.NewServer(&mcp.Implementation{Name: "test_server", Version: "v1.0.0"}, nil)
	_, err := server.Connect(ctx, serverTransport, nil)
	if err != nil {
		t.Fatalf("failed to connect server: %v", err)
	}

	client := mcp.NewClient(&mcp.Implementation{Name: "test_client", Version: "v1.0.0"}, nil)
	sm := NewSessionManager(client, clientTransport)

	headers := map[string]string{"Authorization": "Bearer token123"}

	_, err = sm.GetSession(ctx, headers)
	if err != nil {
		t.Fatalf("failed to create session: %v", err)
	}

	sm.mu.RLock()
	sessionCount := len(sm.sessions)
	sm.mu.RUnlock()
	if sessionCount != 1 {
		t.Errorf("expected 1 session before close, got %d", sessionCount)
	}

	err = sm.Close()
	if err != nil {
		t.Errorf("Close() returned error: %v", err)
	}

	sm.mu.RLock()
	sessionCount = len(sm.sessions)
	sm.mu.RUnlock()
	if sessionCount != 0 {
		t.Errorf("expected 0 sessions after close, got %d", sessionCount)
	}
}

func TestIsSessionValid(t *testing.T) {
	ctx := t.Context()

	clientTransport, serverTransport := mcp.NewInMemoryTransports()
	server := mcp.NewServer(&mcp.Implementation{Name: "test_server", Version: "v1.0.0"}, nil)
	if _, err := server.Connect(ctx, serverTransport, nil); err != nil {
		t.Fatalf("failed to connect server: %v", err)
	}

	client := mcp.NewClient(&mcp.Implementation{Name: "test_client", Version: "v1.0.0"}, nil)
	sm := NewSessionManager(client, clientTransport)
	defer sm.Close()

	session, err := sm.GetSession(ctx, nil)
	if err != nil {
		t.Fatalf("failed to get session: %v", err)
	}

	if !sm.isSessionValid(ctx, session) {
		t.Fatal("expected active session to be valid")
	}

	if err := session.Close(); err != nil {
		t.Fatalf("failed to close session: %v", err)
	}

	if sm.isSessionValid(ctx, session) {
		t.Fatal("expected closed session to be invalid")
	}
}

func TestHeaderTransport(t *testing.T) {
	tests := []struct {
		name           string
		headers        map[string]string
		wantHeaders    map[string]string
		existingHeader map[string]string
	}{
		{
			name:        "adds single header",
			headers:     map[string]string{"Authorization": "Bearer token123"},
			wantHeaders: map[string]string{"Authorization": "Bearer token123"},
		},
		{
			name: "adds multiple headers",
			headers: map[string]string{
				"Authorization": "Bearer token123",
				"X-API-Key":     "key456",
			},
			wantHeaders: map[string]string{
				"Authorization": "Bearer token123",
				"X-API-Key":     "key456",
			},
		},
		{
			name:        "no headers added when empty",
			headers:     map[string]string{},
			wantHeaders: map[string]string{},
		},
		{
			name:           "overwrites existing header",
			headers:        map[string]string{"Authorization": "Bearer new-token"},
			existingHeader: map[string]string{"Authorization": "Bearer old-token"},
			wantHeaders:    map[string]string{"Authorization": "Bearer new-token"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var capturedReq *http.Request
			baseTransport := &mockRoundTripper{
				roundTripFunc: func(req *http.Request) (*http.Response, error) {
					capturedReq = req
					return &http.Response{
						StatusCode: http.StatusOK,
						Body:       io.NopCloser(strings.NewReader("OK")),
						Header:     make(http.Header),
					}, nil
				},
			}

			ht := &headerTransport{
				Base:    baseTransport,
				Headers: tt.headers,
			}

			req, err := http.NewRequest("GET", "http://example.com", nil)
			if err != nil {
				t.Fatalf("failed to create request: %v", err)
			}

			for k, v := range tt.existingHeader {
				req.Header.Set(k, v)
			}

			_, err = ht.RoundTrip(req)
			if err != nil {
				t.Fatalf("RoundTrip failed: %v", err)
			}

			for key, wantValue := range tt.wantHeaders {
				gotValue := capturedReq.Header.Get(key)
				if gotValue != wantValue {
					t.Errorf("header %q: got %q, want %q", key, gotValue, wantValue)
				}
			}

			if len(tt.headers) == 0 && len(tt.existingHeader) == 0 {
				if len(capturedReq.Header) > 0 {
					t.Errorf("expected no headers, got %v", capturedReq.Header)
				}
			}
		})
	}
}

func TestHeaderTransportBase(t *testing.T) {
	tests := []struct {
		name          string
		baseTransport http.RoundTripper
		wantDefault   bool
	}{
		{
			name: "uses provided base transport",
			baseTransport: &mockRoundTripper{
				roundTripFunc: func(req *http.Request) (*http.Response, error) {
					return &http.Response{StatusCode: http.StatusOK, Body: io.NopCloser(strings.NewReader("custom"))}, nil
				},
			},
			wantDefault: false,
		},
		{
			name:          "uses default transport when nil",
			baseTransport: nil,
			wantDefault:   true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ht := &headerTransport{
				Base:    tt.baseTransport,
				Headers: map[string]string{},
			}

			baseRT := ht.base()
			if tt.wantDefault && baseRT != http.DefaultTransport {
				t.Error("expected default transport when Base is nil")
			}
			if !tt.wantDefault && baseRT == http.DefaultTransport {
				t.Error("expected custom transport, got default")
			}
		})
	}
}

func TestWrapHTTPClient(t *testing.T) {
	tests := []struct {
		name       string
		httpClient *http.Client
		headers    map[string]string
	}{
		{
			name:       "wraps nil client",
			httpClient: nil,
			headers:    map[string]string{"Authorization": "Bearer token"},
		},
		{
			name: "wraps existing client",
			httpClient: &http.Client{
				Timeout: 30,
			},
			headers: map[string]string{"X-API-Key": "key123"},
		},
		{
			name: "preserves client settings",
			httpClient: &http.Client{
				Timeout: 60,
			},
			headers: map[string]string{"Authorization": "Bearer token"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			wrapped := wrapHTTPClient(tt.httpClient, tt.headers)

			if wrapped == nil {
				t.Fatal("expected non-nil wrapped client")
			}

			ht, ok := wrapped.Transport.(*headerTransport)
			if !ok {
				t.Fatal("expected Transport to be *headerTransport")
			}

			if diff := cmp.Diff(tt.headers, ht.Headers); diff != "" {
				t.Errorf("headers mismatch (-want +got):\n%s", diff)
			}
			// Verify timeout is preserved
			if tt.httpClient != nil && wrapped.Timeout != tt.httpClient.Timeout {
				t.Errorf("timeout mismatch: got %v, want %v", wrapped.Timeout, tt.httpClient.Timeout)
			}
		})
	}
}

type mockRoundTripper struct {
	roundTripFunc func(*http.Request) (*http.Response, error)
}

func (m *mockRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	return m.roundTripFunc(req)
}
