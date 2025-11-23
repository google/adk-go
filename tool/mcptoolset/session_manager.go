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
	"crypto/md5"
	"encoding/hex"
	"fmt"
	"net/http"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/modelcontextprotocol/go-sdk/mcp"
)

const (
	defaultSessionKey  = "default"
	defaultPingTimeout = 2 * time.Second
)

// sessionManager manages MCP client sessions with header-based pooling
type sessionManager struct {
	client    *mcp.Client
	transport mcp.Transport

	mu       sync.RWMutex
	sessions map[string]*sessionEntry
}

type sessionEntry struct {
	session *mcp.ClientSession
	headers map[string]string
}

// newSessionManager creates a new session manager
func newSessionManager(client *mcp.Client, transport mcp.Transport) *sessionManager {
	return &sessionManager{
		client:    client,
		transport: transport,
		sessions:  make(map[string]*sessionEntry),
	}
}

// headersAffectSession returns true only for HTTP-based transports where
// headers are actually used by the connection.
func (sm *sessionManager) headersAffectSession() bool {
	switch sm.transport.(type) {
	case *mcp.SSEClientTransport, *mcp.StreamableClientTransport:
		return true
	default:
		return false
	}
}

// generateSessionKey creates a hash-based key from headers
func (sm *sessionManager) generateSessionKey(headers map[string]string) string {
	// For non-HTTP transports (e.g., stdio, in-memory), headers don't apply,
	// so we always pool into the same session.
	if !sm.headersAffectSession() {
		return defaultSessionKey
	}
	if len(headers) == 0 {
		return defaultSessionKey
	}

	keys := make([]string, 0, len(headers))
	for k := range headers {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	var pairs []string
	for _, k := range keys {
		pairs = append(pairs, fmt.Sprintf("%q:%q", k, headers[k]))
	}
	jsonStr := "{" + strings.Join(pairs, ",") + "}"

	h := md5.Sum([]byte(jsonStr))
	return hex.EncodeToString(h[:])
}

// GetSession returns a session for the given headers, creating if necessary
func (sm *sessionManager) GetSession(ctx context.Context, headers map[string]string) (*mcp.ClientSession, error) {
	key := sm.generateSessionKey(headers)

	sm.mu.RLock()
	entry, ok := sm.sessions[key]
	sm.mu.RUnlock()

	if ok && sm.isSessionValid(ctx, entry.session) {
		return entry.session, nil
	}

	sm.mu.Lock()
	defer sm.mu.Unlock()

	if entry, ok := sm.sessions[key]; ok && sm.isSessionValid(ctx, entry.session) {
		return entry.session, nil
	}

	wrappedTransport := sm.wrapTransportWithHeaders(headers)

	session, err := sm.client.Connect(ctx, wrappedTransport, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create session: %w", err)
	}

	sm.sessions[key] = &sessionEntry{
		session: session,
		headers: headers,
	}

	return session, nil
}

// isSessionValid checks if a session is still usable
func (sm *sessionManager) isSessionValid(ctx context.Context, session *mcp.ClientSession) bool {
	if session == nil {
		return false
	}

	pingCtx := ctx
	if _, hasDeadline := ctx.Deadline(); !hasDeadline {
		var cancel context.CancelFunc
		pingCtx, cancel = context.WithTimeout(ctx, defaultPingTimeout)
		defer cancel()
	}

	if err := session.Ping(pingCtx, nil); err != nil {
		return false
	}
	return true
}

// wrapTransportWithHeaders creates a transport that injects headers
func (sm *sessionManager) wrapTransportWithHeaders(headers map[string]string) mcp.Transport {
	switch t := sm.transport.(type) {

	case *mcp.SSEClientTransport:
		return &mcp.SSEClientTransport{
			Endpoint:   t.Endpoint,
			HTTPClient: wrapHTTPClient(t.HTTPClient, headers),
		}

	case *mcp.StreamableClientTransport:
		return &mcp.StreamableClientTransport{
			Endpoint:   t.Endpoint,
			HTTPClient: wrapHTTPClient(t.HTTPClient, headers),
		}

	default:
		return sm.transport
	}
}

func wrapHTTPClient(httpClient *http.Client, headers map[string]string) *http.Client {
	if httpClient == nil {
		httpClient = &http.Client{}
	}

	return &http.Client{
		Transport: &headerTransport{
			Base:    httpClient.Transport,
			Headers: headers,
		},
		CheckRedirect: httpClient.CheckRedirect,
		Jar:           httpClient.Jar,
		Timeout:       httpClient.Timeout,
	}
}

// Close closes all sessions
func (sm *sessionManager) Close() error {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	var errs []error
	for _, entry := range sm.sessions {
		if err := entry.session.Close(); err != nil {
			errs = append(errs, err)
		}
	}

	sm.sessions = make(map[string]*sessionEntry)

	if len(errs) > 0 {
		return fmt.Errorf("errors closing sessions: %v", errs)
	}
	return nil
}

type headerTransport struct {
	Base    http.RoundTripper
	Headers map[string]string
}

// RoundTrip adds the configured headers to the request.
func (t *headerTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	if len(t.Headers) == 0 {
		return t.base().RoundTrip(req)
	}

	req2 := req.Clone(req.Context())
	for key, value := range t.Headers {
		req2.Header.Set(key, value)
	}
	return t.base().RoundTrip(req2)
}

func (t *headerTransport) base() http.RoundTripper {
	if t.Base != nil {
		return t.Base
	}
	return http.DefaultTransport
}
