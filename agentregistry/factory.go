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

package agentregistry

import (
	"net/http"
	"net/url"
	"strings"
)

// egressConfig holds the resolved options shared by the factory helpers.
type egressConfig struct {
	httpClient *http.Client
	headers    map[string]string
}

func applyRemoteAgentOptions(opts []RemoteAgentOption) egressConfig {
	var ec egressConfig
	for _, opt := range opts {
		opt(&ec)
	}
	return ec
}

func applyMcpToolsetOptions(opts []McpToolsetOption) egressConfig {
	var ec egressConfig
	for _, opt := range opts {
		opt(&ec)
	}
	return ec
}

// RemoteAgentOption customizes [Client.RemoteAgent].
type RemoteAgentOption func(*egressConfig)

// McpToolsetOption customizes [Client.McpToolset].
type McpToolsetOption func(*egressConfig)

// WithA2AHTTPClient sets the HTTP client used to reach the remote A2A agent.
func WithA2AHTTPClient(c *http.Client) RemoteAgentOption {
	return func(e *egressConfig) { e.httpClient = c }
}

// WithA2AHeaders adds static headers to every request sent to the remote A2A
// agent.
func WithA2AHeaders(h map[string]string) RemoteAgentOption {
	return func(e *egressConfig) { e.headers = h }
}

// WithMcpHTTPClient sets the HTTP client used to reach the MCP server. It
// overrides the default (an Application Default Credentials client for
// *.googleapis.com endpoints).
func WithMcpHTTPClient(c *http.Client) McpToolsetOption {
	return func(e *egressConfig) { e.httpClient = c }
}

// WithMcpHeaders adds static headers to every request sent to the MCP server.
func WithMcpHeaders(h map[string]string) McpToolsetOption {
	return func(e *egressConfig) { e.headers = h }
}

// egressClient selects the HTTP client used to reach an endpoint at rawURL.
// Precedence: an explicit override, then (only when autoADC is set and the
// endpoint is a Google API) the registry's authenticated client, then a default
// client. Static headers, if any, are layered on via a cloned client so shared
// clients are never mutated.
func (c *Client) egressClient(rawURL string, ec egressConfig, autoADC bool) *http.Client {
	base := ec.httpClient
	if base == nil {
		if autoADC && isGoogleAPI(rawURL) {
			base = c.httpClient
		} else {
			base = http.DefaultClient
		}
	}
	return clientWithHeaders(base, ec.headers)
}

// isGoogleAPI reports whether rawURL points at a Google API endpoint. It mirrors
// adk-python's _is_google_api.
func isGoogleAPI(rawURL string) bool {
	u, err := url.Parse(rawURL)
	if err != nil {
		return false
	}
	host := u.Hostname()
	return host == "googleapis.com" || strings.HasSuffix(host, ".googleapis.com")
}

// clientWithHeaders returns a client that adds the given static headers to every
// request. If there are no headers, base is returned unchanged. Otherwise base
// is shallow-copied so the caller's client is not mutated.
func clientWithHeaders(base *http.Client, headers map[string]string) *http.Client {
	if len(headers) == 0 {
		return base
	}
	clone := *base
	clone.Transport = &headerRoundTripper{base: base.Transport, headers: headers}
	return &clone
}

// headerRoundTripper adds a fixed set of headers to each request.
type headerRoundTripper struct {
	base    http.RoundTripper
	headers map[string]string
}

func (h *headerRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	rt := h.base
	if rt == nil {
		rt = http.DefaultTransport
	}
	req = req.Clone(req.Context())
	for k, v := range h.headers {
		req.Header.Set(k, v)
	}
	return rt.RoundTrip(req)
}
