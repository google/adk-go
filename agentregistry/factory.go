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
	"cmp"
	"context"
	"encoding/json"
	"fmt"
	"maps"
	"net/http"
	"net/url"
	"strings"

	"github.com/a2aproject/a2a-go/v2/a2a"
	"github.com/a2aproject/a2a-go/v2/a2aclient"
	"github.com/modelcontextprotocol/go-sdk/mcp"

	"google.golang.org/adk/v2/agent"
	remoteagent "google.golang.org/adk/v2/agent/remoteagent/v2"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/mcptoolset"
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

func applyMCPToolsetOptions(opts []MCPToolsetOption) egressConfig {
	var ec egressConfig
	for _, opt := range opts {
		opt(&ec)
	}
	return ec
}

// addHeaders merges h into e.headers: it adds new keys and overwrites existing
// ones, and never removes a key.
func addHeaders(e *egressConfig, h map[string]string) {
	if len(h) == 0 {
		return
	}
	if e.headers == nil {
		e.headers = make(map[string]string, len(h))
	}
	maps.Copy(e.headers, h)
}

// RemoteAgentOption customizes [Client.RemoteAgent].
type RemoteAgentOption func(*egressConfig)

// MCPToolsetOption customizes [Client.MCPToolset].
type MCPToolsetOption func(*egressConfig)

// WithA2AHTTPClient sets the HTTP client used to reach the remote A2A agent.
// A2A egress is not auto-authenticated, so set this to authenticate requests to
// the remote agent. The default ([http.DefaultClient]) has no timeout; bound
// egress on the client's Transport rather than via http.Client.Timeout, which
// is a deadline over the whole request and would truncate streaming responses.
func WithA2AHTTPClient(c *http.Client) RemoteAgentOption {
	return func(e *egressConfig) { e.httpClient = c }
}

// WithA2AHeaders adds or overwrites static headers on every request sent to the
// remote A2A agent. Repeated calls accumulate; a later value wins on a key conflict.
func WithA2AHeaders(h map[string]string) RemoteAgentOption {
	return func(e *egressConfig) { addHeaders(e, h) }
}

// WithMCPHTTPClient sets the HTTP client used to reach the MCP server. It
// overrides the default (an Application Default Credentials client for
// *.googleapis.com endpoints, else [http.DefaultClient]). The default has no
// timeout; bound egress on the client's Transport rather than via
// http.Client.Timeout, which is a deadline over the whole request and would
// truncate streaming responses.
func WithMCPHTTPClient(c *http.Client) MCPToolsetOption {
	return func(e *egressConfig) { e.httpClient = c }
}

// WithMCPHeaders adds or overwrites static headers on every request sent to the
// MCP server. Repeated calls accumulate; a later value wins on a key conflict.
func WithMCPHeaders(h map[string]string) MCPToolsetOption {
	return func(e *egressConfig) { addHeaders(e, h) }
}

// egressClient selects the HTTP client used to reach an MCP endpoint at rawURL.
// Precedence: an explicit override, then (for a Google API endpoint) the
// registry's authenticated client, then a default client. Static headers, if
// any, are layered on via a cloned client so shared clients are never mutated.
func (c *Client) egressClient(rawURL string, ec egressConfig) *http.Client {
	base := ec.httpClient
	if base == nil {
		if isGoogleAPI(rawURL) {
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

// clientWithHeaders returns a client that adds or overwrites the given static
// headers on every request. If there are no headers, base is returned unchanged.
// Otherwise base is shallow-copied so the caller's client is not mutated.
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

var _ http.RoundTripper = (*headerRoundTripper)(nil)

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

// cardTypeA2AAgentCard is the Card.Type value for an embedded A2A agent card.
const cardTypeA2AAgentCard = "A2A_AGENT_CARD"

// defaultA2AProtocolVersion is used when the registry does not report one.
const defaultA2AProtocolVersion = "0.3.0"

// RemoteAgent resolves a registered A2A agent into an [agent.Agent] usable as a
// sub-agent. name is the full agent resource name.
//
// The agent card is taken from the registry's embedded card when present, and
// otherwise synthesized from the agent's discrete fields. Egress auth is left to
// the caller: pass [WithA2AHTTPClient] (and/or [WithA2AHeaders]) to authenticate
// requests to the remote agent.
func (c *Client) RemoteAgent(ctx context.Context, name string, opts ...RemoteAgentOption) (agent.Agent, error) {
	info, err := c.GetAgent(ctx, name)
	if err != nil {
		return nil, err
	}

	card, agentName, err := agentCard(info, name)
	if err != nil {
		return nil, err
	}

	ec := applyRemoteAgentOptions(opts)
	// A2A egress is not auto-authenticated (parity with adk-python): use the
	// caller's client or the default, never the registry's ADC client.
	egress := clientWithHeaders(cmp.Or(ec.httpClient, http.DefaultClient), ec.headers)

	return remoteagent.NewA2A(remoteagent.A2AConfig{
		Name:           agentName,
		Description:    card.Description,
		AgentCard:      card,
		ClientProvider: remoteagent.NewA2AClientProvider(a2aClientFactory(card, egress)),
	})
}

type transportKey struct {
	binding a2a.TransportProtocol
	version a2a.ProtocolVersion
}

// a2aClientFactory builds an A2A client factory for the given card and HTTP
// client. Besides the SDK's current-version transports, it registers a
// compatibility transport for each (binding, protocolVersion) the card
// advertises. Agent Registry commonly reports an older A2A protocol version
// (e.g. 0.3.0) than the SDK's current one, and the a2a-go factory matches
// transports by (protocol, version); without a matching compat transport,
// client creation fails with "no compatible transports found".
func a2aClientFactory(card *a2a.AgentCard, httpClient *http.Client) *a2aclient.Factory {
	opts := []a2aclient.FactoryOption{
		a2aclient.WithJSONRPCTransport(httpClient),
		a2aclient.WithRESTTransport(httpClient),
	}
	seen := make(map[transportKey]bool)
	for _, iface := range card.SupportedInterfaces {
		// The current-version transports above already cover a2a.Version.
		if iface == nil || iface.ProtocolVersion == "" || iface.ProtocolVersion == a2a.Version {
			continue
		}
		var factory a2aclient.TransportFactory
		switch iface.ProtocolBinding {
		case a2a.TransportProtocolJSONRPC:
			factory = jsonrpcTransportFactory(httpClient)
		case a2a.TransportProtocolHTTPJSON:
			factory = restTransportFactory(httpClient)
		default:
			continue
		}
		key := transportKey{iface.ProtocolBinding, iface.ProtocolVersion}
		if seen[key] {
			continue
		}
		seen[key] = true
		opts = append(opts, a2aclient.WithCompatTransport(iface.ProtocolVersion, iface.ProtocolBinding, factory))
	}
	return a2aclient.NewFactory(opts...)
}

func restTransportFactory(httpClient *http.Client) a2aclient.TransportFactory {
	return a2aclient.TransportFactoryFn(func(_ context.Context, _ *a2a.AgentCard, iface *a2a.AgentInterface) (a2aclient.Transport, error) {
		u, err := url.Parse(iface.URL)
		if err != nil {
			return nil, fmt.Errorf("agentregistry: parsing endpoint URL %q: %w", iface.URL, err)
		}
		return a2aclient.NewRESTTransport(u, httpClient), nil
	})
}

func jsonrpcTransportFactory(httpClient *http.Client) a2aclient.TransportFactory {
	return a2aclient.TransportFactoryFn(func(_ context.Context, _ *a2a.AgentCard, iface *a2a.AgentInterface) (a2aclient.Transport, error) {
		return a2aclient.NewJSONRPCTransport(iface.URL, httpClient), nil
	})
}

// agentCard resolves an [a2a.AgentCard] plus the cleaned agent name for a
// registered agent, using the embedded card when available and otherwise
// synthesizing one. resourceName is used as a fallback name.
func agentCard(info *Agent, resourceName string) (card *a2a.AgentCard, name string, err error) {
	if info.Card != nil && info.Card.Type == cardTypeA2AAgentCard && len(info.Card.Content) > 0 {
		var embedded a2a.AgentCard
		if err := json.Unmarshal(info.Card.Content, &embedded); err != nil {
			return nil, "", fmt.Errorf("agentregistry: decoding embedded agent card for %q: %w", resourceName, err)
		}
		// Fail fast: an interface-less card would otherwise only error at the
		// first invocation, deep in the a2a client, without the resource name.
		if len(embedded.SupportedInterfaces) == 0 {
			return nil, "", fmt.Errorf("agentregistry: embedded agent card for %q has no supported interfaces", resourceName)
		}
		agentName := cleanName(embedded.Name)
		if agentName == "" {
			return nil, "", fmt.Errorf("agentregistry: embedded agent card for %q has an empty name", resourceName)
		}
		return &embedded, agentName, nil
	}

	url, version, transport, ok := connectionURI(info.Protocols, nil, protocolTypeA2AAgent, "")
	if !ok {
		return nil, "", fmt.Errorf("agentregistry: A2A connection URI not found for agent %q", resourceName)
	}

	displayName := info.DisplayName
	if displayName == "" {
		displayName = resourceName
	}
	agentName := cleanName(displayName)
	if agentName == "" {
		return nil, "", fmt.Errorf("agentregistry: cannot derive a non-empty agent name for %q", resourceName)
	}

	// Default to HTTP+JSON for an unrecognized binding (as adk-python does).
	if transport == "" {
		transport = a2a.TransportProtocolHTTPJSON
	}
	protocolVersion := version
	if protocolVersion == "" {
		protocolVersion = defaultA2AProtocolVersion
	}

	card = &a2a.AgentCard{
		Name:        agentName,
		Description: info.Description,
		Version:     info.Version,
		SupportedInterfaces: []*a2a.AgentInterface{{
			URL:             url,
			ProtocolBinding: transport,
			ProtocolVersion: a2a.ProtocolVersion(protocolVersion),
		}},
		Skills:             toA2ASkills(info.Skills),
		DefaultInputModes:  []string{"text"},
		DefaultOutputModes: []string{"text"},
	}
	return card, agentName, nil
}

func toA2ASkills(skills []Skill) []a2a.AgentSkill {
	if len(skills) == 0 {
		return nil
	}
	out := make([]a2a.AgentSkill, 0, len(skills))
	for _, s := range skills {
		out = append(out, a2a.AgentSkill{
			ID:          s.ID,
			Name:        s.Name,
			Description: s.Description,
			Tags:        s.Tags,
			Examples:    s.Examples,
		})
	}
	return out
}

// MCPToolset resolves a registered MCP server into a [tool.Toolset] backed by a
// streamable-HTTP MCP connection. name is the full MCP server resource name.
//
// The endpoint is resolved preferring the JSONRPC binding, then HTTP_JSON. By
// default, requests to *.googleapis.com endpoints are authenticated with the
// registry's Application Default Credentials; use [WithMCPHTTPClient] and/or
// [WithMCPHeaders] to override or augment egress.
func (c *Client) MCPToolset(ctx context.Context, name string, opts ...MCPToolsetOption) (tool.Toolset, error) {
	server, err := c.GetMCPServer(ctx, name)
	if err != nil {
		return nil, err
	}

	uri, ok := mcpEndpointURI(server)
	if !ok {
		return nil, fmt.Errorf("agentregistry: MCP server endpoint URI not found for %q", name)
	}

	ec := applyMCPToolsetOptions(opts)
	egress := c.egressClient(uri, ec)

	return mcptoolset.New(mcptoolset.Config{
		Transport: &mcp.StreamableClientTransport{
			Endpoint:   uri,
			HTTPClient: egress,
		},
	})
}

// mcpEndpointURI returns the MCP server's connection URI, preferring the JSONRPC
// binding and falling back to HTTP_JSON (parity with adk-python).
func mcpEndpointURI(server *MCPServer) (string, bool) {
	if uri, _, _, ok := connectionURI(server.Protocols, server.Interfaces, "", a2a.TransportProtocolJSONRPC); ok {
		return uri, true
	}
	if uri, _, _, ok := connectionURI(server.Protocols, server.Interfaces, "", a2a.TransportProtocolHTTPJSON); ok {
		return uri, true
	}
	return "", false
}
