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

// Package mcptoolset provides an MCP tool set.
package mcptoolset

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"

	"github.com/modelcontextprotocol/go-sdk/mcp"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/auth"
	"google.golang.org/adk/v2/tool"
)

// MetadataProvider supplies request-scoped metadata for MCP tool calls. The
// returned map is sent in the `_meta` field of mcp.CallToolParams, letting an
// agent forward values such as tracing IDs or tenant identifiers from the
// incoming request to the MCP server.
//
// It is called before each tool call is sent; when a tool requires confirmation,
// it runs only for the confirmed call, with the context of the request carrying
// the confirmation. It may be called concurrently. The returned map, and
// anything reachable from it, must not be modified after the provider returns.
// Returning a nil map contributes no metadata, while returning an error or a
// reserved MCP key ("progressToken" or any "io.modelcontextprotocol/*" key)
// fails the tool call.
type MetadataProvider func(ctx agent.Context) (map[string]any, error)

// New returns MCP ToolSet.
// MCP ToolSet connects to a MCP Server, retrieves MCP Tools into ADK Tools and
// passes them to the LLM.
// It uses https://github.com/modelcontextprotocol/go-sdk for MCP communication.
// MCP session is created lazily on the first request to LLM.
// New returns an error if neither Transport nor Endpoint is provided.
//
// The returned toolset implements [io.Closer]. Callers must close it when it
// is no longer in use. Close waits for ongoing connection setup and MCP
// operations to finish, and is safe to call concurrently or more than once.
// Further MCP operations fail with [mcp.ErrConnectionClosed]; closing does not
// affect other sessions created with Config.Client.
// When using [tool.FilterToolset], retain the original toolset to close it.
//
// Usage: create MCP ToolSet with mcptoolset.New() and provide it to the
// LLMAgent in the llmagent.Config.
//
// Example:
//
//	mcpTools, err := mcptoolset.New(mcptoolset.Config{
//		Transport: &mcp.CommandTransport{Command: exec.Command("myserver")},
//	})
//	if err != nil {
//		return err
//	}
//	defer mcpTools.(io.Closer).Close()
//
//	a, err := llmagent.New(llmagent.Config{
//		Name:        "agent_name",
//		Model:       model,
//		Description: "...",
//		Instruction: "...",
//		Toolsets:    []tool.Toolset{mcpTools},
//	})
//	if err != nil {
//		return err
//	}
//	_ = a // Use a with a runner before returning, while mcpTools is still open.
func New(cfg Config) (tool.Toolset, error) {
	transport, err := buildTransport(cfg)
	if err != nil {
		return nil, err
	}
	var client sessionClient
	if cfg.Auth != nil {
		client = newCredentialSessions(cfg.Client, transport, cfg.Auth)
	} else {
		client = newConnectionRefresher(cfg.Client, transport)
	}
	return &set{
		mcpClient:                   client,
		toolFilter:                  cfg.ToolFilter,
		requireConfirmation:         cfg.RequireConfirmation,
		requireConfirmationProvider: cfg.RequireConfirmationProvider,
		metadataProvider:            cfg.MetadataProvider,
		auth:                        cfg.Auth,
	}, nil
}

// buildTransport resolves the MCP transport from cfg. When Transport is nil and
// Endpoint is set, it builds a streamable HTTP transport to that URL. A
// streamable HTTP or SSE transport is copied, and the copy's HTTP client sends
// the ADK tracking headers. When Config.Auth is set, the client also applies a
// per-request auth RoundTripper, which requires a streamable HTTP transport.
func buildTransport(cfg Config) (mcp.Transport, error) {
	transport := cfg.Transport
	if transport == nil && cfg.Endpoint != "" {
		transport = &mcp.StreamableClientTransport{Endpoint: cfg.Endpoint}
	}
	if transport == nil {
		return nil, fmt.Errorf("mcptoolset: set Config.Transport or Config.Endpoint")
	}
	if _, ok := transport.(*mcp.StreamableClientTransport); cfg.Auth != nil && !ok {
		return nil, fmt.Errorf("mcptoolset: Config.Auth requires a streamable HTTP transport; "+
			"set Config.Endpoint or pass a *mcp.StreamableClientTransport (got %T)", transport)
	}
	switch t := transport.(type) {
	case *mcp.StreamableClientTransport:
		if t == nil {
			return nil, fmt.Errorf("mcptoolset: Config.Transport is a nil %T", t)
		}
		stCopy := *t
		stCopy.HTTPClient = httpClient(t.HTTPClient, cfg.Auth)
		return &stCopy, nil
	case *mcp.SSEClientTransport:
		if t == nil {
			return nil, fmt.Errorf("mcptoolset: Config.Transport is a nil %T", t)
		}
		sseCopy := *t
		sseCopy.HTTPClient = httpClient(t.HTTPClient, nil)
		return &sseCopy, nil
	}
	return transport, nil
}

// httpClient returns a shallow copy of base whose Transport adds the ADK
// tracking headers and, when provider is non-nil, applies provider to every
// request. A nil base stands for http.DefaultClient, as it does in the SDK,
// except with a provider (see Config.Auth).
func httpClient(base *http.Client, provider auth.CredentialProvider) *http.Client {
	var c http.Client
	switch {
	case base != nil:
		c = *base
	case provider == nil:
		c = *http.DefaultClient
	}
	c.Transport = &trackingTransport{base: c.Transport}
	if provider != nil {
		c.Transport = &auth.Transport{Provider: consentTolerantProvider{inner: provider}, Base: c.Transport}
	}
	return &c
}

// consentTolerantProvider is what the MCP transport sees. The transport also
// carries initialize and tools/list, which run before any tool call and have
// nowhere to pause, so a provider that still needs interactive consent is
// treated there as "no credential yet": the request goes out unauthenticated
// and the tool's own preflight (consentPreflight) raises the consent on the
// first call. Without this, Tools() failed on the client side for every server
// and the round-trip was unreachable for a user who had not consented.
type consentTolerantProvider struct {
	inner auth.CredentialProvider
}

// noCredential is the credential applied while consent is pending: it sets no
// header, so a server that authenticates listing answers with its own 401.
type noCredential struct{}

func (noCredential) Apply(http.Header) error { return nil }

func (p consentTolerantProvider) Credential(ctx context.Context) (auth.Credential, error) {
	cred, err := p.inner.Credential(ctx)
	if err == nil {
		return cred, nil
	}
	var consent *auth.ConsentRequiredError
	if errors.As(err, &consent) && consent != nil {
		return noCredential{}, nil
	}
	return nil, err
}

// Refresh forwards to the inner provider when it can refresh. A 401 on a
// request that went out with noCredential has nothing to refresh: the user has
// not consented, so the rejection stands.
func (p consentTolerantProvider) Refresh(ctx context.Context, rejected auth.Credential) (auth.Credential, error) {
	if _, pending := rejected.(noCredential); pending {
		return nil, errors.New("mcptoolset: no credential to refresh while consent is pending")
	}
	rp, ok := p.inner.(auth.RefreshingProvider)
	if !ok {
		return nil, errors.New("mcptoolset: provider does not refresh")
	}
	return rp.Refresh(ctx, rejected)
}

// Config provides initial configuration for the MCP ToolSet.
type Config struct {
	// Client is an optional custom MCP client to use. If nil, a default client will be created.
	Client *mcp.Client
	// Transport that will be used to connect to MCP server.
	//
	// A *mcp.StreamableClientTransport or *mcp.SSEClientTransport is copied
	// together with its HTTPClient when New is called, so later changes to
	// either have no effect. A nil HTTPClient stands for http.DefaultClient,
	// except with Auth (see there). The copied client puts the ADK client
	// labels ("google-adk/<version> gl-go/<version>") at the front of the
	// User-Agent and X-Goog-Api-Client headers of every request. The
	// HTTPClient's own Transport runs after that. A User-Agent it
	// sets with Header.Set replaces the labels, and one it adds with
	// Header.Add is not sent, because net/http sends only the first value.
	// Other transport types, including a wrapper such as *mcp.LoggingTransport,
	// are used as given.
	Transport mcp.Transport

	// Endpoint, when set and Transport is nil, builds a streamable HTTP
	// transport (mcp.StreamableClientTransport) targeting this URL. Ignored when
	// Transport is set.
	Endpoint string

	// Auth, when set, resolves and applies a credential to every outgoing MCP
	// HTTP request via a context-aware RoundTripper. It requires a streamable
	// HTTP transport: set Endpoint, or pass a *mcp.StreamableClientTransport.
	// Combining Auth with a non-HTTP transport (e.g. a stdio command) is a
	// configuration error. See package google.golang.org/adk/v2/auth.
	//
	// With Auth set, each distinct credential the provider resolves gets an
	// MCP session of its own, so a server that binds sessions to the user who
	// created them serves every user, while a user-independent credential such
	// as auth.StaticToken keeps one session. A session idle for 10 minutes is
	// closed, and at most 64 are kept open unless more are in use at once.
	//
	// With Auth and a nil HTTPClient, requests use a zero http.Client rather
	// than a copy of http.DefaultClient, so the credential never reaches
	// http.DefaultClient's Transport or CheckRedirect, and cookies set for one
	// user's requests are not kept in its Jar. Requests are sent by
	// http.DefaultTransport.
	//
	// Don't also set OAuthHandler on a supplied *mcp.StreamableClientTransport:
	// Auth is applied last and overwrites the Authorization header, so the two
	// would fight over the same request.
	//
	// A provider that needs interactive 3-legged consent (one that can return
	// auth.ConsentRequiredError) raises it from the tool call, which can pause
	// the run to ask the user. Tool listing runs before any tool call and cannot
	// pause, so while consent is pending the listing requests go out without a
	// credential rather than failing on the client side. A server that itself
	// authenticates listing then answers 401 and Tools fails: such a server
	// needs a credential that resolves without consent — non-interactive, or
	// already consented and cached.
	//
	// An interactive provider cannot be combined with a confirmation on the
	// same call: two human-in-the-loop round-trips on one call are not
	// supported, and a call that needs both fails with an error saying so
	// rather than pausing on a prompt no client receives. With
	// RequireConfirmationProvider this is decided per call, so a provider that
	// gates only some tools leaves the others free to pause for consent.
	Auth auth.CredentialProvider

	// Deprecated: use tool.FilterToolset instead.
	// ToolFilter selects tools for which tool.Predicate returns true.
	// If ToolFilter is nil, then all tools are returned.
	// tool.StringPredicate can be convenient if there's a known fixed list of tool names.
	ToolFilter tool.Predicate

	// RequireConfirmation flags whether the tools from this toolset must always ask for user confirmation
	// before execution. If set to true, the ADK framework will automatically initiate
	// a Human-in-the-Loop (HITL) confirmation request when a tool is invoked.
	RequireConfirmation bool

	// RequireConfirmationProvider allows for dynamic determination of whether
	// user confirmation is needed. This field is a function called at runtime to decide if
	// a confirmation request should be sent. The function takes the toolName and tool's input parameters as arguments.
	// This provider offers more flexibility than the static RequireConfirmation flag,
	// enabling conditional confirmation based on the invocation details.
	// If set, this takes precedence over the RequireConfirmation flag.
	//
	// Required signature for a provider function:
	// func(name string, toolInput any) bool
	// Returning true means confirmation is required.
	RequireConfirmationProvider tool.ConfirmationProvider

	// MetadataProvider, when set, is called before each tool call is sent to
	// build the request's `_meta` entries.
	MetadataProvider MetadataProvider
}

// sessionClient is the MCP connection a set uses: one shared session, or one
// per credential when Config.Auth is set.
type sessionClient interface {
	MCPClient
	io.Closer
}

type set struct {
	mcpClient                   sessionClient
	toolFilter                  tool.Predicate
	requireConfirmation         bool
	requireConfirmationProvider tool.ConfirmationProvider
	metadataProvider            MetadataProvider
	auth                        auth.CredentialProvider
}

func (s *set) Close() error {
	return s.mcpClient.Close()
}

var _ io.Closer = (*set)(nil)

func (*set) Name() string {
	return "mcp_tool_set"
}

func (*set) Description() string {
	return "Connects to a MCP Server, retrieves MCP Tools into ADK Tools."
}

func (*set) IsLongRunning() bool {
	return false
}

// Tools fetch MCP tools from the server, convert to adk tool.Tool and filter by name.
func (s *set) Tools(ctx agent.ReadonlyContext) ([]tool.Tool, error) {
	mcpTools, err := s.mcpClient.ListTools(ctx)
	if err != nil {
		return nil, err
	}

	var adkTools []tool.Tool
	for _, mcpTool := range mcpTools {
		t, err := convertTool(mcpTool, s)
		if err != nil {
			return nil, fmt.Errorf("failed to convert MCP tool %q to adk tool: %w", mcpTool.Name, err)
		}

		if s.toolFilter != nil && !s.toolFilter(ctx, t) {
			continue
		}

		adkTools = append(adkTools, t)
	}

	return adkTools, nil
}
