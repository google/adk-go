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
	"fmt"
	"net/http"

	"github.com/modelcontextprotocol/go-sdk/mcp"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/auth"
	"google.golang.org/adk/v2/internal/llminternal"
	"google.golang.org/adk/v2/tool"
)

// New returns MCP ToolSet.
// MCP ToolSet connects to a MCP Server, retrieves MCP Tools into ADK Tools and
// passes them to the LLM.
// It uses https://github.com/modelcontextprotocol/go-sdk for MCP communication.
// MCP session is created lazily on the first request to LLM.
//
// Usage: create MCP ToolSet with mcptoolset.New() and provide it to the
// LLMAgent in the llmagent.Config.
//
// Example:
//
//	llmagent.New(llmagent.Config{
//		Name:        "agent_name",
//		Model:       model,
//		Description: "...",
//		Instruction: "...",
//		Toolsets: []tool.Set{
//			mcptoolset.New(mcptoolset.Config{
//				Transport: &mcp.CommandTransport{Command: exec.Command("myserver")}
//			}),
//		},
//	})
func New(cfg Config) (tool.Toolset, error) {
	transport, err := buildTransport(cfg)
	if err != nil {
		return nil, err
	}
	var clientOptions *mcp.ClientOptions
	if cfg.ElicitationHandler != nil || cfg.ElicitationCompleteHandler != nil {
		if cfg.Client != nil {
			return nil, fmt.Errorf("mcptoolset: ElicitationHandler and ElicitationCompleteHandler cannot be combined with a custom Client; set them in the client's mcp.ClientOptions instead")
		}
		if cfg.ElicitationHandler == nil {
			return nil, fmt.Errorf("mcptoolset: ElicitationCompleteHandler requires ElicitationHandler to be set; the client cannot service an elicitation without it")
		}
		clientOptions = &mcp.ClientOptions{
			ElicitationHandler:         cfg.ElicitationHandler,
			ElicitationCompleteHandler: cfg.ElicitationCompleteHandler,
			// The capability inferred from ElicitationHandler alone covers only
			// form mode; URL mode must be declared explicitly. RootsV2 preserves
			// the default roots capability, which setting Capabilities would
			// otherwise disable. RootsV2 is deprecated as of the 2026-07-28
			// revision (SEP-2577) but remains the only field that carries
			// ListChanged, so it stays until the SDK offers a replacement.
			Capabilities: &mcp.ClientCapabilities{
				Elicitation: &mcp.ElicitationCapabilities{
					Form: &mcp.FormElicitationCapabilities{},
					URL:  &mcp.URLElicitationCapabilities{},
				},
				RootsV2: &mcp.RootCapabilities{ListChanged: true},
			},
		}
	}
	return &set{
		mcpClient:                   newConnectionRefresher(cfg.Client, transport, clientOptions),
		toolFilter:                  cfg.ToolFilter,
		requireConfirmation:         cfg.RequireConfirmation,
		requireConfirmationProvider: cfg.RequireConfirmationProvider,
	}, nil
}

// buildTransport resolves the MCP transport from cfg. When Transport is nil and
// Endpoint is set, it builds a streamable HTTP transport to that URL. When
// Config.Auth is set, it wraps the transport's HTTP client with a per-request
// auth RoundTripper, which requires a streamable HTTP transport.
func buildTransport(cfg Config) (mcp.Transport, error) {
	transport := cfg.Transport
	if transport == nil && cfg.Endpoint != "" {
		transport = &mcp.StreamableClientTransport{Endpoint: cfg.Endpoint}
	}
	if cfg.Auth == nil {
		return transport, nil
	}
	st, ok := transport.(*mcp.StreamableClientTransport)
	if !ok {
		return nil, fmt.Errorf("mcptoolset: Config.Auth requires a streamable HTTP transport; "+
			"set Config.Endpoint or pass a *mcp.StreamableClientTransport (got %T)", transport)
	}
	stCopy := *st
	stCopy.HTTPClient = authHTTPClient(st.HTTPClient, cfg.Auth)
	return &stCopy, nil
}

// authHTTPClient returns a shallow copy of base whose Transport applies provider
// to every request. base may be nil.
func authHTTPClient(base *http.Client, provider auth.CredentialProvider) *http.Client {
	c := &http.Client{}
	if base != nil {
		*c = *base
	}
	c.Transport = &auth.Transport{Provider: provider, Base: c.Transport}
	return c
}

// Config provides initial configuration for the MCP ToolSet.
type Config struct {
	// Client is an optional custom MCP client to use. If nil, a default client will be created.
	Client *mcp.Client
	// Transport that will be used to connect to MCP server.
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
	// Don't also set OAuthHandler on a supplied *mcp.StreamableClientTransport:
	// Auth is applied last and overwrites the Authorization header, so the two
	// would fight over the same request.
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

	// ElicitationHandler handles elicitation/create requests from the MCP
	// server, including URL-mode elicitations that servers use for
	// out-of-band interactions such as auth challenges. Setting it makes the
	// client advertise the elicitation capability for both form and URL mode.
	//
	// For URL mode the handler must not return until the out-of-band
	// interaction has completed: nothing waits for the server's
	// notifications/elicitation/complete notification on its behalf, and the
	// call is retried as soon as the handler returns. A handler that returns
	// early makes the server re-request input until the retry budget is spent.
	// A server that reports the URL through the CodeURLElicitationRequired
	// (-32042) error code rather than through an elicitation request is not
	// handled at all.
	//
	// One handler serves every tool of the toolset, and the tool calls of a
	// single turn run on separate goroutines, so the handler must be safe for
	// concurrent use.
	//
	// It can only be set when Client is nil; for a custom Client, set the
	// handler in the client's mcp.ClientOptions instead.
	ElicitationHandler func(context.Context, *mcp.ElicitRequest) (*mcp.ElicitResult, error)

	// ElicitationCompleteHandler handles notifications/elicitation/complete
	// notifications, which servers send when an out-of-band (URL-mode)
	// elicitation has been completed. It requires ElicitationHandler to also
	// be set, since a completion notification cannot arrive unless an
	// elicitation was created first. Like ElicitationHandler, it must be safe
	// for concurrent use.
	//
	// It can only be set when Client is nil; for a custom Client, set the
	// handler in the client's mcp.ClientOptions instead.
	ElicitationCompleteHandler func(context.Context, *mcp.ElicitationCompleteNotificationRequest)

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
}

type set struct {
	mcpClient                   MCPClient
	toolFilter                  tool.Predicate
	requireConfirmation         bool
	requireConfirmationProvider tool.ConfirmationProvider
}

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
		t, err := convertTool(mcpTool, s.mcpClient, s.requireConfirmation, s.requireConfirmationProvider)
		if err != nil {
			return nil, fmt.Errorf("failed to convert MCP tool %q to adk tool: %w", mcpTool.Name, err)
		}

		if s.toolFilter != nil && !s.toolFilter(ctx, t) {
			continue
		}

		// Checked after the filter so that excluding the tool by name remains a
		// way to keep the rest of the toolset usable.
		if llminternal.IsReservedToolName(mcpTool.Name) {
			return nil, fmt.Errorf("MCP server advertises tool %q, a name reserved by the framework", mcpTool.Name)
		}

		adkTools = append(adkTools, t)
	}

	return adkTools, nil
}
