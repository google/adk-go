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

// mcptool package provides MCP adapter, allowing to add MCP tools to LLMAgent.
package mcptoolset

import (
	"context"
	"fmt"
	"sync"

	"github.com/modelcontextprotocol/go-sdk/mcp"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/internal/version"
	"google.golang.org/adk/tool"
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
	client := cfg.Client
	if client == nil {
		client = mcp.NewClient(&mcp.Implementation{Name: "adk-mcp-client", Version: version.Version}, nil)
	}
	return &set{
		client:     client,
		transport:  cfg.Transport,
		toolFilter: cfg.ToolFilter,
	}, nil
}

// Config provides initial configuration for the MCP ToolSet.
type Config struct {
	// Client is an optional custom MCP client to use. If nil, a default client will be created.
	Client *mcp.Client
	// Transport that will be used to connect to MCP server.
	Transport mcp.Transport
	// ToolFilter selects tools for which tool.Predicate returns true.
	// If ToolFilter is nil, then all tools are returned.
	// tool.StringPredicate can be convenient if there's a known fixed list of tool names.
	ToolFilter tool.Predicate
}

type set struct {
	client     *mcp.Client
	transport  mcp.Transport
	toolFilter tool.Predicate

	mu      sync.Mutex
	session *mcp.ClientSession
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
	session, err := s.getSession(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to get MCP session: %w", err)
	}

	var adkTools []tool.Tool

	cursor := ""
	for {
		resp, err := session.ListTools(ctx, &mcp.ListToolsParams{
			Cursor: cursor,
		})
		if err != nil {
			return nil, fmt.Errorf("failed to list MCP tools: %w", err)
		}

		for _, mcpTool := range resp.Tools {
			t, err := convertTool(mcpTool, s.getSession)
			if err != nil {
				return nil, fmt.Errorf("failed to convert MCP tool %q to adk tool: %w", mcpTool.Name, err)
			}

			if s.toolFilter != nil && !s.toolFilter(ctx, t) {
				continue
			}

			adkTools = append(adkTools, t)
		}

		if resp.NextCursor == "" {
			break
		}
		cursor = resp.NextCursor
	}

	return adkTools, nil
}

func (s *set) getSession(ctx context.Context) (*mcp.ClientSession, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.session != nil {
		return s.session, nil
	}

	session, err := s.client.Connect(ctx, s.transport, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to init MCP session: %w", err)
	}

	s.session = session
	return s.session, nil
}
