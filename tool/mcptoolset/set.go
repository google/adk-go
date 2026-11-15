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
	"io"
	"os/exec"
	"slices"
	"sync"

	"github.com/modelcontextprotocol/go-sdk/mcp"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/tool"
)

// New returns MCP ToolSet.
// MCP ToolSet connects to a MCP Server, retrieves MCP Tools into ADK Tools and
// passes them to the LLM.
// It uses https://github.com/modelcontextprotocol/go-sdk for MCP communication.
// MCP session is created lazily on the first request to LLM.
// New returns an error if Transport is not provided.
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
	if cfg.Transport == nil {
		return nil, fmt.Errorf("mcptoolset: set Config.Transport")
	}
	return &set{
		mcpClient:                   newConnectionRefresher(cfg.Client, buildTransport(cfg.Transport)),
		toolFilter:                  cfg.ToolFilter,
		requireConfirmation:         cfg.RequireConfirmation,
		requireConfirmationProvider: cfg.RequireConfirmationProvider,
	}, nil
}

// buildTransport wraps a *mcp.CommandTransport with a non-nil Command so it
// can connect more than once (see Config.Transport). Other transports are
// returned as given.
func buildTransport(transport mcp.Transport) mcp.Transport {
	if t, ok := transport.(*mcp.CommandTransport); ok && t != nil && t.Command != nil {
		return newCommandTransport(t)
	}
	return transport
}

// commandTransport lets a CommandTransport connect more than once. An
// *exec.Cmd can be started only once, so the first Connect uses the caller's
// command as is, and every later Connect runs a copy. Without this, a reconnect
// fails with "exec: Stdout already set".
type commandTransport struct {
	first mcp.CommandTransport // the caller's transport, used for the first Connect
	// spec copies the caller's command settings before the first Connect, as
	// the SDK sets Stdin and Stdout on the command it starts.
	spec exec.Cmd

	mu   sync.Mutex
	used bool // the caller's command has been handed to a Connect
}

func newCommandTransport(t *mcp.CommandTransport) *commandTransport {
	src := t.Command
	// Only the fields that configure a process are copied. The copy has no
	// exec.CommandContext context, because os/exec does not expose it. Cancel
	// is left out too: Start rejects a Cancel without a context, and the
	// default Cancel of CommandContext would kill the caller's process, not
	// the copy.
	return &commandTransport{first: *t, spec: exec.Cmd{
		Path:        src.Path,
		Args:        slices.Clone(src.Args),
		Env:         slices.Clone(src.Env),
		Dir:         src.Dir,
		Stdin:       src.Stdin,
		Stderr:      src.Stderr,
		ExtraFiles:  slices.Clone(src.ExtraFiles),
		SysProcAttr: src.SysProcAttr,
		WaitDelay:   src.WaitDelay,
		Err:         src.Err,
	}}
}

// Connect implements mcp.Transport.
func (t *commandTransport) Connect(ctx context.Context) (mcp.Connection, error) {
	t.mu.Lock()
	first := !t.used
	t.used = true
	t.mu.Unlock()

	if first {
		return t.first.Connect(ctx)
	}
	return t.reconnectTransport().Connect(ctx)
}

// reconnectTransport returns the transport for every Connect after the first.
// It copies the whole transport, so every setting on it (such as
// TerminateDuration) carries over, including ones added in later SDK
// versions. Only Command is replaced, with a fresh copy of spec.
func (t *commandTransport) reconnectTransport() *mcp.CommandTransport {
	cmd := t.spec
	tr := t.first
	tr.Command = &cmd
	return &tr
}

// Config provides initial configuration for the MCP ToolSet.
type Config struct {
	// Client is an optional custom MCP client to use. If nil, a default client will be created.
	Client *mcp.Client
	// Transport that will be used to connect to MCP server.
	//
	// A *mcp.CommandTransport with a non-nil Command is wrapped when New is
	// called, so the server can be started again after the connection is
	// lost (an *exec.Cmd can be started only once). The first connect runs
	// the caller's command, and every later connect runs a copy of the
	// command and transport taken at New. Callers can notice this in three
	// ways:
	//   - The transport is copied at New, so changing it afterwards (for
	//     example TerminateDuration) has no effect, even on the first
	//     connect. Changes to the command after New only reach the first
	//     connect.
	//   - The copy can't carry an exec.CommandContext context, so a server
	//     built with CommandContext is started again after that context is
	//     canceled, and the new process ignores it. In the same way, a
	//     server stopped by killing cmd.Process comes back on the next tool
	//     call.
	//   - A CommandTransport inside a wrapper such as *mcp.LoggingTransport
	//     is used as given, so it still can't reconnect.
	//
	// Other transport types are used as given.
	Transport mcp.Transport
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
}

type set struct {
	mcpClient                   *connectionRefresher
	toolFilter                  tool.Predicate
	requireConfirmation         bool
	requireConfirmationProvider tool.ConfirmationProvider
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
		t, err := convertTool(mcpTool, s.mcpClient, s.requireConfirmation, s.requireConfirmationProvider)
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
