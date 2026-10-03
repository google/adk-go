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

package configurable

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os/exec"
	"path/filepath"
	"slices"

	"github.com/modelcontextprotocol/go-sdk/mcp"
)

// mcpConfigError preserves the cause for errors.Is/As without printing config
// contents from decoder, filesystem, or subprocess errors into application logs.
type mcpConfigError struct {
	operation string
	cause     error
}

func (e *mcpConfigError) Error() string {
	// Only expose diagnostics built from schema types and positions, never the
	// decoder's message: it can include field names or values from the input.
	switch cause := e.cause.(type) {
	case *mcpNullArgumentError:
		return fmt.Sprintf("%s: %s", e.operation, cause)
	case *json.SyntaxError:
		return fmt.Sprintf("%s: invalid JSON at byte %d", e.operation, cause.Offset)
	case *json.UnmarshalTypeError:
		return fmt.Sprintf("%s: JSON value must have Go type %s", e.operation, cause.Type)
	}
	if e.cause == io.ErrUnexpectedEOF {
		return e.operation + ": incomplete JSON value"
	}
	return fmt.Sprintf("%s (%T)", e.operation, e.cause)
}
func (e *mcpConfigError) Unwrap() error { return e.cause }

type mcpNullArgumentError struct{ index int }

func (e *mcpNullArgumentError) Error() string {
	return fmt.Sprintf("args[%d]: expected string, got null", e.index)
}

type mcpPolicyKey struct{}

type mcpPolicy struct {
	AllowedServers []allowedMCPServer `json:"allowed_servers"`
}

type allowedMCPServer struct {
	Command         string       `json:"command"`
	Args            mcpArguments `json:"args"`
	resolvedCommand string
}

// mcpArguments rejects null elements, which encoding/json otherwise silently
// converts into empty strings when decoding a []string.
type mcpArguments []string

func (a *mcpArguments) UnmarshalJSON(data []byte) error {
	var values []*string
	if err := json.Unmarshal(data, &values); err != nil {
		return err
	}
	if values == nil {
		*a = nil
		return nil
	}
	args := make(mcpArguments, len(values))
	for i, value := range values {
		if value == nil {
			return &mcpNullArgumentError{index: i}
		}
		args[i] = *value
	}
	*a = args
	return nil
}

type agentCacheKey struct {
	path   string
	policy *mcpPolicy
}

// WithMCPPolicy reads an operator-provided JSON policy and returns a context
// authorizing exactly its command/argument pairs during config loading. Commands
// must be absolute executable paths; args must be an array (empty means no args).
// The policy is owned by the returned context and is not mutable by the caller.
// Never obtain this policy from an untrusted agent config.
//
// This loader is repository-internal, not a supported external API. The internal
// adkcli exposes it through --mcp-policy. Authorization happens at config load;
// it does not revoke previously constructed toolsets or sandbox approved servers.
// Unlike Python's opt-in for all stdio configs, this requires exact approvals
// to meet the command and argument restrictions in issue #1569.
func WithMCPPolicy(ctx context.Context, r io.Reader) (context.Context, error) {
	dec := json.NewDecoder(r)
	dec.DisallowUnknownFields()
	var policy mcpPolicy
	if err := dec.Decode(&policy); err != nil {
		return nil, &mcpConfigError{"decode MCP policy", err}
	}
	if err := dec.Decode(new(any)); err != io.EOF {
		if err == nil {
			return nil, fmt.Errorf("MCP policy must contain exactly one JSON object")
		}
		return nil, &mcpConfigError{"decode trailing MCP policy data", err}
	}
	if policy.AllowedServers == nil {
		return nil, fmt.Errorf("MCP policy requires an allowed_servers array")
	}
	for i := range policy.AllowedServers {
		server := &policy.AllowedServers[i]
		if !filepath.IsAbs(server.Command) {
			return nil, fmt.Errorf("MCP policy allowed_servers[%d].command must be an absolute executable path", i)
		}
		if server.Args == nil {
			return nil, fmt.Errorf("MCP policy allowed_servers[%d].args must be an array (use [] for no arguments)", i)
		}
		command, err := resolveMCPCommand(server.Command)
		if err != nil {
			return nil, fmt.Errorf("MCP policy allowed_servers[%d].command: %w", i, err)
		}
		server.resolvedCommand = command
	}
	return context.WithValue(ctx, mcpPolicyKey{}, &policy), nil
}

func mcpPolicyFromContext(ctx context.Context) *mcpPolicy {
	policy, _ := ctx.Value(mcpPolicyKey{}).(*mcpPolicy)
	return policy
}

func approvedMCPTransport(ctx context.Context, command string, args []string) (mcp.Transport, error) {
	policy := mcpPolicyFromContext(ctx)
	if policy == nil || len(policy.AllowedServers) == 0 {
		return nil, fmt.Errorf("stdio MCP servers are denied by default: supply an operator-approved policy with adkcli --mcp-policy before loading agent configs")
	}
	resolved, err := resolveMCPCommand(command)
	if err != nil {
		return nil, fmt.Errorf("resolve MCP server command: %w", err)
	}
	for _, server := range policy.AllowedServers {
		if server.resolvedCommand == resolved && slices.Equal(server.Args, args) {
			return &mcpCommandTransport{
				command:         server.Command,
				resolvedCommand: server.resolvedCommand,
				args:            slices.Clone(server.Args),
			}, nil
		}
	}
	return nil, fmt.Errorf("MCP server command and arguments are not approved by the operator policy")
}

// mcpCommandTransport preserves the operator's invocation path, which may select
// a virtual environment or a program mode via argv[0]. The config's alias cannot
// select that path. Recheck the target before every connection, since MCP starts
// the process lazily and may reconnect later.
type mcpCommandTransport struct {
	command         string
	resolvedCommand string
	args            []string
}

func (t *mcpCommandTransport) Connect(ctx context.Context) (mcp.Connection, error) {
	resolved, err := resolveMCPCommand(t.command)
	if err != nil {
		return nil, fmt.Errorf("resolve approved MCP command before startup: %w", err)
	}
	if resolved != t.resolvedCommand {
		return nil, fmt.Errorf("MCP server command changed since approval")
	}
	transport := &mcp.CommandTransport{Command: exec.Command(t.command, t.args...)}
	conn, err := transport.Connect(ctx)
	if err != nil {
		return nil, &mcpConfigError{"start approved MCP server", err}
	}
	return conn, nil
}

func resolveMCPCommand(command string) (string, error) {
	path, err := exec.LookPath(command)
	if err != nil {
		return "", &mcpConfigError{"find MCP executable", err}
	}
	// Resolve before cleaning the path: link/../server follows the link before
	// walking to its parent, and must match the path used by exec.Command.
	path, err = filepath.EvalSymlinks(path)
	if err != nil {
		return "", &mcpConfigError{"resolve executable symlinks", err}
	}
	path, err = filepath.Abs(path)
	if err != nil {
		return "", &mcpConfigError{"resolve absolute executable path", err}
	}
	return path, nil
}
