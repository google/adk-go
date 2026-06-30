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
	"fmt"
	"os/exec"
	"path/filepath"
	"sync"
)

// MCPServerPolicy defines which MCP server commands an operator permits to be
// spawned from YAML agent configurations.
//
// The security model treats YAML agent configs as an untrusted input boundary.
// Rather than attempting to sanitize commands (which is fundamentally broken
// when launchers like node, python, npx are themselves code-execution
// primitives), this policy requires the operator to explicitly opt-in to
// specific (command, args-prefix) pairs.
//
// Default behavior: If no policy is set, all MCP subprocess spawning from
// YAML config is blocked (default-deny).
type MCPServerPolicy struct {
	// AllowedServers is a list of permitted MCP server definitions.
	// Each entry specifies an exact command and an optional args prefix.
	AllowedServers []AllowedMCPServer
}

// AllowedMCPServer defines a single permitted MCP server command.
type AllowedMCPServer struct {
	// Command is the exact, resolved absolute path of the binary that is
	// permitted. When matching, the requested command is resolved to its
	// absolute real path (via exec.LookPath + filepath.EvalSymlinks) and
	// compared against this value.
	//
	// Examples:
	//   "/usr/local/bin/npx"
	//   "C:\\Program Files\\nodejs\\npx.cmd"
	Command string

	// ArgsPrefix is an optional prefix that the command arguments must start
	// with. If empty/nil, any arguments are allowed for this command. If set,
	// the actual args must begin with these exact values (in order).
	//
	// This allows constraining a general-purpose launcher to specific
	// packages. For example, allowing npx only with a specific MCP server:
	//   ArgsPrefix: []string{"@modelcontextprotocol/server-filesystem"}
	ArgsPrefix []string
}

var (
	globalPolicyMu sync.RWMutex
	globalPolicy   *MCPServerPolicy
)

// SetGlobalMCPPolicy sets the global MCP server policy. This should be called
// by the operator (the trusted caller that loads agent configs) at startup,
// before any YAML agent configs are loaded.
//
// Pass nil to revert to default-deny (block all MCP subprocess spawning).
func SetGlobalMCPPolicy(policy *MCPServerPolicy) {
	globalPolicyMu.Lock()
	defer globalPolicyMu.Unlock()
	globalPolicy = policy
}

// GetGlobalMCPPolicy returns the current global MCP server policy.
// Returns nil if no policy has been set (default-deny).
func GetGlobalMCPPolicy() *MCPServerPolicy {
	globalPolicyMu.RLock()
	defer globalPolicyMu.RUnlock()
	return globalPolicy
}

// ResolveBinaryPath resolves a command name or path to its absolute, real
// filesystem path. This prevents basename spoofing attacks where a path like
// "../../evil/npx" would match an allowlisted "npx" if only the basename
// were checked.
//
// Resolution steps:
//  1. If the command is a bare name (no path separators), use exec.LookPath
//     to find it in PATH.
//  2. Convert to absolute path via filepath.Abs.
//  3. Resolve symlinks via filepath.EvalSymlinks.
func ResolveBinaryPath(command string) (string, error) {
	if command == "" {
		return "", fmt.Errorf("command cannot be empty")
	}

	var resolvedPath string

	// If the command contains no path separator, look it up in PATH.
	// Otherwise, treat it as a direct path reference.
	if filepath.Base(command) == command {
		// Bare command name — resolve via PATH
		path, err := exec.LookPath(command)
		if err != nil {
			return "", fmt.Errorf("command %q not found in PATH: %w", command, err)
		}
		resolvedPath = path
	} else {
		resolvedPath = command
	}

	// Convert to absolute path
	absPath, err := filepath.Abs(resolvedPath)
	if err != nil {
		return "", fmt.Errorf("failed to resolve absolute path for %q: %w", command, err)
	}

	// Resolve symlinks to get the real path
	realPath, err := filepath.EvalSymlinks(absPath)
	if err != nil {
		return "", fmt.Errorf("failed to resolve symlinks for %q: %w", absPath, err)
	}

	return realPath, nil
}

// ValidateMCPCommand checks whether a resolved command path and its arguments
// are permitted by the given policy.
//
// The command parameter must be a fully resolved absolute path (as returned
// by ResolveBinaryPath). The args parameter is the list of arguments that
// will be passed to the command.
//
// Returns nil if the command is permitted, or an error describing why it was
// blocked.
func ValidateMCPCommand(policy *MCPServerPolicy, resolvedCommand string, args []string) error {
	if policy == nil {
		return fmt.Errorf(
			"no MCP server policy configured; all subprocess spawning is blocked by default. " +
				"Call SetGlobalMCPPolicy() to permit specific commands")
	}

	if len(policy.AllowedServers) == 0 {
		return fmt.Errorf(
			"MCP server policy has no allowed servers; all subprocess spawning is blocked")
	}

	for _, allowed := range policy.AllowedServers {
		if !commandMatches(allowed.Command, resolvedCommand) {
			continue
		}

		// Command matches — now check args prefix
		if len(allowed.ArgsPrefix) == 0 {
			// No args constraint — command is allowed with any args
			return nil
		}

		if argsPrefixMatches(allowed.ArgsPrefix, args) {
			return nil
		}
		// Command matched but args didn't — continue checking other entries
		// (there might be another entry for the same command with different args)
	}

	return fmt.Errorf(
		"command %q with args %v is not permitted by the MCP server policy",
		resolvedCommand, args)
}

// commandMatches checks if a resolved command path matches an allowed command
// specification. The comparison is case-insensitive on Windows-style paths
// and exact on Unix.
func commandMatches(allowed, resolved string) bool {
	// Normalize both paths for comparison
	allowedClean := filepath.Clean(allowed)
	resolvedClean := filepath.Clean(resolved)

	return allowedClean == resolvedClean
}

// argsPrefixMatches checks if the actual args start with the required prefix.
func argsPrefixMatches(prefix, actual []string) bool {
	if len(actual) < len(prefix) {
		return false
	}
	for i, p := range prefix {
		if actual[i] != p {
			return false
		}
	}
	return true
}
