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

package main

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"google.golang.org/adk/v2/internal/configurable"
)

func TestLoadMCPPolicy(t *testing.T) {
	executable, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	data, err := json.Marshal(map[string]any{"allowed_servers": []any{
		map[string]any{"command": executable, "args": []string{"approved"}},
	}})
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "policy.json")
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}
	invalid := filepath.Join(t.TempDir(), "invalid.json")
	if err := os.WriteFile(invalid, []byte(`{"allow_all":true}`), 0o600); err != nil {
		t.Fatal(err)
	}
	tests := []struct {
		name            string
		args, remaining []string
		wantErr         string
		approved        bool
	}{
		{name: "no arguments"},
		{name: "console flags unchanged", args: []string{"-streaming_mode", "sse"}, remaining: []string{"-streaming_mode", "sse"}},
		{name: "web arguments unchanged", args: []string{"web", "api"}, remaining: []string{"web", "api"}},
		{name: "separate value", args: []string{"--mcp-policy", path, "console"}, remaining: []string{"console"}, approved: true},
		{name: "equals value", args: []string{"--mcp-policy=" + path, "web", "api"}, remaining: []string{"web", "api"}, approved: true},
		{name: "single dash", args: []string{"-mcp-policy", path}, approved: true},
		{name: "missing value", args: []string{"--mcp-policy"}, wantErr: "requires"},
		{name: "flag as value", args: []string{"--mcp-policy", "--help"}, wantErr: "requires"},
		{name: "empty value", args: []string{"--mcp-policy="}, wantErr: "requires"},
		{name: "missing file", args: []string{"--mcp-policy", filepath.Join(t.TempDir(), "missing")}, wantErr: "read MCP policy"},
		{name: "invalid policy", args: []string{"--mcp-policy", invalid}, wantErr: "load MCP policy"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx, remaining, err := loadMCPPolicy(context.Background(), tt.args)
			if tt.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
					t.Fatalf("loadMCPPolicy() error = %v, want %q", err, tt.wantErr)
				}
				return
			}
			if err != nil || !slices.Equal(remaining, tt.remaining) {
				t.Fatalf("loadMCPPolicy() remaining = %v, err = %v; want %v", remaining, err, tt.remaining)
			}
			// The parser must return a context that authorizes the factory.
			args := map[string]any{
				"stdio_connection_params": map[string]any{
					"server_params": map[string]any{"command": executable, "args": []any{"approved"}},
				},
				"tool_filter": []any{},
			}
			_, set, err := configurable.ResolveToolReference(ctx, "McpToolset", args)
			if tt.approved {
				if err != nil || set == nil {
					t.Fatalf("CLI policy did not authorize the factory: %v", err)
				}
			} else if err == nil || set != nil {
				t.Fatal("CLI without a policy authorized a subprocess")
			}
		})
	}
}

// Run main in a child process to isolate its flags, globals, and os.Exit calls.
func TestMCPPolicyCLIProcess(t *testing.T) {
	if os.Getenv("ADK_TEST_MCP_POLICY_CLI") != "1" {
		return
	}
	separator := slices.Index(os.Args, "--")
	if separator < 0 {
		t.Fatal("missing CLI argument separator")
	}
	os.Args = append(os.Args[:1], os.Args[separator+1:]...)
	main()
}

func TestMCPPolicyCLIStartup(t *testing.T) {
	executable, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	t.Setenv("GOOGLE_API_KEY", "unused-test-key")
	t.Setenv("GOOGLE_GENAI_USE_VERTEXAI", "false")
	configDir := t.TempDir()
	config, err := json.Marshal(map[string]any{
		"agent_class": "LlmAgent", "name": "policy_agent", "model": "gemini-2.5-flash",
		"tools": []any{map[string]any{
			"name": "McpToolset",
			"args": map[string]any{
				"stdio_connection_params": map[string]any{
					"server_params": map[string]any{"command": executable, "args": []string{}},
				},
				"tool_filter": []string{},
			},
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(configDir, "root_agent.yaml"), config, 0o600); err != nil {
		t.Fatal(err)
	}
	for _, tt := range []struct {
		name       string
		withPolicy bool
		args       []string
		wantError  string
	}{
		{name: "approved", withPolicy: true, args: []string{}},
		{name: "no policy", wantError: "denied by default"},
		{name: "arguments not approved", withPolicy: true, args: []string{"different"}, wantError: "not approved"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			cliArgs := []string{"-test.run=^TestMCPPolicyCLIProcess$", "--"}
			if tt.withPolicy {
				policy, err := json.Marshal(map[string]any{"allowed_servers": []any{
					map[string]any{"command": executable, "args": tt.args},
				}})
				if err != nil {
					t.Fatal(err)
				}
				policyPath := filepath.Join(t.TempDir(), "policy.json")
				if err := os.WriteFile(policyPath, policy, 0o600); err != nil {
					t.Fatal(err)
				}
				cliArgs = append(cliArgs, "--mcp-policy", policyPath)
			}
			cliArgs = append(cliArgs, "console")
			ctx, cancel := context.WithTimeout(t.Context(), 15*time.Second)
			defer cancel()
			cmd := exec.CommandContext(ctx, executable, cliArgs...)
			cmd.Dir = configDir
			cmd.Env = append(os.Environ(), "ADK_TEST_MCP_POLICY_CLI=1")
			// Nil stdin supplies EOF, so the console exits without a model or MCP call.
			output, err := cmd.CombinedOutput()
			if ctx.Err() != nil {
				t.Fatal("CLI startup timed out")
			}
			loaded := strings.Contains(string(output), "Agent loaded successfully: policy_agent")
			if tt.wantError != "" {
				if err == nil || loaded || !strings.Contains(string(output), tt.wantError) {
					t.Fatal("CLI did not reject the unapproved agent during config loading")
				}
				return
			}
			if err != nil || !loaded || !strings.Contains(string(output), "EOF detected, exiting...") {
				t.Fatal("CLI did not load the approved agent and finish the console session")
			}
		})
	}
}

func TestLoadMCPPolicyErrorOmitsPath(t *testing.T) {
	const marker = "private-policy-marker"
	path := filepath.Join(t.TempDir(), marker)
	_, _, err := loadMCPPolicy(t.Context(), []string{"--mcp-policy", path})
	if err == nil || strings.Contains(err.Error(), marker) {
		t.Fatal("file error is missing or exposes policy path")
	}
	var cause *os.PathError
	if !errors.As(err, &cause) || !errors.Is(err, os.ErrNotExist) {
		t.Fatal("file error cause was lost")
	}
}

func TestLoadInvalidMCPPolicyErrorOmitsPath(t *testing.T) {
	const marker = "private-policy-marker"
	path := filepath.Join(t.TempDir(), marker)
	if err := os.WriteFile(path, []byte(`{"allowed_servers":!}`), 0o600); err != nil {
		t.Fatal(err)
	}
	_, _, err := loadMCPPolicy(t.Context(), []string{"--mcp-policy", path})
	if err == nil || strings.Contains(err.Error(), marker) {
		t.Fatal("invalid policy error is missing or exposes policy path")
	}
	var cause *json.SyntaxError
	if !errors.As(err, &cause) {
		t.Fatal("invalid policy error cause was lost")
	}
}
