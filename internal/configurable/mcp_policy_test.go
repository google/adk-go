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
	"errors"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"testing/iotest"
	"time"

	"google.golang.org/adk/v2/agent"
)

// These fixtures are only resolved, never executed. No installed MCP launcher
// or network access is needed to check executable identity.
func policyExecutable(t *testing.T) string {
	t.Helper()
	name := "server"
	if runtime.GOOS == "windows" {
		name += ".exe"
	}
	path := filepath.Join(t.TempDir(), name)
	if err := os.WriteFile(path, []byte("unused executable fixture"), 0o700); err != nil {
		t.Fatal(err)
	}
	return path
}

func policyContext(t *testing.T, command string, args []string) context.Context {
	t.Helper()
	data, err := json.Marshal(mcpPolicy{AllowedServers: []allowedMCPServer{{Command: command, Args: args}}})
	if err != nil {
		t.Fatal(err)
	}
	ctx, err := WithMCPPolicy(context.Background(), strings.NewReader(string(data)))
	if err != nil {
		t.Fatal(err)
	}
	return ctx
}

func mcpConfigArgs(command string, args []string) map[string]any {
	rawArgs := make([]any, len(args))
	for i, arg := range args {
		rawArgs[i] = arg
	}
	return map[string]any{
		"stdio_connection_params": map[string]any{
			"server_params": map[string]any{"command": command, "args": rawArgs},
		},
		"tool_filter": []any{"anything"},
	}
}

func TestResolveToolReferenceMCPPolicy(t *testing.T) {
	executable := policyExecutable(t)
	other := policyExecutable(t)
	approvedArgs := []string{"run", "--rm", "approved-image"}
	ctx := policyContext(t, executable, approvedArgs)
	noArgsCtx := policyContext(t, executable, []string{})
	emptyStringCtx := policyContext(t, executable, []string{""})
	emptyCtx, err := WithMCPPolicy(context.Background(), strings.NewReader(`{"allowed_servers":[]}`))
	if err != nil {
		t.Fatal(err)
	}
	tests := []struct {
		name    string
		ctx     context.Context
		command string
		args    []string
		wantErr string
	}{
		{"no policy", context.Background(), executable, approvedArgs, "denied by default"},
		{"empty policy", emptyCtx, executable, approvedArgs, "denied by default"},
		{"exact match", ctx, executable, approvedArgs, ""},
		{"different executable with same basename", ctx, other, approvedArgs, "not approved"},
		{"extra arguments after approved prefix", ctx, executable, []string{"run", "--rm", "approved-image", "-v", "/:/host"}, "not approved"},
		{"missing argument", ctx, executable, []string{"run", "--rm"}, "not approved"},
		{"reordered arguments", ctx, executable, []string{"--rm", "run", "approved-image"}, "not approved"},
		{"replaced argument", ctx, executable, []string{"run", "--rm", "other-image"}, "not approved"},
		{"no arguments", noArgsCtx, executable, nil, ""},
		{"explicit empty string", emptyStringCtx, executable, []string{""}, ""},
		{"empty string is not zero arguments", emptyStringCtx, executable, nil, "not approved"},
		{"empty args is not a wildcard", noArgsCtx, executable, []string{"-e", "unapproved code"}, "not approved"},
		{"missing executable", ctx, filepath.Join(t.TempDir(), "missing"), nil, "resolve MCP server command"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, set, err := ResolveToolReference(tt.ctx, "McpToolset", mcpConfigArgs(tt.command, tt.args))
			if tt.wantErr == "" {
				if err != nil || set == nil {
					t.Fatalf("ResolveToolReference() = %v, %v; want a toolset", set, err)
				}
			} else if err == nil || !strings.Contains(err.Error(), tt.wantErr) || set != nil {
				t.Fatalf("ResolveToolReference() = %v, %v; want nil toolset and error containing %q", set, err, tt.wantErr)
			}
		})
	}
}

func TestWithMCPPolicyInvalid(t *testing.T) {
	executable := policyExecutable(t)
	quoted, err := json.Marshal(executable)
	if err != nil {
		t.Fatal(err)
	}
	tests := []struct{ name, policy string }{
		{"malformed", "{"},
		{"null", "null"},
		{"missing servers", "{}"},
		{"unknown field", `{"allowed_servers":[],"allow_all":true}`},
		{"trailing object", `{"allowed_servers":[]} {}`},
		{"trailing garbage", `{"allowed_servers":[]} x`},
		{"relative command", `{"allowed_servers":[{"command":"server","args":[]}]}`},
		{"missing args", `{"allowed_servers":[{"command":` + string(quoted) + `}]}`},
		{"null args", `{"allowed_servers":[{"command":` + string(quoted) + `,"args":null}]}`},
		{"null argument", `{"allowed_servers":[{"command":` + string(quoted) + `,"args":[null]}]}`},
		{"null after string argument", `{"allowed_servers":[{"command":` + string(quoted) + `,"args":["approved",null]}]}`},
		{"non-string args", `{"allowed_servers":[{"command":` + string(quoted) + `,"args":[1]}]}`},
		{"prefix option", `{"allowed_servers":[{"command":` + string(quoted) + `,"args_prefix":[]}]}`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx, err := WithMCPPolicy(context.Background(), strings.NewReader(tt.policy))
			if err == nil || ctx != nil {
				t.Fatalf("WithMCPPolicy() = %v, %v; want nil context and error", ctx, err)
			}
		})
	}
}

func TestWithMCPPolicyErrorDiagnostics(t *testing.T) {
	tests := []struct{ name, policy, want string }{
		{"null argument", `{"allowed_servers":[{"args":["private-value",null]}]}`, "decode MCP policy: args[1]: expected string, got null"},
		{"syntax", `{"allowed_servers":!}`, "decode MCP policy: invalid JSON at byte 20"},
		{"wrong type", `{"allowed_servers":[{"command":123456789}]}`, "decode MCP policy: JSON value must have Go type string"},
		{"incomplete", `{"allowed_servers":`, "decode MCP policy: incomplete JSON value"},
		{"trailing syntax", `{"allowed_servers":[]} !`, "decode trailing MCP policy data: invalid JSON at byte 24"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := WithMCPPolicy(t.Context(), strings.NewReader(tt.policy))
			if err == nil || err.Error() != tt.want {
				t.Fatal("policy error does not provide the expected safe diagnostic")
			}
		})
	}
}

func TestResolveToolReferenceMCPPolicyPaths(t *testing.T) {
	executable := policyExecutable(t)
	link := filepath.Join(t.TempDir(), filepath.Base(executable))
	if err := os.Symlink(executable, link); err != nil {
		t.Skipf("symlinks unavailable: %v", err)
	}
	// Both the policy and the config may name a symlink to the same executable.
	ctx := policyContext(t, link, []string{})
	for _, command := range []string{link, executable} {
		if _, _, err := ResolveToolReference(ctx, "McpToolset", mcpConfigArgs(command, nil)); err != nil {
			t.Fatalf("command %q: %v", command, err)
		}
	}
	// PATH lookup must still compare the resolved executable, not its basename.
	t.Setenv("PATH", filepath.Dir(executable))
	if _, _, err := ResolveToolReference(ctx, "McpToolset", mcpConfigArgs(filepath.Base(executable), nil)); err != nil {
		t.Fatal(err)
	}
	other := policyExecutable(t)
	t.Setenv("PATH", filepath.Dir(other))
	if _, _, err := ResolveToolReference(ctx, "McpToolset", mcpConfigArgs(filepath.Base(other), nil)); err == nil {
		t.Fatal("an unapproved executable on PATH was accepted")
	}
	if err := os.Remove(link); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(other, link); err != nil {
		t.Fatal(err)
	}
	if _, _, err := ResolveToolReference(ctx, "McpToolset", mcpConfigArgs(link, nil)); err == nil {
		t.Fatal("retargeting an approved symlink granted approval to another executable")
	}
}

func TestResolveAgentReferenceMCPPolicyIsolation(t *testing.T) {
	// Model construction is local; no agent or model request is run.
	t.Setenv("GOOGLE_API_KEY", "unused-test-key")
	executable := policyExecutable(t)
	dir := t.TempDir()
	parent := filepath.Join(dir, "root_agent.yaml")
	child := filepath.Join(dir, "child.yaml")
	// JSON is also YAML and quotes platform-specific executable paths correctly.
	data, err := json.Marshal(map[string]any{
		"agent_class": "LlmAgent", "name": "child", "model": "gemini-2.5-flash",
		"tools": []any{map[string]any{"name": "McpToolset", "args": mcpConfigArgs(executable, nil)}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(child, data, 0o600); err != nil {
		t.Fatal(err)
	}
	ctx := policyContext(t, executable, []string{})
	first, err := ResolveAgentReference(ctx, parent, "child.yaml")
	if err != nil {
		t.Fatal(err)
	}
	again, err := ResolveAgentReference(ctx, parent, "child.yaml")
	if err != nil || first != again {
		t.Fatalf("same policy did not reuse cached agent: %v", err)
	}
	// Sharing a policy must not collapse different config paths into one cache entry.
	second := filepath.Join(dir, "second.yaml")
	secondData := strings.Replace(string(data), `"name":"child"`, `"name":"second"`, 1)
	if err := os.WriteFile(second, []byte(secondData), 0o600); err != nil {
		t.Fatal(err)
	}
	another, err := ResolveAgentReference(ctx, parent, "second.yaml")
	if err != nil {
		t.Fatal(err)
	}
	if another == first || another.Name() != "second" {
		t.Fatal("different configs shared a cached agent")
	}
	for _, denied := range []context.Context{context.Background(), policyContext(t, executable, []string{"different"})} {
		if _, err := ResolveAgentReference(denied, parent, "child.yaml"); err == nil {
			t.Fatal("cached agent bypassed the new context's MCP policy")
		}
	}
}

// TestMCPPolicySubprocess is run in a child copy of this test binary. It only
// writes a marker and exits, without making any MCP or model network calls.
func TestMCPPolicySubprocess(t *testing.T) {
	if len(os.Args) < 3 || os.Args[len(os.Args)-2] != "adk-mcp-policy-marker" {
		return
	}
	if err := os.WriteFile(os.Args[len(os.Args)-1], []byte(os.Args[0]), 0o600); err != nil {
		os.Exit(1)
	}
	os.Exit(0)
}

func TestMCPPolicyStartsOnlyApprovedCommandLazily(t *testing.T) {
	executable, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	tests := []struct {
		name                               string
		configAlias, policyAlias, retarget bool
	}{
		{name: "direct executable"},
		{name: "approved invocation path", configAlias: true, policyAlias: true},
		{name: "config alias cannot select invocation path", configAlias: true},
		{name: "retargeted policy alias is rejected", configAlias: true, policyAlias: true, retarget: true},
		{name: "retargeted config alias does not affect approved path", configAlias: true, retarget: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			command, approved := executable, executable
			if tt.configAlias {
				command = filepath.Join(t.TempDir(), "approved-launcher")
				if err := os.Symlink(executable, command); err != nil {
					t.Skipf("symlinks unavailable: %v", err)
				}
			}
			if tt.policyAlias {
				approved = command
			}
			marker := filepath.Join(t.TempDir(), "started")
			args := []string{"-test.run=^TestMCPPolicySubprocess$", "--", "adk-mcp-policy-marker", marker}
			ctx := policyContext(t, approved, args)
			_, set, err := ResolveToolReference(ctx, "McpToolset", mcpConfigArgs(command, args))
			if err != nil {
				t.Fatal(err)
			}
			if _, err := os.Stat(marker); !os.IsNotExist(err) {
				t.Fatalf("command started while resolving toolset: %v", err)
			}
			if tt.retarget {
				if err := os.Remove(command); err != nil {
					t.Fatal(err)
				}
				if err := os.Symlink(policyExecutable(t), command); err != nil {
					t.Fatal(err)
				}
			}
			runCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			// The helper is not an MCP server, so the handshake fails after
			// startup. Its marker records the actual invocation path.
			mockCtx := agent.NewStrictContextMock(runCtx)
			_, toolsErr := set.Tools(&mockCtx)
			if tt.policyAlias && tt.retarget {
				if toolsErr != nil && strings.Contains(toolsErr.Error(), "approved-launcher") {
					t.Fatal("changed-target error exposes the policy path")
				}
				if toolsErr == nil || !strings.Contains(toolsErr.Error(), "changed since approval") {
					t.Fatalf("Tools() error = %v; want rejection of changed policy path", toolsErr)
				}
				if _, err := os.Stat(marker); !os.IsNotExist(err) {
					t.Fatalf("retargeted policy path started a process: %v", err)
				}
				return
			}
			actual, err := os.ReadFile(marker)
			if err != nil {
				t.Fatalf("approved command did not start on Tools(): %v (Tools error: %v)", err, toolsErr)
			}
			if string(actual) != approved {
				t.Fatalf("invocation path = %q, want operator-approved path %q", actual, approved)
			}
		})
	}
}

// Cleaning link/../server before resolving the link checks a different path
// from the one the OS executes. A retargeted parent link must still be rejected.
func TestMCPPolicyRejectsRetargetedParentLink(t *testing.T) {
	first, second := policyExecutable(t), policyExecutable(t)
	for _, command := range []string{first, second} {
		if err := os.Mkdir(filepath.Join(filepath.Dir(command), "nested"), 0o700); err != nil {
			t.Fatal(err)
		}
	}
	decoy := policyExecutable(t)
	link := filepath.Join(filepath.Dir(decoy), "link")
	if err := os.Symlink(filepath.Join(filepath.Dir(first), "nested"), link); err != nil {
		t.Skipf("symlinks unavailable: %v", err)
	}
	// Do not use filepath.Join, which would clean away link/.. .
	command := link + string(os.PathSeparator) + ".." + string(os.PathSeparator) + filepath.Base(first)
	ctx := policyContext(t, command, []string{})
	_, set, err := ResolveToolReference(ctx, "McpToolset", mcpConfigArgs(command, nil))
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Remove(link); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(filepath.Join(filepath.Dir(second), "nested"), link); err != nil {
		t.Fatal(err)
	}
	mockCtx := agent.NewStrictContextMock(t.Context())
	if _, err := set.Tools(&mockCtx); err == nil || !strings.Contains(err.Error(), "changed since approval") {
		t.Fatalf("Tools() error = %v; want rejection of changed parent symlink", err)
	}
}

func TestResolveToolReferenceMCPPolicyLaterEntry(t *testing.T) {
	command, other := policyExecutable(t), policyExecutable(t)
	data, err := json.Marshal(mcpPolicy{AllowedServers: []allowedMCPServer{
		{Command: other, Args: []string{"approved"}},
		{Command: command, Args: []string{"different"}},
		{Command: command, Args: []string{"approved"}},
	}})
	if err != nil {
		t.Fatal(err)
	}
	ctx, err := WithMCPPolicy(t.Context(), strings.NewReader(string(data)))
	if err != nil {
		t.Fatal(err)
	}
	if _, set, err := ResolveToolReference(ctx, "McpToolset", mcpConfigArgs(command, []string{"approved"})); err != nil || set == nil {
		t.Fatalf("later policy entry was not accepted: %v", err)
	}
}

func TestMCPPolicyErrorsOmitConfigContent(t *testing.T) {
	const marker = "private-config-marker"
	command := policyExecutable(t)
	ctx := policyContext(t, command, []string{})
	t.Run("denied command", func(t *testing.T) {
		// The filename is config content even when the executable exists.
		path := filepath.Join(t.TempDir(), marker)
		if runtime.GOOS == "windows" {
			path += ".exe"
		}
		if err := os.WriteFile(path, []byte("unused fixture"), 0o700); err != nil {
			t.Fatal(err)
		}
		_, _, err := ResolveToolReference(ctx, "McpToolset", mcpConfigArgs(path, nil))
		if err == nil || strings.Contains(err.Error(), marker) {
			t.Fatal("denial error is missing or exposes config content")
		}
	})
	t.Run("missing command", func(t *testing.T) {
		_, _, err := ResolveToolReference(ctx, "McpToolset", mcpConfigArgs(marker, nil))
		if err == nil || strings.Contains(err.Error(), marker) {
			t.Fatal("lookup error is missing or exposes config content")
		}
		var cause *exec.Error
		if !errors.As(err, &cause) || !errors.Is(err, exec.ErrNotFound) {
			t.Fatal("lookup error cause was lost")
		}
	})
	t.Run("missing policy executable", func(t *testing.T) {
		data, err := json.Marshal(mcpPolicy{AllowedServers: []allowedMCPServer{{Command: filepath.Join(t.TempDir(), marker), Args: []string{}}}})
		if err != nil {
			t.Fatal(err)
		}
		_, err = WithMCPPolicy(t.Context(), strings.NewReader(string(data)))
		if err == nil || strings.Contains(err.Error(), marker) {
			t.Fatal("policy path error is missing or exposes config content")
		}
	})
	t.Run("decoder", func(t *testing.T) {
		_, err := WithMCPPolicy(t.Context(), strings.NewReader(`{"allowed_servers":[],"`+marker+`":true}`))
		if err == nil || strings.Contains(err.Error(), marker) {
			t.Fatal("decoder error is missing or exposes config content")
		}
		var cause *json.SyntaxError
		_, err = WithMCPPolicy(t.Context(), strings.NewReader(`{"allowed_servers":!}`))
		if !errors.As(err, &cause) {
			t.Fatal("decoder error cause was lost")
		}
	})
}

func TestWithMCPPolicyRequiresAbsoluteCommand(t *testing.T) {
	executable := policyExecutable(t)
	t.Setenv("PATH", filepath.Dir(executable))
	data, err := json.Marshal(mcpPolicy{AllowedServers: []allowedMCPServer{{Command: filepath.Base(executable), Args: []string{}}}})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := WithMCPPolicy(t.Context(), strings.NewReader(string(data))); err == nil {
		t.Fatal("relative policy command was accepted because it existed on PATH")
	}
}

func TestMCPPolicyStartupErrorsOmitPaths(t *testing.T) {
	const marker = "private-startup-marker"
	for _, remove := range []bool{false, true} {
		name := "invalid executable"
		if remove {
			name = "removed executable"
		}
		t.Run(name, func(t *testing.T) {
			command := filepath.Join(t.TempDir(), marker)
			if runtime.GOOS == "windows" {
				command += ".exe"
			}
			if err := os.WriteFile(command, []byte("not an executable format"), 0o700); err != nil {
				t.Fatal(err)
			}
			ctx := policyContext(t, command, []string{})
			_, set, err := ResolveToolReference(ctx, "McpToolset", mcpConfigArgs(command, nil))
			if err != nil {
				t.Fatal(err)
			}
			if remove {
				if err := os.Remove(command); err != nil {
					t.Fatal(err)
				}
			}
			mock := agent.NewStrictContextMock(t.Context())
			_, err = set.Tools(&mock)
			if err == nil || strings.Contains(err.Error(), marker) {
				t.Fatal("startup error is missing or exposes command path")
			}
			if remove && !strings.Contains(err.Error(), "resolve approved MCP command before startup") {
				t.Fatal("missing executable was not rejected during startup validation")
			}
		})
	}
}

func TestMCPPolicyTrailingReadErrorOmitsContent(t *testing.T) {
	const marker = "private-reader-marker"
	cause := errors.New(marker)
	reader := io.MultiReader(strings.NewReader(`{"allowed_servers":[]}`), iotest.ErrReader(cause))
	_, err := WithMCPPolicy(t.Context(), reader)
	if err == nil || strings.Contains(err.Error(), marker) {
		t.Fatal("trailing read error is missing or exposes content")
	}
	if !errors.Is(err, cause) {
		t.Fatal("trailing read error cause was lost")
	}
}
