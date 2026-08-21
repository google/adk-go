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
	"os"
	"path/filepath"
	"strings"
	"testing"
)

const traversalError = "path traversal detected"

// newAgentDir lays out an agent directory with a sibling config, a file outside
// the directory, and a symlink inside the directory pointing at that outside
// file. It returns the base directory and the parent agent config path.
func newAgentDir(t *testing.T) (base, parentPath string) {
	t.Helper()

	base = t.TempDir()
	// Resolve the temporary directory itself, since on some platforms it is
	// reached through a symlink (for example /var -> /private/var on macOS).
	if resolved, err := filepath.EvalSymlinks(base); err == nil {
		base = resolved
	}

	agentDir := filepath.Join(base, "agents", "root")
	if err := os.MkdirAll(agentDir, 0o755); err != nil {
		t.Fatalf("MkdirAll(%q) failed: %v", agentDir, err)
	}

	parentPath = filepath.Join(agentDir, "root_agent.yaml")
	if err := os.WriteFile(parentPath, []byte("agent_class: LlmAgent\n"), 0o644); err != nil {
		t.Fatalf("WriteFile(%q) failed: %v", parentPath, err)
	}
	if err := os.WriteFile(filepath.Join(agentDir, "sub_agent.yaml"), []byte("agent_class: LlmAgent\n"), 0o644); err != nil {
		t.Fatalf("WriteFile(sub_agent.yaml) failed: %v", err)
	}

	outside := filepath.Join(base, "outside.yaml")
	if err := os.WriteFile(outside, []byte("agent_class: LlmAgent\n"), 0o644); err != nil {
		t.Fatalf("WriteFile(%q) failed: %v", outside, err)
	}
	if err := os.Symlink(outside, filepath.Join(agentDir, "link.yaml")); err != nil {
		t.Skipf("symlinks are not supported in this environment: %v", err)
	}

	return base, parentPath
}

func TestResolveAgentReferenceRejectsEscapingConfigPath(t *testing.T) {
	_, parentPath := newAgentDir(t)

	tests := []struct {
		name    string
		refPath string
		wantErr string
	}{
		{
			name:    "absolute path",
			refPath: filepath.Join(string(os.PathSeparator), "etc", "passwd"),
			wantErr: "absolute paths are not allowed",
		},
		{
			name:    "parent traversal",
			refPath: filepath.Join("..", "..", "outside.yaml"),
			wantErr: traversalError,
		},
		{
			name:    "symlink escaping the agent directory",
			refPath: "link.yaml",
			wantErr: traversalError,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			// The reference must be rejected by the containment check itself, not
			// merely fail later while the config is being loaded.
			_, err := ResolveAgentReference(context.Background(), parentPath, tc.refPath)
			if err == nil {
				t.Fatalf("ResolveAgentReference(_, %q, %q) succeeded, want error containing %q", parentPath, tc.refPath, tc.wantErr)
			}
			if !strings.Contains(err.Error(), tc.wantErr) {
				t.Errorf("ResolveAgentReference(_, %q, %q) = %v, want error containing %q", parentPath, tc.refPath, err, tc.wantErr)
			}
		})
	}
}

// TestResolveAgentReferenceAllowsPathsInsideAgentDir guards against the
// containment check rejecting legitimate references. The reference is not a
// loadable agent config, so an error is expected; it must not be the traversal
// error.
func TestResolveAgentReferenceAllowsPathsInsideAgentDir(t *testing.T) {
	_, parentPath := newAgentDir(t)

	_, err := ResolveAgentReference(context.Background(), parentPath, "sub_agent.yaml")
	if err != nil && strings.Contains(err.Error(), traversalError) {
		t.Errorf("ResolveAgentReference(_, %q, %q) = %v, want no traversal rejection", parentPath, "sub_agent.yaml", err)
	}
}

// TestResolveToolReferenceMcpToolsetNonStringArgs reproduces the bug where a
// non-string element in the McpToolset "args" or "tool_filter" lists triggered
// an unchecked type assertion that panicked and killed the process. The factory
// must now return a descriptive error instead of crashing.
func TestResolveToolReferenceMcpToolsetNonStringArgs(t *testing.T) {
	newArgs := func(serverArgs, toolFilter []any) map[string]any {
		return map[string]any{
			"stdio_connection_params": map[string]any{
				"server_params": map[string]any{
					"command": "echo",
					"args":    serverArgs,
				},
			},
			"tool_filter": toolFilter,
		}
	}

	tests := []struct {
		name    string
		args    map[string]any
		wantErr string // empty means a valid config is expected
	}{
		{
			name:    "non-string in server args",
			args:    newArgs([]any{"a", 1, 2}, []any{"a"}),
			wantErr: "server_params.args[1]",
		},
		{
			name:    "non-string in tool filter",
			args:    newArgs([]any{"a"}, []any{true}),
			wantErr: "tool_filter[0]",
		},
		{
			name:    "all strings is valid",
			args:    newArgs([]any{"a", "b"}, []any{"t1"}),
			wantErr: "",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			// A non-string element previously triggered a panic; the call must
			// return normally (with or without an error), never crash.
			_, toolset, err := ResolveToolReference(context.Background(), "McpToolset", tc.args)
			if tc.wantErr == "" {
				if err != nil {
					t.Fatalf("ResolveToolReference(McpToolset) = %v, want no error", err)
				}
				if toolset == nil {
					t.Fatal("ResolveToolReference(McpToolset) returned a nil toolset, want non-nil")
				}
				return
			}
			if err == nil {
				t.Fatalf("ResolveToolReference(McpToolset) succeeded, want error containing %q", tc.wantErr)
			}
			if !strings.Contains(err.Error(), tc.wantErr) {
				t.Errorf("ResolveToolReference(McpToolset) = %v, want error containing %q", err, tc.wantErr)
			}
		})
	}
}

// TestResolveAgentReferenceRelativeParentPath covers a parent path that is not
// absolute, which the containment check must normalise before comparing.
func TestResolveAgentReferenceRelativeParentPath(t *testing.T) {
	_, parentPath := newAgentDir(t)
	agentDir := filepath.Dir(parentPath)

	cwd, err := os.Getwd()
	if err != nil {
		t.Fatalf("Getwd failed: %v", err)
	}
	if err := os.Chdir(agentDir); err != nil {
		t.Fatalf("Chdir(%q) failed: %v", agentDir, err)
	}
	t.Cleanup(func() {
		if err := os.Chdir(cwd); err != nil {
			t.Fatalf("restoring working directory failed: %v", err)
		}
	})

	if _, err := ResolveAgentReference(context.Background(), "root_agent.yaml", "sub_agent.yaml"); err != nil &&
		strings.Contains(err.Error(), traversalError) {
		t.Errorf("ResolveAgentReference with a relative parent path = %v, want no traversal rejection", err)
	}

	if _, err := ResolveAgentReference(context.Background(), "root_agent.yaml", filepath.Join("..", "..", "outside.yaml")); err == nil {
		t.Error("ResolveAgentReference with a relative parent path and an escaping reference succeeded, want an error")
	}
}

// TestResolveAgentReferenceFollowsSymlinkInsideDir guards against the
// containment check being too strict: a symlink that stays inside the agent
// directory is a legitimate way to organise configs, and adk-python allows it.
func TestResolveAgentReferenceFollowsSymlinkInsideDir(t *testing.T) {
	_, parentPath := newAgentDir(t)
	agentDir := filepath.Dir(parentPath)

	if err := os.Symlink("sub_agent.yaml", filepath.Join(agentDir, "alias.yaml")); err != nil {
		t.Skipf("symlinks are not supported in this environment: %v", err)
	}

	_, err := ResolveAgentReference(context.Background(), parentPath, "alias.yaml")
	if err != nil && strings.Contains(err.Error(), traversalError) {
		t.Errorf("ResolveAgentReference(_, %q, %q) = %v, want no traversal rejection", parentPath, "alias.yaml", err)
	}
}

// TestResolveAgentReferenceMissingFileIsNotTraversal covers a reference to a
// config that does not exist. Canonicalisation cannot resolve it, but a typo
// must report the missing file rather than a containment failure, which is what
// os.path.realpath gives adk-python for free.
func TestResolveAgentReferenceMissingFileIsNotTraversal(t *testing.T) {
	_, parentPath := newAgentDir(t)

	_, err := ResolveAgentReference(context.Background(), parentPath, filepath.Join("nodes", "typo.yaml"))
	if err == nil {
		t.Fatal("ResolveAgentReference for a missing config succeeded, want an error")
	}
	if strings.Contains(err.Error(), traversalError) {
		t.Errorf("ResolveAgentReference for a missing config = %v, want a not-found error", err)
	}
	if !strings.Contains(err.Error(), "config file not found") {
		t.Errorf("ResolveAgentReference for a missing config = %v, want a not-found error", err)
	}
}

// TestResolveAgentReferenceThroughSymlinkedAgentDir covers an agent directory
// reached through a symlink. Both sides are canonicalised, so a reference
// inside it must still resolve.
func TestResolveAgentReferenceThroughSymlinkedAgentDir(t *testing.T) {
	base, parentPath := newAgentDir(t)
	agentDir := filepath.Dir(parentPath)

	linkedDir := filepath.Join(base, "linked_root")
	if err := os.Symlink(agentDir, linkedDir); err != nil {
		t.Skipf("symlinks are not supported in this environment: %v", err)
	}

	viaLink := filepath.Join(linkedDir, "root_agent.yaml")
	_, err := ResolveAgentReference(context.Background(), viaLink, "sub_agent.yaml")
	if err != nil && strings.Contains(err.Error(), traversalError) {
		t.Errorf("ResolveAgentReference(_, %q, %q) = %v, want no traversal rejection", viaLink, "sub_agent.yaml", err)
	}
}

// TestResolveAgentReferenceRejectsSiblingPrefixDir covers a sibling directory
// whose name begins with the agent directory's name. Containment has to compare
// whole path elements, not raw string prefixes.
func TestResolveAgentReferenceRejectsSiblingPrefixDir(t *testing.T) {
	base, parentPath := newAgentDir(t)

	siblingDir := filepath.Join(base, "agents", "root-evil")
	if err := os.MkdirAll(siblingDir, 0o755); err != nil {
		t.Fatalf("MkdirAll(%q) failed: %v", siblingDir, err)
	}
	if err := os.WriteFile(filepath.Join(siblingDir, "evil.yaml"), []byte("agent_class: LlmAgent\n"), 0o644); err != nil {
		t.Fatalf("WriteFile(evil.yaml) failed: %v", err)
	}

	refPath := filepath.Join("..", "root-evil", "evil.yaml")
	_, err := ResolveAgentReference(context.Background(), parentPath, refPath)
	if err == nil {
		t.Fatalf("ResolveAgentReference(_, %q, %q) succeeded, want error containing %q", parentPath, refPath, traversalError)
	}
	if !strings.Contains(err.Error(), traversalError) {
		t.Errorf("ResolveAgentReference(_, %q, %q) = %v, want error containing %q", parentPath, refPath, err, traversalError)
	}
}

// TestResolveAgentReferenceRejectsThroughDanglingSymlinkedDir covers a
// reference through a symlinked directory whose final component does not
// exist. realPath cannot resolve the reference in one EvalSymlinks call in
// that case, and falls back to resolving the longest existing prefix and
// rejoining the missing tail; that fallback must still see through the
// directory symlink and reject the escape, rather than comparing against the
// reference's unresolved, lexical spelling.
func TestResolveAgentReferenceRejectsThroughDanglingSymlinkedDir(t *testing.T) {
	base, parentPath := newAgentDir(t)
	agentDir := filepath.Dir(parentPath)

	elsewhere := filepath.Join(base, "elsewhere")
	if err := os.MkdirAll(elsewhere, 0o755); err != nil {
		t.Fatalf("MkdirAll(%q) failed: %v", elsewhere, err)
	}
	if err := os.Symlink(elsewhere, filepath.Join(agentDir, "dirlink")); err != nil {
		t.Skipf("symlinks are not supported in this environment: %v", err)
	}

	// "missing.yaml" does not exist under elsewhere, so the reference as a
	// whole cannot be resolved by a single EvalSymlinks call.
	refPath := filepath.Join("dirlink", "missing.yaml")
	_, err := ResolveAgentReference(context.Background(), parentPath, refPath)
	if err == nil {
		t.Fatalf("ResolveAgentReference(_, %q, %q) succeeded, want error containing %q", parentPath, refPath, traversalError)
	}
	if !strings.Contains(err.Error(), traversalError) {
		t.Errorf("ResolveAgentReference(_, %q, %q) = %v, want error containing %q", parentPath, refPath, err, traversalError)
	}
}

// TestResolveAgentReferenceCleansParentSegmentBeforeSymlinkResolution covers a
// reference that walks into a symlinked directory and back out with ".."
// before the final component. filepath.Join cleans "dirlink/.." away
// textually before anything looks at the filesystem, so the reference
// resolves as a plain sibling of the symlink inside the agent directory; the
// symlink's target is never consulted. This matches adk-python's
// normpath-then-realpath order and is pinned here so a future refactor does
// not silently change which file a reference like this loads.
func TestResolveAgentReferenceCleansParentSegmentBeforeSymlinkResolution(t *testing.T) {
	base, parentPath := newAgentDir(t)
	agentDir := filepath.Dir(parentPath)

	elsewhere := filepath.Join(base, "elsewhere")
	if err := os.MkdirAll(elsewhere, 0o755); err != nil {
		t.Fatalf("MkdirAll(%q) failed: %v", elsewhere, err)
	}
	if err := os.Symlink(elsewhere, filepath.Join(agentDir, "dirlink")); err != nil {
		t.Skipf("symlinks are not supported in this environment: %v", err)
	}

	refPath := filepath.Join("dirlink", "..", "sub_agent.yaml")
	_, err := ResolveAgentReference(context.Background(), parentPath, refPath)
	if err != nil && strings.Contains(err.Error(), traversalError) {
		t.Errorf("ResolveAgentReference(_, %q, %q) = %v, want no traversal rejection", parentPath, refPath, err)
	}
}

// TestResolveAgentReferenceRejectsEmptyPath covers the empty reference, which
// must be rejected outright rather than falling through to filepath.Dir's "."
// and silently resolving to the agent directory itself.
func TestResolveAgentReferenceRejectsEmptyPath(t *testing.T) {
	_, parentPath := newAgentDir(t)

	if _, err := ResolveAgentReference(context.Background(), parentPath, ""); err == nil {
		t.Error("ResolveAgentReference with an empty reference succeeded, want an error")
	}
}
