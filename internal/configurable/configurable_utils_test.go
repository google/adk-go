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
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// isContainmentRejection reports whether err is the containment check refusing a
// reference, as opposed to a later failure to load the config that it names.
// Asserting on the sentinels keeps these tests independent of the message text.
func isContainmentRejection(err error) bool {
	return errors.Is(err, errConfigReferenceNotLocal) || errors.Is(err, errConfigReferenceSymlink)
}

// TestIsLinkLike pins the mode predicate the component walk uses. It runs
// everywhere, unlike the junction test, which needs Windows to build one.
func TestIsLinkLike(t *testing.T) {
	for _, tc := range []struct {
		name string
		mode fs.FileMode
		want bool
	}{
		{name: "regular file", mode: 0o644, want: false},
		{name: "directory", mode: fs.ModeDir | 0o755, want: false},
		{name: "symlink", mode: fs.ModeSymlink | 0o777, want: true},
		// A Windows directory junction: os.Lstat reports a reparse point whose
		// tag is not IO_REPARSE_TAG_SYMLINK as irregular, so a ModeSymlink-only
		// test would walk past it.
		{name: "junction or other reparse point", mode: fs.ModeIrregular | 0o666, want: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := isLinkLike(tc.mode); got != tc.want {
				t.Errorf("isLinkLike(%v) = %v, want %v", tc.mode, got, tc.want)
			}
		})
	}
}

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
		wantErr error
	}{
		{
			name:    "absolute path",
			refPath: filepath.Join(string(os.PathSeparator), "etc", "passwd"),
			wantErr: errConfigReferenceNotLocal,
		},
		{
			name:    "parent traversal",
			refPath: filepath.Join("..", "..", "outside.yaml"),
			wantErr: errConfigReferenceNotLocal,
		},
		{
			name:    "symlink escaping the agent directory",
			refPath: "link.yaml",
			wantErr: errConfigReferenceSymlink,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			// The reference must be rejected by the containment check itself, not
			// merely fail later while the config is being loaded.
			_, err := ResolveAgentReference(context.Background(), parentPath, tc.refPath)
			if !errors.Is(err, tc.wantErr) {
				t.Errorf("ResolveAgentReference(_, %q, %q) = %v, want %v", parentPath, tc.refPath, err, tc.wantErr)
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
	if isContainmentRejection(err) {
		t.Errorf("ResolveAgentReference(_, %q, %q) = %v, want no containment rejection", parentPath, "sub_agent.yaml", err)
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

	if _, err := ResolveAgentReference(context.Background(), "root_agent.yaml", "sub_agent.yaml"); isContainmentRejection(err) {
		t.Errorf("ResolveAgentReference with a relative parent path = %v, want no containment rejection", err)
	}

	if _, err := ResolveAgentReference(context.Background(), "root_agent.yaml", filepath.Join("..", "..", "outside.yaml")); err == nil {
		t.Error("ResolveAgentReference with a relative parent path and an escaping reference succeeded, want an error")
	}
}

// TestResolveAgentReferenceSymlinkedParentDir covers a parent directory that is
// itself reached through a symlink, together with a reference that does not
// exist on disk. Resolving only the parent side of the containment check would
// leave the two sides rooted differently — the target has no symlinks to
// resolve, so it keeps its unresolved spelling — and a missing file would be
// reported as a traversal instead of as not found.
func TestResolveAgentReferenceSymlinkedParentDir(t *testing.T) {
	base := t.TempDir()
	if resolved, err := filepath.EvalSymlinks(base); err == nil {
		base = resolved
	}

	realDir := filepath.Join(base, "real")
	if err := os.MkdirAll(realDir, 0o755); err != nil {
		t.Fatalf("MkdirAll(%q) failed: %v", realDir, err)
	}
	if err := os.WriteFile(filepath.Join(realDir, "root_agent.yaml"), []byte("agent_class: LlmAgent\n"), 0o644); err != nil {
		t.Fatalf("WriteFile(root_agent.yaml) failed: %v", err)
	}

	alias := filepath.Join(base, "alias")
	if err := os.Symlink(realDir, alias); err != nil {
		t.Skipf("symlinks are not supported in this environment: %v", err)
	}

	parentPath := filepath.Join(alias, "root_agent.yaml")
	if _, err := ResolveAgentReference(context.Background(), parentPath, "missing.yaml"); isContainmentRejection(err) {
		t.Errorf("ResolveAgentReference(_, %q, %q) = %v, want no containment rejection", parentPath, "missing.yaml", err)
	}

	// The symlinked parent must not weaken the check itself.
	if _, err := ResolveAgentReference(context.Background(), parentPath, filepath.Join("..", "outside.yaml")); err == nil {
		t.Error("ResolveAgentReference through a symlinked parent with an escaping reference succeeded, want an error")
	}
}

// TestResolveConfigReferenceVolumeQualifiedRefs covers references that carry a
// volume name. On Windows a drive-relative reference such as `C:node.yaml` is
// not absolute yet still escapes the parent directory, by resolving against the
// current directory of that drive, and filepath.IsLocal is what rules it out.
//
// The expectation is platform-dependent rather than skipped, so that the test
// asserts something everywhere: on Unix these are ordinary, if odd, relative
// file names, and refusing them would be over-rejection.
func TestResolveConfigReferenceVolumeQualifiedRefs(t *testing.T) {
	_, parentPath := newAgentDir(t)

	for _, refPath := range []string{`C:node.yaml`, `D:\escaped.yaml`, `\\host\share\escaped.yaml`} {
		t.Run(refPath, func(t *testing.T) {
			// Volume-qualified on this platform means the reference escapes and must
			// be refused; otherwise it is a legal file name and must be accepted.
			wantRejected := filepath.VolumeName(refPath) != ""

			_, err := resolveConfigReference(parentPath, refPath)
			if gotRejected := err != nil; gotRejected != wantRejected {
				t.Errorf("resolveConfigReference(%q, %q) rejected = %v (%v), want rejected = %v",
					parentPath, refPath, gotRejected, err, wantRejected)
			}
		})
	}
}

// TestResolveConfigReferenceRefusesLinksThatStayInside pins the one behaviour
// this check deliberately chose: a link is refused for being a link, not for
// where it points, so a link wholly inside the config directory is refused too.
//
// Without this, an implementation that went back to resolving links and
// refusing only those that leave the directory would pass the rest of the suite
// unchanged — and it would carry the hole that motivated refusing them, since a
// link pointing outside at a target that does not exist yet resolves to nothing
// and passes containment.
//
// The projected-volume case is the cost of that choice, and it is the reason
// this is a trade rather than a free win: the kubelet materialises a Kubernetes
// ConfigMap as a `..data` symlink to a timestamped directory plus one symlink
// per key, so every reference in such a mount is refused. Bazel runfiles
// forests and Nix store paths have the same shape.
func TestResolveConfigReferenceRefusesLinksThatStayInside(t *testing.T) {
	const nodeYAML = "name: join_node\nagent_class: JoinNode\n"

	for _, tc := range []struct {
		name string
		// layout populates dir and returns the reference to resolve against
		// dir/root_agent.yaml.
		layout func(t *testing.T, dir string) string
	}{
		{
			name: "alias to a sibling in the same directory",
			layout: func(t *testing.T, dir string) string {
				target := filepath.Join(dir, "nodes", "join.yaml")
				if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
					t.Fatalf("MkdirAll(%q) failed: %v", filepath.Dir(target), err)
				}
				if err := os.WriteFile(target, []byte(nodeYAML), 0o644); err != nil {
					t.Fatalf("WriteFile(%q) failed: %v", target, err)
				}
				if err := os.Symlink(target, filepath.Join(dir, "alias.yaml")); err != nil {
					t.Skipf("symlinks are not supported in this environment: %v", err)
				}
				return "alias.yaml"
			},
		},
		{
			name: "kubernetes projected volume layout",
			layout: func(t *testing.T, dir string) string {
				data := filepath.Join(dir, "..2026_09_01_10_00_00.123456")
				if err := os.MkdirAll(data, 0o755); err != nil {
					t.Fatalf("MkdirAll(%q) failed: %v", data, err)
				}
				if err := os.WriteFile(filepath.Join(data, "join.yaml"), []byte(nodeYAML), 0o644); err != nil {
					t.Fatalf("WriteFile(join.yaml) failed: %v", err)
				}
				if err := os.Symlink(data, filepath.Join(dir, "..data")); err != nil {
					t.Skipf("symlinks are not supported in this environment: %v", err)
				}
				if err := os.Symlink(filepath.Join("..data", "join.yaml"), filepath.Join(dir, "join.yaml")); err != nil {
					t.Skipf("symlinks are not supported in this environment: %v", err)
				}
				return "join.yaml"
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			if resolved, err := filepath.EvalSymlinks(dir); err == nil {
				dir = resolved
			}
			refPath := tc.layout(t, dir)

			parentPath := filepath.Join(dir, "root_agent.yaml")
			if _, err := resolveConfigReference(parentPath, refPath); !errors.Is(err, errConfigReferenceSymlink) {
				t.Errorf("resolveConfigReference(%q, %q) = %v, want %v: a link inside the directory is refused for being a link",
					parentPath, refPath, err, errConfigReferenceSymlink)
			}
		})
	}
}

// TestResolveConfigReferenceCanonicalizesRegistryKey pins the EvalSymlinks call
// on the parent directory. That call is not what makes the containment check
// correct — the component walk uses os.Lstat, which follows every component but
// the last, so removing it changes no verdict. What it does is canonicalise the
// path returned from here, which callers use as the agentRegistry and
// nodeRegistry key. Without it, a directory and a symlink to that directory get
// separate cache entries and each builds its own copy of the same agent.
func TestResolveConfigReferenceCanonicalizesRegistryKey(t *testing.T) {
	base := t.TempDir()
	if resolved, err := filepath.EvalSymlinks(base); err == nil {
		base = resolved
	}

	realDir := filepath.Join(base, "real")
	if err := os.MkdirAll(realDir, 0o755); err != nil {
		t.Fatalf("MkdirAll(%q) failed: %v", realDir, err)
	}
	for _, name := range []string{"root_agent.yaml", "sub_agent.yaml"} {
		cfg := fmt.Sprintf("name: %s\nagent_class: LlmAgent\nmodel: gemini-2.0-flash\n", strings.TrimSuffix(name, ".yaml"))
		if err := os.WriteFile(filepath.Join(realDir, name), []byte(cfg), 0o644); err != nil {
			t.Fatalf("WriteFile(%q) failed: %v", name, err)
		}
	}

	aliasDir := filepath.Join(base, "alias")
	if err := os.Symlink(realDir, aliasDir); err != nil {
		t.Skipf("symlinks are not supported in this environment: %v", err)
	}

	realKey, err := resolveConfigReference(filepath.Join(realDir, "root_agent.yaml"), "sub_agent.yaml")
	if err != nil {
		t.Fatalf("resolveConfigReference through the real directory failed: %v", err)
	}
	aliasKey, err := resolveConfigReference(filepath.Join(aliasDir, "root_agent.yaml"), "sub_agent.yaml")
	if err != nil {
		t.Fatalf("resolveConfigReference through the symlinked directory failed: %v", err)
	}
	if realKey != aliasKey {
		t.Errorf("resolveConfigReference returned %q through the real directory and %q through a symlink to it, want one key",
			realKey, aliasKey)
	}

	// The consequence the key exists for: resolving through both spellings must
	// leave one registry entry, not one per spelling.
	for _, parentDir := range []string{realDir, aliasDir} {
		if _, err := ResolveAgentReference(context.Background(), filepath.Join(parentDir, "root_agent.yaml"), "sub_agent.yaml"); err != nil {
			t.Fatalf("ResolveAgentReference through %q failed: %v", parentDir, err)
		}
	}
	registryMu.RLock()
	defer registryMu.RUnlock()
	if _, ok := agentRegistry[filepath.Join(aliasDir, "sub_agent.yaml")]; ok {
		t.Errorf("agentRegistry holds a separate entry keyed by the symlinked spelling %q, want only %q",
			filepath.Join(aliasDir, "sub_agent.yaml"), realKey)
	}
	if _, ok := agentRegistry[realKey]; !ok {
		t.Errorf("agentRegistry has no entry keyed by the canonical path %q", realKey)
	}
}

// TestResolveConfigReferenceBelowNonDirectory covers a reference whose
// intermediate component is a regular file. Nothing can exist below it, so
// there is no link to find and no reason to report an inspection failure: the
// reference is simply not there, and the caller's own read says so. Reporting
// it separately would give callers a third class of error to distinguish from
// "refused" and "loaded".
func TestResolveConfigReferenceBelowNonDirectory(t *testing.T) {
	_, parentPath := newAgentDir(t)

	// sub_agent.yaml is a regular file, so sub_agent.yaml/nested.yaml cannot exist.
	refPath := filepath.Join("sub_agent.yaml", "nested.yaml")
	if _, err := resolveConfigReference(parentPath, refPath); err != nil {
		t.Errorf("resolveConfigReference(%q, %q) = %v, want no error: the reference cannot exist, which is the caller's read to report",
			parentPath, refPath, err)
	}
}

// TestResolveConfigReferenceEmptyRef covers the empty reference, which is
// usually an unfilled template rather than an attempt to escape. IsLocal
// rejects it along with the escaping spellings, so it needs its own message to
// avoid rendering as a bare "config reference must be ...: " with nothing after
// the colon.
func TestResolveConfigReferenceEmptyRef(t *testing.T) {
	_, parentPath := newAgentDir(t)

	_, err := resolveConfigReference(parentPath, "")
	if !errors.Is(err, errConfigReferenceNotLocal) {
		t.Fatalf("resolveConfigReference(%q, \"\") = %v, want %v", parentPath, err, errConfigReferenceNotLocal)
	}
	if !strings.Contains(err.Error(), "empty") {
		t.Errorf("resolveConfigReference(%q, \"\") = %q, want the message to name the reference as empty", parentPath, err)
	}
}
