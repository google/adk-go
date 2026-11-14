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
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"google.golang.org/adk/v2/internal/utils"
)

// isContainmentRejection reports whether err is the containment check refusing a
// reference, as opposed to a later failure to load the config that it names.
// Asserting on the sentinels keeps these tests independent of the message text.
func isContainmentRejection(err error) bool {
	return errors.Is(err, utils.ErrNotValidRelativePath) || errors.Is(err, utils.ErrSymlinkInRelativePath)
}

// newAgentDir lays out an agent directory with a sibling config and a file
// outside the directory, and returns the base directory, the agent directory
// and the parent agent config path. It creates no link, so the tests built on
// it run where symlink creation is unavailable (Windows without Developer
// Mode); a test that needs the escaping link calls linkOutside.
func newAgentDir(t *testing.T) (base, agentDir, agentPath string) {
	t.Helper()

	base = t.TempDir()
	// Resolve the temporary directory itself, since on some platforms it is
	// reached through a symlink (for example /var -> /private/var on macOS).
	if resolved, err := filepath.EvalSymlinks(base); err == nil {
		base = resolved
	}

	agentDir = filepath.Join(base, "agents", "root")
	if err := os.MkdirAll(agentDir, 0o755); err != nil {
		t.Fatalf("MkdirAll(%q) failed: %v", agentDir, err)
	}

	agentPath = filepath.Join(agentDir, "root_agent.yaml")
	if err := os.WriteFile(agentPath, []byte("agent_class: LlmAgent\n"), 0o644); err != nil {
		t.Fatalf("WriteFile(%q) failed: %v", agentPath, err)
	}
	if err := os.WriteFile(filepath.Join(agentDir, "sub_agent.yaml"), []byte("agent_class: LlmAgent\n"), 0o644); err != nil {
		t.Fatalf("WriteFile(sub_agent.yaml) failed: %v", err)
	}

	outside := filepath.Join(base, "outside.yaml")
	if err := os.WriteFile(outside, []byte("agent_class: LlmAgent\n"), 0o644); err != nil {
		t.Fatalf("WriteFile(%q) failed: %v", outside, err)
	}

	return base, agentDir, agentPath
}

// linkOutside adds agentDir/link.yaml, a symlink to the outside.yaml that
// newAgentDir wrote next to the agent directory. It skips the test where the
// link cannot be created, so only tests that need the link are skipped there.
func linkOutside(t *testing.T, base, agentDir string) {
	t.Helper()
	if err := os.Symlink(filepath.Join(base, "outside.yaml"), filepath.Join(agentDir, "link.yaml")); err != nil {
		t.Skipf("symlinks are not supported in this environment: %v", err)
	}
}

func TestResolveAgentReferenceRejectsEscapingConfigPath(t *testing.T) {
	base, agentDir, _ := newAgentDir(t)
	linkOutside(t, base, agentDir)
	tests := []struct {
		name    string
		refPath string
		wantErr error
	}{
		{
			name:    "absolute path",
			refPath: filepath.Join(string(os.PathSeparator), "etc", "passwd"),
			wantErr: utils.ErrNotValidRelativePath,
		},
		{
			name:    "parent traversal",
			refPath: filepath.Join("..", "..", "outside.yaml"),
			wantErr: utils.ErrNotValidRelativePath,
		},
		{
			name:    "symlink escaping the agent directory",
			refPath: "link.yaml",
			wantErr: utils.ErrSymlinkInRelativePath,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			// The reference must be rejected by the containment check itself, not
			// merely fail later while the config is being loaded.
			_, err := ResolveAgentReference(context.Background(), agentDir, tc.refPath)
			if !errors.Is(err, tc.wantErr) {
				t.Errorf("ResolveAgentReference(_, %q, %q) = %v, want %v", agentDir, tc.refPath, err, tc.wantErr)
			}
		})
	}
}

// TestResolveAgentReferenceAllowsPathsInsideAgentDir guards against the
// containment check rejecting legitimate references. The reference is not a
// loadable agent config, so an error is expected; it must not be the traversal
// error.
func TestResolveAgentReferenceAllowsPathsInsideAgentDir(t *testing.T) {
	_, agentDir, _ := newAgentDir(t)

	_, err := ResolveAgentReference(context.Background(), agentDir, "sub_agent.yaml")
	if isContainmentRejection(err) {
		t.Errorf("ResolveAgentReference(_, %q, %q) = %v, want no containment rejection", agentDir, "sub_agent.yaml", err)
	}
}

// TestResolveToolReferenceMcpToolsetNonStringArgs reproduces the bug where a
// non-string element in the McpToolset "args" or "tool_filter" lists triggered
// an unchecked type assertion that panicked and killed the process. The factory
// must now return a descriptive error instead of crashing.
func TestResolveToolReferenceMcpToolsetNonStringArgs(t *testing.T) {
	command := policyExecutable(t)
	ctx := policyContext(t, command, []string{"a", "b"})
	newArgs := func(serverArgs, toolFilter []any) map[string]any {
		return map[string]any{
			"stdio_connection_params": map[string]any{
				"server_params": map[string]any{
					"command": command,
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
			_, toolset, err := ResolveToolReference(ctx, "McpToolset", tc.args)
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

	agentDir := alias
	// A missing reference is reported by the read as not found, not as a containment refusal.
	_, err := ResolveAgentReference(context.Background(), agentDir, "missing.yaml")
	if err == nil || isContainmentRejection(err) || !strings.Contains(err.Error(), "config file not found") {
		t.Errorf("ResolveAgentReference(_, %q, %q) = %v, want a config file not found error that is not a containment rejection", agentDir, "missing.yaml", err)
	}

	// A link below the alias that names its target by the real path stays
	// inside: this is only true when the base is resolved as well, otherwise the
	// two sides are rooted differently and the reference is refused.
	if err := os.Symlink(filepath.Join(realDir, "root_agent.yaml"), filepath.Join(realDir, "self.yaml")); err != nil {
		t.Fatalf("Symlink(self.yaml) failed: %v", err)
	}
	if _, err := resolveConfigReference(agentDir, "self.yaml"); err != nil {
		t.Errorf("resolveConfigReference(%q, self.yaml) = %v, want success: the link lands inside the real directory", agentDir, err)
	}

	// The symlinked parent must not weaken the check itself. The outside file
	// exists, so an accepted reference would load it rather than fail as not
	// found, and the refusal must come from the lexical check, which runs
	// before any link is looked at.
	if err := os.WriteFile(filepath.Join(base, "outside.yaml"), []byte("agent_class: LlmAgent\n"), 0o644); err != nil {
		t.Fatalf("WriteFile(outside.yaml) failed: %v", err)
	}
	if _, err := ResolveAgentReference(context.Background(), agentDir, filepath.Join("..", "outside.yaml")); !errors.Is(err, utils.ErrNotValidRelativePath) {
		t.Errorf("ResolveAgentReference through a symlinked parent with an escaping reference = %v, want %v", err, utils.ErrNotValidRelativePath)
	}
}

// TestResolveConfigReferenceVolumeQualifiedRefs covers references that carry a
// volume name. On Windows a drive-relative reference such as `C:node.yaml` is
// not absolute yet still escapes the parent directory, by resolving against the
// current directory of that drive. SafeSubpath refuses these on Windows because
// filepath.IsLocal rejects both the ':' of a drive and the leading `\\` of a
// UNC path; the test keys its own expectation off filepath.VolumeName.
//
// The expectation is platform-dependent rather than skipped, so that the test
// asserts something everywhere: on Unix these are ordinary, if odd, relative
// file names, and refusing them would be over-rejection.
func TestResolveConfigReferenceVolumeQualifiedRefs(t *testing.T) {
	_, agentDir, _ := newAgentDir(t)

	for _, refPath := range []string{`C:node.yaml`, `D:\escaped.yaml`, `\\host\share\escaped.yaml`} {
		t.Run(refPath, func(t *testing.T) {
			// Volume-qualified on this platform means the reference escapes and must
			// be refused; otherwise it is a legal file name and must be accepted.
			wantRejected := filepath.VolumeName(refPath) != ""

			_, err := resolveConfigReference(agentDir, refPath)
			if gotRejected := err != nil; gotRejected != wantRejected && !errors.Is(err, os.ErrNotExist) {
				t.Errorf("resolveConfigReference(%q, %q) rejected = %v (%v), want rejected = %v",
					agentDir, refPath, gotRejected, err, wantRejected)
			}
		})
	}
}

// TestResolveConfigReferenceAcceptsLinksThatStayInside pins the behavior chosen
// in review: a link is judged by where it resolves, not by being a link, so a
// link wholly inside the config directory keeps loading (as in adk-python).
//
// The projected-volume case is the one that motivated it: the kubelet
// materializes a Kubernetes ConfigMap as a `..data` symlink to a timestamped
// directory plus one symlink per key, all inside the mount. Links that leave
// the directory stay refused (a Bazel runfiles link, say, which points into the
// execroot); see the SafeSubpath tests in internal/utils.
func TestResolveConfigReferenceAcceptsLinksThatStayInside(t *testing.T) {
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

			got, err := resolveConfigReference(dir, refPath)
			if err != nil {
				t.Fatalf("resolveConfigReference(%q, %q) = %v, want success: a link that stays inside the directory keeps loading", dir, refPath, err)
			}
			want, err := filepath.EvalSymlinks(filepath.Join(dir, refPath))
			if err != nil {
				t.Fatalf("EvalSymlinks failed: %v", err)
			}
			if got != want {
				t.Errorf("resolveConfigReference(%q, %q) = %q, want the real path %q", dir, refPath, got, want)
			}
		})
	}
}

// TestResolveConfigReferenceAliasedDirectoryShareKey covers a reference that
// reaches a config through a symlinked directory below the base. It loads, and
// the agent is keyed by the real path, so the same agent cannot be registered
// under two spellings.
func TestResolveConfigReferenceAliasedDirectoryShareKey(t *testing.T) {
	base := t.TempDir()
	if resolved, err := filepath.EvalSymlinks(base); err == nil {
		base = resolved
	}

	realDir := filepath.Join(base, "real")
	if err := os.MkdirAll(realDir, 0o755); err != nil {
		t.Fatalf("MkdirAll(%q) failed: %v", realDir, err)
	}
	// SequentialAgent, not LlmAgent: an LlmAgent needs a model, and building one
	// constructs a genai client, which fails in any environment without an API key.
	cfg := "name: sub_agent\nagent_class: SequentialAgent\n"
	if err := os.WriteFile(filepath.Join(realDir, "sub_agent.yaml"), []byte(cfg), 0o644); err != nil {
		t.Fatalf("WriteFile(sub_agent.yaml) failed: %v", err)
	}
	aliasDir := filepath.Join(base, "alias")
	if err := os.Symlink(realDir, aliasDir); err != nil {
		t.Skipf("symlinks are not supported in this environment: %v", err)
	}

	realKey, err := resolveConfigReference(base, "real/sub_agent.yaml")
	if err != nil {
		t.Fatalf("resolveConfigReference through the real directory failed: %v", err)
	}
	aliasKey, err := resolveConfigReference(base, "alias/sub_agent.yaml")
	if err != nil {
		t.Fatalf("resolveConfigReference through an inside link failed: %v", err)
	}
	if aliasKey != realKey {
		t.Errorf("alias key = %q, want the real key %q", aliasKey, realKey)
	}

	if _, err := ResolveAgentReference(context.Background(), base, "real/sub_agent.yaml"); err != nil {
		t.Fatalf("ResolveAgentReference through the real directory failed: %v", err)
	}
	if _, err := ResolveAgentReference(context.Background(), base, "alias/sub_agent.yaml"); err != nil {
		t.Errorf("ResolveAgentReference through an inside link failed: %v", err)
	}
	registryMu.RLock()
	defer registryMu.RUnlock()
	if _, ok := agentRegistry[configCacheKey{path: realKey}]; !ok {
		t.Errorf("agentRegistry has no entry keyed by the real path %q", realKey)
	}
	if _, ok := agentRegistry[configCacheKey{path: filepath.Join(aliasDir, "sub_agent.yaml")}]; ok {
		t.Errorf("agentRegistry holds an entry keyed by the symlinked spelling %q", filepath.Join(aliasDir, "sub_agent.yaml"))
	}
}

// TestResolveConfigReferenceBelowNonDirectory covers a reference whose
// intermediate component is a regular file. Nothing can exist below it, so
// there is no link to find and no reason to report an inspection failure: the
// reference is simply not there, and the caller's own read says so. Reporting
// it separately would give callers a third class of error to distinguish from
// "refused" and "loaded".
func TestResolveConfigReferenceBelowNonDirectory(t *testing.T) {
	_, agentDir, _ := newAgentDir(t)

	// sub_agent.yaml is a regular file, so sub_agent.yaml/nested.yaml cannot exist.
	refPath := filepath.Join("sub_agent.yaml", "nested.yaml")
	if _, err := resolveConfigReference(agentDir, refPath); err != nil {
		t.Errorf("resolveConfigReference(%q, %q) = %v, want no error: the reference cannot exist, which is the caller's read to report",
			agentDir, refPath, err)
	}
}

// TestResolveConfigReferenceEmptyRef covers the empty reference, which is
// usually an unfilled template rather than an attempt to escape. SafeSubpath
// refuses it at its first check with a message that says the reference is
// empty, rather than the generic "not a subpath" — the wording this test pins,
// so an unfilled template reads as such.
func TestResolveConfigReferenceEmptyRef(t *testing.T) {
	_, agentDir, _ := newAgentDir(t)

	_, err := resolveConfigReference(agentDir, "")
	if !errors.Is(err, utils.ErrNotValidRelativePath) {
		t.Fatalf("resolveConfigReference(%q, \"\") = %v, want %v", agentDir, err, utils.ErrNotValidRelativePath)
	}
	if !strings.Contains(err.Error(), "empty") {
		t.Errorf("resolveConfigReference(%q, \"\") = %q, want the message to name the reference as empty", agentDir, err)
	}
}

// withinDir reports whether path resolves to a location at or below dir. It is
// the property the whole containment check exists to guarantee: whatever
// resolveConfigReference accepts must satisfy this. filepath.Rel returns an
// error when the two are on different volumes (a Windows drive escape), and
// filepath.IsLocal is false for a relative result that climbs out with "..",
// so both forms of escape are caught.
func withinDir(dir, path string) bool {
	rel, err := filepath.Rel(dir, path)
	if err != nil {
		return false
	}
	return filepath.IsLocal(rel)
}

// TestResolveConfigReferenceOSSpecificRefs exercises references whose meaning
// depends on the operating system, and pins the single property that must hold
// on every platform: a reference is either refused, or resolved to a path that
// stays inside the parent directory — it can never resolve to a path outside
// it. The expectation is branched on runtime.GOOS rather than skipped, so the
// test asserts something concrete on both Windows and Unix.
//
// The cases all turn on path spellings meaning different things on Windows and
// Unix, which SafeSubpath honors through its platform-specific validation ("\"
// is a separator on Windows but a literal byte on Unix; forbidden characters
// and reserved device names apply only on Windows):
//
//   - Windows reserved device names (NUL, CON, COM1, LPT1, ...) name a device
//     rather than a file on Windows and are refused there; on Unix they are
//     ordinary file names and stay inside the directory.
//   - "\" is a path separator on Windows and an ordinary filename character on
//     Unix, so `..\..\outside.yaml` is a traversal that escapes on Windows and
//     a single, oddly-named local file on Unix, while `sub\child.yaml` is a
//     nested path on Windows and one flat file name on Unix — contained either
//     way.
//   - A drive-relative reference such as `C:node.yaml` carries a volume name on
//     Windows, where it resolves against that drive's current directory and so
//     escapes; on Unix the colon is a legal filename character.
//   - An absolute POSIX path is rooted on both platforms and refused on both.
//
// This layout deliberately avoids symlinks (unlike newAgentDir), so the Windows
// branch runs even where creating a symlink needs privileges the test process
// does not have.
func TestResolveConfigReferenceOSSpecificRefs(t *testing.T) {
	base := t.TempDir()
	if resolved, err := filepath.EvalSymlinks(base); err == nil {
		base = resolved
	}
	agentDir := filepath.Join(base, "agents", "root")
	if err := os.MkdirAll(agentDir, 0o755); err != nil {
		t.Fatalf("MkdirAll(%q) failed: %v", agentDir, err)
	}
	agentPath := filepath.Join(agentDir, "root_agent.yaml")
	if err := os.WriteFile(agentPath, []byte("agent_class: LlmAgent\n"), 0o644); err != nil {
		t.Fatalf("WriteFile(%q) failed: %v", agentPath, err)
	}

	windows := runtime.GOOS == "windows"

	tests := []struct {
		name         string
		refPath      string
		wantRejected bool
	}{
		{name: "reserved name NUL", refPath: "NUL", wantRejected: windows},
		{name: "reserved name COM1", refPath: "COM1", wantRejected: windows},
		{name: "reserved name LPT1", refPath: "LPT1", wantRejected: windows},
		{name: "reserved name PhysicalDrive0", refPath: `\\.\PhysicalDrive0`, wantRejected: windows},
		{name: "reserved name direct c:", refPath: `\\.\C:`, wantRejected: windows},
		{name: "reserved name direct c:\\", refPath: `\\.\C:\`, wantRejected: windows},
		{name: "reserved name Charger0", refPath: `\\.\Charger0`, wantRejected: windows},
		{name: "reserved name CON", refPath: "CON", wantRejected: windows},
		{name: "reserved name CONIN$", refPath: "CONIN$", wantRejected: windows},
		{name: "reserved name CONOUT$", refPath: "CONOUT$", wantRejected: windows},
		// Backslash as a separator (Windows) climbs out; as a literal character
		// (Unix) it is one file name that stays put.
		{name: "backslash traversal", refPath: `..\..\outside.yaml`, wantRejected: windows},
		// Accepted on both, and contained on both: a nested path on Windows, a
		// flat file name on Unix. Exercises the containment branch on Windows.
		{name: "backslash subdir", refPath: `sub\child.yaml`, wantRejected: false},
		// Drive-relative: a volume name on Windows, a legal name on Unix.
		{name: "drive-relative ref", refPath: `C:node.yaml`, wantRejected: windows},
		// Rooted on every platform.
		{name: "posix absolute", refPath: "/etc/passwd", wantRejected: true},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := resolveConfigReference(agentDir, tc.refPath)

			if gotRejected := err != nil; gotRejected != tc.wantRejected {
				t.Fatalf("resolveConfigReference(%q, %q) rejected = %v (%v), want rejected = %v",
					agentDir, tc.refPath, gotRejected, err, tc.wantRejected)
			}
			if err != nil {
				// A rejection here is the lexical containment check refusing the
				// reference, never a later failure to load what it names.
				if !errors.Is(err, utils.ErrNotValidRelativePath) {
					t.Errorf("resolveConfigReference(%q, %q) = %v, want %v",
						agentPath, tc.refPath, err, utils.ErrNotValidRelativePath)
				}
				return
			}
			// Accepted: the resolved path must not escape the parent directory.
			if !withinDir(agentDir, got) {
				t.Errorf("resolveConfigReference(%q, %q) = %q, which escapes the parent directory %q",
					agentPath, tc.refPath, got, agentDir)
			}
		})
	}
}

// TestFromConfigRefusesEscapingAgentReferences drives the containment check on
// sub_agents and AgentTool references through the public FromConfig, which is
// where the directory handed to the factories takes effect, and pins that a
// refusal names the reference as written. The outside file exists, so with the
// check removed the reference would load (and fail on its missing name) rather
// than fail as not found.
func TestFromConfigRefusesEscapingAgentReferences(t *testing.T) {
	base, agentDir, _ := newAgentDir(t)
	if err := os.WriteFile(filepath.Join(agentDir, "sub.yaml"), []byte("name: sub\nagent_class: SequentialAgent\n"), 0o644); err != nil {
		t.Fatalf("WriteFile(sub.yaml) failed: %v", err)
	}
	// Building an LlmAgent constructs a genai client, which needs a key but
	// makes no request.
	t.Setenv("GOOGLE_API_KEY", "test-key")

	for _, tc := range []struct {
		name    string
		config  string
		ref     string
		wantErr bool
	}{
		{
			name:    "sub_agents inside",
			config:  "name: root\nagent_class: SequentialAgent\nsub_agents:\n  - config_path: sub.yaml\n",
			ref:     "sub.yaml",
			wantErr: false,
		},
		{
			name:    "sub_agents escaping",
			config:  "name: root\nagent_class: SequentialAgent\nsub_agents:\n  - config_path: ../../outside.yaml\n",
			ref:     "../../outside.yaml",
			wantErr: true,
		},
		{
			name:    "AgentTool escaping",
			config:  "name: root\nagent_class: LlmAgent\nmodel: gemini-2.5-flash\ntools:\n  - name: AgentTool\n    args:\n      agent:\n        config_path: ../../outside.yaml\n",
			ref:     "../../outside.yaml",
			wantErr: true,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			path := filepath.Join(agentDir, strings.ReplaceAll(tc.name, " ", "_")+".yaml")
			if err := os.WriteFile(path, []byte(tc.config), 0o644); err != nil {
				t.Fatalf("WriteFile(%q) failed: %v", path, err)
			}
			_, err := FromConfig(context.Background(), path)
			if !tc.wantErr {
				if err != nil {
					t.Fatalf("FromConfig(%q) = %v, want success", path, err)
				}
				return
			}
			if !isContainmentRejection(err) {
				t.Fatalf("FromConfig(%q) = %v, want a containment rejection of %q (outside file: %s)", path, err, tc.ref, filepath.Join(base, "outside.yaml"))
			}
			if !strings.Contains(err.Error(), fmt.Sprintf("%q", tc.ref)) {
				t.Errorf("FromConfig(%q) = %q, want the error to name the reference %q", path, err, tc.ref)
			}
		})
	}
}

// TestFromConfigDotDotAfterLinkInTarget drives the review case through
// FromConfig: a sub_agents reference to a link whose target is "s/../y.yaml",
// with s linking outside the config directory, must be refused, and one whose
// target is "d/../../in.yaml", which opens the directory's own in.yaml, must
// load. The check judges the file the loader then opens.
func TestFromConfigDotDotAfterLinkInTarget(t *testing.T) {
	write := func(p, s string) {
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatalf("MkdirAll failed: %v", err)
		}
		if err := os.WriteFile(p, []byte(s), 0o644); err != nil {
			t.Fatalf("WriteFile failed: %v", err)
		}
	}
	link := func(dest, at string) {
		if err := os.Symlink(dest, at); err != nil {
			t.Skipf("symlinks are not supported in this environment: %v", err)
		}
	}
	root := t.TempDir()
	if resolved, err := filepath.EvalSymlinks(root); err == nil {
		root = resolved
	}
	agentDir := filepath.Join(root, "agent")
	outside := filepath.Join(root, "outside", "p")

	// s -> a directory outside agentDir; x.yaml -> "s/../y.yaml", which the
	// loader resolves to outside/p/y.yaml.
	write(filepath.Join(outside, "y.yaml"), "agent_class: SequentialAgent\nname: outside_y\n")
	write(filepath.Join(outside, "q", ".keep"), "")
	write(filepath.Join(agentDir, ".keep"), "")
	link(filepath.Join(outside, "q"), filepath.Join(agentDir, "s"))
	link("s/../y.yaml", filepath.Join(agentDir, "x.yaml"))
	write(filepath.Join(agentDir, "escape.yaml"), "agent_class: SequentialAgent\nname: root\nsub_agents:\n  - config_path: x.yaml\n")
	_, err := FromConfig(context.Background(), filepath.Join(agentDir, "escape.yaml"))
	if !errors.Is(err, utils.ErrSymlinkInRelativePath) {
		t.Errorf("FromConfig(escape.yaml) = %v, want %v: x.yaml opens %s", err, utils.ErrSymlinkInRelativePath, filepath.Join(outside, "y.yaml"))
	}
	if err != nil && !strings.Contains(err.Error(), `config reference "x.yaml"`) {
		t.Errorf("FromConfig(escape.yaml) = %q, want the error to name the reference", err)
	}

	// d -> sub1/sub2 (inside); back.yaml -> "d/../../in.yaml", which the loader
	// resolves to agentDir/in.yaml.
	write(filepath.Join(agentDir, "in.yaml"), "agent_class: SequentialAgent\nname: inside\n")
	write(filepath.Join(agentDir, "sub1", "sub2", ".keep"), "")
	link("sub1/sub2", filepath.Join(agentDir, "d"))
	link("d/../../in.yaml", filepath.Join(agentDir, "back.yaml"))
	write(filepath.Join(agentDir, "inside.yaml"), "agent_class: SequentialAgent\nname: root2\nsub_agents:\n  - config_path: back.yaml\n")
	if _, err := FromConfig(context.Background(), filepath.Join(agentDir, "inside.yaml")); err != nil {
		t.Errorf("FromConfig(inside.yaml) = %v, want success: back.yaml opens %s", err, filepath.Join(agentDir, "in.yaml"))
	}
}
