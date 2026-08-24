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
	"os"
	"path/filepath"
	"testing"
)

// isContainmentRejection reports whether err is the containment check refusing a
// reference, as opposed to a later failure to load the config that it names.
// Asserting on the sentinels keeps these tests independent of the message text.
func isContainmentRejection(err error) bool {
	return errors.Is(err, errConfigReferenceNotLocal) || errors.Is(err, errConfigReferenceSymlink)
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

// TestResolveConfigReferenceRejectsVolumeQualifiedRefs covers references that
// carry a volume name. On Windows a drive-relative reference such as
// `C:node.yaml` is not absolute yet still escapes the parent directory, by
// resolving against the current directory of that drive; filepath.IsLocal is
// what rules it out. VolumeName is empty on Unix, where these three are
// ordinary (if odd) relative file names and are correctly accepted, so the
// assertion only applies where the platform gives them a volume.
func TestResolveConfigReferenceRejectsVolumeQualifiedRefs(t *testing.T) {
	_, parentPath := newAgentDir(t)

	for _, refPath := range []string{`C:node.yaml`, `D:\escaped.yaml`, `\\host\share\escaped.yaml`} {
		_, err := resolveConfigReference(parentPath, refPath)
		if filepath.VolumeName(refPath) == "" {
			// Not volume-qualified on this platform; nothing to assert.
			continue
		}
		if err == nil {
			t.Errorf("resolveConfigReference(%q, %q) succeeded, want rejection", parentPath, refPath)
		}
	}
}
