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
