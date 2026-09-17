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

//go:build windows

package configurable

import (
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

// TestResolveConfigReferenceRefusesJunction covers the Windows-only shape of the
// containment check: a directory junction is a reparse point that redirects the
// read like a symlink does, but since Go 1.23 os.Lstat reports it as
// ModeIrregular rather than ModeSymlink, so a ModeSymlink-only test walks past
// it. A junction needs no elevation to create, unlike a symlink, so this is the
// cheaper escape of the two for a caller to arrange.
//
// The target is left missing in one case on purpose: that is the shape the
// containment check exists for, since a link whose target does not exist yet
// resolves to nothing and stops being missing the moment anything creates it.
func TestResolveConfigReferenceRefusesJunction(t *testing.T) {
	for _, tc := range []struct {
		name         string
		createTarget bool
		refPath      string
	}{
		{name: "target exists", createTarget: true, refPath: `escape\outside.yaml`},
		{name: "target does not exist yet", createTarget: false, refPath: `escape\outside.yaml`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			base := t.TempDir()
			if resolved, err := filepath.EvalSymlinks(base); err == nil {
				base = resolved
			}
			agentDir := filepath.Join(base, "agent")
			outsideDir := filepath.Join(base, "outside")
			for _, dir := range []string{agentDir, outsideDir} {
				if err := os.Mkdir(dir, 0o755); err != nil {
					t.Fatalf("Mkdir(%s) = %v, want no error", dir, err)
				}
			}
			parentPath := filepath.Join(agentDir, "agent.yaml")
			if err := os.WriteFile(parentPath, []byte("name: parent\n"), 0o644); err != nil {
				t.Fatalf("WriteFile(parent) = %v, want no error", err)
			}
			if tc.createTarget {
				outside := filepath.Join(outsideDir, "outside.yaml")
				if err := os.WriteFile(outside, []byte("name: outside\n"), 0o644); err != nil {
					t.Fatalf("WriteFile(outside) = %v, want no error", err)
				}
			}

			junction := filepath.Join(agentDir, "escape")
			if out, err := exec.Command("cmd", "/c", "mklink", "/J", junction, outsideDir).CombinedOutput(); err != nil {
				t.Skipf("cannot create a directory junction here: %v: %s", err, out)
			}

			got, err := resolveConfigReference(parentPath, tc.refPath)
			if err == nil {
				t.Fatalf("resolveConfigReference(%q) = %q, want a rejection: the junction leads outside %s",
					tc.refPath, got, agentDir)
			}
			if !isContainmentRejection(err) {
				t.Errorf("resolveConfigReference(%q) = %v, want a containment rejection sentinel", tc.refPath, err)
			}
		})
	}
}
