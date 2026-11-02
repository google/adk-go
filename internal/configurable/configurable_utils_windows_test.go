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

// mkJunction creates a directory junction at link pointing at target, or
// skips the test where cmd's mklink cannot.
func mkJunction(t *testing.T, link, target string) {
	t.Helper()
	if out, err := exec.Command("cmd", "/c", "mklink", "/J", link, target).CombinedOutput(); err != nil {
		t.Skipf("cannot create a directory junction here: %v: %s", err, out)
	}
}

// TestResolveConfigReferenceRefusesJunction covers the Windows-only shape of the
// containment check: a directory junction is a reparse point that redirects the
// read like a symlink does, but since Go 1.23 os.Lstat reports it as
// ModeIrregular rather than ModeSymlink, and filepath.EvalSymlinks does not
// follow it. A junction needs no elevation to create, unlike a symlink, so this
// is the cheaper escape of the two for a caller to arrange.
//
// The target is left missing in one case on purpose: that is the shape the
// containment check exists for, since a link whose target does not exist yet
// stops being missing the moment anything creates it.
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
			if tc.createTarget {
				outside := filepath.Join(outsideDir, "outside.yaml")
				if err := os.WriteFile(outside, []byte("name: outside\n"), 0o644); err != nil {
					t.Fatalf("WriteFile(outside) = %v, want no error", err)
				}
			}
			mkJunction(t, filepath.Join(agentDir, "escape"), outsideDir)

			got, err := resolveConfigReference(agentDir, tc.refPath)
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

// TestResolveConfigReferenceAcceptsJunctionsInside pins the other half on
// Windows: a junction is judged by where it lands, like a symlink, so one that
// stays inside the config directory loads, a config directory below a junction
// can reference its files, and so can a config directory that is itself a
// junction. filepath.EvalSymlinks would refuse all three, since it does not
// follow a junction in the middle of a path.
func TestResolveConfigReferenceAcceptsJunctionsInside(t *testing.T) {
	base := t.TempDir()
	if resolved, err := filepath.EvalSymlinks(base); err == nil {
		base = resolved
	}
	realDir := filepath.Join(base, "real")
	nodesDir := filepath.Join(realDir, "agent", "nodes")
	if err := os.MkdirAll(nodesDir, 0o755); err != nil {
		t.Fatalf("MkdirAll(%s) = %v, want no error", nodesDir, err)
	}
	if err := os.WriteFile(filepath.Join(nodesDir, "join.yaml"), []byte("name: join\n"), 0o644); err != nil {
		t.Fatalf("WriteFile(join.yaml) = %v, want no error", err)
	}
	if err := os.WriteFile(filepath.Join(base, "outside.yaml"), []byte("name: outside\n"), 0o644); err != nil {
		t.Fatalf("WriteFile(outside.yaml) = %v, want no error", err)
	}
	agentDir := filepath.Join(realDir, "agent")
	mkJunction(t, filepath.Join(agentDir, "alias"), nodesDir) // junction inside the config directory
	mkJunction(t, filepath.Join(base, "jroot"), realDir)      // config directory below a junction
	mkJunction(t, filepath.Join(base, "jagent"), agentDir)    // config directory that is a junction

	for _, tc := range []struct {
		name      string
		configDir string
		refPath   string
	}{
		{name: "junction inside the config directory", configDir: agentDir, refPath: `alias\join.yaml`},
		{name: "config directory below a junction", configDir: filepath.Join(base, "jroot", "agent"), refPath: `nodes\join.yaml`},
		{name: "config directory is a junction", configDir: filepath.Join(base, "jagent"), refPath: `nodes\join.yaml`},
		{name: "missing file below a junction is left for the read", configDir: filepath.Join(base, "jagent"), refPath: `nodes\missing.yaml`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := resolveConfigReference(tc.configDir, tc.refPath); err != nil {
				t.Errorf("resolveConfigReference(%q, %q) = %v, want success", tc.configDir, tc.refPath, err)
			}
		})
	}

	// Being reached through a junction does not weaken the check: an escape
	// from below it is still refused.
	for _, configDir := range []string{filepath.Join(base, "jroot", "agent"), filepath.Join(base, "jagent")} {
		if _, err := resolveConfigReference(configDir, `..\..\outside.yaml`); !isContainmentRejection(err) {
			t.Errorf("resolveConfigReference(%q, ..\\..\\outside.yaml) = %v, want a containment rejection", configDir, err)
		}
	}
}
