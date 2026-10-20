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

package utils

import (
	"errors"
	"os"
	"path/filepath"
	"testing"
)

// TestSafeSubpath_OnDisk drives SafeSubpath against a real directory, so the
// component walk runs instead of stopping at a base that does not exist.
func TestSafeSubpath_OnDisk(t *testing.T) {
	base := t.TempDir()
	if resolved, err := filepath.EvalSymlinks(base); err == nil {
		base = resolved
	}
	if err := os.WriteFile(filepath.Join(base, "file.txt"), []byte("x"), 0o644); err != nil {
		t.Fatalf("WriteFile failed: %v", err)
	}

	t.Run("existing file resolves to the file, not the base", func(t *testing.T) {
		got, rel, err := SafeSubpath(base, "file.txt")
		if err != nil {
			t.Fatalf("SafeSubpath(%q, file.txt) = %v, want success", base, err)
		}
		if want := filepath.Join(base, "file.txt"); got != want || rel != "file.txt" {
			t.Errorf("SafeSubpath(%q, file.txt) = (%q, %q), want (%q, %q)", base, got, rel, want, "file.txt")
		}
	})

	// A missing or mistyped reference is the caller's read to report, not a
	// containment refusal.
	for _, tc := range []struct{ name, rel string }{
		{"missing file", "sbu.yaml"},
		{"missing directory", "nodes/n.yaml"},
		{"below a regular file", "file.txt/nested.yaml"},
	} {
		t.Run(tc.name+" is not a containment refusal", func(t *testing.T) {
			got, _, err := SafeSubpath(base, tc.rel)
			if err != nil {
				t.Fatalf("SafeSubpath(%q, %q) = %v, want success: nothing below the missing component can exist", base, tc.rel, err)
			}
			if want := filepath.Join(base, tc.rel); got != want {
				t.Errorf("SafeSubpath(%q, %q) = %q, want %q", base, tc.rel, got, want)
			}
		})
	}

	t.Run("symlinked base is accepted, a symlink below it is not", func(t *testing.T) {
		realDir := filepath.Join(base, "real")
		if err := os.Mkdir(realDir, 0o755); err != nil {
			t.Fatalf("Mkdir failed: %v", err)
		}
		if err := os.WriteFile(filepath.Join(realDir, "f.txt"), nil, 0o644); err != nil {
			t.Fatalf("WriteFile failed: %v", err)
		}
		alias := filepath.Join(base, "alias")
		if err := os.Symlink(realDir, alias); err != nil {
			t.Skipf("symlinks are not supported in this environment: %v", err)
		}
		if _, _, err := SafeSubpath(alias, "f.txt"); err != nil {
			t.Errorf("SafeSubpath(%q, f.txt) = %v, want success: the base is trusted", alias, err)
		}
		if _, _, err := SafeSubpath(base, "alias/f.txt"); !errors.Is(err, ErrSymlinkInRelativePath) {
			t.Errorf("SafeSubpath(%q, alias/f.txt) = %v, want %v", base, err, ErrSymlinkInRelativePath)
		}
	})
}

// TestSafeSubpath_FilesystemRoot covers a base that is a filesystem root, where
// the "base plus separator" prefix is itself root-shaped and no path would ever
// be below it.
func TestSafeSubpath_FilesystemRoot(t *testing.T) {
	root := filepath.VolumeName(os.TempDir()) + string(filepath.Separator)

	got, rel, err := SafeSubpath(root, "nodes/n.yaml")
	if err != nil {
		t.Fatalf("SafeSubpath(%q, nodes/n.yaml) = %v, want success", root, err)
	}
	if want := filepath.Join(root, "nodes", "n.yaml"); got != want || rel != filepath.Join("nodes", "n.yaml") {
		t.Errorf("SafeSubpath(%q, nodes/n.yaml) = (%q, %q), want (%q, ...)", root, got, rel, want)
	}
	if _, _, err := SafeSubpath(root, "../x"); !errors.Is(err, ErrNotValidRelativePath) {
		t.Errorf("SafeSubpath(%q, ../x) = %v, want %v", root, err, ErrNotValidRelativePath)
	}
}
