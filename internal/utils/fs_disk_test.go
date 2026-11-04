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
	"runtime"
	"strings"
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

	t.Run("symlinked base is accepted, a link below it is judged by where it lands", func(t *testing.T) {
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
		// a link that stays inside base keeps loading
		if _, _, err := SafeSubpath(base, "alias/f.txt"); err != nil {
			t.Errorf("SafeSubpath(%q, alias/f.txt) = %v, want success: the link stays inside base", base, err)
		}

		// a link that leaves base is refused, also with a missing tail below it
		outside := t.TempDir()
		if err := os.WriteFile(filepath.Join(outside, "secret.txt"), nil, 0o644); err != nil {
			t.Fatalf("WriteFile failed: %v", err)
		}
		out := filepath.Join(base, "out")
		if err := os.Symlink(outside, out); err != nil {
			t.Skipf("symlinks are not supported in this environment: %v", err)
		}
		for _, rel := range []string{"out/secret.txt", "out/missing.txt", "out/missing/deeper.txt"} {
			if _, _, err := SafeSubpath(base, rel); !errors.Is(err, ErrSymlinkInRelativePath) {
				t.Errorf("SafeSubpath(%q, %s) = %v, want %v", base, rel, err, ErrSymlinkInRelativePath)
			}
		}
		// a relative link destination with ".." is applied where the link sits,
		// as the operating system applies it: one climbs to base/f.txt and stays
		// inside, the other climbs out to the outside directory and is refused
		if err := os.Symlink(filepath.Join("..", "f.txt"), filepath.Join(realDir, "up.txt")); err != nil {
			t.Skipf("symlinks are not supported in this environment: %v", err)
		}
		if _, _, err := SafeSubpath(base, "real/up.txt"); err != nil {
			t.Errorf("SafeSubpath(%q, real/up.txt) = %v, want success: ../f.txt from real/ is base/f.txt", base, err)
		}
		relOut, err := filepath.Rel(realDir, filepath.Join(outside, "secret.txt"))
		if err != nil {
			t.Fatalf("Rel failed: %v", err)
		}
		if err := os.Symlink(relOut, filepath.Join(realDir, "climb.txt")); err != nil {
			t.Skipf("symlinks are not supported in this environment: %v", err)
		}
		if _, _, err := SafeSubpath(base, "real/climb.txt"); !errors.Is(err, ErrSymlinkInRelativePath) {
			t.Errorf("SafeSubpath(%q, real/climb.txt) = %v, want %v: the link climbs out through %q", base, err, ErrSymlinkInRelativePath, relOut)
		}

		// a dangling link is judged by where it points: outside is refused,
		// inside is returned so the caller's read reports the missing file
		if err := os.Symlink(filepath.Join(outside, "nope"), filepath.Join(base, "dangling_out")); err != nil {
			t.Skipf("symlinks are not supported in this environment: %v", err)
		}
		if _, _, err := SafeSubpath(base, "dangling_out"); !errors.Is(err, ErrSymlinkInRelativePath) {
			t.Errorf("SafeSubpath(%q, dangling_out) = %v, want %v", base, err, ErrSymlinkInRelativePath)
		}
		if err := os.Symlink("real/nope", filepath.Join(base, "dangling_in")); err != nil {
			t.Skipf("symlinks are not supported in this environment: %v", err)
		}
		if _, _, err := SafeSubpath(base, "dangling_in"); err != nil {
			t.Errorf("SafeSubpath(%q, dangling_in) = %v, want success: the link points inside base", base, err)
		}
		if err := os.Symlink(filepath.Join(outside, "secret.txt"), filepath.Join(realDir, "sneak.txt")); err != nil {
			t.Skipf("symlinks are not supported in this environment: %v", err)
		}
		if _, _, err := SafeSubpath(base, "real/sneak.txt"); !errors.Is(err, ErrSymlinkInRelativePath) {
			t.Errorf("SafeSubpath(%q, real/sneak.txt) = %v, want %v", base, err, ErrSymlinkInRelativePath)
		}
	})

	t.Run("ConfigMap-style layout keeps loading", func(t *testing.T) {
		// Kubernetes mounts: key -> ..data/key, ..data -> ..<timestamp>/
		cm := t.TempDir()
		ts := filepath.Join(cm, "..2026_10_22")
		if err := os.Mkdir(ts, 0o755); err != nil {
			t.Fatalf("Mkdir failed: %v", err)
		}
		if err := os.WriteFile(filepath.Join(ts, "node.yaml"), nil, 0o644); err != nil {
			t.Fatalf("WriteFile failed: %v", err)
		}
		if err := os.Symlink("..2026_10_22", filepath.Join(cm, "..data")); err != nil {
			t.Skipf("symlinks are not supported in this environment: %v", err)
		}
		if err := os.Symlink(filepath.Join("..data", "node.yaml"), filepath.Join(cm, "node.yaml")); err != nil {
			t.Skipf("symlinks are not supported in this environment: %v", err)
		}
		if _, _, err := SafeSubpath(cm, "node.yaml"); err != nil {
			t.Errorf("SafeSubpath(%q, node.yaml) = %v, want success for a ConfigMap-style link", cm, err)
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

// TestSafeSubpath_RootedLinkTargetWithoutDrive covers a Windows link whose
// target starts with a separator but names no drive, such as \outside\x.yaml.
// os.Readlink returns it unchanged and filepath.IsAbs is false for it, yet the
// operating system reads it from the root of the link's own drive. Joining it
// below the link's directory would judge one path while the load follows
// another. Elsewhere such a target is simply absolute, so the test is
// Windows-only.
func TestSafeSubpath_RootedLinkTargetWithoutDrive(t *testing.T) {
	if runtime.GOOS != "windows" {
		t.Skip("a rooted target without a drive exists only on Windows")
	}
	base := t.TempDir()
	if resolved, err := filepath.EvalSymlinks(base); err == nil {
		base = resolved
	}
	outside := t.TempDir()
	if err := os.WriteFile(filepath.Join(outside, "x.yaml"), nil, 0o644); err != nil {
		t.Fatalf("WriteFile failed: %v", err)
	}
	if filepath.VolumeName(base) != filepath.VolumeName(outside) {
		t.Skip("base and outside are on different drives")
	}
	rooted := strings.TrimPrefix(outside, filepath.VolumeName(outside))
	if filepath.IsAbs(rooted) {
		t.Fatalf("%q is absolute; the case needs a target without a drive", rooted)
	}
	if err := os.Symlink(rooted, filepath.Join(base, "out")); err != nil {
		t.Skipf("symlinks are not supported in this environment: %v", err)
	}
	for _, rel := range []string{"out/x.yaml", "out/missing.yaml"} {
		if _, _, err := SafeSubpath(base, rel); !errors.Is(err, ErrSymlinkInRelativePath) {
			t.Errorf("SafeSubpath(%q, %s) = %v, want %v", base, rel, err, ErrSymlinkInRelativePath)
		}
	}
}

// TestSafeSubpath_DotDotAfterLinkInTarget covers a link whose target contains a
// ".." after another link, such as "s/../y.yaml". filepath.EvalSymlinks, and so
// the loader, follows s first and applies ".." to what s points at; cleaning the
// target as text instead drops "s/.." before anything looks at s, and judges a
// y.yaml beside the link. Found in review.
func TestSafeSubpath_DotDotAfterLinkInTarget(t *testing.T) {
	root := t.TempDir()
	if resolved, err := filepath.EvalSymlinks(root); err == nil {
		root = resolved
	}
	base := filepath.Join(root, "base")
	outside := filepath.Join(root, "outside", "p")
	for _, d := range []string{
		base,
		filepath.Join(base, "sub1", "sub2"),
		filepath.Join(base, "deep"),
		filepath.Join(outside, "q"),
	} {
		if err := os.MkdirAll(d, 0o755); err != nil {
			t.Fatalf("MkdirAll(%q) failed: %v", d, err)
		}
	}
	for _, f := range []string{
		filepath.Join(outside, "y.yaml"),
		filepath.Join(base, "in.yaml"),
		filepath.Join(root, "x.yaml"),
	} {
		if err := os.WriteFile(f, nil, 0o644); err != nil {
			t.Fatalf("WriteFile(%q) failed: %v", f, err)
		}
	}
	link := func(dest, at string) {
		if err := os.Symlink(dest, at); err != nil {
			t.Skipf("symlinks are not supported in this environment: %v", err)
		}
	}
	// s -> a directory outside base; x.yaml -> "s/../y.yaml" opens outside/p/y.yaml.
	link(filepath.Join(outside, "q"), filepath.Join(base, "s"))
	link("s/../y.yaml", filepath.Join(base, "x.yaml"))
	// deep/up -> "..", which stays inside; chain.yaml -> "deep/up/../x.yaml"
	// opens root/x.yaml, one level above base, through links that each stay
	// inside.
	link("..", filepath.Join(base, "deep", "up"))
	link("deep/up/../x.yaml", filepath.Join(base, "chain.yaml"))
	// d -> sub1/sub2 (inside); back.yaml -> "d/../../in.yaml" opens base/in.yaml.
	link("sub1/sub2", filepath.Join(base, "d"))
	link("d/../../in.yaml", filepath.Join(base, "back.yaml"))

	for _, rel := range []string{"x.yaml", "chain.yaml"} {
		if got, _, err := SafeSubpath(base, rel); !errors.Is(err, ErrSymlinkInRelativePath) {
			t.Errorf("SafeSubpath(%q, %s) = (%q, %v), want %v: the link opens a file outside base", base, rel, got, err, ErrSymlinkInRelativePath)
		}
	}
	got, _, err := SafeSubpath(base, "back.yaml")
	if err != nil {
		t.Fatalf("SafeSubpath(%q, back.yaml) = %v, want success: the link opens base/in.yaml", base, err)
	}
	if want := filepath.Join(base, "in.yaml"); got != want {
		t.Errorf("SafeSubpath(%q, back.yaml) = %q, want %q, the file the link opens", base, got, want)
	}
}
