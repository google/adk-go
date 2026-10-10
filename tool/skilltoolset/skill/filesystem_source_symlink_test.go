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

package skill

import (
	"errors"
	"io"
	"os"
	"path/filepath"
	"testing"
)

// TestFileSystemSource_LoadResource_SymlinkEscape exercises LoadResource
// against a real on-disk filesystem (os.DirFS), since fstest.MapFS has no
// notion of symbolic links and so cannot represent this attack at all.
//
// A skill is free to ship a symlink inside assets/, references/, or
// scripts/ (a normal thing for a skill bundle to do, e.g. linking a shared
// asset). Before this change, LoadResource validated resourcePath as a
// string but then opened it with the OS, which transparently follows any
// symlink in the path -- so a symlink pointing outside the skill's own
// directory let LoadResource return the content of an arbitrary file on
// disk. This test fails without the rejectSymlinkPath check in
// LoadResource.
func TestFileSystemSource_LoadResource_SymlinkEscape(t *testing.T) {
	root := t.TempDir()

	// A file outside any skill directory that no skill should ever be able
	// to read.
	secretDir := filepath.Join(root, "outside-secret")
	if err := os.MkdirAll(secretDir, 0o755); err != nil {
		t.Fatal(err)
	}
	secretPath := filepath.Join(secretDir, "secret.txt")
	if err := os.WriteFile(secretPath, []byte("top-secret"), 0o600); err != nil {
		t.Fatal(err)
	}

	skillDir := filepath.Join(root, "test-skill")
	assetsDir := filepath.Join(skillDir, "assets")
	if err := os.MkdirAll(assetsDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(skillDir, "SKILL.md"),
		[]byte("---\nname: test-skill\ndescription: test\n---\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	// An ordinary file, to confirm the fix does not break normal access.
	if err := os.WriteFile(filepath.Join(assetsDir, "image.png"),
		[]byte("image-data"), 0o644); err != nil {
		t.Fatal(err)
	}
	// The symlink a malicious (or merely careless) skill author ships
	// inside an otherwise-ordinary resource directory.
	if err := os.Symlink(secretDir, filepath.Join(assetsDir, "link")); err != nil {
		t.Skipf("symlinks not supported on this filesystem: %v", err)
	}

	source := NewFileSystemSource(os.DirFS(root))

	t.Run("Error_Symlink_Escape", func(t *testing.T) {
		resource, err := source.LoadResource(t.Context(), "test-skill", "assets/link/secret.txt")
		if !errors.Is(err, ErrInvalidResourcePath) {
			t.Fatalf("LoadResource(assets/link/secret.txt) error = %v, want %v", err, ErrInvalidResourcePath)
		}
		if resource != nil {
			_ = resource.Close()
		}
	})

	t.Run("Success_Ordinary_File_Still_Served", func(t *testing.T) {
		resource, err := source.LoadResource(t.Context(), "test-skill", "assets/image.png")
		if err != nil {
			t.Fatalf("LoadResource(assets/image.png) error = %v, want nil", err)
		}
		defer func() { _ = resource.Close() }()
		got, err := io.ReadAll(resource)
		if err != nil {
			t.Fatalf("read resource: %v", err)
		}
		if string(got) != "image-data" {
			t.Errorf("LoadResource(assets/image.png) = %q, want %q", got, "image-data")
		}
	})
}
