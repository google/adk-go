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
	"bytes"
	"errors"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"testing"
	"testing/fstest"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
)

func TestFileSystemSource_ListFrontmatters(t *testing.T) {
	tests := []struct {
		name    string
		source  Source
		want    []*Frontmatter
		wantErr error
	}{
		{
			name: "Success",
			source: NewFileSystemSource(plainFS{fstest.MapFS{
				"math-skill/SKILL.md": &fstest.MapFile{
					Data: []byte("---\nname: math-skill\ndescription: test\n---\n"),
				},
				"weather-skill/SKILL.md": &fstest.MapFile{
					Data: []byte("---\nname: weather-skill\ndescription: test\n---\n"),
				},
				"random-file.txt":   &fstest.MapFile{Data: []byte("should be ignored")},
				"SKILL.md":          &fstest.MapFile{Data: []byte("should be ignored")},
				"dir/not-skill.txt": &fstest.MapFile{Data: []byte("should be ignored")},
				"sub/dir/SKILL.md":  &fstest.MapFile{Data: []byte("should be ignored")},
			}}),
			want: []*Frontmatter{
				{Name: "math-skill", Description: "test"},
				{Name: "weather-skill", Description: "test"},
			},
		},
		{
			name: "Allowed tools formats",
			source: NewFileSystemSource(plainFS{fstest.MapFS{
				"spec-skill/SKILL.md": &fstest.MapFile{
					Data: []byte("---\nname: spec-skill\ndescription: test\nallowed-tools: Bash(git:*) Bash(jq:*) Read\n---\nInstructions."),
				},
				"list-skill/SKILL.md": &fstest.MapFile{
					Data: []byte("---\nname: list-skill\ndescription: test\nallowed-tools:\n  - Bash(git:*)\n  - Bash(jq:*)\n  - Read\n---\nInstructions."),
				},
				"other-skill/SKILL.md": &fstest.MapFile{
					Data: []byte("---\nname: other-skill\ndescription: test\n---\nInstructions."),
				},
			}}),
			want: []*Frontmatter{
				{Name: "spec-skill", Description: "test", AllowedTools: []string{"Bash(git:*)", "Bash(jq:*)", "Read"}},
				{Name: "list-skill", Description: "test", AllowedTools: []string{"Bash(git:*)", "Bash(jq:*)", "Read"}},
				{Name: "other-skill", Description: "test"},
			},
		},
		{
			name: "Name mismatch",
			source: NewFileSystemSource(plainFS{fstest.MapFS{
				"test-skill/SKILL.md": &fstest.MapFile{Data: []byte("---\nname: wrong-skill\ndescription: test\n---\n")},
			}}),
			wantErr: ErrInvalidSkillName,
		},
		{
			name: "Invalid frontmatter",
			source: NewFileSystemSource(plainFS{fstest.MapFS{
				"test-skill/SKILL.md": &fstest.MapFile{Data: []byte("---[INVALID_YAML")},
			}}),
			wantErr: ErrInvalidFrontmatter,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := tt.source.ListFrontmatters(t.Context())

			if !errors.Is(err, tt.wantErr) {
				t.Errorf("ListFrontmatters() expected error %v, got %v", tt.wantErr, err)
			}
			if diff := cmp.Diff(tt.want, got, cmpopts.SortSlices(func(a, b *Frontmatter) bool { return a.Name < b.Name })); diff != "" {
				t.Errorf("ListFrontmatters() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestFileSystemSource_LoadFrontmatter(t *testing.T) {
	tests := []struct {
		name    string
		source  Source
		want    *Frontmatter
		wantErr error
	}{
		{
			name: "Success",
			source: NewFileSystemSource(plainFS{fstest.MapFS{
				"test-skill/SKILL.md": &fstest.MapFile{Data: []byte("---\nname: test-skill\ndescription: test\n---\n")},
			}}),
			want: &Frontmatter{Name: "test-skill", Description: "test"},
		},
		{
			name: "Name mismatch",
			source: NewFileSystemSource(plainFS{fstest.MapFS{
				"test-skill/SKILL.md": &fstest.MapFile{Data: []byte("---\nname: wrong-skill\ndescription: test\n---\n")},
			}}),
			wantErr: ErrInvalidSkillName,
		},
		{
			name: "Invalid frontmatter",
			source: NewFileSystemSource(plainFS{fstest.MapFS{
				"test-skill/SKILL.md": &fstest.MapFile{Data: []byte("---[INVALID_YAML")},
			}}),
			wantErr: ErrInvalidFrontmatter,
		},
		{
			name:    "Skill not found",
			source:  NewFileSystemSource(plainFS{fstest.MapFS{}}),
			wantErr: ErrSkillNotFound,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := tt.source.LoadFrontmatter(t.Context(), "test-skill")

			if !errors.Is(err, tt.wantErr) {
				t.Errorf("LoadFrontmatter(%q) expected error %v, got %v", "test-skill", tt.wantErr, err)
			}
			if diff := cmp.Diff(tt.want, got); diff != "" {
				t.Errorf("LoadFrontmatter(%q) mismatch (-want +got):\n%s", "test-skill", diff)
			}
		})
	}
}

func TestFileSystemSource_LoadInstructions(t *testing.T) {
	tests := []struct {
		name    string
		source  Source
		want    string
		wantErr error
	}{
		{
			name: "Success",
			source: NewFileSystemSource(plainFS{fstest.MapFS{
				"test-skill/SKILL.md": &fstest.MapFile{Data: []byte("---\nname: test-skill\ndescription: test\n---\nMath instructions.")},
			}}),
			want: "Math instructions.",
		},
		{
			name: "Name mismatch",
			source: NewFileSystemSource(plainFS{fstest.MapFS{
				"test-skill/SKILL.md": &fstest.MapFile{Data: []byte("---\nname: wrong-name\ndescription: test\n---\n")},
			}}),
			wantErr: ErrInvalidSkillName,
		},
		{
			name: "Invalid YAML",
			source: NewFileSystemSource(plainFS{fstest.MapFS{
				"test-skill/SKILL.md": &fstest.MapFile{Data: []byte("---[INVALID_YAML")},
			}}),
			wantErr: ErrInvalidFrontmatter,
		},
		{
			name:    "Skill not found",
			source:  NewFileSystemSource(plainFS{fstest.MapFS{}}),
			wantErr: ErrSkillNotFound,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := tt.source.LoadInstructions(t.Context(), "test-skill")

			if !errors.Is(err, tt.wantErr) {
				t.Errorf("LoadInstructions(%q) expected error %v, got %v", "test-skill", tt.wantErr, err)
			}
			if got != tt.want {
				t.Errorf("LoadInstructions(%q) = %q, want %q", "skill-name", got, tt.want)
			}
		})
	}
}

func TestFileSystemSource_LoadResource(t *testing.T) {
	tests := []struct {
		name         string
		resourcePath string
		source       Source
		wantErr      error
		want         []byte
	}{
		{
			name:         "Success Asset",
			resourcePath: "assets/image.png",
			source: NewFileSystemSource(plainFS{fstest.MapFS{
				"test-skill/SKILL.md":         &fstest.MapFile{Data: []byte("---\nname: test-skill\ndescription: test\n---\n")},
				"test-skill/assets/image.png": &fstest.MapFile{Data: []byte("image-data")},
			}}),
			want: []byte("image-data"),
		},
		{
			name:         "Success Reference",
			resourcePath: "references/doc.txt",
			source: NewFileSystemSource(plainFS{fstest.MapFS{
				"test-skill/SKILL.md":           &fstest.MapFile{Data: []byte("---\nname: test-skill\ndescription: test\n---\n")},
				"test-skill/references/doc.txt": &fstest.MapFile{Data: []byte("doc-data")},
			}}),
			want: []byte("doc-data"),
		},
		{
			name:         "Success Script",
			resourcePath: "scripts/script.py",
			source: NewFileSystemSource(plainFS{fstest.MapFS{
				"test-skill/SKILL.md":          &fstest.MapFile{Data: []byte("---\nname: test-skill\ndescription: test\n---\n")},
				"test-skill/scripts/script.py": &fstest.MapFile{Data: []byte("python-code")},
			}}),
			want: []byte("python-code"),
		},
		{
			name:         "Success Clean Path resolves traversal safely",
			resourcePath: "assets/../assets/images/../images/./image.png",
			source: NewFileSystemSource(plainFS{fstest.MapFS{
				"test-skill/SKILL.md":                &fstest.MapFile{Data: []byte("---\nname: test-skill\ndescription: test\n---\n")},
				"test-skill/assets/images/image.png": &fstest.MapFile{Data: []byte("image-data")},
			}}),
			want: []byte("image-data"),
		},
		{
			name:         "Error Skill Not Found",
			resourcePath: "assets/image.png",
			source:       NewFileSystemSource(plainFS{fstest.MapFS{}}),
			wantErr:      ErrSkillNotFound,
		},
		{
			name:         "Error Not a Skill",
			resourcePath: "assets/image.png",
			source: NewFileSystemSource(plainFS{fstest.MapFS{
				// No SKILL.md file - not a skill.
				"test-skill/assets/image.png": &fstest.MapFile{Data: []byte("image-data")},
			}}),
			wantErr: ErrSkillNotFound,
		},
		{
			name:         "Error Invalid Skill Name",
			resourcePath: "assets/image.png",
			source: NewFileSystemSource(plainFS{fstest.MapFS{
				"test-skill/SKILL.md":         &fstest.MapFile{Data: []byte("---\nname: wrong-name\ndescription: test\n---\n")},
				"test-skill/assets/image.png": &fstest.MapFile{Data: []byte("image-data")},
			}}),
			wantErr: ErrInvalidSkillName,
		},
		{
			name:         "Error Invalid Frontmatter",
			resourcePath: "assets/image.png",
			source: NewFileSystemSource(plainFS{fstest.MapFS{
				"test-skill/SKILL.md":         &fstest.MapFile{Data: []byte("Invalid YAML")},
				"test-skill/assets/image.png": &fstest.MapFile{Data: []byte("image-data")},
			}}),
			wantErr: ErrInvalidFrontmatter,
		},
		{
			name:         "Error Traversal Attempt",
			resourcePath: "../../etc/passwd",
			source: NewFileSystemSource(plainFS{fstest.MapFS{
				"test-skill/SKILL.md": &fstest.MapFile{Data: []byte("---\nname: test-skill\ndescription: test\n---\n")},
			}}),
			wantErr: ErrInvalidResourcePath,
		},
		{
			name:         "Error Unauthorized Directory",
			resourcePath: "unauthorized/file.txt",
			source: NewFileSystemSource(plainFS{fstest.MapFS{
				"test-skill/SKILL.md":              &fstest.MapFile{Data: []byte("---\nname: test-skill\ndescription: test\n---\n")},
				"test-skill/unauthorized/file.txt": &fstest.MapFile{Data: []byte("secret")},
			}}),
			wantErr: ErrInvalidResourcePath,
		},
		{
			name:         "Error File Not Found",
			resourcePath: "assets/missing.png",
			source: NewFileSystemSource(plainFS{fstest.MapFS{
				"test-skill/SKILL.md": &fstest.MapFile{Data: []byte("---\nname: test-skill\ndescription: test\n---\n")},
			}}),
			wantErr: ErrResourceNotFound,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			resource, err := tc.source.LoadResource(t.Context(), "test-skill", tc.resourcePath)

			if !errors.Is(err, tc.wantErr) {
				t.Errorf("LoadResource(%q, %q) error = %v, want %v", "test-skill", tc.resourcePath, err, tc.wantErr)
			}
			if err != nil {
				return
			}
			defer func() {
				_ = resource.Close()
			}()
			got, err := io.ReadAll(resource)
			if err != nil {
				t.Fatalf("LoadResource(%q, %q) failed to read resource: %v", "test-skill", tc.resourcePath, err)
			}
			if !bytes.Equal(got, tc.want) {
				t.Errorf("LoadResource(%q, %q) = %q, want %q", "test-skill", tc.resourcePath, got, tc.want)
			}
		})
	}
}

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

// TestFileSystemSource_LoadResource_SymlinkWithinSkillAllowed exercises the
// case raised in review: a skill sharing one file between its resource
// directories via a symlink that stays inside the skill's own directory.
// This must keep working, or WithCompletePreloadSource fails for any
// skill that ships a symlink at all, not only ones that escape.
func TestFileSystemSource_LoadResource_SymlinkWithinSkillAllowed(t *testing.T) {
	root := t.TempDir()

	skillDir := filepath.Join(root, "test-skill")
	referencesDir := filepath.Join(skillDir, "references")
	assetsDir := filepath.Join(skillDir, "assets")
	if err := os.MkdirAll(referencesDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(assetsDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(skillDir, "SKILL.md"),
		[]byte("---\nname: test-skill\ndescription: test\n---\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(referencesDir, "shared.md"),
		[]byte("shared-content"), 0o644); err != nil {
		t.Fatal(err)
	}
	// A symlink that resolves to another resource directory of the same
	// skill -- this must be served, not rejected.
	if err := os.Symlink(filepath.Join("..", "references", "shared.md"),
		filepath.Join(assetsDir, "shared.md")); err != nil {
		t.Skipf("symlinks not supported on this filesystem: %v", err)
	}

	source := NewFileSystemSource(os.DirFS(root))
	resource, err := source.LoadResource(t.Context(), "test-skill", "assets/shared.md")
	if err != nil {
		t.Fatalf("LoadResource(assets/shared.md) error = %v, want nil", err)
	}
	defer func() { _ = resource.Close() }()
	got, err := io.ReadAll(resource)
	if err != nil {
		t.Fatalf("read resource: %v", err)
	}
	if string(got) != "shared-content" {
		t.Errorf("LoadResource(assets/shared.md) = %q, want %q", got, "shared-content")
	}
}

// TestFileSystemSource_LoadResource_SymlinkAsLastComponentEscape covers a
// symlink as the final path component (assets/passwd -> /etc/passwd),
// distinct from TestFileSystemSource_LoadResource_SymlinkEscape above,
// which only exercises a symlinked directory in the middle of the path.
func TestFileSystemSource_LoadResource_SymlinkAsLastComponentEscape(t *testing.T) {
	root := t.TempDir()

	secretPath := filepath.Join(root, "secret.txt")
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
	if err := os.Symlink(secretPath, filepath.Join(assetsDir, "passwd")); err != nil {
		t.Skipf("symlinks not supported on this filesystem: %v", err)
	}

	source := NewFileSystemSource(os.DirFS(root))
	resource, err := source.LoadResource(t.Context(), "test-skill", "assets/passwd")
	if !errors.Is(err, ErrInvalidResourcePath) {
		t.Fatalf("LoadResource(assets/passwd) error = %v, want %v", err, ErrInvalidResourcePath)
	}
	if resource != nil {
		_ = resource.Close()
	}
}

// TestFileSystemSource_LoadResource_AbsoluteSymlinkTargetEscape covers a
// symlink whose stored target is itself an absolute path, which must be
// rejected outright since it cannot be reinterpreted as a path inside this
// filesystem.
func TestFileSystemSource_LoadResource_AbsoluteSymlinkTargetEscape(t *testing.T) {
	root := t.TempDir()

	secretPath := filepath.Join(root, "secret.txt")
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
	// os.Symlink(secretPath, ...) with an absolute secretPath (t.TempDir()
	// is always absolute) stores an absolute target, exercising this case
	// directly rather than through a symlinked intermediate directory.
	if err := os.Symlink(secretPath, filepath.Join(assetsDir, "link")); err != nil {
		t.Skipf("symlinks not supported on this filesystem: %v", err)
	}

	source := NewFileSystemSource(os.DirFS(root))
	resource, err := source.LoadResource(t.Context(), "test-skill", "assets/link")
	if !errors.Is(err, ErrInvalidResourcePath) {
		t.Fatalf("LoadResource(assets/link) error = %v, want %v", err, ErrInvalidResourcePath)
	}
	if resource != nil {
		_ = resource.Close()
	}
}

// TestFileSystemSource_WithCompletePreloadSource_SkillWithInternalSymlink
// reproduces the scenario from review: a source with two skills, one plain
// and one carrying a symlink that stays inside its own directory.
// WithCompletePreloadSource must still succeed; before this fix, the
// second skill's internal symlink made the whole preload -- including the
// unrelated first skill -- fail with err != nil.
func TestFileSystemSource_WithCompletePreloadSource_SkillWithInternalSymlink(t *testing.T) {
	root := t.TempDir()

	plainDir := filepath.Join(root, "plain")
	if err := os.MkdirAll(filepath.Join(plainDir, "assets"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(plainDir, "SKILL.md"),
		[]byte("---\nname: plain\ndescription: test\n---\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(plainDir, "assets", "a.txt"), []byte("a"), 0o644); err != nil {
		t.Fatal(err)
	}

	linkedDir := filepath.Join(root, "linked")
	if err := os.MkdirAll(filepath.Join(linkedDir, "references"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(linkedDir, "assets"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(linkedDir, "SKILL.md"),
		[]byte("---\nname: linked\ndescription: test\n---\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(linkedDir, "references", "shared.md"), []byte("shared"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(filepath.Join("..", "references", "shared.md"),
		filepath.Join(linkedDir, "assets", "shared.md")); err != nil {
		t.Skipf("symlinks not supported on this filesystem: %v", err)
	}

	source := NewFileSystemSource(os.DirFS(root))
	if _, _, err := WithCompletePreloadSource(t.Context(), source); err != nil {
		t.Fatalf("WithCompletePreloadSource() error = %v, want nil", err)
	}
}

// TestFileSystemSource_LoadResource_WindowsStyleSymlinkTarget uses an
// unwrapped fstest.MapFS to inject a symlink target with backslash
// separators directly, the way os.Symlink(filepath.Join(...), ...) stores
// one on Windows. This runs the same on every host OS, unlike a real
// os.Symlink call, whose separator depends on the OS running the test.
func TestFileSystemSource_LoadResource_WindowsStyleSymlinkTarget(t *testing.T) {
	tests := []struct {
		name       string
		target     string
		wantErr    error
		wantServed bool
	}{
		{
			name:       "Relative backslash target within skill is served",
			target:     `..\references\shared.md`,
			wantServed: true,
		},
		{
			name:    "Backslash target escaping the skill is rejected",
			target:  `..\..\outside\secret.txt`,
			wantErr: ErrInvalidResourcePath,
		},
		{
			name:    "Backslash drive-letter target is rejected outright",
			target:  `C:\Users\victim\.ssh\id_rsa`,
			wantErr: ErrInvalidResourcePath,
		},
		{
			name:    "Backslash UNC target is rejected outright",
			target:  `\\fileserver\share\secret.txt`,
			wantErr: ErrInvalidResourcePath,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mapFS := fstest.MapFS{
				"test-skill/SKILL.md":             &fstest.MapFile{Data: []byte("---\nname: test-skill\ndescription: test\n---\n")},
				"test-skill/references/shared.md": &fstest.MapFile{Data: []byte("shared-content")},
				"outside/secret.txt":              &fstest.MapFile{Data: []byte("top-secret")},
				"test-skill/assets/link": &fstest.MapFile{
					Mode: fs.ModeSymlink,
					Data: []byte(tt.target),
				},
			}

			source := NewFileSystemSource(mapFS)
			resource, err := source.LoadResource(t.Context(), "test-skill", "assets/link")

			if tt.wantErr != nil {
				if !errors.Is(err, tt.wantErr) {
					t.Fatalf("LoadResource(assets/link) error = %v, want %v", err, tt.wantErr)
				}
				if resource != nil {
					_ = resource.Close()
				}
				return
			}

			if err != nil {
				t.Fatalf("LoadResource(assets/link) error = %v, want nil", err)
			}
			defer func() { _ = resource.Close() }()
			got, err := io.ReadAll(resource)
			if err != nil {
				t.Fatalf("read resource: %v", err)
			}
			if string(got) != "shared-content" {
				t.Errorf("LoadResource(assets/link) = %q, want %q", got, "shared-content")
			}
		})
	}
}

func TestFileSystemSource_ListResources(t *testing.T) {
	tests := []struct {
		name       string
		searchPath string
		source     Source
		wantErr    error
		want       []string
	}{
		{
			name:       "Success Root",
			searchPath: ".",
			source: NewFileSystemSource(plainFS{fstest.MapFS{
				"test-skill/SKILL.md":               &fstest.MapFile{Data: []byte("---\nname: test-skill\ndescription: test\n---\n")},
				"test-skill/unauthorized/file.txt":  &fstest.MapFile{Data: []byte("")},
				"test-skill/references/doc.txt":     &fstest.MapFile{Data: []byte("")},
				"test-skill/references/sub/doc.txt": &fstest.MapFile{Data: []byte("")},
				"test-skill/assets/image.png":       &fstest.MapFile{Data: []byte("")},
				"test-skill/scripts/script.py":      &fstest.MapFile{Data: []byte("")},
			}}),
			want: []string{
				"references/doc.txt",
				"references/sub/doc.txt",
				"assets/image.png",
				"scripts/script.py",
			},
		},
		{
			name:       "Success Root Empty search path",
			searchPath: ".",
			source: NewFileSystemSource(plainFS{fstest.MapFS{
				"test-skill/SKILL.md":           &fstest.MapFile{Data: []byte("---\nname: test-skill\ndescription: test\n---\n")},
				"test-skill/references/doc.txt": &fstest.MapFile{Data: []byte("")},
			}}),
			want: []string{"references/doc.txt"},
		},
		{
			name:       "Success Specific Dir",
			searchPath: "assets",
			source: NewFileSystemSource(plainFS{fstest.MapFS{
				"test-skill/SKILL.md":                &fstest.MapFile{Data: []byte("---\nname: test-skill\ndescription: test\n---\n")},
				"test-skill/references/doc.txt":      &fstest.MapFile{Data: []byte("")},
				"test-skill/assets/image.png":        &fstest.MapFile{Data: []byte("")},
				"test-skill/assets/images/image.png": &fstest.MapFile{Data: []byte("")},
			}}),
			want: []string{"assets/image.png", "assets/images/image.png"},
		},
		{
			name:       "Error Skill Not Found",
			searchPath: ".",
			source:     NewFileSystemSource(plainFS{fstest.MapFS{}}),
			wantErr:    ErrSkillNotFound,
		},
		{
			name:       "Error Skill name mismatch",
			searchPath: ".",
			source: NewFileSystemSource(plainFS{fstest.MapFS{
				"test-skill/SKILL.md": &fstest.MapFile{Data: []byte("---\nname: wrong-skill\ndescription: test\n---\n")},
			}}),
			wantErr: ErrInvalidSkillName,
		},
		{
			name:       "Error Invalid frontmatter",
			searchPath: ".",
			source: NewFileSystemSource(plainFS{fstest.MapFS{
				"test-skill/SKILL.md": &fstest.MapFile{Data: []byte("Invalid YAML")},
			}}),
			wantErr: ErrInvalidFrontmatter,
		},
		{
			name:       "Error Unauthorized Directory",
			searchPath: "unauthorized",
			source: NewFileSystemSource(plainFS{fstest.MapFS{
				"test-skill/SKILL.md":              &fstest.MapFile{Data: []byte("---\nname: test-skill\ndescription: test\n---\n")},
				"test-skill/unauthorized/file.txt": &fstest.MapFile{Data: []byte("")},
			}}),
			wantErr: ErrInvalidResourcePath,
		},
		{
			name:       "Error Directory Not Found",
			searchPath: "references/missing_dir",
			source: NewFileSystemSource(plainFS{fstest.MapFS{
				"test-skill/SKILL.md": &fstest.MapFile{Data: []byte("---\nname: test-skill\ndescription: test\n---\n")},
			}}),
			wantErr: ErrResourceNotFound,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := tt.source.ListResources(t.Context(), "test-skill", tt.searchPath)

			if !errors.Is(err, tt.wantErr) {
				t.Errorf("ListResources(%q, %q) error = %v, want %v", "test-skill", tt.searchPath, err, tt.wantErr)
			}
			if diff := cmp.Diff(tt.want, got, cmpopts.SortSlices(func(a, b string) bool { return a < b })); diff != "" {
				t.Errorf("ListResources(%q, %q) diff (-want +got):\n%s", "test-skill", tt.searchPath, diff)
			}
		})
	}
}

// plainFS is a minimal implementation of fs.FS that deliberately hides optional
// interface extensions like fs.ReadDirFS or fs.StatFS.
//
// We test exclusively against this minimal wrapper to ensure our Source
// implementation strictly relies only on the baseline fs.FS contract (Open)
// and doesn't accidentally depend on optional filesystem features.
type plainFS struct {
	fs fs.FS
}

func (p plainFS) Open(name string) (fs.File, error) {
	return p.fs.Open(name)
}
