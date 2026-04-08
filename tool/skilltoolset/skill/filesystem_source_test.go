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
