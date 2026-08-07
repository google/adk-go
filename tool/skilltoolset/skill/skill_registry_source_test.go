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
	"archive/zip"
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"maps"
	"net/http"
	"slices"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/api/option"
)

const testSkillMD = `---
name: cymbal_skill
description: A skill for managing Cymbal projects.
metadata:
  version: "1.2"
---
# Cymbal

Run the deploy script.
`

// zipSkill builds a skill payload from a map of path to file content.
func zipSkill(t *testing.T, files map[string]string) []byte {
	t.Helper()

	var buf bytes.Buffer
	w := zip.NewWriter(&buf)
	// Write in a stable order so archive order never depends on map iteration.
	for _, name := range slices.Sorted(maps.Keys(files)) {
		f, err := w.Create(name)
		if err != nil {
			t.Fatalf("creating zip entry %q: %v", name, err)
		}
		if _, err := io.WriteString(f, files[name]); err != nil {
			t.Fatalf("writing zip entry %q: %v", name, err)
		}
	}
	if err := w.Close(); err != nil {
		t.Fatalf("closing zip: %v", err)
	}
	return buf.Bytes()
}

// testSkillFiles is the payload of the "cymbal-skill" fixture.
func testSkillFiles() map[string]string {
	return map[string]string{
		"SKILL.md":              testSkillMD,
		"scripts/deploy.sh":     "#!/bin/sh\necho deploy\n",
		"scripts/lib/util.sh":   "true\n",
		"references/api.md":     "# API\n",
		"assets/logo.svg":       "<svg/>",
		"notes.txt":             "not a resource",
		"../escape/evil.sh":     "pwned",
		"assets/nested/pic.png": "png",
	}
}

// newTestSource starts a fake Skill Registry and returns a Source bound to it.
// getSkill serves GET on a single skill; list serves the collection.
func newTestSource(t *testing.T, list string, getSkill func(skillID string) (int, string)) Source {
	t.Helper()

	c, _ := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		urlPath := r.URL.Path
		if strings.HasSuffix(urlPath, "/skills") {
			_, _ = w.Write([]byte(list))
			return
		}
		status, body := getSkill(urlPath[strings.LastIndex(urlPath, "/")+1:])
		w.WriteHeader(status)
		_, _ = w.Write([]byte(body))
	})

	src, err := NewSkillRegistrySource(t.Context(), SkillRegistrySourceConfig{Client: c})
	if err != nil {
		t.Fatalf("NewSkillRegistrySource() error = %v", err)
	}
	return src
}

// skillJSON renders a skill resource carrying the standard test payload.
func skillJSON(t *testing.T, id string, files map[string]string) string {
	t.Helper()

	body, err := json.Marshal(map[string]any{
		"name":             "projects/1234/locations/us-central1/skills/" + id,
		"displayName":      "cymbal_skill",
		"description":      "A skill for managing Cymbal projects.",
		"license":          "Apache-2.0",
		"compatibility":    "needs bash",
		"state":            SkillStateActive,
		"zippedFilesystem": zipSkill(t, files),
	})
	if err != nil {
		t.Fatalf("marshaling skill: %v", err)
	}
	return string(body)
}

func TestNewSkillRegistrySource_Config(t *testing.T) {
	if _, err := NewSkillRegistrySource(t.Context(), SkillRegistrySourceConfig{Location: "us-central1"},
		option.WithHTTPClient(http.DefaultClient)); err == nil {
		t.Error("NewSkillRegistrySource() with no project = nil error, want error")
	}
}

func TestSkillRegistrySource_ListFrontmatters(t *testing.T) {
	// Only ACTIVE skills, and skills whose state the API left unset, are usable.
	list := `{"skills":[
	  {"name":"projects/1234/locations/us-central1/skills/zebra-skill","description":"Zebras.","state":"ACTIVE"},
	  {"name":"projects/1234/locations/us-central1/skills/cymbal-skill","displayName":"cymbal_skill","description":"Cymbal.","license":"Apache-2.0","compatibility":"needs bash","state":"ACTIVE"},
	  {"name":"projects/1234/locations/us-central1/skills/half-built","description":"Nope.","state":"CREATING"},
	  {"name":"projects/1234/locations/us-central1/skills/broken","description":"Nope.","state":"FAILED"},
	  {"name":"projects/1234/locations/us-central1/skills/stateless","description":"Fine."}
	]}`
	src := newTestSource(t, list, nil)

	got, err := src.ListFrontmatters(t.Context())
	if err != nil {
		t.Fatalf("ListFrontmatters() error = %v", err)
	}

	// Names come from the skill ID, not the display name, and are sorted.
	want := []*Frontmatter{
		{Name: "cymbal-skill", Description: "Cymbal.", License: "Apache-2.0", Compatibility: "needs bash"},
		{Name: "stateless", Description: "Fine."},
		{Name: "zebra-skill", Description: "Zebras."},
	}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("ListFrontmatters() diff (-want +got):\n%s", diff)
	}
}

func TestSkillRegistrySource_LoadFrontmatter(t *testing.T) {
	src := newTestSource(t, "", func(string) (int, string) {
		return http.StatusOK, skillJSON(t, "cymbal-skill", testSkillFiles())
	})

	got, err := src.LoadFrontmatter(t.Context(), "cymbal-skill")
	if err != nil {
		t.Fatalf("LoadFrontmatter() error = %v", err)
	}

	// The registered metadata wins over the SKILL.md frontmatter, so the name
	// is the skill ID rather than the "cymbal_skill" in the file.
	want := &Frontmatter{
		Name:          "cymbal-skill",
		Description:   "A skill for managing Cymbal projects.",
		License:       "Apache-2.0",
		Compatibility: "needs bash",
	}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("LoadFrontmatter() diff (-want +got):\n%s", diff)
	}
}

func TestSkillRegistrySource_LoadInstructions(t *testing.T) {
	tests := []struct {
		name     string
		skillMD  string
		want     string
		wantErr  error
		skipFile bool
	}{
		{
			name:    "strips frontmatter",
			skillMD: testSkillMD,
			want:    "# Cymbal\n\nRun the deploy script.\n",
		},
		{
			name:    "windows line endings",
			skillMD: "---\r\nname: x\r\n---\r\n# Body\r\n",
			want:    "# Body\r\n",
		},
		{
			name:    "empty body",
			skillMD: "---\nname: x\n---\n",
			want:    "",
		},
		{
			// Frontmatter this package cannot model must not hide the body:
			// unlike ParseBytes, the source never decodes the YAML.
			name:    "unknown frontmatter fields",
			skillMD: "---\nname: cymbal_skill\nunknown-field: [1, 2]\n---\nBody\n",
			want:    "Body\n",
		},
		{
			name:    "no opening separator",
			skillMD: "# Just markdown\n",
			wantErr: ErrInvalidFrontmatter,
		},
		{
			name:    "no closing separator",
			skillMD: "---\nname: x\n",
			wantErr: ErrInvalidFrontmatter,
		},
		{
			name:     "no SKILL.md in payload",
			skipFile: true,
			wantErr:  ErrSkillNotFound,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			files := map[string]string{"assets/logo.svg": "<svg/>"}
			if !tt.skipFile {
				files["SKILL.md"] = tt.skillMD
			}
			src := newTestSource(t, "", func(string) (int, string) {
				return http.StatusOK, skillJSON(t, "cymbal-skill", files)
			})

			got, err := src.LoadInstructions(t.Context(), "cymbal-skill")
			if tt.wantErr != nil {
				if !errors.Is(err, tt.wantErr) {
					t.Fatalf("LoadInstructions() error = %v, want %v", err, tt.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("LoadInstructions() error = %v", err)
			}
			if got != tt.want {
				t.Errorf("LoadInstructions() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestSkillRegistrySource_ListResources(t *testing.T) {
	src := newTestSource(t, "", func(string) (int, string) {
		return http.StatusOK, skillJSON(t, "cymbal-skill", testSkillFiles())
	})

	tests := []struct {
		name    string
		subpath string
		want    []string
		wantErr error
	}{
		{
			// SKILL.md, the stray root file and the traversal entry are all
			// excluded; only the three resource directories are listed.
			name:    "root",
			subpath: ".",
			want: []string{
				"assets/logo.svg",
				"assets/nested/pic.png",
				"references/api.md",
				"scripts/deploy.sh",
				"scripts/lib/util.sh",
			},
		},
		{
			name:    "empty subpath lists root",
			subpath: "",
			want: []string{
				"assets/logo.svg",
				"assets/nested/pic.png",
				"references/api.md",
				"scripts/deploy.sh",
				"scripts/lib/util.sh",
			},
		},
		{
			name:    "top level resource directory",
			subpath: "scripts",
			want:    []string{"scripts/deploy.sh", "scripts/lib/util.sh"},
		},
		{
			name:    "nested directory",
			subpath: "assets/nested",
			want:    []string{"assets/nested/pic.png"},
		},
		{
			name:    "single file",
			subpath: "references/api.md",
			want:    []string{"references/api.md"},
		},
		{
			name:    "unknown directory",
			subpath: "scripts/missing",
			wantErr: ErrResourceNotFound,
		},
		{
			name:    "outside the resource directories",
			subpath: "notes.txt",
			wantErr: ErrInvalidResourcePath,
		},
		{
			name:    "traversal",
			subpath: "../escape",
			wantErr: ErrInvalidResourcePath,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := src.ListResources(t.Context(), "cymbal-skill", tt.subpath)
			if tt.wantErr != nil {
				if !errors.Is(err, tt.wantErr) {
					t.Fatalf("ListResources() error = %v, want %v", err, tt.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("ListResources() error = %v", err)
			}
			if diff := cmp.Diff(tt.want, got); diff != "" {
				t.Errorf("ListResources() diff (-want +got):\n%s", diff)
			}
		})
	}
}

func TestSkillRegistrySource_LoadResource(t *testing.T) {
	src := newTestSource(t, "", func(string) (int, string) {
		return http.StatusOK, skillJSON(t, "cymbal-skill", testSkillFiles())
	})

	tests := []struct {
		name         string
		resourcePath string
		want         string
		wantErr      error
	}{
		{name: "script", resourcePath: "scripts/deploy.sh", want: "#!/bin/sh\necho deploy\n"},
		{name: "nested asset", resourcePath: "assets/nested/pic.png", want: "png"},
		{name: "cleaned path", resourcePath: "references/../references/api.md", want: "# API\n"},
		{name: "missing", resourcePath: "scripts/absent.sh", wantErr: ErrResourceNotFound},
		{name: "skill file", resourcePath: "SKILL.md", wantErr: ErrInvalidResourcePath},
		{name: "root file", resourcePath: "notes.txt", wantErr: ErrInvalidResourcePath},
		{name: "directory", resourcePath: "scripts", wantErr: ErrInvalidResourcePath},
		{name: "traversal", resourcePath: "../escape/evil.sh", wantErr: ErrInvalidResourcePath},
		{name: "traversal through a resource dir", resourcePath: "scripts/../../escape/evil.sh", wantErr: ErrInvalidResourcePath},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rc, err := src.LoadResource(t.Context(), "cymbal-skill", tt.resourcePath)
			if tt.wantErr != nil {
				if !errors.Is(err, tt.wantErr) {
					t.Fatalf("LoadResource() error = %v, want %v", err, tt.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("LoadResource() error = %v", err)
			}
			defer func() { _ = rc.Close() }()

			got, err := io.ReadAll(rc)
			if err != nil {
				t.Fatalf("reading resource: %v", err)
			}
			if string(got) != tt.want {
				t.Errorf("LoadResource() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestNewSkillArchive(t *testing.T) {
	// A skill payload is untrusted input: entries that would escape the skill
	// root are dropped, and the survivors are indexed under cleaned paths.
	files := map[string]string{
		"SKILL.md":            "body",
		"./scripts/deploy.sh": "deploy",
		"assets/nested/a.png": "a",
		"../escape.sh":        "escape",
		"..":                  "dots",
		"/etc/passwd":         "absolute",
	}
	archive, err := newSkillArchive(zipSkill(t, files))
	if err != nil {
		t.Fatalf("newSkillArchive() error = %v", err)
	}

	// "../escape.sh", ".." and "/etc/passwd" are all gone; "./scripts/deploy.sh"
	// survives under its cleaned name.
	want := []string{"SKILL.md", "assets/nested/a.png", "scripts/deploy.sh"}
	if diff := cmp.Diff(want, archive.paths); diff != "" {
		t.Errorf("archive paths diff (-want +got):\n%s", diff)
	}
	if len(archive.files) != len(want) {
		t.Errorf("archive indexed %d files, want %d", len(archive.files), len(want))
	}

	t.Run("empty payload", func(t *testing.T) {
		if _, err := newSkillArchive(nil); !errors.Is(err, ErrSkillNotFound) {
			t.Errorf("newSkillArchive(nil) error = %v, want ErrSkillNotFound", err)
		}
	})
}

func TestSkillRegistrySource_Errors(t *testing.T) {
	t.Run("missing skill", func(t *testing.T) {
		src := newTestSource(t, "", func(string) (int, string) {
			return http.StatusNotFound, `{"error":{"status":"NOT_FOUND","message":"nope"}}`
		})
		if _, err := src.LoadFrontmatter(t.Context(), "gone"); !errors.Is(err, ErrSkillNotFound) {
			t.Errorf("LoadFrontmatter() error = %v, want ErrSkillNotFound", err)
		}
		if _, err := src.LoadInstructions(t.Context(), "gone"); !errors.Is(err, ErrSkillNotFound) {
			t.Errorf("LoadInstructions() error = %v, want ErrSkillNotFound", err)
		}
	})

	t.Run("empty payload", func(t *testing.T) {
		src := newTestSource(t, "", func(string) (int, string) {
			return http.StatusOK, `{"name":"projects/1234/locations/us-central1/skills/hollow"}`
		})
		if _, err := src.LoadInstructions(t.Context(), "hollow"); !errors.Is(err, ErrSkillNotFound) {
			t.Errorf("LoadInstructions() error = %v, want ErrSkillNotFound", err)
		}
	})

	t.Run("corrupt payload", func(t *testing.T) {
		body, err := json.Marshal(map[string]any{
			"name":             "projects/1234/locations/us-central1/skills/corrupt",
			"zippedFilesystem": []byte("this is not a zip archive"),
		})
		if err != nil {
			t.Fatalf("marshaling skill: %v", err)
		}
		src := newTestSource(t, "", func(string) (int, string) { return http.StatusOK, string(body) })

		_, err = src.LoadInstructions(t.Context(), "corrupt")
		if err == nil || !strings.Contains(err.Error(), "zipped filesystem") {
			t.Errorf("LoadInstructions() error = %v, want it to mention the zipped filesystem", err)
		}
	})

	t.Run("list failure", func(t *testing.T) {
		src := newTestSource(t, "", nil)
		// The list handler returns the empty body configured above, which is
		// not valid JSON, so the decode must surface as an error.
		if _, err := src.ListFrontmatters(t.Context()); err == nil {
			t.Error("ListFrontmatters() = nil error, want a decoding error")
		}
	})
}

// TestSkillRegistrySource_WithCompletePreload checks the source satisfies the
// contract the preload decorator relies on: ListResources(name, ".") must list
// every resource of the skill.
func TestSkillRegistrySource_WithCompletePreload(t *testing.T) {
	list := `{"skills":[{"name":"projects/1234/locations/us-central1/skills/cymbal-skill","description":"Cymbal.","state":"ACTIVE"}]}`
	src := newTestSource(t, list, func(string) (int, string) {
		return http.StatusOK, skillJSON(t, "cymbal-skill", testSkillFiles())
	})

	cached, _, err := WithCompletePreloadSource(t.Context(), src)
	if err != nil {
		t.Fatalf("WithCompletePreloadSource() error = %v", err)
	}

	instructions, err := cached.LoadInstructions(t.Context(), "cymbal-skill")
	if err != nil {
		t.Fatalf("LoadInstructions() error = %v", err)
	}
	if want := "# Cymbal\n\nRun the deploy script.\n"; instructions != want {
		t.Errorf("LoadInstructions() = %q, want %q", instructions, want)
	}

	resources, err := cached.ListResources(t.Context(), "cymbal-skill", ".")
	if err != nil {
		t.Fatalf("ListResources() error = %v", err)
	}
	want := []string{
		"assets/logo.svg",
		"assets/nested/pic.png",
		"references/api.md",
		"scripts/deploy.sh",
		"scripts/lib/util.sh",
	}
	if diff := cmp.Diff(want, resources); diff != "" {
		t.Errorf("ListResources() diff (-want +got):\n%s", diff)
	}
}
