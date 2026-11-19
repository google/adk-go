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

package skillregistry

import (
	"archive/zip"
	"bytes"
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	agentregistry "google.golang.org/api/agentregistry/v1alpha"
	"google.golang.org/api/googleapi"

	"google.golang.org/adk/v2/tool/skilltoolset/skill"
)

const (
	nameCymbal = "projects/p/locations/global/skills/cymbal"
	revCymbal  = "rev-cymbal"
)

// zipEntry is one file placed into the archive served by fakeClient.GetZip.
type zipEntry struct {
	name    string
	content []byte
}

// buildZip returns a *zip.Reader over entries, preserving their order.
func buildZip(t *testing.T, entries []zipEntry) *zip.Reader {
	t.Helper()
	b := zipBytes(t, entries)
	r, err := zip.NewReader(bytes.NewReader(b), int64(len(b)))
	if err != nil {
		t.Fatalf("opening zip: %v", err)
	}
	return r
}

// fakeClient is an in-memory [Client]. It serves the source's three access
// paths — ListFrontmatters, GetSkill (resolving the revision) and GetZip
// (keyed by revision) — and records nothing beyond its configured data.
type fakeClient struct {
	t *testing.T

	frontmatters []*agentregistry.Frontmatter    // ListFrontmatters result
	listFMErr    error                           // ListFrontmatters error
	byName       map[string]*agentregistry.Skill // GetSkill result, by resource name
	getErr       map[string]error                // GetSkill error, by resource name
	zips         map[string][]zipEntry           // GetZip archives, by revision
	zipErr       map[string]error                // GetZip error, by revision
}

var _ Client = (*fakeClient)(nil)

func (c *fakeClient) ListFrontmatters() ([]*agentregistry.Frontmatter, error) {
	if c.listFMErr != nil {
		return nil, c.listFMErr
	}
	return c.frontmatters, nil
}

func (c *fakeClient) GetSkill(name string) (*agentregistry.Skill, error) {
	if err := c.getErr[name]; err != nil {
		return nil, err
	}
	s, ok := c.byName[name]
	if !ok {
		return nil, errors.New("fakeClient: no skill for the given name")
	}
	return s, nil
}

func (c *fakeClient) GetZip(rev string) (*zip.Reader, error) {
	if err := c.zipErr[rev]; err != nil {
		return nil, err
	}
	entries, ok := c.zips[rev]
	if !ok {
		return nil, errors.New("fakeClient: no zip for the given revision")
	}
	return buildZip(c.t, entries), nil
}

// ResourceID mirrors the real client: it wraps a short name as a full resource
// name, so the test can assert the source reports the resource name rather than
// whatever the frontmatter carried.
func (c *fakeClient) ResourceID(name string) string {
	return "projects/p/locations/global/skills/" + name
}

// The methods below are not exercised by the source paths.
func (c *fakeClient) ListSkills() ([]*agentregistry.Skill, error) { return nil, nil }

func (c *fakeClient) GetRevision(string) (*agentregistry.SkillRevision, error) { return nil, nil }

func (c *fakeClient) FindFrontmatters(string) ([]*agentregistry.Frontmatter, error) {
	return nil, nil
}

func (c *fakeClient) ParseSkillName(string) (projectID, location, skillName string, resErr error) {
	return "", "", "", nil
}

// newCymbalClient serves a single skill with a SKILL.md and a few resources.
func newCymbalClient(t *testing.T) *fakeClient {
	t.Helper()
	md, err := skill.Build(&skill.Frontmatter{Name: "cymbal", Description: "Cymbal."}, "# Cymbal\n\nRun it.\n")
	if err != nil {
		t.Fatalf("skill.Build() error = %v", err)
	}
	return &fakeClient{
		t: t,
		byName: map[string]*agentregistry.Skill{
			nameCymbal: {
				Name:            nameCymbal,
				DefaultRevision: revCymbal,
				Frontmatter: &agentregistry.Frontmatter{
					Name:          "fm-name",
					Description:   "Cymbal.",
					License:       "Apache-2.0",
					Compatibility: "needs bash",
					Metadata:      map[string]string{"version": "1.2"},
				},
			},
		},
		zips: map[string][]zipEntry{
			revCymbal: {
				{name: "SKILL.md", content: md},
				{name: "assets/logo.svg", content: []byte("<svg/>")},
				{name: "assets/nested/pic.png", content: []byte("png")},
				{name: "scripts/run.sh", content: []byte("echo hi")},
			},
		},
		getErr: map[string]error{},
		zipErr: map[string]error{},
	}
}

func newSource(t *testing.T, c Client) *SkillRegistrySource {
	t.Helper()
	src, err := NewSkillRegistrySource(t.Context(), SkillRegistrySourceConfig{Client: c})
	if err != nil {
		t.Fatalf("NewSkillRegistrySource() error = %v", err)
	}
	return src
}

func TestSkillRegistrySource_ListFrontmatters(t *testing.T) {
	c := &fakeClient{
		t: t,
		frontmatters: []*agentregistry.Frontmatter{
			{Name: "alpha", Description: "Alpha.", License: "Apache-2.0"},
			nil, // a nil frontmatter from the client is skipped, not returned
			{Name: "beta", Description: "Beta."},
		},
	}
	src := newSource(t, c)

	got, err := src.ListFrontmatters(t.Context())
	if err != nil {
		t.Fatalf("ListFrontmatters() error = %v", err)
	}
	// The name is the full resource name (via ResourceID), not the raw field.
	want := []*skill.Frontmatter{
		{Name: "projects/p/locations/global/skills/alpha", Description: "Alpha.", License: "Apache-2.0"},
		{Name: "projects/p/locations/global/skills/beta", Description: "Beta."},
	}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("ListFrontmatters() diff (-want +got):\n%s", diff)
	}
}

func TestSkillRegistrySource_ListFrontmatters_Error(t *testing.T) {
	boom := errors.New("registry down")
	src := newSource(t, &fakeClient{t: t, listFMErr: boom})
	if _, err := src.ListFrontmatters(t.Context()); !errors.Is(err, boom) {
		t.Fatalf("ListFrontmatters() error = %v, want errors.Is %v", err, boom)
	}
}

func TestSkillRegistrySource_LoadFrontmatter(t *testing.T) {
	t.Run("returns the registered metadata", func(t *testing.T) {
		src := newSource(t, newCymbalClient(t))
		got, err := src.LoadFrontmatter(t.Context(), nameCymbal)
		if err != nil {
			t.Fatalf("LoadFrontmatter() error = %v", err)
		}
		want := &skill.Frontmatter{
			// The reported name is the registry frontmatter name wrapped by
			// ResourceID into a full resource name.
			Name:          "projects/p/locations/global/skills/fm-name",
			Description:   "Cymbal.",
			License:       "Apache-2.0",
			Compatibility: "needs bash",
			Metadata:      map[string]string{"version": "1.2"},
		}
		if diff := cmp.Diff(want, got); diff != "" {
			t.Errorf("LoadFrontmatter() diff (-want +got):\n%s", diff)
		}
	})

	t.Run("GetSkill error is propagated", func(t *testing.T) {
		boom := errors.New("boom")
		c := newCymbalClient(t)
		c.getErr[nameCymbal] = boom
		src := newSource(t, c)
		if _, err := src.LoadFrontmatter(t.Context(), nameCymbal); !errors.Is(err, boom) {
			t.Fatalf("LoadFrontmatter() error = %v, want errors.Is %v", err, boom)
		}
	})

	t.Run("a skill without frontmatter is an error", func(t *testing.T) {
		c := newCymbalClient(t)
		c.byName[nameCymbal].Frontmatter = nil
		src := newSource(t, c)
		if _, err := src.LoadFrontmatter(t.Context(), nameCymbal); err == nil {
			t.Fatal("LoadFrontmatter() error = nil, want an error for a skill with no frontmatter")
		}
	})
}

func TestSkillRegistrySource_LoadInstructions(t *testing.T) {
	t.Run("returns the SKILL.md body", func(t *testing.T) {
		src := newSource(t, newCymbalClient(t))
		got, err := src.LoadInstructions(t.Context(), nameCymbal)
		if err != nil {
			t.Fatalf("LoadInstructions() error = %v", err)
		}
		if want := "# Cymbal\n\nRun it.\n"; got != want {
			t.Errorf("LoadInstructions() = %q, want %q", got, want)
		}
	})

	t.Run("no SKILL.md returns ErrResourceNotFound", func(t *testing.T) {
		c := newCymbalClient(t)
		c.zips[revCymbal] = []zipEntry{{name: "assets/logo.svg", content: []byte("<svg/>")}}
		src := newSource(t, c)
		if _, err := src.LoadInstructions(t.Context(), nameCymbal); !errors.Is(err, skill.ErrResourceNotFound) {
			t.Fatalf("LoadInstructions() error = %v, want errors.Is %v", err, skill.ErrResourceNotFound)
		}
	})

	t.Run("invalid frontmatter in SKILL.md is a parse error", func(t *testing.T) {
		c := newCymbalClient(t)
		c.zips[revCymbal] = []zipEntry{{name: "SKILL.md", content: []byte("no frontmatter here")}}
		src := newSource(t, c)
		_, err := src.LoadInstructions(t.Context(), nameCymbal)
		if err == nil {
			t.Fatal("LoadInstructions() error = nil, want a parse error")
		}
		if !strings.Contains(err.Error(), "parse") {
			t.Errorf("LoadInstructions() error = %v, want it to mention a parse failure", err)
		}
	})

	t.Run("download failure is propagated", func(t *testing.T) {
		boom := errors.New("registry down")
		c := newCymbalClient(t)
		c.zipErr[revCymbal] = boom
		src := newSource(t, c)
		if _, err := src.LoadInstructions(t.Context(), nameCymbal); !errors.Is(err, boom) {
			t.Fatalf("LoadInstructions() error = %v, want errors.Is %v", err, boom)
		}
	})
}

func TestSkillRegistrySource_ListResources(t *testing.T) {
	src := newSource(t, newCymbalClient(t))

	tests := []struct {
		name    string
		subpath string
		want    []string
		wantErr error
	}{
		{
			name:    "root lists every file including SKILL.md",
			subpath: ".",
			want:    []string{"SKILL.md", "assets/logo.svg", "assets/nested/pic.png", "scripts/run.sh"},
		},
		{
			name:    "empty subpath also lists root",
			subpath: "",
			want:    []string{"SKILL.md", "assets/logo.svg", "assets/nested/pic.png", "scripts/run.sh"},
		},
		{
			name:    "directory lists its subtree",
			subpath: "assets",
			want:    []string{"assets/logo.svg", "assets/nested/pic.png"},
		},
		{
			name:    "trailing slash is accepted",
			subpath: "assets/",
			want:    []string{"assets/logo.svg", "assets/nested/pic.png"},
		},
		{
			name:    "single file matches itself",
			subpath: "scripts/run.sh",
			want:    []string{"scripts/run.sh"},
		},
		{
			// "script" is not a path segment of "scripts/..." and must not match.
			name:    "partial segment does not match",
			subpath: "script",
			want:    []string{},
		},
		{name: "traversal is rejected", subpath: "../other", wantErr: skill.ErrInvalidResourcePath},
		{name: "absolute is rejected", subpath: "/assets", wantErr: skill.ErrInvalidResourcePath},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := src.ListResources(t.Context(), nameCymbal, tt.subpath)
			if tt.wantErr != nil {
				if !errors.Is(err, tt.wantErr) {
					t.Fatalf("ListResources() error = %v, want errors.Is %v", err, tt.wantErr)
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

func TestSkillRegistrySource_ListResources_DropsZipSlip(t *testing.T) {
	c := newCymbalClient(t)
	c.zips[revCymbal] = []zipEntry{
		{name: "assets/", content: nil},                    // directory entry
		{name: "./assets/ok.txt", content: []byte("ok")},   // survives, cleaned to assets/ok.txt
		{name: "../escape.sh", content: []byte("pwned")},   // escapes
		{name: "/etc/passwd", content: []byte("absolute")}, // absolute
	}
	src := newSource(t, c)
	got, err := src.ListResources(t.Context(), nameCymbal, ".")
	if err != nil {
		t.Fatalf("ListResources() error = %v", err)
	}
	if diff := cmp.Diff([]string{"assets/ok.txt"}, got); diff != "" {
		t.Errorf("ListResources() diff (-want +got):\n%s", diff)
	}

	// The cleaned entry is reachable by its cleaned path via LoadResource.
	rc, err := src.LoadResource(t.Context(), nameCymbal, "assets/ok.txt")
	if err != nil {
		t.Fatalf("LoadResource() error = %v", err)
	}
	_ = rc.Close()
}

func TestSkillRegistrySource_LoadResource(t *testing.T) {
	src := newSource(t, newCymbalClient(t))

	t.Run("returns the file content", func(t *testing.T) {
		rc, err := src.LoadResource(t.Context(), nameCymbal, "scripts/run.sh")
		if err != nil {
			t.Fatalf("LoadResource() error = %v", err)
		}
		defer func() { _ = rc.Close() }()
		got, err := io.ReadAll(rc)
		if err != nil {
			t.Fatalf("reading resource: %v", err)
		}
		if string(got) != "echo hi" {
			t.Errorf("content = %q, want %q", got, "echo hi")
		}
	})

	t.Run("interior dot resolves to the same file", func(t *testing.T) {
		rc, err := src.LoadResource(t.Context(), nameCymbal, "assets/./logo.svg")
		if err != nil {
			t.Fatalf("LoadResource() error = %v", err)
		}
		_ = rc.Close()
	})

	t.Run("missing file returns ErrResourceNotFound", func(t *testing.T) {
		if _, err := src.LoadResource(t.Context(), nameCymbal, "scripts/absent.sh"); !errors.Is(err, skill.ErrResourceNotFound) {
			t.Fatalf("LoadResource() error = %v, want errors.Is %v", err, skill.ErrResourceNotFound)
		}
	})

	t.Run("traversal and absolute paths are rejected", func(t *testing.T) {
		for _, p := range []string{"../escape.sh", "scripts/../../escape.sh", "/etc/passwd", ".."} {
			if _, err := src.LoadResource(t.Context(), nameCymbal, p); !errors.Is(err, skill.ErrInvalidResourcePath) {
				t.Errorf("LoadResource(%q) error = %v, want errors.Is %v", p, err, skill.ErrInvalidResourcePath)
			}
		}
	})
}

// A GetSkill failure must surface from every method that resolves the revision,
// not just LoadFrontmatter.
func TestSkillRegistrySource_GetSkillErrorPropagates(t *testing.T) {
	boom := errors.New("registry down")
	newErrSrc := func(t *testing.T) *SkillRegistrySource {
		c := newCymbalClient(t)
		c.getErr[nameCymbal] = boom
		return newSource(t, c)
	}

	t.Run("LoadInstructions", func(t *testing.T) {
		if _, err := newErrSrc(t).LoadInstructions(t.Context(), nameCymbal); !errors.Is(err, boom) {
			t.Fatalf("LoadInstructions() error = %v, want errors.Is %v", err, boom)
		}
	})
	t.Run("ListResources", func(t *testing.T) {
		if _, err := newErrSrc(t).ListResources(t.Context(), nameCymbal, "."); !errors.Is(err, boom) {
			t.Fatalf("ListResources() error = %v, want errors.Is %v", err, boom)
		}
	})
	t.Run("LoadResource", func(t *testing.T) {
		if _, err := newErrSrc(t).LoadResource(t.Context(), nameCymbal, "scripts/run.sh"); !errors.Is(err, boom) {
			t.Fatalf("LoadResource() error = %v, want errors.Is %v", err, boom)
		}
	})
}

// A registry not-found (HTTP 404) must surface as skill.ErrSkillNotFound so a
// merged lookup falls through to the next source. It must reach every method
// that resolves the skill, not just LoadFrontmatter.
func TestSkillRegistrySource_NotFoundMapsToErrSkillNotFound(t *testing.T) {
	notFound := &googleapi.Error{Code: http.StatusNotFound, Message: "skill not found"}
	newSrc := func(t *testing.T) *SkillRegistrySource {
		c := newCymbalClient(t)
		c.getErr[nameCymbal] = notFound
		return newSource(t, c)
	}

	t.Run("LoadFrontmatter", func(t *testing.T) {
		if _, err := newSrc(t).LoadFrontmatter(t.Context(), nameCymbal); !errors.Is(err, skill.ErrSkillNotFound) {
			t.Fatalf("LoadFrontmatter() error = %v, want errors.Is %v", err, skill.ErrSkillNotFound)
		}
	})
	t.Run("LoadInstructions", func(t *testing.T) {
		if _, err := newSrc(t).LoadInstructions(t.Context(), nameCymbal); !errors.Is(err, skill.ErrSkillNotFound) {
			t.Fatalf("LoadInstructions() error = %v, want errors.Is %v", err, skill.ErrSkillNotFound)
		}
	})
	t.Run("ListResources", func(t *testing.T) {
		if _, err := newSrc(t).ListResources(t.Context(), nameCymbal, "."); !errors.Is(err, skill.ErrSkillNotFound) {
			t.Fatalf("ListResources() error = %v, want errors.Is %v", err, skill.ErrSkillNotFound)
		}
	})
	t.Run("LoadResource", func(t *testing.T) {
		if _, err := newSrc(t).LoadResource(t.Context(), nameCymbal, "scripts/run.sh"); !errors.Is(err, skill.ErrSkillNotFound) {
			t.Fatalf("LoadResource() error = %v, want errors.Is %v", err, skill.ErrSkillNotFound)
		}
	})
}

// A non-404 error is a real failure, not a missing skill, so it must not be
// mapped to skill.ErrSkillNotFound.
func TestSkillRegistrySource_NonNotFoundIsNotErrSkillNotFound(t *testing.T) {
	// A 403 is a googleapi.Error too, to prove the mapping keys on the code.
	forbidden := &googleapi.Error{Code: http.StatusForbidden, Message: "permission denied"}
	c := newCymbalClient(t)
	c.getErr[nameCymbal] = forbidden
	src := newSource(t, c)

	_, err := src.LoadFrontmatter(t.Context(), nameCymbal)
	if errors.Is(err, skill.ErrSkillNotFound) {
		t.Fatalf("LoadFrontmatter() error = %v, must not be errors.Is %v", err, skill.ErrSkillNotFound)
	}
	if !errors.Is(err, forbidden) {
		t.Fatalf("LoadFrontmatter() error = %v, want errors.Is %v", err, forbidden)
	}
}

// A skill whose DefaultRevision is empty must not trigger a GetZip("") call.
func TestSkillRegistrySource_EmptyRevision(t *testing.T) {
	c := newCymbalClient(t)
	c.byName[nameCymbal].DefaultRevision = ""
	src := newSource(t, c)
	if _, err := src.LoadInstructions(t.Context(), nameCymbal); err == nil {
		t.Fatal("LoadInstructions() error = nil, want an error for a skill with no default revision")
	}
}
