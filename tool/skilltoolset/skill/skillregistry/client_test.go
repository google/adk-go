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
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"slices"
	"testing"

	"google.golang.org/api/option"
)

func TestResourceID(t *testing.T) {
	c := &client{parent: "projects/p/locations/global"}
	if got, want := c.ResourceID("my-skill"), "projects/p/locations/global/skills/my-skill"; got != want {
		t.Errorf("ResourceID() = %q, want %q", got, want)
	}
}

func TestIsUsableSkill(t *testing.T) {
	for _, tt := range []struct {
		state string
		want  bool
	}{
		{"STATE_ACTIVE", true},
		{"STATE_DEPRECATED", true},
		{"STATE_CREATING", false},
		{"STATE_FAILED", false},
		{"", false},
	} {
		if got := isUsableSkill(tt.state); got != tt.want {
			t.Errorf("isUsableSkill(%q) = %v, want %v", tt.state, got, tt.want)
		}
	}
}

func TestParseSkillName(t *testing.T) {
	c := &client{}

	t.Run("valid", func(t *testing.T) {
		p, l, n, err := c.ParseSkillName("projects/my-proj/locations/global/skills/cloud.google.com-report")
		if err != nil {
			t.Fatalf("ParseSkillName() error = %v", err)
		}
		if p != "my-proj" || l != "global" || n != "cloud.google.com-report" {
			t.Errorf("ParseSkillName() = (%q, %q, %q), want (my-proj, global, cloud.google.com-report)", p, l, n)
		}
	})

	t.Run("malformed", func(t *testing.T) {
		for _, name := range []string{"", "skills/only", "projects/p/skills/x", "not a name"} {
			if _, _, _, err := c.ParseSkillName(name); err == nil {
				t.Errorf("ParseSkillName(%q) = nil error, want an error", name)
			}
		}
	})
}

// newHTTPClient starts an httptest server serving handler and returns a client
// wired to it, with no credentials required.
func newHTTPClient(t *testing.T, cfg Config, handler http.HandlerFunc) Client {
	t.Helper()
	srv := httptest.NewServer(handler)
	t.Cleanup(srv.Close)
	c, err := NewClient(t.Context(), cfg,
		option.WithEndpoint(srv.URL),
		option.WithoutAuthentication(),
		option.WithHTTPClient(srv.Client()),
	)
	if err != nil {
		t.Fatalf("NewClient() error = %v", err)
	}
	return c
}

// skillJSON renders one skill resource.
func skillJSON(name, rev, state string) map[string]any {
	return map[string]any{
		"name":            name,
		"defaultRevision": rev,
		"state":           state,
		"frontmatter":     map[string]any{"description": "desc of " + name},
	}
}

func TestClientListSkills(t *testing.T) {
	// Two skills on page 1, one on page 2; one of them is unusable.
	page1 := map[string]any{
		"skills": []map[string]any{
			skillJSON("projects/p/locations/l/skills/alpha", "r1", "STATE_ACTIVE"),
			skillJSON("projects/p/locations/l/skills/half", "r2", "STATE_CREATING"),
		},
		"nextPageToken": "tok2",
	}
	page2 := map[string]any{
		"skills": []map[string]any{
			skillJSON("projects/p/locations/l/skills/beta", "r3", "STATE_DEPRECATED"),
		},
	}
	handler := func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Query().Get("pageToken") == "tok2" {
			_ = json.NewEncoder(w).Encode(page2)
			return
		}
		_ = json.NewEncoder(w).Encode(page1)
	}

	t.Run("follows pagination and filters unusable skills", func(t *testing.T) {
		c := newHTTPClient(t, Config{ProjectID: "p", Location: "l"}, handler)
		skills, err := c.ListSkills()
		if err != nil {
			t.Fatalf("ListSkills() error = %v", err)
		}
		var names []string
		for _, s := range skills {
			names = append(names, s.Name)
		}
		slices.Sort(names)
		want := []string{
			"projects/p/locations/l/skills/alpha",
			"projects/p/locations/l/skills/beta",
		}
		if !slices.Equal(names, want) {
			t.Errorf("ListSkills() names = %v, want %v (half is STATE_CREATING)", names, want)
		}
	})

	t.Run("IncludeUnusableSkills keeps every state", func(t *testing.T) {
		c := newHTTPClient(t, Config{ProjectID: "p", Location: "l", IncludeUnusableSkills: true}, handler)
		skills, err := c.ListSkills()
		if err != nil {
			t.Fatalf("ListSkills() error = %v", err)
		}
		if len(skills) != 3 {
			t.Errorf("ListSkills() returned %d skills, want 3 with IncludeUnusableSkills", len(skills))
		}
	})
}

func TestClientGetSkill(t *testing.T) {
	const name = "projects/p/locations/l/skills/alpha"
	handler := func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(skillJSON(name, "r1", "STATE_ACTIVE"))
	}
	c := newHTTPClient(t, Config{ProjectID: "p", Location: "l"}, handler)

	got, err := c.GetSkill(name)
	if err != nil {
		t.Fatalf("GetSkill() error = %v", err)
	}
	if got.Name != name {
		t.Errorf("GetSkill().Name = %q, want %q", got.Name, name)
	}
	if got.DefaultRevision != "r1" {
		t.Errorf("GetSkill().DefaultRevision = %q, want %q", got.DefaultRevision, "r1")
	}
}

func TestClientGetZip(t *testing.T) {
	archive := zipBytes(t, []zipEntry{
		{name: "SKILL.md", content: []byte("# skill")},
		{name: "assets/a.txt", content: []byte("alpha")},
	})
	handler := func(w http.ResponseWriter, r *http.Request) {
		// Download sets ?alt=media; serve the raw archive.
		if r.URL.Query().Get("alt") != "media" {
			http.Error(w, "want alt=media", http.StatusBadRequest)
			return
		}
		w.Header().Set("Content-Type", "application/zip")
		_, _ = w.Write(archive)
	}
	c := newHTTPClient(t, Config{ProjectID: "p", Location: "l"}, handler)

	zr, err := c.GetZip("projects/p/locations/l/skills/alpha/revisions/1")
	if err != nil {
		t.Fatalf("GetZip() error = %v", err)
	}
	var names []string
	for _, f := range zr.File {
		names = append(names, f.Name)
	}
	slices.Sort(names)
	if want := []string{"SKILL.md", "assets/a.txt"}; !slices.Equal(names, want) {
		t.Errorf("GetZip() entries = %v, want %v", names, want)
	}
}

func TestClientListFrontmatters(t *testing.T) {
	list := map[string]any{
		"skills": []map[string]any{
			skillJSON("projects/p/locations/l/skills/alpha", "r1", "STATE_ACTIVE"),
			skillJSON("projects/p/locations/l/skills/beta", "r2", "STATE_ACTIVE"),
		},
	}
	handler := func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1alpha/projects/p/locations/l/skills" {
			_ = json.NewEncoder(w).Encode(list)
			return
		}
		// Otherwise it is a per-skill Get; echo a skill carrying a frontmatter.
		name := r.URL.Path[len("/v1alpha/"):]
		_ = json.NewEncoder(w).Encode(skillJSON(name, "r", "STATE_ACTIVE"))
	}
	c := newHTTPClient(t, Config{ProjectID: "p", Location: "l"}, handler)

	fms, err := c.ListFrontmatters()
	if err != nil {
		t.Fatalf("ListFrontmatters() error = %v", err)
	}
	if len(fms) != 2 {
		t.Fatalf("ListFrontmatters() returned %d frontmatters, want 2", len(fms))
	}
	for _, fm := range fms {
		if fm.Description == "" {
			t.Errorf("frontmatter has an empty description: %+v", fm)
		}
	}
}

// zipBytes builds a zip archive in memory for the given entries, preserving
// their order.
func zipBytes(t *testing.T, entries []zipEntry) []byte {
	t.Helper()
	var buf bytes.Buffer
	zw := zip.NewWriter(&buf)
	for _, e := range entries {
		w, err := zw.Create(e.name)
		if err != nil {
			t.Fatalf("creating zip entry %q: %v", e.name, err)
		}
		if _, err := w.Write(e.content); err != nil {
			t.Fatalf("writing zip entry %q: %v", e.name, err)
		}
	}
	if err := zw.Close(); err != nil {
		t.Fatalf("closing zip: %v", err)
	}
	return buf.Bytes()
}
