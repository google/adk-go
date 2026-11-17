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
	"slices"
	"testing"

	"github.com/google/go-cmp/cmp"
)

func TestInMemorySource(t *testing.T) {
	weather := &Frontmatter{Name: "weather", Description: "Tells the weather"}
	math := &Frontmatter{Name: "math", Description: "Does sums"}
	source, err := NewInMemorySource(
		InMemorySkill{
			Frontmatter:  weather,
			Instructions: "  Look outside.\n",
			Resources: map[string][]byte{
				"references/guide.md":       []byte("guide"),
				"references/api/v1/spec.md": []byte("spec"),
				"references/api.md":         []byte("api"),
				"assets/logo.png":           {0x89, 'P', 'N', 'G'},
				"scripts/run.sh":            []byte("echo run"),
			},
		},
		InMemorySkill{Frontmatter: math, Instructions: "Add."},
	)
	if err != nil {
		t.Fatalf("NewInMemorySource() error = %v", err)
	}
	ctx := t.Context()

	t.Run("ListFrontmatters keeps the order given", func(t *testing.T) {
		got, err := source.ListFrontmatters(ctx)
		if err != nil {
			t.Fatalf("ListFrontmatters() error = %v", err)
		}
		if diff := cmp.Diff([]*Frontmatter{weather, math}, got); diff != "" {
			t.Errorf("ListFrontmatters() mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("LoadFrontmatter", func(t *testing.T) {
		got, err := source.LoadFrontmatter(ctx, "math")
		if err != nil {
			t.Fatalf("LoadFrontmatter() error = %v", err)
		}
		if diff := cmp.Diff(math, got); diff != "" {
			t.Errorf("LoadFrontmatter() mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("LoadInstructions returns them as given", func(t *testing.T) {
		got, err := source.LoadInstructions(ctx, "weather")
		if err != nil {
			t.Fatalf("LoadInstructions() error = %v", err)
		}
		if want := "  Look outside.\n"; got != want {
			t.Errorf("LoadInstructions() = %q, want %q", got, want)
		}
	})

	t.Run("LoadResource", func(t *testing.T) {
		for _, tc := range []struct {
			resourcePath string
			want         []byte
		}{
			{"references/guide.md", []byte("guide")},
			{"assets/logo.png", []byte{0x89, 'P', 'N', 'G'}},
			{"references/api/../guide.md", []byte("guide")},
			{"./scripts//run.sh", []byte("echo run")},
		} {
			resource, err := source.LoadResource(ctx, "weather", tc.resourcePath)
			if err != nil {
				t.Errorf("LoadResource(%q) error = %v", tc.resourcePath, err)
				continue
			}
			got, err := io.ReadAll(resource)
			if err != nil {
				t.Errorf("LoadResource(%q) read error = %v", tc.resourcePath, err)
			}
			if err := resource.Close(); err != nil {
				t.Errorf("LoadResource(%q) close error = %v", tc.resourcePath, err)
			}
			if !slices.Equal(got, tc.want) {
				t.Errorf("LoadResource(%q) = %q, want %q", tc.resourcePath, got, tc.want)
			}
		}
	})

	t.Run("ListResources", func(t *testing.T) {
		all := []string{"assets/logo.png", "references/api.md", "references/api/v1/spec.md", "references/guide.md", "scripts/run.sh"}
		for _, tc := range []struct {
			skill, subpath string
			want           []string
		}{
			{"weather", ".", all},
			{"weather", "/", all},
			{"weather", "references", []string{"references/api.md", "references/api/v1/spec.md", "references/guide.md"}},
			{"weather", "references/", []string{"references/api.md", "references/api/v1/spec.md", "references/guide.md"}},
			{"weather", "references/api", []string{"references/api/v1/spec.md"}},
			{"weather", "references/guide.md", []string{"references/guide.md"}},
			{"math", ".", nil},
		} {
			got, err := source.ListResources(ctx, tc.skill, tc.subpath)
			if err != nil {
				t.Errorf("ListResources(%q, %q) error = %v", tc.skill, tc.subpath, err)
				continue
			}
			if diff := cmp.Diff(tc.want, got); diff != "" {
				t.Errorf("ListResources(%q, %q) mismatch (-want +got):\n%s", tc.skill, tc.subpath, diff)
			}
		}
	})

	t.Run("a skill that is not there", func(t *testing.T) {
		if _, err := source.LoadFrontmatter(ctx, "absent"); !errors.Is(err, ErrSkillNotFound) {
			t.Errorf("LoadFrontmatter() error = %v, want %v", err, ErrSkillNotFound)
		}
		if _, err := source.LoadInstructions(ctx, "absent"); !errors.Is(err, ErrSkillNotFound) {
			t.Errorf("LoadInstructions() error = %v, want %v", err, ErrSkillNotFound)
		}
		if _, err := source.LoadResource(ctx, "absent", "references/guide.md"); !errors.Is(err, ErrSkillNotFound) {
			t.Errorf("LoadResource() error = %v, want %v", err, ErrSkillNotFound)
		}
		if _, err := source.ListResources(ctx, "absent", "."); !errors.Is(err, ErrSkillNotFound) {
			t.Errorf("ListResources() error = %v, want %v", err, ErrSkillNotFound)
		}
	})

	t.Run("a resource that is not there", func(t *testing.T) {
		for _, resourcePath := range []string{"references/absent.md", "references", "guide.md"} {
			if _, err := source.LoadResource(ctx, "weather", resourcePath); !errors.Is(err, ErrResourceNotFound) {
				t.Errorf("LoadResource(%q) error = %v, want %v", resourcePath, err, ErrResourceNotFound)
			}
		}
		// "refer" is the start of a directory's name, not a directory.
		for _, subpath := range []string{"references/absent", "references/guide.md/", "refer", "notes"} {
			paths, err := source.ListResources(ctx, "weather", subpath)
			if !errors.Is(err, ErrResourceNotFound) {
				t.Errorf("ListResources(%q) = %v, %v, want %v", subpath, paths, err, ErrResourceNotFound)
			}
		}
		if _, err := source.ListResources(ctx, "math", "references"); !errors.Is(err, ErrResourceNotFound) {
			t.Errorf("ListResources() of a skill without resources: error = %v, want %v", err, ErrResourceNotFound)
		}
	})
}

func TestNewInMemorySource_Errors(t *testing.T) {
	valid := &Frontmatter{Name: "weather", Description: "Tells the weather"}
	for _, tc := range []struct {
		name   string
		skills []InMemorySkill
		want   error
	}{
		{
			name:   "no frontmatter",
			skills: []InMemorySkill{{Instructions: "Look outside."}},
			want:   ErrInvalidFrontmatter,
		},
		{
			name:   "frontmatter that does not validate",
			skills: []InMemorySkill{{Frontmatter: &Frontmatter{Name: "Bad_Name", Description: "d"}}},
			want:   ErrInvalidFrontmatter,
		},
		{
			name:   "two skills of one name",
			skills: []InMemorySkill{{Frontmatter: valid}, {Frontmatter: &Frontmatter{Name: "weather", Description: "Another"}}},
			want:   ErrDuplicateSkill,
		},
		{name: "resource outside the resource directories", skills: withResource(valid, "notes/todo.md"), want: ErrInvalidResourcePath},
		{name: "resource beside SKILL.md", skills: withResource(valid, "guide.md"), want: ErrInvalidResourcePath},
		{name: "resource directory itself", skills: withResource(valid, "references/"), want: ErrInvalidResourcePath},
		{name: "resource path with a parent element", skills: withResource(valid, "references/../SKILL.md"), want: ErrInvalidResourcePath},
		{name: "resource path that stays inside but is not shortest", skills: withResource(valid, "references/a/../guide.md"), want: ErrInvalidResourcePath},
		{name: "resource path with a dot element", skills: withResource(valid, "references/./guide.md"), want: ErrInvalidResourcePath},
		{name: "resource path with a repeated slash", skills: withResource(valid, "assets//logo.png"), want: ErrInvalidResourcePath},
		{name: "resource path with a trailing slash", skills: withResource(valid, "scripts/run.sh/"), want: ErrInvalidResourcePath},
		{name: "absolute resource path", skills: withResource(valid, "/references/guide.md"), want: ErrInvalidResourcePath},
	} {
		t.Run(tc.name, func(t *testing.T) {
			source, err := NewInMemorySource(tc.skills...)
			if !errors.Is(err, tc.want) {
				t.Errorf("NewInMemorySource() error = %v, want %v", err, tc.want)
			}
			if source != nil {
				t.Errorf("NewInMemorySource() = %v, want nil", source)
			}
		})
	}
}

func withResource(frontmatter *Frontmatter, resourcePath string) []InMemorySkill {
	return []InMemorySkill{{Frontmatter: frontmatter, Resources: map[string][]byte{resourcePath: []byte("content")}}}
}

func TestNewInMemorySource_NoSkills(t *testing.T) {
	source, err := NewInMemorySource()
	if err != nil {
		t.Fatalf("NewInMemorySource() error = %v", err)
	}
	got, err := source.ListFrontmatters(t.Context())
	if err != nil {
		t.Fatalf("ListFrontmatters() error = %v", err)
	}
	if len(got) != 0 {
		t.Errorf("ListFrontmatters() = %v, want none", got)
	}
}

// The source must not change when the caller changes what it passed in or
// what it got back.
func TestInMemorySource_IsIndependentOfItsCaller(t *testing.T) {
	resources := map[string][]byte{"references/b.md": []byte("b"), "references/a.md": []byte("a")}
	skills := []InMemorySkill{
		{Frontmatter: &Frontmatter{Name: "first", Description: "d"}, Resources: resources},
		{Frontmatter: &Frontmatter{Name: "second", Description: "d"}},
	}
	source, err := NewInMemorySource(skills...)
	if err != nil {
		t.Fatalf("NewInMemorySource() error = %v", err)
	}
	ctx := t.Context()

	// Through the slice and through the map: the source must hold neither.
	skills[0].Resources["references/c.md"] = []byte("c")
	delete(resources, "references/a.md")
	slices.Reverse(skills)

	frontmatters, err := source.ListFrontmatters(ctx)
	if err != nil {
		t.Fatalf("ListFrontmatters() error = %v", err)
	}
	slices.Reverse(frontmatters)
	paths, err := source.ListResources(ctx, "first", ".")
	if err != nil {
		t.Fatalf("ListResources() error = %v", err)
	}
	slices.Reverse(paths)

	frontmatters, err = source.ListFrontmatters(ctx)
	if err != nil {
		t.Fatalf("ListFrontmatters() error = %v", err)
	}
	var names []string
	for _, frontmatter := range frontmatters {
		names = append(names, frontmatter.Name)
	}
	if diff := cmp.Diff([]string{"first", "second"}, names); diff != "" {
		t.Errorf("ListFrontmatters() names mismatch (-want +got):\n%s", diff)
	}
	for _, subpath := range []string{".", "references"} {
		paths, err = source.ListResources(ctx, "first", subpath)
		if err != nil {
			t.Fatalf("ListResources(%q) error = %v", subpath, err)
		}
		if diff := cmp.Diff([]string{"references/a.md", "references/b.md"}, paths); diff != "" {
			t.Errorf("ListResources(%q) mismatch (-want +got):\n%s", subpath, diff)
		}
	}
	if _, err := source.LoadResource(ctx, "first", "references/c.md"); !errors.Is(err, ErrResourceNotFound) {
		t.Errorf("LoadResource() of a resource added after construction: error = %v, want %v", err, ErrResourceNotFound)
	}
}
