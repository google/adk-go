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

package simplecache

import (
	"context"
	"errors"
	"io"
	"testing"

	"github.com/google/go-cmp/cmp"

	"google.golang.org/adk/v2/tool/skilltoolset/skill"
	"google.golang.org/adk/v2/tool/skilltoolset/skill/skillregistry"
)

// fakeCache is a recording [skillregistry.Cache]. Each method returns its
// configured value and error and records the arguments it was called with, so a
// test can assert that CachedSkillRegistrySource forwards them unchanged.
type fakeCache struct {
	// Configured return values.
	frontmatters  []*skill.Frontmatter
	frontmatter   *skill.Frontmatter
	instructions  string
	resourceData  string
	resourcePaths []string

	// Configured errors, returned alongside the values above.
	listFMErr    error
	loadFMErr    error
	loadInstrErr error
	loadResErr   error
	listResErr   error

	// Recorded arguments.
	gotLoadFMName    string
	gotLoadInstrName string
	gotLoadResName   string
	gotLoadResPath   string
	gotListResName   string
	gotListResSub    string
}

var _ skillregistry.Cache = (*fakeCache)(nil)

func (f *fakeCache) WarmUp() error { return nil }

func (f *fakeCache) StopAutorefresh() {
}

func (f *fakeCache) ListFrontmatters() ([]*skill.Frontmatter, error) {
	return f.frontmatters, f.listFMErr
}

func (f *fakeCache) LoadFrontmatter(name string) (*skill.Frontmatter, error) {
	f.gotLoadFMName = name
	return f.frontmatter, f.loadFMErr
}

func (f *fakeCache) LoadInstructions(name string) (string, error) {
	f.gotLoadInstrName = name
	return f.instructions, f.loadInstrErr
}

func (f *fakeCache) LoadResource(name, resourcePath string) (string, error) {
	f.gotLoadResName = name
	f.gotLoadResPath = resourcePath
	return f.resourceData, f.loadResErr
}

func (f *fakeCache) ListResources(name, subpath string) ([]string, error) {
	f.gotListResName = name
	f.gotListResSub = subpath
	return f.resourcePaths, f.listResErr
}

// mustSource builds a CachedSkillRegistrySource over c, failing the test if the
// constructor errors.
func mustSource(t *testing.T, c skillregistry.Cache) *CachedSkillRegistrySource {
	t.Helper()
	src, err := NewCachedSkillRegistrySource(context.Background(), c)
	if err != nil {
		t.Fatalf("NewCachedSkillRegistrySource() error = %v", err)
	}
	return src
}

func TestNewCachedSkillRegistrySource(t *testing.T) {
	fc := &fakeCache{}
	src, err := NewCachedSkillRegistrySource(context.Background(), fc)
	if err != nil {
		t.Fatalf("NewCachedSkillRegistrySource() error = %v", err)
	}
	if src == nil {
		t.Fatal("NewCachedSkillRegistrySource() returned a nil source")
	}
	if src.cache != fc {
		t.Error("source does not wrap the provided cache")
	}
}

func TestCachedSourceListFrontmatters(t *testing.T) {
	ctx := context.Background()

	t.Run("passes the cache result through", func(t *testing.T) {
		want := []*skill.Frontmatter{{Name: "alpha"}, {Name: "beta"}}
		src := mustSource(t, &fakeCache{frontmatters: want})

		got, err := src.ListFrontmatters(ctx)
		if err != nil {
			t.Fatalf("ListFrontmatters() error = %v", err)
		}
		if diff := cmp.Diff(want, got); diff != "" {
			t.Errorf("ListFrontmatters diff (-want +got):\n%s", diff)
		}
	})

	t.Run("error is propagated", func(t *testing.T) {
		boom := errors.New("cache down")
		src := mustSource(t, &fakeCache{listFMErr: boom})

		if _, err := src.ListFrontmatters(ctx); !errors.Is(err, boom) {
			t.Fatalf("ListFrontmatters() error = %v, want errors.Is %v", err, boom)
		}
	})
}

func TestCachedSourceLoadFrontmatter(t *testing.T) {
	ctx := context.Background()

	t.Run("forwards the name and returns the frontmatter", func(t *testing.T) {
		want := &skill.Frontmatter{Name: "alpha", Description: "Alpha."}
		fc := &fakeCache{frontmatter: want}
		src := mustSource(t, fc)

		got, err := src.LoadFrontmatter(ctx, "alpha")
		if err != nil {
			t.Fatalf("LoadFrontmatter() error = %v", err)
		}
		if got != want {
			t.Errorf("LoadFrontmatter() = %+v, want %+v", got, want)
		}
		if fc.gotLoadFMName != "alpha" {
			t.Errorf("cache.LoadFrontmatter called with name %q, want %q", fc.gotLoadFMName, "alpha")
		}
	})

	// Source implementations must surface the package's sentinel errors; the
	// adapter has to pass ErrSkillNotFound through untouched.
	t.Run("ErrSkillNotFound is propagated", func(t *testing.T) {
		src := mustSource(t, &fakeCache{loadFMErr: skill.ErrSkillNotFound})

		if _, err := src.LoadFrontmatter(ctx, "missing"); !errors.Is(err, skill.ErrSkillNotFound) {
			t.Fatalf("LoadFrontmatter() error = %v, want errors.Is %v", err, skill.ErrSkillNotFound)
		}
	})
}

func TestCachedSourceLoadInstructions(t *testing.T) {
	ctx := context.Background()

	t.Run("forwards the name and returns the instruction", func(t *testing.T) {
		fc := &fakeCache{instructions: "# body"}
		src := mustSource(t, fc)

		got, err := src.LoadInstructions(ctx, "alpha")
		if err != nil {
			t.Fatalf("LoadInstructions() error = %v", err)
		}
		if got != "# body" {
			t.Errorf("LoadInstructions() = %q, want %q", got, "# body")
		}
		if fc.gotLoadInstrName != "alpha" {
			t.Errorf("cache.LoadInstructions called with name %q, want %q", fc.gotLoadInstrName, "alpha")
		}
	})

	t.Run("error is propagated", func(t *testing.T) {
		boom := errors.New("boom")
		src := mustSource(t, &fakeCache{loadInstrErr: boom})

		if _, err := src.LoadInstructions(ctx, "alpha"); !errors.Is(err, boom) {
			t.Fatalf("LoadInstructions() error = %v, want errors.Is %v", err, boom)
		}
	})
}

func TestCachedSourceListResources(t *testing.T) {
	ctx := context.Background()

	t.Run("forwards name and subpath and returns the paths", func(t *testing.T) {
		want := []string{"assets/a.txt", "assets/b.txt"}
		fc := &fakeCache{resourcePaths: want}
		src := mustSource(t, fc)

		got, err := src.ListResources(ctx, "alpha", "assets/")
		if err != nil {
			t.Fatalf("ListResources() error = %v", err)
		}
		if diff := cmp.Diff(want, got); diff != "" {
			t.Errorf("ListResources diff (-want +got):\n%s", diff)
		}
		// name and subpath must reach the cache in the right positions.
		if fc.gotListResName != "alpha" || fc.gotListResSub != "assets/" {
			t.Errorf("cache.ListResources(name=%q, subpath=%q), want (%q, %q)",
				fc.gotListResName, fc.gotListResSub, "alpha", "assets/")
		}
	})

	t.Run("error is propagated", func(t *testing.T) {
		boom := errors.New("boom")
		src := mustSource(t, &fakeCache{listResErr: boom})

		if _, err := src.ListResources(ctx, "alpha", "."); !errors.Is(err, boom) {
			t.Fatalf("ListResources() error = %v, want errors.Is %v", err, boom)
		}
	})
}

func TestCachedSourceLoadResource(t *testing.T) {
	ctx := context.Background()

	t.Run("wraps the content in a readable, closable stream", func(t *testing.T) {
		fc := &fakeCache{resourceData: "file contents"}
		src := mustSource(t, fc)

		rc, err := src.LoadResource(ctx, "alpha", "assets/a.txt")
		if err != nil {
			t.Fatalf("LoadResource() error = %v", err)
		}
		if rc == nil {
			t.Fatal("LoadResource() returned a nil reader on success")
		}
		got, err := io.ReadAll(rc)
		if err != nil {
			t.Fatalf("io.ReadAll() error = %v", err)
		}
		if string(got) != "file contents" {
			t.Errorf("content = %q, want %q", got, "file contents")
		}
		if err := rc.Close(); err != nil {
			t.Errorf("Close() error = %v", err)
		}
		// name and resourcePath must reach the cache in the right positions.
		if fc.gotLoadResName != "alpha" || fc.gotLoadResPath != "assets/a.txt" {
			t.Errorf("cache.LoadResource(name=%q, path=%q), want (%q, %q)",
				fc.gotLoadResName, fc.gotLoadResPath, "alpha", "assets/a.txt")
		}
	})

	t.Run("error returns a nil reader and propagates the error", func(t *testing.T) {
		src := mustSource(t, &fakeCache{loadResErr: skill.ErrResourceNotFound})

		rc, err := src.LoadResource(ctx, "alpha", "missing")
		if !errors.Is(err, skill.ErrResourceNotFound) {
			t.Fatalf("LoadResource() error = %v, want errors.Is %v", err, skill.ErrResourceNotFound)
		}
		if rc != nil {
			t.Errorf("LoadResource() returned a non-nil reader on error: %v", rc)
		}
	})

	// Empty content must still be a non-nil, readable, closable stream rather
	// than a nil reader the caller would have to special-case.
	t.Run("empty content yields a non-nil empty stream", func(t *testing.T) {
		src := mustSource(t, &fakeCache{resourceData: ""})

		rc, err := src.LoadResource(ctx, "alpha", "empty.txt")
		if err != nil {
			t.Fatalf("LoadResource() error = %v", err)
		}
		if rc == nil {
			t.Fatal("LoadResource() returned a nil reader for empty content, want an empty stream")
		}
		got, err := io.ReadAll(rc)
		if err != nil {
			t.Fatalf("io.ReadAll() error = %v", err)
		}
		if len(got) != 0 {
			t.Errorf("content = %q, want empty", got)
		}
		if err := rc.Close(); err != nil {
			t.Errorf("Close() error = %v", err)
		}
	})
}
