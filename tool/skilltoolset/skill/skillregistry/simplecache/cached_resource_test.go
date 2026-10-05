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
	"archive/zip"
	"bytes"
	"errors"
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	agentregistry "google.golang.org/api/agentregistry/v1alpha"

	"google.golang.org/adk/v2/tool/skilltoolset/skill/skillregistry"
)

// zipEntry is one file placed into the archive served by zipClientStub.GetZip.
type zipEntry struct {
	name    string
	content []byte
}

// zipClientStub is a [skillregistry.Client] that serves a single in-memory zip
// from GetZip and counts the calls. Only GetZip is exercised by the resource
// paths; the other methods satisfy the interface and are not expected to run.
type zipClientStub struct {
	mu        sync.Mutex
	entries   []zipEntry // archive contents, in order
	zipErr    error      // when non-nil, GetZip returns it instead of a zip
	callCount int        // number of GetZip calls
	revs      []string   // revisions passed to GetZip, in call order
}

var _ skillregistry.Client = (*zipClientStub)(nil)

func (c *zipClientStub) GetZip(rev string) (*zip.Reader, error) {
	c.mu.Lock()
	c.callCount++
	c.revs = append(c.revs, rev)
	entries, zipErr := c.entries, c.zipErr
	c.mu.Unlock()
	if zipErr != nil {
		return nil, zipErr
	}
	return buildZip(entries)
}

// calls reports how many times GetZip has been invoked.
func (c *zipClientStub) calls() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.callCount
}

// lastRev returns the revision passed to the most recent GetZip call, or "" if
// GetZip has not been called.
func (c *zipClientStub) lastRev() string {
	c.mu.Lock()
	defer c.mu.Unlock()
	if len(c.revs) == 0 {
		return ""
	}
	return c.revs[len(c.revs)-1]
}

// The methods below are unused by the resource-loading paths.
func (c *zipClientStub) ListSkills() ([]*agentregistry.Skill, error) { return nil, nil }

func (c *zipClientStub) ListFrontmatters() ([]*agentregistry.Frontmatter, error) {
	return nil, nil
}
func (c *zipClientStub) GetSkill(string) (*agentregistry.Skill, error)            { return nil, nil }
func (c *zipClientStub) GetRevision(string) (*agentregistry.SkillRevision, error) { return nil, nil }

func (c *zipClientStub) FindFrontmatters(string) ([]*agentregistry.Frontmatter, error) {
	return nil, nil
}
func (c *zipClientStub) ResourceID(name string) string { return name }

func (c *zipClientStub) ParseSkillName(string) (projectID, location, skillName string, resErr error) {
	return "", "", "", nil
}

// buildZip returns a [zip.Reader] over entries, preserving their order.
func buildZip(entries []zipEntry) (*zip.Reader, error) {
	var buf bytes.Buffer
	zw := zip.NewWriter(&buf)
	for _, e := range entries {
		w, err := zw.Create(e.name)
		if err != nil {
			return nil, err
		}
		if _, err := w.Write(e.content); err != nil {
			return nil, err
		}
	}
	if err := zw.Close(); err != nil {
		return nil, err
	}
	b := buf.Bytes()
	return zip.NewReader(bytes.NewReader(b), int64(len(b)))
}

func TestLoadResources(t *testing.T) {
	entries := []zipEntry{
		{name: "SKILL.md", content: []byte("# skill")},
		{name: "assets/a.txt", content: []byte("alpha")},
		{name: "scripts/run.sh", content: []byte("echo hi")},
	}

	t.Run("success", func(t *testing.T) {
		c := &zipClientStub{entries: entries}
		r := newResources(c)

		if err := r.loadResources("rev-1"); err != nil {
			t.Fatalf("loadResources() error = %v", err)
		}
		if !r.loaded {
			t.Error("loaded = false after a successful load, want true")
		}
		if got, want := len(r.res), len(entries); got != want {
			t.Fatalf("len(res) = %d, want %d", got, want)
		}
		// res keeps the archive order; pathToRes indexes the same objects by name.
		for i, e := range entries {
			if r.res[i].path != e.name {
				t.Errorf("res[%d].path = %q, want %q", i, r.res[i].path, e.name)
			}
			got, ok := r.pathToRes[e.name]
			if !ok {
				t.Fatalf("pathToRes is missing %q", e.name)
			}
			if got != r.res[i] {
				t.Errorf("pathToRes[%q] is not res[%d] (same pointer)", e.name, i)
			}
			if string(got.content) != string(e.content) {
				t.Errorf("content for %q = %q, want %q", e.name, got.content, e.content)
			}
		}
	})

	t.Run("re-load replaces previous state", func(t *testing.T) {
		c := &zipClientStub{entries: entries}
		r := newResources(c)
		if err := r.loadResources("rev-1"); err != nil {
			t.Fatalf("first loadResources() error = %v", err)
		}

		// The second revision serves a different, smaller archive.
		c.entries = []zipEntry{{name: "assets/only.txt", content: []byte("x")}}
		if err := r.loadResources("rev-2"); err != nil {
			t.Fatalf("second loadResources() error = %v", err)
		}

		if got, want := len(r.res), 1; got != want {
			t.Fatalf("len(res) = %d, want %d", got, want)
		}
		if _, ok := r.pathToRes["SKILL.md"]; ok {
			t.Error("pathToRes still has SKILL.md from the first load")
		}
		if _, ok := r.pathToRes["assets/only.txt"]; !ok {
			t.Error("pathToRes is missing the entry from the second load")
		}
	})

	t.Run("GetZip error is propagated and leaves the cache unloaded", func(t *testing.T) {
		boom := errors.New("registry down")
		r := newResources(&zipClientStub{zipErr: boom})

		err := r.loadResources("rev-1")
		if !errors.Is(err, boom) {
			t.Fatalf("loadResources() error = %v, want errors.Is %v", err, boom)
		}
		if r.loaded {
			t.Error("loaded = true after a failed load, want false")
		}
	})
}

func TestListResources(t *testing.T) {
	entries := []zipEntry{
		{name: "SKILL.md", content: []byte("# skill")},
		{name: "assets/a.txt", content: []byte("a")},
		{name: "assets/b.txt", content: []byte("b")},
		{name: "scripts/run.sh", content: []byte("run")},
	}

	t.Run("lazy loads once and filters by prefix, preserving order", func(t *testing.T) {
		c := &zipClientStub{entries: entries}
		r := newResources(c)

		got, err := r.listResources("rev-1", "assets/")
		if err != nil {
			t.Fatalf("listResources() error = %v", err)
		}
		want := []string{"assets/a.txt", "assets/b.txt"}
		if diff := cmp.Diff(want, got); diff != "" {
			t.Errorf("listResources diff (-want +got):\n%s", diff)
		}
		if c.calls() != 1 {
			t.Errorf("GetZip called %d times on first use, want 1", c.calls())
		}

		// A second call is served from cache with no extra fetch.
		if _, err := r.listResources("rev-1", "assets/"); err != nil {
			t.Fatalf("second listResources() error = %v", err)
		}
		if c.calls() != 1 {
			t.Errorf("GetZip called %d times after caching, want 1", c.calls())
		}
	})

	t.Run("empty prefix lists every resource", func(t *testing.T) {
		r := newResources(&zipClientStub{entries: entries})
		got, err := r.listResources("rev-1", "")
		if err != nil {
			t.Fatalf("listResources() error = %v", err)
		}
		if len(got) != len(entries) {
			t.Errorf("len = %d, want %d", len(got), len(entries))
		}
	})

	// Known gap (see review item M1): subpath "." matches nothing because entries
	// are stored under their archive paths ("assets/..."), none of which start
	// with ".". WithCompletePreloadSource relies on ListResources(name, ".")
	// listing everything, so this is a bug. The test pins today's behavior so a
	// fix has to update it deliberately.
	t.Run("dot subpath currently matches nothing", func(t *testing.T) {
		r := newResources(&zipClientStub{entries: entries})
		got, err := r.listResources("rev-1", ".")
		if err != nil {
			t.Fatalf("listResources() error = %v", err)
		}
		if len(got) != 0 {
			t.Errorf(`listResources(rev, ".") = %v, want empty (current behavior)`, got)
		}
	})

	// The match is a plain strings.HasPrefix, not path-aware: "script" (no
	// trailing slash) still matches "scripts/...". Pinned as current behavior.
	t.Run("prefix match is not path-aware", func(t *testing.T) {
		r := newResources(&zipClientStub{entries: entries})
		got, err := r.listResources("rev-1", "script")
		if err != nil {
			t.Fatalf("listResources() error = %v", err)
		}
		want := []string{"scripts/run.sh"}
		if diff := cmp.Diff(want, got); diff != "" {
			t.Errorf("listResources diff (-want +got):\n%s", diff)
		}
	})

	t.Run("no match returns an empty list and no error", func(t *testing.T) {
		r := newResources(&zipClientStub{entries: entries})
		got, err := r.listResources("rev-1", "references/")
		if err != nil {
			t.Fatalf("listResources() error = %v", err)
		}
		if len(got) != 0 {
			t.Errorf("listResources = %v, want empty", got)
		}
	})

	// Regression for the write-lock leak: the lazy-load error path must release
	// the write lock, otherwise the next call blocks forever on r.mu.
	t.Run("error path releases the lock so a retry can succeed", func(t *testing.T) {
		boom := errors.New("registry down")
		c := &zipClientStub{entries: entries, zipErr: boom}
		r := newResources(c)

		if _, err := r.listResources("rev-1", "assets/"); !errors.Is(err, boom) {
			t.Fatalf("listResources() error = %v, want errors.Is %v", err, boom)
		}

		// With the leak, this second call deadlocks acquiring the read lock.
		// Guard with a timeout so a regression fails fast instead of hanging the
		// whole suite.
		c.mu.Lock()
		c.zipErr = nil
		c.mu.Unlock()

		type result struct {
			paths []string
			err   error
		}
		done := make(chan result, 1)
		go func() {
			paths, err := r.listResources("rev-1", "assets/")
			done <- result{paths, err}
		}()
		select {
		case got := <-done:
			if got.err != nil {
				t.Fatalf("second listResources() error = %v", got.err)
			}
			if len(got.paths) != 2 {
				t.Errorf("second listResources() = %v, want the 2 assets after recovery", got.paths)
			}
		case <-time.After(5 * time.Second):
			t.Fatal("listResources deadlocked after a failed load: the write lock was leaked")
		}
	})
}

func TestGetResource(t *testing.T) {
	entries := []zipEntry{
		{name: "SKILL.md", content: []byte("# instructions")},
		{name: "assets/a.txt", content: []byte("alpha")},
	}

	t.Run("lazy loads once and returns the resource by exact path", func(t *testing.T) {
		c := &zipClientStub{entries: entries}
		r := newResources(c)

		res, err := r.getResource("rev-1", "assets/a.txt")
		if err != nil {
			t.Fatalf("getResource() error = %v", err)
		}
		if res.path != "assets/a.txt" {
			t.Errorf("path = %q, want %q", res.path, "assets/a.txt")
		}
		if string(res.content) != "alpha" {
			t.Errorf("content = %q, want %q", res.content, "alpha")
		}
		if c.calls() != 1 {
			t.Errorf("GetZip called %d times, want 1", c.calls())
		}

		// A second lookup is served from cache.
		if _, err := r.getResource("rev-1", "SKILL.md"); err != nil {
			t.Fatalf("second getResource() error = %v", err)
		}
		if c.calls() != 1 {
			t.Errorf("GetZip called %d times after caching, want 1", c.calls())
		}
	})

	// getResource matches by exact path with no allow-listing, so SKILL.md (the
	// instructions file) is reachable as a resource. Pinned as current behavior
	// (see review item M2: the resource hardening is absent here).
	t.Run("exposes SKILL.md by exact path", func(t *testing.T) {
		r := newResources(&zipClientStub{entries: entries})
		res, err := r.getResource("rev-1", "SKILL.md")
		if err != nil {
			t.Fatalf("getResource() error = %v", err)
		}
		if string(res.content) != "# instructions" {
			t.Errorf("content = %q, want %q", res.content, "# instructions")
		}
	})

	t.Run("missing path returns an error", func(t *testing.T) {
		r := newResources(&zipClientStub{entries: entries})
		if _, err := r.getResource("rev-1", "nope.txt"); err == nil {
			t.Fatal("getResource() error = nil, want an error for a missing path")
		}
	})

	// getResource releases the write lock on its error path (via defer), so a
	// failed load leaves the cache retryable rather than stuck.
	t.Run("error path is propagated and retried", func(t *testing.T) {
		boom := errors.New("registry down")
		c := &zipClientStub{entries: entries, zipErr: boom}
		r := newResources(c)

		if _, err := r.getResource("rev-1", "assets/a.txt"); !errors.Is(err, boom) {
			t.Fatalf("getResource() error = %v, want errors.Is %v", err, boom)
		}
		if c.calls() != 1 {
			t.Errorf("GetZip called %d times, want 1", c.calls())
		}

		c.mu.Lock()
		c.zipErr = nil
		c.mu.Unlock()

		res, err := r.getResource("rev-1", "assets/a.txt")
		if err != nil {
			t.Fatalf("getResource() after recovery error = %v", err)
		}
		if string(res.content) != "alpha" {
			t.Errorf("content = %q, want %q", res.content, "alpha")
		}
		if c.calls() != 2 {
			t.Errorf("GetZip called %d times after retry, want 2", c.calls())
		}
	})
}

// Concurrent getResource/listResources callers must load the archive exactly
// once and stay race-free. Run under -race, this exercises the double-checked
// locking that guards the lazy load.
func TestResourcesConcurrentAccess(t *testing.T) {
	entries := []zipEntry{
		{name: "SKILL.md", content: []byte("# skill")},
		{name: "assets/a.txt", content: []byte("alpha")},
		{name: "assets/b.txt", content: []byte("beta")},
		{name: "scripts/run.sh", content: []byte("run")},
	}
	c := &zipClientStub{entries: entries}
	r := newResources(c)

	const workers = 32
	var wg sync.WaitGroup
	errs := make(chan error, workers)
	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			if i%2 == 0 {
				res, err := r.getResource("rev-1", "assets/a.txt")
				if err != nil {
					errs <- fmt.Errorf("getResource: %w", err)
					return
				}
				if string(res.content) != "alpha" {
					errs <- fmt.Errorf("getResource content = %q, want alpha", res.content)
				}
				return
			}
			got, err := r.listResources("rev-1", "assets/")
			if err != nil {
				errs <- fmt.Errorf("listResources: %w", err)
				return
			}
			if len(got) != 2 {
				errs <- fmt.Errorf("listResources len = %d, want 2", len(got))
			}
		}(i)
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Error(err)
	}

	if c.calls() != 1 {
		t.Errorf("GetZip called %d times under concurrency, want exactly 1", c.calls())
	}
}
