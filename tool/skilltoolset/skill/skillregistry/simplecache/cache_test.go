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
	"errors"
	"fmt"
	"slices"
	"sync"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	agentregistry "google.golang.org/api/agentregistry/v1alpha"

	"google.golang.org/adk/v2/tool/skilltoolset/skill"
	"google.golang.org/adk/v2/tool/skilltoolset/skill/skillregistry"
)

const (
	nameAlpha = "projects/p/locations/global/skills/alpha"
	nameBeta  = "projects/p/locations/global/skills/beta"
	revAlpha  = "rev-alpha"
	revBeta   = "rev-beta"
	bodyAlpha = "# Alpha instructions\n"
	bodyBeta  = "# Beta instructions\n"
)

// fullClient is a concurrency-safe [skillregistry.Client] that serves both the
// warm-up path (ListSkills -> GetSkill -> ParseSkillName) and the resource path
// (GetZip, keyed by revision). The other package fakes each cover only one of
// those paths (fakeClient has no zip, zipClientStub has no skills), so the cache
// tests need this combined one.
type fullClient struct {
	mu sync.Mutex

	listed []*agentregistry.Skill          // ListSkills result, in order
	byName map[string]*agentregistry.Skill // GetSkill result, by resource name
	zips   map[string][]zipEntry           // GetZip archives, by revision

	listErr error            // when non-nil, ListSkills returns it
	getErr  map[string]error // GetSkill error, by resource name
	zipErr  map[string]error // GetZip error, by revision

	listCalls   int
	getZipCalls int

	// onList, when non-nil, receives (non-blocking) at the start of every
	// ListSkills call, so a test can observe periodic reloads.
	onList chan struct{}
}

var _ skillregistry.Client = (*fullClient)(nil)

func (c *fullClient) ListSkills() ([]*agentregistry.Skill, error) {
	c.mu.Lock()
	c.listCalls++
	listed, listErr, onList := c.listed, c.listErr, c.onList
	c.mu.Unlock()
	if onList != nil {
		select {
		case onList <- struct{}{}:
		default:
		}
	}
	if listErr != nil {
		return nil, listErr
	}
	return listed, nil
}

func (c *fullClient) GetSkill(name string) (*agentregistry.Skill, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if err := c.getErr[name]; err != nil {
		return nil, err
	}
	s, ok := c.byName[name]
	if !ok {
		return nil, fmt.Errorf("fullClient: no skill for the given name")
	}
	return s, nil
}

func (c *fullClient) GetZip(rev string) (*zip.Reader, error) {
	c.mu.Lock()
	c.getZipCalls++
	zipErr := c.zipErr[rev]
	entries, ok := c.zips[rev]
	c.mu.Unlock()
	if zipErr != nil {
		return nil, zipErr
	}
	if !ok {
		return nil, fmt.Errorf("fullClient: no zip for the given revision")
	}
	return buildZip(entries)
}

func (c *fullClient) ParseSkillName(name string) (projectID, location, skillName string, resErr error) {
	m := skillIDRegex.FindStringSubmatch(name)
	if len(m) != 4 {
		return "", "", "", fmt.Errorf("fullClient: cannot parse the given name")
	}
	return m[1], m[2], m[3], nil
}

// calls reports how many times ListSkills and GetZip have been invoked.
func (c *fullClient) calls() (list, getZip int) {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.listCalls, c.getZipCalls
}

// The methods below are unused by the cache paths.
func (c *fullClient) ListFrontmatters() ([]*agentregistry.Frontmatter, error) { return nil, nil }

func (c *fullClient) GetRevision(string) (*agentregistry.SkillRevision, error) {
	return nil, nil
}

func (c *fullClient) FindFrontmatters(string) ([]*agentregistry.Frontmatter, error) {
	return nil, nil
}
func (c *fullClient) ResourceID(name string) string { return name }

// newFullClient builds a fullClient serving two skills (alpha, beta), each with
// a valid SKILL.md and one extra resource.
func newFullClient(t *testing.T) *fullClient {
	t.Helper()
	mdAlpha, err := skill.Build(&skill.Frontmatter{Name: "alpha", Description: "Alpha."}, bodyAlpha)
	if err != nil {
		t.Fatalf("skill.Build(alpha) error = %v", err)
	}
	mdBeta, err := skill.Build(&skill.Frontmatter{Name: "beta", Description: "Beta."}, bodyBeta)
	if err != nil {
		t.Fatalf("skill.Build(beta) error = %v", err)
	}
	return &fullClient{
		listed: []*agentregistry.Skill{
			{Name: nameAlpha, DefaultRevision: revAlpha},
			{Name: nameBeta, DefaultRevision: revBeta},
		},
		byName: map[string]*agentregistry.Skill{
			nameAlpha: {Name: nameAlpha, DefaultRevision: revAlpha, Frontmatter: &agentregistry.Frontmatter{Description: "Alpha."}},
			nameBeta:  {Name: nameBeta, DefaultRevision: revBeta, Frontmatter: &agentregistry.Frontmatter{Description: "Beta."}},
		},
		zips: map[string][]zipEntry{
			revAlpha: {{name: "SKILL.md", content: mdAlpha}, {name: "assets/a.txt", content: []byte("alpha-asset")}},
			revBeta:  {{name: "SKILL.md", content: mdBeta}, {name: "scripts/run.sh", content: []byte("echo beta")}},
		},
		getErr: map[string]error{},
		zipErr: map[string]error{},
	}
}

// newWarmCache constructs a *cache over c and warms it up, without starting the
// periodic refresher goroutine. It is the seam for exercising the read methods
// directly.
func newWarmCache(t *testing.T, c skillregistry.Client) *cache {
	t.Helper()
	ca := &cache{
		client:                  c,
		frontmattersParallelism: 4,
		updateInterval:          time.Hour,
		stop:                    make(chan struct{}),
	}
	if err := ca.WarmUp(); err != nil {
		t.Fatalf("WarmUp() error = %v", err)
	}
	return ca
}

// frontmatterNames returns the sorted Name of each frontmatter, so order-free
// comparisons are stable (ListFrontmatters iterates a map).
func frontmatterNames(fms []*skill.Frontmatter) []string {
	names := make([]string, 0, len(fms))
	for _, fm := range fms {
		names = append(names, fm.Name)
	}
	slices.Sort(names)
	return names
}

func TestNewCache(t *testing.T) {
	t.Run("warms up and starts the periodic refresher", func(t *testing.T) {
		c := newFullClient(t)
		ca, err := NewCache(CacheConfig{Client: c, UpdateInterval: time.Hour})
		if err != nil {
			t.Fatalf("NewCache() error = %v", err)
		}
		t.Cleanup(func() { close(ca.(*cache).stop) })

		got, err := ca.ListFrontmatters()
		if err != nil {
			t.Fatalf("ListFrontmatters() error = %v", err)
		}
		if diff := cmp.Diff([]string{"alpha", "beta"}, frontmatterNames(got)); diff != "" {
			t.Errorf("frontmatter names diff (-want +got):\n%s", diff)
		}
		if list, _ := c.calls(); list == 0 {
			t.Error("warm-up did not call ListSkills")
		}
	})

	t.Run("SkipWarmUp leaves the cache empty", func(t *testing.T) {
		c := newFullClient(t)
		ca, err := NewCache(CacheConfig{Client: c, UpdateInterval: time.Hour, SkipWarmUp: true})
		if err != nil {
			t.Fatalf("NewCache() error = %v", err)
		}
		t.Cleanup(func() { close(ca.(*cache).stop) })

		got, err := ca.ListFrontmatters()
		if err != nil {
			t.Fatalf("ListFrontmatters() error = %v", err)
		}
		if len(got) != 0 {
			t.Errorf("ListFrontmatters() = %d entries, want 0 before warm-up", len(got))
		}
		if list, _ := c.calls(); list != 0 {
			t.Errorf("ListSkills called %d times with SkipWarmUp, want 0", list)
		}
	})

	t.Run("warm-up failure is returned and no cache is created", func(t *testing.T) {
		boom := errors.New("registry down")
		ca, err := NewCache(CacheConfig{Client: &fullClient{listErr: boom}, UpdateInterval: time.Hour})
		if !errors.Is(err, boom) {
			t.Fatalf("NewCache() error = %v, want errors.Is %v", err, boom)
		}
		if ca != nil {
			t.Error("NewCache() returned a non-nil cache on warm-up failure")
		}
	})

	// Regression for the zero-interval default: storing cfg.UpdateInterval (0)
	// instead of the computed default made runPeriodicalReloads call
	// time.NewTicker(0), which panics and crashes the process.
	t.Run("zero interval falls back to the one-hour default", func(t *testing.T) {
		ca, err := NewCache(CacheConfig{Client: newFullClient(t), SkipWarmUp: true})
		if err != nil {
			t.Fatalf("NewCache() error = %v", err)
		}
		impl := ca.(*cache)
		t.Cleanup(func() { close(impl.stop) })
		if impl.updateInterval != time.Hour {
			t.Errorf("updateInterval = %v, want %v (the default)", impl.updateInterval, time.Hour)
		}
	})
}

func TestCacheListFrontmatters(t *testing.T) {
	t.Run("returns every warmed frontmatter", func(t *testing.T) {
		ca := newWarmCache(t, newFullClient(t))

		got, err := ca.ListFrontmatters()
		if err != nil {
			t.Fatalf("ListFrontmatters() error = %v", err)
		}
		if diff := cmp.Diff([]string{"alpha", "beta"}, frontmatterNames(got)); diff != "" {
			t.Errorf("frontmatter names diff (-want +got):\n%s", diff)
		}
	})

	t.Run("empty registry yields an empty, non-nil list", func(t *testing.T) {
		ca := newWarmCache(t, &fullClient{})

		got, err := ca.ListFrontmatters()
		if err != nil {
			t.Fatalf("ListFrontmatters() error = %v", err)
		}
		if got == nil {
			t.Fatal("ListFrontmatters() = nil, want an empty slice")
		}
		if len(got) != 0 {
			t.Errorf("ListFrontmatters() = %d entries, want 0", len(got))
		}
	})
}

func TestCacheLoadFrontmatter(t *testing.T) {
	ca := newWarmCache(t, newFullClient(t))

	t.Run("returns the frontmatter by prefixed name", func(t *testing.T) {
		fm, err := ca.LoadFrontmatter("alpha")
		if err != nil {
			t.Fatalf("LoadFrontmatter() error = %v", err)
		}
		if fm.Name != "alpha" {
			t.Errorf("Name = %q, want %q", fm.Name, "alpha")
		}
		if fm.Description != "Alpha." {
			t.Errorf("Description = %q, want %q", fm.Description, "Alpha.")
		}
	})

	t.Run("unknown skill returns ErrSkillNotFound", func(t *testing.T) {
		if _, err := ca.LoadFrontmatter("missing"); !errors.Is(err, skill.ErrSkillNotFound) {
			t.Fatalf("LoadFrontmatter() error = %v, want errors.Is %v", err, skill.ErrSkillNotFound)
		}
	})
}

func TestCacheLoadInstructions(t *testing.T) {
	t.Run("loads, parses, and caches SKILL.md", func(t *testing.T) {
		c := newFullClient(t)
		ca := newWarmCache(t, c)

		got, err := ca.LoadInstructions("alpha")
		if err != nil {
			t.Fatalf("LoadInstructions() error = %v", err)
		}
		if got != bodyAlpha {
			t.Errorf("instructions = %q, want %q", got, bodyAlpha)
		}

		// A second call is served from cache, with no extra download.
		if _, err := ca.LoadInstructions("alpha"); err != nil {
			t.Fatalf("second LoadInstructions() error = %v", err)
		}
		if _, getZip := c.calls(); getZip != 1 {
			t.Errorf("GetZip called %d times, want 1 (cached)", getZip)
		}
	})

	t.Run("unknown skill returns ErrSkillNotFound", func(t *testing.T) {
		ca := newWarmCache(t, newFullClient(t))
		if _, err := ca.LoadInstructions("missing"); !errors.Is(err, skill.ErrSkillNotFound) {
			t.Fatalf("LoadInstructions() error = %v, want errors.Is %v", err, skill.ErrSkillNotFound)
		}
	})

	t.Run("download failure is propagated", func(t *testing.T) {
		boom := errors.New("registry down")
		c := newFullClient(t)
		c.zipErr[revAlpha] = boom
		ca := newWarmCache(t, c)

		if _, err := ca.LoadInstructions("alpha"); !errors.Is(err, boom) {
			t.Fatalf("LoadInstructions() error = %v, want errors.Is %v", err, boom)
		}
	})
}

func TestCacheLoadResource(t *testing.T) {
	t.Run("returns the resource content", func(t *testing.T) {
		ca := newWarmCache(t, newFullClient(t))

		got, err := ca.LoadResource("alpha", "assets/a.txt")
		if err != nil {
			t.Fatalf("LoadResource() error = %v", err)
		}
		if got != "alpha-asset" {
			t.Errorf("content = %q, want %q", got, "alpha-asset")
		}
	})

	t.Run("unknown skill returns ErrSkillNotFound", func(t *testing.T) {
		ca := newWarmCache(t, newFullClient(t))
		if _, err := ca.LoadResource("missing", "assets/a.txt"); !errors.Is(err, skill.ErrSkillNotFound) {
			t.Fatalf("LoadResource() error = %v, want errors.Is %v", err, skill.ErrSkillNotFound)
		}
	})

	// A missing resource within a known skill currently returns a generic error,
	// not skill.ErrResourceNotFound (see review item M1). Pin that an error is
	// returned; a fix for M1 should tighten this to the sentinel.
	t.Run("missing resource returns an error", func(t *testing.T) {
		ca := newWarmCache(t, newFullClient(t))
		if _, err := ca.LoadResource("alpha", "nope.txt"); err == nil {
			t.Fatal("LoadResource() error = nil, want an error for a missing resource")
		}
	})
}

func TestCacheListResources(t *testing.T) {
	t.Run("filters resources by prefix", func(t *testing.T) {
		ca := newWarmCache(t, newFullClient(t))

		got, err := ca.ListResources("alpha", "assets/")
		if err != nil {
			t.Fatalf("ListResources() error = %v", err)
		}
		if diff := cmp.Diff([]string{"assets/a.txt"}, got); diff != "" {
			t.Errorf("ListResources diff (-want +got):\n%s", diff)
		}
	})

	t.Run("unknown skill returns ErrSkillNotFound", func(t *testing.T) {
		ca := newWarmCache(t, newFullClient(t))
		if _, err := ca.ListResources("missing", "assets/"); !errors.Is(err, skill.ErrSkillNotFound) {
			t.Fatalf("ListResources() error = %v, want errors.Is %v", err, skill.ErrSkillNotFound)
		}
	})

	t.Run("download failure is propagated", func(t *testing.T) {
		boom := errors.New("registry down")
		c := newFullClient(t)
		c.zipErr[revAlpha] = boom
		ca := newWarmCache(t, c)

		if _, err := ca.ListResources("alpha", "assets/"); !errors.Is(err, boom) {
			t.Fatalf("ListResources() error = %v, want errors.Is %v", err, boom)
		}
	})

	// The "." subpath currently matches nothing (see review item M3): entries are
	// stored under their archive paths ("assets/..."), none of which start with
	// ".". WithCompletePreloadSource relies on ListResources(name, ".") listing
	// everything, so this is a known gap, pinned here as current behavior.
	t.Run("dot subpath currently lists nothing", func(t *testing.T) {
		ca := newWarmCache(t, newFullClient(t))

		got, err := ca.ListResources("alpha", ".")
		if err != nil {
			t.Fatalf("ListResources() error = %v", err)
		}
		if len(got) != 0 {
			t.Errorf(`ListResources(name, ".") = %v, want empty (current behavior)`, got)
		}
	})
}

func TestCacheReset(t *testing.T) {
	t.Run("reset swaps in the newly fetched skill set", func(t *testing.T) {
		c := newFullClient(t)
		ca := newWarmCache(t, c)

		// The registry now lists only alpha.
		c.mu.Lock()
		c.listed = []*agentregistry.Skill{{Name: nameAlpha, DefaultRevision: revAlpha}}
		c.mu.Unlock()

		if err := ca.reset(); err != nil {
			t.Fatalf("reset() error = %v", err)
		}

		got, err := ca.ListFrontmatters()
		if err != nil {
			t.Fatalf("ListFrontmatters() error = %v", err)
		}
		if diff := cmp.Diff([]string{"alpha"}, frontmatterNames(got)); diff != "" {
			t.Errorf("after reset, frontmatter names diff (-want +got):\n%s", diff)
		}
	})

	t.Run("a failed reset keeps the previous cache", func(t *testing.T) {
		c := newFullClient(t)
		ca := newWarmCache(t, c)

		boom := errors.New("registry down")
		c.mu.Lock()
		c.listErr = boom
		c.mu.Unlock()

		if err := ca.reset(); !errors.Is(err, boom) {
			t.Fatalf("reset() error = %v, want errors.Is %v", err, boom)
		}

		// The old two-skill cache must still serve reads.
		got, err := ca.ListFrontmatters()
		if err != nil {
			t.Fatalf("ListFrontmatters() error = %v", err)
		}
		if diff := cmp.Diff([]string{"alpha", "beta"}, frontmatterNames(got)); diff != "" {
			t.Errorf("after a failed reset, frontmatter names diff (-want +got):\n%s", diff)
		}
	})
}

func TestCacheWarmUp(t *testing.T) {
	t.Run("failure is returned", func(t *testing.T) {
		boom := errors.New("registry down")
		ca := &cache{
			client:                  &fullClient{listErr: boom},
			frontmattersParallelism: 4,
			updateInterval:          time.Hour,
			stop:                    make(chan struct{}),
		}
		if err := ca.WarmUp(); !errors.Is(err, boom) {
			t.Fatalf("WarmUp() error = %v, want errors.Is %v", err, boom)
		}
	})

	// ListSkills succeeds but a per-skill GetSkill fails, so the error surfaces
	// from the frontmatter-loading stage rather than the skill-listing one.
	t.Run("frontmatter load failure is returned", func(t *testing.T) {
		boom := errors.New("skill fetch failed")
		c := newFullClient(t)
		c.getErr[nameAlpha] = boom
		ca := &cache{
			client:                  c,
			frontmattersParallelism: 4,
			updateInterval:          time.Hour,
			stop:                    make(chan struct{}),
		}
		if err := ca.WarmUp(); !errors.Is(err, boom) {
			t.Fatalf("WarmUp() error = %v, want errors.Is %v", err, boom)
		}
	})
}

func TestRunPeriodicalReloads(t *testing.T) {
	t.Run("ticks trigger reloads and stop ends the loop", func(t *testing.T) {
		c := newFullClient(t)
		c.onList = make(chan struct{}, 8)
		ca := &cache{
			client:                  c,
			frontmattersParallelism: 4,
			updateInterval:          10 * time.Millisecond,
			stop:                    make(chan struct{}),
		}

		done := make(chan struct{})
		go func() {
			ca.runPeriodicalReloads()
			close(done)
		}()

		// Wait for a tick-driven reload (reset calls ListSkills).
		select {
		case <-c.onList:
		case <-time.After(2 * time.Second):
			t.Fatal("no periodic reload within 2s")
		}

		close(ca.stop)
		select {
		case <-done:
		case <-time.After(2 * time.Second):
			t.Fatal("runPeriodicalReloads did not return after stop")
		}
	})

	t.Run("a failing reload does not stop the loop", func(t *testing.T) {
		c := &fullClient{listErr: errors.New("registry down"), onList: make(chan struct{}, 8)}
		ca := &cache{
			client:                  c,
			frontmattersParallelism: 4,
			updateInterval:          10 * time.Millisecond,
			stop:                    make(chan struct{}),
		}

		done := make(chan struct{})
		go func() {
			ca.runPeriodicalReloads()
			close(done)
		}()

		// Two ticks despite every reload failing: the loop keeps going.
		for i := 0; i < 2; i++ {
			select {
			case <-c.onList:
			case <-time.After(2 * time.Second):
				t.Fatalf("periodic reload %d did not run", i+1)
			}
		}

		close(ca.stop)
		select {
		case <-done:
		case <-time.After(2 * time.Second):
			t.Fatal("runPeriodicalReloads did not return after stop")
		}
	})
}

// Concurrent readers of one skill must share a single archive download and stay
// race-free. Run under -race, this drives the cache's RLock read paths and the
// per-skill lazy-load dedup at once.
func TestCacheConcurrentReads(t *testing.T) {
	c := newFullClient(t)
	ca := newWarmCache(t, c)

	const workers = 40
	var wg sync.WaitGroup
	errs := make(chan error, workers)
	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			switch i % 4 {
			case 0:
				got, err := ca.LoadInstructions("alpha")
				if err != nil {
					errs <- fmt.Errorf("LoadInstructions: %w", err)
				} else if got != bodyAlpha {
					errs <- fmt.Errorf("LoadInstructions = %q, want %q", got, bodyAlpha)
				}
			case 1:
				got, err := ca.LoadResource("alpha", "assets/a.txt")
				if err != nil {
					errs <- fmt.Errorf("LoadResource: %w", err)
				} else if got != "alpha-asset" {
					errs <- fmt.Errorf("LoadResource = %q, want %q", got, "alpha-asset")
				}
			case 2:
				got, err := ca.ListResources("alpha", "assets/")
				if err != nil {
					errs <- fmt.Errorf("ListResources: %w", err)
				} else if len(got) != 1 {
					errs <- fmt.Errorf("ListResources = %v, want 1 entry", got)
				}
			case 3:
				fm, err := ca.LoadFrontmatter("alpha")
				if err != nil {
					errs <- fmt.Errorf("LoadFrontmatter: %w", err)
				} else if fm.Name != "alpha" {
					errs <- fmt.Errorf("LoadFrontmatter Name = %q, want alpha", fm.Name)
				}
			}
		}(i)
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Error(err)
	}

	// All instruction/resource reads share one archive, loaded exactly once.
	if _, getZip := c.calls(); getZip != 1 {
		t.Errorf("GetZip called %d times under concurrency, want exactly 1", getZip)
	}
}
