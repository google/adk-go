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
	"regexp"
	"strings"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	agentregistry "google.golang.org/api/agentregistry/v1alpha"

	"google.golang.org/adk/v2/tool/skilltoolset/skill"
	"google.golang.org/adk/v2/tool/skilltoolset/skill/skillregistry"
)

// fakeClient is a minimal in-memory [Client] for exercising the skill and
// frontmatter loading paths. Those paths only call ListSkills, GetSkill and
// ParseSkillName; the remaining methods satisfy the interface and are not
// expected to be called.
type fakeClient struct {
	listed   []*agentregistry.Skill          // ListSkills result
	listErr  error                           // ListSkills error
	skills   map[string]*agentregistry.Skill // GetSkill result, by resource name
	getErr   map[string]error                // GetSkill error, by resource name
	parseErr map[string]error                // ParseSkillName error, by resource name
}

var _ skillregistry.Client = (*fakeClient)(nil)

func (c *fakeClient) ListSkills() ([]*agentregistry.Skill, error) {
	if c.listErr != nil {
		return nil, c.listErr
	}
	return c.listed, nil
}

func (c *fakeClient) GetSkill(name string) (*agentregistry.Skill, error) {
	if err := c.getErr[name]; err != nil {
		return nil, err
	}
	s, ok := c.skills[name]
	if !ok {
		return nil, fmt.Errorf("fakeClient: no skill for the given name")
	}
	return s, nil
}

var skillIDRegex = regexp.MustCompile("projects/([^/]+)/locations/([^/]+)/skills/([^/]+)")

// ParseSkillName returns the configured error for name, otherwise parses it with
// the same regex the real client uses, so the derived prefix is realistic.
func (c *fakeClient) ParseSkillName(name string) (projectID, location, skillName string, resErr error) {
	if err := c.parseErr[name]; err != nil {
		return "", "", "", err
	}
	m := skillIDRegex.FindStringSubmatch(name)
	if len(m) != 4 {
		return "", "", "", fmt.Errorf("fakeClient: cannot parse the given name")
	}
	return m[1], m[2], m[3], nil
}

// The methods below are unused by the loading paths.
func (c *fakeClient) ListFrontmatters() ([]*agentregistry.Frontmatter, error) { return nil, nil }

func (c *fakeClient) GetRevision(string) (*agentregistry.SkillRevision, error) {
	return nil, nil
}

func (c *fakeClient) GetZip(string) (*zip.Reader, error) { return nil, nil }

func (c *fakeClient) FindFrontmatters(string) ([]*agentregistry.Frontmatter, error) { return nil, nil }
func (c *fakeClient) ResourceID(name string) string                                 { return name }

func TestNewDataCache(t *testing.T) {
	c := &fakeClient{}

	t.Run("indexes every skill by resource name", func(t *testing.T) {
		in := []*agentregistry.Skill{
			{Name: "projects/p/locations/global/skills/alpha", DefaultRevision: "rev-a"},
			{Name: "projects/p/locations/global/skills/beta", DefaultRevision: "rev-b"},
		}

		dc := newDataCache(c, in)

		if len(dc.skills) != len(in) {
			t.Fatalf("len(skills) = %d, want %d", len(dc.skills), len(in))
		}
		if len(dc.nameToSkill) != len(in) {
			t.Errorf("len(nameToSkill) = %d, want %d", len(dc.nameToSkill), len(in))
		}
		for i, want := range in {
			cs := dc.skills[i]
			if cs.origSkill.Name != want.Name {
				t.Errorf("skills[%d].origSkill.Name = %q, want %q", i, cs.origSkill.Name, want.Name)
			}
			// Resources are fetched by revision, so it must survive the copy.
			if cs.origSkill.DefaultRevision != want.DefaultRevision {
				t.Errorf("skills[%d].origSkill.DefaultRevision = %q, want %q", i, cs.origSkill.DefaultRevision, want.DefaultRevision)
			}
			if got := dc.nameToSkill[want.Name]; got != cs {
				t.Errorf("nameToSkill[%q] is not skills[%d]", want.Name, i)
			}
		}
	})

	// loadFrontmatters writes into nameToSkill, so it must be usable even when
	// the registry has no skills.
	t.Run("no skills", func(t *testing.T) {
		dc := newDataCache(c, nil)

		if len(dc.skills) != 0 {
			t.Errorf("len(skills) = %d, want 0", len(dc.skills))
		}
		if dc.nameToSkill == nil {
			t.Error("nameToSkill is nil, want an empty map")
		}
	})
}

func TestLoadSkills(t *testing.T) {
	names := []string{
		"projects/p/locations/global/skills/alpha",
		"projects/p/locations/global/skills/beta",
	}

	t.Run("success", func(t *testing.T) {
		c := &fakeClient{}
		for _, n := range names {
			c.listed = append(c.listed, &agentregistry.Skill{Name: n})
		}

		before := time.Now()
		dc, err := loadSkills(c)
		after := time.Now()
		if err != nil {
			t.Fatalf("loadSkills() error = %v", err)
		}

		var got []string
		for _, cs := range dc.skills {
			got = append(got, cs.origSkill.Name)
		}
		if diff := cmp.Diff(names, got); diff != "" {
			t.Errorf("skill names diff (-want +got):\n%s", diff)
		}
		for _, n := range names {
			if _, ok := dc.nameToSkill[n]; !ok {
				t.Errorf("nameToSkill is missing %q", n)
			}
		}
		if dc.readingSkillsStart.Before(before) || dc.readingSkillsEnd.After(after) || dc.readingSkillsEnd.Before(dc.readingSkillsStart) {
			t.Errorf("reading window [%v, %v] is not an ordered window inside the call [%v, %v]",
				dc.readingSkillsStart, dc.readingSkillsEnd, before, after)
		}
	})

	t.Run("no skills", func(t *testing.T) {
		dc, err := loadSkills(&fakeClient{})
		if err != nil {
			t.Fatalf("loadSkills() error = %v", err)
		}
		if len(dc.skills) != 0 {
			t.Errorf("len(skills) = %d, want 0", len(dc.skills))
		}
	})

	t.Run("ListSkills failure is propagated", func(t *testing.T) {
		boom := errors.New("registry down")

		dc, err := loadSkills(&fakeClient{listErr: boom})
		if !errors.Is(err, boom) {
			t.Fatalf("loadSkills() error = %v, want errors.Is %v", err, boom)
		}
		if dc != nil {
			t.Errorf("loadSkills() returned a non-nil cache on error")
		}
	})
}

func TestFrontmatterWorker(t *testing.T) {
	const name = "projects/p/locations/global/skills/alpha"
	boom := errors.New("boom")

	tests := []struct {
		name    string
		client  *fakeClient
		wantErr error // expected via errors.Is; nil means the success path
	}{
		{
			name: "success",
			client: &fakeClient{
				skills: map[string]*agentregistry.Skill{
					name: {Name: name, Frontmatter: &agentregistry.Frontmatter{
						Name:          "whatever-the-file-says",
						Description:   "Alpha skill.",
						License:       "Apache-2.0",
						Compatibility: "needs bash",
						Metadata:      map[string]string{"version": "1.2"},
					}},
				},
			},
		},
		{
			name:    "GetSkill error",
			client:  &fakeClient{getErr: map[string]error{name: boom}},
			wantErr: boom,
		},
		{
			name: "ParseSkillName error",
			client: &fakeClient{
				skills: map[string]*agentregistry.Skill{
					name: {Name: name, Frontmatter: &agentregistry.Frontmatter{Description: "Alpha."}},
				},
				parseErr: map[string]error{name: boom},
			},
			wantErr: boom,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			sk := newCachedSkill(tt.client, &agentregistry.Skill{Name: name})
			toProcess := make(chan *cachedSkill, 1)
			done := make(chan frontmatterReadDone, 1)
			toProcess <- sk
			close(toProcess)

			// With toProcess closed and done buffered, the worker processes the
			// single skill and returns without blocking.
			frontmatterWorker(0, tt.client, toProcess, done)
			res := <-done

			if tt.wantErr != nil {
				if !errors.Is(res.err, tt.wantErr) {
					t.Fatalf("frontmatterWorker err = %v, want errors.Is %v", res.err, tt.wantErr)
				}
				if res.prefixedSkillName != "" {
					t.Errorf("prefixedSkillName = %q on error, want empty", res.prefixedSkillName)
				}
				// The skill name is customer data and must stay out of the error.
				if strings.Contains(res.err.Error(), name) {
					t.Errorf("error message leaks the raw skill name: %v", res.err)
				}
				return
			}

			if res.err != nil {
				t.Fatalf("frontmatterWorker err = %v, want nil", res.err)
			}
			if res.prefixedSkillName != "alpha" {
				t.Errorf("prefixedSkillName = %q, want %q", res.prefixedSkillName, "alpha")
			}
			if res.skill != sk {
				t.Errorf("done carried a different *cachedSkill than the input")
			}
			// The registry-derived prefix overrides the name from the file's
			// frontmatter; every other field is copied through from GetSkill.
			want := skill.Frontmatter{
				Name:          "alpha",
				Description:   "Alpha skill.",
				License:       "Apache-2.0",
				Compatibility: "needs bash",
				Metadata:      map[string]string{"version": "1.2"},
			}
			if diff := cmp.Diff(want, sk.frontmatter); diff != "" {
				t.Errorf("frontmatter diff (-want +got):\n%s", diff)
			}
		})
	}
}

// loadFrontmatters reads exactly one result per skill, so a worker must keep
// draining the queue after a failure instead of abandoning the rest.
func TestFrontmatterWorker_ReportsEverySkill(t *testing.T) {
	names := []string{
		"projects/p/locations/global/skills/alpha",
		"projects/p/locations/global/skills/beta",
		"projects/p/locations/global/skills/gamma",
	}
	c, dc := newFrontmatterTestCache(names)
	c.getErr[names[0]] = errors.New("boom")
	c.parseErr[names[1]] = errors.New("bad name")

	toProcess := make(chan *cachedSkill, len(dc.skills))
	done := make(chan frontmatterReadDone, len(dc.skills))
	for _, sk := range dc.skills {
		toProcess <- sk
	}
	close(toProcess)

	frontmatterWorker(0, c, toProcess, done)
	close(done)

	var failed, succeeded int
	for res := range done {
		if res.err != nil {
			failed++
		} else {
			succeeded++
		}
	}
	if failed != 2 || succeeded != 1 {
		t.Errorf("got %d failed and %d succeeded results, want 2 and 1", failed, succeeded)
	}
}

// newFrontmatterTestCache builds a fakeClient and a dataCache for the given
// resource names. Each skill resolves through GetSkill to a frontmatter whose
// description is "desc:"+name.
func newFrontmatterTestCache(names []string) (*fakeClient, *dataCache) {
	c := &fakeClient{
		skills:   make(map[string]*agentregistry.Skill),
		getErr:   make(map[string]error),
		parseErr: make(map[string]error),
	}
	listed := make([]*agentregistry.Skill, 0, len(names))
	for _, n := range names {
		c.skills[n] = &agentregistry.Skill{
			Name:        n,
			Frontmatter: &agentregistry.Frontmatter{Description: "desc:" + n},
		}
		listed = append(listed, &agentregistry.Skill{Name: n})
	}
	return c, newDataCache(c, listed)
}

func TestLoadFrontmatters(t *testing.T) {
	names := []string{
		"projects/p/locations/global/skills/alpha",
		"projects/p/locations/global/skills/beta",
		"projects/p/locations/global/skills/gamma",
	}

	// Fewer, as many, and more workers than skills. More than one worker
	// exercises the parallel path under -race.
	for _, nWorkers := range []int{1, len(names), 10} {
		t.Run(fmt.Sprintf("success with %d workers", nWorkers), func(t *testing.T) {
			c, dc := newFrontmatterTestCache(names)

			if err := loadFrontmatters(c, nWorkers, dc); err != nil {
				t.Fatalf("loadFrontmatters() error = %v", err)
			}

			for i, n := range names {
				_, _, prefix, err := c.ParseSkillName(n)
				if err != nil {
					t.Fatalf("ParseSkillName(%q) error = %v", n, err)
				}
				cs, ok := dc.nameToSkill[prefix]
				if !ok {
					t.Fatalf("nameToSkill is missing the prefixed key %q", prefix)
				}
				if cs != dc.skills[i] {
					t.Errorf("nameToSkill[%q] is not skills[%d]", prefix, i)
				}
				if cs.frontmatter.Name != prefix {
					t.Errorf("frontmatter.Name = %q, want %q", cs.frontmatter.Name, prefix)
				}
				if want := "desc:" + n; cs.frontmatter.Description != want {
					t.Errorf("frontmatter.Description = %q, want %q", cs.frontmatter.Description, want)
				}
			}
		})
	}

	t.Run("no skills", func(t *testing.T) {
		c, dc := newFrontmatterTestCache(nil)

		if err := loadFrontmatters(c, 2, dc); err != nil {
			t.Fatalf("loadFrontmatters() error = %v", err)
		}
		if len(dc.nameToSkill) != 0 {
			t.Errorf("len(nameToSkill) = %d, want 0", len(dc.nameToSkill))
		}
	})

	t.Run("GetSkill failure is propagated", func(t *testing.T) {
		boom := errors.New("registry down")
		c, dc := newFrontmatterTestCache(names)
		c.getErr[names[1]] = boom // one of several skills fails

		if err := loadFrontmatters(c, 2, dc); !errors.Is(err, boom) {
			t.Fatalf("loadFrontmatters() error = %v, want errors.Is %v", err, boom)
		}
	})

	t.Run("ParseSkillName failure is propagated", func(t *testing.T) {
		boom := errors.New("bad name")
		c, dc := newFrontmatterTestCache(names)
		c.parseErr[names[2]] = boom

		if err := loadFrontmatters(c, 2, dc); !errors.Is(err, boom) {
			t.Fatalf("loadFrontmatters() error = %v, want errors.Is %v", err, boom)
		}
	})

	// A non-positive worker count would otherwise block forever on the done
	// channel with skills to process, so it must be rejected up front.
	for _, nWorkers := range []int{0, -1} {
		t.Run(fmt.Sprintf("%d workers is rejected", nWorkers), func(t *testing.T) {
			c, dc := newFrontmatterTestCache(names)

			if err := loadFrontmatters(c, nWorkers, dc); err == nil {
				t.Errorf("loadFrontmatters(nWorkers=%d) = nil error, want error", nWorkers)
			}
		})
	}
}
