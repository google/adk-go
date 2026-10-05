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
	"errors"
	"fmt"
	"strings"
	"sync"
	"testing"

	agentregistry "google.golang.org/api/agentregistry/v1alpha"

	"google.golang.org/adk/v2/tool/skilltoolset/skill"
)

func TestNewCachedSkill(t *testing.T) {
	c := &zipClientStub{}
	in := &agentregistry.Skill{
		Name:            "projects/p/locations/global/skills/alpha",
		DefaultRevision: "rev-a",
	}

	cs := newCachedSkill(c, in)

	if cs.client != c {
		t.Error("client was not stored")
	}
	if cs.origSkill.Name != in.Name {
		t.Errorf("origSkill.Name = %q, want %q", cs.origSkill.Name, in.Name)
	}
	// The revision drives every resource fetch, so it must survive construction.
	if cs.origSkill.DefaultRevision != in.DefaultRevision {
		t.Errorf("origSkill.DefaultRevision = %q, want %q", cs.origSkill.DefaultRevision, in.DefaultRevision)
	}
	if cs.resources == nil {
		t.Fatal("resources is nil")
	}
	if cs.resources.client != c {
		t.Error("resources was not wired to the provided client")
	}
	if cs.gotInstructions {
		t.Error("gotInstructions = true on a fresh skill, want false")
	}
	if cs.instructions != "" {
		t.Errorf("instructions = %q on a fresh skill, want empty", cs.instructions)
	}

	// origSkill is a value copy (origSkill: *s): later mutations of the caller's
	// struct must not leak into the cached skill.
	in.Name = "mutated"
	in.DefaultRevision = "mutated-rev"
	if cs.origSkill.Name == "mutated" || cs.origSkill.DefaultRevision == "mutated-rev" {
		t.Errorf("origSkill aliases the caller's *Skill; got %+v", cs.origSkill)
	}
}

func TestCachedSkillResourceDelegation(t *testing.T) {
	entries := []zipEntry{
		{name: "SKILL.md", content: []byte("# skill")},
		{name: "assets/a.txt", content: []byte("alpha")},
		{name: "assets/b.txt", content: []byte("beta")},
	}

	t.Run("getResource forwards DefaultRevision and returns the content", func(t *testing.T) {
		c := &zipClientStub{entries: entries}
		cs := newCachedSkill(c, &agentregistry.Skill{DefaultRevision: "rev-42"})

		res, err := cs.getResource("assets/a.txt")
		if err != nil {
			t.Fatalf("getResource() error = %v", err)
		}
		if string(res.content) != "alpha" {
			t.Errorf("content = %q, want %q", res.content, "alpha")
		}
		if got := c.lastRev(); got != "rev-42" {
			t.Errorf("GetZip called with rev %q, want %q", got, "rev-42")
		}
	})

	t.Run("listResources forwards DefaultRevision and filters by prefix", func(t *testing.T) {
		c := &zipClientStub{entries: entries}
		cs := newCachedSkill(c, &agentregistry.Skill{DefaultRevision: "rev-7"})

		got, err := cs.listResources("assets/")
		if err != nil {
			t.Fatalf("listResources() error = %v", err)
		}
		if len(got) != 2 {
			t.Errorf("listResources = %v, want the 2 assets", got)
		}
		if rev := c.lastRev(); rev != "rev-7" {
			t.Errorf("GetZip called with rev %q, want %q", rev, "rev-7")
		}
	})
}

func TestCachedSkillGetInstruction(t *testing.T) {
	const body = "# Alpha instructions\n"
	skillMD, err := skill.Build(&skill.Frontmatter{Name: "alpha", Description: "Alpha skill."}, body)
	if err != nil {
		t.Fatalf("skill.Build() error = %v", err)
	}

	t.Run("parses SKILL.md once and caches the instruction", func(t *testing.T) {
		c := &zipClientStub{entries: []zipEntry{{name: "SKILL.md", content: skillMD}}}
		cs := newCachedSkill(c, &agentregistry.Skill{DefaultRevision: "rev-1"})

		got, err := cs.getInstruction()
		if err != nil {
			t.Fatalf("getInstruction() error = %v", err)
		}
		if got != body {
			t.Errorf("instruction = %q, want %q", got, body)
		}
		if !cs.gotInstructions {
			t.Error("gotInstructions = false after a successful parse, want true")
		}
		if cs.instructions != body {
			t.Errorf("cached instructions = %q, want %q", cs.instructions, body)
		}

		// A second call is served from cache with no extra download.
		got2, err := cs.getInstruction()
		if err != nil {
			t.Fatalf("second getInstruction() error = %v", err)
		}
		if got2 != body {
			t.Errorf("second instruction = %q, want %q", got2, body)
		}
		if c.calls() != 1 {
			t.Errorf("GetZip called %d times, want 1", c.calls())
		}
	})

	t.Run("cache hit returns without touching the client", func(t *testing.T) {
		// A client whose download always fails; the preset cache must win.
		c := &zipClientStub{zipErr: errors.New("GetZip must not be called")}
		cs := newCachedSkill(c, &agentregistry.Skill{DefaultRevision: "rev-1"})
		cs.gotInstructions = true
		cs.instructions = "preset"

		got, err := cs.getInstruction()
		if err != nil {
			t.Fatalf("getInstruction() error = %v", err)
		}
		if got != "preset" {
			t.Errorf("instruction = %q, want %q", got, "preset")
		}
		if c.calls() != 0 {
			t.Errorf("GetZip called %d times on a cache hit, want 0", c.calls())
		}
	})

	t.Run("missing SKILL.md is reported as a getResource error", func(t *testing.T) {
		c := &zipClientStub{entries: []zipEntry{{name: "assets/a.txt", content: []byte("x")}}}
		cs := newCachedSkill(c, &agentregistry.Skill{DefaultRevision: "rev-1"})

		_, err := cs.getInstruction()
		if err == nil {
			t.Fatal("getInstruction() error = nil, want an error for a missing SKILL.md")
		}
		if !strings.Contains(err.Error(), "cannot getResource") {
			t.Errorf("error = %v, want it to mention %q", err, "cannot getResource")
		}
		if cs.gotInstructions {
			t.Error("gotInstructions = true after a failed load, want false")
		}
	})

	t.Run("GetZip failure is propagated", func(t *testing.T) {
		boom := errors.New("registry down")
		c := &zipClientStub{entries: []zipEntry{{name: "SKILL.md", content: skillMD}}, zipErr: boom}
		cs := newCachedSkill(c, &agentregistry.Skill{DefaultRevision: "rev-1"})

		if _, err := cs.getInstruction(); !errors.Is(err, boom) {
			t.Fatalf("getInstruction() error = %v, want errors.Is %v", err, boom)
		}
	})

	t.Run("invalid frontmatter is reported as a parse error", func(t *testing.T) {
		c := &zipClientStub{entries: []zipEntry{{name: "SKILL.md", content: []byte("no frontmatter here")}}}
		cs := newCachedSkill(c, &agentregistry.Skill{DefaultRevision: "rev-1"})

		_, err := cs.getInstruction()
		if err == nil {
			t.Fatal("getInstruction() error = nil, want a parse error")
		}
		if !strings.Contains(err.Error(), "cannot parse") {
			t.Errorf("error = %v, want it to mention %q", err, "cannot parse")
		}
		if cs.gotInstructions {
			t.Error("gotInstructions = true after a parse failure, want false")
		}
	})
}

// Concurrent getInstruction callers must parse SKILL.md exactly once and agree
// on the result. Run under -race, this exercises the locking around the
// instruction cache.
func TestCachedSkillGetInstructionConcurrent(t *testing.T) {
	const body = "# Concurrent instructions\n"
	skillMD, err := skill.Build(&skill.Frontmatter{Name: "alpha", Description: "Alpha."}, body)
	if err != nil {
		t.Fatalf("skill.Build() error = %v", err)
	}
	c := &zipClientStub{entries: []zipEntry{{name: "SKILL.md", content: skillMD}}}
	cs := newCachedSkill(c, &agentregistry.Skill{DefaultRevision: "rev-1"})

	const workers = 32
	var wg sync.WaitGroup
	errs := make(chan error, workers)
	for range workers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			got, err := cs.getInstruction()
			if err != nil {
				errs <- err
				return
			}
			if got != body {
				errs <- fmt.Errorf("instruction = %q, want %q", got, body)
			}
		}()
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
