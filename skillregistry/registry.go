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

// Package skillregistry implements dynamic skill discovery and loading.
// Mirrors the toolregistry pattern: skills live in a Registry, the LLM
// lists candidates via list_skills, and activates one by name via
// load_skill. The instructions from loaded skills are injected into the
// LLM system instruction by SkillsInstructionPlugin so the model has the
// guidance it just chose without paying the context cost up front.
//
// Loaded skill names persist in session state under StateKeyLoadedSkills
// so activation survives across turns within an invocation chain.
package skillregistry

import (
	"errors"
	"fmt"
	"sort"
	"strings"
	"sync"

	"google.golang.org/adk/skill"
)

// StateKeyLoadedSkills is the session-state key under which loaded skill
// names are tracked. Stored as []string. Unprefixed so it persists
// across turns.
const StateKeyLoadedSkills = "_adk_loaded_skills"

// Filter narrows what list_skills returns. Empty fields match everything.
type Filter struct {
	// Query is a case-insensitive substring matched against the skill's
	// name and description.
	Query string

	// Tags requires all listed tags be present in the skill's
	// frontmatter metadata under the "tags" key (when present).
	Tags []string
}

// Builder constructs a skill on first activation. Lets callers register
// cheap metadata without paying for skill loading (parsing markdown,
// loading resources) up front.
type Builder func() (*skill.Skill, error)

type entry struct {
	frontmatter skill.Frontmatter
	build       Builder
	cached      *skill.Skill
}

// Registry is the central catalog of dynamically-loadable skills.
//
// Concurrency: all methods are safe for concurrent use.
type Registry struct {
	mu      sync.RWMutex
	entries map[string]*entry
}

// New constructs an empty Registry.
func New() *Registry { return &Registry{entries: map[string]*entry{}} }

// Register adds a skill factory keyed by frontmatter.Name. Re-registering
// an existing name overwrites the prior entry.
func (r *Registry) Register(fm skill.Frontmatter, build Builder) error {
	if err := fm.Validate(); err != nil {
		return err
	}
	if build == nil {
		return errors.New("skillregistry: Register: Builder must not be nil")
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	r.entries[fm.Name] = &entry{frontmatter: fm, build: build}
	return nil
}

// RegisterSkill is a convenience wrapper for the common case where the
// skill is already constructed. Equivalent to Register with a Builder
// that returns the supplied skill.
func (r *Registry) RegisterSkill(s *skill.Skill) error {
	if s == nil {
		return errors.New("skillregistry: RegisterSkill: skill must not be nil")
	}
	return r.Register(s.Frontmatter, func() (*skill.Skill, error) { return s, nil })
}

// Get returns the skill registered under name, lazily constructing it
// on first access and caching the result.
func (r *Registry) Get(name string) (*skill.Skill, error) {
	r.mu.RLock()
	e, ok := r.entries[name]
	r.mu.RUnlock()
	if !ok {
		return nil, fmt.Errorf("skillregistry: skill %q not found", name)
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if e.cached != nil {
		return e.cached, nil
	}
	s, err := e.build()
	if err != nil {
		return nil, fmt.Errorf("skillregistry: build %q: %w", name, err)
	}
	e.cached = s
	return s, nil
}

// Has reports whether name is registered.
func (r *Registry) Has(name string) bool {
	r.mu.RLock()
	defer r.mu.RUnlock()
	_, ok := r.entries[name]
	return ok
}

// List returns frontmatters for every registered skill matching f,
// sorted by name.
func (r *Registry) List(f Filter) []skill.Frontmatter {
	r.mu.RLock()
	defer r.mu.RUnlock()

	q := strings.ToLower(strings.TrimSpace(f.Query))
	out := make([]skill.Frontmatter, 0, len(r.entries))
	for _, e := range r.entries {
		if !matchesFilter(e.frontmatter, q, f.Tags) {
			continue
		}
		out = append(out, e.frontmatter)
	}
	sort.SliceStable(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out
}

// Names returns the sorted list of registered skill names.
func (r *Registry) Names() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()
	out := make([]string, 0, len(r.entries))
	for k := range r.entries {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

// matchesFilter reports whether fm matches q (case-insensitive substring
// across Name and Description) and contains every tag in requiredTags.
// Tags are extracted from frontmatter.Metadata["tags"] when present.
func matchesFilter(fm skill.Frontmatter, q string, requiredTags []string) bool {
	if q != "" {
		hay := strings.ToLower(fm.Name) + " " + strings.ToLower(fm.Description)
		if !strings.Contains(hay, q) {
			return false
		}
	}
	if len(requiredTags) == 0 {
		return true
	}
	tagSet := map[string]struct{}{}
	if rawTags, ok := fm.Metadata["tags"]; ok {
		switch tags := rawTags.(type) {
		case []string:
			for _, t := range tags {
				tagSet[strings.ToLower(t)] = struct{}{}
			}
		case []any:
			for _, t := range tags {
				if s, ok := t.(string); ok {
					tagSet[strings.ToLower(s)] = struct{}{}
				}
			}
		}
	}
	for _, want := range requiredTags {
		if _, ok := tagSet[strings.ToLower(want)]; !ok {
			return false
		}
	}
	return true
}

// LoadedNames returns the slice of currently-loaded skill names from
// the session state. Coerces []any (from JSON deserialization) to
// []string transparently.
func LoadedNames(state interface {
	Get(key string) (any, error)
}) []string {
	if state == nil {
		return nil
	}
	v, err := state.Get(StateKeyLoadedSkills)
	if err != nil {
		return nil
	}
	switch x := v.(type) {
	case []string:
		return x
	case []any:
		out := make([]string, 0, len(x))
		for _, item := range x {
			if s, ok := item.(string); ok {
				out = append(out, s)
			}
		}
		return out
	}
	return nil
}
