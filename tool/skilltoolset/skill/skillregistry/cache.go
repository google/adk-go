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
	"fmt"
	"log"
	"sync"

	"google.golang.org/adk/v2/tool/skilltoolset/skill"
)

type CacheConfig struct {
	Client Client
}

type Cache interface {
	IntialRead() error
	ListFrontmatters() ([]*skill.Frontmatter, error)
	LoadFrontmatter(name string) (*skill.Frontmatter, error)
	LoadInstructions(name string) (string, error)
	LoadResource(name string, resourcePath string) (string, error)
	ListResources(name string, subpath string) ([]string, error)
}

// NewCache returns a top level cache.
// Top level cache is handling updates and reads. Uses a mutex to synchronization.
// The SkillRegistry provides:
//  1. the informative list of skills without Frontmatters
//  2. frontmatters (per skill)
//  3. zip files (per skill)
//
// General idea is:
//  1. to read all the skills (without Frontmatters) - done by a sequence of API calls (with paging)
//     Reading all the skills should be done upfront, before the first real request
//     Then the list is periodically updated (new skills are created, old are deleted, changed are updated)
//  2. all the initial ...
//
// TODO(kdroste): describe the rest
func NewCache(cfg CacheConfig) (Cache, error) {
	return &topLevelCache{
		mu:                      sync.Mutex{},
		client:                  cfg.Client,
		frontmattersParallelism: 15,
	}, nil
}

type topLevelCache struct {
	mu                      sync.Mutex
	client                  Client
	data                    dataCache
	frontmattersParallelism int
}

func (c *topLevelCache) ListFrontmatters() ([]*skill.Frontmatter, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	res := make([]*skill.Frontmatter, 0)
	for _, sk := range c.data.skills {
		res = append(res, &sk.frontmatter)
	}
	return res, nil
}

// ListResources implements [Cache].
func (c *topLevelCache) ListResources(name string, subpath string) ([]string, error) {
	s, ok := c.data.nameToSkill[name]
	if !ok { // not found
		return nil, skill.ErrSkillNotFound
	}

	res, err := s.listResources(subpath)
	if err != nil {
		return nil, fmt.Errorf("cannot listResources: %w", err)
	}

	return res, nil
}

// LoadFrontmatter implements [Cache].
func (c *topLevelCache) LoadFrontmatter(name string) (*skill.Frontmatter, error) {
	s, ok := c.data.nameToSkill[name]
	if !ok { // not found
		return nil, skill.ErrSkillNotFound
	}
	return &s.frontmatter, nil
}

// TODO: cache
// LoadInstructions implements [Cache].
func (c *topLevelCache) LoadInstructions(name string) (string, error) {
	s, ok := c.data.nameToSkill[name]
	if !ok { // not found
		return "", skill.ErrSkillNotFound
	}

	res, err := s.getResource("SKILL.md")
	if err != nil {
		return "", fmt.Errorf("cannot getResource: %w", err)
	}

	_, instr, err := skill.ParseBytes(res.content)
	if err != nil {
		return "", fmt.Errorf("cannot parse: %w", err)
	}
	return instr, nil
}

// LoadResource implements [Cache].
func (c *topLevelCache) LoadResource(name string, resourcePath string) (string, error) {
	s, ok := c.data.nameToSkill[name]
	if !ok { // not found
		return "", skill.ErrSkillNotFound
	}

	res, err := s.getResource(resourcePath)
	if err != nil {
		return "", fmt.Errorf("cannot getResource: %w", err)
	}
	return string(res.content), nil
}

// InitialRead pre-loads skills.
// The list of skills is preloaded (sequencially). The frontmatters are preloaded (in parallel)
// The zip files are NOT preloaded - including SKILL.md (it means the Instructions are not preloaded as well)
func (c *topLevelCache) IntialRead() error {
	sc, err := loadSkills(c.client)
	if err != nil {
		return fmt.Errorf("cannot loadSkills: %w", err)
	}
	c.data = *sc
	err = loadFrontmatters(c.client, c.frontmattersParallelism, sc)
	if err != nil {
		return fmt.Errorf("cannot loadFrontmatters: %w", err)
	}
	for _, d := range c.data.skills {
		log.Printf("CachedSkill: %+v  FrontMatter: %+v", d.origSkill, d.frontmatter)
		break
	}

	return nil
}

var _ Cache = (*topLevelCache)(nil)
