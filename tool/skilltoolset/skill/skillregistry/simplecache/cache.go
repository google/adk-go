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

// simplecache provides cache functionality over SkillRegistry
// Cache is handling updates and reads. Uses a mutex for the synchronization.
// The SkillRegistry provides:
//  1. the informative list of skills without Frontmatters
//  2. frontmatters (per skill)
//  3. zip files (per skill)
//
// General idea is:
//  1. to read all the skills (without Frontmatters) - done by a sequence of API calls (with paging) - the whole set is cached
//  2. to read all the frontmatters - done by fetching per skill in parallel - the whole set is cached
//  3. to fetch zip files on demand - whatever has been downloaded is cached
//
// Important: General assumption is that the size of the cached objects fits to the memory
//
// Updating the cache:
//
//	periodically the cache is updated:
//	 1. while the cache is being updated ([runPeriodicalReloads]), the requests are served using the old cache
//	 2. update process:
//	     a. the new list of skills is fetched (sequence due to paging)
//	     b. all frontmatters are fetched (in parallel)
//	     c. the new cache overrides the old one (the cached files are discarded)
package simplecache

import (
	"fmt"
	"log"
	"sync"
	"time"

	"google.golang.org/adk/v2/tool/skilltoolset/skill"
	"google.golang.org/adk/v2/tool/skilltoolset/skill/skillregistry"
)

// CacheConfig contains parametres for the cache layer
type CacheConfig struct {
	// the underlying client to Skill Registry
	Client skillregistry.Client
	// UpdateInterval controlls how often the cache is refreshed.
	// Defaults to 1h if set to 0
	UpdateInterval time.Duration
	// NewCache will do WarmUp iff SkipWarmUp is false
	SkipWarmUp bool
}

type cache struct {
	client                  skillregistry.Client // the underlying client to Skill Registry
	mu                      sync.RWMutex         // used to synch reada and writes to the cache.
	preloadMutex            sync.Mutex           // for cache updates only - used to separate [preload]
	data                    dataCache            // cache data
	frontmattersParallelism int                  // controlls the amount of workers reading frontmatters in parallel
	updateInterval          time.Duration        // controlls how often the cache is refreshed
	stop                    chan struct{}        // stop is controlling go routine runPeriodicalReset, allowing to stop the ticker
}

// NewCache returns a top level cache.
func NewCache(cfg CacheConfig) (skillregistry.Cache, error) {
	updateInterval := cfg.UpdateInterval
	if updateInterval == 0 {
		updateInterval = 1 * time.Hour
	}
	c := &cache{
		mu:                      sync.RWMutex{},
		client:                  cfg.Client,
		frontmattersParallelism: 15,
		updateInterval:          cfg.UpdateInterval,
		stop:                    make(chan struct{}),
		preloadMutex:            sync.Mutex{},
	}
	if !cfg.SkipWarmUp {
		err := c.WarmUp()
		if err != nil {
			return nil, fmt.Errorf("cannot WarmUp: %w", err)
		}
	}
	go c.runPeriodicalReloads()
	return c, nil
}

// runPeriodicalReloads starts periodic cache cleanup and preload
func (c *cache) runPeriodicalReloads() {
	ticker := time.NewTicker(c.updateInterval)
	for {
		select {
		case <-ticker.C:
			err := c.reset()
			if err != nil {
				log.Printf("runPeriodicalReset: cannot reset: %v", err)
			}
		case <-c.stop:
			ticker.Stop()
			return
		}
	}
}

// ListFrontmatters implements [Cache].
// Returns the list of all frontmatters. In case of ADK with SkillRegistry it requires
// caching date - otherwise you would need to query the list first and then query each skill.
func (c *cache) ListFrontmatters() ([]*skill.Frontmatter, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	res := make([]*skill.Frontmatter, 0)
	for _, sk := range c.data.skills {
		res = append(res, &sk.frontmatter)
	}
	return res, nil
}

// ListResources implements [Cache].
func (c *cache) ListResources(name string, subpath string) ([]string, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()
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
func (c *cache) LoadFrontmatter(name string) (*skill.Frontmatter, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	s, ok := c.data.nameToSkill[name]
	if !ok { // not found
		return nil, skill.ErrSkillNotFound
	}
	return &s.frontmatter, nil
}

// LoadInstructions implements [Cache].
func (c *cache) LoadInstructions(name string) (string, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	s, ok := c.data.nameToSkill[name]
	if !ok { // not found
		return "", skill.ErrSkillNotFound
	}

	instr, err := s.getInstruction()
	if err != nil {
		return "", fmt.Errorf("cannot getInstruction: %w", err)
	}

	return instr, nil
}

// LoadResource implements [Cache].
// The resources are not preloaded. SkillRegistry provides a zip archive.
// Zip has to be unpacked. All the content of the archive goes to the cache.
func (c *cache) LoadResource(name string, resourcePath string) (string, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

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

// WarmUp pre-loads skills.
// The list of skills is preloaded (sequencially). The frontmatters are preloaded (in parallel)
// The zip files are NOT preloaded - including SKILL.md (it means the Instructions are not preloaded as well)
func (c *cache) WarmUp() error {
	return c.preload()
}

// preload clears the cache, re-loading skills and frontmatters. The resources already in cache are forgotten.
func (c *cache) preload() error {
	c.preloadMutex.Lock()
	defer c.preloadMutex.Unlock()

	newCache, err := loadSkills(c.client)
	if err != nil {
		return fmt.Errorf("cannot loadSkills: %w", err)
	}

	err = loadFrontmatters(c.client, c.frontmattersParallelism, newCache)
	if err != nil {
		return fmt.Errorf("cannot loadFrontmatters: %w", err)
	}

	// got everything, swap
	// lock mu only now, for the swap
	c.mu.Lock()
	defer c.mu.Unlock()
	c.data = *newCache

	return nil
}

func (c *cache) reset() error {
	return c.preload()
}

var _ skillregistry.Cache = (*cache)(nil)
