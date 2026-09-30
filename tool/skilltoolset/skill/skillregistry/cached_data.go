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
	"time"

	"google.golang.org/adk/v2/tool/skilltoolset/skill"
	agentregistry "google.golang.org/api/agentregistry/v1alpha"
)

type dataCache struct {
	skills      []*cachedSkill
	idToSkill   map[string]*cachedSkill
	nameToSkill map[string]*cachedSkill

	readingSkillsStart time.Time
	readingSkillsEnd   time.Time
}

// newDataCache creates dataCache using the provided list of [agentregistry.Skill].
func newDataCache(client Client, skills []*agentregistry.Skill) *dataCache {
	dc := &dataCache{
		skills:      make([]*cachedSkill, 0, len(skills)),
		idToSkill:   make(map[string]*cachedSkill),
		nameToSkill: make(map[string]*cachedSkill),
	}
	dc.skills = make([]*cachedSkill, 0, len(skills))
	dc.idToSkill = make(map[string]*cachedSkill)
	dc.nameToSkill = make(map[string]*cachedSkill)

	// add all skills to the internal structures
	for _, sk := range skills {
		cs := newCachedSkill(client, sk)

		dc.skills = append(dc.skills, cs)
		dc.idToSkill[sk.Uid] = cs
		dc.nameToSkill[sk.Name] = cs
	}

	return dc
}

// loadSkills loads the full list of skills (sequencial due to paging) and returns a new dataCache containing the skills
// An error is returned when getting the list of skills fails.
// No frontmatters are loaded. No zip files are loaded.
func loadSkills(c Client) (*dataCache, error) {
	log.Printf("loadSkills")
	start := time.Now()

	skills, err := c.ListSkills()
	if err != nil {
		return nil, fmt.Errorf("cannot ListSkills: %w", err)
	}

	dc := newDataCache(c, skills)
	dc.readingSkillsStart = start
	dc.readingSkillsEnd = time.Now()

	log.Printf("loadSkills done")
	return dc, nil
}

// loadFrontmatters loads fronmatters for skills in dataCache.
// It's being done in parallel using nWorkers
func loadFrontmatters(c Client, nWorkers int, dc *dataCache) error {
	log.Printf("loadFrontmatters")

	toProcess := make(chan *cachedSkill, len(dc.skills))
	done := make(chan struct{}, len(dc.skills))

	for i := 0; i < nWorkers; i++ {
		go frontmatterWorker(i, c, toProcess, done)
	}

	for _, sk := range dc.skills {
		toProcess <- sk
	}
	close(toProcess)

	// return for all workers to finish
	for i := 0; i < len(dc.skills); i++ {
		<-done
	}
	log.Printf("loadFrontmatters done")
	return nil
}

// frontmatterWorker is used to load frontmatters in parallel. It uses the provided
// client to load frontmatter.
// TODO: error handling
func frontmatterWorker(i int, c Client, toProcess chan *cachedSkill, done chan struct{}) {
	for sk := range toProcess {
		// log.Printf("frontmatterWorker %d processing %v", i, sk.origSkill.Name)
		s, err := c.GetSkill(sk.origSkill.Name)
		if err != nil {
			log.Printf("frontmatterWorker %d failed for %v: %v", i, sk.origSkill.Name, err)
			continue
		}
		sk.frontmatter = skill.Frontmatter{
			Name:          sk.origSkill.Name,
			Description:   s.Frontmatter.Description,
			License:       s.Frontmatter.License,
			Compatibility: s.Frontmatter.Compatibility,
			Metadata:      s.Frontmatter.Metadata,
		}
		// log.Printf("frontmatterWorker %d processed  %v", i, sk.origSkill.Name)
		done <- struct{}{}
	}
}
