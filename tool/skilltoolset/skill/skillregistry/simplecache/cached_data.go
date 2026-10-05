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
	"fmt"
	"log"
	"time"

	"google.golang.org/adk/v2/tool/skilltoolset/skill"
	"google.golang.org/adk/v2/tool/skilltoolset/skill/skillregistry"
	agentregistry "google.golang.org/api/agentregistry/v1alpha"
)

type dataCache struct {
	skills      []*cachedSkill
	nameToSkill map[string]*cachedSkill

	readingSkillsStart time.Time
	readingSkillsEnd   time.Time
}

// newDataCache creates dataCache using the provided list of [agentregistry.Skill].
func newDataCache(client skillregistry.Client, skills []*agentregistry.Skill) *dataCache {
	dc := &dataCache{
		skills:      make([]*cachedSkill, 0, len(skills)),
		nameToSkill: make(map[string]*cachedSkill),
	}
	dc.skills = make([]*cachedSkill, 0, len(skills))
	dc.nameToSkill = make(map[string]*cachedSkill)

	// add all skills to the internal structures
	for _, sk := range skills {
		cs := newCachedSkill(client, sk)

		dc.skills = append(dc.skills, cs)
		dc.nameToSkill[sk.Name] = cs
	}

	return dc
}

// loadSkills loads the full list of skills (sequencial due to paging) and returns a new dataCache containing the skills
// An error is returned when getting the list of skills fails.
// No frontmatters are loaded. No zip files are loaded.
func loadSkills(c skillregistry.Client) (*dataCache, error) {
	log.Printf("loadSkills start")
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

// frontmatterReadDone is used to report the results of parallel workers readeing frontmatters
type frontmatterReadDone struct {
	// prefixedSkillName field is tricky. SkillRegistry uses in fact the key = (project, location, provider, displayname)
	// provider is "private" for your own skills or contains provider code (like "cloud.google.com").
	// In ADK project and location is client-specific. But we have only name to identify the skill.
	// So, we combine provider and pure name into one prefixedSkillName. By doing this we have the same key like SkillRegistry: (project, location, provider + "-" + displayname)
	// From now on, we override Frontmatter name (which doesn't have to be unique or match SkillRegistry) to this value.
	prefixedSkillName string

	// non-nil in case when the frontmatter cannot be read
	err error

	// the skill which was the input parameter for worker (needed to be able to match the results with the right skill)
	skill *cachedSkill
}

// loadFrontmatters loads fronmatters for skills in dataCache.
// It's being done in parallel using nWorkers
func loadFrontmatters(c skillregistry.Client, nWorkers int, dc *dataCache) error {
	if nWorkers <= 0 {
		return fmt.Errorf("nWorkers must be > 0")
	}
	log.Printf("loadFrontmatters")

	toProcess := make(chan *cachedSkill, len(dc.skills))
	done := make(chan frontmatterReadDone, len(dc.skills))

	for i := 0; i < nWorkers; i++ {
		go frontmatterWorker(i, c, toProcess, done)
	}

	for _, sk := range dc.skills {
		toProcess <- sk
	}
	close(toProcess)

	// return for all workers to finish
	for i := 0; i < len(dc.skills); i++ {
		res := <-done
		if res.err != nil {
			// fail on the first failure
			return res.err
		}
		if res.prefixedSkillName != "" {
			dc.nameToSkill[res.prefixedSkillName] = res.skill
		}
	}
	log.Printf("loadFrontmatters done")
	return nil
}

// frontmatterWorker is used to load frontmatters in parallel. It uses the provided
// client to load frontmatter.
func frontmatterWorker(i int, c skillregistry.Client, toProcess chan *cachedSkill, done chan frontmatterReadDone) {
	for sk := range toProcess {
		s, err := c.GetSkill(sk.origSkill.Name)
		if err != nil {
			done <- frontmatterReadDone{prefixedSkillName: "", skill: sk, err: fmt.Errorf("frontmatterWorker %d failed: %w", i, err)}
			continue
		}

		// we assume that the location is aligned with the client, so we can safely strip it.
		// prefixedSkillName is by SkillRegistry built using the provider and the display name.
		// for your own skills with name "akill-name" you will see private-skill-name.
		_, _, prefixedSkillName, err := c.ParseSkillName(sk.origSkill.Name)
		if err != nil {
			done <- frontmatterReadDone{prefixedSkillName: "", skill: sk, err: fmt.Errorf("frontmatterWorker %d failed to parse skillID: %w", i, err)}
			continue
		}

		// skillName here is a prefix + displayName (provided by SkillRegistry)
		sk.frontmatter = skill.Frontmatter{
			Name:          prefixedSkillName, // here we need the unique name, so we are overriding whatever comes from the Frontmatter
			Description:   s.Frontmatter.Description,
			License:       s.Frontmatter.License,
			Compatibility: s.Frontmatter.Compatibility,
			Metadata:      s.Frontmatter.Metadata,
		}

		// return the parsed skillName
		done <- frontmatterReadDone{prefixedSkillName: prefixedSkillName, skill: sk, err: nil}
	}
}
