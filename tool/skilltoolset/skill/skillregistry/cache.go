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

// Package skillregistry provides access to the Skill Registry of the Gemini
// Enterprise Agent Platform as a [skill.Source].
//
// [NewSkillRegistrySource] talks to the registry directly. Most callers should
// wrap it in the cache from the simplecache subpackage
// (simplecache.NewCachedSkillRegistrySource) so that skills are not
// re-downloaded on every read.
package skillregistry

import (
	"google.golang.org/adk/v2/tool/skilltoolset/skill"
)

// Cache is an in-memory cache over a [Client]. It serves frontmatter, skill
// instructions and resources and refreshes itself in the background.
// Implementations are safe for concurrent use; build one with
// simplecache.NewCache.
type Cache interface {
	// WarmUp loads what's needed at start.
	WarmUp() error
	// Refresh refreshes the cache. Blocking. The cache can be used during the refresh.
	Refesh() error
	// ListFrontmatters returns the cached frontmatter of every skill.
	ListFrontmatters() ([]*skill.Frontmatter, error)
	// LoadFrontmatter returns the cached frontmatter of one skill, or
	// [skill.ErrSkillNotFound] if the name is unknown.
	LoadFrontmatter(name string) (*skill.Frontmatter, error)
	// LoadInstructions returns the Markdown body of a skill's SKILL.md,
	// downloading and caching the skill's archive on first use.
	LoadInstructions(name string) (string, error)
	// LoadResource returns the contents of one file in a skill's filesystem,
	// downloading and caching the skill's archive on first use.
	LoadResource(name, resourcePath string) (string, error)
	// ListResources returns the paths of the files in a skill's filesystem under
	// subpath; an empty subpath (or ".") lists them all.
	ListResources(name, subpath string) ([]string, error)
}
