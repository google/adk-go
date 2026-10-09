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

// Package skillregistry provides the access to SkillRegistry on GCP.
// You should use the cached version from [simplecache.NewCachedSkillRegistrySource]
package skillregistry

import (
	"google.golang.org/adk/v2/tool/skilltoolset/skill"
)

// Cache is an interface which allows you to reset and query the cache
type Cache interface {
	WarmUp() error
	StopAutorefresh()
	ListFrontmatters() ([]*skill.Frontmatter, error)
	LoadFrontmatter(name string) (*skill.Frontmatter, error)
	LoadInstructions(name string) (string, error)
	LoadResource(name, resourcePath string) (string, error)
	ListResources(name, subpath string) ([]string, error)
}
