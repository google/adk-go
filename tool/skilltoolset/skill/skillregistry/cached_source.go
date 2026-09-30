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
	"context"
	"io"
	"strings"

	"google.golang.org/adk/v2/tool/skilltoolset/skill"
)

type CachedSkillRegistrySource struct {
	cache Cache
}

func NewCachedSkillRegistrySource(ctx context.Context, cache Cache) (*CachedSkillRegistrySource, error) {
	return &CachedSkillRegistrySource{cache: cache}, nil
}

// ListFrontmatters implements [skill.Source].
func (c *CachedSkillRegistrySource) ListFrontmatters(ctx context.Context) ([]*skill.Frontmatter, error) {
	return c.cache.ListFrontmatters()
}

// ListResources implements [skill.Source].
func (c *CachedSkillRegistrySource) ListResources(ctx context.Context, name string, subpath string) ([]string, error) {
	return c.cache.ListResources(name, subpath)
}

// LoadFrontmatter implements [skill.Source].
func (c *CachedSkillRegistrySource) LoadFrontmatter(ctx context.Context, name string) (*skill.Frontmatter, error) {
	return c.cache.LoadFrontmatter(name)
}

// LoadInstructions implements [skill.Source].
func (c *CachedSkillRegistrySource) LoadInstructions(ctx context.Context, name string) (string, error) {
	return c.cache.LoadInstructions(name)
}

// LoadResource implements [skill.Source].
func (c *CachedSkillRegistrySource) LoadResource(ctx context.Context, name string, resourcePath string) (io.ReadCloser, error) {
	content, err := c.cache.LoadResource(name, resourcePath)
	if err != nil {
		return nil, err
	}
	return io.NopCloser(strings.NewReader(content)), nil
}

var _ skill.Source = (*CachedSkillRegistrySource)(nil)
