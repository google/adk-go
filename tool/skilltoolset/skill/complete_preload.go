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

package skill

import (
	"context"
	"fmt"
	"io"
	"path"
	"slices"
	"strings"
	"sync/atomic"
)

const maxResourceSize = 10 * 1024 * 1024 // 10MB

// completePreloadSource serves every call from an in-memory copy of base's
// skills, which reload replaces as a whole.
type completePreloadSource struct {
	base    Source
	preload atomic.Pointer[inMemorySource]
}

// WithCompletePreloadSource returns a Source proxy that wraps the given source
// and fully preloads all skill data (Frontmatters, Instructions, and Resources)
// into memory upon creation.
// This offers the fastest access speed after initialization, at the expense of
// higher initial load time and memory usage.
// Additionally, 'reload' callback is returned: calling it actualizes data by
// loading all the data anew.
//
// NOTE: source.ListResources must list all resources for subpath ".".
func WithCompletePreloadSource(ctx context.Context, source Source) (Source, func(context.Context) error, error) {
	s := &completePreloadSource{base: source}
	if err := s.reload(ctx); err != nil {
		return nil, nil, err
	}
	return s, s.reload, nil
}

func (s *completePreloadSource) ListFrontmatters(ctx context.Context) ([]*Frontmatter, error) {
	return s.preload.Load().ListFrontmatters(ctx)
}

func (s *completePreloadSource) LoadFrontmatter(ctx context.Context, name string) (*Frontmatter, error) {
	return s.preload.Load().LoadFrontmatter(ctx, name)
}

func (s *completePreloadSource) LoadInstructions(ctx context.Context, name string) (string, error) {
	return s.preload.Load().LoadInstructions(ctx, name)
}

func (s *completePreloadSource) LoadResource(ctx context.Context, name, resourcePath string) (io.ReadCloser, error) {
	return s.preload.Load().LoadResource(ctx, name, resourcePath)
}

func (s *completePreloadSource) ListResources(ctx context.Context, name, subpath string) ([]string, error) {
	return s.preload.Load().ListResources(ctx, name, subpath)
}

func (s *completePreloadSource) reload(ctx context.Context) error {
	frontmatters, err := s.base.ListFrontmatters(ctx)
	if err != nil {
		return fmt.Errorf("list frontmatters: %w", err)
	}
	frontmatters = slices.Clone(frontmatters)
	slices.SortFunc(frontmatters, func(a, b *Frontmatter) int {
		return strings.Compare(a.Name, b.Name)
	})

	skills := make([]InMemorySkill, 0, len(frontmatters))
	for i, frontmatter := range frontmatters {
		if err := ctx.Err(); err != nil {
			return err
		}

		// The frontmatters are sorted by name, so a repeated name follows
		// its first use.
		if i > 0 && frontmatters[i-1].Name == frontmatter.Name {
			return fmt.Errorf("%w: %q", ErrDuplicateSkill, frontmatter.Name)
		}

		instructions, err := s.base.LoadInstructions(ctx, frontmatter.Name)
		if err != nil {
			return fmt.Errorf("load instructions for skill %q: %w", frontmatter.Name, err)
		}

		resourcePaths, err := s.base.ListResources(ctx, frontmatter.Name, ".")
		if err != nil {
			return fmt.Errorf("list resources for skill %q: %w", frontmatter.Name, err)
		}

		resources := make(map[string][]byte)
		for _, resourcePath := range resourcePaths {
			if err := ctx.Err(); err != nil {
				return err
			}
			resource, err := s.base.LoadResource(ctx, frontmatter.Name, resourcePath)
			if err != nil {
				return fmt.Errorf("resource path %q in skill %q: %w", resourcePath, frontmatter.Name, err)
			}
			data, err := io.ReadAll(io.LimitReader(resource, maxResourceSize+1))
			_ = resource.Close()
			if err != nil {
				return fmt.Errorf("read resource path %q in skill %q: %w", resourcePath, frontmatter.Name, err)
			}
			if len(data) > maxResourceSize {
				return fmt.Errorf("resource %q in skill %q exceeds %d bytes limit", resourcePath, frontmatter.Name, maxResourceSize)
			}
			resources[path.Clean(resourcePath)] = data
		}

		skills = append(skills, InMemorySkill{
			Frontmatter:  frontmatter,
			Instructions: instructions,
			Resources:    resources,
		})
	}

	s.preload.Store(newInMemorySource(skills))
	return nil
}
